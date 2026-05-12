"""ReBRAC-style policy regularization baseline for D4RL MuJoCo.

This is a compact PyTorch implementation for early C-track smoke tests. It keeps
the project interface aligned with ``bc.py`` / ``td3_bc.py`` and uses the shared
``D4RLDataset`` for timeout-safe transitions plus next-action targets.

Reference:
    Tarasov et al., ReBRAC: ReBehavior Cloning Regularization for Offline RL,
    arXiv 2023.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    D4RLDataset,
    ExperimentLogger,
    eval_episodes,
    get_obs_act_dims,
    make_env,
    set_seed,
    write_result,
)


def mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int,
        layernorm: bool = False, output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    dim = in_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(dim, hidden))
        layers.append(nn.ReLU())
        if layernorm:
            layers.append(nn.LayerNorm(hidden))
        dim = hidden
    layers.append(nn.Linear(dim, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 hidden: int, n_layers: int, layernorm: bool):
        super().__init__()
        self.max_action = float(max_action)
        self.net = mlp(obs_dim, act_dim, hidden, n_layers, layernorm, nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class CriticEnsemble(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int,
                 n_layers: int, layernorm: bool, num_critics: int):
        super().__init__()
        self.qs = nn.ModuleList([
            mlp(obs_dim + act_dim, 1, hidden, n_layers, layernorm)
            for _ in range(num_critics)
        ])

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        x = torch.cat([obs, act], dim=-1)
        return torch.stack([q(x) for q in self.qs], dim=0)


@dataclass
class ReBRACConfig:
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_bc_coef: float = 1.0
    critic_bc_coef: float = 1.0
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    hidden: int = 256
    actor_layers: int = 3
    critic_layers: int = 3
    num_critics: int = 2
    actor_ln: bool = False
    critic_ln: bool = True
    normalize_obs: bool = False
    normalize_q: bool = True


class ReBRACAgent:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 config: ReBRACConfig, device: str = "cpu",
                 obs_mean: np.ndarray | None = None, obs_std: np.ndarray | None = None):
        self.device = device
        self.config = config
        self.max_action = float(max_action)

        self.actor = Actor(obs_dim, act_dim, max_action, config.hidden,
                           config.actor_layers, config.actor_ln).to(device)
        self.actor_target = Actor(obs_dim, act_dim, max_action, config.hidden,
                                  config.actor_layers, config.actor_ln).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = CriticEnsemble(obs_dim, act_dim, config.hidden, config.critic_layers,
                                     config.critic_ln, config.num_critics).to(device)
        self.critic_target = CriticEnsemble(obs_dim, act_dim, config.hidden, config.critic_layers,
                                            config.critic_ln, config.num_critics).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.total_it = 0
        self.obs_mean = np.zeros(obs_dim, dtype=np.float32) if obs_mean is None else obs_mean.astype(np.float32)
        self.obs_std = np.ones(obs_dim, dtype=np.float32) if obs_std is None else obs_std.astype(np.float32)

    def update(self, batch: dict) -> dict:
        self.total_it += 1
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        next_act_data = batch["next_act"]
        done = batch["done"]

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_act = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q = self.critic_target(next_obs, next_act).min(dim=0).values
            critic_bc = ((next_act - next_act_data) ** 2).sum(dim=-1, keepdim=True)
            target_q = target_q - self.config.critic_bc_coef * critic_bc
            target_q = rew + (1.0 - done) * self.config.discount * target_q

        cur_q = self.critic(obs, act)
        critic_loss = ((cur_q - target_q.unsqueeze(0)) ** 2).mean(dim=(1, 2)).sum()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {
            "critic_loss": float(critic_loss.item()),
            "q_min": float(cur_q.detach().min(dim=0).values.mean().item()),
        }

        if self.total_it % self.config.policy_freq == 0:
            pi = self.actor(obs)
            q = self.critic(obs, pi).min(dim=0).values
            lam = 1.0
            if self.config.normalize_q:
                lam = 1.0 / q.abs().mean().detach().clamp(min=1e-6)
            actor_bc = ((pi - act) ** 2).sum(dim=-1, keepdim=True)
            actor_loss = (self.config.actor_bc_coef * actor_bc - lam * q).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

            info.update({
                "actor_loss": float(actor_loss.item()),
                "bc_mse_policy": float(actor_bc.detach().mean().item()),
                "lambda": float(lam.item() if isinstance(lam, torch.Tensor) else lam),
            })

        return info

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = (obs.astype(np.float32) - self.obs_mean) / self.obs_std
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(obs_t).cpu().numpy().squeeze(0)
        return np.clip(action, -self.max_action, self.max_action)

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - self.config.tau).add_(self.config.tau * p.data)


def train(env_name: str, seed: int, steps: int, batch_size: int,
          eval_freq: int, eval_episodes_n: int, result_dir: str,
          use_aim: bool, use_wandb: bool, config: ReBRACConfig,
          algo_name: str = "rebrac_lite"):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    agent = ReBRACAgent(
        obs_dim, act_dim, max_action, config, device=device,
        obs_mean=dataset.obs_mean.squeeze(0),
        obs_std=dataset.obs_std.squeeze(0),
    )

    logger = ExperimentLogger(
        algo=algo_name,
        env_name=env_name,
        seed=seed,
        use_aim=use_aim,
        use_wandb=use_wandb,
        config=dict(algo=algo_name, base_algo="rebrac", env=env_name, seed=seed, steps=steps, **asdict(config)),
    )

    print(f"[{algo_name}] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    print(f"         steps={steps:,} batch={batch_size} critics={config.num_critics} "
          f"actor_bc={config.actor_bc_coef} critic_bc={config.critic_bc_coef}")
    t0 = time.time()
    last_info: dict = {}

    try:
        for step in range(1, steps + 1):
            last_info = agent.update(dataset.sample(batch_size))
            if step % eval_freq == 0 or step == steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, algo_name, env_name, seed, step, "offline", metrics)
                elapsed = time.time() - t0
                log = {**last_info, **metrics, "step": step}
                print(
                    f"  step={step:>8,} | norm={metrics['normalized_score']:6.2f} "
                    f"| raw={metrics['raw_return']:8.1f} | "
                    f"critic={last_info.get('critic_loss', float('nan')):.4f} "
                    f"| actor={last_info.get('actor_loss', float('nan')):.4f} "
                    f"| t={elapsed:.0f}s"
                )
                logger.log(log, step=step, context={"phase": "offline"})
    finally:
        logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--algo_name", default="rebrac_lite")
    parser.add_argument("--aim", action="store_true", help="启用 Aim local tracking")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb logging")

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--actor_layers", type=int, default=3)
    parser.add_argument("--critic_layers", type=int, default=3)
    parser.add_argument("--num_critics", type=int, default=2)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--actor_bc_coef", type=float, default=1.0)
    parser.add_argument("--critic_bc_coef", type=float, default=1.0)
    parser.add_argument("--actor_lr", type=float, default=1e-3)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--actor_ln", action="store_true")
    parser.add_argument("--no_critic_ln", action="store_true")
    parser.add_argument("--normalize_obs", action="store_true")
    parser.add_argument("--no_normalize_q", action="store_true")
    args = parser.parse_args()

    config = ReBRACConfig(
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        actor_bc_coef=args.actor_bc_coef,
        critic_bc_coef=args.critic_bc_coef,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden=args.hidden,
        actor_layers=args.actor_layers,
        critic_layers=args.critic_layers,
        num_critics=args.num_critics,
        actor_ln=args.actor_ln,
        critic_ln=not args.no_critic_ln,
        normalize_obs=args.normalize_obs,
        normalize_q=not args.no_normalize_q,
    )
    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.result_dir,
          args.aim, args.wandb, config, args.algo_name)


if __name__ == "__main__":
    main()
