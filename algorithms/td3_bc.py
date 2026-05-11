"""TD3+BC baseline for D4RL MuJoCo.

This is the first real C-track algorithm implementation in the shared project
pipeline. It keeps the interface aligned with ``algorithms/bc.py`` and uses the
shared D4RL dataset / evaluator / result writer.

Reference:
    Fujimoto and Gu, A Minimalist Approach to Offline Reinforcement Learning,
    NeurIPS 2021.

Usage:
    python -m algorithms.td3_bc --env hopper-medium-v2 --seed 0 --steps 100000
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


def mlp(in_dim: int, out_dim: int, hidden: int, n_layers: int = 2,
        output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float, hidden: int = 256):
        super().__init__()
        self.max_action = float(max_action)
        self.net = mlp(obs_dim, act_dim, hidden, output_activation=nn.Tanh())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.q1(torch.cat([obs, act], dim=-1))


@dataclass
class TD3BCConfig:
    discount: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    alpha: float = 2.5
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: int = 256
    normalize_obs: bool = True


class TD3BCAgent:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 config: TD3BCConfig, device: str = "cpu",
                 obs_mean: np.ndarray | None = None, obs_std: np.ndarray | None = None):
        self.device = device
        self.config = config
        self.max_action = float(max_action)

        self.actor = Actor(obs_dim, act_dim, max_action, config.hidden).to(device)
        self.actor_target = Actor(obs_dim, act_dim, max_action, config.hidden).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(obs_dim, act_dim, config.hidden).to(device)
        self.critic_target = Critic(obs_dim, act_dim, config.hidden).to(device)
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
        done = batch["done"]

        with torch.no_grad():
            noise = (torch.randn_like(act) * self.config.policy_noise).clamp(
                -self.config.noise_clip, self.config.noise_clip
            )
            next_act = (self.actor_target(next_obs) + noise).clamp(-self.max_action, self.max_action)
            target_q1, target_q2 = self.critic_target(next_obs, next_act)
            target_q = torch.min(target_q1, target_q2)
            target_q = rew + (1.0 - done) * self.config.discount * target_q

        cur_q1, cur_q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        info = {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(cur_q1.detach().mean().item()),
        }

        if self.total_it % self.config.policy_freq == 0:
            pi = self.actor(obs)
            q = self.critic.q1_only(obs, pi)
            lam = self.config.alpha / q.abs().mean().detach().clamp(min=1e-6)
            actor_loss = -lam * q.mean() + F.mse_loss(pi, act)

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

            info.update({
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(F.mse_loss(pi.detach(), act).item()),
                "lambda": float(lam.item()),
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
          use_aim: bool, use_wandb: bool, config: TD3BCConfig):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    agent = TD3BCAgent(
        obs_dim, act_dim, max_action, config, device=device,
        obs_mean=dataset.obs_mean.squeeze(0),
        obs_std=dataset.obs_std.squeeze(0),
    )

    logger = ExperimentLogger(
        algo="td3_bc",
        env_name=env_name,
        seed=seed,
        use_aim=use_aim,
        use_wandb=use_wandb,
        config=dict(algo="td3_bc", env=env_name, seed=seed, steps=steps, **asdict(config)),
    )

    print(f"[TD3+BC] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    print(f"         steps={steps:,} batch={batch_size} normalize_obs={config.normalize_obs}")
    t0 = time.time()
    last_info: dict = {}

    try:
        for step in range(1, steps + 1):
            last_info = agent.update(dataset.sample(batch_size))

            if step % eval_freq == 0 or step == steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, "td3_bc", env_name, seed, step, "offline", metrics)
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
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq", type=int, default=5_000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--aim", action="store_true", help="启用 Aim local tracking")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb logging")

    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy_noise", type=float, default=0.2)
    parser.add_argument("--noise_clip", type=float, default=0.5)
    parser.add_argument("--policy_freq", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=2.5)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--no_normalize_obs", action="store_true")
    args = parser.parse_args()

    config = TD3BCConfig(
        discount=args.discount,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_freq=args.policy_freq,
        alpha=args.alpha,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden=args.hidden,
        normalize_obs=not args.no_normalize_obs,
    )
    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.result_dir,
          args.aim, args.wandb, config)


if __name__ == "__main__":
    main()
