"""Implicit Q-Learning (IQL) baseline for D4RL MuJoCo.

This is a compact shared-pipeline implementation. It is meant to answer a
project-level question first: whether SSAR's strong result is mostly explained
by an IQL-style value signal. Use official implementations later for paper-grade
reproduction.
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
from torch.distributions import Normal

from common import (
    D4RLDataset,
    ExperimentLogger,
    eval_episodes,
    get_obs_act_dims,
    make_env,
    set_seed,
    write_result,
)


def mlp(in_dim: int, out_dim: int, hidden: int,
        n_layers: int = 2, output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float, hidden: int = 256,
                 min_log_std: float = -5.0, max_log_std: float = 2.0):
        super().__init__()
        self.max_action = float(max_action)
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std
        self.trunk = mlp(obs_dim, hidden, hidden, n_layers=1)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.max_action * torch.tanh(self.mean(h))
        log_std = self.log_std(h).clamp(self.min_log_std, self.max_log_std)
        return mean, log_std

    def log_prob(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        return dist.log_prob(act).sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return mean.clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


class Value(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = mlp(obs_dim, 1, hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


@dataclass
class IQLConfig:
    discount: float = 0.99
    tau: float = 0.005
    expectile: float = 0.7
    beta: float = 3.0
    max_weight: float = 100.0
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    value_lr: float = 3e-4
    hidden: int = 256
    normalize_obs: bool = True


def expectile_loss(diff: torch.Tensor, expectile: float) -> torch.Tensor:
    weight = torch.where(diff > 0, expectile, 1.0 - expectile)
    return (weight * diff.pow(2)).mean()


class IQLAgent:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 config: IQLConfig, device: str,
                 obs_mean: np.ndarray, obs_std: np.ndarray):
        self.device = device
        self.config = config
        self.max_action = float(max_action)

        self.actor = GaussianActor(obs_dim, act_dim, max_action, config.hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, config.hidden).to(device)
        self.critic_target = Critic(obs_dim, act_dim, config.hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.value = Value(obs_dim, config.hidden).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.value_opt = torch.optim.Adam(self.value.parameters(), lr=config.value_lr)

        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            target_q1, target_q2 = self.critic_target(obs, act)
            target_q = torch.min(target_q1, target_q2)
        v = self.value(obs)
        value_loss = expectile_loss(target_q - v, self.config.expectile)

        self.value_opt.zero_grad()
        value_loss.backward()
        self.value_opt.step()

        with torch.no_grad():
            next_v = self.value(next_obs)
            q_target = rew + (1.0 - done) * self.config.discount * next_v
        q1, q2 = self.critic(obs, act)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        with torch.no_grad():
            q1_det, q2_det = self.critic(obs, act)
            q_det = torch.min(q1_det, q2_det)
            adv = q_det - self.value(obs)
            exp_adv = torch.exp(self.config.beta * adv).clamp(max=self.config.max_weight)
        log_prob = self.actor.log_prob(obs, act)
        actor_loss = -(exp_adv * log_prob).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "value_loss": float(value_loss.item()),
            "adv_mean": float(adv.mean().item()),
            "adv_weight_mean": float(exp_adv.mean().item()),
            "q_mean": float(q_det.mean().item()),
            "v_mean": float(v.detach().mean().item()),
        }

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        obs = (obs.astype(np.float32) - self.obs_mean) / self.obs_std
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor.deterministic(obs_t).cpu().numpy().squeeze(0)
        return np.clip(action, -self.max_action, self.max_action)

    def _soft_update(self, src: nn.Module, dst: nn.Module) -> None:
        for p, tp in zip(src.parameters(), dst.parameters()):
            tp.data.mul_(1.0 - self.config.tau).add_(self.config.tau * p.data)


def train(env_name: str, seed: int, steps: int, batch_size: int,
          eval_freq: int, eval_episodes_n: int, result_dir: str,
          use_aim: bool, use_wandb: bool, config: IQLConfig,
          algo_name: str = "iql"):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    agent = IQLAgent(
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
        config=dict(algo=algo_name, base_algo="iql", env=env_name, seed=seed, steps=steps, **asdict(config)),
    )

    print(f"[{algo_name}] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    print(f"         steps={steps:,} batch={batch_size} expectile={config.expectile} beta={config.beta}")
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
                    f"| value={last_info.get('value_loss', float('nan')):.4f} "
                    f"| actor={last_info.get('actor_loss', float('nan')):.4f} "
                    f"| w={last_info.get('adv_weight_mean', float('nan')):.2f} "
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
    parser.add_argument("--algo_name", default="iql")
    parser.add_argument("--aim", action="store_true", help="启用 Aim local tracking")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb logging")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--expectile", type=float, default=0.7)
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--max_weight", type=float, default=100.0)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--no_normalize_obs", action="store_true")
    args = parser.parse_args()

    config = IQLConfig(
        discount=args.discount,
        tau=args.tau,
        expectile=args.expectile,
        beta=args.beta,
        max_weight=args.max_weight,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        value_lr=args.value_lr,
        hidden=args.hidden,
        normalize_obs=not args.no_normalize_obs,
    )
    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.result_dir,
          args.aim, args.wandb, config, args.algo_name)


if __name__ == "__main__":
    main()
