"""Conservative Q-Learning (CQL) baseline for D4RL MuJoCo.

Compact SAC+CQL style implementation for the shared pipeline. This is a
project-level anchor, not a strict paper reproduction.
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

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
EPS = 1e-6


def mlp(in_dim: int, out_dim: int, hidden: int,
        n_layers: int = 2, output_activation: Optional[nn.Module] = None) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    if output_activation is not None:
        layers.append(output_activation)
    return nn.Sequential(*layers)


class TanhGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, max_action: float, hidden: int = 256):
        super().__init__()
        self.max_action = float(max_action)
        self.trunk = mlp(obs_dim, hidden, hidden, n_layers=1)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(obs)
        dist = Normal(mean, log_std.exp())
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + EPS)
        return self.max_action * action, log_prob.sum(dim=-1, keepdim=True)

    @torch.no_grad()
    def deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _ = self(obs)
        return (self.max_action * torch.tanh(mean)).clamp(-self.max_action, self.max_action)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, 1, hidden)
        self.q2 = mlp(obs_dim + act_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)


@dataclass
class CQLConfig:
    discount: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    cql_alpha: float = 5.0
    num_random: int = 10
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    hidden: int = 256
    normalize_obs: bool = True


class CQLAgent:
    def __init__(self, obs_dim: int, act_dim: int, max_action: float,
                 config: CQLConfig, device: str,
                 obs_mean: np.ndarray, obs_std: np.ndarray):
        self.device = device
        self.config = config
        self.max_action = float(max_action)
        self.act_dim = act_dim

        self.actor = TanhGaussianActor(obs_dim, act_dim, max_action, config.hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, config.hidden).to(device)
        self.critic_target = Critic(obs_dim, act_dim, config.hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = obs_std.astype(np.float32)

    def update(self, batch: dict) -> dict:
        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action, next_logp = self.actor.sample(next_obs)
            tq1, tq2 = self.critic_target(next_obs, next_action)
            target_q = torch.min(tq1, tq2) - self.config.alpha * next_logp
            target_q = rew + (1.0 - done) * self.config.discount * target_q

        q1, q2 = self.critic(obs, act)
        bellman_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        cql_loss = self._cql_penalty(obs, q1, q2)
        critic_loss = bellman_loss + self.config.cql_alpha * cql_loss

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_action, logp = self.actor.sample(obs)
        aq1, aq2 = self.critic(obs, new_action)
        actor_loss = (self.config.alpha * logp - torch.min(aq1, aq2)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.critic, self.critic_target)

        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "cql_loss": float(cql_loss.item()),
            "q_mean": float(q1.detach().mean().item()),
        }

    def _cql_penalty(self, obs: torch.Tensor, data_q1: torch.Tensor, data_q2: torch.Tensor) -> torch.Tensor:
        batch_size = obs.shape[0]
        n = self.config.num_random
        obs_rep = obs.unsqueeze(1).repeat(1, n, 1).reshape(batch_size * n, -1)

        random_actions = torch.empty(
            batch_size * n, self.act_dim, dtype=obs.dtype, device=obs.device
        ).uniform_(-self.max_action, self.max_action)
        policy_actions, policy_logp = self.actor.sample(obs_rep)

        q1_rand, q2_rand = self.critic(obs_rep, random_actions)
        q1_policy, q2_policy = self.critic(obs_rep, policy_actions)
        q1_cat = torch.cat([
            q1_rand.view(batch_size, n, 1),
            (q1_policy - policy_logp).view(batch_size, n, 1),
        ], dim=1)
        q2_cat = torch.cat([
            q2_rand.view(batch_size, n, 1),
            (q2_policy - policy_logp).view(batch_size, n, 1),
        ], dim=1)

        cql_q1 = torch.logsumexp(q1_cat, dim=1).mean() - data_q1.mean()
        cql_q2 = torch.logsumexp(q2_cat, dim=1).mean() - data_q2.mean()
        return cql_q1 + cql_q2

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
          use_aim: bool, use_wandb: bool, config: CQLConfig,
          algo_name: str = "cql"):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    agent = CQLAgent(
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
        config=dict(algo=algo_name, base_algo="cql", env=env_name, seed=seed, steps=steps, **asdict(config)),
    )

    print(f"[{algo_name}] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    print(f"         steps={steps:,} batch={batch_size} cql_alpha={config.cql_alpha} num_random={config.num_random}")
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
                    f"| cql={last_info.get('cql_loss', float('nan')):.4f} "
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
    parser.add_argument("--algo_name", default="cql")
    parser.add_argument("--aim", action="store_true", help="启用 Aim local tracking")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb logging")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--cql_alpha", type=float, default=5.0)
    parser.add_argument("--num_random", type=int, default=10)
    parser.add_argument("--actor_lr", type=float, default=3e-4)
    parser.add_argument("--critic_lr", type=float, default=3e-4)
    parser.add_argument("--no_normalize_obs", action="store_true")
    args = parser.parse_args()

    config = CQLConfig(
        discount=args.discount,
        tau=args.tau,
        alpha=args.alpha,
        cql_alpha=args.cql_alpha,
        num_random=args.num_random,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        hidden=args.hidden,
        normalize_obs=not args.no_normalize_obs,
    )
    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.result_dir,
          args.aim, args.wandb, config, args.algo_name)


if __name__ == "__main__":
    main()
