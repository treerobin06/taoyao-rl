"""Behavior Cloning (BC) — smoke test 用的最简 offline RL baseline.

不是项目的主算法（C 线主推 TD3+BC / ReBRAC / PRDC / A2PR，另外两个方向跑
DMG / SCQ）；BC 只用来验证 data + train + eval + tracking 整条 pipeline 通畅。

用法（独立运行）：
    python -m algorithms.bc --env hopper-medium-v2 --seed 0 --steps 50000

预期：hopper-medium-v2 上 50k steps 训练后 normalized_score ~30-50（BC 论文 ~52）
"""
import argparse
import os
import sys
import time

# Allow running both as module and as script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from common import (
    D4RLDataset,
    ExperimentLogger,
    eval_episodes,
    get_obs_act_dims,
    make_env,
    set_seed,
    write_result,
)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden=256, n_layers=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, out_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class BCAgent:
    def __init__(self, obs_dim, act_dim, hidden=256, lr=3e-4, device="cpu"):
        self.device = device
        self.actor = MLP(obs_dim, act_dim, hidden).to(device)
        self.opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

    def update(self, batch):
        pred = self.actor(batch["obs"])
        loss = ((pred - batch["act"]) ** 2).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return {"bc_loss": loss.item()}

    @torch.no_grad()
    def act(self, obs):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(obs_t).cpu().numpy().squeeze(0)


def train(env_name: str, seed: int, steps: int, batch_size: int,
          eval_freq: int, eval_episodes_n: int, hidden: int,
          result_dir: str, use_aim: bool, use_wandb: bool):

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    dataset = D4RLDataset(env, device=device, normalize_obs=False)

    agent = BCAgent(obs_dim, act_dim, hidden=hidden, device=device)

    logger = ExperimentLogger(
        algo="bc",
        env_name=env_name,
        seed=seed,
        use_aim=use_aim,
        use_wandb=use_wandb,
        config=dict(algo="bc", env=env_name, seed=seed, steps=steps, hidden=hidden),
    )

    print(f"[BC] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    t0 = time.time()

    try:
        for step in range(1, steps + 1):
            batch = dataset.sample(batch_size)
            info = agent.update(batch)

            if step % eval_freq == 0 or step == steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, "bc", env_name, seed, step,
                             "offline", metrics)
                elapsed = time.time() - t0
                log = {**info, **metrics, "step": step}
                print(f"  step={step:>7,} | norm={metrics['normalized_score']:6.2f} | "
                      f"raw={metrics['raw_return']:8.1f} | "
                      f"loss={info['bc_loss']:.4f} | t={elapsed:.0f}s")
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
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--aim", action="store_true", help="启用 Aim local tracking")
    parser.add_argument("--wandb", action="store_true", help="启用 wandb logging")
    args = parser.parse_args()

    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.hidden,
          args.result_dir, args.aim, args.wandb)


if __name__ == "__main__":
    main()
