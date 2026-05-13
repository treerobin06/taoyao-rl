"""Minimal offline-to-online TD3+BC / ATLAS fine-tuning runner.

This script is intentionally small and explicit. It answers the course-project
gap: after offline training on D4RL replay data, can a policy continue with real
environment interaction, and how does the offline BC regularizer affect that
fine-tuning curve?

The online phase uses a mixed replay batch:
- offline transitions from the D4RL dataset;
- online transitions collected by the current actor with exploration noise.

BC / label-file regularization is applied only to offline samples. Online
transitions get zero BC weight, so the actor is not trained to imitate its noisy
exploration actions.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import asdict
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn.functional as F

from algorithms.td3_bc import TD3BCAgent, TD3BCConfig
from algorithms.trusted_td3_bc import build_label_trust_weights
from common import (
    D4RLDataset,
    ExperimentLogger,
    eval_episodes,
    get_obs_act_dims,
    make_env,
    set_seed,
    write_result,
)


class OnlineReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int, device: str):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: np.ndarray, next_obs: np.ndarray,
            reward: float, done: bool) -> None:
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.next_obs[self.ptr] = next_obs
        self.rewards[self.ptr, 0] = float(reward)
        self.dones[self.ptr, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict:
        if self.size <= 0:
            raise ValueError("cannot sample from an empty online replay buffer")
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]).to(self.device),
            "act": torch.from_numpy(self.actions[idx]).to(self.device),
            "rew": torch.from_numpy(self.rewards[idx]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[idx]).to(self.device),
            "done": torch.from_numpy(self.dones[idx]).to(self.device),
            "bc_weight": torch.zeros((batch_size, 1), dtype=torch.float32, device=self.device),
        }


def _sample_offline(dataset: D4RLDataset, bc_weights: np.ndarray, batch_size: int) -> dict:
    idx = np.random.randint(0, dataset.size, size=batch_size)
    return {
        "obs": torch.from_numpy(dataset.observations[idx]).to(dataset.device),
        "act": torch.from_numpy(dataset.actions[idx]).to(dataset.device),
        "rew": torch.from_numpy(dataset.rewards[idx]).to(dataset.device).unsqueeze(-1),
        "next_obs": torch.from_numpy(dataset.next_obs[idx]).to(dataset.device),
        "done": torch.from_numpy(dataset.dones[idx]).to(dataset.device).unsqueeze(-1),
        "bc_weight": torch.from_numpy(bc_weights[idx]).to(dataset.device),
    }


def _concat_batches(parts: list[dict]) -> dict:
    keys = parts[0].keys()
    return {k: torch.cat([part[k] for part in parts], dim=0) for k in keys}


def sample_mixed_batch(
    dataset: D4RLDataset,
    offline_bc_weights: np.ndarray,
    online_buffer: OnlineReplayBuffer,
    batch_size: int,
    online_fraction: float,
) -> dict:
    online_n = int(round(batch_size * float(np.clip(online_fraction, 0.0, 1.0))))
    online_n = min(online_n, online_buffer.size, batch_size)
    offline_n = batch_size - online_n

    parts = []
    if offline_n > 0:
        parts.append(_sample_offline(dataset, offline_bc_weights, offline_n))
    if online_n > 0:
        parts.append(online_buffer.sample(online_n))
    return _concat_batches(parts)


def td3_update(
    agent: TD3BCAgent,
    batch: dict,
    bc_coef: float,
    trust_gate: str = "none",
    gate_temperature: float = 10.0,
    gate_min_weight: float = 0.0,
    gate_margin: float = 0.0,
) -> dict:
    """One TD3-style update with configurable offline BC regularization."""
    agent.total_it += 1
    config = agent.config

    obs = batch["obs"]
    act = batch["act"]
    rew = batch["rew"]
    next_obs = batch["next_obs"]
    done = batch["done"]
    bc_weight = batch["bc_weight"]

    with torch.no_grad():
        noise = (torch.randn_like(act) * config.policy_noise).clamp(
            -config.noise_clip, config.noise_clip
        )
        next_act = (agent.actor_target(next_obs) + noise).clamp(-agent.max_action, agent.max_action)
        target_q1, target_q2 = agent.critic_target(next_obs, next_act)
        target_q = torch.min(target_q1, target_q2)
        target_q = rew + (1.0 - done) * config.discount * target_q

    cur_q1, cur_q2 = agent.critic(obs, act)
    critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)

    agent.critic_opt.zero_grad()
    critic_loss.backward()
    agent.critic_opt.step()

    info = {
        "critic_loss": float(critic_loss.item()),
        "q_mean": float(cur_q1.detach().mean().item()),
        "bc_coef": float(bc_coef),
        "batch_bc_weight": float(bc_weight.mean().item()),
    }

    if agent.total_it % config.policy_freq == 0:
        pi = agent.actor(obs)
        q = agent.critic.q1_only(obs, pi)
        lam = config.alpha / q.abs().mean().detach().clamp(min=1e-6)
        effective_bc_weight = bc_weight
        gate_info = {}
        if trust_gate == "qgap":
            with torch.no_grad():
                q_data = agent.critic.q1_only(obs, act)
                qgap = q_data - q.detach() - float(gate_margin)
                temp = max(float(gate_temperature), 1e-6)
                gate = torch.sigmoid(qgap / temp)
                min_w = float(np.clip(gate_min_weight, 0.0, 1.0))
                gate = min_w + (1.0 - min_w) * gate
                effective_bc_weight = bc_weight * gate
                offline_mask = bc_weight > 0
                gate_info = {
                    "qgate_mean": float(gate[offline_mask].mean().item()) if offline_mask.any() else 0.0,
                    "qgate_qgap_mean": float(qgap[offline_mask].mean().item()) if offline_mask.any() else 0.0,
                    "qgate_effective_bc_mean": float(effective_bc_weight.mean().item()),
                }
        elif trust_gate != "none":
            raise ValueError(f"unsupported trust_gate={trust_gate}")

        per_sample_bc = ((pi - act) ** 2).mean(dim=-1, keepdim=True)
        bc_loss = (effective_bc_weight * per_sample_bc).sum() / effective_bc_weight.sum().clamp(min=1e-6)
        actor_loss = -lam * q.mean() + float(bc_coef) * bc_loss

        agent.actor_opt.zero_grad()
        actor_loss.backward()
        agent.actor_opt.step()

        agent._soft_update(agent.critic, agent.critic_target)
        agent._soft_update(agent.actor, agent.actor_target)

        info.update({
            "actor_loss": float(actor_loss.item()),
            "bc_loss": float(bc_loss.item()),
            "lambda": float(lam.item()),
            "effective_bc_weight": float(effective_bc_weight.mean().item()),
            **gate_info,
        })
    return info


def _linear_value(start: float, end: float, step: int, total_steps: int) -> float:
    if total_steps <= 1:
        return float(end)
    frac = min(max(step / float(total_steps), 0.0), 1.0)
    return float(start + frac * (end - start))


def _safe_reset(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, reward, bool(terminated or truncated), info
    return out


def train_o2o(
    env_name: str,
    seed: int,
    offline_steps: int,
    online_steps: int,
    batch_size: int,
    eval_freq_offline: int,
    eval_freq_online: int,
    eval_episodes_n: int,
    result_dir: str,
    use_aim: bool,
    use_wandb: bool,
    config: TD3BCConfig,
    algo_name: str,
    label_path: Optional[str],
    label_score_key: str,
    label_min_weight: float,
    online_batch_fraction: float,
    online_replay_capacity: int,
    exploration_noise: float,
    offline_bc_coef: float,
    online_bc_coef_start: float,
    online_bc_coef_end: float,
    online_trust_gate: str,
    online_gate_temperature: float,
    online_gate_min_weight: float,
    online_gate_margin: float,
    online_gate_start_step: int,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    online_env = make_env(env_name, seed=seed + 10_000)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    if label_path:
        offline_bc_weights, label_info = build_label_trust_weights(
            dataset=dataset,
            label_path=label_path,
            score_key=label_score_key,
            min_weight=label_min_weight,
            threshold=0.5,
            binarize=False,
        )
    else:
        offline_bc_weights = np.ones((dataset.size, 1), dtype=np.float32)
        label_info = {"label_path": "", "label_weight_mean": 1.0, "trusted_fraction": 1.0}

    agent = TD3BCAgent(
        obs_dim, act_dim, max_action, config, device=device,
        obs_mean=dataset.obs_mean.squeeze(0),
        obs_std=dataset.obs_std.squeeze(0),
    )
    online_buffer = OnlineReplayBuffer(obs_dim, act_dim, online_replay_capacity, device)

    logger = ExperimentLogger(
        algo=algo_name,
        env_name=env_name,
        seed=seed,
        use_aim=use_aim,
        use_wandb=use_wandb,
        config=dict(
            algo=algo_name,
            base_algo="td3_bc_o2o",
            env=env_name,
            seed=seed,
            offline_steps=offline_steps,
            online_steps=online_steps,
            batch_size=batch_size,
            online_batch_fraction=online_batch_fraction,
            exploration_noise=exploration_noise,
            offline_bc_coef=offline_bc_coef,
            online_bc_coef_start=online_bc_coef_start,
            online_bc_coef_end=online_bc_coef_end,
            online_trust_gate=online_trust_gate,
            online_gate_temperature=online_gate_temperature,
            online_gate_min_weight=online_gate_min_weight,
            online_gate_margin=online_gate_margin,
            online_gate_start_step=online_gate_start_step,
            label_score_key=label_score_key,
            label_min_weight=label_min_weight,
            **asdict(config),
            **{f"label_{k}": v for k, v in label_info.items() if isinstance(v, (int, float, str, bool))},
        ),
    )

    print(f"[{algo_name}] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    print(
        f"         offline={offline_steps:,} online={online_steps:,} batch={batch_size} "
        f"online_frac={online_batch_fraction:.2f} online_bc={online_bc_coef_start}->{online_bc_coef_end}"
    )
    if label_path:
        print(
            f"         label_file={label_path} key={label_score_key} "
            f"weight_mean={label_info['label_weight_mean']:.3f}"
        )

    t0 = time.time()
    last_info: dict = {}

    try:
        for step in range(1, offline_steps + 1):
            batch = _sample_offline(dataset, offline_bc_weights, batch_size)
            last_info = td3_update(agent, batch, bc_coef=offline_bc_coef)
            if step % eval_freq_offline == 0 or step == offline_steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, algo_name, env_name, seed, step, "offline", metrics)
                log = {**last_info, **metrics, "step": step}
                print(
                    f"  offline step={step:>8,} | norm={metrics['normalized_score']:6.2f} "
                    f"| raw={metrics['raw_return']:8.1f} | bc_coef={offline_bc_coef:.3f} "
                    f"| t={time.time() - t0:.0f}s"
                )
                logger.log(log, step=step, context={"phase": "offline"})

        obs_raw = _safe_reset(online_env)
        for online_step in range(1, online_steps + 1):
            action = agent.act(obs_raw)
            noise = np.random.normal(0.0, exploration_noise * max_action, size=action.shape)
            noisy_action = np.clip(action + noise, -max_action, max_action).astype(np.float32)
            next_obs_raw, reward, done, _ = _safe_step(online_env, noisy_action)

            online_buffer.add(
                dataset.normalize_obs(obs_raw).astype(np.float32),
                noisy_action,
                dataset.normalize_obs(next_obs_raw).astype(np.float32),
                float(reward),
                bool(done),
            )
            obs_raw = _safe_reset(online_env) if done else next_obs_raw

            bc_coef = _linear_value(online_bc_coef_start, online_bc_coef_end, online_step, online_steps)
            batch = sample_mixed_batch(
                dataset,
                offline_bc_weights,
                online_buffer,
                batch_size=batch_size,
                online_fraction=online_batch_fraction,
            )
            gate_mode = (
                online_trust_gate
                if online_step >= int(online_gate_start_step)
                else "none"
            )
            last_info = td3_update(
                agent,
                batch,
                bc_coef=bc_coef,
                trust_gate=gate_mode,
                gate_temperature=online_gate_temperature,
                gate_min_weight=online_gate_min_weight,
                gate_margin=online_gate_margin,
            )

            global_step = offline_steps + online_step
            if online_step % eval_freq_online == 0 or online_step == online_steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, algo_name, env_name, seed, global_step, "online_finetune", metrics)
                log = {
                    **last_info,
                    **metrics,
                    "step": global_step,
                    "online_step": online_step,
                    "online_buffer_size": online_buffer.size,
                }
                print(
                    f"  online  step={online_step:>8,} | norm={metrics['normalized_score']:6.2f} "
                    f"| raw={metrics['raw_return']:8.1f} | bc_coef={bc_coef:.3f} "
                    f"| gate={gate_mode} | online_buf={online_buffer.size:,} | t={time.time() - t0:.0f}s"
                )
                logger.log(log, step=global_step, context={"phase": "online_finetune"})
    finally:
        logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--offline_steps", type=int, default=50_000)
    parser.add_argument("--online_steps", type=int, default=10_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq_offline", type=int, default=10_000)
    parser.add_argument("--eval_freq_online", type=int, default=1_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--result_dir", default="results/o2o_minimal")
    parser.add_argument("--algo_name", default="td3_bc_o2o")
    parser.add_argument("--aim", action="store_true")
    parser.add_argument("--wandb", action="store_true")

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

    parser.add_argument("--label_path", default=None)
    parser.add_argument("--label_score_key", default="trust_score")
    parser.add_argument("--label_min_weight", type=float, default=0.05)
    parser.add_argument("--online_batch_fraction", type=float, default=0.5)
    parser.add_argument("--online_replay_capacity", type=int, default=200_000)
    parser.add_argument("--exploration_noise", type=float, default=0.1)
    parser.add_argument("--offline_bc_coef", type=float, default=1.0)
    parser.add_argument("--online_bc_coef_start", type=float, default=1.0)
    parser.add_argument("--online_bc_coef_end", type=float, default=0.0)
    parser.add_argument("--online_trust_gate", choices=["none", "qgap"], default="none")
    parser.add_argument("--online_gate_temperature", type=float, default=10.0)
    parser.add_argument("--online_gate_min_weight", type=float, default=0.0)
    parser.add_argument("--online_gate_margin", type=float, default=0.0)
    parser.add_argument("--online_gate_start_step", type=int, default=1)
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
    train_o2o(
        env_name=args.env,
        seed=args.seed,
        offline_steps=args.offline_steps,
        online_steps=args.online_steps,
        batch_size=args.batch_size,
        eval_freq_offline=args.eval_freq_offline,
        eval_freq_online=args.eval_freq_online,
        eval_episodes_n=args.eval_episodes,
        result_dir=args.result_dir,
        use_aim=args.aim,
        use_wandb=args.wandb,
        config=config,
        algo_name=args.algo_name,
        label_path=args.label_path,
        label_score_key=args.label_score_key,
        label_min_weight=args.label_min_weight,
        online_batch_fraction=args.online_batch_fraction,
        online_replay_capacity=args.online_replay_capacity,
        exploration_noise=args.exploration_noise,
        offline_bc_coef=args.offline_bc_coef,
        online_bc_coef_start=args.online_bc_coef_start,
        online_bc_coef_end=args.online_bc_coef_end,
        online_trust_gate=args.online_trust_gate,
        online_gate_temperature=args.online_gate_temperature,
        online_gate_min_weight=args.online_gate_min_weight,
        online_gate_margin=args.online_gate_margin,
        online_gate_start_step=args.online_gate_start_step,
    )


if __name__ == "__main__":
    main()
