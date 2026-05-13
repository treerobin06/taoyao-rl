"""Cheap trusted-action selectors on top of TD3+BC.

This is the local contribution-candidate after the baseline/source screen.
It tests whether cheap trust signals can replace part of SSAR's expensive
IQL-qv trusted action selection.

The algorithm keeps critic training on the full replay dataset, but weights the
actor BC regularizer by either a trajectory-return mask or an online Q-gap proxy.
These selectors are mechanism probes, not final paper claims.
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
import torch.nn.functional as F

from algorithms.td3_bc import TD3BCAgent, TD3BCConfig
from common import (
    D4RLDataset,
    ExperimentLogger,
    eval_episodes,
    get_obs_act_dims,
    make_env,
    set_seed,
    write_result,
)


@dataclass
class TrustedTD3BCConfig(TD3BCConfig):
    selector_mode: str = "return"
    top_trajectory_fraction: float = 0.2
    min_trajectory_return: Optional[float] = None
    untrusted_bc_weight: float = 0.05
    trust_bc_coef: float = 1.0
    boundary_eps: float = 1e-4
    qgap_temperature: float = 10.0
    qgap_min_weight: float = 0.05
    qgap_warmup_steps: int = 5_000
    qgap_use_min_q: bool = True
    consistency_threshold: float = 0.25
    consistency_temperature: float = 0.05
    consistency_min_weight: float = 0.05
    consistency_warmup_steps: int = 5_000
    label_path: Optional[str] = None
    label_score_key: str = "trust_score"
    label_min_weight: float = 0.05
    label_threshold: float = 0.5
    label_binarize: bool = False


def _trajectory_boundaries(dataset: D4RLDataset, eps: float) -> list[tuple[int, int, float]]:
    """Infer trajectory slices from qlearning transitions.

    qlearning_dataset removes timeout boundary transitions, so a discontinuity
    between next_obs[i] and obs[i + 1] marks a new trajectory. True terminals
    also mark a boundary.
    """
    slices: list[tuple[int, int, float]] = []
    start = 0
    size = dataset.size
    for i in range(size - 1):
        terminal = bool(dataset.dones[i])
        discontinuity = np.linalg.norm(dataset.observations[i + 1] - dataset.next_obs[i]) > eps
        if terminal or discontinuity:
            end = i + 1
            ret = float(dataset.rewards[start:end].sum())
            if end > start:
                slices.append((start, end, ret))
            start = i + 1
    if start < size:
        slices.append((start, size, float(dataset.rewards[start:size].sum())))
    return slices


def build_return_trust_weights(
    dataset: D4RLDataset,
    top_fraction: float,
    min_return: Optional[float],
    untrusted_weight: float,
    eps: float,
) -> tuple[np.ndarray, dict]:
    """Return per-transition BC weights and selector diagnostics."""
    top_fraction = float(np.clip(top_fraction, 0.0, 1.0))
    untrusted_weight = float(np.clip(untrusted_weight, 0.0, 1.0))
    trajs = _trajectory_boundaries(dataset, eps)

    weights = np.full((dataset.size, 1), untrusted_weight, dtype=np.float32)
    trusted = np.zeros(dataset.size, dtype=bool)

    if min_return is not None:
        for start, end, ret in trajs:
            if ret >= min_return:
                trusted[start:end] = True
    else:
        target = int(round(dataset.size * top_fraction))
        selected = 0
        for start, end, _ in sorted(trajs, key=lambda x: x[2], reverse=True):
            if selected >= target:
                break
            trusted[start:end] = True
            selected += end - start

    weights[trusted] = 1.0
    returns = np.asarray([ret for _, _, ret in trajs], dtype=np.float32)
    info = {
        "num_trajectories": len(trajs),
        "trusted_fraction": float(trusted.mean()) if dataset.size else 0.0,
        "trusted_transitions": int(trusted.sum()),
        "untrusted_bc_weight": untrusted_weight,
        "trajectory_return_min": float(returns.min()) if len(returns) else 0.0,
        "trajectory_return_mean": float(returns.mean()) if len(returns) else 0.0,
        "trajectory_return_max": float(returns.max()) if len(returns) else 0.0,
    }
    if min_return is not None:
        info["min_trajectory_return"] = float(min_return)
    else:
        info["top_trajectory_fraction"] = top_fraction
    return weights, info


def build_label_trust_weights(
    dataset: D4RLDataset,
    label_path: str,
    score_key: str,
    min_weight: float,
    threshold: float,
    binarize: bool,
) -> tuple[np.ndarray, dict]:
    """Load per-transition teacher/selector labels as BC weights."""
    labels = np.load(label_path, allow_pickle=False)
    if score_key not in labels:
        raise KeyError(f"{label_path} does not contain key {score_key}; available={labels.files}")
    score = labels[score_key].astype(np.float32).reshape(-1, 1)
    if len(score) != dataset.size:
        raise ValueError(f"label length {len(score)} != dataset size {dataset.size}")

    score = np.nan_to_num(score, nan=0.0, posinf=1.0, neginf=0.0)
    if binarize:
        score = (score >= float(threshold)).astype(np.float32)
    score = np.clip(score, 0.0, 1.0)
    min_weight = float(np.clip(min_weight, 0.0, 1.0))
    weights = min_weight + (1.0 - min_weight) * score
    trusted = score >= float(threshold)
    info = {
        "label_path": label_path,
        "label_score_key": score_key,
        "label_min_weight": min_weight,
        "label_threshold": float(threshold),
        "label_binarize": bool(binarize),
        "trusted_fraction": float(trusted.mean()) if dataset.size else 0.0,
        "label_score_min": float(score.min()) if dataset.size else 0.0,
        "label_score_mean": float(score.mean()) if dataset.size else 0.0,
        "label_score_max": float(score.max()) if dataset.size else 0.0,
        "label_weight_mean": float(weights.mean()) if dataset.size else 0.0,
    }
    return weights.astype(np.float32), info


class TrustedTD3BCAgent(TD3BCAgent):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_action: float,
        config: TrustedTD3BCConfig,
        device: str,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
    ):
        super().__init__(obs_dim, act_dim, max_action, config, device, obs_mean, obs_std)
        self.config: TrustedTD3BCConfig = config

    @torch.no_grad()
    def _batch_trust_weight(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, dict]:
        if self.config.selector_mode not in {"qgap_soft", "consistency"}:
            raise ValueError(f"_batch_trust_weight got unsupported selector_mode={self.config.selector_mode}")

        warmup_steps = (
            self.config.qgap_warmup_steps
            if self.config.selector_mode == "qgap_soft"
            else self.config.consistency_warmup_steps
        )
        if self.total_it <= warmup_steps:
            warm = torch.ones((obs.shape[0], 1), dtype=obs.dtype, device=obs.device)
            return warm, {
                "qgap_mean": 0.0,
                "qgap_std": 0.0,
                "q_data_mean": 0.0,
                "q_pi_mean": 0.0,
                "qgap_weight_min": 1.0,
                "qgap_weight_max": 1.0,
                "qgap_warmup": 1.0,
                "consistency_mse_mean": 0.0,
                "consistency_weight_min": 1.0,
                "consistency_weight_max": 1.0,
                "consistency_warmup": 1.0,
            }

        pi = self.actor(obs)
        if self.config.selector_mode == "consistency":
            mse = ((pi - act) ** 2).mean(dim=-1, keepdim=True)
            temp = max(float(self.config.consistency_temperature), 1e-6)
            soft = torch.sigmoid((float(self.config.consistency_threshold) - mse) / temp)
            min_w = float(np.clip(self.config.consistency_min_weight, 0.0, 1.0))
            weight = min_w + (1.0 - min_w) * soft
            return weight.detach(), {
                "consistency_mse_mean": float(mse.mean().item()),
                "consistency_weight_min": float(weight.min().item()),
                "consistency_weight_max": float(weight.max().item()),
                "consistency_warmup": 0.0,
            }

        data_q1, data_q2 = self.critic(obs, act)
        pi_q1, pi_q2 = self.critic(obs, pi)
        if self.config.qgap_use_min_q:
            data_q = torch.min(data_q1, data_q2)
            pi_q = torch.min(pi_q1, pi_q2)
        else:
            data_q = data_q1
            pi_q = pi_q1

        q_gap = data_q - pi_q
        temp = max(float(self.config.qgap_temperature), 1e-6)
        soft = torch.sigmoid(q_gap / temp)
        min_w = float(np.clip(self.config.qgap_min_weight, 0.0, 1.0))
        weight = min_w + (1.0 - min_w) * soft
        return weight.detach(), {
            "qgap_mean": float(q_gap.mean().item()),
            "qgap_std": float(q_gap.std(unbiased=False).item()),
            "q_data_mean": float(data_q.mean().item()),
            "q_pi_mean": float(pi_q.mean().item()),
            "qgap_weight_min": float(weight.min().item()),
            "qgap_weight_max": float(weight.max().item()),
            "qgap_warmup": 0.0,
        }

    def update(self, batch: dict) -> dict:
        self.total_it += 1

        obs = batch["obs"]
        act = batch["act"]
        rew = batch["rew"]
        next_obs = batch["next_obs"]
        done = batch["done"]
        trust_weight = batch["trust_weight"]

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

        trust_extra: dict = {}
        if self.config.selector_mode in {"qgap_soft", "consistency"}:
            trust_weight, trust_extra = self._batch_trust_weight(obs, act)

        info = {
            "critic_loss": float(critic_loss.item()),
            "q_mean": float(cur_q1.detach().mean().item()),
            "batch_trust_weight": float(trust_weight.mean().item()),
            **trust_extra,
        }

        if self.total_it % self.config.policy_freq == 0:
            pi = self.actor(obs)
            q = self.critic.q1_only(obs, pi)
            lam = self.config.alpha / q.abs().mean().detach().clamp(min=1e-6)

            per_sample_bc = ((pi - act) ** 2).mean(dim=-1, keepdim=True)
            bc_loss = (trust_weight * per_sample_bc).sum() / trust_weight.sum().clamp(min=1e-6)
            actor_loss = -lam * q.mean() + self.config.trust_bc_coef * bc_loss

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

            info.update({
                "actor_loss": float(actor_loss.item()),
                "bc_loss": float(bc_loss.item()),
                "lambda": float(lam.item()),
            })

        return info


def sample_trusted(dataset: D4RLDataset, trust_weights: np.ndarray, batch_size: int) -> dict:
    idx = np.random.randint(0, dataset.size, size=batch_size)
    return {
        "obs": torch.from_numpy(dataset.observations[idx]).to(dataset.device),
        "act": torch.from_numpy(dataset.actions[idx]).to(dataset.device),
        "rew": torch.from_numpy(dataset.rewards[idx]).to(dataset.device).unsqueeze(-1),
        "next_obs": torch.from_numpy(dataset.next_obs[idx]).to(dataset.device),
        "done": torch.from_numpy(dataset.dones[idx]).to(dataset.device).unsqueeze(-1),
        "trust_weight": torch.from_numpy(trust_weights[idx]).to(dataset.device),
    }


def _numeric_metrics(values: dict) -> dict:
    numeric_types = (int, float, bool, np.integer, np.floating, np.bool_)
    return {k: v for k, v in values.items() if isinstance(v, numeric_types)}


def train(env_name: str, seed: int, steps: int, batch_size: int,
          eval_freq: int, eval_episodes_n: int, result_dir: str,
          use_aim: bool, use_wandb: bool, config: TrustedTD3BCConfig,
          algo_name: str = "trusted_td3_bc"):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env(env_name, seed=seed)
    eval_env = make_env(env_name, seed=seed + 100)
    obs_dim, act_dim = get_obs_act_dims(env)
    max_action = float(env.action_space.high[0])

    dataset = D4RLDataset(env, device=device, normalize_obs=config.normalize_obs)
    if config.selector_mode == "return":
        trust_weights, trust_info = build_return_trust_weights(
            dataset,
            config.top_trajectory_fraction,
            config.min_trajectory_return,
            config.untrusted_bc_weight,
            config.boundary_eps,
        )
    elif config.selector_mode == "label_file":
        if not config.label_path:
            raise ValueError("selector_mode=label_file requires --label_path")
        trust_weights, trust_info = build_label_trust_weights(
            dataset,
            config.label_path,
            config.label_score_key,
            config.label_min_weight,
            config.label_threshold,
            config.label_binarize,
        )
    elif config.selector_mode in {"qgap_soft", "consistency"}:
        trust_weights = np.ones((dataset.size, 1), dtype=np.float32)
        trust_info = {
            "selector_mode": config.selector_mode,
            "qgap_temperature": config.qgap_temperature,
            "qgap_min_weight": config.qgap_min_weight,
            "qgap_warmup_steps": config.qgap_warmup_steps,
            "qgap_use_min_q": config.qgap_use_min_q,
            "consistency_threshold": config.consistency_threshold,
            "consistency_temperature": config.consistency_temperature,
            "consistency_min_weight": config.consistency_min_weight,
            "consistency_warmup_steps": config.consistency_warmup_steps,
        }
    else:
        raise ValueError(f"Unknown selector_mode={config.selector_mode}")

    agent = TrustedTD3BCAgent(
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
        config=dict(
            algo=algo_name,
            base_algo="trusted_td3_bc",
            env=env_name,
            seed=seed,
            steps=steps,
            **asdict(config),
            **{f"selector_{k}": v for k, v in trust_info.items()},
        ),
    )

    print(f"[{algo_name}] {env_name} | seed={seed} | device={device} | obs={obs_dim} act={act_dim}")
    if config.selector_mode == "return":
        print(
            f"         steps={steps:,} batch={batch_size} selector=return "
            f"trusted={trust_info['trusted_fraction']:.3f} trajs={trust_info['num_trajectories']} "
            f"untrusted_w={config.untrusted_bc_weight}"
        )
    elif config.selector_mode == "label_file":
        print(
            f"         steps={steps:,} batch={batch_size} selector=label_file "
            f"key={config.label_score_key} trusted={trust_info['trusted_fraction']:.3f} "
            f"weight_mean={trust_info['label_weight_mean']:.3f}"
        )
    elif config.selector_mode == "qgap_soft":
        print(
            f"         steps={steps:,} batch={batch_size} selector=qgap_soft "
            f"temp={config.qgap_temperature} min_w={config.qgap_min_weight} "
            f"warmup={config.qgap_warmup_steps}"
        )
    else:
        print(
            f"         steps={steps:,} batch={batch_size} selector=consistency "
            f"threshold={config.consistency_threshold} temp={config.consistency_temperature} "
            f"min_w={config.consistency_min_weight} warmup={config.consistency_warmup_steps}"
        )
    t0 = time.time()
    last_info: dict = {}

    try:
        for step in range(1, steps + 1):
            last_info = agent.update(sample_trusted(dataset, trust_weights, batch_size))

            if step % eval_freq == 0 or step == steps:
                metrics = eval_episodes(agent, eval_env, n_episodes=eval_episodes_n)
                write_result(result_dir, algo_name, env_name, seed, step, "offline", metrics)
                elapsed = time.time() - t0
                log = {**last_info, **metrics, "step": step, **_numeric_metrics(trust_info)}
                print(
                    f"  step={step:>8,} | norm={metrics['normalized_score']:6.2f} "
                    f"| raw={metrics['raw_return']:8.1f} | "
                    f"critic={last_info.get('critic_loss', float('nan')):.4f} "
                    f"| actor={last_info.get('actor_loss', float('nan')):.4f} "
                    f"| trust_w={last_info.get('batch_trust_weight', float('nan')):.3f} "
                    f"| qgap={last_info.get('qgap_mean', float('nan')):.3f} "
                    f"| cmse={last_info.get('consistency_mse_mean', float('nan')):.3f} "
                    f"| t={elapsed:.0f}s"
                )
                logger.log(log, step=step, context={"phase": "offline"})
    finally:
        logger.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_freq", type=int, default=10_000)
    parser.add_argument("--eval_episodes", type=int, default=5)
    parser.add_argument("--result_dir", default="results")
    parser.add_argument("--algo_name", default="trusted_td3_bc")
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

    parser.add_argument("--selector_mode", choices=["return", "qgap_soft", "consistency", "label_file"], default="return")
    parser.add_argument("--top_trajectory_fraction", type=float, default=0.2)
    parser.add_argument("--min_trajectory_return", type=float, default=None)
    parser.add_argument("--untrusted_bc_weight", type=float, default=0.05)
    parser.add_argument("--trust_bc_coef", type=float, default=1.0)
    parser.add_argument("--boundary_eps", type=float, default=1e-4)
    parser.add_argument("--qgap_temperature", type=float, default=10.0)
    parser.add_argument("--qgap_min_weight", type=float, default=0.05)
    parser.add_argument("--qgap_warmup_steps", type=int, default=5_000)
    parser.add_argument("--qgap_use_q1", action="store_true")
    parser.add_argument("--consistency_threshold", type=float, default=0.25)
    parser.add_argument("--consistency_temperature", type=float, default=0.05)
    parser.add_argument("--consistency_min_weight", type=float, default=0.05)
    parser.add_argument("--consistency_warmup_steps", type=int, default=5_000)
    parser.add_argument("--label_path", default=None)
    parser.add_argument("--label_score_key", default="trust_score")
    parser.add_argument("--label_min_weight", type=float, default=0.05)
    parser.add_argument("--label_threshold", type=float, default=0.5)
    parser.add_argument("--label_binarize", action="store_true")
    args = parser.parse_args()

    config = TrustedTD3BCConfig(
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
        selector_mode=args.selector_mode,
        top_trajectory_fraction=args.top_trajectory_fraction,
        min_trajectory_return=args.min_trajectory_return,
        untrusted_bc_weight=args.untrusted_bc_weight,
        trust_bc_coef=args.trust_bc_coef,
        boundary_eps=args.boundary_eps,
        qgap_temperature=args.qgap_temperature,
        qgap_min_weight=args.qgap_min_weight,
        qgap_warmup_steps=args.qgap_warmup_steps,
        qgap_use_min_q=not args.qgap_use_q1,
        consistency_threshold=args.consistency_threshold,
        consistency_temperature=args.consistency_temperature,
        consistency_min_weight=args.consistency_min_weight,
        consistency_warmup_steps=args.consistency_warmup_steps,
        label_path=args.label_path,
        label_score_key=args.label_score_key,
        label_min_weight=args.label_min_weight,
        label_threshold=args.label_threshold,
        label_binarize=args.label_binarize,
    )
    train(args.env, args.seed, args.steps, args.batch_size,
          args.eval_freq, args.eval_episodes, args.result_dir,
          args.aim, args.wandb, config, args.algo_name)


if __name__ == "__main__":
    main()
