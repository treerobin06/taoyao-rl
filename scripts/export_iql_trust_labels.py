#!/usr/bin/env python3
"""Export SSAR/IQL trust labels for each D4RL transition.

The SSAR IQL-qv cache stores only Q and V networks. SSAR marks a transition as
trusted when Q(s, a_data) >= V(s). This script reproduces that label export in
the shared project pipeline so downstream local algorithms can reuse or distill
the expensive teacher signal.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from common import D4RLDataset, get_obs_act_dims, make_env, set_seed


class Squeeze(nn.Module):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.squeeze(dim=self.dim)


class SSARMLP(nn.Module):
    def __init__(
        self,
        dims: list[int],
        activation_fn: Callable[[], nn.Module] = nn.ReLU,
        output_activation_fn: Optional[Callable[[], nn.Module]] = None,
        squeeze_output: bool = False,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation_fn())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        if output_activation_fn is not None:
            layers.append(output_activation_fn())
        if squeeze_output:
            layers.append(Squeeze(-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SSARTwinQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [obs_dim + act_dim, *([hidden] * n_hidden), 1]
        self.q1 = SSARMLP(dims, squeeze_output=True)
        self.q2 = SSARMLP(dims, squeeze_output=True)

    def both(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x), self.q2(x)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, act)
        return torch.minimum(q1, q2)


class SSARValue(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256, n_hidden: int = 2):
        super().__init__()
        dims = [obs_dim, *([hidden] * n_hidden), 1]
        self.v = SSARMLP(dims, squeeze_output=True)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.v(obs)


def load_teacher(checkpoint: str, obs_dim: int, act_dim: int, device: str) -> tuple[SSARTwinQ, SSARValue]:
    state = torch.load(checkpoint, map_location=device)
    if "qf" not in state or "vf" not in state:
        raise KeyError(f"{checkpoint} does not look like an SSAR IQL-qv checkpoint")
    qf = SSARTwinQ(obs_dim, act_dim).to(device)
    vf = SSARValue(obs_dim).to(device)
    qf.load_state_dict(state["qf"])
    vf.load_state_dict(state["vf"])
    qf.eval()
    vf.eval()
    return qf, vf


@torch.no_grad()
def score_dataset(
    dataset: D4RLDataset,
    qf: SSARTwinQ,
    vf: SSARValue,
    batch_size: int,
    device: str,
    temperature: float,
    threshold: float,
) -> dict[str, np.ndarray]:
    qs: list[np.ndarray] = []
    vs: list[np.ndarray] = []
    advs: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    hard: list[np.ndarray] = []
    temp = max(float(temperature), 1e-6)

    for start in range(0, dataset.size, batch_size):
        end = min(start + batch_size, dataset.size)
        obs = torch.as_tensor(dataset.observations[start:end], dtype=torch.float32, device=device)
        act = torch.as_tensor(dataset.actions[start:end], dtype=torch.float32, device=device)
        q = qf(obs, act)
        v = vf(obs)
        adv = q - v
        score = torch.sigmoid((adv - threshold) / temp)
        qs.append(q.detach().cpu().numpy())
        vs.append(v.detach().cpu().numpy())
        advs.append(adv.detach().cpu().numpy())
        scores.append(score.detach().cpu().numpy())
        hard.append((adv >= threshold).detach().cpu().numpy().astype(np.float32))

    return {
        "q": np.concatenate(qs).astype(np.float32),
        "v": np.concatenate(vs).astype(np.float32),
        "advantage": np.concatenate(advs).astype(np.float32),
        "trust_score": np.concatenate(scores).astype(np.float32),
        "hard_trust": np.concatenate(hard).astype(np.float32),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=16_384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_normalize_obs", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    env = make_env(args.env, seed=args.seed)
    obs_dim, act_dim = get_obs_act_dims(env)
    dataset = D4RLDataset(env, device="cpu", normalize_obs=not args.no_normalize_obs)
    qf, vf = load_teacher(args.checkpoint, obs_dim, act_dim, args.device)
    labels = score_dataset(
        dataset,
        qf,
        vf,
        batch_size=args.batch_size,
        device=args.device,
        temperature=args.temperature,
        threshold=args.threshold,
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "env": args.env,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "size": int(dataset.size),
        "temperature": args.temperature,
        "threshold": args.threshold,
        "normalize_obs": not args.no_normalize_obs,
        "created_at": int(time.time()),
        "hard_trust_fraction": float(labels["hard_trust"].mean()),
        "advantage_mean": float(labels["advantage"].mean()),
        "advantage_std": float(labels["advantage"].std()),
        "trust_score_mean": float(labels["trust_score"].mean()),
    }
    np.savez_compressed(
        out,
        **labels,
        obs_mean=dataset.obs_mean.astype(np.float32),
        obs_std=dataset.obs_std.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
