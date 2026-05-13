#!/usr/bin/env python3
"""Train a small selector to amortize exported IQL/SSAR trust labels."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import D4RLDataset, get_obs_act_dims, make_env, set_seed


class AtlasSelector(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1))


def _load_target(labels_path: str, key: str, size: int) -> np.ndarray:
    labels = np.load(labels_path, allow_pickle=False)
    if key not in labels:
        raise KeyError(f"{labels_path} does not contain key {key}; available={labels.files}")
    target = labels[key].astype(np.float32).reshape(-1, 1)
    if len(target) != size:
        raise ValueError(f"label length {len(target)} != dataset size {size}")
    return target


@torch.no_grad()
def _predict_all(model: AtlasSelector, obs: np.ndarray, act: np.ndarray,
                 batch_size: int, device: str) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(obs), batch_size):
        end = min(start + batch_size, len(obs))
        obs_t = torch.as_tensor(obs[start:end], dtype=torch.float32, device=device)
        act_t = torch.as_tensor(act[start:end], dtype=torch.float32, device=device)
        preds.append(torch.sigmoid(model(obs_t, act_t)).cpu().numpy())
    return np.concatenate(preds).astype(np.float32).reshape(-1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="hopper-medium-replay-v2")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--target_key", default="hard_trust")
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_normalize_obs", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    env = make_env(args.env, seed=args.seed)
    obs_dim, act_dim = get_obs_act_dims(env)
    dataset = D4RLDataset(env, device="cpu", normalize_obs=not args.no_normalize_obs)
    target = _load_target(args.labels, args.target_key, dataset.size)

    rng = np.random.default_rng(args.seed)
    indices = np.arange(dataset.size)
    if args.max_samples and args.max_samples < dataset.size:
        indices = rng.choice(indices, size=args.max_samples, replace=False)
    rng.shuffle(indices)
    val_n = int(round(len(indices) * float(np.clip(args.val_fraction, 0.0, 0.5))))
    val_idx = indices[:val_n]
    train_idx = indices[val_n:]

    model = AtlasSelector(obs_dim, act_dim, args.hidden).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    binary_target = args.target_key == "hard_trust"

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_idx)
        losses: list[float] = []
        for start in range(0, len(train_idx), args.batch_size):
            batch_idx = train_idx[start:start + args.batch_size]
            obs = torch.as_tensor(dataset.observations[batch_idx], dtype=torch.float32, device=args.device)
            act = torch.as_tensor(dataset.actions[batch_idx], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(target[batch_idx], dtype=torch.float32, device=args.device)
            logits = model(obs, act)
            if binary_target:
                loss = F.binary_cross_entropy_with_logits(logits, y)
            else:
                loss = F.mse_loss(torch.sigmoid(logits), y.clamp(0.0, 1.0))
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        with torch.no_grad():
            if len(val_idx):
                obs = torch.as_tensor(dataset.observations[val_idx], dtype=torch.float32, device=args.device)
                act = torch.as_tensor(dataset.actions[val_idx], dtype=torch.float32, device=args.device)
                y = torch.as_tensor(target[val_idx], dtype=torch.float32, device=args.device)
                prob = torch.sigmoid(model(obs, act))
                if binary_target:
                    val_loss = F.binary_cross_entropy(prob.clamp(1e-6, 1 - 1e-6), y)
                    acc = ((prob >= 0.5) == (y >= 0.5)).float().mean()
                else:
                    val_loss = F.mse_loss(prob, y.clamp(0.0, 1.0))
                    acc = torch.tensor(float("nan"), device=args.device)
                print(
                    f"epoch={epoch} train_loss={np.mean(losses):.6f} "
                    f"val_loss={float(val_loss.item()):.6f} val_acc={float(acc.item()):.4f}"
                )
            else:
                print(f"epoch={epoch} train_loss={np.mean(losses):.6f}")

    atlas_score = _predict_all(model, dataset.observations, dataset.actions, args.batch_size, args.device)
    min_w = float(np.clip(args.min_weight, 0.0, 1.0))
    atlas_weight = (min_w + (1.0 - min_w) * atlas_score).astype(np.float32)

    pred_out = Path(args.output_predictions)
    pred_out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "env": args.env,
        "seed": args.seed,
        "labels": args.labels,
        "target_key": args.target_key,
        "size": int(dataset.size),
        "epochs": args.epochs,
        "max_samples": int(args.max_samples),
        "min_weight": min_w,
        "created_at": int(time.time()),
        "atlas_score_mean": float(atlas_score.mean()),
        "atlas_score_std": float(atlas_score.std()),
        "atlas_weight_mean": float(atlas_weight.mean()),
    }
    np.savez_compressed(
        pred_out,
        atlas_score=atlas_score,
        trust_score=atlas_weight,
        hard_trust=(atlas_score >= 0.5).astype(np.float32),
        teacher_target=target.reshape(-1).astype(np.float32),
        obs_mean=dataset.obs_mean.astype(np.float32),
        obs_std=dataset.obs_std.astype(np.float32),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )

    model_out = Path(args.output_model)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden": args.hidden,
            "metadata": metadata,
        },
        model_out,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {pred_out}")
    print(f"wrote {model_out}")


if __name__ == "__main__":
    main()
