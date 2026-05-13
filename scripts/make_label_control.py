#!/usr/bin/env python3
"""Create control label files for ATLAS/label-file ablations.

Controls keep the label-file interface unchanged while removing or weakening
the teacher signal. The main diagnostic is `shuffle`: it preserves the score
distribution but breaks the transition-to-score alignment.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _metadata_to_dict(value: np.ndarray | str | bytes | None) -> dict:
    if value is None:
        return {}
    try:
        if isinstance(value, np.ndarray):
            raw = value.item() if value.shape == () else value.tolist()
        else:
            raw = value
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            return json.loads(raw)
    except Exception:
        return {}
    return {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source .npz label file.")
    parser.add_argument("--output", required=True, help="Output .npz control label file.")
    parser.add_argument("--score_key", default="atlas_score")
    parser.add_argument(
        "--mode",
        choices=["shuffle", "random_uniform", "constant_mean", "random_subset"],
        default="shuffle",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hard_threshold", type=float, default=0.5)
    args = parser.parse_args()

    src = np.load(args.input, allow_pickle=False)
    if args.score_key not in src:
        raise KeyError(f"{args.input} does not contain {args.score_key}; available={src.files}")
    score = src[args.score_key].astype(np.float32).reshape(-1)
    rng = np.random.default_rng(args.seed)

    if args.mode == "shuffle":
        control = score.copy()
        rng.shuffle(control)
    elif args.mode == "random_uniform":
        control = rng.uniform(0.0, 1.0, size=score.shape).astype(np.float32)
    elif args.mode == "random_subset":
        trusted_fraction = float((score >= args.hard_threshold).mean())
        trusted_count = int(round(trusted_fraction * score.size))
        control = np.zeros_like(score, dtype=np.float32)
        if trusted_count > 0:
            chosen = rng.choice(score.size, size=trusted_count, replace=False)
            control[chosen] = 1.0
    else:
        control = np.full_like(score, float(score.mean()), dtype=np.float32)

    payload = {key: src[key] for key in src.files if key != "metadata"}
    payload[args.score_key] = control.astype(np.float32)
    payload["trust_score"] = control.astype(np.float32)
    payload["hard_trust"] = (control >= args.hard_threshold).astype(np.float32)

    metadata = _metadata_to_dict(src["metadata"] if "metadata" in src else None)
    metadata.update(
        {
            "control_mode": args.mode,
            "control_seed": args.seed,
            "source_file": args.input,
            "score_key": args.score_key,
            "size": int(score.size),
            "source_mean": float(score.mean()),
            "source_std": float(score.std()),
            "control_mean": float(control.mean()),
            "control_std": float(control.std()),
            "created_at": int(time.time()),
        }
    )
    payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
