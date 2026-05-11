#!/usr/bin/env python3
"""Export W&B runs to local CSV/JSONL files.

This script is intentionally small and dependency-light. It only needs the
existing ``wandb`` package from requirements.txt and writes plain files that can
be inspected, archived, or loaded later by pandas/rliable.

Examples:
    python scripts/export_wandb.py --entity <your-team-or-username> --project taoyao-rl
    WANDB_ENTITY=taoyao-team python scripts/export_wandb.py --history-keys step,normalized_score,raw_return
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable


def jsonable(value: Any) -> Any:
    """Convert common W&B/numpy-ish values to JSON-serializable objects."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def plain_dict(obj: Any) -> dict[str, Any]:
    """Best-effort conversion for wandb config/summary objects."""
    if obj is None:
        return {}
    if hasattr(obj, "_json_dict"):
        data = obj._json_dict
    else:
        try:
            data = dict(obj)
        except Exception:
            data = {}
    return {str(k): jsonable(v) for k, v in data.items() if not str(k).startswith("_")}


def is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def flatten_scalars(prefix: str, data: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}.{k}": v for k, v in data.items() if is_scalar(v)}


def safe_name(text: str) -> str:
    text = text.strip() or "unnamed"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)[:120]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(jsonable(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    preferred = ["step", "_step", "normalized_score", "raw_return", "episode_length", "return_std"]
    keys = [k for k in preferred if k in keys] + [k for k in keys if k not in preferred]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export W&B runs for local analysis.")
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"),
                        help="W&B username/team. Defaults to WANDB_ENTITY.")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "taoyao-rl"),
                        help="W&B project name. Defaults to WANDB_PROJECT or taoyao-rl.")
    parser.add_argument("--out", default="results/wandb_export",
                        help="Output directory.")
    parser.add_argument("--history-keys", default="",
                        help="Comma-separated metric keys. Empty exports all available history keys.")
    parser.add_argument("--max-runs", type=int, default=0,
                        help="Limit number of runs; 0 means no limit.")
    parser.add_argument("--state", default="",
                        help="Optional run state filter, e.g. finished, running, failed.")
    parser.add_argument("--filters-json", default="",
                        help="Extra W&B API filters as JSON, merged with --state.")
    parser.add_argument("--order", default="-created_at",
                        help="W&B API run ordering, e.g. -created_at or -summary_metrics.normalized_score.")
    parser.add_argument("--page-size", type=int, default=1000,
                        help="History scan page size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.entity:
        raise SystemExit("Missing --entity. Set WANDB_ENTITY or pass --entity <username-or-team>.")

    import wandb

    filters: dict[str, Any] = {}
    if args.filters_json:
        filters.update(json.loads(args.filters_json))
    if args.state:
        filters["state"] = args.state

    out_dir = Path(args.out)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    path = f"{args.entity}/{args.project}"
    runs = api.runs(path=path, filters=filters or None, order=args.order)
    history_keys = [k.strip() for k in args.history_keys.split(",") if k.strip()]

    summary_rows: list[dict[str, Any]] = []
    exported = 0
    for run in runs:
        if args.max_runs and exported >= args.max_runs:
            break

        config = plain_dict(run.config)
        summary = plain_dict(run.summary)
        run_file_prefix = f"{safe_name(run.name)}__{run.id}"

        write_json(runs_dir / f"{run_file_prefix}_config.json", config)
        write_json(runs_dir / f"{run_file_prefix}_summary.json", summary)

        scan_kwargs: dict[str, Any] = {"page_size": args.page_size}
        if history_keys:
            scan_kwargs["keys"] = history_keys
        history_rows = [jsonable(row) for row in run.scan_history(**scan_kwargs)]
        history_jsonl = runs_dir / f"{run_file_prefix}_history.jsonl"
        write_jsonl(history_jsonl, history_rows)
        if history_rows:
            write_csv(runs_dir / f"{run_file_prefix}_history.csv", history_rows)

        row = {
            "entity": args.entity,
            "project": args.project,
            "run_id": run.id,
            "name": run.name,
            "state": run.state,
            "url": run.url,
            "created_at": str(getattr(run, "created_at", "")),
            "updated_at": str(getattr(run, "updated_at", "")),
            "group": getattr(run, "group", None),
            "job_type": getattr(run, "job_type", None),
            "tags": ",".join(getattr(run, "tags", []) or []),
            "history_rows": len(history_rows),
        }
        row.update(flatten_scalars("config", config))
        row.update(flatten_scalars("summary", summary))
        summary_rows.append(row)

        print(f"exported {run.name} ({run.id}) history_rows={len(history_rows)}")
        exported += 1

    write_json(out_dir / "manifest.json", {
        "entity": args.entity,
        "project": args.project,
        "filters": filters,
        "order": args.order,
        "history_keys": history_keys,
        "runs": exported,
    })
    if summary_rows:
        write_csv(out_dir / "runs_summary.csv", summary_rows)

    print(f"Done. Exported {exported} run(s) from {path} to {out_dir}")


if __name__ == "__main__":
    main()
