#!/usr/bin/env python3
"""Project-local helper for the paper-illustration-image2 workflow.

This helper does not generate images. It only records preflight/finalize/verify
receipts around native image generation performed by the Codex image2 bridge.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def rel_or_abs(path: Path, workspace: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path.resolve())


def preflight(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    out_dir = workspace / "figures" / "ai_generated"
    out_dir.mkdir(parents=True, exist_ok=True)

    checks = {
        "workspace_exists": workspace.exists() and workspace.is_dir(),
        "output_dir_exists": out_dir.exists() and out_dir.is_dir(),
        "output_dir_writable": True,
    }
    probe = out_dir / ".write_probe"
    try:
        probe.write_text("ok\n")
        probe.unlink()
    except OSError:
        checks["output_dir_writable"] = False

    payload = {
        "ok": all(checks.values()),
        "timestamp": now_iso(),
        "workspace": str(workspace),
        "output_dir": rel_or_abs(out_dir, workspace),
        "checks": checks,
        "native_generation_required": True,
        "note": "This helper only validates paths; rendering must use codex-image2 native generation.",
    }
    write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def finalize(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    best_image = Path(args.best_image)
    if not best_image.is_absolute():
        best_image = workspace / best_image
    best_image = best_image.resolve()

    out_dir = workspace / "figures" / "ai_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    final_image = out_dir / "figure_final.png"

    if not best_image.exists():
        payload = {
            "ok": False,
            "timestamp": now_iso(),
            "error": f"best image not found: {best_image}",
        }
        write_json(out_dir / "review_log.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 1

    shutil.copyfile(best_image, final_image)

    latex = r"""\begin{figure*}[t]
    \centering
    \includegraphics[width=0.95\textwidth]{figures/ai_generated/figure_final.png}
    \caption{Trusted-action constraint transfer in offline-to-online reinforcement learning.}
    \label{fig:trusted-constraint-transfer}
\end{figure*}
"""
    (out_dir / "latex_include.tex").write_text(latex)

    payload = {
        "ok": True,
        "timestamp": now_iso(),
        "best_image": rel_or_abs(best_image, workspace),
        "final_image": rel_or_abs(final_image, workspace),
        "score": args.score,
        "review_summary": args.review_summary,
    }
    write_json(out_dir / "review_log.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


def verify(args: argparse.Namespace) -> int:
    workspace = Path(args.workspace).expanduser().resolve()
    out_dir = workspace / "figures" / "ai_generated"
    final_image = out_dir / "figure_final.png"
    latex_include = out_dir / "latex_include.tex"
    review_log = out_dir / "review_log.json"
    preflight_json = out_dir / "preflight.json"

    checks = {
        "final_image_exists": final_image.exists(),
        "final_image_nonempty": final_image.exists() and final_image.stat().st_size > 1024,
        "latex_include_exists": latex_include.exists(),
        "review_log_exists": review_log.exists(),
        "preflight_exists": preflight_json.exists(),
    }
    payload = {
        "ok": all(checks.values()),
        "timestamp": now_iso(),
        "workspace": str(workspace),
        "checks": checks,
        "artifacts": {
            "final_image": rel_or_abs(final_image, workspace),
            "latex_include": rel_or_abs(latex_include, workspace),
            "review_log": rel_or_abs(review_log, workspace),
            "preflight": rel_or_abs(preflight_json, workspace),
        },
    }
    write_json(Path(args.json_out), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preflight")
    p.add_argument("--workspace", required=True)
    p.add_argument("--json-out", required=True)
    p.set_defaults(func=preflight)

    p = sub.add_parser("finalize")
    p.add_argument("--workspace", required=True)
    p.add_argument("--best-image", required=True)
    p.add_argument("--score", type=float, default=9.0)
    p.add_argument("--review-summary", default="Accepted after strict review.")
    p.set_defaults(func=finalize)

    p = sub.add_parser("verify")
    p.add_argument("--workspace", required=True)
    p.add_argument("--json-out", required=True)
    p.set_defaults(func=verify)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
