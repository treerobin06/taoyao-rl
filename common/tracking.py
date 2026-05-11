"""Optional experiment tracking backends.

JSONL files remain the source of truth for aggregation. This module adds light
adapters for local Aim tracking and optional W&B cloud sync so algorithms do not
need to duplicate backend-specific code.
"""
from __future__ import annotations

import os
from typing import Any


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _clean_config(config: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in config.items():
        if _is_scalar(value):
            out[key] = value
        elif isinstance(value, dict):
            out[key] = _clean_config(value)
        else:
            out[key] = str(value)
    return out


class ExperimentLogger:
    """Log metrics to Aim and/or W&B.

    The logger is intentionally optional. If both backends are disabled, calls
    become no-ops and training still writes normal JSONL records.
    """

    def __init__(self, algo: str, env_name: str, seed: int, config: dict[str, Any],
                 use_aim: bool = False, use_wandb: bool = False):
        self.algo = algo
        self.env_name = env_name
        self.seed = seed
        self.config = _clean_config(config)
        self.run_name = f"{algo}_{env_name}_s{seed}"
        self.aim_run = None
        self.wandb_run = None

        if use_aim:
            self._init_aim()
        if use_wandb:
            self._init_wandb()

    def _init_aim(self) -> None:
        try:
            from aim import Run
        except ImportError as exc:
            raise RuntimeError(
                "Aim tracking requested but aim is not installed. "
                "Run `uv pip install -r requirements.txt` or set USE_AIM=0."
            ) from exc

        repo = os.environ.get("AIM_REPO", ".")
        experiment = os.environ.get("AIM_EXPERIMENT", os.environ.get("WANDB_PROJECT", "taoyao-rl"))
        self.aim_run = Run(
            repo=repo,
            experiment=experiment,
            system_tracking_interval=10,
            log_system_params=True,
            capture_terminal_logs=True,
        )
        self.aim_run.name = self.run_name
        self.aim_run["hparams"] = self.config

    def _init_wandb(self) -> None:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking requested but wandb is not installed. "
                "Run `uv pip install -r requirements.txt` or set USE_WANDB=0."
            ) from exc

        self.wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT", "taoyao-rl"),
            entity=os.environ.get("WANDB_ENTITY"),
            name=self.run_name,
            config=self.config,
        )

    def log(self, metrics: dict[str, Any], step: int | None = None,
            context: dict[str, str] | None = None) -> None:
        step = int(step if step is not None else metrics.get("step", 0))
        context = context or {}

        if self.aim_run is not None:
            for key, value in metrics.items():
                if key == "step" or value is None or not _is_scalar(value):
                    continue
                self.aim_run.track(value, name=key, step=step, context=context)

        if self.wandb_run is not None:
            import wandb
            wandb_metrics = dict(metrics)
            wandb_metrics.pop("step", None)
            wandb.log(wandb_metrics, step=step)

    def finish(self) -> None:
        if self.aim_run is not None:
            self.aim_run.close()
            self.aim_run = None
        if self.wandb_run is not None:
            import wandb
            wandb.finish()
            self.wandb_run = None
