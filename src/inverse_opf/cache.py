"""Tiny on-disk cache for experiment results.

Each ``(experiment, run_name, seed)`` triple owns one directory::

    outputs/<run_name>/<seed>/
        metrics.json     # the dict returned by the experiment function
        config.yaml      # the config used to produce it (for provenance)

If ``metrics.json`` exists, :func:`load_or_compute` returns it unless
``force=True``.  This lets long-running experiments be resumed seed-by-seed
without re-running expensive solves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml


def seed_dir(run_name: str, seed: int, root: str | Path = "outputs") -> Path:
    return Path(root) / run_name / str(seed)


def write_config(path: Path, config_dict: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)


def write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_jsonable)


def read_metrics(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_compute(
    run_name: str,
    seed: int,
    config_dict: dict[str, Any],
    fn: Callable[[], dict[str, Any]],
    *,
    root: str | Path = "outputs",
    force: bool = False,
) -> dict[str, Any]:
    """Return cached metrics if present (and not ``force``), else compute+save."""
    d = seed_dir(run_name, seed, root)
    metrics_path = d / "metrics.json"
    if metrics_path.exists() and not force:
        return read_metrics(metrics_path)
    write_config(d / "config.yaml", config_dict)
    metrics = fn()
    write_metrics(metrics_path, metrics)
    return metrics


def _jsonable(obj: Any):
    """JSON fallback for numpy / torch scalars and arrays."""
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().tolist()
    except ImportError:
        pass
    raise TypeError(f"object of type {type(obj).__name__} not JSON-serializable")
