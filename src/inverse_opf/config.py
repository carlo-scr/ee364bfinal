"""Unified configuration system.

Every hyperparameter for any experiment lives on one of the nested
dataclasses below.  Configs are loaded from YAML and validated by
:func:`ExperimentConfig.from_yaml`.

Example YAML::

    experiment: methods_comparison
    run_name: headline_5seed
    seeds: [0, 1, 2, 3, 4]
    network: {n_buses: 10, n_lines: 14}
    data: {n_train: 200, n_val: 100, obs_noise: 0.5}
    model: {n_strata: 24}
    training: {steps: 600, lr: 0.05, loss: huber}
    forward: {tau: 0.01, eps: 0.1, physics: transport}

Unknown top-level keys raise; unknown nested keys are forwarded to the
matching dataclass kwargs so each experiment can declare its own
extension dict under ``extra:``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# ----- Nested config dataclasses ---------------------------------------------


@dataclass
class NetworkConfig:
    n_buses: int = 10
    n_lines: int = 14


@dataclass
class TrueParamsConfig:
    f_min: float = 5.0
    f_max: float = 60.0
    gmin_scale: float = 1.05
    gmax_scale: float = 1.30
    pmin_scale: float = 0.8
    pmax_scale: float = 1.2


@dataclass
class DataConfig:
    n_train: int = 200
    n_val: int = 100
    n_test: int = 0
    demand_mean: float = 40.0
    demand_std: float = 8.0
    obs_noise: float = 0.5
    diurnal_amp: float = 0.0
    diurnal_cost_amp: float = 0.20
    missing_frac: float = 0.0  # for missing-data experiment


@dataclass
class ModelConfig:
    n_strata: int = 24
    use_stratified_costs: bool = False
    f_init: float = 20.0
    gmax_init: float = 80.0
    pmax_init: float = 120.0


@dataclass
class TrainingHyper:
    steps: int = 600
    lr: float = 5e-2
    lr_min: float = 1e-4
    lr_schedule: str = "cosine"  # "none" | "cosine"
    warmup_frac: float = 0.0
    loss: str = "huber"
    huber_delta: float = 1.0
    l2_weight: float = 1e-5
    laplacian_weight: float = 0.05
    slack_l1_weight: float = 1e-3
    clip_grad_norm: float = 5.0
    early_stopping_patience: int = 120
    restore_best: bool = True


@dataclass
class ForwardConfig:
    physics: str = "transport"  # "transport" | "dc"
    tau: float = 1e-2
    eps: float = 1e-1
    slack_lift: bool = False
    slack_lift_weight: float = 1.0


@dataclass
class ExperimentConfig:
    experiment: str
    run_name: str
    seeds: list[int] = field(default_factory=lambda: [0])
    network: NetworkConfig = field(default_factory=NetworkConfig)
    true_params: TrueParamsConfig = field(default_factory=TrueParamsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingHyper = field(default_factory=TrainingHyper)
    forward: ForwardConfig = field(default_factory=ForwardConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    # ----- I/O -------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        raw = dict(raw)  # shallow copy
        if "experiment" not in raw:
            raise ValueError("config must specify 'experiment' (registry key)")
        if "run_name" not in raw:
            raise ValueError("config must specify 'run_name'")

        nested_cls = {
            "network": NetworkConfig,
            "true_params": TrueParamsConfig,
            "data": DataConfig,
            "model": ModelConfig,
            "training": TrainingHyper,
            "forward": ForwardConfig,
        }
        kwargs: dict[str, Any] = {
            "experiment": raw.pop("experiment"),
            "run_name": raw.pop("run_name"),
            "seeds": list(raw.pop("seeds", [0])),
            "extra": dict(raw.pop("extra", {})),
        }
        for key, sub_cls in nested_cls.items():
            sub_raw = raw.pop(key, None) or {}
            _check_unknown(key, sub_raw, sub_cls)
            kwargs[key] = sub_cls(**sub_raw)
        if raw:
            raise ValueError(f"unknown top-level config keys: {sorted(raw.keys())}")
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_unknown(prefix: str, data: dict[str, Any], cls: type) -> None:
    allowed = {f.name for f in fields(cls)}
    extra = set(data) - allowed
    if extra:
        raise ValueError(
            f"unknown keys in '{prefix}': {sorted(extra)} (allowed: {sorted(allowed)})"
        )
