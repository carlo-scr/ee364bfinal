"""Synthetic dataset generators for inverse DC-OPF experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dc_opf import DCOpfLayer, DCOpfProblemData
from .graph import random_connected_incidence, random_susceptance


@dataclass
class SyntheticDataset:
    d_train: torch.Tensor
    g_train_obs: torch.Tensor
    strata_train: torch.Tensor
    d_val: torch.Tensor
    g_val_obs: torch.Tensor
    strata_val: torch.Tensor
    true_f: torch.Tensor
    true_f_table: torch.Tensor
    true_gmax: torch.Tensor
    true_pmax: torch.Tensor
    incidence: np.ndarray
    susceptance: np.ndarray
    physics: str = "transport"


def _sample_demands(n_samples: int, n_buses: int, mean: float, std: float,
                    rng: np.random.Generator,
                    diurnal_amp: float = 0.0,
                    strata: np.ndarray | None = None,
                    n_strata: int = 1) -> np.ndarray:
    base = rng.normal(mean, std, size=(n_samples, n_buses))
    if diurnal_amp > 0.0 and strata is not None and n_strata > 1:
        t = strata.astype(float) / float(n_strata)
        # Bus-specific phase so different demand peaks shift through the day.
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_buses,))
        load_shape = 1.0 + diurnal_amp * np.sin(2.0 * np.pi * t[:, None] + phase[None, :])
        base = base * load_shape
    return np.clip(base, 1.0, None)


def make_synthetic_dataset(
    n_buses: int,
    n_lines: int,
    n_train: int,
    n_val: int,
    demand_mean: float,
    demand_std: float,
    observation_noise_std: float,
    true_f_range: tuple[float, float],
    gmax_scale: tuple[float, float],
    pmax_scale: tuple[float, float],
    n_strata: int,
    seed: int,
    physics: str = "transport",
    diurnal_amp: float = 0.0,
    diurnal_cost_amp: float = 0.25,
    gmax_abs_range: tuple[float, float] | None = None,
) -> SyntheticDataset:
    rng = np.random.default_rng(seed)
    incidence = random_connected_incidence(n_buses, n_lines, rng)
    susceptance = random_susceptance(n_lines, rng)

    n_total = n_train + n_val
    strata = rng.integers(0, n_strata, size=(n_total,))

    d_total = _sample_demands(
        n_total, n_buses, demand_mean, demand_std, rng,
        diurnal_amp=diurnal_amp, strata=strata, n_strata=n_strata,
    )

    base_f = rng.uniform(true_f_range[0], true_f_range[1], size=(n_buses,))
    if n_strata <= 1:
        true_f_table = base_f.reshape(1, -1)
    else:
        t = np.arange(n_strata, dtype=float) / float(n_strata)
        amp = rng.uniform(0.05, max(0.06, diurnal_cost_amp), size=(n_buses,))
        phase = rng.uniform(0.0, 2.0 * np.pi, size=(n_buses,))
        mod = 1.0 + amp[None, :] * np.sin(2.0 * np.pi * t[:, None] + phase[None, :])
        true_f_table = np.clip(base_f[None, :] * mod, true_f_range[0], true_f_range[1])

    true_f = true_f_table.mean(axis=0)

    peak_demand = np.max(d_total, axis=0)
    if gmax_abs_range is not None:
        # Absolute (tight) capacity sampling, independent of peak demand.
        # Used by the identifiability experiment to ensure many generators
        # actually bind their capacity constraint.
        g_lo, g_hi = float(gmax_abs_range[0]), float(gmax_abs_range[1])
        true_gmax = rng.uniform(g_lo, g_hi, size=(n_buses,))
        # Guarantee feasibility: total capacity must exceed peak total demand
        # by a small slack. If not, scale uniformly upward (preserves the
        # relative tight/loose pattern across generators).
        peak_total = float(np.sum(d_total, axis=1).max())
        cap_total = float(true_gmax.sum())
        required = 1.05 * peak_total
        if cap_total < required:
            true_gmax = true_gmax * (required / cap_total)
    else:
        g_low = max(1.05, float(gmax_scale[0]))
        g_high = max(g_low + 1e-6, float(gmax_scale[1]))
        true_gmax = rng.uniform(g_low, g_high, size=(n_buses,)) * peak_demand

    p_low = max(0.8, float(pmax_scale[0]))
    p_high = max(p_low + 1e-6, float(pmax_scale[1]))
    avg_bus_peak = float(np.mean(peak_demand))
    true_pmax = rng.uniform(p_low, p_high, size=(n_lines,)) * avg_bus_peak

    data = DCOpfProblemData(incidence=incidence, susceptance=susceptance)
    opf = DCOpfLayer(data, physics=physics)

    d_torch = torch.tensor(d_total, dtype=torch.float32)
    true_f_table_t = torch.tensor(true_f_table, dtype=torch.float32)
    f_torch = true_f_table_t[torch.tensor(strata, dtype=torch.long)]
    gmax_torch = torch.tensor(true_gmax, dtype=torch.float32).unsqueeze(0).repeat(n_total, 1)
    pmax_torch = torch.tensor(true_pmax, dtype=torch.float32).unsqueeze(0).repeat(n_total, 1)

    with torch.no_grad():
        g_star, _ = opf.solve(d_torch, f_torch, gmax_torch, pmax_torch)

    noise = torch.randn_like(g_star) * observation_noise_std
    g_obs = torch.relu(g_star + noise)

    return SyntheticDataset(
        d_train=d_torch[:n_train],
        g_train_obs=g_obs[:n_train],
        strata_train=torch.tensor(strata[:n_train], dtype=torch.long),
        d_val=d_torch[n_train:],
        g_val_obs=g_obs[n_train:],
        strata_val=torch.tensor(strata[n_train:], dtype=torch.long),
        true_f=torch.tensor(true_f, dtype=torch.float32),
        true_f_table=true_f_table_t,
        true_gmax=torch.tensor(true_gmax, dtype=torch.float32),
        true_pmax=torch.tensor(true_pmax, dtype=torch.float32),
        incidence=incidence,
        susceptance=susceptance,
        physics=physics,
    )


# -----------------------------------------------------------------------------
# PJM-like single-bus stack (8 fuels). Realistic-ish heat rates / capacities.
# -----------------------------------------------------------------------------

PJM_FUELS = [
    # name,            cost ($/MWh),  capacity (GW)
    ("nuclear",         8.0,  35.0),
    ("hydro",          12.0,   8.0),
    ("wind",           15.0,  15.0),
    ("solar",          18.0,  12.0),
    ("coal",           28.0,  45.0),
    ("ccgt",           35.0,  60.0),
    ("oil_st",         85.0,   8.0),
    ("ct_peaker",     140.0,  15.0),
]


def make_pjm_like_dataset(
    n_train: int = 600,
    n_val: int = 200,
    seed: int = 0,
    observation_noise_frac: float = 0.02,
    n_strata: int = 24,
    diurnal_load_amp: float = 0.35,
    weekly_amp: float = 0.05,
    renewable_curtailment: bool = True,
) -> SyntheticDataset:
    """Single-bus, 8-fuel PJM-like stack with diurnally-varying load and
    renewable availability that effectively shifts the merit order by hour.

    All "buses" map to fuel buckets; we use a star network with one demand
    bus connected to each generator bus by a high-capacity line (so the
    transport model degenerates to a single-bus economic dispatch).
    """
    rng = np.random.default_rng(seed)
    n_fuel = len(PJM_FUELS)
    n_buses = n_fuel + 1                 # 1 extra "load" bus
    load_bus = n_fuel
    n_lines = n_fuel
    incidence = np.zeros((n_buses, n_lines))
    for k in range(n_fuel):
        incidence[k, k] = 1.0
        incidence[load_bus, k] = -1.0
    susceptance = np.ones(n_lines)

    base_costs = np.array([f[1] for f in PJM_FUELS], dtype=float)
    fuel_caps  = np.array([f[2] for f in PJM_FUELS], dtype=float)

    n_total = n_train + n_val
    strata = rng.integers(0, n_strata, size=(n_total,))
    week_phase = rng.uniform(0.0, 2 * np.pi, size=(n_total,))

    # Realistic system load ~ 80-130 GW with diurnal + weekly noise.
    base_load = 100.0
    diurnal = 1.0 + diurnal_load_amp * np.sin(2 * np.pi * strata / n_strata - 1.0)
    weekly = 1.0 + weekly_amp * np.sin(week_phase)
    noise_load = 1.0 + 0.05 * rng.standard_normal(n_total)
    total_load = base_load * diurnal * weekly * noise_load

    # Renewable availability: solar peaks midday (stratum 12-16),
    # wind anti-correlated, hydro nearly flat. We *modulate gmax per sample*
    # via stratified caps, but for the basic generator we just inject this as
    # cost variation (cheaper when renewable available, more expensive when not).
    fuel_idx = {name: i for i, (name, _, _) in enumerate(PJM_FUELS)}
    t = np.arange(n_strata) / n_strata
    solar_avail = np.clip(np.sin(np.pi * (t - 0.25) / 0.5), 0.0, None)  # daylight bump
    wind_avail = 1.0 + 0.4 * np.sin(2 * np.pi * t + 1.0)

    # Build per-stratum cost table so that solar/wind are very cheap when
    # available and effectively unavailable otherwise (encode by raising cost).
    f_table = np.tile(base_costs, (n_strata, 1))
    s_idx = fuel_idx["solar"]; w_idx = fuel_idx["wind"]
    f_table[:, s_idx] = np.where(solar_avail > 0.05,
                                  base_costs[s_idx] / np.clip(solar_avail, 0.05, None),
                                  500.0)
    f_table[:, w_idx] = base_costs[w_idx] / np.clip(wind_avail, 0.2, None)
    if renewable_curtailment:
        # also slightly modulate ccgt (gas) by hour (gas price wiggle)
        cc = fuel_idx["ccgt"]
        f_table[:, cc] = base_costs[cc] * (1.0 + 0.10 * np.sin(2 * np.pi * t + 0.5))

    # n_buses parameter for the inverse OPF includes the load bus; it has
    # zero generation cap and is *forced* to dispatch zero (cheap to encode by
    # giving it a tiny gmax).
    true_f = np.concatenate([f_table.mean(axis=0), [1e3]])
    true_f_table_full = np.zeros((n_strata, n_buses))
    true_f_table_full[:, :n_fuel] = f_table
    true_f_table_full[:, load_bus] = 1e3
    true_gmax = np.concatenate([fuel_caps, [1e-3]])
    # Lines sized well above peak load.
    true_pmax = np.full(n_lines, 1.5 * total_load.max())

    # Build per-bus demand: only the load bus has nonzero demand.
    d_total = np.zeros((n_total, n_buses))
    d_total[:, load_bus] = total_load

    # Solve forward for ground-truth dispatch.
    data = DCOpfProblemData(incidence=incidence, susceptance=susceptance)
    opf = DCOpfLayer(data, physics="transport")
    d_torch = torch.tensor(d_total, dtype=torch.float32)
    f_table_t = torch.tensor(true_f_table_full, dtype=torch.float32)
    f_torch = f_table_t[torch.tensor(strata, dtype=torch.long)]
    gmax_torch = torch.tensor(true_gmax, dtype=torch.float32).unsqueeze(0).repeat(n_total, 1)
    pmax_torch = torch.tensor(true_pmax, dtype=torch.float32).unsqueeze(0).repeat(n_total, 1)
    with torch.no_grad():
        g_star, _ = opf.solve(d_torch, f_torch, gmax_torch, pmax_torch)

    # Multiplicative-style observation noise.
    sigma = observation_noise_frac * g_star.abs().mean().item()
    noise = torch.randn_like(g_star) * sigma
    g_obs = torch.relu(g_star + noise)

    return SyntheticDataset(
        d_train=d_torch[:n_train],
        g_train_obs=g_obs[:n_train],
        strata_train=torch.tensor(strata[:n_train], dtype=torch.long),
        d_val=d_torch[n_train:],
        g_val_obs=g_obs[n_train:],
        strata_val=torch.tensor(strata[n_train:], dtype=torch.long),
        true_f=torch.tensor(true_f, dtype=torch.float32),
        true_f_table=f_table_t,
        true_gmax=torch.tensor(true_gmax, dtype=torch.float32),
        true_pmax=torch.tensor(true_pmax, dtype=torch.float32),
        incidence=incidence,
        susceptance=susceptance,
        physics="transport",
    )
