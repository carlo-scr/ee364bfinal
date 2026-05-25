"""Post-hoc analysis helpers used by Section 2/3/6 experiments.

Each function is pure (no side effects, no plotting) and returns plain
``numpy``/``dict`` objects so they can be cached and tabulated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .dc_opf import DCOpfLayer, DCOpfProblemData


# ---------------------------------------------------------------------------
# Congestion / binding-constraint statistics  (Section 2)
# ---------------------------------------------------------------------------


def congestion_stats(opf: DCOpfLayer, d: torch.Tensor, f: torch.Tensor,
                     gmax: torch.Tensor, pmax: torch.Tensor,
                     tol_frac: float = 1e-2) -> dict:
    """Fraction of samples in which each line/generator constraint binds.

    Parameters
    ----------
    opf : DCOpfLayer
    d : (T, n_buses) demand tensor
    f, gmax, pmax : 1-D parameter tensors (will be broadcast over T)
    tol_frac : a constraint is "binding" if the slack is within
        ``tol_frac * capacity``.

    Returns dict with:
      - line_freq : (n_lines,) per-line binding fraction
      - gen_freq  : (n_buses,) per-gen upper-bound binding fraction
      - line_any_frac : fraction of samples with >=1 binding line
      - gen_any_frac  : fraction of samples with >=1 binding gen-upper
    """
    T = d.shape[0]
    f_b = f.unsqueeze(0).repeat(T, 1) if f.dim() == 1 else f
    gmax_b = gmax.unsqueeze(0).repeat(T, 1) if gmax.dim() == 1 else gmax
    pmax_b = pmax.unsqueeze(0).repeat(T, 1) if pmax.dim() == 1 else pmax
    with torch.no_grad():
        g, p = opf.solve(d, f_b, gmax_b, pmax_b)
    g_np = g.cpu().numpy()
    p_np = p.cpu().numpy()
    gmax_np = gmax.cpu().numpy()
    pmax_np = pmax.cpu().numpy()
    line_bind = np.abs(p_np) >= (1.0 - tol_frac) * pmax_np[None, :]
    gen_bind = g_np >= (1.0 - tol_frac) * gmax_np[None, :]
    return {
        "line_freq": line_bind.mean(axis=0),
        "gen_freq": gen_bind.mean(axis=0),
        "line_any_frac": float(line_bind.any(axis=1).mean()),
        "gen_any_frac": float(gen_bind.any(axis=1).mean()),
        "line_mean_freq": float(line_bind.mean()),
        "gen_mean_freq": float(gen_bind.mean()),
    }


# ---------------------------------------------------------------------------
# KMeans strata recovery  (Section 3)
# ---------------------------------------------------------------------------


def kmeans_strata(demands: np.ndarray | torch.Tensor, n_strata: int,
                  seed: int = 0) -> np.ndarray:
    """Assign each demand row to one of ``n_strata`` clusters via KMeans.

    Used to replace the ground-truth hour-of-day stratum index when only
    raw demand vectors are observed.
    """
    from sklearn.cluster import KMeans
    X = demands.detach().cpu().numpy() if torch.is_tensor(demands) else np.asarray(demands)
    km = KMeans(n_clusters=int(n_strata), n_init=10, random_state=int(seed))
    return km.fit_predict(X).astype(np.int64)


def strata_agreement(pred: np.ndarray, truth: np.ndarray) -> float:
    """Best-permutation agreement between two label vectors.

    Uses the Hungarian algorithm on the confusion matrix.  Returns a
    score in [0, 1] (1 = perfect cluster recovery).
    """
    from scipy.optimize import linear_sum_assignment
    pred = np.asarray(pred).astype(int)
    truth = np.asarray(truth).astype(int)
    k = max(pred.max(), truth.max()) + 1
    M = np.zeros((k, k), dtype=int)
    for a, b in zip(pred, truth, strict=False):
        M[a, b] += 1
    r, c = linear_sum_assignment(-M)
    return float(M[r, c].sum()) / float(len(pred))


# ---------------------------------------------------------------------------
# Capacity screening  (Section 4)
# ---------------------------------------------------------------------------


def screen_capacities(g_obs: np.ndarray, p_est: np.ndarray | None,
                      gen_thresh: float = 0.05, line_thresh: float = 0.05
                      ) -> dict:
    """Identify generators and lines that are effectively unused.

    A generator is "screened out" if its peak observed dispatch is below
    ``gen_thresh`` times the global peak generation; same for lines.
    Returns boolean keep-masks (so callers can rebuild a smaller problem).
    """
    g_peak = g_obs.max(axis=0)
    g_global = max(g_peak.max(), 1e-9)
    keep_gen = g_peak >= gen_thresh * g_global
    keep_line = np.ones(p_est.shape[1] if p_est is not None else 0, dtype=bool)
    if p_est is not None and p_est.size > 0:
        p_peak = np.abs(p_est).max(axis=0)
        p_global = max(p_peak.max(), 1e-9)
        keep_line = p_peak >= line_thresh * p_global
    return {
        "keep_gen_mask": keep_gen,
        "keep_line_mask": keep_line,
        "n_gen_kept": int(keep_gen.sum()),
        "n_line_kept": int(keep_line.sum()),
        "gen_screen_ratio": float(keep_gen.sum()) / float(keep_gen.size),
        "line_screen_ratio": (float(keep_line.sum()) / float(keep_line.size)
                              if keep_line.size > 0 else 1.0),
    }


# ---------------------------------------------------------------------------
# Emissions: per-fuel CO2 factors + marginal emission factors  (Section 6)
# ---------------------------------------------------------------------------


# Representative EIA-style CO2 intensities (kg/MWh) loosely matching the PJM
# fuel ordering nuclear / hydro / wind / solar / coal / CCGT / oil / peaker.
EIA_FUEL_CO2 = np.array([0.0, 0.0, 0.0, 0.0, 900.0, 400.0, 750.0, 650.0])


def assign_eia_factors(f_true: np.ndarray, rng_seed: int = 0) -> np.ndarray:
    """Assign per-bus CO2 factors consistent with the merit-order rank.

    We sort buses by ``f_true`` and map the sorted order onto a tiled copy
    of the canonical EIA factors.  Result: cheap buses get clean factors,
    expensive buses get dirty factors -- matching the synthetic stack.
    """
    n = int(f_true.shape[0])
    rank = np.argsort(f_true)
    template = np.tile(EIA_FUEL_CO2, int(np.ceil(n / EIA_FUEL_CO2.size)))[:n]
    out = np.zeros(n, dtype=float)
    out[rank] = template
    return out


def marginal_emission_factor(opf: DCOpfLayer, d: torch.Tensor,
                              f: torch.Tensor, gmax: torch.Tensor,
                              pmax: torch.Tensor, co2_factors: np.ndarray,
                              perturb: float = 1.0,
                              bus: int | None = None) -> float:
    """One-sided finite-difference MEF: kg CO2 per MWh of *added* demand.

    If ``bus`` is None the perturbation is added uniformly across all
    buses (system-wide MEF).  Otherwise only that bus is perturbed.
    """
    T = d.shape[0]
    f_b = f.unsqueeze(0).repeat(T, 1) if f.dim() == 1 else f
    gmax_b = gmax.unsqueeze(0).repeat(T, 1) if gmax.dim() == 1 else gmax
    pmax_b = pmax.unsqueeze(0).repeat(T, 1) if pmax.dim() == 1 else pmax
    with torch.no_grad():
        g_base, _ = opf.solve(d, f_b, gmax_b, pmax_b)
        d_pert = d.clone()
        if bus is None:
            d_pert = d_pert + perturb / d.shape[1]
        else:
            d_pert[:, bus] = d_pert[:, bus] + perturb
        g_pert, _ = opf.solve(d_pert, f_b, gmax_b, pmax_b)
    delta_g = (g_pert - g_base).cpu().numpy()        # T x n_buses
    delta_co2 = (delta_g * co2_factors[None, :]).sum(axis=1)  # T
    # MEF per MWh of added demand.
    return float(delta_co2.mean() / perturb)


# ---------------------------------------------------------------------------
# Counterfactual demand scaling  (Section 6)
# ---------------------------------------------------------------------------


@dataclass
class CounterfactualResult:
    scale: float
    g_mean: float
    total_co2_mean: float
    cost_mean: float


def counterfactual_dispatch(opf: DCOpfLayer, d: torch.Tensor,
                             f: torch.Tensor, gmax: torch.Tensor,
                             pmax: torch.Tensor, co2_factors: np.ndarray,
                             scale: float) -> CounterfactualResult:
    """Solve the OPF at demand * scale and report mean dispatch / CO2 / cost."""
    T = d.shape[0]
    d_scaled = d * float(scale)
    f_b = f.unsqueeze(0).repeat(T, 1) if f.dim() == 1 else f
    gmax_b = gmax.unsqueeze(0).repeat(T, 1) if gmax.dim() == 1 else gmax
    pmax_b = pmax.unsqueeze(0).repeat(T, 1) if pmax.dim() == 1 else pmax
    with torch.no_grad():
        g, _ = opf.solve(d_scaled, f_b, gmax_b, pmax_b)
    g_np = g.cpu().numpy()
    f_np = f.cpu().numpy()
    return CounterfactualResult(
        scale=float(scale),
        g_mean=float(g_np.sum(axis=1).mean()),
        total_co2_mean=float((g_np * co2_factors[None, :]).sum(axis=1).mean()),
        cost_mean=float((g_np * f_np[None, :]).sum(axis=1).mean()),
    )
