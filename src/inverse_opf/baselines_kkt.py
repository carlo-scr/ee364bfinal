"""KKT-residual inverse optimization baseline (Keshavarz, Wang, Boyd 2011).

Given observations (d_t, g_t^obs, p_t^obs?) and the forward DC-OPF QP, the
KKT stationarity conditions are linear in the parameter f when (g, p) are
plugged in as fixed. We exploit this to write a *single convex QP in
(f, lambda multipliers)* per inverse problem, no implicit differentiation
required.

Forward (transport, no slack lift) Lagrangian:
    L = 0.5 g^T diag(f+eps) g + 0.5 tau ||p||^2
        + nu^T (g - d - A p)
        + mu_g_lo^T (-g) + mu_g_hi^T (g - gmax)
        + mu_p_lo^T (-p - pmax) + mu_p_hi^T (p - pmax)

Stationarity:
    dL/dg = (f+eps) .* g + nu - mu_g_lo + mu_g_hi = 0
    dL/dp = tau * p   - A^T nu - mu_p_lo + mu_p_hi = 0

We aggregate over T samples and minimize the squared L2 norm of these
residuals subject to:
    f >= 0
    multipliers >= 0
    complementarity-aware sparsity: force mu = 0 on inactive constraints (via
    constraint masks derived from g^obs).

This is a convex QP in (f, multipliers). It serves as the standard convex-opt
baseline against our differentiable layer.

When ``learn_capacities=True`` we also estimate gmax and pmax. To keep the
problem convex, we *fix the active set* using the observations
(active = constraint observed binding) and parameterize gmax = max_t g_obs_t
plus a learned positive shift; in practice we use a simple two-stage scheme:
1) estimate gmax / pmax via per-bus / per-line empirical maxima inflated by a
   small slack (KKT cannot identify a binding constraint's exact value);
2) solve the KKT-residual QP for f.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch

from .dc_opf import DCOpfProblemData


@dataclass
class KKTBaselineResult:
    f: np.ndarray
    gmax: np.ndarray
    pmax: np.ndarray
    residual_norm: float


def _flow_from_balance(d: np.ndarray, g: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Recover p from balance g - d = A p using least squares (per sample)."""
    # A is n_buses x n_lines, possibly underdetermined; use lstsq.
    rhs = g - d
    p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    return p


def kkt_residual_inverse(
    data: DCOpfProblemData,
    d_obs: torch.Tensor | np.ndarray,
    g_obs: torch.Tensor | np.ndarray,
    tau: float = 1e-2,
    eps: float = 1e-1,
    learn_capacities: bool = False,
    activity_tol: float = 1e-2,
    f_upper: float = 200.0,
) -> KKTBaselineResult:
    A = data.incidence
    n = data.n_buses
    m = data.n_lines

    d = d_obs.detach().cpu().numpy() if isinstance(d_obs, torch.Tensor) else np.asarray(d_obs)
    g = g_obs.detach().cpu().numpy() if isinstance(g_obs, torch.Tensor) else np.asarray(g_obs)
    T = d.shape[0]

    # Recover line flows from power balance.
    p = np.stack([_flow_from_balance(d[t], g[t], A) for t in range(T)], axis=0)

    # Capacity estimates: empirical maxima inflated by 5% so observed maxima
    # are not on the boundary of the feasible set.
    if learn_capacities:
        gmax = np.maximum(g.max(axis=0) * 1.05, 1e-3)
        pmax = np.maximum(np.abs(p).max(axis=0) * 1.05, 1e-3)
    else:
        gmax = g.max(axis=0) * 1.05
        pmax = np.abs(p).max(axis=0) * 1.05

    # Active-set masks from observations. A constraint is "active" if the
    # observation is within activity_tol * scale of the bound.
    g_scale = max(g.max(), 1.0)
    p_scale = max(np.abs(p).max(), 1.0)
    active_g_lo = g <= activity_tol * g_scale          # T x n
    active_g_hi = g >= gmax - activity_tol * g_scale   # T x n
    active_p_lo = p <= -pmax + activity_tol * p_scale  # T x m
    active_p_hi = p >=  pmax - activity_tol * p_scale  # T x m

    # CVXPY variables.
    f = cp.Variable(n, nonneg=True)
    nu = cp.Variable((T, n))                         # equality multipliers
    mu_g_lo = cp.Variable((T, n), nonneg=True)
    mu_g_hi = cp.Variable((T, n), nonneg=True)
    mu_p_lo = cp.Variable((T, m), nonneg=True)
    mu_p_hi = cp.Variable((T, m), nonneg=True)

    constraints = [f <= f_upper]
    # Force inactive multipliers to zero (complementary slackness).
    for t in range(T):
        for i in range(n):
            if not active_g_lo[t, i]:
                constraints.append(mu_g_lo[t, i] == 0.0)
            if not active_g_hi[t, i]:
                constraints.append(mu_g_hi[t, i] == 0.0)
        for e in range(m):
            if not active_p_lo[t, e]:
                constraints.append(mu_p_lo[t, e] == 0.0)
            if not active_p_hi[t, e]:
                constraints.append(mu_p_hi[t, e] == 0.0)

    # Stationarity residuals (linear cost: dL/dg = f + eps*g + nu - mu_lo + mu_hi).
    res_g = cp.reshape(f, (1, n), order="C") + eps * g + nu - mu_g_lo + mu_g_hi
    res_p = tau * p - nu @ A - mu_p_lo + mu_p_hi
    obj = cp.sum_squares(res_g) + cp.sum_squares(res_p)

    prob = cp.Problem(cp.Minimize(obj), constraints)
    try:
        prob.solve(solver=cp.SCS, verbose=False)
    except Exception:
        prob.solve(solver=cp.CLARABEL, verbose=False)

    f_val = np.clip(np.asarray(f.value).reshape(-1), 0.0, None) if f.value is not None else np.full(n, np.nan)
    return KKTBaselineResult(
        f=f_val,
        gmax=gmax,
        pmax=pmax,
        residual_norm=float(np.sqrt(prob.value)) if prob.value is not None else float("nan"),
    )
