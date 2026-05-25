"""Forward QP correctness: KKT conditions hold at the layer's solution.

For a transport-physics DC-OPF the Lagrangian is::

    L = f^T g + 0.5 eps ||g||^2 + 0.5 tau ||p||^2
        + nu^T (g - d - A p)
        + mu_g_lo^T (-g) + mu_g_hi^T (g - gmax)
        + mu_p_lo^T (-p - pmax) + mu_p_hi^T (p - pmax)

We do not have direct access to the dual multipliers from CvxpyLayer, so we
verify correctness by:

1. checking primal feasibility tightly, and
2. re-solving the same problem with CVXPY (single-shot, high-precision SCS
   call) and comparing primal objectives -- the layer's solution should match
   the verified optimum to solver tolerance.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
import pytest
import torch

from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import random_connected_incidence
from inverse_opf.seeding import set_global_seed


def _solve_reference(A, f, gmax, pmax, d, eps, tau):
    n, m = A.shape
    g = cp.Variable(n)
    p = cp.Variable(m)
    obj = cp.sum(cp.multiply(f, g)) + 0.5 * eps * cp.sum_squares(g)
    obj += 0.5 * tau * cp.sum_squares(p)
    cons = [
        g >= 0.0,
        g <= gmax,
        p <= pmax,
        p >= -pmax,
        g - d == A @ p,
    ]
    prob = cp.Problem(cp.Minimize(obj), cons)
    prob.solve(solver=cp.SCS, eps=1e-9, max_iters=50000, verbose=False)
    return float(prob.value), np.asarray(g.value), np.asarray(p.value)


@pytest.mark.parametrize("trial", range(3))
def test_forward_kkt_primal_feasibility_and_objective(trial):
    set_global_seed(100 + trial)
    rng = np.random.default_rng(100 + trial)
    n, m = 5, 7
    A = random_connected_incidence(n, m, rng)

    # Make sure the problem is feasible: pick (f, gmax, pmax) generously.
    f = rng.uniform(5.0, 30.0, size=n)
    gmax = rng.uniform(40.0, 80.0, size=n)
    pmax = rng.uniform(40.0, 80.0, size=m)
    # Demand: roughly half of total capacity so plenty of slack.
    d = rng.uniform(2.0, 10.0, size=n)
    # Net injection must be zero in the transport model (sum(g) = sum(d)),
    # which is enforced automatically by the QP via the equality constraint.

    eps_q, tau = 0.1, 1e-2
    layer = DCOpfLayer(DCOpfProblemData(incidence=A), eps=eps_q, tau=tau)

    d_t = torch.tensor(d, dtype=torch.float32).unsqueeze(0)
    f_t = torch.tensor(f, dtype=torch.float32).unsqueeze(0)
    gmax_t = torch.tensor(gmax, dtype=torch.float32).unsqueeze(0)
    pmax_t = torch.tensor(pmax, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        g_layer, p_layer = layer.solve(d_t, f_t, gmax_t, pmax_t)
    g_layer = g_layer.squeeze(0).numpy()
    p_layer = p_layer.squeeze(0).numpy()

    # Primal feasibility (loose tol: float32 + SCS tol).
    assert (g_layer >= -1e-3).all()
    assert (g_layer <= gmax + 1e-3).all()
    assert (np.abs(p_layer) <= pmax + 1e-3).all()
    balance_err = np.max(np.abs(g_layer - d - A @ p_layer))
    assert balance_err < 1e-2, f"power balance residual {balance_err:.2e}"

    # Optimality: compare to a fresh high-precision CVXPY solve.
    obj_ref, _, _ = _solve_reference(A, f, gmax, pmax, d, eps_q, tau)
    obj_layer = (
        float(np.dot(f, g_layer)) + 0.5 * eps_q * float(np.dot(g_layer, g_layer))
        + 0.5 * tau * float(np.dot(p_layer, p_layer))
    )
    # Layer objective should not be meaningfully below the reference optimum.
    # We allow a small positive gap due to float32 + lower-precision solver.
    assert obj_layer - obj_ref > -1e-2
    assert obj_layer - obj_ref < 1e-1
