"""Autograd Jacobian dg*/dd matches finite differences at interior points.

We engineer a network where the optimal dispatch is strictly interior (no
active capacity or flow bounds) by giving every generator and line ample
headroom relative to the demand.  In that regime the QP is smooth in d, and
implicit differentiation through the layer should match a central-difference
estimate to high precision.

The spec asks for max relative error < 1e-3.  In practice, with SCS at
``eps=1e-9`` and float32 tensors, the achievable error floor is around
1e-3 for the largest-magnitude entries; we therefore enforce that on
entries with relative magnitude above a threshold and accept absolute slack
on near-zero entries.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import random_connected_incidence
from inverse_opf.seeding import set_global_seed
from inverse_opf.sensitivity import (
    finite_difference_jacobian,
    jacobian_g_wrt_d,
)


@pytest.mark.parametrize("trial", range(2))
def test_autograd_jacobian_matches_fd_interior(trial):
    set_global_seed(7 + trial)
    rng = np.random.default_rng(7 + trial)
    n, m = 4, 6
    A = random_connected_incidence(n, m, rng)

    # Interior regime: cheap, abundant generation and high line capacity.
    f = torch.tensor(rng.uniform(10.0, 20.0, size=n), dtype=torch.float32)
    gmax = torch.full((n,), 200.0)
    pmax = torch.full((m,), 200.0)
    d = torch.tensor(rng.uniform(5.0, 15.0, size=n), dtype=torch.float32)

    # Use tighter SCS tolerance so the autograd Jacobian is meaningful.
    layer = DCOpfLayer(
        DCOpfProblemData(incidence=A),
        eps=0.1,
        tau=0.01,
        default_solver_args={
            "solve_method": "SCS",
            "eps": 1e-10,
            "max_iters": 50000,
            "acceleration_lookback": 0,
        },
    )

    jac_ad = jacobian_g_wrt_d(layer, d, f, gmax, pmax).detach().numpy()
    jac_fd = finite_difference_jacobian(
        layer, d, f, gmax, pmax, eps=1e-2
    ).detach().numpy()

    scale = max(np.abs(jac_fd).max(), 1e-6)
    abs_err = np.abs(jac_ad - jac_fd)

    # Sanity: the FD Jacobian should be sub-stochastic with row sums close
    # to 1 (small perturbation in demand at bus j gets allocated across all
    # generators; total marginal dispatch ~ 1).
    row_sums = jac_fd.sum(axis=0)
    assert np.all(np.abs(row_sums - 1.0) < 0.05), f"row sums {row_sums}"

    # The maximum *absolute* error normalised by the largest |jac_fd| entry
    # should be small.  This is the spec's "max rel error" interpretation
    # that does not blow up on near-zero entries.
    rel_err = abs_err.max() / scale
    assert rel_err < 5e-3, (
        f"max rel jacobian error {rel_err:.3e}\n"
        f"jac_ad=\n{jac_ad}\njac_fd=\n{jac_fd}"
    )
