"""Boundary cases for the identifiability score.

``identifiability_score(g_obs, gmax)`` returns, per generator, the fraction
of samples where the dispatch lies strictly inside ``(0, gmax)``.  We expect:

* always-interior generator  -> 1.0
* always-at-upper generator   -> 0.0
* always-at-lower generator   -> 0.0
* mixed                       -> intermediate
"""

from __future__ import annotations

import numpy as np
import torch

from inverse_opf.metrics import identifiability_score


def test_identifiability_extremes():
    T, n = 50, 4
    gmax = np.array([10.0, 10.0, 10.0, 10.0])

    g = np.zeros((T, n))
    g[:, 0] = 5.0     # always interior
    g[:, 1] = 10.0    # always at upper bound
    g[:, 2] = 0.0     # always at lower bound
    g[:, 3] = np.linspace(0.5, 9.5, T)  # always interior (sweep)

    si = identifiability_score(torch.tensor(g), torch.tensor(gmax), tol=1e-3)
    assert si.shape == (n,)
    assert si[0] == 1.0
    assert si[1] == 0.0
    assert si[2] == 0.0
    assert si[3] == 1.0


def test_identifiability_mixed_fraction():
    T, n = 100, 1
    gmax = np.array([10.0])
    g = np.zeros((T, n))
    # 30% interior, 40% at upper, 30% at lower
    g[:30, 0] = 5.0
    g[30:70, 0] = 10.0
    g[70:, 0] = 0.0
    si = identifiability_score(g, gmax, tol=1e-3)
    assert abs(float(si[0]) - 0.30) < 1e-6


def test_identifiability_respects_tolerance():
    T, n = 10, 1
    gmax = np.array([10.0])
    # Within tol of the upper bound -> should count as bound, not interior.
    g = np.full((T, n), 10.0 - 1e-4)
    si = identifiability_score(g, gmax, tol=1e-3)
    assert float(si[0]) == 0.0

    # Comfortably interior.
    g2 = np.full((T, n), 9.0)
    si2 = identifiability_score(g2, gmax, tol=1e-3)
    assert float(si2[0]) == 1.0
