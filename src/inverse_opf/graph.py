from __future__ import annotations

import numpy as np


def cycle_laplacian(n_strata: int) -> np.ndarray:
    n = int(n_strata)
    l = np.zeros((n, n), dtype=float)
    for i in range(n):
        j = (i + 1) % n
        l[i, i] += 1.0
        l[j, j] += 1.0
        l[i, j] -= 1.0
        l[j, i] -= 1.0
    return l


def product_cycle_path_laplacian(n_hours: int, n_months: int) -> np.ndarray:
    h = int(n_hours); m = int(n_months); n = h * m
    l = np.zeros((n, n), dtype=float)
    def idx(hh, mm): return hh * m + mm
    for hh in range(h):
        hh_next = (hh + 1) % h
        for mm in range(m):
            i = idx(hh, mm); j = idx(hh_next, mm)
            l[i, i] += 1.0; l[j, j] += 1.0
            l[i, j] -= 1.0; l[j, i] -= 1.0
    for mm in range(m - 1):
        for hh in range(h):
            i = idx(hh, mm); j = idx(hh, mm + 1)
            l[i, i] += 1.0; l[j, j] += 1.0
            l[i, j] -= 1.0; l[j, i] -= 1.0
    return l


def random_connected_incidence(n_buses: int, n_lines: int, rng: np.random.Generator) -> np.ndarray:
    if n_lines < n_buses - 1:
        raise ValueError("Need at least n_buses - 1 lines for connectivity")
    edges: set[tuple[int, int]] = set()
    for i in range(1, n_buses):
        j = int(rng.integers(0, i))
        u, v = sorted((i, j))
        edges.add((u, v))
    while len(edges) < n_lines:
        u = int(rng.integers(0, n_buses)); v = int(rng.integers(0, n_buses))
        if u == v:
            continue
        a, b = sorted((u, v))
        edges.add((a, b))
    edge_list = list(edges)
    a = np.zeros((n_buses, n_lines), dtype=float)
    for k, (u, v) in enumerate(edge_list):
        a[u, k] = 1.0; a[v, k] = -1.0
    return a


def random_susceptance(n_lines: int, rng: np.random.Generator,
                       low: float = 0.5, high: float = 2.0) -> np.ndarray:
    """Per-line susceptance, drawn log-uniform in [low, high]."""
    log_lo = np.log(low); log_hi = np.log(high)
    return np.exp(rng.uniform(log_lo, log_hi, size=n_lines))
