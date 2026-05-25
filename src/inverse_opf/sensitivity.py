"""Sensitivity utilities: Jacobians of g* w.r.t. demand."""

from __future__ import annotations

import torch

from .dc_opf import DCOpfLayer


def jacobian_g_wrt_d(
    opf_layer: DCOpfLayer,
    d: torch.Tensor,
    f: torch.Tensor,
    gmax: torch.Tensor,
    pmax: torch.Tensor,
) -> torch.Tensor:
    """dg*/dd at a single operating point. (n_buses, n_buses)."""
    f_b = f.detach().unsqueeze(0)
    gmax_b = gmax.detach().unsqueeze(0)
    pmax_b = pmax.detach().unsqueeze(0)

    def g_of_d(d_in: torch.Tensor) -> torch.Tensor:
        g, _ = opf_layer.solve(d_in.unsqueeze(0), f_b, gmax_b, pmax_b)
        return g.squeeze(0)

    return torch.autograd.functional.jacobian(g_of_d, d.detach().clone(), vectorize=False)


def finite_difference_jacobian(
    opf_layer: DCOpfLayer,
    d: torch.Tensor,
    f: torch.Tensor,
    gmax: torch.Tensor,
    pmax: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Central-difference estimate of dg*/dd. For validation only."""
    n = d.shape[0]
    jac = torch.zeros(n, n)
    f_b = f.detach().unsqueeze(0)
    gmax_b = gmax.detach().unsqueeze(0)
    pmax_b = pmax.detach().unsqueeze(0)
    with torch.no_grad():
        for j in range(n):
            d_plus = d.clone(); d_plus[j] += eps
            d_minus = d.clone(); d_minus[j] -= eps
            g_plus, _ = opf_layer.solve(d_plus.unsqueeze(0), f_b, gmax_b, pmax_b)
            g_minus, _ = opf_layer.solve(d_minus.unsqueeze(0), f_b, gmax_b, pmax_b)
            jac[:, j] = (g_plus.squeeze(0) - g_minus.squeeze(0)) / (2.0 * eps)
    return jac


def marginal_emission_factors(jacobian: torch.Tensor, carbon_intensity: torch.Tensor) -> torch.Tensor:
    return jacobian.T @ carbon_intensity


def counterfactual_dispatch(
    opf_layer: DCOpfLayer,
    d: torch.Tensor,
    f: torch.Tensor,
    gmax: torch.Tensor,
    pmax: torch.Tensor,
    delta_d: torch.Tensor,
) -> torch.Tensor:
    with torch.no_grad():
        d_new = (d + delta_d).unsqueeze(0)
        g_new, _ = opf_layer.solve(
            d_new, f.unsqueeze(0), gmax.unsqueeze(0), pmax.unsqueeze(0),
        )
    return g_new.squeeze(0)
