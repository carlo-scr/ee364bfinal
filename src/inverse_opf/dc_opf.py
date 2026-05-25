"""Differentiable DC-OPF QP layer.

Two physics modes:

* ``physics="transport"``: network-flow / transportation model (no susceptances).
  Forward problem:
      min  0.5 g^T diag(f) g + 0.5 tau ||p||^2 + lambda_s 1^T s
      s.t. g - d = A p,  0 <= g <= gmax,  |p| <= pmax,  s >= 0,
           g <= gmax + s   (lifted slack -- only when slack_lift=True)

* ``physics="dc"``: linearized AC ("DC") OPF with susceptance-weighted flows
  and explicit phase angles, slack bus pinned at 0:
      p = B_e A^T theta,
      A p + g = d,
      with g, p, |p|<=pmax, theta_ref = 0.

Both formulations are convex QPs with strongly convex objective (eps>0 floor on
``f``, plus tau on ||p||^2), so the solution map is unique and differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer


@dataclass(frozen=True)
class DCOpfProblemData:
    """Network description.

    ``incidence``       : signed n_buses x n_lines incidence matrix (any orientation).
    ``susceptance``     : optional length-n_lines vector of line susceptances b_e
                          (only used when physics="dc"). Defaults to ones.
    ``slack_bus``       : index of the reference bus for physics="dc".
    """
    incidence: np.ndarray
    susceptance: np.ndarray | None = None
    slack_bus: int = 0

    @property
    def n_buses(self) -> int:
        return int(self.incidence.shape[0])

    @property
    def n_lines(self) -> int:
        return int(self.incidence.shape[1])

    def b_vec(self) -> np.ndarray:
        if self.susceptance is None:
            return np.ones(self.n_lines, dtype=float)
        return np.asarray(self.susceptance, dtype=float)


class DCOpfLayer:
    """Differentiable DC-OPF layer using cvxpylayers."""

    def __init__(
        self,
        data: DCOpfProblemData,
        tau: float = 1e-2,
        eps: float = 1e-1,
        physics: str = "transport",
        slack_lift: bool = False,
        slack_lift_weight: float = 1.0,
        default_solver_args: dict | None = None,
    ) -> None:
        self.data = data
        self.tau = float(tau)
        self.eps = float(eps)
        if physics not in {"transport", "dc"}:
            raise ValueError(f"Unknown physics: {physics}")
        self.physics = physics
        self.slack_lift = bool(slack_lift)
        self.slack_lift_weight = float(slack_lift_weight)
        self.default_solver_args = default_solver_args or {
            "solve_method": "SCS",
            "eps": 1e-8,
            "max_iters": 20000,
            "acceleration_lookback": 0,
        }
        self._layer = self._build_layer()

    def _build_layer(self) -> CvxpyLayer:
        n = self.data.n_buses
        m = self.data.n_lines
        A = self.data.incidence

        g = cp.Variable(n)
        p = cp.Variable(m)

        d_param = cp.Parameter(n)
        f_param = cp.Parameter(n, nonneg=True)
        gmax_param = cp.Parameter(n, nonneg=True)
        pmax_param = cp.Parameter(m, nonneg=True)

        objective = cp.sum(cp.multiply(f_param, g)) + 0.5 * self.eps * cp.sum_squares(g)
        objective += 0.5 * self.tau * cp.sum_squares(p)

        constraints: list = [
            g >= 0.0,
            p <= pmax_param,
            p >= -pmax_param,
        ]

        if self.physics == "dc":
            theta = cp.Variable(n)
            B = np.diag(self.data.b_vec())
            constraints += [
                p == B @ A.T @ theta,
                A @ p + g == d_param,
                theta[self.data.slack_bus] == 0.0,
            ]
        else:
            constraints += [g - d_param == A @ p]

        if self.slack_lift:
            s = cp.Variable(n, nonneg=True)
            constraints += [g <= gmax_param + s]
            objective += self.slack_lift_weight * cp.sum(s)
            variables = [g, p, s]
        else:
            constraints += [g <= gmax_param]
            variables = [g, p]

        problem = cp.Problem(cp.Minimize(objective), constraints)
        return CvxpyLayer(
            problem,
            parameters=[d_param, f_param, gmax_param, pmax_param],
            variables=variables,
        )

    def solve(
        self,
        d: torch.Tensor,
        f: torch.Tensor,
        gmax: torch.Tensor,
        pmax: torch.Tensor,
        solver_args: dict | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        solver_args = solver_args or self.default_solver_args
        out = self._layer(d, f, gmax, pmax, solver_args=solver_args)
        # Always return (g, p) regardless of slack_lift presence.
        return out[0], out[1]
