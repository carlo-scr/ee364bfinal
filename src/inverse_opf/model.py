from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class InverseParameters:
    f: torch.Tensor
    gmax: torch.Tensor
    pmax: torch.Tensor


class InverseOPFModel(torch.nn.Module):
    """Learns static parameters (f, gmax, pmax)."""

    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        f_init: float = 20.0,
        gmax_init: float = 80.0,
        pmax_init: float = 120.0,
    ) -> None:
        super().__init__()
        self.f_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_buses,), f_init)))
        self.gmax_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_buses,), gmax_init)))
        self.pmax_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_lines,), pmax_init)))

    @staticmethod
    def _softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    @staticmethod
    def _positive(x: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return torch.nn.functional.softplus(x) + floor

    def current_parameters(self) -> InverseParameters:
        return InverseParameters(
            f=self._positive(self.f_raw),
            gmax=self._positive(self.gmax_raw),
            pmax=self._positive(self.pmax_raw),
        )


class StratifiedInverseOPFModel(torch.nn.Module):
    """Learns per-stratum costs with shared capacities.

    f[s, i] is the generation cost at stratum s and bus i.
    """

    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        n_strata: int,
        f_init: float = 20.0,
        gmax_init: float = 80.0,
        pmax_init: float = 120.0,
    ) -> None:
        super().__init__()
        self.n_strata = int(n_strata)
        self.f_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_strata, n_buses), f_init)))
        self.gmax_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_buses,), gmax_init)))
        self.pmax_raw = torch.nn.Parameter(self._softplus_inverse(torch.full((n_lines,), pmax_init)))

    @staticmethod
    def _softplus_inverse(y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        y = torch.clamp(y, min=eps)
        return y + torch.log(-torch.expm1(-y))

    @staticmethod
    def _positive(x: torch.Tensor, floor: float = 1e-3) -> torch.Tensor:
        return torch.nn.functional.softplus(x) + floor

    def costs_for_strata(self, strata_idx: torch.Tensor) -> torch.Tensor:
        f_all = self._positive(self.f_raw)
        return f_all[strata_idx]

    def shared_capacities(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._positive(self.gmax_raw), self._positive(self.pmax_raw)

    def full_cost_table(self) -> torch.Tensor:
        return self._positive(self.f_raw)
