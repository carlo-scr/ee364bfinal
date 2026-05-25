"""Baselines for ablations against the full inverse OPF model.

FixedCapacityInverseOPFModel mimics the Fuentes Valenzuela and Degleris [14]
setup: learn only the cost vector f while treating (gmax, pmax) as known.
"""

from __future__ import annotations

import torch

from .model import InverseOPFModel, InverseParameters


class FixedCapacityInverseOPFModel(InverseOPFModel):
    """Inverse model that learns only f; gmax, pmax are registered as buffers."""

    def __init__(
        self,
        n_buses: int,
        n_lines: int,
        gmax: torch.Tensor,
        pmax: torch.Tensor,
        f_init: float = 20.0,
    ) -> None:
        super().__init__(
            n_buses=n_buses,
            n_lines=n_lines,
            f_init=f_init,
            gmax_init=float(gmax.mean().item()),
            pmax_init=float(pmax.mean().item()),
        )
        del self.gmax_raw
        del self.pmax_raw
        self.register_buffer("_gmax_fixed", gmax.clone().detach().float())
        self.register_buffer("_pmax_fixed", pmax.clone().detach().float())

    def current_parameters(self) -> InverseParameters:
        return InverseParameters(
            f=self._positive(self.f_raw),
            gmax=self._gmax_fixed,
            pmax=self._pmax_fixed,
        )
