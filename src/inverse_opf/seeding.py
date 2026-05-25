"""Deterministic seeding for reproducible experiments.

Call :func:`set_global_seed` once at the start of every experiment script and
pass the returned ``np.random.Generator`` (or a sub-stream via :func:`spawn`)
into any dataset / sampling code that supports it.  Anything that still uses
the global ``numpy.random`` or Python ``random`` modules also gets seeded for
backward compatibility.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class SeedBundle:
    """Holds the seed plus a stream-aware NumPy generator."""

    seed: int
    rng: np.random.Generator

    def spawn(self, label: str) -> np.random.Generator:
        """Return an independent generator derived from this seed + label.

        Using ``SeedSequence`` keeps sub-streams statistically independent so
        e.g. dataset sampling and noise injection do not share state.
        """
        ss = np.random.SeedSequence([self.seed, _label_to_int(label)])
        return np.random.default_rng(ss)


def set_global_seed(seed: int, *, deterministic_torch: bool = True) -> SeedBundle:
    """Seed every global RNG and return a fresh NumPy generator.

    Parameters
    ----------
    seed
        Master seed.
    deterministic_torch
        If True, set ``torch.use_deterministic_algorithms(True)`` and the
        relevant cuDNN flags.  Safe on CPU; on GPU may slow some kernels.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        # CUBLAS workspace required by torch.use_deterministic_algorithms when
        # CUDA is available; harmless on CPU.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some torch builds disallow this; fall through silently.
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return SeedBundle(seed=seed, rng=np.random.default_rng(seed))


def _label_to_int(label: str) -> int:
    """Map a short label to a stable integer for SeedSequence sub-streams."""
    # Simple FNV-1a 32-bit; deterministic across Python versions/platforms.
    h = 0x811c9dc5
    for ch in label.encode("utf-8"):
        h ^= ch
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h
