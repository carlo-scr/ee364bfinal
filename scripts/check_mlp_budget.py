"""Verify whether MLP underperformance vs Ridge is a training-budget artifact.

Runs the MLP baseline at 1000 (default) and 5000 steps on the same synthetic
dataset used by the headline methods_comparison experiment, plus Ridge for
reference. Reports val NRMSE means/stds across 5 seeds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from _common import StandardConfig, build_dataset, true_val_dispatch
from inverse_opf.baselines_regression import mlp_baseline, ridge_baseline
from inverse_opf.metrics import normalized_rmse

SEEDS = [0, 1, 2, 3, 4]
sc = StandardConfig()  # same defaults as methods_comparison

results = {"ridge": [], "mlp_1000": [], "mlp_5000": []}
for seed in SEEDS:
    torch.manual_seed(seed); np.random.seed(seed)
    ds = build_dataset(seed, sc)
    g_true = true_val_dispatch(ds, sc).cpu().numpy()
    # Ridge
    r = ridge_baseline(ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, alpha=1.0)
    pred = r.predict(ds.d_val.detach().cpu().numpy())
    results["ridge"].append(normalized_rmse(pred, g_true))
    # MLP 1000 steps
    torch.manual_seed(seed)
    m1 = mlp_baseline(ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, steps=1000)
    pred = m1.predict(ds.d_val.detach().cpu().numpy())
    results["mlp_1000"].append(normalized_rmse(pred, g_true))
    # MLP 5000 steps
    torch.manual_seed(seed)
    m5 = mlp_baseline(ds.d_train, ds.g_train_obs, ds.d_val, ds.g_val_obs, steps=5000)
    pred = m5.predict(ds.d_val.detach().cpu().numpy())
    results["mlp_5000"].append(normalized_rmse(pred, g_true))
    print(f"seed {seed}: ridge={results['ridge'][-1]:.3f}  "
          f"mlp@1000={results['mlp_1000'][-1]:.3f}  "
          f"mlp@5000={results['mlp_5000'][-1]:.3f}")

print()
for k, v in results.items():
    v = np.array(v)
    print(f"{k:>10s}: NRMSE mean={v.mean():.3f}  std={v.std():.3f}")
