"""Smoke tests for the inverse OPF pipeline."""

from __future__ import annotations

import numpy as np
import torch

from inverse_opf.baselines import FixedCapacityInverseOPFModel
from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
from inverse_opf.graph import cycle_laplacian, random_connected_incidence
from inverse_opf.model import InverseOPFModel, StratifiedInverseOPFModel
from inverse_opf.sensitivity import (
    counterfactual_dispatch,
    jacobian_g_wrt_d,
    marginal_emission_factors,
)
from inverse_opf.synthetic import make_synthetic_dataset
from inverse_opf.training import TrainingConfig, train_inverse_model


def _small_dataset(n_strata: int = 4):
    return make_synthetic_dataset(
        n_buses=6,
        n_lines=8,
        n_train=40,
        n_val=20,
        demand_mean=30.0,
        demand_std=5.0,
        observation_noise_std=0.5,
        true_f_range=(5.0, 40.0),
        gmax_scale=(1.05, 1.3),
        pmax_scale=(0.8, 1.2),
        n_strata=n_strata,
        seed=0,
    )


def test_incidence_is_signed():
    rng = np.random.default_rng(0)
    a = random_connected_incidence(6, 8, rng)
    assert a.shape == (6, 8)
    assert np.all(np.abs(a.sum(axis=0)) < 1e-9)


def test_forward_feasibility():
    ds = _small_dataset()
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    gmax = ds.true_gmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1)
    pmax = ds.true_pmax.unsqueeze(0).repeat(ds.d_train.shape[0], 1)
    f = ds.true_f_table[ds.strata_train]
    with torch.no_grad():
        g, p = opf.solve(ds.d_train, f, gmax, pmax)
    assert torch.all(g >= -1e-2)
    assert torch.all(g <= gmax + 1e-2)
    assert torch.all(p.abs() <= pmax + 1e-2)


def test_training_improves_recovery():
    ds = _small_dataset()
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    model = InverseOPFModel(n_buses=6, n_lines=8)
    cfg = TrainingConfig(steps=60, lr=0.02, loss="mse", l2_weight=1e-5, slack_l1_weight=0.0)
    res = train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=ds.d_train,
        g_train_obs=ds.g_train_obs,
        d_val=ds.d_val,
        g_val_obs=ds.g_val_obs,
        train_cfg=cfg,
    )
    assert res.best_val_rmse <= res.history[0]["val_rmse"] + 1e-6


def test_stratified_laplacian_runs():
    ds = _small_dataset(n_strata=4)
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    model = StratifiedInverseOPFModel(n_buses=6, n_lines=8, n_strata=4)
    lap = torch.tensor(cycle_laplacian(4), dtype=torch.float32)
    cfg = TrainingConfig(
        steps=20, lr=0.05, loss="huber", laplacian_weight=0.05, slack_l1_weight=1e-3
    )
    res = train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=ds.d_train,
        g_train_obs=ds.g_train_obs,
        d_val=ds.d_val,
        g_val_obs=ds.g_val_obs,
        train_cfg=cfg,
        strata_train=ds.strata_train,
        strata_val=ds.strata_val,
        laplacian=lap,
    )
    assert np.isfinite(res.best_val_rmse)


def test_fixed_capacity_baseline():
    ds = _small_dataset()
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    model = FixedCapacityInverseOPFModel(
        n_buses=6,
        n_lines=8,
        gmax=ds.true_gmax,
        pmax=ds.true_pmax,
    )
    cfg = TrainingConfig(steps=20, lr=0.05, loss="mse")
    train_inverse_model(
        model=model,
        opf_layer=opf,
        d_train=ds.d_train,
        g_train_obs=ds.g_train_obs,
        d_val=ds.d_val,
        g_val_obs=ds.g_val_obs,
        train_cfg=cfg,
    )
    params = model.current_parameters()
    assert torch.allclose(params.gmax, ds.true_gmax)
    assert torch.allclose(params.pmax, ds.true_pmax)


def test_mef_and_counterfactual():
    ds = _small_dataset()
    opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence))
    f = ds.true_f_table.mean(dim=0)
    d0 = ds.d_val[0]
    jac = jacobian_g_wrt_d(opf, d0, f, ds.true_gmax, ds.true_pmax)
    assert jac.shape == (6, 6)
    c = torch.ones(6)
    mef = marginal_emission_factors(jac, c)
    assert mef.shape == (6,)
    g_cf = counterfactual_dispatch(
        opf, d0, f, ds.true_gmax, ds.true_pmax, delta_d=torch.ones_like(d0) * 0.5
    )
    assert g_cf.shape == (6,)
