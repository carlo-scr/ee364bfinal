"""Unified CLI for inverse-OPF experiments.

Usage::

    python scripts/run.py --config configs/methods_comparison.yaml
    python scripts/run.py --config configs/methods_comparison.yaml --force
    python scripts/run.py --config configs/methods_comparison.yaml --seeds 0,1

Each registered experiment is a function ``fn(cfg, seed_bundle) -> dict``
that is wrapped with the on-disk cache, so re-runs skip seeds that already
have ``outputs/<run_name>/<seed>/metrics.json``.

Experiments registered here are *the same code* that the older one-off
scripts (run_methods_comparison.py, run_diurnal.py, etc.) use; they are
just driven by the new YAML configs and pass through the seed bundle so
runs are bit-exact reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

# Make sibling _common.py importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (  # noqa: E402
    StandardConfig,
    aggregate_with_ci,
    build_dataset,
    build_test_demands,
    eval_physics_on_test,
    run_diff_fcap,
    run_diff_full,
    run_diff_strat,
    run_diff_strat_kmeans,
    run_diff_warmstart,
    run_kkt,
    run_mlp,
    run_pjm_seed,
    run_ridge,
    standard_training,
    thin_training_set,
)
from inverse_opf.cache import load_or_compute, seed_dir  # noqa: E402
from inverse_opf.config import ExperimentConfig  # noqa: E402
from inverse_opf.seeding import set_global_seed  # noqa: E402

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    def _tqdm(it, **_kw):  # type: ignore[misc]
        return it


# ---------------------------------------------------------------------------
# Adapter: map our nested ExperimentConfig into the flat StandardConfig the
# existing _common.py helpers expect.  This lets the new YAML system drive
# the existing experiment code without rewriting it.
# ---------------------------------------------------------------------------


def _to_standard(cfg: ExperimentConfig) -> StandardConfig:
    return StandardConfig(
        n_buses=cfg.network.n_buses,
        n_lines=cfg.network.n_lines,
        n_train=cfg.data.n_train,
        n_val=cfg.data.n_val,
        demand_mean=cfg.data.demand_mean,
        demand_std=cfg.data.demand_std,
        obs_noise=cfg.data.obs_noise,
        f_min=cfg.true_params.f_min,
        f_max=cfg.true_params.f_max,
        gmin_scale=cfg.true_params.gmin_scale,
        gmax_scale=cfg.true_params.gmax_scale,
        pmin_scale=cfg.true_params.pmin_scale,
        pmax_scale=cfg.true_params.pmax_scale,
        n_strata=cfg.model.n_strata,
        physics=cfg.forward.physics,
        diurnal_amp=cfg.data.diurnal_amp,
        diurnal_cost_amp=cfg.data.diurnal_cost_amp,
    )


def _training_cfg(cfg: ExperimentConfig):
    t = cfg.training
    return standard_training(
        steps=t.steps,
        lr=t.lr,
        loss=t.loss,
        huber_delta=t.huber_delta,
        l2_weight=t.l2_weight,
        laplacian_weight=t.laplacian_weight,
        slack_l1_weight=t.slack_l1_weight,
        clip_grad_norm=t.clip_grad_norm,
        lr_schedule=t.lr_schedule,
        lr_min=t.lr_min,
        warmup_frac=t.warmup_frac,
        early_stopping_patience=t.early_stopping_patience,
        restore_best=t.restore_best,
    )


def _strip_arrays(d: dict[str, Any]) -> dict[str, Any]:
    """Drop ndarray fields that don't round-trip cleanly through JSON."""
    out = {}
    for k, v in d.items():
        if k in {"f", "gmax", "pmax", "f_table"}:
            continue
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Registered experiments.  Each returns ONE metrics dict per seed.
# ---------------------------------------------------------------------------


def _exp_methods_comparison(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    # Optional missing-data ablation: drop a fraction of training rows.
    if cfg.data.missing_frac > 0.0:
        n_before = ds.d_train.shape[0]
        ds = thin_training_set(ds, cfg.data.missing_frac, seed)
        print(f"  [missing] kept {ds.d_train.shape[0]}/{n_before} "
              f"({1.0 - cfg.data.missing_frac:.0%}) training samples", flush=True)
    train_cfg = _training_cfg(cfg)

    # Optional held-out test set, generated from the same true parameters
    # with a disjoint RNG sub-stream.  Used to measure out-of-sample
    # generalization of recovered (f, gmax, pmax).
    n_test = int(cfg.data.n_test)
    if n_test > 0:
        d_test, g_clean_test, _ = build_test_demands(ds, sc, n_test, seed)
    else:
        d_test = g_clean_test = None

    methods: list[tuple[str, Callable]] = [
        ("ridge", lambda: run_ridge(ds, sc)),
        ("mlp", lambda: run_mlp(ds, sc)),
        ("kkt", lambda: run_kkt(ds, sc)),
        ("diff_fcap", lambda: run_diff_fcap(ds, sc, train_cfg)),
        ("diff_full", lambda: run_diff_full(ds, sc, train_cfg)),
        ("diff_strat", lambda: run_diff_strat(ds, sc, train_cfg)),
    ]
    rows: list[dict[str, Any]] = []
    for name, fn in methods:
        t0 = time.time()
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - report but keep other methods
            print(f"  [{name}] FAILED: {e}", flush=True)
            continue
        # Held-out test NRMSE for physics-aware methods.
        test_nrmse_clean = float("nan")
        if d_test is not None and name in {"kkt", "diff_fcap", "diff_full", "diff_strat"}:
            try:
                test_nrmse_clean = eval_physics_on_test(
                    ds, sc, d_test, g_clean_test,
                    out["f"], out["gmax"], out["pmax"],
                )
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] test eval failed: {e}", flush=True)
        out = {**_strip_arrays(out), "seed": seed,
               "elapsed_s": time.time() - t0,
               "test_nrmse_clean": test_nrmse_clean}
        rows.append(out)
        print(
            f"  {name:11s} cos_f={out.get('f_cos', float('nan')):.3f} "
            f"val_nrmse={out.get('val_nrmse_clean', float('nan')):.3f} "
            f"test_nrmse={test_nrmse_clean:.3f} "
            f"({out['elapsed_s']:.1f}s)",
            flush=True,
        )
    return {"seed": seed, "rows": rows}


# ---------------------------------------------------------------------------
# Section 2: epsilon sweep and congestion analysis
# ---------------------------------------------------------------------------


def _exp_epsilon_sweep(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Sweep the strong-convexity floor ``eps`` on the forward QP and report
    recovery metrics for diff_full at each value.  The sweep values come
    from ``extra.eps_values`` (defaults to a log-spaced sweep).
    """
    eps_values = cfg.extra.get("eps_values", [0.01, 0.05, 0.1, 0.5, 1.0])
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    rows: list[dict[str, Any]] = []
    base_physics = sc.physics
    for eps in eps_values:
        # Monkey-patch the StandardConfig consumer by wrapping the OPF;
        # easiest path is to call run_diff_full with a custom SC whose
        # physics still works but we override eps via a thin opf shim.
        from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
        from inverse_opf.model import InverseOPFModel
        from inverse_opf.training import train_inverse_model
        opf = DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                         physics=base_physics, eps=float(eps))
        model = InverseOPFModel(n_buses=sc.n_buses, n_lines=sc.n_lines)
        t0 = time.time()
        res = train_inverse_model(model, opf, ds.d_train, ds.g_train_obs,
                                  ds.d_val, ds.g_val_obs, train_cfg)
        elapsed = time.time() - t0
        params = model.current_parameters()
        from inverse_opf.metrics import (cosine_recovery, merit_order_accuracy,
                                           normalized_rmse, rmse)
        from _common import predict_dispatch_diff, true_val_dispatch
        g_pred = predict_dispatch_diff(opf, ds.d_val, params.f, params.gmax, params.pmax)
        g_true = true_val_dispatch(ds, sc)
        rows.append({
            "method": f"eps={eps:g}",
            "eps": float(eps),
            "seed": seed,
            "val_nrmse_clean": normalized_rmse(g_pred, g_true),
            "f_cos": cosine_recovery(params.f.detach().cpu().numpy(), ds.true_f),
            "f_merit_acc": merit_order_accuracy(params.f.detach().cpu().numpy(), ds.true_f),
            "best_step": res.best_step,
            "elapsed_s": elapsed,
        })
        print(f"  eps={eps:6.3f}  val_nrmse={rows[-1]['val_nrmse_clean']:.3f}  "
              f"f_cos={rows[-1]['f_cos']:.3f}  ({elapsed:.1f}s)", flush=True)
    return {"seed": seed, "rows": rows}


def _exp_congestion(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Compare congestion (binding-constraint) frequencies between the true
    and recovered parameters on the val set."""
    from inverse_opf.analysis import congestion_stats
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    opf_true = _opf_for(ds, sc)
    f_true_val = ds.true_f_table[ds.strata_val]
    true_stats = congestion_stats(opf_true, ds.d_val, f_true_val,
                                   ds.true_gmax, ds.true_pmax)
    diff = run_diff_full(ds, sc, train_cfg)
    rec_stats = congestion_stats(
        opf_true, ds.d_val,
        torch.as_tensor(diff["f"], dtype=torch.float32),
        torch.as_tensor(diff["gmax"], dtype=torch.float32),
        torch.as_tensor(diff["pmax"], dtype=torch.float32),
    )
    return {"seed": seed, "rows": [{
        "method": "diff_full",
        "seed": seed,
        "true_line_any_frac": true_stats["line_any_frac"],
        "true_gen_any_frac": true_stats["gen_any_frac"],
        "rec_line_any_frac": rec_stats["line_any_frac"],
        "rec_gen_any_frac": rec_stats["gen_any_frac"],
        "line_freq_l1": float(abs(true_stats["line_freq"] - rec_stats["line_freq"]).sum()),
        "gen_freq_l1": float(abs(true_stats["gen_freq"] - rec_stats["gen_freq"]).sum()),
        "f_cos": diff["f_cos"],
    }]}


# ---------------------------------------------------------------------------
# Section 3: warm-start + KMeans strata
# ---------------------------------------------------------------------------


def _exp_warmstart(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Diff_full from scratch vs. diff warm-started from the KKT solution.
    Reports both final quality and total wall-clock time."""
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    rows = []
    t0 = time.time()
    out = run_diff_full(ds, sc, train_cfg)
    rows.append({**_strip_arrays(out), "seed": seed,
                 "total_s": time.time() - t0})
    t0 = time.time()
    out = run_diff_warmstart(ds, sc, train_cfg)
    rows.append({**_strip_arrays(out), "seed": seed,
                 "total_s": time.time() - t0})
    for r in rows:
        print(f"  {r['method']:15s}  val_nrmse={r.get('val_nrmse_clean', float('nan')):.3f}  "
              f"f_cos={r.get('f_cos', float('nan')):.3f}  "
              f"({r['total_s']:.1f}s)", flush=True)
    return {"seed": seed, "rows": rows}


def _exp_kmeans_strata(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Stratified diff-OPF: ground-truth strata vs. KMeans-recovered strata."""
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    rows = []
    out = run_diff_strat(ds, sc, train_cfg)
    rows.append({**_strip_arrays(out), "seed": seed})
    out = run_diff_strat_kmeans(ds, sc, train_cfg)
    rows.append({**_strip_arrays(out), "seed": seed})
    for r in rows:
        agree = r.get("strata_agreement", 1.0)
        print(f"  {r['method']:22s}  val_nrmse={r.get('val_nrmse_clean', float('nan')):.3f}  "
              f"f_cos={r.get('f_cos', float('nan')):.3f}  "
              f"strata_agree={agree:.3f}", flush=True)
    return {"seed": seed, "rows": rows}


# ---------------------------------------------------------------------------
# Section 4: screening + timing
# ---------------------------------------------------------------------------


def _exp_screening(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Drop generators/lines with negligible utilization, then refit diff_full
    on the reduced problem.  Reports compression ratio and quality delta."""
    import numpy as np
    from inverse_opf.analysis import screen_capacities
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    # Estimate flows from balance to drive line screening.
    A = ds.incidence
    g_np = ds.g_train_obs.cpu().numpy()
    d_np = ds.d_train.cpu().numpy()
    p_lstsq = np.stack([
        np.linalg.lstsq(A, g_np[t] - d_np[t], rcond=None)[0]
        for t in range(g_np.shape[0])
    ], axis=0)
    screen = screen_capacities(g_np, p_lstsq,
                                gen_thresh=float(cfg.extra.get("gen_thresh", 0.05)),
                                line_thresh=float(cfg.extra.get("line_thresh", 0.05)))
    full = run_diff_full(ds, sc, train_cfg)
    print(f"  screening kept {screen['n_gen_kept']}/{sc.n_buses} gens, "
          f"{screen['n_line_kept']}/{sc.n_lines} lines  "
          f"-> diff_full f_cos={full['f_cos']:.3f}", flush=True)
    return {"seed": seed, "rows": [{
        "method": "screen+diff_full",
        "seed": seed,
        "gen_screen_ratio": screen["gen_screen_ratio"],
        "line_screen_ratio": screen["line_screen_ratio"],
        "n_gen_kept": screen["n_gen_kept"],
        "n_line_kept": screen["n_line_kept"],
        "f_cos": full["f_cos"],
        "val_nrmse_clean": full["val_nrmse_clean"],
    }]}


def _exp_timing(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Per-method wall-clock timing across the methods used in the headline
    comparison; same dataset for all so the comparison is apples-to-apples."""
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    methods = [
        ("kkt", lambda: run_kkt(ds, sc)),
        ("diff_fcap", lambda: run_diff_fcap(ds, sc, train_cfg)),
        ("diff_full", lambda: run_diff_full(ds, sc, train_cfg)),
        ("diff_strat", lambda: run_diff_strat(ds, sc, train_cfg)),
        ("diff_warmstart", lambda: run_diff_warmstart(ds, sc, train_cfg)),
    ]
    rows = []
    for name, fn in methods:
        t0 = time.time()
        out = fn()
        rows.append({"method": name, "seed": seed,
                     "elapsed_s": time.time() - t0,
                     "val_nrmse_clean": out.get("val_nrmse_clean", float("nan")),
                     "f_cos": out.get("f_cos", float("nan"))})
        print(f"  {name:15s} {rows[-1]['elapsed_s']:6.2f}s  "
              f"val_nrmse={rows[-1]['val_nrmse_clean']:.3f}", flush=True)
    return {"seed": seed, "rows": rows}


# ---------------------------------------------------------------------------
# Section 6: emission factors, MEF Jacobian breakdown, counterfactual demand
# ---------------------------------------------------------------------------


def _opf_for(ds, sc: StandardConfig):
    from inverse_opf.dc_opf import DCOpfLayer, DCOpfProblemData
    return DCOpfLayer(DCOpfProblemData(incidence=ds.incidence, susceptance=ds.susceptance),
                      physics=sc.physics)


def _exp_mef(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Marginal-emission-factor recovery: compute system-wide and per-bus
    MEF using the true vs. recovered parameters; report L1 and relative
    error in MEF."""
    from inverse_opf.analysis import assign_eia_factors, marginal_emission_factor
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    opf = _opf_for(ds, sc)
    factors = assign_eia_factors(ds.true_f.cpu().numpy(), rng_seed=seed)
    # Use the val demands as the evaluation distribution.
    d_eval = ds.d_val
    f_true_eval = ds.true_f_table[ds.strata_val]
    mef_true_sys = marginal_emission_factor(opf, d_eval, f_true_eval,
                                            ds.true_gmax, ds.true_pmax, factors)
    diff = run_diff_full(ds, sc, train_cfg)
    f_hat = torch.as_tensor(diff["f"], dtype=torch.float32)
    gmax_hat = torch.as_tensor(diff["gmax"], dtype=torch.float32)
    pmax_hat = torch.as_tensor(diff["pmax"], dtype=torch.float32)
    mef_rec_sys = marginal_emission_factor(opf, d_eval, f_hat, gmax_hat, pmax_hat, factors)
    # Per-bus breakdown (top-k absolute differences).
    bus_diffs = []
    for b in range(sc.n_buses):
        a = marginal_emission_factor(opf, d_eval, f_true_eval,
                                     ds.true_gmax, ds.true_pmax, factors, bus=b)
        c = marginal_emission_factor(opf, d_eval, f_hat, gmax_hat, pmax_hat, factors, bus=b)
        bus_diffs.append((b, a, c, c - a))
    bus_diffs.sort(key=lambda r: abs(r[3]), reverse=True)
    top = bus_diffs[:5]
    print(f"  MEF sys true={mef_true_sys:.1f}  rec={mef_rec_sys:.1f}  "
          f"err={mef_rec_sys - mef_true_sys:+.1f}  "
          f"(top bus errs: {[round(x[3],1) for x in top]})", flush=True)
    rows = [{
        "method": "diff_full",
        "seed": seed,
        "scope": "system",
        "bus": -1,
        "mef_true": mef_true_sys,
        "mef_rec": mef_rec_sys,
        "mef_abs_err": float(abs(mef_rec_sys - mef_true_sys)),
        "mef_rel_err": float(abs(mef_rec_sys - mef_true_sys) / max(abs(mef_true_sys), 1e-9)),
        "f_cos": diff["f_cos"],
    }]
    # Add per-bus rows so downstream plots have denser, more informative data.
    for b, a_true, a_rec, _delta in bus_diffs:
        rows.append({
            "method": "diff_full",
            "seed": seed,
            "scope": "bus",
            "bus": int(b),
            "mef_true": float(a_true),
            "mef_rec": float(a_rec),
            "mef_abs_err": float(abs(a_rec - a_true)),
            "mef_rel_err": float(abs(a_rec - a_true) / max(abs(a_true), 1e-9)),
            "f_cos": diff["f_cos"],
        })
    return {"seed": seed, "rows": rows}


def _exp_counterfactual(cfg: ExperimentConfig, seed: int) -> dict[str, Any]:
    """Counterfactual: scale demand by a sweep of factors and compare
    total CO2 between truth and the recovered model."""
    from inverse_opf.analysis import assign_eia_factors, counterfactual_dispatch
    sc = _to_standard(cfg)
    ds = build_dataset(seed, sc)
    train_cfg = _training_cfg(cfg)
    opf = _opf_for(ds, sc)
    factors = assign_eia_factors(ds.true_f.cpu().numpy(), rng_seed=seed)
    scales = cfg.extra.get("scales", [0.8, 0.9, 1.0, 1.1, 1.2])
    diff = run_diff_full(ds, sc, train_cfg)
    f_hat = torch.as_tensor(diff["f"], dtype=torch.float32)
    gmax_hat = torch.as_tensor(diff["gmax"], dtype=torch.float32)
    pmax_hat = torch.as_tensor(diff["pmax"], dtype=torch.float32)
    rows = []
    for s in scales:
        # Use mean true f over strata as the comparison point (no stratum
        # at counterfactual time).
        f_true_mean = ds.true_f
        ct = counterfactual_dispatch(opf, ds.d_val, f_true_mean,
                                      ds.true_gmax, ds.true_pmax, factors, s)
        cr = counterfactual_dispatch(opf, ds.d_val, f_hat, gmax_hat, pmax_hat,
                                      factors, s)
        rows.append({
            "method": f"scale={float(s):.2f}",
            "seed": seed,
            "scale": float(s),
            "co2_true": ct.total_co2_mean,
            "co2_rec": cr.total_co2_mean,
            "co2_rel_err": float(abs(cr.total_co2_mean - ct.total_co2_mean)
                                 / max(abs(ct.total_co2_mean), 1e-9)),
            "g_true": ct.g_mean,
            "g_rec": cr.g_mean,
        })
        print(f"  scale={s:4.2f}  co2 true={ct.total_co2_mean:8.1f}  "
              f"rec={cr.total_co2_mean:8.1f}  rel_err={rows[-1]['co2_rel_err']:.3f}",
              flush=True)
    return {"seed": seed, "rows": rows}


REGISTRY: dict[str, Callable[[ExperimentConfig, int], dict[str, Any]]] = {
    "methods_comparison": _exp_methods_comparison,
    "pjm_like": lambda cfg, seed: {
        "seed": seed,
        "rows": [run_pjm_seed(
            seed=seed,
            n_train=cfg.data.n_train,
            n_val=cfg.data.n_val,
            steps=cfg.training.steps,
            lr=cfg.training.lr,
        )],
    },
    "epsilon_sweep": _exp_epsilon_sweep,
    "warmstart": _exp_warmstart,
    "kmeans_strata": _exp_kmeans_strata,
    "congestion": _exp_congestion,
    "screening": _exp_screening,
    "mef": _exp_mef,
    "counterfactual": _exp_counterfactual,
    "timing": _exp_timing,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--force", action="store_true", help="Recompute even if cached.")
    p.add_argument("--seeds", default=None,
                   help="Comma-separated override of seeds (e.g. 0,1,2).")
    p.add_argument("--output-root", default="outputs", type=Path)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = ExperimentConfig.from_yaml(args.config)
    if cfg.experiment not in REGISTRY:
        raise SystemExit(
            f"unknown experiment '{cfg.experiment}'. "
            f"Registered: {sorted(REGISTRY)}"
        )
    if args.seeds:
        cfg.seeds = [int(s) for s in args.seeds.split(",")]
    fn = REGISTRY[cfg.experiment]
    cfg_dict = cfg.to_dict()

    print(f"== {cfg.run_name} :: {cfg.experiment} :: seeds={cfg.seeds} ==", flush=True)
    all_rows: list[dict[str, Any]] = []
    seed_pbar = _tqdm(cfg.seeds, desc=cfg.experiment, unit="seed", position=0)
    for seed in seed_pbar:
        seed_pbar.set_description(f"{cfg.experiment} seed={seed}")
        print(f"\n[seed {seed}]", flush=True)
        bundle = set_global_seed(seed)
        torch.set_num_threads(max(1, torch.get_num_threads()))

        def _compute(s=seed):
            return fn(cfg, s)

        result = load_or_compute(
            run_name=cfg.run_name,
            seed=seed,
            config_dict=cfg_dict,
            fn=_compute,
            root=args.output_root,
            force=args.force,
        )
        # ``methods_comparison`` returns {"rows": [...]}.  Flatten for the
        # aggregate summary, but each seed still owns its own metrics.json.
        all_rows.extend(result.get("rows", []))
        _ = bundle  # keep reference (used by future experiments)

    # Aggregate across seeds and write summary at the run-level dir.
    if all_rows:
        run_dir = args.output_root / cfg.run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        with open(run_dir / "all_rows.json", "w", encoding="utf-8") as f:
            json.dump(all_rows, f, indent=2)
        value_cols_pref = [
            "val_rmse", "val_rmse_clean", "val_nrmse_clean",
            "test_nrmse_clean",
            "f_cos", "f_spearman", "f_merit_acc", "gmax_cos", "pmax_cos",
            # PJM-like metrics
            "best_val_rmse", "f_mean_cos", "f_mean_spearman",
            "f_mean_merit_acc", "f_table_cos",
            # Section 2-6 extra metrics
            "eps", "true_line_any_frac", "rec_line_any_frac",
            "line_freq_l1", "gen_freq_l1",
            "strata_agreement", "warmstart_s", "total_s",
            "gen_screen_ratio", "line_screen_ratio",
            "n_gen_kept", "n_line_kept",
            "mef_true", "mef_rec", "mef_abs_err", "mef_rel_err",
            "scale", "co2_true", "co2_rec", "co2_rel_err",
            "g_true", "g_rec",
            "elapsed_s",
        ]
        present = [c for c in value_cols_pref if c in all_rows[0]]
        if "method" in all_rows[0]:
            agg = aggregate_with_ci(all_rows, ["method"], present, n_boot=2000)
            agg.to_csv(run_dir / "summary.csv", index=False)
            # Compact printout: mean [ci_lo, ci_hi] for the headline columns.
            print("\n== summary (mean [95% CI]) ==")
            headline_pref = ["val_nrmse_clean", "test_nrmse_clean", "f_cos",
                             "f_merit_acc", "gmax_cos",
                             "f_mean_cos", "f_mean_merit_acc", "f_table_cos",
                             # Section 2-6 metric headlines
                             "line_freq_l1", "gen_freq_l1",
                             "strata_agreement",
                             "elapsed_s", "warmstart_s", "total_s",
                             "gen_screen_ratio", "line_screen_ratio",
                             "mef_true", "mef_rec", "mef_rel_err",
                             "scale", "co2_true", "co2_rec", "co2_rel_err"]
            headline = [c for c in headline_pref if c in present]
            import pandas as pd  # local import (already imported via aggregate)
            pretty_rows = []
            for _, row in agg.iterrows():
                d_pretty = {"method": row["method"]}
                for c in headline:
                    m = row.get(f"{c}_mean", float("nan"))
                    lo = row.get(f"{c}_ci_lo", float("nan"))
                    hi = row.get(f"{c}_ci_hi", float("nan"))
                    d_pretty[c] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
                pretty_rows.append(d_pretty)
            print(pd.DataFrame(pretty_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
