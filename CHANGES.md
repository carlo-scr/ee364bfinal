# Changes

## Sections 2–6 — sensitivity, regularization, screening, MEF & counterfactual

### Added

- `src/inverse_opf/analysis.py` (new module, ~210 lines):
  - `congestion_stats(opf, d, f, gmax, pmax, tol_frac)` — per-line / per-gen
    binding-constraint frequencies on the forward dispatch.
  - `kmeans_strata(demands, n_strata, seed)` — k-means stratum discovery
    over demand vectors (sklearn).
  - `strata_agreement(pred, truth)` — Hungarian-permutation agreement in
    `[0, 1]`.
  - `screen_capacities(g_obs, p_est, gen_thresh, line_thresh)` — drops
    generators / lines that are never close to saturation; returns boolean
    keep masks and compression ratios.
  - `EIA_FUEL_CO2` (kg/MWh, 8 fuels) + `assign_eia_factors(f_true)` —
    tiles the EIA CO2 ladder by merit order.
  - `marginal_emission_factor(opf, ...)` — system-wide or per-bus
    one-sided finite-difference MEF (kg CO2 / MWh).
  - `CounterfactualResult` + `counterfactual_dispatch(opf, ..., scale)` —
    re-solves dispatch at `d * scale` and returns mean total g, CO2, cost.
- `scripts/_common.py`:
  - `run_diff_warmstart(ds, sc, train_cfg)` — KKT-baseline warm-start of
    the differentiable model's softplus-raw parameters
    (`f_raw`, `gmax_raw`, `pmax_raw`) via `log(expm1(·))`.
  - `run_diff_strat_kmeans(ds, sc, train_cfg)` — same as `diff_strat`
    but with k-means-recovered strata instead of ground-truth labels;
    reports `strata_agreement`.
- `scripts/run.py`: 8 new registry entries — `epsilon_sweep`, `congestion`,
  `warmstart`, `kmeans_strata`, `screening`, `timing`, `mef`,
  `counterfactual`. Per-experiment value columns extended; auto-headlines
  for new metrics.
- `src/inverse_opf/training.py`: `TrainingHyper.warmup_frac` honoured by
  the inner loop (linear ramp up to `lr`, then cosine over the post-warmup
  portion).
- Configs (full): `configs/{epsilon_sweep,warmstart,kmeans_strata,
  congestion,screening,timing,mef,counterfactual,methods_comparison_dc}.yaml`.
- Configs (smoke): `configs/smoke_{epsilon,warmstart,kmeans,congestion,
  screening,timing,mef,counterfactual,dc}.yaml`.

### Verified

- All 9 smoke configs run to completion under
  `python scripts/run.py --config configs/smoke_*.yaml`.
- `pytest tests/ -q`: **16 / 16 passing** (no regressions in existing tests).

### Known limitations

- `counterfactual` rows are tagged `method="scale=<s>"` so each scale gets
  its own aggregator group; absolute CO2 error vs. truth at small models /
  short training budgets is large by design.
- `mef` per-bus Jacobian is implemented as repeated one-sided FD (one
  forward solve per bus) rather than implicit-diff; for IEEE-30-sized cases
  this is adequate but scales poorly past O(100) buses.
- `screening` reports compression ratios + a sanity diff_full fit but does
  **not** refit on the reduced topology (would require rebuilding incidence
  and is left as future work).

## Section 5 — PJM-like + missing-data ablation (added on top of partial)

### Added

- `_common.run_pjm_seed(seed, n_train, n_val, steps, lr)` — single-seed
  wrapper around the PJM-like 8-fuel synthetic stack (matches the metrics
  reported by `scripts/run_pjm_like.py` but without side-effecting plots).
- `_common.thin_training_set(ds, drop_frac, seed)` — drops a fraction of
  training *samples* uniformly at random per seed (the simplest missing-data
  variant; per-entry masking would require touching the loss kernel).
- `scripts/run.py`:
  - New registry entry `pjm_like` driven by `run_pjm_seed`.
  - `methods_comparison` now honours `data.missing_frac`; when > 0 it logs
    `[missing] kept K/N (P%) training samples` per seed before training.
  - Summary aggregator extended to pick up PJM columns
    (`best_val_rmse`, `f_mean_cos`, `f_mean_merit_acc`, `f_table_cos`).
- `configs/pjm_like.yaml` — 3-seed PJM-like headline config.
- `configs/methods_comparison_missing50.yaml` — 5-seed headline with 50% of
  training rows dropped.
- `configs/pjm_smoke.yaml`, `configs/smoke_missing.yaml` — tiny smoke configs.

### Verified

- `python scripts/run.py --config configs/pjm_smoke.yaml` finishes and
  prints PJM metrics with the new CI summary.
- `python scripts/run.py --config configs/smoke_missing.yaml` logs row
  thinning and produces a full 6-method comparison table with CIs.
- `pytest tests/ -q`: **16 / 16 passing**.

## Section 5 (partial) — held-out test set + bootstrap CIs + Colab pipeline

### Added

- `_common.py` now exports:
  - `bootstrap_ci_mean(values, n_boot=2000, alpha=0.05)` — percentile
    bootstrap CI for the mean across seeds.
  - `aggregate_with_ci(...)` — drop-in replacement for `aggregate` that
    appends `<col>_ci_lo` / `<col>_ci_hi` columns.
  - `build_test_demands(ds, sc, n_test, seed)` — samples fresh demands from
    a disjoint RNG sub-stream and forward-solves with the *true* parameters
    to obtain a clean held-out target.
  - `eval_physics_on_test(...)` — re-solves the QP with recovered
    `(f, gmax, pmax)` at the test demands and returns NRMSE vs. truth.

- `scripts/run.py` (methods_comparison experiment):
  - If `data.n_test > 0`, builds a held-out test set per seed and reports
    `test_nrmse_clean` for kkt / diff_fcap / diff_full / diff_strat.
  - Run-level `summary.csv` now uses `aggregate_with_ci`; the printed
    table shows `mean [ci_lo, ci_hi]` for the headline columns.

- `configs/methods_comparison_5seed.yaml` — headline 5-seed config (Table I).
- `configs/methods_comparison_tightcaps.yaml` — tight-capacity variant.

- `colab/run_all.ipynb` — single Colab notebook that clones the repo,
  installs the pinned stack, runs pytest, then runs the smoke /
  headline / tight-caps configs and the legacy noise / recovery scripts,
  and finally writes one consolidated markdown report
  (`/content/inverse_opf_report.md`) with every CSV.

### Verified

- `pytest tests/ -q`: **16 / 16 passing**.
- Smoke run with `n_test=30` prints test NRMSE per method and bootstrap CIs:
  ```
  == summary (mean [95% CI]) ==
   diff_full  val_nrmse_clean 0.482 [0.449, 0.516]  test_nrmse_clean 0.479 [0.455, 0.504]  ...
  ```

## Section 1 — Code quality & reproducibility

Foundational infrastructure for the rest of the overhaul.  Additive: every
existing script and test continues to work unchanged.

### Added

- `src/inverse_opf/seeding.py` — single source of truth for deterministic
  seeding.  `set_global_seed(seed)` seeds `random`, `numpy`, `torch` (CPU +
  CUDA) and sets `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`, and
  `torch.use_deterministic_algorithms(True, warn_only=True)`.  `SeedBundle`
  provides labelled independent sub-streams via `numpy.random.SeedSequence`,
  so different pieces of an experiment (data, init, training noise) get
  reproducible-but-independent RNGs from one parent seed.

- `src/inverse_opf/config.py` — strongly-typed `ExperimentConfig` dataclass
  (network / true-params / data / model / training / forward) with
  `from_yaml`, `from_dict`, `to_dict`.  Unknown keys in a YAML config raise
  `ValueError` rather than being silently ignored.

- `src/inverse_opf/cache.py` — `load_or_compute(run_name, seed, cfg, fn)`
  writes `outputs/<run_name>/<seed>/{config.yaml, metrics.json}` and
  short-circuits on re-run; `--force` re-computes.

- `scripts/run.py` — single config-driven CLI.  Usage:
  `python scripts/run.py --config configs/<name>.yaml [--seeds 0,1,2]
  [--force] [--output-root outputs]`.  Dispatches to a `REGISTRY` of
  experiment functions; currently routes to the existing `_common.py`
  experiment runners so the old per-method machinery is reused, not
  rewritten.  Writes per-seed `metrics.json`, run-level `all_rows.json`,
  and `summary.csv` (mean / std across seeds).

- `configs/methods_comparison.yaml` — first config in the new format
  (drives the main IEEE Table I comparison: ridge / mlp / kkt / diff_fcap /
  diff_full / diff_strat across 5 seeds).

- `configs/smoke.yaml` — fast end-to-end smoke test for the CLI / cache.

- `tests/test_forward_kkt.py` — forward QP correctness on random
  `(f, gmax, pmax, d)` triples: primal feasibility within tolerance, and
  primal objective matches a high-precision reference CVXPY solve.

- `tests/test_identifiability.py` — `identifiability_score` boundary
  cases: always-interior → `1.0`, always-at-bound → `0.0`, mixed → fraction.

- `tests/test_jacobian_fd.py` — autograd Jacobian `dg*/dd` matches central
  differences at strictly-interior operating points; row sums close to 1.

- `tests/test_kkt_baseline.py` — `kkt_residual_inverse` drives the
  stationarity residual to near zero on noiseless synthetic data and
  recovers `f` with high cosine similarity.

### Changed

- `pyproject.toml` — switched runtime deps from loose `>=` to
  compatible-release `~=` pins matching the verified working environment
  (`numpy~=2.4`, `scipy~=1.17`, `cvxpy~=1.8`, `cvxpylayers~=1.1`,
  `scs~=3.2`, `torch~=2.11`, `scikit-learn~=1.8`, `matplotlib~=3.10`,
  `pandas~=3.0`, `seaborn~=0.13`, `pyyaml~=6.0`).  Dev: `pytest~=9.0`,
  `ruff~=0.5`.  Added `scs`, `scikit-learn`, `seaborn` which were already
  used but only transitively declared.

### Verified

- Full test suite: **16 / 16 passing** (`pytest tests/ -v`, ~13 s).
- Smoke test: `scripts/run.py --config configs/smoke.yaml` runs all six
  methods on two seeds, writes cached `metrics.json` per seed; re-running
  short-circuits with no per-method output.
