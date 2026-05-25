# Inverse Optimal Power Flow via Differentiable Convex Optimization

EE 364B Final Project — Carlo Schreiber.

This repository implements a differentiable DC-OPF layer used to learn
dispatch parameters from observed (demand, generation) pairs. Beyond the
Fuentes Valenzuela & Degleris [14] baseline, it adds three convex extensions
that the proposal argues are necessary for fitting real grid data:

1. **Extended parameter vector.** Learn line and generator capacities
   $g_{\max}, p_{\max}$ in addition to the cost vector $f$, so dispatch driven
   by binding constraints can be explained.
2. **Stratified, time-varying parameters.** A Laplacian-regularized stratified
   model lets $f^{(s)}$ vary across hour-of-day / season strata while sharing
   strength across adjacent strata.
3. **Robust loss with active-set sparsity.** Huber residual loss plus an
   $\ell_1$ slack penalty on a lifted formulation, designed to tolerate
   sensor noise, derates, and unit-commitment effects.

## Repository layout

```
src/inverse_opf/        # core package
  dc_opf.py             # differentiable DC-OPF QP layer (cvxpylayers)
  model.py              # InverseOPFModel, StratifiedInverseOPFModel
  baselines.py          # FixedCapacityInverseOPFModel (F&D-style)
  losses.py             # Huber residual, slack L1, L2, Laplacian smoothness
  training.py           # gradient training loop with regularizers
  synthetic.py          # synthetic dataset generator (stratified costs)
  graph.py              # cycle / cycle x path Laplacians, incidence matrices
  sensitivity.py        # MEF Jacobians and counterfactual dispatch
  metrics.py, io.py, plotting.py
configs/                # YAML experiment configurations
scripts/                # entry-point experiments
tests/                  # pytest smoke tests
paper/figures/          # PDFs auto-mirrored from runs
outputs/<run_name>/     # per-run CSV / JSON / PNG artifacts
```

## Reproducing all paper figures

```bash
make install
make test
make figures   # ablation, recovery, noise sweep, MEF analysis
```

Individual targets are also available: `make synthetic`, `make ablation`,
`make recovery`, `make noise`, `make mef`. All scripts accept
`--config <path>`; sweep scripts also accept `--seeds` and `--sizes` /
`--noises`.

## Figures

| Figure                                  | Description                                                                                                  | Source                            |
|-----------------------------------------|--------------------------------------------------------------------------------------------------------------|-----------------------------------|
| `paper/figures/ablation_summary.pdf`    | Best validation RMSE and $f, g_{\max}, p_{\max}$ recovery across the four ablation variants.                 | `scripts/run_ablation.py`         |
| `paper/figures/recovery_curve.pdf`      | $\cos\angle(\hat f, f^\star)$ vs. dataset size, mean $\pm$ 1 std over seeds.                                 | `scripts/run_recovery_curve.py`   |
| `paper/figures/noise_sweep.pdf`         | Cosine recovery and validation RMSE vs. observation noise $\sigma$.                                          | `scripts/run_noise_sweep.py`      |
| `paper/figures/merit_order.pdf`         | Learned merit-order curve overlaid on the ground-truth ordering.                                             | `scripts/run_mef_analysis.py`     |
| `paper/figures/mef_heatmap.pdf`         | Marginal emission factor Jacobian $\partial g^*/\partial d$ at a sample operating point.                     | `scripts/run_mef_analysis.py`     |

## Method summary

**Forward problem.** For incidence $A \in \mathbb{R}^{n\times m}$, generation
$g$, flows $p$, demand $d$:

$$
\min_{g,p}\ \tfrac{1}{2}\, g^\top \mathrm{diag}(f)\, g + \tfrac{\tau}{2}\|p\|_2^2
\quad \text{s.t.}\quad g - d = A p,\ 0 \le g \le g_{\max},\ |p|\le p_{\max}.
$$

**Inverse problem.** Given $T$ observations $\{(d_t, g_t^{\mathrm{obs}})\}$,

$$
\min_{\theta\in\Theta}\ \tfrac{1}{T}\sum_t \ell\bigl(g^\star(\theta, d_t) - g_t^{\mathrm{obs}}\bigr)
+ \lambda R(\theta),
$$

with $\ell$ Huber, $R$ a sum of L2 + Laplacian smoothness on $f^{(s)}$ +
$\ell_1$ slack penalty, and $\theta = (f, g_{\max}, p_{\max})$. Gradients
$\nabla_\theta g^\star$ come from `cvxpylayers` via implicit KKT
differentiation [1, 3, 5]. Optimization uses Adam with gradient clipping.

**Stratified model.** $f^{(s)}\in\mathbb{R}^n$ for each stratum $s$ in a
hour-of-day cycle (or hour × month grid via
`graph.product_cycle_path_laplacian`); regularization uses
$\mathrm{tr}(F^\top L F)$ with $L$ the cycle/grid Laplacian.

**Downstream sensitivity.** `sensitivity.jacobian_g_wrt_d` returns
$\partial g^\star/\partial d$ via autograd; `marginal_emission_factors`
applies $J^\top c$ for any per-bus carbon intensity $c$;
`counterfactual_dispatch` recomputes $g^\star$ under demand perturbations.

## Implementation notes

* Default solver is SCS with tightened tolerances; ECOS can be enabled by
  passing `solver_args={"solve_method": "ECOS"}` to `DCOpfLayer.solve`.
* Positive parameters use `softplus` parameterization with inverse-softplus
  initialization so requested initial scales are honored exactly.
* Synthetic capacities are demand-aware ($g_{\max}$ above sampled peak demand),
  guaranteeing pointwise feasibility for the forward QP.
# Inverse Optimal Power Flow (EE364B Final Project)

This repository implements a practical MVP of the proposal:

- Forward model: differentiable DC-OPF QP.
- Inverse model: fit unknown parameters from observed `(d, g_obs)`.
- Extensions: learned generation and line capacities, robust Huber loss, and stratified (time-varying) costs with Laplacian regularization.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/run_synthetic_experiment.py --config configs/synthetic_small.yaml
```

Artifacts are written to `outputs/<run_name>/`.

## Structure

- `src/inverse_opf/dc_opf.py`: CVXPY + cvxpylayers forward OPF layer.
- `src/inverse_opf/losses.py`: robust losses and regularizers.
- `src/inverse_opf/model.py`: inverse model parameterization.
- `src/inverse_opf/training.py`: gradient-based training loop.
- `src/inverse_opf/synthetic.py`: synthetic data generation.
- `src/inverse_opf/metrics.py`: recovery and fit metrics.
- `scripts/run_synthetic_experiment.py`: end-to-end experiment script.

## Notes

- This code targets synthetic validation first and is designed so real data ingestion can be added in `data/` + a preprocessing module.
- The OPF formulation uses a strongly convex objective on generation and flow, making the solution map well-behaved for differentiation.
- When `model.use_stratified_costs: true`, synthetic data is generated with truly time-varying per-stratum costs and reported with both mean-cost and full-table recovery metrics.
