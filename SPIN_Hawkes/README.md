# SPIN-Hawkes

This repository contains the code associated with the **SPIN-H** preprint.

It includes implementations of the spatially structured sigmoidal Gaussian Cox process (SSGC) and 
its Spatially Informed Hawkes (SPIN-H) extension.


---

## Installation guide

### Clone the repository

```bash
git clone https://github.com/IbraSey/SPIN_H.git
cd SPIN_H
```

The repository does not yet contain packaging metadata (`pyproject.toml` or
`setup.py`). Run scripts from the repository root so that the public `package`
module and the sibling modules are importable:

```bash
python experiments/second_test_spinh.py
```

Long inference and experiment loops use `tqdm` progress bars. The dependency
must therefore be available in the Python environment (`python -m pip install tqdm`).
The full experiment runners also require `joblib`, `threadpoolctl` and `psutil`
for isolated workers, native-thread limits and memory monitoring.

---

## Package layout

```text
data/
  catalog.py
experiments/                       runnable experiment protocols
  exp_ssgc/
    experiment_1.py
    experiment_2.py
    experiment_3.py
    experiment_4.py
  exp_spinh/
    test_simulations.py              final simulated Experiments 1 and 2
    test_fcat17.py                   final FCAT-17 comparison
    test_full_campaign.py            complete multi-core draft campaign
    experiment_5.py ... 9.py         exploratory development studies
  first_test_ssgc.py                first SSGC test
  first_test_spinh.py               first SPIN-H test
  second_test_spinh.py              SPIN-H VI smoke test
results/                            generated figures and experiment outputs
simulation/
  process.py                        point-process simulation
  tessellation.py                  tessellation
spatial/
  domain.py
package/
  models/
    base.py                         common model interface
    ssgc.py                         SSGC model
    spinh.py                        SPIN-H model
    kernels.py                      ETAS productivity, temporal and spatial kernels
  inference/
    ssgc_gibbs.py                   SSGC Gibbs implementation
    spinh_gibbs.py                  SPIN-H Gibbs implementation
    VI.py                           shared SSGC/SPIN-H variational inference
    backends.py
    results.py                      posterior summaries and diagnostics
  config.py                         model and inference configuration
visualization/
  plots.py
README.md
```


---

## Variational inference

Inference is selected from the model itself. An SSGC fit automatically omits
branching and ETAS components:

```python
from package import SSGCVIConfig

fit = ssgc_model.vi(
    catalog,
    config=SSGCVIConfig(
        n_iter=200,
        gp_backend="sparse",
        fixed_beta=None,  # learn beta when magnitudes are available
    ),
)
```

Use `SPINHVIConfig` with `spinh_model.vi(...)` for the complete Hawkes model.
Both calls return `VIResults`; the historical `SPINHVIResults` name remains
available for compatibility. The numerical campaigns use the HSGP backend for
MF-VI, but the exact backend remains available explicitly with
`SPINHVIConfig(gp_backend="exact")`.

### Posterior marginals

Gibbs and MF-VI results expose the same eight scalar-parameter sampling
interface. Their marginal posterior densities can therefore be compared in one
figure:

```python
from package import plot_spinh_parameter_marginals

comparison = plot_spinh_parameter_marginals(
    {
        "Gibbs HSGP": gibbs_hsgp,
        "Gibbs HSGP + truncated candidates": gibbs_truncated,
        "MF-VI HSGP": vi_hsgp,
        "MF-VI HSGP + truncated candidates": vi_truncated,
    },
    reference_posterior=gibbs_exact,
    reference_label="Gibbs GP exact",
    true_parameters=generating_etas,
    true_beta=generating_beta,
    savefigure=True,
)
```

The panels follow the order `beta`, `A`, `alpha`, `c`, `p`, `d`, `q`,
`gamma`. The exact-GP Gibbs curve is a numerical reference posterior, not an
analytically known posterior. In simulation studies, dotted vertical lines mark
the generating values separately. A list or tuple of Gibbs results can be
passed for any label to pool several chains.

### Sparse parent truncation for simulated catalogs

The dense parent support can be replaced by the temporal graph

```text
(j, i) is retained when 0 < t[i] - t[j] <= parent_time_window.
```

For simulated catalogs, choose `parent_time_window` from the temporal or
spatio-temporal ETAS-kernel diagnostic. A relative kernel height can optionally
be converted into a time lag using the known simulation parameters:

```python
from package import SPINHGibbsConfig, SPINHVIConfig

relative_height = 1e-3
parent_time_window = spinh_model.parent_time_window_from_kernel(relative_height)

vi_fit = spinh_model.vi(
    catalog,
    config=SPINHVIConfig(
        parent_time_window=parent_time_window,
        gp_backend="sparse",
    ),
)
kernel_diagnostic = vi_fit.plot_etas_kernel_dispersion(
    parent_time_window=parent_time_window,
)

gibbs_fit = spinh_model.gibbs(
    catalog,
    config=SPINHGibbsConfig(
        n_iter=8000,
        thin=5,
        adaptation_start=200,
        etas_adaptation_end=2000,
        parent_time_window=parent_time_window,
    ),
    gp_backend="sparse",
)

gibbs_fit.plot_traces(burn_in=0.25)
gibbs_fit.plot_acf(burn_in=0.25, max_lag=200, etas_only=True)
```

The sampler keeps complete scalar traces even when large latent states are
thinned. The default `mala_step=None` selects a catalogue-dependent step from
the initial zonal curvature; an explicit value overrides this choice.
Productivity `A` is sampled exactly from its Gamma full conditional.
When `A` and `alpha` are both free, the default partially collapsed update
integrates out `A` during the Metropolis step for `alpha`, then redraws `A`
exactly. The remaining ETAS blocks use adaptive Metropolis proposals whose
adaptation can be frozen before retained draws.

The plotted cutoff reports both the relative kernel height and the omitted
temporal mass. The same finite support is used in the eventwise triggering sum
and in the temporal compensator, so the approximate likelihood is internally
consistent. With truncation enabled, VI stores `q(Z)` as a CSR matrix and Gibbs
caches only candidate-pair lags and distances. `None`, the default, retains the
original dense support. Candidate counts, retained fractions and memory use are
available in `fit.diagnostics["branching_truncation"]` for VI and
`fit.raw["branching_truncation"]` for Gibbs.

Run the reproducible truncation-complexity benchmark with:

```bash
python -m experiments.benchmark_branching_truncation \
  --sizes 500,1000,2000,4000 \
  --event-rate 20 \
  --time-window 0.5 \
  --output results/branching_truncation_benchmark.csv
```


---

## SSGC experiments

The four simulated-data studies and the FCAT-17 block cross-validation are
available in `experiments/exp_ssgc/`. See
`experiments/exp_ssgc/README.md` for the `smoke` and deliverable-quality `full`
commands, output tables, figures and inference-method selectors.

Plotting functions use `visualization.save_figure`. Vector figures are saved as
PDF; rasterized artists use 300 dpi within the PDF. Default figure outputs are
written below `results/figures/` in the working directory.


---

## SPIN-H experiments

The draft-aligned simulated experiments and FCAT-17 comparison are documented
in `experiments/exp_spinh/README.md`. To validate every path quickly, run:

```bash
python experiments/exp_spinh/test_full_campaign.py --profile smoke
```

The production campaign is launched with:

```bash
python experiments/exp_spinh/test_full_campaign.py
```

It requests all usable workers, capped by the selected experiment and available
memory, and resumes compatible task checkpoints after an interruption.
For the simulated experiments alone, use
`experiments/exp_spinh/test_simulations.py`. Its first cell holds the launch
options; `--action postprocess --experiment 1` or `--experiment 2` rebuilds
saved simulation outputs without inference.


---

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{seydi:package:2026,
    title = {XXX}, 
    author = {Ibrahim Seydi and Sophie Donnet and Merlin Keller and Joseph Muré and Julien Stoehr},
    year = {2026},
    eprint = {XXX.XXX},
    archivePrefix = {arXiv},
    url = {https://arxiv.org/abs/XXX}, 
    note = {Code available at https://github.com/IbraSey/SPIN_H}
}
```


---

## License

A license has not yet been added to this repository.
