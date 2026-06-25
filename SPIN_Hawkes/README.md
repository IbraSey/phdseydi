# SPIN-Hawkes

This repository contains the full codebase associated with the preprint ** SPIN-H ** (https://arxiv.org/abs/...). 

It includes implementations of the spatially structured sigmoidal Gaussian Cox process (SSGC) and 
its Spatially Informed Hawkes (SPIN-H) extension.


---

## Installation guide

### Clone the repository

```bash
git clone https://github.com/IbraSey/SPIN_H.git
cd SPIN-H
```

### Python package



---

## Package layout

```text
artifacts/                      Generated figures, ignored by git
experiments/                    Runnable usage
  exp_ssgc/
    experiment_1.py
    experiment_2.py
    experiment_3.py
    experiment_4.py
  exp_spin_h/
    experiment_5.py
    experiment_6.py
    experiment_7.py
    experiment_8.py
    experiment_9.py
    experiment_10.py
spin_h/
  models/
    ssgc.py                     SSGC model
    spinh.py                    SPIN-H model
    kernels.py                  ETAS productivity, temporal and spatial kernels
  inference/
    base.py                     Inference facade
    gibbs.py                    Current Gibbs implementation
    backends.py
    results.py                  Posterior summaries and diagnostics
  simulation/                   Point-process simulation and partition
    process.py                  Point-process simulation
    tessellation.py             Tessellation
  visualization/
    fields.py
    diagnostics.py
  data/
    catalog.py
  spatial/
    domain.py
  config.py                     Model and inference configuration
tests/                          Automated tests
README.md
```


---

## SSGC experiments

The four SSGC studies are available in `examples/experiments/ssgc/` and can be run as Python modules. 
See `examples/experiments/ssgc/README.md` for the study matrix.

All active plotting functions use `gp.visualization.save_figure`. 
Saved images are written at **50 dpi** (à modifier potentiellement, dans fields.py) below `artifacts/figures/`.


---

## Citation

If you use this repository, please cite the associated manuscript:

```bibtex
@article{seydi:spin_h:2026,
    title = {XXX}, 
    author = {Ibrahim Seydi and Sophie Donnet and Merlin Keller and Joseph Muré and Julien Stoehr},
    year = {2026},
    eprint = {XXX.XXX},
    archivePrefix = {arXiv},
    url = {https://arxiv.org/abs/XXX}, 
    note = {Code available at https://github.com/jstoehr/eHMC}
}
```


---

## License

See the package-specific license files `...`, `...`, `...`.


