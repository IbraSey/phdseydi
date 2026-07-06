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
spin_h/
  figures/                        generated figures
  experiments/                    runnable experiment protocols
    exp_ssgc/
      experiment_1.py
      experiment_2.py
      experiment_3.py
      experiment_4.py
  models/
    base.py                       common model interface
    ssgc.py                       SSGC model
    spinh.py                      SPIN-H model
    kernels.py                    ETAS productivity, temporal and spatial kernels
  inference/
    ssgc_gibbs.py                 SSGC Gibbs implementation
    spinh_gibbs.py                SPIN-H Gibbs implementation
    backends.py
    results.py                    posterior summaries and diagnostics
  simulation/
    process.py                    point-process simulation
    tessellation.py               tessellation
  visualization/
    plots.py
  data/
    catalog.py
  spatial/
    domain.py
  config.py                       model and inference configuration
  first_test_spinh.py              first SPIN-H test
  how_to_use.md                    user guide
  README.md
```


---

## SSGC experiments

The four SSGC studies are available in `examples/experiments/ssgc/` and can be run as Python modules. 
See `examples/experiments/ssgc/README.md` for the study matrix.

All active plotting functions use `gp.visualization.save_figure`. 
Vector figures are saved as PDF; raster figures are saved as PNG at **600 dpi** (à modifier potentiellement, dans visualization/plots.py) below `figures/`.


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
