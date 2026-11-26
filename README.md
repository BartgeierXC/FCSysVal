# FCSysVal publication toolchain

Code to reproduce the system-level validation of the physical PEM fuel cell system model in the accompanying publication. The model itself is the Modelica library [VirtualFCS-SysVal](https://github.com/BartgeierXC/VirtualFCS-SysVal). The measurement data and the simulation results of the publication are published in the Zenodo record [10.5281/zenodo.24310446](https://doi.org/10.5281/zenodo.24310446).

This repository holds the validation runs only: the quasi-stationary validation dataset D_0 (run D0), the transient runs T01 to T04 and the coverage of the validation operating domain. Parameter Identification is not part of it.

## Installation

Python 3.13.5 with the pinned packages:

    pip install -r requirements.txt

## Data and local paths

All measured data come from the raw measurement file of the Zenodo record, `raw_2026-07-16_2026-07-31.csv`. Copy `.env.example` to `.env` and set the paths, or set them as environment variables. `.env` is not versioned.

| Variable | Meaning |
|---|---|
| `FCS_MEASUREMENT_CSV` | Raw measurement file of the record (required) |
| `FCS_SIMULATION_ROOT` | Folder holding the unpacked simulation results, `iteration1/<run>/` and `iteration2/<run>/` |
| `MODELICA_LUT_PATH` | Simulation input table written by the run notebooks |
| `FCS_DOWNLOAD_DIR` | Folder for intermediate files |

## Reproduce the published results

    python scripts/build_figures.py --parameter-set all

This writes `results/iteration1/`, `results/iteration2/` and the coverage files to `results/`. `--skip-convergence` skips the pool convergence check, which takes longest. The full build takes about 10 minutes.

## Workflow of a single run

Each run has one config `configs/run_<ID>.yml` and one notebook `notebooks/run_<ID>.ipynb`. The notebooks show each step of the workflow with its intermediate results. The logic of the steps is in `src`.

1. Set `WRITE_INPUT = True` in the run notebook and execute it up to section 4. It loads the measured data and writes the simulation input table. With the default `False`, the input table on disk stays unchanged.
2. Simulate the model in OpenModelica with the printed settings. Simulations are started manually.
3. Evaluate the stored result with the same notebook.

Simulated and measured signals are compared on the measured time base, in seconds since the start of the run.

## Repository layout

| Folder | Content |
|---|---|
| `configs/` | Run and simulation configurations, operating conditions of D_0, measurement uncertainty budget |
| `notebooks/` | One notebook per run, and the coverage notebook |
| `scripts/` | Build of all figures and tables |
| `src/` | Data processing, workflow steps, figures, utilities |
| `tests/` | Unit tests |
| `data/parameters/` | Signal names |

## Tests

    python -m pytest tests

Five tests of unit conversions that the published runs do not use are skipped, their reasons are given in the test file.

## License

MIT, Copyright (c) 2026 Zentrum für Sonnenenergie- und Wasserstoff-Forschung Baden-Württemberg (ZSW). See `LICENSE`.
