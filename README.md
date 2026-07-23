# EH — Shallow-time Earth Habitability

A reduced-complexity, fully reproducible proof-of-concept that projects
terrestrial habitability under coupled climate and ocean change, from the
preindustrial (1750) to 2300 CE.

It runs a single seeded pipeline:

```
real forcing/obs → two-layer energy-balance emulator → tempered SMC
calibration → six carried climate–ocean variables → tiered Composite
Hazard Score (CHS) → Habitable Area Fraction (HAF)
```

on a 0.5° land grid. Every headline number and figure in the manuscript is
regenerated from public data by `scripts/run_pipeline.py`.

## Layout

```
src/eh_shallow/     Python package (model logic)
  config.py         constants, grid, calibration windows, priors, RunConfig
  data.py           GMST (HadCRUT5), OHC (NOAA/NCEI), CO2/ERF, land mask
  emulator.py       two-layer Geoffroy EBM (exact matrix-exponential solver)
  smc.py            tempered Sequential Monte Carlo calibration of (ECS, γ)
  ocean.py          PyCO2SYS carbonate diagnostics (pH, Ω_arag)
  chs_haf.py        Composite Hazard Score field + Habitable Area Fraction
  whi.py            water-hazard baseline field + descriptive Random Forest
  metrics.py        out-of-sample skill + human-climate-niche check
  pipeline.py       end-to-end orchestration → metrics.json + results.npz
  figures.py        publication figures (Nature style)
scripts/            run driver, SLURM job, cluster env setup
tests/              regression tests (e.g. calibration baseline guard)
manuscript/         Nature-style main.tex + SI.tex + figures
environment.yml     pinned reproducible conda environment
```

## Quick start

```bash
conda env create -f environment.yml && conda activate eh
python scripts/run_pipeline.py --fast     # subsampled smoke test (~2-3 min)
python scripts/run_pipeline.py            # full headline run + figures
PYTHONPATH=src python -m pytest tests/ -q  # tests
```

Outputs are written to `outputs/` (git-ignored). Override paths via
`EH_DATA_DIR`, `EH_OUTPUT_DIR`, `EH_FIG_DIR`, `EH_PAPER_FIG_DIR`.

## Heavy runs (SLURM)

`scripts/cluster_setup.sh` builds the conda env; `scripts/slurm_run.sh` submits
the full-resolution run. See headers in those scripts.

## Reproducibility

All randomness flows through one seeded NumPy generator (`config.SEED`); a run is
bit-reproducible. The calibrated run gives ECS = 2.84 K (5–95%: 2.47–3.14),
out-of-sample GMST skill +0.49 vs persistence, and HAF declining from 0.84
(preindustrial) to 0.50 by 2100 under SSP2-4.5.

## Author

Masoud Rostami — Potsdam Institute for Climate Impact Research (PIK).
