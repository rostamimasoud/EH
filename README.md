# EH — Shallow-time Earth Habitability

A reduced-complexity, fully reproducible pipeline that projects terrestrial
habitability under coupled climate and ocean change, from the preindustrial
(1750) to 2300 CE, on a real 0.5° global land grid.

Pipeline:

```
real forcing/obs → two-box energy-balance emulator → tempered SMC calibration
→ six carried climate–ocean variables → tiered Composite Hazard Score (CHS)
→ Habitable Area Fraction (HAF)
```

The tier-(ii) baseline hazard field is a **real gridded Water Hazard Index (WHI)**
resampled to the model grid (released, redistributable derivative at
`eh_shallow/released/whi_field_0p5deg.npz`), so the CHS map carries genuine
emergent geography (subtropical dry belts, continental interiors).

## Layout

```
eh_shallow/          the package
  data.py            HadCRUT5, AR6 ERF, NOAA/NCEI OHC, CO2 pathway (download+cache)
  emulator.py        two-box EBM (FaIR core if available, else forward-Euler) + chemistry
  smc.py             tempered Sequential Monte Carlo calibration of (ECS, γ)
  grid.py            real 0.5° grid, land mask, warming pattern, baseline field, HAF table
  chs.py             Composite Hazard Score + Habitable Area Fraction
  whi.py             real gridded WHI baseline + descriptive Random Forest
  niche.py, cropyield.py, structural.py, stationarity.py   validation modules
  plots.py           publication figures
  run.py             end-to-end driver (python -m eh_shallow.run)
  released/          redistributable WHI field, RF importances, land mask
scripts/             run driver, SLURM job, cluster env setup
tests/               shallow-time test suite
manuscript/          main.tex + SI.tex + references.bib + figures
environment.yml      pinned reproducible conda environment
```

## Quick start

```bash
conda env create -f environment.yml && conda activate eh
python -m eh_shallow.run --n-particles 400 --baseline auto --outdir outputs
PYTHONPATH=. python -m pytest tests/ -q
```

The land mask and WHI field ship with the code, so a run needs network only to
fetch the observational series (HadCRUT5, AR6 ERF, NOAA/NCEI OHC), which are then
cached under `eh_shallow/_cache/`.

## Heavy runs (SLURM, PIK HPC)

`scripts/cluster_setup.sh` builds the conda env; `scripts/slurm_run.sh` seeds the
land-mask cache and submits the full run. Compute nodes are offline, so the
observational series must be pre-cached on a login node first (the setup script
does this).

## Reproducibility

Deterministic given `--seed`. The calibrated run (seed 0, 400 particles, two-box
core) gives ECS ≈ 3.69 K (5–95%: 3.03–4.31), out-of-sample GMST skill +0.43 vs
persistence, and HAF declining from 0.90 (preindustrial) to 0.63 (2100, SSP2-4.5)
and 0.21 (2100, SSP5-8.5).

## Author

Masoud Rostami — Potsdam Institute for Climate Impact Research (PIK).
