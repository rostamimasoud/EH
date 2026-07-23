# `eh_shallow` — prototype vertical slice of the shallow (1750–2300) model

A minimal, **runnable** end-to-end implementation of the pipeline in
`../EH_shallow.tex`, built to prove the architecture on real data before the
full 33-variable / 0.5° system is constructed. It is deliberately a *reduced*
model (the "fast prior-exploration" form the proposal describes), with the
production components named at each seam.

```
real forcing (AR6 ERF) + CO2  ─▶  FaIR 2-layer energy-balance climate core
   ─▶  6 multi-sphere variables  ─▶  tiered disaggregation onto a real 0.5° grid
   ─▶  Composite Hazard Score (CHS) per cell  ─▶  Habitable Area Fraction (HAF)
   ─▶  tempered-SMC calibration vs HadCRUT5 GMST + NOAA/NCEI ocean heat content
```

## Run

```bash
pip install -r eh_shallow/requirements.txt        # already satisfied in the project env
python -m eh_shallow.run --n-particles 500 --n-temps 12 --outdir eh_shallow/out
```

Outputs (in `--outdir`): `haf.pdf` (headline Habitable-Area-Fraction trajectory),
`haf_scenarios.pdf` (HAF under all four SSPs), `chs_map_2100.pdf` (2-D hazard map),
`gmst_fit.pdf` (posterior GMST fan vs HadCRUT5, calibration vs withheld),
`ohc_fit.pdf` (posterior OHC fan vs NOAA/NCEI 0–2000 m), `posterior.pdf` (prior
vs posterior for ECS, γ), and `metrics.json`. Deterministic for a given `--seed`.
First run downloads ~110 kB of public data into `eh_shallow/_cache/`.

## What is real vs reduced

| Stage | This prototype | Production (per the proposal) |
|---|---|---|
| Forcing | **real** IPCC AR6 ERF 1750–2019 + all four SSP extensions (1-2.6 / 2-4.5 / 3-7.0 / 5-8.5) | RCMIP/CMIP6 emissions through FaIR |
| Climate core | **FaIR** `EnergyBalanceModel` (2-layer Geoffroy 2013, exact matrix-exponential) when `fair` is installed; in-house forward-Euler fallback otherwise | 3-layer AR6-calibrated FaIR + emissions-driven FaIR carbon cycle |
| Ocean chemistry | **PyCO2SYS** full carbonate system (surface equilibrium with atmospheric CO₂ at fixed alkalinity, SST-coupled) when installed; linear pH/Ω fallback | full DIC/alkalinity budget with air–sea disequilibrium |
| Calibration target | **real** HadCRUT5 GMST + **real** NOAA/NCEI 0–2000 m OHC (jointly) | + IAP/Argo OHC, Argo profiles, satellite altimetry |
| Inference | self-contained tempered SMC (numpy): ESS resampling, MALA-like rejuvenation, Student-t(ν=4), σ_struct | blackjax/numpyro SMC on JAX, vectorised over particles |
| Spatial field | **real 0.5° grid** (Natural Earth land mask via `regionmask`, cos-lat areas) with tiered disaggregation | + tier-(ii) gridded local forcings (WHI/withdrawals, SSP land-use) |
| Variables in CHS | 6 (GMST, CO₂, SST, OHC, pH, Ω) | 33, with held-out RF/WHI weights |

## Climate core: FaIR

The two-layer Geoffroy (2013) EBM is integrated by FaIR's
`EnergyBalanceModel` (exact matrix-exponential solver + validated TOA-imbalance
and OHC bookkeeping) whenever the `fair` package is importable. It is the *same*
physical model as the in-house 2-box (identical `C_S`, `C_D`, and ECS/γ meaning):
FaIR's emergent ECS = `forcing_4co2 * 0.5 / ocean_heat_transfer[0]`, so setting
`forcing_4co2 = 2·F2X` and `ocean_heat_transfer = [F2X/ECS, γ]` reproduces it
exactly, only without forward-Euler discretisation error (≈0.03 K at 2020, up to
≈0.27 K in the fast transient). At ECS=3, γ=0.7 FaIR gives a 2020 GMST of 1.13 K
vs the observed 1.12 K. If `fair` is absent the code falls back to forward Euler
so the prototype still runs offline. `metrics.json` records which core ran.
Next: a 3-layer AR6-calibrated EBM and the emissions-driven FaIR carbon cycle.

## Ocean chemistry: PyCO2SYS

Surface pH and aragonite saturation (Ω_arag) are solved by **PyCO2SYS** (the full
nonlinear carbonate system) whenever the package is importable: the surface mixed
layer is taken in equilibrium with atmospheric CO₂ (`par1 = pCO2`) at fixed total
alkalinity (2300 µmol/kg, S = 35), with temperature = 18 °C + the FaIR-driven SST
anomaly — so warming couples into the carbonate equilibria. This replaces the
linear `pH = 8.2 − 0.0011·ΔCO₂` form, capturing Revelle buffering (the pH drop per
ppm shrinks as CO₂ rises) and the temperature dependence (warming raises Ω_arag at
fixed pCO₂). pH/Ω do **not** enter the SMC likelihood, so the sampler runs the
cheap linear form (`chem=False`, ~50 ms/solve × thousands of particles avoided)
and only the projection/CHS path calls PyCO2SYS. `metrics.json` records the active
chemistry model.

## Spatial grid: real 0.5° + tiered disaggregation

The CHS/HAF run on a **real 0.5° global grid** (`grid.py`): a Natural Earth land
mask (via `regionmask`, cached to `_cache/`; reproduces Earth's ~29% land area
fraction), `cos(lat)` cell areas, and the proposal's **tiered** disaggregation:

- **tier (i)** — surface temperature → pattern-scaled onto a present-day land
  warming pattern (polar + land/ocean amplification);
- **tier (iii)** — variables with no defensible land field (CO₂, and the ocean
  quantities SST/OHC/pH/Ω) → spatially-uniform modifiers, so they shift the CHS
  everywhere equally and never manufacture artificial hotspots;
- **tier (ii)** — the locally-forced vars (groundwater, land use) that alone may
  claim *emergent* hotspots — needs the gridded WHI stack and is **not yet wired
  in**; until then the present-day baseline vulnerability field `B` is a documented
  stand-in (latitudinal structure + the `σ_agg` aggregation/Jensen heterogeneity).

The per-cell hazard separates as `CHS(cell,t) = P_temp(cell)·sT(t) + B(cell) +
U(t)`. Because the preindustrial drivers are ~0, the HAF reference level τ is a
fixed percentile of `B`, so `HAF(sT, U)` is a fixed 2-D function — precomputed
once as a lookup table, making the whole 200-draw posterior ensemble (≈86k land
cells × 551 yr) cost ~1 s instead of materialising a field per draw. This also
honestly exposes what the *old* latitude-band stand-in hid: pattern-scaling the
full CHS (the Jensen fallacy) produced a smooth decline; the correct tiered
scheme leaves 5/6 variables uniform, so refugia persistence comes from the
spatial spread of `B` (real present-day habitability heterogeneity, `B_STD`).
`chs_map_2100.pdf` shows the resulting hazard map.

## Design choices that encode the review fixes

- **Calibration window is 1850–1980; 1981–2020 is withheld** (out-of-sample),
  matching the leakage fix in the proposal. `metrics.json` reports the
  1981–2020 RMSE.
- **σ_struct** (0.20 K) is added to the likelihood — without it the GMST-only
  fit over 1850–1980 over-fits ECS to ~5 K (the model lacks internal
  variability). With it, the posterior is AR6-consistent (~3.7 K).
- **OHC breaks the ECS–γ degeneracy.** GMST alone leaves the deep-ocean
  heat-uptake coefficient γ prior-dominated (posterior ≈ prior, width ~0.82).
  Adding NOAA/NCEI 0–2000 m ocean heat content as a second observable
  (2005–2020, on its own 2005–2014 baseline, σ_struct,OHC = 10 ZJ for the
  full-depth-vs-0–2000 m mismatch) narrows γ to ~0.36–0.79 (width 0.48) while
  keeping ECS AR6-consistent. OHC postdates the 1980 GMST cutoff, so it is an
  *additional* constraint window, not a relaxation of the leakage fix.
- **CHS standardisation** anchors the mean at the preindustrial baseline but
  scales by the full-period spread, because the preindustrial-only variance is
  ~0 for monotonic variables — a concrete demonstration of the `baseline-1`
  review finding (`chs.py:_standardise`).
- **HAF is reported across τ = 80th–99th percentile** (dotted band in `haf.pdf`)
  plus a smooth-exceedance alternative, per the "threshold-free → baseline-
  referenced" fix.

## Known limitations (next increments)

1. Wire **tier (ii)**: the gridded local forcings (water withdrawals, SSP
   land-use) and the **RF/WHI weights** (`weights.py`) once the WHI stack
   (`Rostami2025Aquifer`) is available, with the five WHI constituents held out.
   This replaces the `B` stand-in and enables genuine *emergent* hotspots/refugia.
2. Move to a 3-layer AR6-calibrated FaIR EBM + the emissions-driven FaIR carbon
   cycle; add air–sea CO₂ disequilibrium to the chemistry.

*Done:* (a) OHC added as a second observable, breaking the ECS–γ degeneracy
(see `ohc_fit.pdf` / `posterior.pdf`); (b) FaIR adopted as the climate core
(2-layer EBM via the exact matrix-exponential solver); (c) PyCO2SYS adopted as
the ocean-chemistry core (full carbonate system, SST-coupled); (d) real 0.5°
grid + tiered disaggregation (`grid.py`, `chs_map_2100.pdf`); (e) all four SSP
scenarios + the scenario-spread term σ²_scenario (`haf_scenarios.pdf`). The CHS
is standardised against a common reference (SSP2-4.5 posterior mean) so HAF is
comparable across draws and scenarios — without it a high-emission scenario
self-standardises to its own variance and looks spuriously *more* habitable.

## Files

`data.py` (download/cache, forcing, CO₂, GMST, OHC) · `emulator.py` (`run_emulator`) ·
`chs.py` (CHS, HAF) · `smc.py` (tempered SMC) · `plots.py` (Nature-style PDFs) ·
`run.py` (CLI) · `../tests/test_eh_shallow.py` (smoke test).
