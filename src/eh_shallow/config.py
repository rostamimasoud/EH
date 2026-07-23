"""Global configuration for the shallow-time Earth Habitability (EH) proof-of-concept.

All physical constants, grid definitions, calibration windows, dataset URLs, and
run parameters live here so that a single import fixes the reproducible state of
the pipeline. Values follow the executed proof-of-concept documented in the
perspective Supplementary Information (Sections S13--S19).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repository root = two levels above this file (src/eh_shallow/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]

# Data cache and run outputs live inside the repo but are git-ignored.
DATA_DIR = Path(os.environ.get("EH_DATA_DIR", REPO_ROOT / "data"))
OUTPUT_DIR = Path(os.environ.get("EH_OUTPUT_DIR", REPO_ROOT / "outputs"))
FIG_DIR = Path(os.environ.get("EH_FIG_DIR", OUTPUT_DIR / "figures"))

# The manuscript figures directory (Nature paper). The pipeline mirrors the
# publication figures here so LaTeX picks them up directly. Override with
# EH_PAPER_FIG_DIR for a different manuscript location.
PAPER_FIG_DIR = Path(
    os.environ.get(
        "EH_PAPER_FIG_DIR",
        REPO_ROOT.parent / "EH_shallow" / "figures",
    )
)

for _d in (DATA_DIR, OUTPUT_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 0

# --------------------------------------------------------------------------- #
# Time axis (annual, 1750--2300 CE)
# --------------------------------------------------------------------------- #
YEAR_START = 1750
YEAR_END = 2300
PREINDUSTRIAL = (1750, 1800)      # baseline window for standardisation and tau
CALIB_GMST = (1850, 1980)         # GMST calibration window (post-1980 withheld)
VALID_GMST = (1981, 2020)         # out-of-sample GMST validation window
CALIB_OHC = (2005, 2020)          # OHC constraint window
OHC_REF = (2005, 2014)            # OHC reference period
IPCC_BASELINE = (1850, 1900)      # warming reference for Delta T

# --------------------------------------------------------------------------- #
# Grid (0.5 degree global)
# --------------------------------------------------------------------------- #
GRID_RES = 0.5
N_LON = int(360 / GRID_RES)       # 720
N_LAT = int(180 / GRID_RES)       # 360

# --------------------------------------------------------------------------- #
# Two-layer (Geoffroy) energy-balance model constants
# Geoffroy et al. (2013), J. Climate; standard FaIR-class values.
# Heat capacities in W yr m^-2 K^-1.
# --------------------------------------------------------------------------- #
C_SURF = 7.3        # surface / mixed-layer heat capacity
C_DEEP = 106.0      # deep-ocean heat capacity
F_2X = 3.93         # ERF of CO2 doubling (W m^-2), AR6
F_OCEAN = 0.71      # ocean fraction of net TOA imbalance stored as OHC

# --------------------------------------------------------------------------- #
# Ocean carbonate chemistry (PyCO2SYS diagnostics)
# --------------------------------------------------------------------------- #
ALKALINITY = 2300.0     # total alkalinity (umol/kg), held fixed
SST_PI_MEAN = 18.0      # preindustrial open-marine SST reference (deg C)
CO2_PI = 278.0          # preindustrial atmospheric CO2 (ppm)

# --------------------------------------------------------------------------- #
# SSP scenarios: label -> nominal 2100 total ERF (W m^-2) used by the reduced
# forcing extension when the full RCMIP/AR6 SSP ERF series is unavailable.
# --------------------------------------------------------------------------- #
SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")
SSP_PRETTY = {
    "ssp126": "SSP1-2.6",
    "ssp245": "SSP2-4.5",
    "ssp370": "SSP3-7.0",
    "ssp585": "SSP5-8.5",
}
SSP_ERF_2100 = {"ssp126": 2.6, "ssp245": 4.5, "ssp370": 7.0, "ssp585": 8.5}
HEADLINE_SSP = "ssp245"

# --------------------------------------------------------------------------- #
# SMC calibration
# --------------------------------------------------------------------------- #
N_PARTICLES = 400
N_TEMP_STEPS = 12           # inverse-temperature ladder
ESS_FRACTION = 0.5          # resample when ESS < N/2
STUDENT_T_NU = 4            # robust likelihood degrees of freedom
SIGMA_STRUCT_GMST = 0.20    # structural uncertainty on GMST (K)
SIGMA_OHC = 10.0            # obs+structural uncertainty on OHC (ZJ)

# Priors (executed emulator): only ECS and gamma are calibrated.
ECS_PRIOR = ("normal", 3.0, 0.5)        # N(3.0, 0.5) K
GAMMA_PRIOR = ("uniform", 0.3, 1.2)     # U(0.3, 1.2) W m^-2 K^-1

# --------------------------------------------------------------------------- #
# CHS / HAF
# --------------------------------------------------------------------------- #
# Six carried climate--ocean variables, equal weights (w_m = 1/6).
CHS_VARIABLES = ("gmst", "co2", "sst", "ohc", "ph", "omega_arag")
TAU_PERCENTILE = 90.0                    # baseline reference percentile for tau
TAU_PERCENTILE_BAND = (80.0, 99.0)       # reported sensitivity band


@dataclass
class RunConfig:
    """Runtime knobs; defaults reproduce the headline proof-of-concept."""
    seed: int = SEED
    n_particles: int = N_PARTICLES
    n_temp_steps: int = N_TEMP_STEPS
    grid_res: float = GRID_RES
    ssps: tuple = SSPS
    headline_ssp: str = HEADLINE_SSP
    tau_percentile: float = TAU_PERCENTILE
    write_paper_figs: bool = True
    fast: bool = False        # if True, subsample grid/particles for a smoke test

    def __post_init__(self):
        if self.fast:
            self.n_particles = min(self.n_particles, 120)
