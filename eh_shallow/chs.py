"""Composite Hazard Score (CHS) and Habitable Area Fraction (HAF).

Implements the hardened metric from EH_shallow.tex on a REAL 0.5-degree grid
(see `grid.py`) with the proposal's *tiered* disaggregation:

  - each variable is *oriented* so larger = more hazard, then standardised using
    the PREINDUSTRIAL baseline (1750-1800) mean (full-period scale -- see the
    `_standardise` note);
  - tier (i): surface-temperature-driven variables are pattern-scaled onto the
    present-day land warming pattern; tier (iii): variables with no defensible
    land field are spatially-uniform modifiers (tier (ii), the locally-forced
    vars, needs the gridded WHI stack and is not yet wired in);
  - CHS(cell, t) = P_temp(cell)*sT(t) + B(cell) + U(t), with sT the tier-(i)
    driver, U the tier-(iii) uniform sum, and B the present-day baseline field;
  - HAF = area-weighted fraction of LAND with CHS below a reference level tau
    (90th percentile of the preindustrial CHS distribution ~ percentile of B).
    We also report a SMOOTH exceedance and the sensitivity to the percentile,
    per the proposal's fix to the "threshold-free" overclaim.
"""
from __future__ import annotations

import numpy as np

from . import data, grid

# CHS variables and their hazard orientation (+1: larger is worse; -1: smaller is worse)
CHS_VARS = {
    "gmst": +1, "co2": +1, "sst": +1, "ohc": +1, "ph": -1, "omega": -1,
}
# Tier assignment for disaggregation onto the grid (EH_shallow.tex):
#   tier (i): a defensible LAND spatial pattern (surface air temperature);
#   tier (iii): no defensible land field -> spatially-uniform global modifier.
# (sst/ohc/ph/omega are ocean quantities; on the land HAF they act as uniform
#  modifiers. tier (ii) -- groundwater/land-use, the only tier that may claim
#  emergent hotspots -- arrives with the WHI stack.)
TIER_I = ("gmst",)
TIER_III = ("co2", "sst", "ohc", "ph", "omega")


def _weights(weights: dict | None) -> dict:
    names = list(CHS_VARS)
    if weights is None:
        return {k: 1.0 / len(names) for k in names}
    s = sum(weights.values())
    return {k: weights[k] / s for k in names}


def _standardise(series: np.ndarray, years: np.ndarray, sign: int,
                 scale: tuple | None = None) -> np.ndarray:
    """Orient by `sign`, anchor the mean at the preindustrial baseline, and scale.

    NOTE (demonstrates the `baseline-1` review finding): the preindustrial window
    is near-constant for monotonic variables, so its *variance* is ~0 and using it
    as the z-score scale makes the score explode. We therefore anchor the MEAN at
    the preindustrial baseline (so anomalies are measured against it) but take the
    SCALE from the full analysis period -- a pragmatic, documented choice. A
    production metric would standardise per-variable against a physically
    meaningful range, not a degenerate preindustrial variance.

    `scale` = (mu, sd) overrides the per-series constants with a COMMON reference
    (see `reference_scales`), so the CHS is comparable across draws and scenarios
    -- without it, a high-emission scenario self-standardises to its own large
    variance and would spuriously look *more* habitable than a low one.
    """
    oriented = sign * series
    if scale is not None:
        mu, sd = scale
        return (oriented - mu) / sd
    m = (years >= data.PREINDUSTRIAL[0]) & (years <= data.PREINDUSTRIAL[1])
    mu = oriented[m].mean()
    sd = oriented.std()  # full-period scale (preindustrial-only sd is ~0)
    if sd == 0 or not np.isfinite(sd):
        sd = 1.0
    return (oriented - mu) / sd


def reference_scales(out_ref: dict) -> dict:
    """Per-variable (mu, sd) standardisation constants from a reference run.

    Fixing the scale to one reference scenario/draw makes the CHS a well-defined
    metric that is comparable across the posterior ensemble and across SSPs.
    """
    years = out_ref["year"]
    sc = {}
    for k, sign in CHS_VARS.items():
        oriented = sign * out_ref[k]
        m = (years >= data.PREINDUSTRIAL[0]) & (years <= data.PREINDUSTRIAL[1])
        sd = oriented.std()
        sc[k] = (float(oriented[m].mean()),
                 float(sd) if (sd and np.isfinite(sd)) else 1.0)
    return sc


def composite_hazard(out: dict, weights: dict | None = None,
                     scales: dict | None = None) -> np.ndarray:
    """Global (area-mean) CHS(t) from an emulator output dict.

    Equals sT(t) + U(t): the area-mean of the per-cell field, since P_temp has
    global mean 1 and B is centred over land.
    """
    sT, U = tier_series(out, weights, scales)
    return sT + U


def tier_series(out: dict, weights: dict | None = None, scales: dict | None = None):
    """Split the standardised CHS into the tier-(i) driver sT(t) and uniform U(t).

    sT(t) = sum_{tier i} w_m * vtilde_m(t)  (weight folded in, so the tier-(i)
    field P_temp(cell)*sT(t) has area-mean sT(t)); U(t) = sum_{tier iii} w_m * vtilde_m.
    Pass `scales` (from `reference_scales`) for a common cross-scenario scale.
    """
    years = out["year"]
    w = _weights(weights)
    sc = scales or {}
    sT = np.zeros(len(years))
    for k in TIER_I:
        sT += w[k] * _standardise(out[k], years, CHS_VARS[k], sc.get(k))
    U = np.zeros(len(years))
    for k in TIER_III:
        U += w[k] * _standardise(out[k], years, CHS_VARS[k], sc.get(k))
    return sT, U


def haf_ensemble(sT_all: np.ndarray, U_all: np.ndarray, years: np.ndarray,
                 percentile: float = 90.0):
    """HAF(t) for a whole ensemble of draws via the precomputed grid lookup.

    sT_all, U_all: [n_draw, n_time]. Returns HAF [n_draw, n_time]. The HAF(sT, c)
    table is built once (covering the ensemble's range) and queried for all draws.
    """
    G = grid.build()
    tau = grid.tau_of(percentile, G)
    sT_all = np.atleast_2d(sT_all)
    U_all = np.atleast_2d(U_all)
    c = tau - U_all
    tab = grid.make_table(float(sT_all.max()) * 1.15 + 1e-3,
                          float(c.min()) - 0.5, float(c.max()) + 0.5, G)
    return tab(sT_all, c)


def haf_trajectory(out: dict, percentile: float = 90.0, weights: dict | None = None,
                   scales: dict | None = None):
    """Single-draw HAF on the real grid: hard indicator + smooth exceedance + tau."""
    years = out["year"]
    sT, U = tier_series(out, weights, scales)
    G = grid.build()
    tau = grid.tau_of(percentile, G)
    haf = haf_ensemble(sT[None, :], U[None, :], years, percentile)[0]
    # smooth exceedance for this draw: per-cell logistic, area-weighted over land
    # (float32 to bound the transient [n_land x n_time] field)
    P = G["P"].astype(np.float32)
    B = G["B"].astype(np.float32)
    area = G["area"]
    field = P[:, None] * sT[None, :].astype(np.float32) + B[:, None] + U[None, :].astype(np.float32)
    scale = max(float(G["B"].std()), 1e-6)
    smooth = (area[:, None] * _sigmoid((tau - field) / scale)).sum(0) / area.sum()
    return {"haf": haf, "haf_smooth": smooth, "tau": tau,
            "land_area_frac": G["land_area_frac"]}


def haf_percentile_sensitivity(out: dict, percentiles=(80, 85, 90, 95, 99),
                               weights: dict | None = None, scales: dict | None = None):
    """HAF(t) for several tau-percentiles (the proposal's sensitivity report)."""
    sT, U = tier_series(out, weights, scales)
    years = out["year"]
    return {p: haf_ensemble(sT[None, :], U[None, :], years, percentile=p)[0]
            for p in percentiles}


def chs_field(out: dict, year: int, weights: dict | None = None,
              scales: dict | None = None):
    """2-D CHS field [nlat, nlon] (ocean masked NaN) at `year`, for mapping."""
    sT, U = tier_series(out, weights, scales)
    years = out["year"]
    i = int(np.argmin(np.abs(years - year)))
    return grid.field_at(float(sT[i]), float(U[i]))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))
