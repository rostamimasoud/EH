"""Composite Hazard Score (CHS) and Habitable Area Fraction (HAF).

Tiered spatial disaggregation (SI Section S18):

    CHS(cell, t) = (1/6) [ P(cell) z_gmst(t)
                           + z_co2(t) + z_sst(t) + z_ohc(t)
                           + z_pH(t) + z_Omega(t) ]
                   + B(cell)

  * Tier (i): surface-temperature-driven land pattern P(cell) z_gmst(t).
  * Tier (ii): observed water-hazard baseline field B(cell) (only tier carrying
    emergent spatial hotspots).
  * Tier (iii): ocean / global-mean variables enter as spatially uniform terms.

Each of the six carried variables is standardised against the preindustrial
baseline mean, with the *scale* taken over the full analysis period (the
preindustrial variance of the anthropogenic variables is degenerate), and
oriented so larger means more hazardous (pH and aragonite saturation are
sign-flipped). The HAF is the area-weighted land fraction where CHS < tau, with
tau the 90th percentile of the preindustrial CHS distribution.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .whi import baseline_field, land_warming_pattern

# Orientation: +1 if larger = more hazardous, -1 if larger = less hazardous.
_ORIENT = {"gmst": +1, "co2": +1, "sst": +1, "ohc": +1,
           "ph": -1, "omega_arag": -1}


def standardise(series: dict, years: np.ndarray) -> dict:
    """Standardise each carried variable to an oriented z-score time series.

    Mean taken over the preindustrial window; scale over the full period.
    """
    pi = (years >= C.PREINDUSTRIAL[0]) & (years <= C.PREINDUSTRIAL[1])
    z = {}
    for name in C.CHS_VARIABLES:
        v = np.asarray(series[name], dtype=float)
        mu = v[pi].mean()
        sd = v.std() + 1e-12
        z[name] = _ORIENT[name] * (v - mu) / sd
    return z


def chs_field(grid: dict, zt: dict, t_index: int, B: np.ndarray,
              P: np.ndarray) -> np.ndarray:
    """Composite Hazard Score field at a single time index."""
    clim = (P * zt["gmst"][t_index]
            + zt["co2"][t_index] + zt["sst"][t_index] + zt["ohc"][t_index]
            + zt["ph"][t_index] + zt["omega_arag"][t_index]) / 6.0
    return clim + B


def _area_mean_free_chs(grid, zt, t_index):
    """Spatially-uniform (area-mean) part of CHS: s_T(t) + U(t)."""
    z = zt
    return (z["gmst"][t_index] + z["co2"][t_index] + z["sst"][t_index]
            + z["ohc"][t_index] + z["ph"][t_index]
            + z["omega_arag"][t_index]) / 6.0


def preindustrial_chs_distribution(grid, zt, years, B, P):
    """Pool CHS over land cells across the preindustrial window."""
    pi_idx = np.where((years >= C.PREINDUSTRIAL[0])
                      & (years <= C.PREINDUSTRIAL[1]))[0]
    mask = grid["mask"]
    vals = []
    for ti in pi_idx:
        f = chs_field(grid, zt, ti, B, P)
        vals.append(f[mask])
    return np.concatenate(vals)


def haf_trajectory(grid, zt, years, tau, B=None, P=None):
    """HAF(t): area-weighted land fraction with CHS < tau."""
    if B is None:
        B = baseline_field(grid)
    if P is None:
        P = land_warming_pattern(grid)
    mask = grid["mask"]
    area = grid["area"]
    w = area[mask]
    wtot = w.sum()
    haf = np.empty(years.size)
    for i in range(years.size):
        f = chs_field(grid, zt, i, B, P)[mask]
        haf[i] = np.sum(w * (f < tau)) / wtot
    return haf


def land_arrays(grid, B=None, P=None):
    """Flattened land-cell arrays (P, B, area) for the fast HAF path."""
    if B is None:
        B = baseline_field(grid)
    if P is None:
        P = land_warming_pattern(grid)
    mask = grid["mask"]
    return P[mask], B[mask], grid["area"][mask]


def haf_fast(Pland, Bland, wland, zt, tau):
    """Vectorised HAF(t) over land cells.

    CHS(cell,t) = P(cell) a(t) + c(t) + B(cell), with
      a(t) = z_gmst(t)/6 ,
      c(t) = [z_co2 + z_sst + z_ohc + z_pH + z_Omega](t)/6 .
    HAF(t) = weighted fraction of land where P a(t) + B(cell) < tau - c(t).
    """
    a = zt["gmst"] / 6.0
    c = (zt["co2"] + zt["sst"] + zt["ohc"] + zt["ph"]
         + zt["omega_arag"]) / 6.0
    wtot = wland.sum()
    haf = np.empty(a.size)
    for t in range(a.size):
        cond = (Pland * a[t] + Bland) < (tau - c[t])
        haf[t] = np.dot(wland, cond) / wtot
    return haf


def compute_tau(grid, zt, years, percentile, B=None, P=None):
    if B is None:
        B = baseline_field(grid)
    if P is None:
        P = land_warming_pattern(grid)
    dist = preindustrial_chs_distribution(grid, zt, years, B, P)
    return float(np.percentile(dist, percentile))


def haf_percentile_band(grid, zt, years, B=None, P=None,
                        band=C.TAU_PERCENTILE_BAND):
    """HAF trajectories at the low/high ends of the reference-percentile band."""
    if B is None:
        B = baseline_field(grid)
    if P is None:
        P = land_warming_pattern(grid)
    out = {}
    for p in band:
        tau = compute_tau(grid, zt, years, p, B, P)
        out[p] = haf_trajectory(grid, zt, years, tau, B, P)
    return out
