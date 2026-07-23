"""Validation metrics: out-of-sample skill and the human-climate-niche check."""
from __future__ import annotations

import numpy as np

from . import config as C


def rmse(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.sqrt(np.nanmean((a - b) ** 2)))


def skill_score(rmse_model, rmse_ref):
    """Positive => model beats the reference baseline."""
    return float(1.0 - rmse_model / rmse_ref)


def persistence_forecast(years, vals, split_year):
    """Persistence baseline: hold the last calibration value flat."""
    last = vals[years <= split_year][-1]
    return np.full(np.sum(years > split_year), last)


def linear_trend_forecast(years, vals, calib):
    """Extrapolate the linear trend fitted on the calibration window."""
    m = (years >= calib[0]) & (years <= calib[1])
    p = np.polyfit(years[m], vals[m], 1)
    return np.polyval(p, years[years > calib[1]])


def gmst_out_of_sample(sim_years, sim_gmst, obs_years, obs_gmst):
    """RMSE and skill of the emulator over the withheld 1981-2020 window."""
    # Reference the simulated GMST to the observations' 1850-1900 baseline so the
    # comparison is on the same anomaly footing (see smc._log_likelihood_single).
    base = (sim_years >= C.IPCC_BASELINE[0]) & (sim_years <= C.IPCC_BASELINE[1])
    sim_gmst = np.asarray(sim_gmst) - np.asarray(sim_gmst)[base].mean()
    lo, hi = C.VALID_GMST
    m = (obs_years >= lo) & (obs_years <= hi)
    oy, ov = obs_years[m], obs_gmst[m]
    idx = np.searchsorted(sim_years, oy)
    model = sim_gmst[idx]
    r_model = rmse(model, ov)

    pers = persistence_forecast(obs_years, obs_gmst, C.CALIB_GMST[1])
    pers = pers[: ov.size]
    lin = linear_trend_forecast(obs_years, obs_gmst, C.CALIB_GMST)
    lin = lin[: ov.size]
    return {
        "rmse_model": r_model,
        "rmse_persistence": rmse(pers, ov),
        "rmse_lineartrend": rmse(lin, ov),
        "skill_vs_persistence": skill_score(r_model, rmse(pers, ov)),
        "skill_vs_lineartrend": skill_score(r_model, rmse(lin, ov)),
    }


def near_unlivable_fraction(grid, land_temp_2d):
    """Area-weighted land fraction with mean annual T > 29 deg C (Xu et al.)."""
    mask = grid["mask"]
    w = grid["area"][mask]
    hot = land_temp_2d[mask] > 29.0
    return float(np.sum(w * hot) / np.sum(w))


def niche_habitable_fraction(mean_annual_temp_land, area_w):
    """Complement of the near-unlivable fraction (temperature-only niche)."""
    hot = mean_annual_temp_land > 29.0
    return float(1.0 - np.sum(area_w * hot) / np.sum(area_w))
