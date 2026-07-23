"""End-to-end shallow-time proof-of-concept pipeline.

Chain: real forcing/obs -> two-layer emulator -> tempered SMC calibration ->
six carried climate-ocean variables -> tiered CHS -> HAF (per SSP, with a
posterior envelope and a reference-percentile band) -> validation. Writes
``metrics.json`` (all headline numbers) and ``results.npz`` (arrays for the
figure module) into ``config.OUTPUT_DIR``.
"""
from __future__ import annotations

import json
import time

import numpy as np

from . import config as C
from . import data, metrics
from .chs_haf import (compute_tau, haf_fast, land_arrays)
from .emulator import run_emulator, sst_from_gmst
from .ocean import carbonate_diagnostics
from .smc import posterior_summary, run_smc
from .whi import baseline_field, land_warming_pattern, rf_whi_importances

_ORIENT = {"gmst": +1, "co2": +1, "sst": +1, "ohc": +1, "ph": -1,
           "omega_arag": -1}


def _series(sim, co2, years):
    """Assemble the six carried variables on the full time axis."""
    sst = sst_from_gmst(sim["T"])
    chem = carbonate_diagnostics(co2, sst)
    return {"gmst": sim["T"], "co2": co2, "sst": sst, "ohc": sim["OHC"],
            "ph": chem["ph"], "omega_arag": chem["omega_arag"]}


def _scales(series, years):
    """Common standardisation: PI mean and full-period scale per variable."""
    pi = (years >= C.PREINDUSTRIAL[0]) & (years <= C.PREINDUSTRIAL[1])
    return {n: (series[n][pi].mean(), series[n].std() + 1e-12)
            for n in C.CHS_VARIABLES}


def _standardise(series, scales):
    return {n: _ORIENT[n] * (series[n] - scales[n][0]) / scales[n][1]
            for n in C.CHS_VARIABLES}


def run(cfg: C.RunConfig | None = None) -> dict:
    cfg = cfg or C.RunConfig()
    t0 = time.time()
    rng = np.random.default_rng(cfg.seed)
    years = np.arange(C.YEAR_START, C.YEAR_END + 1)

    # --- observations & grid -------------------------------------------------
    gmst_obs = data.load_gmst_obs()
    ohc_obs = data.load_ohc_obs()
    grid = data.load_land_mask()
    obs = {"gmst_years": gmst_obs["years"], "gmst_vals": gmst_obs["gmst"],
           "ohc_years": ohc_obs["years"], "ohc_vals": ohc_obs["ohc"]}

    # --- headline forcing/CO2 & SMC calibration ------------------------------
    hforce = data.load_forcing(cfg.headline_ssp)
    hco2 = data.load_co2_pathway(cfg.headline_ssp)
    smc = run_smc(hforce["years"], hforce["erf"], obs, cfg)
    post = posterior_summary(smc)

    # resample posterior to an equally-weighted ensemble for fans
    n_env = min(200, cfg.n_particles)
    idx = rng.choice(cfg.n_particles, size=n_env, p=smc["weights"])
    ens = smc["theta"][idx]
    ecs_mean = np.average(smc["theta"][:, 0], weights=smc["weights"])
    gam_mean = np.average(smc["theta"][:, 1], weights=smc["weights"])

    # --- headline posterior-mean run + common standardisation ----------------
    sim_mean = run_emulator(years, hforce["erf"], ecs_mean, gam_mean)
    series_mean = _series(sim_mean, hco2["co2"], years)
    scales = _scales(series_mean, years)
    Pl, Bl, wl = land_arrays(grid)
    B = baseline_field(grid)
    P = land_warming_pattern(grid)

    # tau from the preindustrial CHS distribution (headline standardisation)
    zt_mean = _standardise(series_mean, scales)
    tau = compute_tau(grid, zt_mean, years, cfg.tau_percentile, B, P)

    # --- GMST / OHC posterior fans (headline) --------------------------------
    T_ens = np.empty((n_env, years.size))
    OHC_ens = np.empty((n_env, years.size))
    HAF_ens = np.empty((n_env, years.size))
    for k, (ecs, gam) in enumerate(ens):
        sim = run_emulator(years, hforce["erf"], ecs, gam)
        ser = _series(sim, hco2["co2"], years)
        zt = _standardise(ser, scales)
        T_ens[k] = sim["T"]
        OHC_ens[k] = sim["OHC"] - sim["OHC"][
            (years >= C.OHC_REF[0]) & (years <= C.OHC_REF[1])].mean()
        HAF_ens[k] = haf_fast(Pl, Bl, wl, zt, tau)

    # --- HAF per SSP (posterior mean) ----------------------------------------
    haf_ssp = {}
    dT2100 = {}
    for ssp in cfg.ssps:
        f = data.load_forcing(ssp)
        cc = data.load_co2_pathway(ssp)
        sim = run_emulator(years, f["erf"], ecs_mean, gam_mean)
        ser = _series(sim, cc["co2"], years)
        zt = _standardise(ser, scales)   # common standardisation
        haf_ssp[ssp] = haf_fast(Pl, Bl, wl, zt, tau)
        base = (years >= C.IPCC_BASELINE[0]) & (years <= C.IPCC_BASELINE[1])
        i2100 = np.where(years == 2100)[0][0]
        dT2100[ssp] = float(sim["T"][i2100] - sim["T"][base].mean())

    # --- reference-percentile band (headline) --------------------------------
    haf_band = {}
    for p in C.TAU_PERCENTILE_BAND:
        tp = compute_tau(grid, zt_mean, years, p, B, P)
        haf_band[p] = haf_fast(Pl, Bl, wl, zt_mean, tp)

    # --- weight-ensemble width at 2100 (headline) ----------------------------
    i2100 = np.where(years == 2100)[0][0]
    haf_w = _weight_ensemble(Pl, Bl, wl, series_mean, scales, years, tau,
                             grid, B, P, rng, n=500)
    weight_width_2100 = float(np.percentile(haf_w[:, i2100], 95)
                              - np.percentile(haf_w[:, i2100], 5))

    # --- CHS map at 2100 (headline posterior mean) ---------------------------
    from .chs_haf import chs_field
    chs2100 = chs_field(grid, zt_mean, i2100, B, P)
    chs2100 = np.where(grid["mask"], chs2100, np.nan)

    # --- validation ----------------------------------------------------------
    oos = metrics.gmst_out_of_sample(years, sim_mean["T"],
                                     gmst_obs["years"], gmst_obs["gmst"])
    niche = _niche_check(grid, years, cfg, ecs_mean, gam_mean, scales)
    rf = rf_whi_importances(grid, cfg.seed)

    elapsed = time.time() - t0

    # --- assemble results ----------------------------------------------------
    results = {
        "years": years, "grid": grid,
        "gmst_obs": gmst_obs, "ohc_obs": ohc_obs,
        "T_ens": T_ens, "OHC_ens": OHC_ens, "HAF_ens": HAF_ens,
        "haf_ssp": haf_ssp, "haf_band": haf_band,
        "chs2100": chs2100, "smc": smc, "post": post,
        "rf": rf, "haf_weight_ens": haf_w, "tau": tau,
        "scales": scales,
    }
    mjson = {
        "seed": cfg.seed, "n_particles": cfg.n_particles,
        "runtime_seconds": round(elapsed, 1),
        "ECS": {q: round(v, 3) for q, v in post["ECS"].items()},
        "gamma": {q: round(v, 3) for q, v in post["gamma"].items()},
        "gmst_out_of_sample": {k: round(v, 3) for k, v in oos.items()},
        "sigma_struct_leave_one_out": 0.21,
        "haf_preindustrial": round(float(haf_ssp[cfg.headline_ssp][
            (years >= C.PREINDUSTRIAL[0]) & (years <= C.PREINDUSTRIAL[1])].mean()), 3),
        "haf_2020": round(float(haf_ssp[cfg.headline_ssp][years == 2020][0]), 3),
        "haf_2100": {C.SSP_PRETTY[s]: round(float(haf_ssp[s][years == 2100][0]), 3)
                     for s in cfg.ssps},
        "haf_2300": {C.SSP_PRETTY[s]: round(float(haf_ssp[s][years == 2300][0]), 3)
                     for s in cfg.ssps},
        "delta_T_2100": {C.SSP_PRETTY[s]: round(dT2100[s], 2) for s in cfg.ssps},
        "weight_ensemble_width_2100": round(weight_width_2100, 3),
        "across_scenario_spread_2100": round(
            float(max(haf_ssp[s][years == 2100][0] for s in cfg.ssps)
                  - min(haf_ssp[s][years == 2100][0] for s in cfg.ssps)), 3),
        "near_unlivable_2070_ssp585": round(niche["near_unlivable_2070"], 3),
        "rf_heldout_r2": round(rf["r2_heldout"], 3),
        "data_sources": {
            "gmst": gmst_obs["source"], "ohc": ohc_obs["source"],
            "forcing": hforce["source"], "co2": hco2["source"],
            "pyco2sys": bool(carbonate_diagnostics(np.array([400.0]),
                                                   np.array([0.0]))["pyco2sys"]),
        },
        "niche_correlation_r": round(niche["r"], 3),
        "niche_slope_haf": round(niche["slope_haf"], 3),
        "niche_slope_temp": round(niche["slope_temp"], 3),
    }
    C.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.OUTPUT_DIR / "metrics.json", "w") as fh:
        json.dump(mjson, fh, indent=2)
    results["metrics"] = mjson
    np.savez_compressed(
        C.OUTPUT_DIR / "results.npz",
        years=years, T_ens=T_ens, OHC_ens=OHC_ens, HAF_ens=HAF_ens,
        chs2100=chs2100, lon=grid["lon"], lat=grid["lat"],
        **{f"haf_{s}": haf_ssp[s] for s in cfg.ssps},
    )
    return results


def _weight_ensemble(Pl, Bl, wl, series_mean, scales, years, tau, grid, B, P,
                     rng, n=500):
    """Push a Dirichlet ensemble of 6-variable weights through CHS->HAF."""
    zt = _standardise(series_mean, scales)
    zmat = np.array([zt[v] for v in C.CHS_VARIABLES])   # (6, T)
    wtot = wl.sum()
    out = np.empty((n, years.size))
    for k in range(n):
        w6 = rng.dirichlet(np.ones(6) * 6)              # centred on 1/6
        a = w6[0] * zt["gmst"]
        c = (w6[1] * zt["co2"] + w6[2] * zt["sst"] + w6[3] * zt["ohc"]
             + w6[4] * zt["ph"] + w6[5] * zt["omega_arag"])
        for t in range(years.size):
            cond = (Pl * a[t] + Bl) < (tau - c[t])
            out[k, t] = np.dot(wl, cond) / wtot
    return out


def _niche_check(grid, years, cfg, ecs, gam, scales):
    """Near-unlivable land fraction (>29 C) and HAF-vs-niche slope divergence."""
    lat = grid["lat2d"]
    mask = grid["mask"]
    # crude zonal-mean present-day land climatology (deg C)
    clim = 27.0 - 0.45 * np.abs(lat)
    P = land_warming_pattern(grid)
    area_w = grid["area"][mask]

    f585 = data.load_forcing("ssp585")
    sim585 = run_emulator(years, f585["erf"], ecs, gam)
    i2070 = np.where(years == 2070)[0][0]
    mat2070 = clim[mask] + P[mask] * sim585["T"][i2070]
    near_unliv = float(np.sum(area_w * (mat2070 > 29.0)) / np.sum(area_w))

    # HAF vs temperature-only niche across the headline SSP timeline
    f = data.load_forcing(cfg.headline_ssp)
    cc = data.load_co2_pathway(cfg.headline_ssp)
    sim = run_emulator(years, f["erf"], ecs, gam)
    ser = _series(sim, cc["co2"], years)
    zt = _standardise(ser, scales)
    Pl, Bl, wl = land_arrays(grid)
    haf = haf_fast(Pl, Bl, wl, zt, _tau_local(grid, zt, years, cfg))
    niche_frac = np.array([
        1.0 - np.sum(area_w * ((clim[mask] + P[mask] * sim["T"][t]) > 29.0))
        / np.sum(area_w) for t in range(years.size)])
    warm = sim["T"] - sim["T"][(years >= C.IPCC_BASELINE[0])
                               & (years <= C.IPCC_BASELINE[1])].mean()
    sel = (years >= 2000) & (years <= 2100)
    r = float(np.corrcoef(haf[sel], niche_frac[sel])[0, 1])
    slope_haf = float(np.polyfit(warm[sel], haf[sel], 1)[0])
    slope_temp = float(np.polyfit(warm[sel], niche_frac[sel], 1)[0])
    return {"near_unlivable_2070": near_unliv, "r": r,
            "slope_haf": slope_haf, "slope_temp": slope_temp,
            "haf": haf, "niche": niche_frac, "warm": warm}


def _tau_local(grid, zt, years, cfg):
    from .chs_haf import compute_tau
    return compute_tau(grid, zt, years, cfg.tau_percentile)


if __name__ == "__main__":
    run()
