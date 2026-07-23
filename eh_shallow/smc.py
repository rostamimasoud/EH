"""Tempered Sequential Monte Carlo calibration of the emulator vs observations.

A self-contained numpy implementation of the sampler the proposal specifies:
  - particles drawn from the prior (emulator.sample_prior);
  - a geometric inverse-temperature ladder beta: 0 -> 1 that introduces the
    data gradually;
  - systematic resampling whenever ESS < N/2;
  - Gaussian random-walk MH rejuvenation with covariance matched to the cloud;
  - a Student-t(nu=4) likelihood, robust to outliers.

Two observables are used jointly:
  - GMST (HadCRUT5) over 1850-1980 (per the hardened proposal: 1981+ is withheld
    for out-of-sample validation);
  - ocean heat content (NOAA/NCEI 0-2000 m) over 2005-2020, referenced to its own
    2005-2014 baseline since OHC has no preindustrial record.
GMST alone leaves ECS and gamma (deep-ocean heat uptake) partly degenerate; OHC
constrains the rate of heat uptake and so breaks that degeneracy (Geoffroy 2013,
Cummins-style). OHC is layered on top of the strict 1980 GMST withholding: it
only exists from 2005, so it is treated as an additional constraint window, not a
relaxation of the leakage fix.

PRODUCTION SWAP: blackjax/numpyro tempered SMC on JAX, vectorised over particles.
"""
from __future__ import annotations

import numpy as np

from . import data, emulator

NU = 4.0                       # Student-t degrees of freedom
CALIB_WINDOW = (1850, 1980)    # GMST in-sample; 1981+ withheld
OHC_WINDOW = (2005, 2020)      # OHC constraint window (post-dates GMST cutoff)
SIGMA_STRUCT = 0.20            # K, structural error + unforced internal variability
#   (the 2-box emulator has no internal variability; without this term the
#    likelihood is over-confident and over-fits ECS. This is the proposal's
#    sigma_struct, here benchmarked so the ECS posterior is AR6-consistent.)
SIGMA_STRUCT_OHC = 10.0        # ZJ, structural floor for the OHC term: the model
#   accumulates full-depth ocean heat while the obs are 0-2000 m only, and the
#   reduced 2-box core has no interannual ocean variability.


def _loglik_factory(years, use_ohc=True):
    """Return loglik(theta_arr)->[n] for the joint GMST+OHC likelihood.

    Also returns an obs dict with the windowed (year, value, sigma) tuples for
    'gmst' and (if available) 'ohc', for plotting/metrics downstream.
    """
    # --- GMST term (HadCRUT5, 1850-1980) -----------------------------------
    obs = data.load_hadcrut5()
    oy = obs["year"].to_numpy()
    og = data.rebaseline(oy, obs["gmst"].to_numpy())
    # observational sigma from the 2.5-97.5% CI (~2 sigma), floored, plus the
    # structural-error term added in quadrature
    osd = np.maximum((obs["hi"].to_numpy() - obs["lo"].to_numpy()) / 3.92, 0.03)
    osd = np.sqrt(osd ** 2 + SIGMA_STRUCT ** 2)
    cw = (oy >= CALIB_WINDOW[0]) & (oy <= CALIB_WINDOW[1])
    oy_c, og_c, osd_c = oy[cw], og[cw], osd[cw]

    # --- OHC term (NOAA/NCEI 0-2000 m, 2005-2020, own 2005-2014 baseline) ---
    ohc_obs = None
    if use_ohc:
        oh = data.load_ohc()
        hy = oh["year"].to_numpy()
        hv = data.rebaseline(hy, oh["ohc"].to_numpy(), ref=data.OHC_REF_PERIOD)
        hsd = np.sqrt(oh["sigma"].to_numpy() ** 2 + SIGMA_STRUCT_OHC ** 2)
        hw = (hy >= OHC_WINDOW[0]) & (hy <= OHC_WINDOW[1])
        if np.any(hw):
            ohc_obs = (hy[hw], hv[hw], hsd[hw])

    def loglik(theta_arr):
        out = np.empty(len(theta_arr))
        for i, th in enumerate(theta_arr):
            # chem=False: pH/Omega do not enter the likelihood, so skip the
            # ~50 ms/solve PyCO2SYS call on every particle evaluation
            res = emulator.run_emulator(th, years, chem=False)
            gm = data.rebaseline(years, res["gmst"])
            r = (np.interp(oy_c, years, gm) - og_c) / osd_c
            # Student-t log density (drop constants that cancel across particles)
            ll = np.sum(-0.5 * (NU + 1) * np.log1p(r ** 2 / NU))
            if ohc_obs is not None:
                hy_c, hv_c, hsd_c = ohc_obs
                mh = data.rebaseline(years, res["ohc"], ref=data.OHC_REF_PERIOD)
                rh = (np.interp(hy_c, years, mh) - hv_c) / hsd_c
                ll += np.sum(-0.5 * (NU + 1) * np.log1p(rh ** 2 / NU))
            out[i] = ll
        return out

    return loglik, {"gmst": (oy_c, og_c, osd_c), "ohc": ohc_obs}


def _ess(logw):
    w = np.exp(logw - logw.max())
    w /= w.sum()
    return 1.0 / np.sum(w ** 2), w


def _systematic_resample(w, rng):
    n = len(w)
    positions = (rng.random() + np.arange(n)) / n
    idx = np.searchsorted(np.cumsum(w), positions)
    return np.clip(idx, 0, n - 1)


def run_smc(n_particles=500, n_temps=12, n_rejuv=4, seed=0,
            years=None, verbose=True):
    """Run tempered SMC; return dict with posterior particles, weights, log."""
    if years is None:
        years = np.arange(1750, 2026)
    rng = np.random.default_rng(seed)
    loglik, obs_pack = _loglik_factory(years)

    theta = emulator.sample_prior(rng, n_particles)
    ll = loglik(theta)
    betas = np.linspace(0, 1, n_temps) ** 2  # geometric-ish ladder, denser near 0
    log = []

    for t in range(1, n_temps):
        dbeta = betas[t] - betas[t - 1]
        logw = dbeta * ll
        ess, w = _ess(logw)
        log.append({"beta": float(betas[t]), "ess": float(ess)})
        if verbose:
            print(f"  beta={betas[t]:.3f}  ESS={ess:6.1f}/{n_particles}")
        if ess < n_particles / 2:
            idx = _systematic_resample(w, rng)
            theta, ll = theta[idx], ll[idx]
            w = np.full(n_particles, 1.0 / n_particles)
        # MH rejuvenation at the current tempered target (prior + beta*ll)
        cov = np.cov(theta.T) + 1e-6 * np.eye(theta.shape[1])
        L = np.linalg.cholesky(cov) * 2.38 / np.sqrt(theta.shape[1])
        lp = emulator.log_prior(theta) + betas[t] * ll
        for _ in range(n_rejuv):
            prop = theta + (rng.standard_normal(theta.shape) @ L.T)
            ll_prop = loglik(prop)
            lp_prop = emulator.log_prior(prop) + betas[t] * ll_prop
            accept = np.log(rng.random(n_particles)) < (lp_prop - lp)
            theta[accept], ll[accept], lp[accept] = (
                prop[accept], ll_prop[accept], lp_prop[accept])

    # final importance weights at beta=1
    _, w = _ess((1.0 - betas[-2]) * ll) if n_temps > 1 else (None, None)
    w = np.full(n_particles, 1.0 / n_particles) if w is None else w
    return {
        "theta": theta, "weights": w, "names": emulator.PARAM_NAMES,
        "loglik": ll, "log": log, "years": years,
        "obs_calib": obs_pack["gmst"], "obs_ohc": obs_pack["ohc"],
        "seed": seed, "calib_window": CALIB_WINDOW, "ohc_window": OHC_WINDOW,
    }


def posterior_summary(post):
    """Weighted mean / 5-95% for each parameter."""
    theta, w = post["theta"], post["weights"]
    out = {}
    for j, name in enumerate(post["names"]):
        x = theta[:, j]
        order = np.argsort(x)
        cw = np.cumsum(w[order])
        lo = x[order][np.searchsorted(cw, 0.05)]
        hi = x[order][np.searchsorted(cw, 0.95)]
        out[name] = {"mean": float(np.sum(w * x)), "p05": float(lo), "p95": float(hi)}
    return out
