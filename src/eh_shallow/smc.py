"""Tempered Sequential Monte Carlo calibration of the two-parameter emulator.

Calibrates (ECS, gamma) jointly against HadCRUT5 GMST over 1850-1980 and
NOAA/NCEI 0-2000 m ocean heat content over 2005-2020, with 1981-2020 GMST
withheld for out-of-sample validation. The likelihood is a robust Student-t
(nu=4). Tempering uses a fixed inverse-temperature ladder; systematic
resampling fires when ESS < N/2; Gaussian random-walk Metropolis rejuvenation
uses a proposal covariance matched to the particle cloud.

All randomness is driven by a single seeded ``numpy`` generator, so a run is
bit-reproducible.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

from . import config as C
from .emulator import run_emulator


def _sample_prior(rng, n):
    _, e_mu, e_sd = C.ECS_PRIOR
    _, g_lo, g_hi = C.GAMMA_PRIOR
    ecs = rng.normal(e_mu, e_sd, n)
    gamma = rng.uniform(g_lo, g_hi, n)
    return np.column_stack([ecs, gamma])


def _log_prior(theta):
    _, e_mu, e_sd = C.ECS_PRIOR
    _, g_lo, g_hi = C.GAMMA_PRIOR
    ecs, gamma = theta[:, 0], theta[:, 1]
    lp = -0.5 * ((ecs - e_mu) / e_sd) ** 2 - np.log(e_sd)
    lp = np.where((gamma >= g_lo) & (gamma <= g_hi), lp, -np.inf)
    lp = np.where(ecs > 0.5, lp, -np.inf)
    return lp


def _student_t_ll(resid, sigma, nu=C.STUDENT_T_NU):
    """Summed Student-t log-likelihood for residual vector ``resid``."""
    c = (gammaln((nu + 1) / 2) - gammaln(nu / 2)
         - 0.5 * np.log(np.pi * nu) - np.log(sigma))
    return np.sum(c - (nu + 1) / 2 * np.log1p((resid / sigma) ** 2 / nu))


def _log_likelihood_single(theta, forcing_years, forcing, obs):
    ecs, gamma = float(theta[0]), float(theta[1])
    if ecs <= 0.5:
        return -np.inf
    sim = run_emulator(forcing_years, forcing, ecs, gamma)
    yrs = sim["years"]

    # GMST term (calibration window 1850-1980). Reference the simulated GMST to
    # the same 1850-1900 baseline as the observations before differencing.
    # Otherwise the emulator's 1750->1850 warming -- which grows with ECS --
    # enters as a spurious, ECS-dependent offset that inverts the likelihood
    # (drives the posterior to the lower ECS bound). Mirrors the OHC term's
    # 2005-2014 re-referencing below.
    gy, gv = obs["gmst_years"], obs["gmst_vals"]
    m = (gy >= C.CALIB_GMST[0]) & (gy <= C.CALIB_GMST[1])
    base = (yrs >= C.IPCC_BASELINE[0]) & (yrs <= C.IPCC_BASELINE[1])
    sim_T = sim["T"] - sim["T"][base].mean()
    idx = np.searchsorted(yrs, gy[m])
    resid_g = sim_T[idx] - gv[m]
    ll = _student_t_ll(resid_g, C.SIGMA_STRUCT_GMST)

    # OHC term (2005-2020), both referenced to 2005-2014.
    oy, ov = obs["ohc_years"], obs["ohc_vals"]
    mo = (oy >= C.CALIB_OHC[0]) & (oy <= C.CALIB_OHC[1])
    idxo = np.searchsorted(yrs, oy[mo])
    sim_ohc = sim["OHC"][idxo]
    ref = (yrs >= C.OHC_REF[0]) & (yrs <= C.OHC_REF[1])
    sim_ohc = sim_ohc - sim["OHC"][ref].mean()
    resid_o = sim_ohc - ov[mo]
    ll += _student_t_ll(resid_o, C.SIGMA_OHC)
    return ll


def _log_likelihood(theta, forcing_years, forcing, obs):
    return np.array([_log_likelihood_single(t, forcing_years, forcing, obs)
                     for t in theta])


def run_smc(forcing_years, forcing, obs, cfg: C.RunConfig):
    """Tempered SMC. Returns posterior particles, weights, and diagnostics."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_particles
    theta = _sample_prior(rng, n)
    logL = _log_likelihood(theta, forcing_years, forcing, obs)
    logL = np.where(np.isfinite(logL), logL, -1e12)
    logw = np.full(n, -np.log(n))

    betas = np.linspace(0, 1, cfg.n_temp_steps + 1)
    ess_trace = []
    for b0, b1 in zip(betas[:-1], betas[1:]):
        db = b1 - b0
        logw = logw + db * logL
        logw -= _logsumexp(logw)
        w = np.exp(logw)
        ess = 1.0 / np.sum(w ** 2)
        ess_trace.append(ess)
        if ess < C.ESS_FRACTION * n:
            idx = _systematic_resample(w, rng)
            theta, logL = theta[idx], logL[idx]
            logw = np.full(n, -np.log(n))
        # Metropolis rejuvenation at inverse temperature b1.
        theta, logL = _rejuvenate(theta, logL, b1, forcing_years, forcing,
                                  obs, rng)

    logw -= _logsumexp(logw)
    w = np.exp(logw)
    return {"theta": theta, "weights": w, "logL": logL, "ess": ess_trace,
            "names": ("ECS", "gamma")}


def _rejuvenate(theta, logL, beta, fy, forcing, obs, rng, moves=2):
    n = theta.shape[0]
    cov = np.cov(theta.T) + 1e-6 * np.eye(2)
    step = 2.38 / np.sqrt(2) * np.linalg.cholesky(cov)
    lp = _log_prior(theta)
    for _ in range(moves):
        prop = theta + (rng.standard_normal((n, 2)) @ step.T)
        lp_prop = _log_prior(prop)
        finite = np.isfinite(lp_prop)
        logL_prop = np.where(finite, -1e12, -1e12)
        if finite.any():
            ll = _log_likelihood(prop[finite], fy, forcing, obs)
            logL_prop[finite] = np.where(np.isfinite(ll), ll, -1e12)
        log_alpha = (beta * (logL_prop - logL)) + (lp_prop - lp)
        accept = np.log(rng.random(n)) < log_alpha
        theta = np.where(accept[:, None], prop, theta)
        logL = np.where(accept, logL_prop, logL)
        lp = np.where(accept, lp_prop, lp)
    return theta, logL


def _systematic_resample(w, rng):
    n = w.size
    positions = (rng.random() + np.arange(n)) / n
    return np.searchsorted(np.cumsum(w), positions)


def _logsumexp(x):
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def posterior_summary(res, q=(5, 50, 95)):
    """Weighted percentiles of each parameter."""
    theta, w = res["theta"], res["weights"]
    out = {}
    for j, name in enumerate(res["names"]):
        out[name] = _weighted_percentile(theta[:, j], w, q)
    return out


def _weighted_percentile(x, w, q):
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cdf = np.cumsum(ws) - 0.5 * ws
    cdf /= np.sum(ws)
    return {p: float(np.interp(p / 100.0, cdf, xs)) for p in q}
