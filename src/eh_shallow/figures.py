"""Publication figures for the shallow-time proof-of-concept (Nature style).

All figures are vector PDFs sized to Nature column widths (89 mm single,
183 mm double) with small sans-serif type. The Composite Hazard Score map
rasterises only the pcolormesh layer at 300 dpi while coastlines and text stay
vector. Figures are written to ``config.FIG_DIR`` and mirrored to the
manuscript ``config.PAPER_FIG_DIR``.
"""
from __future__ import annotations

import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

MM = 1 / 25.4
COL1, COL2 = 89 * MM, 183 * MM

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 8,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "lines.linewidth": 1.0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight",
    "savefig.dpi": 300, "figure.dpi": 150,
})

GREEN = "#2a9d5c"
BLUE = "#2166ac"
SSP_COLORS = {"ssp126": "#1b7837", "ssp245": "#5aae61",
              "ssp370": "#d6604d", "ssp585": "#b2182b"}


def _fan(ax, x, ens, color, label):
    lo, mid, hi = np.percentile(ens, [5, 50, 95], axis=0)
    ax.fill_between(x, lo, hi, color=color, alpha=0.25, lw=0)
    ax.plot(x, mid, color=color, lw=1.2, label=label)


def fig_gmst(results, path):
    years = results["years"]
    obs = results["gmst_obs"]
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.75))
    m = years <= 2020
    # Reference the ensemble to the observations' 1850-1900 baseline.
    base = (years >= C.IPCC_BASELINE[0]) & (years <= C.IPCC_BASELINE[1])
    T_ens = results["T_ens"] - results["T_ens"][:, base].mean(axis=1,
                                                              keepdims=True)
    _fan(ax, years[m], T_ens[:, m], GREEN, "Emulator posterior")
    ax.scatter(obs["years"], obs["gmst"], s=3, color="0.25",
               label=f"{obs['source']}", zorder=5)
    ax.axvspan(*C.CALIB_GMST, color="0.85", alpha=0.5, lw=0)
    ax.axvspan(C.VALID_GMST[0], C.VALID_GMST[1], color="#ffe0b3", alpha=0.4, lw=0)
    ax.set_xlim(1850, 2020)
    ax.set_xlabel("Year")
    ax.set_ylabel("GMST anomaly (K, vs 1850--1900)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def fig_ohc(results, path):
    years = results["years"]
    obs = results["ohc_obs"]
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.75))
    m = (years >= 1990) & (years <= 2020)
    _fan(ax, years[m], results["OHC_ens"][:, m], BLUE, "Emulator posterior")
    o = (obs["years"] >= 2005) & (obs["years"] <= 2020)
    ax.scatter(obs["years"][o], obs["ohc"][o], s=4, color="0.25",
               label=obs["source"], zorder=5)
    ax.axvspan(*C.CALIB_OHC, color="0.85", alpha=0.5, lw=0)
    ax.set_xlim(1990, 2020)
    ax.set_xlabel("Year")
    ax.set_ylabel("OHC 0--2000 m anomaly (ZJ)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, path)


def fig_posterior(results, path):
    smc = results["smc"]
    theta, w = smc["theta"], smc["weights"]
    rng = np.random.default_rng(C.SEED)
    prior_ecs = rng.normal(*C.ECS_PRIOR[1:], size=4000)
    prior_gam = rng.uniform(*C.GAMMA_PRIOR[1:], size=4000)
    idx = rng.choice(theta.shape[0], size=2000, p=w)
    fig, axes = plt.subplots(1, 2, figsize=(COL2 * 0.66, COL1 * 0.7))
    for ax, (i, name, prior, unit) in zip(axes, [
        (0, "ECS", prior_ecs, "K"),
        (1, r"$\gamma$", prior_gam, r"W m$^{-2}$ K$^{-1}$")]):
        ax.hist(prior, bins=40, density=True, color="0.7", alpha=0.6,
                label="Prior")
        ax.hist(theta[idx, i], bins=40, density=True, color=GREEN, alpha=0.7,
                label="Posterior")
        ax.set_xlabel(f"{name} ({unit})")
        ax.set_yticks([])
    axes[0].legend(frameon=False)
    _save(fig, path)


def fig_haf(results, path):
    years = results["years"]
    haf = results["haf_ssp"][C.HEADLINE_SSP]
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.75))
    lo, hi = (results["haf_band"][C.TAU_PERCENTILE_BAND[0]],
              results["haf_band"][C.TAU_PERCENTILE_BAND[1]])
    ax.fill_between(years, np.minimum(lo, hi), np.maximum(lo, hi),
                    color="0.7", alpha=0.4, lw=0,
                    label="80th--99th pct reference band")
    e_lo, e_hi = np.percentile(results["HAF_ens"], [5, 95], axis=0)
    ax.fill_between(years, e_lo, e_hi, color=GREEN, alpha=0.3, lw=0,
                    label="5--95% posterior")
    ax.plot(years, haf, color=GREEN, lw=1.3)
    ax.set_xlim(1750, 2300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Habitable Area Fraction")
    ax.set_title(C.SSP_PRETTY[C.HEADLINE_SSP], fontsize=7)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def fig_haf_scenarios(results, path):
    years = results["years"]
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.75))
    for ssp in C.SSPS:
        ax.plot(years, results["haf_ssp"][ssp], color=SSP_COLORS[ssp],
                lw=1.3, label=C.SSP_PRETTY[ssp])
    ax.set_xlim(1750, 2300)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year")
    ax.set_ylabel("Habitable Area Fraction")
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def fig_chs_map(results, path):
    lon, lat = results["grid"]["lon"], results["grid"]["lat"]
    chs = results["chs2100"]
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        fig = plt.figure(figsize=(COL2, COL2 * 0.5))
        ax = plt.axes(projection=ccrs.Robinson())
        vmax = np.nanpercentile(np.abs(chs), 98)
        pm = ax.pcolormesh(lon, lat, chs, transform=ccrs.PlateCarree(),
                           cmap="YlOrRd", vmin=np.nanmin(chs), vmax=vmax,
                           rasterized=True, shading="auto")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.3)
        ax.set_global()
        cb = fig.colorbar(pm, ax=ax, orientation="horizontal", pad=0.03,
                          shrink=0.6, aspect=40)
        cb.set_label("Composite Hazard Score (2100, SSP2-4.5)")
    except Exception:
        fig, ax = plt.subplots(figsize=(COL2, COL2 * 0.5))
        pm = ax.pcolormesh(lon, lat, chs, cmap="YlOrRd", rasterized=True,
                           shading="auto")
        fig.colorbar(pm, ax=ax, orientation="horizontal",
                     label="Composite Hazard Score (2100, SSP2-4.5)")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    _save(fig, path)


def fig_weight_ensemble(results, path):
    years = results["years"]
    hw = results["haf_weight_ens"]
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.75))
    for (a, b, al) in [(5, 95, 0.25), (25, 75, 0.4)]:
        lo, hi = np.percentile(hw, [a, b], axis=0)
        ax.fill_between(years, lo, hi, color=GREEN, alpha=al, lw=0,
                        label=f"{a}--{b}%")
    ax.plot(years, np.median(hw, axis=0), color=GREEN, lw=1.2)
    ax.set_xlim(1750, 2300); ax.set_ylim(0, 1)
    ax.set_xlabel("Year"); ax.set_ylabel("HAF (weight ensemble)")
    ax.set_title(C.SSP_PRETTY[C.HEADLINE_SSP], fontsize=7)
    ax.legend(frameon=False, loc="lower left")
    _save(fig, path)


def fig_rf_importances(results, path):
    rf = results["rf"]
    order = np.argsort(rf["importances"])
    fig, ax = plt.subplots(figsize=(COL1, COL1 * 0.7))
    ax.barh(np.array(rf["names"])[order], rf["importances"][order],
            color=BLUE, alpha=0.8)
    ax.set_xlabel("Permutation importance")
    ax.set_title(f"WHI baseline field ($R^2_{{oos}}$={rf['r2_heldout']:.2f})",
                 fontsize=7)
    _save(fig, path)


def make_all(results):
    figs = {
        "gmst_fit.pdf": fig_gmst,
        "ohc_fit.pdf": fig_ohc,
        "posterior.pdf": fig_posterior,
        "haf.pdf": fig_haf,
        "haf_scenarios.pdf": fig_haf_scenarios,
        "chs_map_2100.pdf": fig_chs_map,
        "haf_weight_ensemble.pdf": fig_weight_ensemble,
        "whi_rf_importances.pdf": fig_rf_importances,
    }
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    # Mirror to the manuscript figure tree only when it is a distinct directory;
    # on the cluster PAPER_FIG_DIR may resolve to FIG_DIR itself (no manuscript
    # tree), in which case copying a figure onto itself would raise SameFileError.
    mirror = (C.PAPER_FIG_DIR.exists()
              and C.PAPER_FIG_DIR.resolve() != C.FIG_DIR.resolve())
    written = []
    for name, fn in figs.items():
        p = C.FIG_DIR / name
        fn(results, p)
        written.append(name)
        if mirror:
            shutil.copy(p, C.PAPER_FIG_DIR / name)
    return written


def _save(fig, path):
    fig.savefig(path)
    plt.close(fig)
