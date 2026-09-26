"""Publication figure layer for the multi-sphere habitability analysis.

Builds the main-text and supplementary figures from ``figdata.npz`` (written by
the end-to-end driver) together with the independent validation modules. Vector
PDF output, sans-serif type, 88 mm single-column and 180 mm double-column
widths.

    python -m eh_shallow.paperfigs --datadir outputs --outdir outputs/paper
"""
from __future__ import annotations

import argparse
import copy as _copy
import json
import os

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.4, "ytick.major.size": 2.4,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 400, "pdf.fonttype": 42, "ps.fonttype": 42,
    "legend.frameon": False, "lines.solid_capstyle": "round",
})

MM = 1 / 25.4
ONE_COL, TWO_COL = 88 * MM, 180 * MM


SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")
SSP_COLOR = {"ssp126": "#1b6ca8", "ssp245": "#2e9e5b",
             "ssp370": "#e07b22", "ssp585": "#b62025"}
SSP_LABEL = {"ssp126": "SSP1-2.6", "ssp245": "SSP2-4.5",
             "ssp370": "SSP3-7.0", "ssp585": "SSP5-8.5"}

# Sequential hazard ramp: pale sand through ochre to deep crimson, then near-black
# for the most hazardous tail, so the dry belts read as distinct from the interiors.
HAZARD = LinearSegmentedColormap.from_list("hazard", [
    "#f7f4ec", "#f6e2b3", "#f3c169", "#ea9a3c", "#d96b2c",
    "#b8342a", "#8c1d35", "#4d0f27"])
# Emergence ramp: urgent (early) is hot, late is cool, so "soon" is visually loud.
EMERGE = LinearSegmentedColormap.from_list("emerge", [
    "#7f0000", "#c2321f", "#ef7215", "#f7c33f", "#bfe07c",
    "#5fc0a6", "#2b7fbd", "#2d3a8c"])


def _panel(ax, letter, x=-0.14, y=1.03):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left")


def _land_arrays(d):
    """Return the land-flattened baseline field, warming pattern and area weight."""
    land = d["land2d"].astype(bool)
    lat2d = np.repeat(d["lat"][:, None], d["lon"].size, axis=1)
    B = d["B2d"][land]
    P = d["P2d"][land]
    A = np.cos(np.deg2rad(lat2d))[land]
    return B, P, A


def _robinson(fig, rect):
    """Axes with a Robinson projection and coastlines when cartopy is importable."""
    try:
        import cartopy.crs as ccrs
        ax = fig.add_subplot(rect, projection=ccrs.Robinson())
        ax.set_global()
        return ax, ccrs.PlateCarree()
    except Exception:
        ax = fig.add_subplot(rect)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        return ax, None


def _coast(ax):
    try:
        ax.coastlines(linewidth=0.25, color="0.25")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# The habitability response surface: H(s_T, U)
# ---------------------------------------------------------------------------
def response_surface(B, P, A, tau, sT_grid, U_grid):
    """Habitable area fraction as a function of the two aggregate drivers.

    Under the tiered decomposition the per-cell score is
    ``P(c) s_T + B(c) + U``, so the area-weighted habitable fraction depends on
    space and time only through the two scalars ``(s_T, U)``.
    """
    Atot = A.sum()
    H = np.empty((U_grid.size, sT_grid.size))
    for i, u in enumerate(U_grid):
        lim = tau - u
        for j, s in enumerate(sT_grid):
            H[i, j] = A[(P * s + B) < lim].sum() / Atot
    return H


# ---------------------------------------------------------------------------
# Figure 1 -- trajectories, uncertainty, and land area at stake
# ---------------------------------------------------------------------------
def fig_trajectories(d, m, path):
    yr = d["years"]
    fig = plt.figure(figsize=(TWO_COL, 128 * MM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92],
                          hspace=0.42, wspace=0.30,
                          left=0.075, right=0.985, top=0.95, bottom=0.085)

    # (a) scenario trajectories, with an observational-era inset -------------
    axa = fig.add_subplot(gs[0, :])
    axa.axvspan(1750, 2020, color="#f0f0f0", zorder=0, lw=0)
    for s in SSPS:
        axa.plot(yr, d[f"haf_{s}"], lw=1.3, color=SSP_COLOR[s], label=SSP_LABEL[s])
    hpi = m["committed_vs_avoidable"]["haf_preindustrial"]
    axa.axhline(hpi, color="0.45", lw=0.6, ls=(0, (4, 3)))
    axa.text(2296, hpi + 0.018, "preindustrial reference", fontsize=5.8,
             color="0.35", ha="right")
    axa.text(1885, 0.955, "observed to 2020", fontsize=6, color="0.4", ha="center")
    axa.set_xlim(1750, 2300); axa.set_ylim(0, 1.0)
    axa.set_xlabel("Year")
    axa.set_ylabel("Habitable area fraction")
    axa.legend(loc="lower left", bbox_to_anchor=(0.355, 0.03), ncol=2,
               handlelength=1.5, columnspacing=1.0)
    _panel(axa, "a", x=-0.065)

    ins = axa.inset_axes([0.055, 0.115, 0.265, 0.335])
    ins.axvspan(1750, 2020, color="#f0f0f0", zorder=0, lw=0)
    for s in SSPS:
        ins.plot(yr, d[f"haf_{s}"], lw=0.9, color=SSP_COLOR[s])
    h2020 = m["committed_vs_avoidable"]["haf_2020"]
    ins.plot([2020], [h2020], marker="o", ms=3.2, color="k", zorder=6)
    ins.annotate(f"2020\n{h2020:.2f}", xy=(2020, h2020), xytext=(1972, 0.845),
                 fontsize=5.2, color="k",
                 arrowprops=dict(arrowstyle="-", lw=0.4, color="0.4"))
    ins.set_xlim(1950, 2060); ins.set_ylim(0.70, 0.92)
    ins.set_yticks([0.75, 0.85])
    ins.set_xticks([1960, 2000, 2040])
    ins.tick_params(labelsize=5.2, pad=1.2)
    ins.patch.set_alpha(0.92)
    for sp in ins.spines.values():
        sp.set_linewidth(0.4)
    ins.set_title("recent decades", fontsize=5.6, pad=2)

    # (b) posterior and threshold sensitivity, with a 2100 spread inset ------
    axb = fig.add_subplot(gs[1, 0])
    lo, mid, hi = np.percentile(d["ens_haf"], [5, 50, 95], axis=0)
    band = np.array([d[f"pctile_{p}"] for p in (80, 85, 90, 95, 99)])
    axb.fill_between(yr, band.min(0), band.max(0), color="#e6e6e6",
                     lw=0, label="threshold 80th to 99th percentile")
    axb.fill_between(yr, lo, hi, color="#bfe0cb", lw=0, label="posterior 5th to 95th")
    axb.plot(yr, mid, color="#1d7f47", lw=1.2, label="posterior median")
    axb.set_xlim(1750, 2300); axb.set_ylim(0, 1.0)
    axb.set_xlabel("Year"); axb.set_ylabel("Habitable area fraction")
    # legend in the free upper-right corner so it clears the inset below it
    axb.legend(loc="upper right", bbox_to_anchor=(1.005, 1.035),
               handlelength=1.4, fontsize=5.4, labelspacing=0.35)
    _panel(axb, "b")

    # inset: the three sources of spread in the 2100 value, on a common scale
    insb = axb.inset_axes([0.095, 0.095, 0.395, 0.295])
    i2100 = yr == 2100
    par = d["ens_haf"][:, i2100].ravel()
    wgt = d["haf_weights"][:, i2100].ravel()
    scen = np.array([d[f"haf_{s}"][i2100][0] for s in SSPS])
    bars = [("pathway", scen.max() - scen.min(), "0.20"),
            ("weights", np.diff(np.percentile(wgt, [5, 95]))[0], "#6a51a3"),
            ("parameters", np.diff(np.percentile(par, [5, 95]))[0], "#1d7f47")]
    insb.bar(range(3), [b[1] for b in bars], 0.62,
             color=[b[2] for b in bars], edgecolor="none")
    for j, b in enumerate(bars):
        insb.text(j, b[1] + 0.012, f"{b[1]:.2f}", ha="center", fontsize=4.8,
                  color=b[2])
    insb.set_xticks(range(3))
    insb.set_xticklabels([b[0] for b in bars], fontsize=4.8)
    insb.set_ylim(0, 0.52); insb.set_yticks([0, 0.2, 0.4])
    insb.tick_params(labelsize=4.8, pad=1.0)
    insb.patch.set_alpha(0.92)
    for sp in insb.spines.values():
        sp.set_linewidth(0.4)
    insb.set_title("width of the 2100 range", fontsize=5.0, pad=2)

    # (c) habitable land area at stake --------------------------------------
    axc = fig.add_subplot(gs[1, 1])
    x = np.arange(4); bw = 0.36
    loss100 = [m["committed_vs_avoidable"]["by_scenario_2100"][s]
               ["total_loss_vs_PI_2100_mkm2"] for s in SSPS]
    land_mkm2 = m["committed_vs_avoidable"]["domain_land_million_km2"]
    loss300 = [(m["committed_vs_avoidable"]["haf_preindustrial"]
                - m["scenarios"][s]["haf_2300"]) * land_mkm2 for s in SSPS]
    committed = m["committed_vs_avoidable"]["committed_loss_by_2020_million_km2"]
    axc.bar(x - bw / 2, loss100, bw, color=[SSP_COLOR[s] for s in SSPS],
            edgecolor="none", label="by 2100")
    axc.bar(x + bw / 2, loss300, bw, color=[SSP_COLOR[s] for s in SSPS],
            edgecolor="none", alpha=0.42, label="by 2300")
    axc.axhline(committed, color="k", lw=0.8, ls=(0, (4, 3)))
    axc.set_xticks(x); axc.set_xticklabels([SSP_LABEL[s] for s in SSPS], fontsize=6)
    axc.set_ylabel("Habitable land lost since 1750 (million km$^2$)")
    axc.set_ylim(0, 152)
    axc.legend(handles=[Patch(facecolor="0.35", label="by 2100"),
                        Patch(facecolor="0.35", alpha=0.42, label="by 2300"),
                        Line2D([], [], color="k", lw=0.8, ls=(0, (4, 3)),
                               label=f"already lost by 2020, "
                                     f"{committed:.0f} million km$^2$")],
               loc="upper center", bbox_to_anchor=(0.52, 1.03), ncol=2,
               handlelength=1.3, columnspacing=0.9, fontsize=5.4)
    _panel(axc, "c")

    insc = axc.inset_axes([0.075, 0.475, 0.355, 0.285])
    rate = [m["peak_loss_rate"][s]["peak_decadal_loss_mkm2"] for s in SSPS]
    insc.bar(x, rate, 0.62, color=[SSP_COLOR[s] for s in SSPS], edgecolor="none")
    for k, s in enumerate(SSPS):
        insc.text(k, rate[k] + 0.9, f"{rate[k]:.0f}", ha="center", fontsize=4.6,
                  color="0.25")
    insc.set_xticks(x); insc.set_xticklabels(["1-2.6", "2-4.5", "3-7.0", "5-8.5"],
                                             fontsize=4.8)
    insc.set_ylim(0, 29)
    insc.set_yticks([0, 10, 20])
    insc.tick_params(labelsize=5.0, pad=1.0)
    insc.patch.set_alpha(0.92)
    for sp in insc.spines.values():
        sp.set_linewidth(0.4)
    insc.set_title("fastest decade (million km$^2$)", fontsize=5.0, pad=2)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 -- the habitability response surface
# ---------------------------------------------------------------------------
def fig_response_surface(d, path, n=120):
    B, P, A = _land_arrays(d)
    tau = float(d["tau"])
    sT_all = np.concatenate([d[f"sT_{s}"] for s in SSPS])
    U_all = np.concatenate([d[f"U_{s}"] for s in SSPS])
    sg = np.linspace(min(0.0, sT_all.min()), sT_all.max() * 1.05, n)
    ug = np.linspace(min(0.0, U_all.min()), U_all.max() * 1.05, n)
    H = response_surface(B, P, A, tau, sg, ug)
    SG, UG = np.meshgrid(sg, ug)

    yr = d["years"]
    kof = lambda y: int(np.argmin(abs(yr - y)))  # noqa: E731

    def on_surface(x, y):
        return np.array([H[np.argmin(abs(ug - b)), np.argmin(abs(sg - a))]
                         for a, b in zip(x, y)])

    fig = plt.figure(figsize=(TWO_COL, 152 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 0.80],
                          wspace=0.24, hspace=0.34,
                          left=0.075, right=0.945, top=0.985, bottom=0.065)

    # (a) the surface itself, with the pathways climbing across it -----------
    axa = fig.add_subplot(gs[0, 0], projection="3d")
    axa.plot_surface(SG, UG, H, cmap="YlGnBu", rstride=1, cstride=1,
                     linewidth=0, antialiased=True,
                     alpha=0.96, shade=True, vmin=0, vmax=1)
    axa.contour(SG, UG, H, levels=[0.2, 0.4, 0.6, 0.8], colors="0.45",
                linewidths=0.4, offset=0.0)
    hist = slice(0, kof(2020) + 1)
    xr3, yr3 = d["sT_ssp585"], d["U_ssp585"]
    axa.plot(xr3, yr3, on_surface(xr3, yr3) + 0.025, color="w", lw=2.2,
             zorder=8)
    axa.plot(xr3, yr3, on_surface(xr3, yr3) + 0.028, color="0.10", lw=1.0,
             zorder=9)
    for s in SSPS:
        j = kof(2100)
        axa.scatter([d[f"sT_{s}"][j]], [d[f"U_{s}"][j]],
                    [on_surface([d[f"sT_{s}"][j]], [d[f"U_{s}"][j]])[0] + 0.03],
                    s=15, color=SSP_COLOR[s], edgecolors="w", linewidths=0.4,
                    depthshade=False, zorder=10)
        j = kof(2300)
        axa.scatter([d[f"sT_{s}"][j]], [d[f"U_{s}"][j]],
                    [on_surface([d[f"sT_{s}"][j]], [d[f"U_{s}"][j]])[0] + 0.03],
                    s=13, marker="s", color=SSP_COLOR[s], edgecolors="w",
                    linewidths=0.4, depthshade=False, zorder=10)
    k = kof(2020)
    axa.scatter([d["sT_ssp245"][k]], [d["U_ssp245"][k]],
                [on_surface([d["sT_ssp245"][k]], [d["U_ssp245"][k]])[0] + 0.03],
                marker="*", s=30, color="k", depthshade=False, zorder=11)
    axa.set_xlabel("Thermal driver $s_T$", labelpad=-3, fontsize=6.2)
    axa.set_ylabel("Ocean and carbon\ndriver $U$", labelpad=-3, fontsize=6.2)
    axa.set_zlabel("Habitable area fraction", labelpad=9.0, fontsize=6.2)
    axa.tick_params(axis="x", labelsize=5.0, pad=-3.5)
    axa.tick_params(axis="y", labelsize=5.0, pad=-3.0)
    axa.tick_params(axis="z", labelsize=5.0, pad=1.0)
    axa.set_xlim(0, sg.max()); axa.set_ylim(0, ug.max()); axa.set_zlim(0, 1)
    axa.set_xticks([0, 0.2, 0.4, 0.6]); axa.set_yticks([0, 2, 4, 6])
    axa.set_zticks([0, 0.5, 1.0])
    axa.view_init(elev=30, azim=-118)
    try:
        axa.set_box_aspect((1, 1.05, 0.85), zoom=1.0)
    except TypeError:
        axa.set_box_aspect((1, 1.05, 0.85))
    except Exception:
        pass
    for pane in (axa.xaxis, axa.yaxis, axa.zaxis):
        pane.pane.set_facecolor("w"); pane.pane.set_edgecolor("0.88")
        pane._axinfo["grid"]["color"] = "0.92"
        pane._axinfo["grid"]["linewidth"] = 0.3
    axa.text2D(0.02, 0.92, "a", transform=axa.transAxes, fontsize=9,
               fontweight="bold")

    # (b) the surface from above. Every pathway advances along one shared
    # direction in driver space, so the track is drawn once and each pathway is
    # marked by how far along it the system travels by 2100 and by 2300.
    axb = fig.add_subplot(gs[0, 1])
    cf = axb.contourf(SG, UG, H, levels=np.linspace(0, 1, 26), cmap="YlGnBu",
                      vmin=0, vmax=1)
    for c in cf.collections:
        c.set_rasterized(True)
    cs = axb.contour(SG, UG, H, levels=[0.2, 0.4, 0.6, 0.8], colors="0.3",
                     linewidths=0.45)
    axb.clabel(cs, fmt="%.1f", fontsize=5, inline=True)
    xr, yr_ = d["sT_ssp585"], d["U_ssp585"]
    axb.plot(xr, yr_, color="w", lw=3.0, solid_capstyle="round", zorder=4)
    axb.plot(xr, yr_, color="0.12", lw=1.4, solid_capstyle="round", zorder=5)
    for s in SSPS:
        for ymark, mk, ms in ((2100, "o", 4.4), (2300, "s", 4.0)):
            j = kof(ymark)
            axb.plot([d[f"sT_{s}"][j]], [d[f"U_{s}"][j]], marker=mk, ms=ms,
                     color=SSP_COLOR[s], mec="w", mew=0.6, zorder=8)
    axb.plot([d["sT_ssp245"][k]], [d["U_ssp245"][k]], marker="*", ms=9,
             color="k", mec="w", mew=0.5, zorder=9)
    axb.annotate("2020", xy=(d["sT_ssp245"][k], d["U_ssp245"][k]),
                 xytext=(8, -3), textcoords="offset points", fontsize=5.6)
    axb.annotate("shared direction\nof travel", xy=(0.425, 4.05),
                 xytext=(0.455, 2.25), fontsize=5.4, color="0.12", ha="left",
                 arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.12",
                                 mutation_scale=5))
    axb.set_xlabel("Thermal driver $s_T$")
    axb.set_ylabel("Ocean and carbon driver $U$")
    axb.set_xlim(0, sg.max()); axb.set_ylim(0, ug.max())
    hand = [Line2D([], [], color="0.12", lw=1.4, label="driver track")]
    hand += [Line2D([], [], color=SSP_COLOR[s], lw=0, marker="o", ms=4.0,
                    label=SSP_LABEL[s]) for s in SSPS]
    hand += [Line2D([], [], color="0.35", lw=0, marker="o", ms=4.0, label="in 2100"),
             Line2D([], [], color="0.35", lw=0, marker="s", ms=3.8, label="in 2300")]
    axb.legend(handles=hand, loc="lower right", handlelength=1.0, ncol=2,
               columnspacing=0.8, labelspacing=0.30, fontsize=5.2,
               borderaxespad=0.5)
    _panel(axb, "b", x=-0.11)
    cb = fig.colorbar(cf, ax=axb, pad=0.02, shrink=0.94, ticks=[0, 0.25, 0.5, 0.75, 1])
    cb.set_label("Habitable area fraction", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)

    # (c) habitable area against the warming reached ------------------------
    axc = fig.add_subplot(gs[1, 0])
    for s_ in SSPS:
        g, h = d[f"gmst_{s_}"], d[f"haf_{s_}"]
        sel = slice(kof(2000), None)
        axc.plot(g[sel], h[sel], color=SSP_COLOR[s_], lw=1.4, label=SSP_LABEL[s_])
        j = kof(2100)
        axc.plot([g[j]], [h[j]], marker="o", ms=3.6, color=SSP_COLOR[s_],
                 mec="w", mew=0.5, zorder=5)
    for s_ in ("ssp126", "ssp245"):
        g, h = d[f"gmst_{s_}"], d[f"haf_{s_}"]
        axc.annotate("", xy=(g[kof(2300)], h[kof(2300)]),
                     xytext=(g[kof(2120)], h[kof(2120)]),
                     arrowprops=dict(arrowstyle="-|>", lw=0.9,
                                     color=SSP_COLOR[s_], mutation_scale=6))
    axc.annotate("area keeps falling\nwhile warming is steady",
                 xy=(d["gmst_ssp245"][kof(2300)], d["haf_ssp245"][kof(2300)]),
                 xytext=(3.35, 0.30), fontsize=5.6, color="0.25",
                 arrowprops=dict(arrowstyle="-|>", lw=0.5, color="0.35",
                                 mutation_scale=5))
    axc.set_xlabel("Warming reached (K, relative to 1850 to 1900)")
    axc.set_ylabel("Habitable area fraction")
    axc.set_ylim(0, 0.95)
    axc.legend(loc="lower left", handlelength=1.4)
    _panel(axc, "c")

    # (d) how steeply the surface falls, and the substitution rate -----------
    axd = fig.add_subplot(gs[1, 1])
    xs, ys = d["sT_ssp585"][kof(2020):], d["U_ssp585"][kof(2020):]
    band = 0.15
    Hs, phi, psi = [], [], []
    for a_, b_ in zip(xs, ys):
        f = P * a_ + B + b_
        Hs.append(A[f < tau].sum() / A.sum())
        near = np.abs(f - tau) < band
        w_ = A[near]
        phi.append(w_.sum() / A.sum() / (2 * band))
        psi.append(float(np.average(P[near], weights=w_)) if w_.sum() > 0 else np.nan)
    Hs, phi, psi = np.array(Hs), np.array(phi), np.array(psi)
    k = 9
    ker = np.ones(k) / k
    phis = np.convolve(np.pad(phi, k // 2, mode="edge"), ker, mode="valid")[:len(phi)]
    axd.plot(Hs, phis, color="#1b6ca8", lw=1.5)
    axd.set_xlim(0.95, 0.0)
    axd.set_xlabel("Habitable area fraction")
    axd.set_ylabel("Land at the threshold, per unit driver", color="#1b6ca8")
    axd.tick_params(axis="y", colors="#1b6ca8")
    axd.spines["left"].set_color("#1b6ca8")
    axt = axd.twinx()
    axt.plot(Hs, psi, color="#b62025", lw=1.5)
    axt.set_ylabel("Substitution rate $\\Psi$", color="#b62025")
    axt.tick_params(axis="y", colors="#b62025")
    axt.spines["right"].set_visible(True)
    axt.spines["right"].set_color("#b62025")
    axt.spines["top"].set_visible(False)
    axd.axvline(d["haf_ssp245"][kof(2100)], color="0.6", lw=0.6, ls=(0, (3, 2)))
    axd.text(d["haf_ssp245"][kof(2100)], axd.get_ylim()[1] * 0.97,
             " SSP2-4.5 in 2100", fontsize=5.2, color="0.4", va="top")
    _panel(axd, "d")

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 -- where the hazard concentrates
# ---------------------------------------------------------------------------
def fig_hazard_map(d, path):
    lon, lat = d["lon"], d["lat"]
    chs = d["chs2100"]
    tau = float(d["tau"])
    land = d["land2d"].astype(bool)
    lat2d = np.repeat(lat[:, None], lon.size, axis=1)
    A2d = np.cos(np.deg2rad(lat2d))

    fig = plt.figure(figsize=(TWO_COL, 96 * MM))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.05, 1.0], height_ratios=[1, 1],
                          wspace=0.20, hspace=0.45,
                          left=0.01, right=0.975, top=0.97, bottom=0.10)

    axm, tr = _robinson(fig, gs[:, 0])
    kw = dict(cmap=HAZARD, shading="auto", rasterized=True,
              vmin=np.nanpercentile(chs, 1), vmax=np.nanpercentile(chs, 99.5))
    if tr is not None:
        im = axm.pcolormesh(lon, lat, chs, transform=tr, **kw)
    else:
        im = axm.pcolormesh(lon, lat, chs, **kw)
    _coast(axm)
    cb = fig.colorbar(im, ax=axm, orientation="horizontal", pad=0.03,
                      shrink=0.62, aspect=34)
    cb.set_label("Composite hazard score in 2100 (SSP2-4.5)", fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    axm.text(0.005, 0.97, "a", transform=axm.transAxes, fontsize=9,
             fontweight="bold", va="top")

    # inset: the observed surface-water hazard field that carries the geography
    insm = axm.inset_axes([0.005, 0.02, 0.27, 0.30])
    insm.pcolormesh(lon, lat, np.where(land, d["B2d"], np.nan),
                    cmap="YlGnBu", shading="auto", rasterized=True)
    insm.set_xticks([]); insm.set_yticks([])
    for sp in insm.spines.values():
        sp.set_linewidth(0.4); sp.set_color("0.5")
    insm.set_title("observed surface-water hazard", fontsize=5.0, pad=1.5)

    # (b) zonal profile
    axz = fig.add_subplot(gs[0, 1])
    wz = np.where(np.isfinite(chs), A2d, 0.0)
    num = np.nansum(np.where(np.isfinite(chs), chs * A2d, 0.0), axis=1)
    den = wz.sum(axis=1)
    prof = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)
    axz.plot(prof, lat, color="#8c1d35", lw=1.0)
    axz.axvline(tau, color="0.35", lw=0.7, ls=(0, (3, 2)))
    axz.text(tau, 86, " threshold", fontsize=5.2, color="0.35", va="top")
    axz.set_ylim(-60, 85); axz.set_yticks([-60, -30, 0, 30, 60])
    axz.set_yticklabels(["60$^\\circ$S", "30$^\\circ$S", "0",
                         "30$^\\circ$N", "60$^\\circ$N"], fontsize=5.6)
    axz.set_xlabel("Zonal mean hazard score", fontsize=6)
    _panel(axz, "b", x=-0.30)

    # (c) distribution of land by hazard score, split by era
    axh = fig.add_subplot(gs[1, 1])
    w = A2d[land] / A2d[land].sum()
    bins = np.linspace(np.nanpercentile(d["chs2020"], 0.5),
                       np.nanpercentile(chs, 99.8), 44)
    axh.hist(d["chs2020"][land], bins=bins, weights=w, color="#7fb3d5",
             alpha=0.85, label="2020")
    axh.hist(chs[land], bins=bins, weights=w, histtype="step", lw=1.0,
             color="#8c1d35", label="2100")
    axh.axvline(tau, color="0.35", lw=0.7, ls=(0, (3, 2)))
    axh.set_xlabel("Composite hazard score", fontsize=6)
    axh.set_ylabel("Share of land area", fontsize=6)
    axh.legend(loc="upper right", handlelength=1.1)
    _panel(axh, "c", x=-0.30)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 -- when the threshold is crossed
# ---------------------------------------------------------------------------
def fig_emergence(d, m, path):
    lon, lat = d["lon"], d["lat"]
    lat2d = np.repeat(lat[:, None], lon.size, axis=1)
    A2d = np.cos(np.deg2rad(lat2d))
    ALREADY, NEVER = "#151515", "#dcdcdc"

    fig = plt.figure(figsize=(TWO_COL, 150 * MM))
    gs = fig.add_gridspec(2, 1, hspace=0.06, left=0.01, right=0.99,
                          top=0.985, bottom=0.085)

    for row, (s, letter) in enumerate((("ssp245", "a"), ("ssp585", "b"))):
        toe = d[f"toe_{s}"].copy()
        ax, tr = _robinson(fig, gs[row, 0])
        cmap = _copy.copy(EMERGE)
        cmap.set_bad(NEVER)
        cmap.set_under(ALREADY)
        kw = dict(cmap=cmap, vmin=2021, vmax=2300, shading="auto",
                  rasterized=True)
        if tr is not None:
            im = ax.pcolormesh(lon, lat, toe, transform=tr, **kw)
        else:
            im = ax.pcolormesh(lon, lat, toe, **kw)
        _coast(ax)
        ax.text(0.005, 0.97, letter, transform=ax.transAxes, fontsize=9,
                fontweight="bold", va="top")
        ax.text(0.5, 0.985, SSP_LABEL[s], transform=ax.transAxes, fontsize=7.5,
                ha="center", va="top", color=SSP_COLOR[s], fontweight="bold")

        # inset: how much presently habitable land has crossed by each year
        ins = ax.inset_axes([0.035, 0.05, 0.215, 0.33])
        land = d["land2d"].astype(bool)
        yrs = np.arange(2020, 2301)
        for s2 in SSPS:
            t2 = d[f"toe_{s2}"]
            emerges = land & np.isfinite(t2) & (t2 >= 2021)
            habitable = land & (np.isnan(t2) | (t2 >= 2021))
            aw, tw = A2d[emerges], t2[emerges]
            tot = A2d[habitable].sum()
            cum = np.array([aw[tw <= y].sum() for y in yrs]) / max(tot, 1e-12)
            ins.plot(yrs, 100 * cum, lw=1.6 if s2 == s else 1.0,
                     color=SSP_COLOR[s2], alpha=1.0 if s2 == s else 0.55)
        ins.set_xlim(2020, 2300); ins.set_ylim(0, 100)
        ins.set_xticks([2050, 2150, 2250])
        ins.tick_params(labelsize=5.0, pad=1.2)
        ins.set_ylabel("% of 2020 habitable\nland crossed", fontsize=5.0,
                       labelpad=1.5)
        for sp in ins.spines.values():
            sp.set_linewidth(0.4)
        ins.patch.set_alpha(0.85)

        ax.legend(handles=[Patch(facecolor=ALREADY, label="above threshold by 2020"),
                           Patch(facecolor=NEVER, label="no crossing before 2300")],
                  loc="lower right", fontsize=5.4, handlelength=1.2,
                  borderaxespad=0.4)
        if row == 1:
            cax = fig.add_axes([0.32, 0.055, 0.38, 0.016])
            cb = fig.colorbar(im, cax=cax, orientation="horizontal", extend="min")
            cb.set_label("Year the habitability threshold is crossed", fontsize=6.5)
            cb.ax.tick_params(labelsize=5.5)
            cb.outline.set_linewidth(0.4)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5 -- independent checks
# ---------------------------------------------------------------------------
def fig_validation(d, niche_r, stat_r, path):
    yr = d["years"]
    fig = plt.figure(figsize=(TWO_COL, 120 * MM))
    gs = fig.add_gridspec(2, 2, hspace=0.46, wspace=0.30,
                          left=0.095, right=0.975, top=0.95, bottom=0.085)

    # (a) composite measure against the independent temperature niche
    axa = fig.add_subplot(gs[0, 0])
    ny = niche_r["years"]
    axa.plot(ny, niche_r["haf245"], lw=1.3, color="#1d7f47",
             label="this study")
    axa.plot(ny, niche_r["per_ssp"]["ssp245"]["niche_hab"], lw=1.3, ls=(0, (4, 2)),
             color="#6a51a3", label="temperature niche")
    axa.set_xlim(1850, 2300); axa.set_ylim(0, 1.0)
    axa.set_xlabel("Year"); axa.set_ylabel("Habitable land fraction (SSP2-4.5)")
    axa.legend(loc="lower left", handlelength=1.6)
    _panel(axa, "a")
    # lower right: the curves all sit above 0.45, so this corner is clear
    insa = axa.inset_axes([0.605, 0.085, 0.355, 0.335])
    fut = (ny >= 2000)
    insa.scatter(niche_r["per_ssp"]["ssp245"]["niche_hab"][fut],
                 np.asarray(niche_r["haf245"])[fut], s=1.6, color="0.35",
                 edgecolors="none")
    lim = [0, 1]
    insa.plot(lim, lim, color="#b62025", lw=0.6, ls=(0, (3, 2)))
    insa.set_xlim(0.2, 1.0); insa.set_ylim(0.2, 1.0)
    insa.tick_params(labelsize=5.0, pad=1.0)
    insa.set_xlabel("niche", fontsize=5.0, labelpad=1)
    insa.set_ylabel("this study", fontsize=5.0, labelpad=1)
    insa.set_title(f"$r$ = {niche_r['haf_vs_niche_corr_ssp245']:.2f}; partial "
                   f"$r$ = {niche_r['haf_vs_niche_partial_corr_given_gmst']:.2f}",
                   fontsize=5.0, pad=2)

    # (b) near-unlivable land, against the published estimate
    axb = fig.add_subplot(gs[0, 1])
    for s in SSPS:
        axb.plot(ny, 100 * niche_r["per_ssp"][s]["unlivable"], lw=1.2,
                 color=SSP_COLOR[s], label=SSP_LABEL[s])
    axb.plot([2070], [100 * niche_r["xu2020_rcp85_2070_unlivable"]], marker="*",
             ms=10, color="k", ls="none", label="published estimate, 2070")
    axb.set_xlim(1950, 2300); axb.set_ylim(0, 60)
    axb.set_xlabel("Year")
    axb.set_ylabel("Land above 29 $^\\circ$C (%)")
    axb.legend(loc="upper left", handlelength=1.4, ncol=1)
    _panel(axb, "b")

    # (c) warming pattern against the observed record
    axc = fig.add_subplot(gs[1, 0])
    el = stat_r["warm_epochs_list"]
    warm = [stat_r["warming"][e] for e in el]
    corr = [stat_r["stats"][e][0] for e in el]
    sc = axc.scatter(warm, corr, c=corr, cmap="YlGnBu", s=34, vmin=0.3, vmax=1.0,
                     edgecolors="0.3", linewidths=0.4, zorder=5)
    axc.plot(warm, corr, color="0.55", lw=0.7, zorder=4)
    for e, w_, c_ in zip(el, warm, corr):
        axc.annotate(f"{e[0]}-{e[1]}", xy=(w_, c_), xytext=(0, -9),
                     textcoords="offset points", fontsize=4.8, ha="center",
                     color="0.35")
    axc.set_xlabel("Observed warming of the epoch (K)")
    axc.set_ylabel("Correlation with the assumed pattern")
    axc.set_ylim(0.2, 1.05)
    _panel(axc, "c", x=-0.155, y=1.05)
    insc = axc.inset_axes([0.52, 0.10, 0.45, 0.42])
    insc.pcolormesh(stat_r["lon"], stat_r["lat"], stat_r["beta_ref"],
                    cmap="RdYlBu_r", vmin=0, vmax=3, shading="auto",
                    rasterized=True)
    insc.set_xticks([]); insc.set_yticks([])
    for sp in insc.spines.values():
        sp.set_linewidth(0.4); sp.set_color("0.5")
    insc.set_title("observed warming pattern", fontsize=5.0, pad=1.5)
    cbc = fig.colorbar(sc, ax=axc, pad=0.02, shrink=0.8)
    cbc.set_label("correlation", fontsize=5.6)
    cbc.ax.tick_params(labelsize=5.0)
    cbc.outline.set_linewidth(0.4)

    # (d) withheld-period test of the calibrated core
    axd = fig.add_subplot(gs[1, 1])
    cw = d["calib_window"]
    lo, mid, hi = np.percentile(d["ens_gmst"], [5, 50, 95], axis=0)
    axd.axvspan(cw[0], cw[1], color="#f0f0f0", lw=0, zorder=0)
    axd.fill_between(yr, lo, hi, color="#c6dbef", lw=0, label="model 5th to 95th")
    axd.plot(yr, mid, color="#1b6ca8", lw=1.0, label="model median")
    axd.plot(d["obs_gmst_year"], d["obs_gmst"], color="k", lw=0.7,
             label="observed temperature")
    axd.axvline(cw[1], color="0.5", lw=0.6, ls=(0, (3, 2)))
    axd.text(cw[1] + 3, 1.47, "withheld", fontsize=5.4, color="0.35")
    axd.text(cw[0] + 4, -0.42, "calibration", fontsize=5.4, color="0.45")
    axd.set_xlim(1850, 2025); axd.set_ylim(-0.5, 1.6)
    axd.set_xlabel("Year")
    axd.set_ylabel("Temperature anomaly (K)")
    axd.legend(loc="lower right", handlelength=1.4, fontsize=5.4)
    _panel(axd, "d", x=-0.135, y=1.05)
    # upper left is clear: the temperature curve is low through the 19th century
    insd = axd.inset_axes([0.055, 0.595, 0.325, 0.325])
    mo = (d["years"] >= 2000) & (d["years"] <= 2030)
    olo, omid, ohi = np.percentile(d["ens_ohc"][:, mo], [5, 50, 95], axis=0)
    insd.fill_between(d["years"][mo], olo, ohi, color="#fdd0a2", lw=0)
    insd.plot(d["years"][mo], omid, color="#e6550d", lw=0.9)
    insd.errorbar(d["obs_ohc_year"], d["obs_ohc"], yerr=d["obs_ohc_sd"], fmt="o",
                  ms=1.6, lw=0.4, color="k", capsize=0.8)
    insd.set_xlim(2003, 2023)
    insd.tick_params(labelsize=5.0, pad=1.0)
    insd.set_title("ocean heat content (ZJ)", fontsize=5.0, pad=2)

    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Supplementary figures built from the same dump
# ---------------------------------------------------------------------------
def fig_posterior(d, m, path):
    from . import emulator
    names = [str(x) for x in d["theta_names"]]
    pretty = {"ecs": "Equilibrium climate sensitivity (K)",
              "gamma": "Ocean heat-uptake coefficient (W m$^{-2}$ K$^{-1}$)"}
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(TWO_COL, 52 * MM),
                             gridspec_kw={"wspace": 0.30})
    rng = np.random.default_rng(0)
    prior = emulator.sample_prior(rng, 20000)
    w = d["post_weights"] / d["post_weights"].sum()
    for j, name in enumerate(names):
        ax = axes[j]
        ax.hist(prior[:, j], bins=50, density=True, color="0.86",
                histtype="stepfilled", label="prior")
        ax.hist(d["theta"][:, j], bins=34, density=True, weights=w,
                color="#6a51a3", alpha=0.85, histtype="stepfilled",
                label="posterior")
        q5, q50, q95 = _wquant(d["theta"][:, j], w, [0.05, 0.5, 0.95])
        ax.axvline(q50, color="k", lw=0.8)
        ax.axvspan(q5, q95, color="k", alpha=0.07, lw=0)
        ax.set_xlabel(pretty.get(name, name)); ax.set_yticks([])
        ax.set_ylabel("Probability density" if j == 0 else "")
        if j == 0:
            ax.legend(loc="upper right", handlelength=1.2)
        _panel(ax, "abc"[j], x=-0.10)
    ax = axes[-1]
    hb = ax.hexbin(d["theta"][:, 0], d["theta"][:, 1], C=w, reduce_C_function=np.sum,
                   gridsize=26, cmap="Purples", linewidths=0.1)
    ax.set_xlabel(pretty["ecs"]); ax.set_ylabel(pretty["gamma"])
    cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.9)
    cb.set_label("posterior mass", fontsize=5.6)
    cb.ax.tick_params(labelsize=5.0); cb.outline.set_linewidth(0.4)
    _panel(ax, "c", x=-0.16)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _wquant(x, w, qs):
    o = np.argsort(x)
    xs, ws = x[o], w[o]
    c = np.cumsum(ws) / ws.sum()
    return np.interp(qs, c, xs)


def fig_weighting(m, path):
    ow = m["objective_weights"]
    methods = ["equal", "critic", "entropy", "pca"]
    pretty = {"equal": "equal", "critic": "CRITIC", "entropy": "entropy",
              "pca": "leading component"}
    cvars = list(ow["weights"]["equal"].keys())
    vlabel = {"gmst": "surface\ntemperature", "co2": "atmospheric\nCO$_2$",
              "sst": "sea-surface\ntemperature", "ohc": "ocean heat\ncontent",
              "ph": "ocean pH", "omega": "aragonite\nsaturation"}
    fig, (axw, axh) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM),
                                   gridspec_kw={"width_ratios": [2.3, 1],
                                                "wspace": 0.24})
    x = np.arange(len(cvars)); bw = 0.8 / len(methods)
    pal = ["0.25", "#1b6ca8", "#e07b22", "#2e9e5b"]
    for j, mth in enumerate(methods):
        axw.bar(x + (j - 1.5) * bw, [ow["weights"][mth][k] for k in cvars], bw,
                label=pretty[mth], color=pal[j], edgecolor="none")
    axw.axhline(1 / len(cvars), color="0.6", lw=0.6, ls=(0, (2, 2)))
    axw.set_xticks(x); axw.set_xticklabels([vlabel[c] for c in cvars], fontsize=5.4)
    axw.set_ylabel("Weight in the composite score")
    axw.legend(ncol=2, handlelength=1.2, loc="upper left")
    _panel(axw, "a", x=-0.07)
    hy = [ow["haf_2100_by_method"][mth] for mth in methods]
    axh.bar(range(4), hy, 0.62, color=pal, edgecolor="none")
    for k, v in enumerate(hy):
        axh.text(k, v + 0.012, f"{v:.2f}", ha="center", fontsize=5.4)
    axh.set_xticks(range(4))
    axh.set_xticklabels([pretty[mth] for mth in methods], rotation=22, ha="right",
                        fontsize=5.6)
    axh.set_ylabel("Habitable area fraction in 2100")
    axh.set_ylim(0, 0.82)
    _panel(axh, "b", x=-0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_threshold(d, path):
    yr = d["years"]
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(TWO_COL, 62 * MM),
                                   gridspec_kw={"wspace": 0.26})
    pcts = (80, 85, 90, 95, 99)
    cols = plt.get_cmap("viridis")(np.linspace(0.12, 0.88, len(pcts)))
    for p, c in zip(pcts, cols):
        axa.plot(yr, d[f"pctile_{p}"], lw=1.1, color=c,
                 label=f"{p}th percentile")
    axa.set_xlim(1750, 2300); axa.set_ylim(0, 1.0)
    axa.set_xlabel("Year"); axa.set_ylabel("Habitable area fraction")
    axa.legend(loc="lower left", handlelength=1.4)
    _panel(axa, "a", x=-0.11)

    ref = d["pctile_90"]
    for p, c in zip(pcts, cols):
        axb.plot(yr, d[f"pctile_{p}"] - ref, lw=1.1, color=c)
    axb.axhline(0, color="0.5", lw=0.6)
    axb.set_xlim(1750, 2300)
    axb.set_xlabel("Year")
    axb.set_ylabel("Difference from the 90th-percentile threshold")
    _panel(axb, "b", x=-0.13)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_rf(rf, path):
    """Descriptive characterisation of the observed water-hazard field."""
    names = list(rf["names"])[:12][::-1]
    imp = list(rf["importance"])[:12][::-1]
    fig, (axb, axr) = plt.subplots(1, 2, figsize=(TWO_COL, 66 * MM),
                                   gridspec_kw={"width_ratios": [1.7, 1],
                                                "wspace": 0.30})
    axb.barh(range(len(names)), imp, color="#1b6ca8", edgecolor="none")
    axb.set_yticks(range(len(names)))
    axb.set_yticklabels([n.replace("_", " ") for n in names], fontsize=5.4)
    axb.set_xlabel("Permutation importance (normalised)")
    _panel(axb, "a", x=-0.36)
    r2 = [rf["cv_r2_heldout"], rf["cv_r2_with_constituents"]]
    axr.bar([0, 1], r2, 0.55, color=["#1b6ca8", "0.72"], edgecolor="none")
    for k, v in enumerate(r2):
        axr.text(k, v + 0.02, f"{v:.2f}", ha="center", fontsize=6)
    axr.set_xticks([0, 1])
    axr.set_xticklabels(["independent\npredictors only",
                         "including the field's\nown constituents"], fontsize=5.2)
    axr.set_ylabel("Held-out $R^{2}$")
    axr.set_ylim(0, 1.05)
    axr.annotate("", xy=(1, r2[1]), xytext=(0, r2[0]),
                 arrowprops=dict(arrowstyle="<->", lw=0.6, color="0.35"))
    axr.text(0.5, (r2[0] + r2[1]) / 2 + 0.05,
             f"difference {rf['leakage_r2']:.2f}", ha="center", fontsize=5.4,
             color="0.3")
    _panel(axr, "b", x=-0.26)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_structural(r, path):
    """Spread among independent surface-temperature reconstructions."""
    names, recon, gm, years = r["names"], r["recon"], r["gm"], r["years"]
    col = {"HadCRUT5": "#111111", "GISTEMP": "#1b6ca8", "Berkeley": "#b62025"}
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 64 * MM),
                                   gridspec_kw={"width_ratios": [1.65, 1],
                                                "wspace": 0.28})
    cw, ow = r["cal_window"], r["oos_window"]
    axL.axvspan(cw[0], cw[1], color="#f0f0f0", lw=0, zorder=0)
    axL.axvspan(ow[0], ow[1], color="#fdeaea", lw=0, zorder=0)
    for n in names:
        oy, og, _ = recon[n]
        held = r["per_reconstruction"][n]["held_out"]
        axL.plot(oy, og, lw=0.8, color=col.get(n, "0.4"),
                 ls=(0, (3, 1.5)) if held else "-",
                 label=n + (", withheld" if held else ", used in calibration"))
    axL.plot(years, gm, lw=1.3, color="#1d7f47", label="calibrated emulator")
    axL.set_xlim(1860, 2020); axL.set_ylim(-0.6, 1.6)
    axL.set_xlabel("Year")
    axL.set_ylabel("Temperature anomaly (K, relative to 1880 to 1920)")
    axL.legend(loc="upper left", handlelength=1.6)
    _panel(axL, "a", x=-0.10)

    x = np.arange(len(names)); bw = 0.36
    ins = [r["per_reconstruction"][n]["rmse_insample_K"] for n in names]
    oos = [r["per_reconstruction"][n]["rmse_oos_K"] for n in names]
    axR.bar(x - bw / 2, ins, bw, color="0.72", edgecolor="none",
            label="calibration window")
    axR.bar(x + bw / 2, oos, bw, color="#b62025", edgecolor="none",
            label="withheld window")
    axR.axhline(r["sigma_struct_assumed_in_smc_K"], color="#1d7f47", lw=1.0,
                ls=(0, (4, 2)))
    axR.text(len(names) - 0.4, r["sigma_struct_assumed_in_smc_K"] + 0.006,
             f"assumed {r['sigma_struct_assumed_in_smc_K']:.2f} K", fontsize=5.2,
             ha="right", color="#1d7f47")
    axR.set_xticks(x); axR.set_xticklabels(names, rotation=18, ha="right",
                                           fontsize=5.6)
    axR.set_ylabel("Temperature error (K)")
    axR.legend(loc="upper left", handlelength=1.2)
    _panel(axR, "b", x=-0.24)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_cropyield(r, path):
    """Comparison against observed crop-yield variability."""
    chs_v, cv_v, area_v = r["_chs_v"], r["_cv_v"], r["_area_v"]
    lat_v, lon_v = r["_lat_v"], r["_lon_v"]
    pl = r["pooled"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(TWO_COL, 64 * MM),
                                   gridspec_kw={"width_ratios": [1.15, 1],
                                                "wspace": 0.30})
    edges = np.percentile(chs_v, np.arange(0, 101, 10)); edges[-1] += 1e-9
    idx = np.clip(np.digitize(chs_v, edges) - 1, 0, 9)
    rng = np.random.default_rng(0)
    cen, mu_, lo_, hi_ = [], [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() < 5:
            continue
        cvb, ab, latb, lonb = cv_v[m], area_v[m], lat_v[m], lon_v[m]
        bid = (np.floor(latb / 10.0).astype(int) * 100000
               + np.floor(lonb / 10.0).astype(int))
        groups = {}
        for k, g in enumerate(bid):
            groups.setdefault(g, []).append(k)
        gidx = [np.array(v) for v in groups.values()]
        mu = np.average(cvb, weights=ab) * 100
        boot = [np.average(cvb[sel], weights=ab[sel]) * 100 for sel in
                (np.concatenate([gidx[q] for q in
                                 rng.integers(0, len(gidx), len(gidx))])
                 for _ in range(200))]
        cen.append(b + 1); mu_.append(mu)
        lo_.append(mu - np.percentile(boot, 5))
        hi_.append(np.percentile(boot, 95) - mu)
    axL.errorbar(cen, mu_, yerr=[lo_, hi_], marker="o", ms=3.2, lw=1.1,
                 color="#b62025", capsize=1.8, mec="w", mew=0.4)
    axL.set_xlabel("Composite hazard score, decile of present-day value")
    axL.set_ylabel("Detrended yield variability (%, area-weighted)")
    axL.set_xticks(range(1, 11))
    lo, hi = pl["spearman_chs_vs_cv_block_ci90"]
    axL.set_title(f"rank correlation {pl['spearman_chs_vs_cv']:.2f} "
                  f"[{lo:.2f}, {hi:.2f}]", fontsize=6)
    _panel(axL, "a", x=-0.13)

    crops = list(r["crops"])
    labels = crops + ["water hazard\nfield alone", "warming\npattern alone"]
    x = np.arange(len(labels)); bw = 0.38
    cvb = [r["per_crop"][c]["spearman_chs_vs_cv"] for c in crops] + \
          [pl["spearman_whi_alone_vs_cv"], pl["spearman_warmpattern_alone_vs_cv"]]
    myb = [r["per_crop"][c]["spearman_chs_vs_meanyield"] for c in crops] + \
          [np.nan, np.nan]
    axR.bar(x - bw / 2, cvb, bw, color="#b62025", edgecolor="none",
            label="against yield variability")
    axR.bar(x + bw / 2, myb, bw, color="#1b6ca8", edgecolor="none",
            label="against mean yield")
    axR.axhline(0, color="0.5", lw=0.6)
    axR.set_xticks(x); axR.set_xticklabels(labels, fontsize=4.8)
    axR.set_ylabel("Rank correlation with the hazard score")
    axR.legend(loc="lower left", handlelength=1.2)
    _panel(axR, "b", x=-0.20)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="publication figure layer")
    ap.add_argument("--datadir", default="outputs")
    ap.add_argument("--outdir", default=os.path.join("outputs", "paper"))
    ap.add_argument("--skip-validation", action="store_true",
                    help="omit the figure that needs the independent niche and "
                         "pattern analyses")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    d = np.load(os.path.join(a.datadir, "figdata.npz"), allow_pickle=True)
    with open(os.path.join(a.datadir, "metrics.json")) as f:
        m = json.load(f)

    out = lambda n: os.path.join(a.outdir, n)  # noqa: E731
    print("fig1 trajectories");     fig_trajectories(d, m, out("fig1_trajectories.pdf"))
    print("fig2 response surface"); fig_response_surface(d, out("fig2_response_surface.pdf"))
    print("fig3 hazard map");       fig_hazard_map(d, out("fig3_hazard_map.pdf"))
    print("fig4 emergence");        fig_emergence(d, m, out("fig4_emergence.pdf"))
    print("figS1 posterior");       fig_posterior(d, m, out("figS1_posterior.pdf"))
    print("figS2 weighting");       fig_weighting(m, out("figS2_weighting.pdf"))
    print("figS3 threshold");       fig_threshold(d, out("figS3_threshold.pdf"))
    if not a.skip_validation:
        from . import cropyield, niche, stationarity, structural, whi
        print("fig5 validation (running the independent checks)")
        fig_validation(d, niche.run(), stationarity.run(),
                       out("fig5_validation.pdf"))
        print("figS4 predictor characterisation")
        fig_rf(whi.rf_importances(), out("figS4_predictors.pdf"))
        print("figS5 reconstruction spread")
        fig_structural(structural.leave_one_out(), out("figS5_structural.pdf"))
        print("figS6 crop-yield comparison")
        fig_cropyield(cropyield.run(), out("figS6_cropyield.pdf"))
    print(f"done -> {a.outdir}")


if __name__ == "__main__":
    main()
