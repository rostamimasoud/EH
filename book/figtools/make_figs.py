#!/usr/bin/env python3
"""Render the book's data figures.

Reads the finished model output and redraws it for a book page. The science is
identical to the published work; the drawing is not. Each figure here is a new
composition at book measure, with book colours and a single panel where the
paper used a multi panel strip.

Sources, all read only:
    ../sources/EH/outputs/results.npz
    ../sources/EH/outputs/metrics.json
    ../sources/EH/eh_shallow/released/whi_field_0p5deg.npz

Usage, from EH_Book/:
    python3 figtools/make_figs.py --all
    python3 figtools/make_figs.py haf_scenarios whi_map
    python3 figtools/make_figs.py --list
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bookstyle as bs                                        # noqa: E402

import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.ticker import MultipleLocator                 # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.dirname(BOOK)

OUT = os.path.join(BOOK, "figures")
RESULTS = os.path.join(ROOT, "sources", "EH", "outputs", "results.npz")
METRICS = os.path.join(ROOT, "sources", "EH", "outputs", "metrics.json")
WHI = os.path.join(ROOT, "sources", "EH", "eh_shallow", "released",
                   "whi_field_0p5deg.npz")
EXTRA = os.path.join(BOOK, "figures", "book_extra.npz")
ARCTIC = os.path.join(os.path.dirname(ROOT), "0_paper21_Arctic_Ice", "runs")

_cache = {}


def results():
    if "r" not in _cache:
        _cache["r"] = np.load(RESULTS)
    return _cache["r"]


def metrics():
    if "m" not in _cache:
        with open(METRICS, encoding="utf-8") as fh:
            _cache["m"] = json.load(fh)
    return _cache["m"]


def whi_field():
    if "w" not in _cache:
        _cache["w"] = np.load(WHI)
    return _cache["w"]


def arctic(name):
    """One output file of the Arctic sea ice study, read only."""
    key = "arctic_" + name
    if key not in _cache:
        path = os.path.join(ARCTIC, name + ".json")
        if not os.path.exists(path):
            raise SystemExit("missing Arctic output: %s" % path)
        with open(path, encoding="utf-8") as fh:
            _cache[key] = json.load(fh)
    return _cache[key]


def extra():
    """The recomputed archive. Produced by make_extra.py; see its docstring."""
    if "x" not in _cache:
        if not os.path.exists(EXTRA):
            raise SystemExit(
                "figures/book_extra.npz is missing. Run make_extra.py first.")
        _cache["x"] = np.load(EXTRA, allow_pickle=True)
    return _cache["x"]


def _band(ax, x, ens, colour, label=None, lw=1.3):
    """Median line with a 5 to 95 per cent envelope."""
    lo = np.percentile(ens, 5, axis=0)
    hi = np.percentile(ens, 95, axis=0)
    mid = np.percentile(ens, 50, axis=0)
    ax.fill_between(x, lo, hi, color=colour, alpha=0.16, linewidth=0)
    ax.plot(x, mid, color=colour, lw=lw, label=label)
    return mid


def _map_axes(ax, lon, lat, field, cmap, vmin, vmax):
    """Plain equirectangular map with a light graticule.

    The papers used a Robinson projection through cartopy. A plate carree frame
    is used here instead: it needs no coastline download, it prints cleanly at
    book size, and it makes the latitude bands that matter for the warming
    pattern easy to read off the vertical axis.
    """
    masked = np.ma.masked_invalid(field)
    mesh = ax.pcolormesh(lon, lat, masked, cmap=cmap, vmin=vmin, vmax=vmax,
                         shading="auto", rasterized=True)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 85)
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.5)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-30, 0, 30, 60])
    ax.set_xticklabels(["120W", "60W", "0", "60E", "120E"])
    ax.set_yticklabels(["30S", "0", "30N", "60N"])
    for y in (-30, 0, 30, 60):
        ax.axhline(y, color="0.75", lw=0.25, zorder=0)
    for x in (-120, -60, 0, 60, 120):
        ax.axvline(x, color="0.75", lw=0.25, zorder=0)
    return mesh


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def haf_trajectory(path):
    """Headline trajectory with the parametric envelope. Book: one wide panel."""
    r = results()
    m = metrics()
    years, ens = r["years"], r["HAF_ens"]

    fig, ax = bs.figure(ratio=0.52)
    _band(ax, years, ens, bs.SPHERE["ocean"])

    for year, label, offset in ((1800, "preindustrial", (8, 6)),
                                (2020, "today", (8, 8)),
                                (2100, "2100", (-30, -14))):
        idx = int(np.argmin(np.abs(years - year)))
        value = np.percentile(ens[:, idx], 50)
        ax.plot([year], [value], "o", ms=3.2, color=bs.RULE, zorder=5)
        ax.annotate("%s\n%.2f" % (label, value), (year, value),
                    textcoords="offset points", xytext=offset,
                    fontsize=6.5, color=bs.RULE)

    ax.axvspan(1750, 2020, color="0.92", zorder=0, lw=0)
    ax.text(1885, 0.13, "observed forcing", fontsize=6.5, color="0.45",
            ha="center")

    ax.set_xlim(1750, 2300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("year")
    ax.set_ylabel("habitable area fraction")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    return bs.finish(fig, path)


def haf_scenarios(path):
    """The four pathways on one axis, labelled at the right margin."""
    r = results()
    years = r["years"]

    fig, ax = bs.figure(ratio=0.58)
    for key in ("ssp126", "ssp245", "ssp370", "ssp585"):
        series = r["haf_" + key]
        ax.plot(years, series, color=bs.SSP[key], lw=1.3)
        ax.annotate(bs.SSP_LABEL[key], (2300, series[-1]),
                    textcoords="offset points", xytext=(4, -2),
                    fontsize=6.5, color=bs.SSP[key], va="center")

    ax.axvline(2020, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
    ax.text(2026, 0.94, "today", fontsize=6.5, color=bs.RULE)

    ax.set_xlim(1750, 2300)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("year")
    ax.set_ylabel("habitable area fraction")
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    fig.subplots_adjust(right=0.82)
    return bs.finish(fig, path)


def loss_bars(path):
    """Land already lost against land still to lose. A book only composition."""
    m = metrics()
    c = m["committed_vs_avoidable"]
    committed = c["committed_loss_by_2020_million_km2"]
    keys = ("ssp126", "ssp245", "ssp370", "ssp585")

    fig, ax = bs.figure(ratio=0.52)
    ypos = np.arange(len(keys))[::-1]
    for i, key in zip(ypos, keys):
        addl = c["by_scenario_2100"][key]["additional_loss_2020_2100_mkm2"]
        ax.barh(i, committed, color="0.55", height=0.62,
                edgecolor="white", linewidth=0.6)
        ax.barh(i, addl, left=committed, color=bs.SSP[key], height=0.62,
                edgecolor="white", linewidth=0.6)
        ax.text(committed + addl + 1.5, i, "%.0f" % (committed + addl),
                va="center", fontsize=6.8, color=bs.RULE)

    ax.set_yticks(ypos)
    ax.set_yticklabels([bs.SSP_LABEL[k] for k in keys])
    ax.set_xlim(0, 100)
    ax.set_xlabel("habitable land lost by 2100, million square kilometres")
    ax.text(committed / 2.0, ypos[0] + 0.62, "already committed",
            ha="center", fontsize=6.5, color="0.35")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return bs.finish(fig, path)


def chs_map(path):
    """Composite hazard at 2100 on the middle pathway."""
    r = results()
    fig, ax = bs.figure(ratio=0.50)
    field = r["chs2100"].astype(float)
    finite = field[np.isfinite(field)]
    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)
    mesh = _map_axes(ax, r["lon"], r["lat"], field, bs.HAZARD, vmin, vmax)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.82,
                        pad=0.02, aspect=18)
    cbar.set_label("composite hazard", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return bs.finish(fig, path)


def whi_map(path):
    """The gridded surface water hazard field, as used by the metric."""
    w = whi_field()
    field = w["whi"].astype(float)
    field[field <= 0] = np.nan

    fig, ax = bs.figure(ratio=0.50)
    finite = field[np.isfinite(field)]
    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 98)
    mesh = _map_axes(ax, w["lon"], w["lat"], field, bs.HAZARD, vmin, vmax)
    cbar = fig.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.82,
                        pad=0.02, aspect=18)
    cbar.set_label("water hazard", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return bs.finish(fig, path)


def weights(path):
    """How four principled schemes distribute weight, and what it does to the
    answer. Horizontal bars: the variable names are too long to sit under a
    narrow book column without colliding."""
    m = metrics()["objective_weights"]
    order = ["gmst", "co2", "sst", "ohc", "ph", "omega"]
    names = ["surface temperature", "carbon dioxide", "sea surface temperature",
             "ocean heat content", "ocean pH", "aragonite saturation"]
    schemes = ["equal", "critic", "entropy", "pca"]
    labels = ["equal", "CRITIC", "entropy", "principal component"]
    colours = ["0.62", bs.SPHERE["ocean"], bs.SPHERE["atmosphere"],
               bs.SPHERE["hydrosphere"]]

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.92),
        gridspec_kw={"height_ratios": [2.6, 1.0], "hspace": 0.42})

    y = np.arange(len(order))[::-1]
    height = 0.19
    for j, (scheme, colour) in enumerate(zip(schemes, colours)):
        vals = [m["weights"][scheme][k] for k in order]
        ax.barh(y + (1.5 - j) * height, vals, height * 0.92, color=colour,
                label=labels[j], edgecolor="none")
    ax.axvline(1 / 6.0, color=bs.RULE, lw=0.6, ls=(0, (2, 2)))
    ax.text(1 / 6.0 + 0.006, y[0] + 0.44, "equal share", fontsize=6,
            color=bs.RULE)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.6)
    ax.set_xlabel("weight")
    ax.set_xlim(0, 0.44)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(fontsize=6.2, loc="lower right", ncol=2, handlelength=1.0,
              columnspacing=1.0)

    haf = [m["haf_2100_by_method"][s] for s in schemes]
    yb = np.arange(4)[::-1]
    bx.barh(yb, haf, 0.6, color=colours, edgecolor="none")
    for yy, v in zip(yb, haf):
        bx.text(v + 0.004, yy, "%.3f" % v, va="center", fontsize=6.4,
                color=bs.RULE)
    bx.set_yticks(yb)
    bx.set_yticklabels(labels, fontsize=6.6)
    bx.set_xlim(0.55, 0.72)
    bx.set_xlabel("habitable area fraction at 2100")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    return bs.finish(fig, path)


def gmst_fit(path):
    """Calibration against surface temperature, with the withheld window."""
    r, m = results(), metrics()
    years, ens = r["years"], r["T_ens"]
    hist = (years >= 1850) & (years <= 2020)

    fig, ax = bs.figure(ratio=0.52)
    ax.axvspan(1850, 1980, color="0.92", lw=0, zorder=0)
    _band(ax, years[hist], ens[:, hist], bs.SPHERE["atmosphere"])
    ax.axvline(1980, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
    ax.text(1860, 1.35, "calibrated here", fontsize=6.5, color="0.4")
    ax.text(1988, 1.35, "withheld", fontsize=6.5, color=bs.RULE)
    ax.set_xlim(1850, 2020)
    ax.set_xlabel("year")
    ax.set_ylabel("warming since 1850 to 1900 (K)")
    return bs.finish(fig, path)


def ohc_fit(path):
    """Calibration against ocean heat content."""
    r = results()
    years, ens = r["years"], r["OHC_ens"]
    win = (years >= 1960) & (years <= 2020)

    fig, ax = bs.figure(ratio=0.52)
    _band(ax, years[win], ens[:, win], bs.SPHERE["ocean"])
    ax.set_xlim(1960, 2020)
    ax.set_xlabel("year")
    ax.set_ylabel("ocean heat content (ZJ)")
    return bs.finish(fig, path)



# ---------------------------------------------------------------------------
# Figures built from the recomputed archive
# ---------------------------------------------------------------------------

def _paths_axes(ax, years, getter, xlim=(1750, 2300), label_x=2300):
    for key in ("ssp126", "ssp245", "ssp370", "ssp585"):
        y = getter(key)
        ax.plot(years, y, color=bs.SSP[key], lw=1.2)
        ax.annotate(bs.SSP_LABEL[key], (label_x, y[-1]),
                    textcoords="offset points", xytext=(4, -2),
                    fontsize=6.3, color=bs.SSP[key], va="center")
    ax.axvline(2020, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
    ax.set_xlim(*xlim)
    ax.set_xlabel("year")


def pattern_map(path):
    """The land warming pattern: how much faster each place warms."""
    x = extra()
    field = x["pattern"].astype(float)
    fig, ax = bs.figure(ratio=0.50)
    finite = field[np.isfinite(field)]
    mesh = _map_axes(ax, x["lon"], x["lat"], field, "YlOrBr",
                     float(np.nanmin(finite)), float(np.nanmax(finite)))
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02, aspect=18)
    cbar.set_label("warming relative to the global mean", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return bs.finish(fig, path)


def gmst_paths(path):
    """Surface warming under each pathway."""
    x = extra()
    years = x["years"]
    fig, ax = bs.figure(ratio=0.55)
    _paths_axes(ax, years, lambda k: x["gmst_" + k])
    ax.set_ylabel("warming (K)")
    fig.subplots_adjust(right=0.82)
    return bs.finish(fig, path)


def co2_paths(path):
    """Atmospheric carbon dioxide under each pathway."""
    x = extra()
    fig, ax = bs.figure(ratio=0.55)
    _paths_axes(ax, x["years"], lambda k: x["co2_" + k])
    ax.set_ylabel("carbon dioxide (parts per million)")
    fig.subplots_adjust(right=0.82)
    return bs.finish(fig, path)


def ocean_heat_paths(path):
    """Heat accumulating in the upper ocean under each pathway."""
    x = extra()
    fig, ax = bs.figure(ratio=0.55)
    _paths_axes(ax, x["years"], lambda k: x["ohc_" + k])
    ax.set_ylabel("ocean heat content (ZJ)")
    fig.subplots_adjust(right=0.82)
    return bs.finish(fig, path)


def carbonate_paths(path):
    """Surface ocean chemistry: acidity and aragonite saturation."""
    x = extra()
    years = x["years"]
    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.80), sharex=True,
        gridspec_kw={"hspace": 0.18})

    for key in ("ssp126", "ssp245", "ssp370", "ssp585"):
        ax.plot(years, x["ph_" + key], color=bs.SSP[key], lw=1.2)
        bx.plot(years, x["omega_" + key], color=bs.SSP[key], lw=1.2)
        bx.annotate(bs.SSP_LABEL[key], (2300, x["omega_" + key][-1]),
                    textcoords="offset points", xytext=(4, -2),
                    fontsize=6.3, color=bs.SSP[key], va="center")

    bx.axhline(1.0, color=bs.RULE, lw=0.7, ls=(0, (3, 2)))
    bx.text(1770, 1.12, "aragonite dissolves below this line",
            fontsize=6.3, color=bs.RULE)
    for a in (ax, bx):
        a.axvline(2020, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
        a.set_xlim(1750, 2300)
    ax.set_ylabel("surface ocean pH")
    bx.set_ylabel("aragonite saturation")
    bx.set_xlabel("year")
    fig.subplots_adjust(right=0.82)
    return bs.finish(fig, path)


def whi_distribution(path):
    """How water hazard is distributed over land. The tail is the point."""
    x = extra()
    raw = x["whi_raw"].astype(float)
    land = x["land2d"].astype(bool)
    vals = raw[land & np.isfinite(raw)]

    fig, ax = bs.figure(ratio=0.50)
    ax.hist(vals, bins=60, color=bs.SPHERE["hydrosphere"], alpha=0.85,
            edgecolor="white", linewidth=0.2)
    med = float(np.median(vals))
    p95 = float(np.percentile(vals, 95))
    ax.axvline(med, color=bs.RULE, lw=0.9)
    ax.axvline(p95, color="#A8402F", lw=0.9, ls=(0, (3, 2)))
    ax.annotate("median", (med, ax.get_ylim()[1] * 0.92),
                textcoords="offset points", xytext=(4, 0), fontsize=6.5,
                color=bs.RULE)
    ax.annotate("95th percentile", (p95, ax.get_ylim()[1] * 0.75),
                textcoords="offset points", xytext=(4, 0), fontsize=6.5,
                color="#A8402F")
    ax.set_xlabel("water hazard index")
    ax.set_ylabel("number of land cells")
    return bs.finish(fig, path)


def whi_zonal(path):
    """Water hazard against latitude, with the spread inside each band.

    The point of the figure is that the spread within a latitude band is far
    larger than the variation between bands, which is why a zonal or global
    summary of water hazard conveys almost nothing.
    """
    x = extra()
    raw = x["whi_raw"].astype(float)
    land = x["land2d"].astype(bool)
    lat = x["lat"]

    edges = np.arange(-60, 86, 5)
    mids, med, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (lat >= a) & (lat < b)
        if not sel.any():
            continue
        block = raw[sel][land[sel]]
        block = block[np.isfinite(block)]
        if block.size < 20:
            continue
        mids.append(0.5 * (a + b))
        med.append(np.median(block))
        lo.append(np.percentile(block, 10))
        hi.append(np.percentile(block, 90))

    fig, ax = bs.figure(ratio=0.62)
    ax.fill_betweenx(mids, lo, hi, color=bs.SPHERE["hydrosphere"], alpha=0.22,
                     linewidth=0)
    ax.plot(med, mids, color=bs.SPHERE["hydrosphere"], lw=1.3)
    ax.set_ylabel("latitude")
    ax.set_xlabel("water hazard index")
    ax.set_ylim(-60, 85)
    ax.axhline(0, color="0.8", lw=0.4)
    ax.text(0.97, 0.03, "shading spans the 10th to 90th percentile\nwithin each"
            " band", transform=ax.transAxes, fontsize=6.2, color="0.4",
            ha="right", va="bottom")
    return bs.finish(fig, path)


def whi_predictors(path):
    """Which independent predictors explain the water hazard field, and how
    poorly. The low skill is the result worth showing."""
    x = extra()
    imp = json.loads(str(x["whi_importance_json"]))["rf"]
    names = imp["names"]
    vals = np.asarray(imp["importance"], dtype=float)
    keep = vals > 0.001
    names = [n.replace("_", " ") for n, k in zip(names, keep) if k]
    vals = vals[keep]
    order = np.argsort(vals)

    fig, ax = bs.figure(ratio=0.55)
    ypos = np.arange(len(vals))
    ax.barh(ypos, vals[order], 0.66, color=bs.SPHERE["hydrosphere"])
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.2)
    ax.set_xlabel("share of the explained variation")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.97, 0.06,
            "held out skill on independent predictors: %.2f\n"
            "with the field's own constituents: %.2f"
            % (imp["cv_r2_heldout"], imp["cv_r2_with_constituents"]),
            transform=ax.transAxes, fontsize=6.2, color=bs.RULE, ha="right")
    return bs.finish(fig, path)


def twobox_response(path):
    """The calibrated model's response to a sudden doubling of carbon dioxide.

    Integrated here with the calibrated parameters, to show the two timescales
    that make the ocean the place where commitment lives.
    """
    x = extra()
    ecs, gamma = [float(v) for v in x["theta_mean"]]
    c_s, c_d, f2x = 7.3, 106.0, 3.93
    lam = f2x / ecs

    n = 600
    t = np.arange(n)
    ts, td = np.zeros(n), np.zeros(n)
    for i in range(1, n):
        dts = (f2x - lam * ts[i - 1] - gamma * (ts[i - 1] - td[i - 1])) / c_s
        dtd = gamma * (ts[i - 1] - td[i - 1]) / c_d
        ts[i] = ts[i - 1] + dts
        td[i] = td[i - 1] + dtd

    fig, ax = bs.figure(ratio=0.52)
    ax.plot(t, ts, color=bs.SPHERE["atmosphere"], lw=1.4, label="surface")
    ax.plot(t, td, color=bs.SPHERE["ocean"], lw=1.4, label="deep ocean")
    ax.axhline(ecs, color=bs.RULE, lw=0.7, ls=(0, (3, 2)))
    ax.text(320, ecs + 0.08, "eventual warming", fontsize=6.5, color=bs.RULE)

    i10 = int(np.argmin(np.abs(ts - 0.63 * ecs)))
    ax.annotate("most of the surface response\narrives within decades",
                (i10, ts[i10]), textcoords="offset points", xytext=(26, -22),
                fontsize=6.3, color=bs.RULE,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=bs.RULE))

    ax.set_xlim(0, n)
    ax.set_ylim(0, ecs * 1.12)
    ax.set_xlabel("years after the forcing is applied")
    ax.set_ylabel("warming (K)")
    ax.legend(loc="lower right", fontsize=6.5)
    return bs.finish(fig, path)


def posterior_marginals(path):
    """What calibration did to each parameter: prior against posterior."""
    x = extra()
    post = x["post_particles"]
    w = x["post_weights"] / x["post_weights"].sum()
    prior = x["prior_particles"]
    labels = ["equilibrium climate sensitivity (K)",
              "ocean heat uptake (W m$^{-2}$ K$^{-1}$)"]
    colours = [bs.SPHERE["atmosphere"], bs.SPHERE["ocean"]]

    fig, axes = plt.subplots(
        1, 2, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.40),
        gridspec_kw={"wspace": 0.30})

    for j, (ax, lab, col) in enumerate(zip(axes, labels, colours)):
        lo = min(prior[:, j].min(), post[:, j].min())
        hi = max(prior[:, j].max(), post[:, j].max())
        bins = np.linspace(lo, hi, 46)
        ax.hist(prior[:, j], bins=bins, density=True, color="0.78",
                edgecolor="none", label="prior")
        ax.hist(post[:, j], bins=bins, weights=w, density=True, color=col,
                alpha=0.82, edgecolor="none", label="posterior")
        ax.set_xlabel(lab, fontsize=7)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        if j == 0:
            ax.legend(fontsize=6.3, loc="upper right")
    axes[0].set_ylabel("density", fontsize=7)
    return bs.finish(fig, path)


def posterior_joint(path):
    """The two parameters together, showing what the ocean constraint did.

    Surface temperature alone cannot separate a sensitive climate that takes up
    heat quickly from an insensitive one that does not. The joint cloud is what
    breaking that trade off looks like.
    """
    x = extra()
    post = x["post_particles"]
    w = x["post_weights"] / x["post_weights"].sum()
    prior = x["prior_particles"]

    fig, ax = bs.figure(ratio=0.66)
    ax.scatter(prior[::12, 0], prior[::12, 1], s=1.2, color="0.84",
               edgecolors="none", label="prior", rasterized=True)
    size = 4.0 + 900.0 * w
    ax.scatter(post[:, 0], post[:, 1], s=size, color=bs.SPHERE["ocean"],
               alpha=0.62, edgecolors="none", label="posterior")
    ax.set_xlabel("equilibrium climate sensitivity (K)")
    ax.set_ylabel("ocean heat uptake (W m$^{-2}$ K$^{-1}$)")
    ax.legend(fontsize=6.5, loc="upper left")
    ax.text(0.97, 0.04, "marker size is the particle weight",
            transform=ax.transAxes, fontsize=6.2, color="0.45", ha="right")
    return bs.finish(fig, path)


def threshold_sensitivity(path):
    """Does the answer depend on where the threshold is put?"""
    x = extra()
    years = x["years"]
    sens = np.atleast_2d(x["sens_haf"])
    pcts = x["sens_pct"]
    shades = ["#DCE6EC", "#B4C9D6", "#7FA4B8", "#4A7C95", "#1F4E63"]

    fig, ax = bs.figure(ratio=0.55)
    for row, pct, col in zip(sens, pcts, shades):
        ax.plot(years, row, color=col, lw=1.2)
        ax.annotate("%dth" % int(pct), (2300, row[-1]),
                    textcoords="offset points", xytext=(4, -2), fontsize=6.3,
                    color=col, va="center")
    ax.axvline(2020, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
    ax.set_xlim(1750, 2300)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("year")
    ax.set_ylabel("habitable area fraction")
    ax.text(0.02, 0.06, "each curve puts the threshold at a different\n"
            "percentile of the preindustrial field",
            transform=ax.transAxes, fontsize=6.2, color="0.4")
    fig.subplots_adjust(right=0.84)
    return bs.finish(fig, path)


def chs_evolution(path):
    """The composite hazard field at four dates on the middle pathway."""
    x = extra()
    dates = (1850, 2020, 2100, 2300)
    fields = [x["chs_ssp245_%d" % y].astype(float) for y in dates]
    allv = np.concatenate([f[np.isfinite(f)] for f in fields])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)

    fig, axes = plt.subplots(
        4, 1, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 1.34),
        gridspec_kw={"hspace": 0.16})
    for ax, year, field in zip(axes, dates, fields):
        mesh = _map_axes(ax, x["lon"], x["lat"], field, bs.HAZARD, vmin, vmax)
        ax.set_title(str(year), fontsize=7.5, loc="left", pad=2)
        if ax is not axes[-1]:
            ax.set_xticklabels([])
            ax.set_xlabel("")
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.55, pad=0.02, aspect=26)
    cbar.set_label("composite hazard", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return bs.finish(fig, path)


def water_contribution(path):
    """What the water hazard term supplies, shown by removing it.

    Upper panel: the composite field at 2100 with the surface water hazard term
    switched off, leaving climate and ocean alone. Lower panel: the full field.
    The difference is the geography the hydrosphere contributes.
    """
    x = extra()
    without = x["chs_nowater_2100"].astype(float)
    withw = x["chs_ssp245_2100"].astype(float)
    allv = np.concatenate([without[np.isfinite(without)],
                           withw[np.isfinite(withw)]])
    vmin, vmax = np.percentile(allv, 1), np.percentile(allv, 99)

    fig, axes = plt.subplots(
        2, 1, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.76),
        gridspec_kw={"hspace": 0.20})
    titles = ["climate and ocean only", "with the surface water hazard field"]
    for ax, field, title in zip(axes, (without, withw), titles):
        mesh = _map_axes(ax, x["lon"], x["lat"], field, bs.HAZARD, vmin, vmax)
        ax.set_title(title, fontsize=7.5, loc="left", pad=2)
    axes[0].set_xticklabels([])
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.7, pad=0.02, aspect=22)
    cbar.set_label("composite hazard", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    return bs.finish(fig, path)


def _toe_map(path, key, label):
    x = extra()
    toe = x[key].astype(float)
    sentinel = 2019.0
    already = np.isclose(toe, sentinel)
    future = toe > sentinel

    fig, ax = bs.figure(ratio=0.52)
    shown = np.where(future, toe, np.nan)
    mesh = _map_axes(ax, x["lon"], x["lat"], shown, bs.WHEN, 2020, 2300)
    already_field = np.where(already, 1.0, np.nan)
    ax.pcolormesh(x["lon"], x["lat"], np.ma.masked_invalid(already_field),
                  cmap=plt.matplotlib.colors.ListedColormap(["#241016"]),
                  shading="auto", rasterized=True)
    land = x["land2d"].astype(bool)
    never = np.where(land & ~already & ~future, 1.0, np.nan)
    ax.pcolormesh(x["lon"], x["lat"], np.ma.masked_invalid(never),
                  cmap=plt.matplotlib.colors.ListedColormap(["#C9C9C4"]),
                  shading="auto", rasterized=True)

    cbar = fig.colorbar(mesh, ax=ax, shrink=0.82, pad=0.02, aspect=18)
    cbar.set_label("year of crossing", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5)
    cbar.outline.set_linewidth(0.5)
    ax.set_title(label, fontsize=7.5, loc="left", pad=2)
    return bs.finish(fig, path)


def toe_map_middle(path):
    """When each place crosses, on the middle pathway."""
    return _toe_map(path, "toe_ssp245", bs.SSP_LABEL["ssp245"])


def toe_map_high(path):
    """When each place crosses, on the high emissions pathway."""
    return _toe_map(path, "toe_ssp585", bs.SSP_LABEL["ssp585"])


def toe_distribution(path):
    """How the crossings are spread through time under two pathways."""
    x = extra()
    area = None
    fig, ax = bs.figure(ratio=0.50)
    for key, ssp in (("toe_ssp245", "ssp245"), ("toe_ssp585", "ssp585")):
        toe = x[key].astype(float)
        vals = toe[np.isfinite(toe) & (toe > 2019.5)]
        ax.hist(vals, bins=np.arange(2020, 2310, 10), histtype="step",
                color=bs.SSP[ssp], lw=1.3, label=bs.SSP_LABEL[ssp])
    ax.set_xlabel("decade of crossing")
    ax.set_ylabel("land cells crossing")
    ax.set_xlim(2020, 2300)
    ax.legend(fontsize=6.5)
    return bs.finish(fig, path)


def uncertainty_bars(path):
    """The three sources of spread, on one axis. The comparison is the point."""
    m = metrics()
    we = m["weight_ensemble"]
    values = [m["scenario_spread_haf_2100"],
              we["haf_2100_width_weights"],
              we["haf_2100_width_param_posterior"]]
    labels = ["emissions\npathway", "how the spheres\nare weighted",
              "calibrated\nparameters"]
    colours = ["#A8402F", bs.SPHERE["hydrosphere"], bs.SPHERE["ocean"]]

    fig, ax = bs.figure(ratio=0.50)
    ypos = np.arange(3)[::-1]
    ax.barh(ypos, values, 0.55, color=colours)
    for y, v in zip(ypos, values):
        ax.text(v + 0.008, y, "%.3f" % v, va="center", fontsize=7,
                color=bs.RULE)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 0.47)
    ax.set_xlabel("spread in the habitable area fraction at 2100")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return bs.finish(fig, path)


def tier_decomposition(path):
    """The two parts of the composite signal, and how they grow apart."""
    x = extra()
    years = x["years"]
    fig, ax = bs.figure(ratio=0.52)
    for key in ("ssp126", "ssp585"):
        ax.plot(years, x["tierT_" + key], color=bs.SSP[key], lw=1.3)
        ax.plot(years, x["tierU_" + key], color=bs.SSP[key], lw=1.0,
                ls=(0, (3, 2)))
    ax.axvline(2020, color=bs.RULE, lw=0.5, ls=(0, (4, 3)))
    ax.set_xlim(1750, 2300)
    ax.set_xlabel("year")
    ax.set_ylabel("standardised contribution")
    ax.text(0.02, 0.92, "solid: the patterned surface warming term\n"
            "dashed: the spatially uniform ocean and carbon term",
            transform=ax.transAxes, fontsize=6.3, color="0.4", va="top")
    for key in ("ssp126", "ssp585"):
        ax.annotate(bs.SSP_LABEL[key], (2300, x["tierU_" + key][-1]),
                    textcoords="offset points", xytext=(4, 0), fontsize=6.3,
                    color=bs.SSP[key], va="center")
    fig.subplots_adjust(right=0.84)
    return bs.finish(fig, path)


ZOOMS = [
    ("Western North America", -125, -113, 31, 42),
    ("Indo Gangetic plain", 67, 92, 20, 34),
    ("North China Plain", 110, 123, 30, 42),
    ("Middle East and the Nile", 25, 60, 12, 38),
]


def whi_map_full(path):
    """The water hazard field at full page size, with regional detail.

    A full page is worth spending on this field because it is the only component
    of the composite measure that carries real geography, and because the
    regional structure is invisible at the size of an ordinary text figure.
    """
    x = extra()
    raw = x["whi_raw"].astype(float)
    land = x["land2d"].astype(bool)
    field = np.where(land & np.isfinite(raw), raw, np.nan)
    lon, lat = x["lon"], x["lat"]
    finite = field[np.isfinite(field)]
    vmin, vmax = np.percentile(finite, 2), np.percentile(finite, 99)

    fig = plt.figure(figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 1.49))
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.28, 1.0, 1.0], hspace=0.26, wspace=0.16,
        left=0.10, right=0.90, top=0.985, bottom=0.085)

    ax = fig.add_subplot(gs[0, :])
    mesh = _map_axes(ax, lon, lat, field, bs.HAZARD, vmin, vmax)
    ax.set_title("the global field", fontsize=8, loc="left", pad=3)

    for name, x0, x1, y0, y1 in ZOOMS:
        ax.add_patch(plt.Rectangle(
            (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#1F3A4D",
            linewidth=0.7, zorder=6))

    ilon = (lon >= -180) & (lon <= 180)
    for k, (name, x0, x1, y0, y1) in enumerate(ZOOMS):
        bx = fig.add_subplot(gs[1 + k // 2, k % 2])
        sx = (lon >= x0) & (lon <= x1)
        sy = (lat >= y0) & (lat <= y1)
        sub = field[np.ix_(sy, sx)]
        bx.pcolormesh(lon[sx], lat[sy], np.ma.masked_invalid(sub),
                      cmap=bs.HAZARD, vmin=vmin, vmax=vmax, shading="auto",
                      rasterized=True)
        bx.set_aspect("equal")
        bx.set_title(name, fontsize=7, loc="left", pad=2)
        bx.tick_params(labelsize=5.8, width=0.4)
        for spine in bx.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.5)

    cax = fig.add_axes([0.24, 0.045, 0.52, 0.012])
    cbar = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cbar.set_label("water hazard index", fontsize=7)
    cbar.ax.tick_params(labelsize=6.3, width=0.5)
    cbar.outline.set_linewidth(0.5)
    fig.savefig(path)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Cryosphere: Antarctic ice shelf cavity thresholds
#
# Values are the published continuation results of the companion study on
# bistability of Antarctic ice shelf cavities. Columns are the two folds of the
# bistable window in sea ice production (metres per year), and the jump in
# basal melt rate across the transition (metres per year).
# ---------------------------------------------------------------------------

CAVITIES = [
    # name,            lower fold, upper fold, melt jump
    ("Ross",                 1.67, 29.13,  6.09),
    ("Getz",                 0.42, 25.38,  8.60),
    ("Pine Island",          1.04, 20.21, 10.59),
    ("Riiser Larsen",        1.38, 19.66,  4.46),
    ("Fimbul",               1.62, 19.55,  4.12),
    ("Amery",                2.04, 13.41,  3.92),
    ("Shackleton",           0.54, 13.18,  5.83),
    ("Larsen C",             1.50, 10.26,  2.82),
    ("Totten",               0.17,  6.41,  7.43),
]


def cryo_windows(path):
    """The bistable window and the melt jump for each Antarctic cavity.

    Composed for this book. The published figure places a derived forcing point
    inside each window; this one ranks the cavities by the width of the window
    and sets the size of the melt jump beside it, because the argument the book
    makes concerns how much room there is and how far the state moves when the
    room runs out.
    """
    rows = sorted(CAVITIES, key=lambda r: r[2] - r[1])
    names = [r[0] for r in rows]
    lo = np.array([r[1] for r in rows])
    hi = np.array([r[2] for r in rows])
    jump = np.array([r[3] for r in rows])
    y = np.arange(len(rows))

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.58),
        gridspec_kw={"width_ratios": [2.4, 1.0], "wspace": 0.08})

    for i, (a, b) in enumerate(zip(lo, hi)):
        ax.plot([a, b], [i, i], color=bs.SPHERE["ocean"], lw=3.2,
                solid_capstyle="butt", alpha=0.75)
        ax.plot([a], [i], "o", ms=3.0, color=bs.RULE, zorder=4)
        ax.plot([b], [i], "o", ms=3.0, color="#A8402F", zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=6.8)
    ax.set_xlim(0, 31)
    ax.set_xlabel("sea ice production at the two folds (m per year)")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.text(0.98, 0.04, "present day forcing lies inside\nevery window",
            transform=ax.transAxes, fontsize=6.2, color="0.4", ha="right")

    bx.barh(y, jump, 0.6, color=bs.SPHERE["hydrosphere"], edgecolor="none")
    for i, v in enumerate(jump):
        bx.text(v + 0.25, i, "%.1f" % v, va="center", fontsize=6.2,
                color=bs.RULE)
    bx.set_yticks(y)
    bx.set_yticklabels([])
    bx.set_xlim(0, 13.5)
    bx.set_xlabel("melt jump (m per year)")
    bx.spines["left"].set_visible(False)
    bx.tick_params(axis="y", length=0)
    return bs.finish(fig, path)


def cryo_width_vs_jump(path):
    """Window width against the size of the jump, one point per cavity.

    A composition that does not appear in the source study. It asks whether the
    cavities with the most room to spare are also the ones that move least when
    they go, and the answer is that they are not.
    """
    width = np.array([r[2] - r[1] for r in CAVITIES])
    jump = np.array([r[3] for r in CAVITIES])
    names = [r[0] for r in CAVITIES]

    fig, ax = bs.figure(ratio=0.62)
    ax.scatter(width, jump, s=34, color=bs.SPHERE["ocean"], alpha=0.8,
               edgecolors="white", linewidths=0.6, zorder=3)
    for x0, y0, n in zip(width, jump, names):
        ax.annotate(n, (x0, y0), textcoords="offset points", xytext=(5, 3),
                    fontsize=6.0, color=bs.RULE)
    ax.set_xlabel("width of the bistable window (m per year)")
    ax.set_ylabel("melt jump across the transition (m per year)")
    ax.set_xlim(3, 31)
    ax.set_ylim(1.5, 12.5)
    ax.text(0.97, 0.95, "no relationship: a wide margin does not\n"
            "mean a small consequence", transform=ax.transAxes,
            fontsize=6.2, color="0.4", ha="right", va="top")
    return bs.finish(fig, path)


# ---------------------------------------------------------------------------
# Arctic sea ice: composed from the companion study's model output
# ---------------------------------------------------------------------------

def arctic_parameter(path):
    """Everything hangs on one unmeasured number.

    A composition made for this book. The source study reports the branches and
    the forcing budget in separate figures; putting them on a shared horizontal
    axis is what shows that the same parameter which decides whether a threshold
    exists also decides whether it has already been passed.
    """
    b = arctic("budget")
    rows = [e for e in b["entries"]]
    h = np.array([e["albedo_scale"] for e in rows])
    frac = np.array([e.get("fraction_applied", np.nan) for e in rows], dtype=float)
    lo = np.array([e.get("fraction_low", np.nan) for e in rows], dtype=float)
    hi = np.array([e.get("fraction_high", np.nan) for e in rows], dtype=float)

    bif = arctic("bifurcation")
    width, jump, hb = [], [], []
    for br in bif["branches"]:
        hy = br.get("hysteresis") or {}
        w = hy.get("width")
        j = hy.get("jump")
        if w is not None and j is not None:
            hb.append(br["albedo_scale"]); width.append(w); jump.append(j)
    hb = np.array(hb); width = np.array(width); jump = np.array(jump)

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.86), sharex=True,
        gridspec_kw={"hspace": 0.14})

    ax.plot(hb, width, "o-", ms=3.4, lw=1.3, color=bs.SPHERE["atmosphere"],
            label="width of the bistable window")
    ax2 = ax.twinx()
    ax2.plot(hb, jump, "s--", ms=3.0, lw=1.1, color="#A8402F",
             label="drop in late summer thickness")
    ax2.set_ylabel("thickness drop (m)", fontsize=7, color="#A8402F")
    ax2.tick_params(axis="y", labelsize=6.5, colors="#A8402F", width=0.5)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#A8402F")
    ax2.spines["right"].set_linewidth(0.6)
    ax.set_ylabel("window (W m$^{-2}$)", fontsize=7)
    ax.set_ylim(0, 2.3)
    ax2.set_ylim(0, 1.95)
    ax.axvspan(0.26, 0.55, color="0.92", zorder=0, lw=0)
    ax.text(0.275, 2.02, "no threshold here", fontsize=6.4, color="0.4")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=6.3, loc="upper left",
              bbox_to_anchor=(0.0, 0.86))

    ok = np.isfinite(frac)
    bx.fill_between(h[ok], 100 * lo[ok], 100 * hi[ok],
                    color=bs.SPHERE["ocean"], alpha=0.18, linewidth=0)
    bx.plot(h[ok], 100 * frac[ok], "o-", ms=3.4, lw=1.3,
            color=bs.SPHERE["ocean"])
    bx.axhline(100, color=bs.RULE, lw=0.7, ls=(0, (3, 2)))
    bx.text(0.22, 101.5, "all of it already applied", fontsize=6.4,
            color=bs.RULE)
    bx.set_ylim(0, 118)
    bx.set_ylabel("share of the threshold forcing\nalready applied (per cent)",
                  fontsize=7)
    bx.set_xlabel("thickness over which the surface darkens as the ice thins (m)")
    bx.axvspan(0.26, 0.55, color="0.92", zorder=0, lw=0)
    return bs.finish(fig, path)


def arctic_area_mass(path):
    """Mass goes faster than area, so satellite extent understates the loss."""
    r = arctic("retreat")
    st = sorted(r["states"], key=lambda s: s["forcing"])
    f = np.array([s["forcing"] for s in st])
    ext = np.array([s["extent_min"] for s in st])
    vol = np.array([s["volume_min"] for s in st])
    ext0, vol0 = ext[0], vol[0]

    fig, ax = bs.figure(ratio=0.56)
    ax.plot(f, 100 * ext / ext0, "o-", ms=3.6, lw=1.4,
            color=bs.SPHERE["atmosphere"], label="area")
    ax.plot(f, 100 * vol / vol0, "s-", ms=3.4, lw=1.4,
            color=bs.SPHERE["ocean"], label="mass")

    i0 = int(np.argmin(np.abs(f - 0.0)))
    ax.axvline(0, color=bs.RULE, lw=0.6, ls=(0, (4, 3)))
    ax.annotate("today", (0, 8), textcoords="offset points", xytext=(5, 0),
                fontsize=6.5, color=bs.RULE)
    ax.plot([0, 0], [100 * vol[i0] / vol0, 100 * ext[i0] / ext0],
            color="#A8402F", lw=2.4, solid_capstyle="butt", zorder=5)
    ax.annotate("a fifth of the area gone,\nnearly a third of the substance",
                (0, 100 * vol[i0] / vol0), textcoords="offset points",
                xytext=(14, -6), fontsize=6.4, color="#A8402F")

    ax.set_xlabel("greenhouse forcing relative to today (W m$^{-2}$)")
    ax.set_ylabel("September ice, per cent of preindustrial")
    ax.set_xlim(-2.6, 8.4)
    ax.set_ylim(0, 108)
    ax.legend(fontsize=6.8, loc="upper right")
    return bs.finish(fig, path)


def arctic_synergy(path):
    """Where the threshold sits depends on the Atlantic inflow as well.

    The regime boundary in the plane of greenhouse forcing and inflow
    temperature, recovered from the study's regime map. Drawn as a filled
    region here rather than as a grid of markers.
    """
    rm = arctic("bifurcation")["regime_map"]
    f = np.asarray(rm["forcing"], dtype=float)
    ta = np.asarray(rm["atlantic_temperature"], dtype=float)
    per = np.asarray(rm["perennial_possible"], dtype=float)
    if per.shape == (len(f), len(ta)):
        per = per.T

    fig, ax = bs.figure(ratio=0.62)
    ax.contourf(f, ta, per, levels=[-0.5, 0.5, 1.5],
                colors=["#F3E4DE", "#DCE8EF"])
    ax.contour(f, ta, per, levels=[0.5], colors=[bs.RULE], linewidths=1.0)

    ax.text(0.05, 0.72, "perennial ice\npossible", transform=ax.transAxes,
            fontsize=7.0, color=bs.SPHERE["atmosphere"])
    ax.text(0.58, 0.72, "perennial ice\ncannot persist", transform=ax.transAxes,
            fontsize=7.0, color="#8C3A2C")
    ax.set_xlabel("greenhouse forcing relative to today (W m$^{-2}$)")
    ax.set_ylabel("temperature of the Atlantic inflow (K above freezing)")
    ax.text(0.97, 0.05, "each kelvin of inflow warming removes about\n"
            "0.7 W m$^{-2}$ from the forcing the ice can take",
            transform=ax.transAxes, fontsize=6.2, color="0.3", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none",
                      alpha=0.8))
    return bs.finish(fig, path)


FIGURES = {
    "haf_trajectory": haf_trajectory,
    "haf_scenarios": haf_scenarios,
    "loss_bars": loss_bars,
    "chs_map": chs_map,
    "whi_map": whi_map,
    "whi_map_full": whi_map_full,
    "cryo_windows": cryo_windows,
    "arctic_parameter": arctic_parameter,
    "arctic_area_mass": arctic_area_mass,
    "arctic_synergy": arctic_synergy,
    "cryo_width_vs_jump": cryo_width_vs_jump,
    "weights": weights,
    "gmst_fit": gmst_fit,
    "ohc_fit": ohc_fit,
    "pattern_map": pattern_map,
    "gmst_paths": gmst_paths,
    "co2_paths": co2_paths,
    "ocean_heat_paths": ocean_heat_paths,
    "carbonate_paths": carbonate_paths,
    "whi_distribution": whi_distribution,
    "whi_zonal": whi_zonal,
    "whi_predictors": whi_predictors,
    "twobox_response": twobox_response,
    "posterior_marginals": posterior_marginals,
    "posterior_joint": posterior_joint,
    "threshold_sensitivity": threshold_sensitivity,
    "chs_evolution": chs_evolution,
    "water_contribution": water_contribution,
    "toe_map_middle": toe_map_middle,
    "toe_map_high": toe_map_high,
    "toe_distribution": toe_distribution,
    "uncertainty_bars": uncertainty_bars,
    "tier_decomposition": tier_decomposition,
}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for name in sorted(FIGURES):
            print(" ", name)
        return 0

    if not os.path.isdir(OUT):
        os.makedirs(OUT)

    names = sorted(FIGURES) if ("--all" in sys.argv or not args) else args
    bs.apply()

    failed = 0
    for name in names:
        if name not in FIGURES:
            print("  unknown figure: %s" % name)
            failed += 1
            continue
        path = os.path.join(OUT, name + ".pdf")
        try:
            FIGURES[name](path)
            print("  %-22s %7.1f kB" % (name, os.path.getsize(path) / 1024.0))
        except Exception as exc:                              # noqa: BLE001
            print("  %-22s FAILED  %s: %s" % (name, type(exc).__name__, exc))
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
