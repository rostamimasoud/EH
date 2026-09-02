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
        extra = c["by_scenario_2100"][key]["additional_loss_2020_2100_mkm2"]
        ax.barh(i, committed, color="0.55", height=0.62,
                edgecolor="white", linewidth=0.6)
        ax.barh(i, extra, left=committed, color=bs.SSP[key], height=0.62,
                edgecolor="white", linewidth=0.6)
        ax.text(committed + extra + 1.5, i, "%.0f" % (committed + extra),
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
    """How four principled weighting schemes distribute weight, and what it
    does to the answer. Book layout: schemes as grouped rows, not a strip."""
    m = metrics()["objective_weights"]
    order = ["gmst", "co2", "sst", "ohc", "ph", "omega"]
    names = ["surface\ntemperature", "carbon\ndioxide", "sea surface\ntemperature",
             "ocean heat\ncontent", "ocean\npH", "aragonite\nsaturation"]
    schemes = ["equal", "critic", "entropy", "pca"]
    labels = ["equal", "CRITIC", "entropy", "principal\ncomponent"]
    colours = ["0.62", bs.SPHERE["ocean"], bs.SPHERE["atmosphere"],
               bs.SPHERE["hydrosphere"]]

    fig, (ax, bx) = plt.subplots(
        1, 2, figsize=(bs.TEXT_WIDTH, bs.TEXT_WIDTH * 0.46),
        gridspec_kw={"width_ratios": [2.5, 1.0], "wspace": 0.45})

    x = np.arange(len(order))
    width = 0.2
    for j, (scheme, colour) in enumerate(zip(schemes, colours)):
        vals = [m["weights"][scheme][k] for k in order]
        ax.bar(x + (j - 1.5) * width, vals, width * 0.92, color=colour,
               label=labels[j].replace("\n", " "), edgecolor="none")
    ax.axhline(1 / 6.0, color=bs.RULE, lw=0.5, ls=(0, (2, 2)))
    ax.text(5.42, 1 / 6.0 + 0.012, "equal share", fontsize=6, color=bs.RULE,
            ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=6)
    ax.set_ylabel("weight")
    ax.set_ylim(0, 0.45)
    ax.legend(fontsize=6, loc="upper left", ncol=2, handlelength=1.1)

    haf = [m["haf_2100_by_method"][s] for s in schemes]
    bx.bar(np.arange(4), haf, 0.62, color=colours, edgecolor="none")
    bx.set_xticks(np.arange(4))
    bx.set_xticklabels(labels, fontsize=6)
    bx.set_ylim(0.5, 0.72)
    bx.set_ylabel("habitable area fraction, 2100", fontsize=7)
    bx.yaxis.set_major_locator(MultipleLocator(0.05))
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


FIGURES = {
    "haf_trajectory": haf_trajectory,
    "haf_scenarios": haf_scenarios,
    "loss_bars": loss_bars,
    "chs_map": chs_map,
    "whi_map": whi_map,
    "weights": weights,
    "gmst_fit": gmst_fit,
    "ohc_fit": ohc_fit,
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
            size = os.path.getsize(path) / 1024.0
            print("  %-18s %7.1f kB" % (name, size))
        except Exception as exc:                              # noqa: BLE001
            print("  %-18s FAILED  %s: %s"
                  % (name, type(exc).__name__, exc))
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
