#!/usr/bin/env python3
"""Recompute the quantities the stored results file does not carry.

The published run saved a compact results archive: the parameter ensemble, the
habitable area trajectories, and the composite hazard field at 2100. A book
needs a good deal more than that, so this script re runs the parts of the
pipeline that produce the missing fields and caches everything into one archive
the figure code can read.

What it adds, none of which is in the published archive:

    the calibrated posterior particle cloud and the prior it came from
    the six carried variables as trajectories, under every pathway
    the composite hazard field at several dates, under two pathways
    the composite hazard field with the water hazard term removed
    the time of emergence map under two pathways
    the land warming pattern and the land mask
    the threshold sensitivity curve
    the tier decomposition of the composite score
    the water hazard field and the distribution of its values over land

Everything is deterministic. Same seed, same output, to the last digit.

Usage, from EH_Book/:
    python3 figtools/make_extra.py
    python3 figtools/make_extra.py --particles 400
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOOK = os.path.dirname(HERE)
ROOT = os.path.dirname(BOOK)
PKG = os.path.join(ROOT, "sources", "EH")
sys.path.insert(0, PKG)

from eh_shallow import chs, emulator, grid, smc, whi          # noqa: E402

OUT = os.path.join(BOOK, "figures", "book_extra.npz")
SSPS = ("ssp126", "ssp245", "ssp370", "ssp585")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particles", type=int, default=400)
    ap.add_argument("--temps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    years = np.arange(1750, 2301)
    store = {}

    print("building the grid ...")
    G = grid.build(seed=args.seed)
    store["lon"] = G["lon"]
    store["lat"] = G["lat"]
    store["land2d"] = G["land2d"].astype(np.int8)

    # Install the real gridded water hazard field as the tier two baseline,
    # exactly as the published run does. Without this the baseline is an
    # analytic stand in and every map below would be latitude bands.
    B_whi, whi_cov, whi_raw = whi.load_whi_field(G)
    grid.set_baseline(B_whi, G)
    store["whi_coverage"] = np.array(float(whi_cov))
    store["whi_raw"] = np.asarray(whi_raw, dtype=float)

    store["pattern"] = np.asarray(G["P2d"], dtype=float)
    store["baseline"] = np.asarray(G["B2d"], dtype=float)
    store["area_land"] = G["area"]

    print("calibrating, %d particles ..." % args.particles)
    post = smc.run_smc(n_particles=args.particles, n_temps=args.temps,
                       seed=args.seed, years=np.arange(1750, 2026))
    names = list(post["names"])
    store["post_particles"] = np.asarray(post["theta"])
    store["post_weights"] = np.asarray(post["weights"])
    store["post_names"] = np.array(names)
    store["post_loglik"] = np.asarray(post["loglik"])

    rng = np.random.default_rng(args.seed + 101)
    prior = emulator.sample_prior(rng, 20000)
    store["prior_particles"] = prior

    summ = smc.posterior_summary(post)
    mean_theta = {n: summ[n]["mean"] for n in names}
    store["theta_mean"] = np.array([mean_theta[n] for n in names])

    print("running the emulator under each pathway ...")
    ref = emulator.run_emulator(mean_theta, years, ssp="ssp245")
    scales = chs.reference_scales(ref)
    w_crit = chs.objective_weights(ref, scales=scales, method="critic")

    store["years"] = years
    for ssp in SSPS:
        out = emulator.run_emulator(mean_theta, years, ssp=ssp)
        for var in ("gmst", "ohc", "co2", "ph", "omega", "sst",
                    "toa_imbalance"):
            store["%s_%s" % (var, ssp)] = np.asarray(out[var], dtype=float)
        sT, U = chs.tier_series(out, weights=w_crit, scales=scales)
        store["tierT_%s" % ssp] = np.asarray(sT, dtype=float)
        store["tierU_%s" % ssp] = np.asarray(U, dtype=float)

    print("composite hazard fields ...")
    for ssp in ("ssp245", "ssp585"):
        out = emulator.run_emulator(mean_theta, years, ssp=ssp)
        for year in (1850, 2020, 2100, 2300):
            store["chs_%s_%d" % (ssp, year)] = chs.chs_field(
                out, year, weights=w_crit, scales=scales)

    # The same field with the water hazard term switched off. This is what the
    # map would look like if the measure carried climate and ocean alone, and it
    # is the cleanest way to show how much geography the hydrosphere supplies.
    print("counterfactual field without the water hazard term ...")
    out245 = emulator.run_emulator(mean_theta, years, ssp="ssp245")
    sT, U = chs.tier_series(out245, weights=w_crit, scales=scales)
    i2100 = int(np.argmin(np.abs(years - 2100)))
    land = G["land2d"]
    nowater = np.full(land.shape, np.nan)
    nowater[land] = G["P"] * float(sT[i2100]) + float(U[i2100])
    store["chs_nowater_2100"] = nowater
    store["tau_90"] = np.array(grid.tau_of(90.0, G))

    print("time of emergence ...")
    for ssp in ("ssp245", "ssp585"):
        out = emulator.run_emulator(mean_theta, years, ssp=ssp)
        toe2d, summary = chs.time_of_emergence(
            out, percentile=90.0, weights=w_crit, scales=scales, G=G)
        store["toe_%s" % ssp] = toe2d
        store["toe_summary_%s" % ssp] = np.array(
            [summary.get("frac_of_habitable_that_emerges_by_2300", np.nan),
             summary.get("median_emergence_year", np.nan)], dtype=float)

    print("threshold sensitivity ...")
    pct = np.array([80, 85, 90, 95, 99], dtype=float)
    sens = chs.haf_percentile_sensitivity(out245, weights=w_crit,
                                          scales=scales)
    store["sens_pct"] = pct
    store["sens_haf"] = np.asarray(
        [sens[p] for p in (80, 85, 90, 95, 99)], dtype=float) \
        if isinstance(sens, dict) else np.asarray(sens, dtype=float)

    imp_path = os.path.join(PKG, "eh_shallow", "released",
                            "whi_rf_importances.json")
    if os.path.exists(imp_path):
        with open(imp_path, encoding="utf-8") as fh:
            imp = json.load(fh)
        store["whi_importance_json"] = np.array(json.dumps(imp))

    store["weights_critic"] = np.array(
        [w_crit[k] for k in chs.CHS_VARS], dtype=float)
    store["weight_names"] = np.array(list(chs.CHS_VARS))

    np.savez_compressed(OUT, **store)
    size = os.path.getsize(OUT) / 1e6
    print("")
    print("wrote %s  (%.1f MB, %d arrays)"
          % (os.path.relpath(OUT, BOOK), size, len(store)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
