"""End-to-end driver for the shallow-model prototype.

    python -m eh_shallow.run --n-particles 500 --outdir eh_shallow/out

Pipeline: load real data -> tempered-SMC calibrate ECS/gamma vs HadCRUT5
(1850-1980) -> propagate the posterior to 2300 -> CHS -> HAF -> figures +
metrics.json. Deterministic given --seed.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import chs, data, emulator, grid, plots, smc


def main(argv=None):
    ap = argparse.ArgumentParser(description="shallow-model prototype slice")
    ap.add_argument("--ssp", default="ssp245")
    ap.add_argument("--n-particles", type=int, default=500)
    ap.add_argument("--n-temps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=os.path.join(os.path.dirname(__file__), "out"))
    ap.add_argument("--n-project", type=int, default=200,
                    help="posterior draws propagated to 2300 for the envelopes")
    ap.add_argument("--n-weights", type=int, default=500,
                    help="weight vectors drawn over the simplex for the weight-ensemble HAF")
    ap.add_argument("--baseline", default="auto", choices=["auto", "whi", "standin"],
                    help="tier-(ii) field: real WHI if available (auto/whi) or analytic stand-in")
    ap.add_argument("--allow-synthetic-fallback", action="store_true",
                    help="permit figures built on synthetic fallback data (offline / "
                         "stand-in baseline). OFF by default: the run REFUSES rather than "
                         "silently emit a non-real publication figure. Output is NOT for publication.")
    args = ap.parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    calib_years = np.arange(1750, 2026)
    proj_years = np.arange(1750, 2301)

    print(f"[1/5] data: HadCRUT5 = {data.load_hadcrut5().attrs.get('source')}")
    print(f"      forcing  = {data.load_ar6_erf().attrs.get('source')}")
    print(f"      core     = {emulator.climate_core(calib_years)}")
    print(f"      chemistry= {emulator.chemistry_core()}")

    print(f"[2/5] tempered SMC ({args.n_particles} particles, {args.n_temps} temps)")
    post = smc.run_smc(n_particles=args.n_particles, n_temps=args.n_temps,
                       seed=args.seed, years=calib_years)
    summ = smc.posterior_summary(post)
    for k, v in summ.items():
        print(f"      {k}: {v['mean']:.3f}  [{v['p05']:.3f}, {v['p95']:.3f}]")

    print(f"[3/5] propagate {args.n_project} posterior draws to 2300")
    rng = np.random.default_rng(args.seed + 1)
    w = post["weights"] / post["weights"].sum()
    draws = rng.choice(len(w), size=args.n_project, p=w)
    # reference run at the posterior mean fixes the COMMON CHS standardisation,
    # so HAF is comparable across draws and across SSPs (not self-standardised).
    mean_theta = {n: summ[n]["mean"] for n in post["names"]}
    out_mean = emulator.run_emulator(mean_theta, proj_years, ssp=args.ssp)
    scales = chs.reference_scales(out_mean)
    ens_gmst, ens_ohc, sT_all, U_all = [], [], [], []
    for di in draws:
        out = emulator.run_emulator(post["theta"][di], proj_years, ssp=args.ssp)
        ens_gmst.append(data.rebaseline(proj_years, out["gmst"]))
        ens_ohc.append(data.rebaseline(proj_years, out["ohc"], ref=data.OHC_REF_PERIOD))
        sT, U = chs.tier_series(out, scales=scales)   # tier-(i) driver + tier-(iii) uniform
        sT_all.append(sT)
        U_all.append(U)
    ens_gmst = np.array(ens_gmst)
    ens_ohc = np.array(ens_ohc)
    # HAF for the whole ensemble via the precomputed 0.5-deg grid lookup
    G = grid.build()
    print(f"      grid     = {G['source']} "
          f"({G['n_cells']} cells, {G['n_land']} land, "
          f"land area {100*G['land_area_frac']:.0f}%)")
    # Carry the REAL gridded WHI as the tier-(ii) baseline field B (replacing the
    # analytic stand-in) when available, so the headline HAF uses real geography.
    baseline_src, whi_cov = "analytic stand-in", None
    if args.baseline in ("auto", "whi"):
        try:
            from . import whi as _whi
            if _whi.has_whi():
                B_whi, whi_cov, _ = _whi.load_whi_field(G)
                grid.set_baseline(B_whi, G)
                baseline_src = ("real gridded WHI" if os.path.exists(_whi.WHI_PATH)
                                else "released gridded WHI field")
            elif args.baseline == "whi":
                print("      [warn] WHI raster not found; using stand-in")
        except Exception as e:
            print(f"      [warn] WHI baseline failed: {e}; using stand-in")
    print(f"      baseline = {baseline_src}"
          + (f" ({whi_cov*100:.0f}% land coverage)" if whi_cov else ""))
    ens_haf = chs.haf_ensemble(np.array(sT_all), np.array(U_all), proj_years,
                               percentile=90)
    pctile_sens = chs.haf_percentile_sensitivity(out_mean, scales=scales)

    # weight-ENSEMBLE HAF (Major-1 demonstration): absent the learned RF/WHI
    # weights, span the simplex of weightings over the carried CHS variables and
    # propagate each through the HAF; report the envelope + equal-weight fallback.
    print(f"      weight ensemble: {args.n_weights} draws over the simplex")
    wrng = np.random.default_rng(args.seed + 7)
    cvars = list(chs.CHS_VARS)
    W = wrng.dirichlet(np.ones(len(cvars)), size=args.n_weights)  # flat on simplex
    sTw, Uw = [], []
    for wi in W:
        wd = {k: float(v) for k, v in zip(cvars, wi)}
        sT_i, U_i = chs.tier_series(out_mean, weights=wd, scales=scales)
        sTw.append(sT_i); Uw.append(U_i)
    haf_weights = chs.haf_ensemble(np.array(sTw), np.array(Uw), proj_years, percentile=90)
    sT_eq, U_eq = chs.tier_series(out_mean, scales=scales)  # weights=None -> equal
    haf_equal = chs.haf_ensemble(sT_eq[None, :], U_eq[None, :], proj_years, 90)[0]

    # Provenance guard: never emit publication figures from synthetic fallback data.
    # By the time we get here HadCRUT5 / AR6 ERF / OHC have been loaded (directly and
    # via the SMC); the baseline source is known. Refuse by default if any is synthetic.
    extra_prov = []
    if "stand-in" in baseline_src:
        extra_prov.append(f"analytic stand-in baseline ({baseline_src}; WHI raster absent)")
    synthetic = data.fallbacks_used() + extra_prov
    if synthetic and not args.allow_synthetic_fallback:
        data.assert_real_data(context="eh_shallow figures", extra=extra_prov)
    elif synthetic:
        bar = "!" * 72
        print(f"\n{bar}\nWARNING: SYNTHETIC FALLBACK DATA IN USE -- OUTPUT IS NOT FOR PUBLICATION:")
        for b in synthetic:
            print(f"  - {b}")
        print(f"{bar}\n")

    print("[4/5] figures")
    obs = data.load_hadcrut5()
    oy = obs["year"].to_numpy()
    og = data.rebaseline(oy, obs["gmst"].to_numpy())
    plots.plot_gmst_fit(proj_years, ens_gmst, oy, og, post["calib_window"],
                        os.path.join(args.outdir, "gmst_fit.pdf"))
    if post["obs_ohc"] is not None:
        ohy, ohv, ohsd = post["obs_ohc"]
        plots.plot_ohc_fit(proj_years, ens_ohc, ohy, ohv, ohsd,
                           post["ohc_window"], os.path.join(args.outdir, "ohc_fit.pdf"))
    plots.plot_haf(proj_years, ens_haf, os.path.join(args.outdir, "haf.pdf"),
                   pctile_sens=pctile_sens)
    plots.plot_posterior(post, os.path.join(args.outdir, "posterior.pdf"))
    plots.plot_chs_map(chs.chs_field(out_mean, 2100, scales=scales), G, year=2100,
                       path=os.path.join(args.outdir, "chs_map_2100.pdf"))
    # multi-SSP scenario spread (sigma_scenario) at the posterior mean, on the
    # SAME (ssp245-referenced) standardisation so the scenarios are comparable
    scen_gmst, scen_haf = {}, {}
    for s in data.SSPS:
        o = emulator.run_emulator(mean_theta, proj_years, ssp=s)
        scen_gmst[s] = data.rebaseline(proj_years, o["gmst"])
        sT_s, U_s = chs.tier_series(o, scales=scales)
        scen_haf[s] = chs.haf_ensemble(sT_s[None, :], U_s[None, :], proj_years, 90)[0]
    plots.plot_haf_scenarios(proj_years, scen_haf,
                             os.path.join(args.outdir, "haf_scenarios.pdf"))
    plots.plot_haf_weight_ensemble(proj_years, haf_weights, haf_equal,
                                   os.path.join(args.outdir, "haf_weight_ensemble.pdf"))

    print("[5/5] metrics.json")
    # out-of-sample (1981-2020) GMST skill of the posterior-mean run
    val = (oy >= 1981) & (oy <= 2020)
    gm_mean = data.rebaseline(proj_years, out_mean["gmst"])
    pred = np.interp(oy[val], proj_years, gm_mean)
    rmse = float(np.sqrt(np.mean((pred - og[val]) ** 2)))
    # baseline skill: persistence (hold the last calibration value) and a linear
    # trend fit on 1850-1980, both scored over the withheld 1981-2020 window.
    cal = (oy >= 1850) & (oy <= 1980)
    last_cal = float(og[oy == 1980][0]) if np.any(oy == 1980) else float(og[cal][-1])
    trend = np.polyval(np.polyfit(oy[cal], og[cal], 1), oy)
    rmse_pers = float(np.sqrt(np.mean((last_cal - og[val]) ** 2)))
    rmse_trend = float(np.sqrt(np.mean((trend[val] - og[val]) ** 2)))
    # cleanest sub-window 1981-2004 (before the OHC 2005-2020 constraint opens)
    gm_at_oy = np.interp(oy, proj_years, gm_mean)
    cln = (oy >= 1981) & (oy <= 2004)
    def _rmse_over(a, m):
        return float(np.sqrt(np.mean((a[m] - og[m]) ** 2)))
    rmse_clean = _rmse_over(gm_at_oy, cln)
    rmse_clean_pers = _rmse_over(np.full(len(oy), last_cal), cln)
    rmse_clean_trend = _rmse_over(trend, cln)
    # OHC fit (2005-2020) of the posterior-mean run, on the 2005-2014 baseline
    ohc_rmse = None
    if post["obs_ohc"] is not None:
        ohy, ohv, _ = post["obs_ohc"]
        oh_mean = data.rebaseline(proj_years, out_mean["ohc"], ref=data.OHC_REF_PERIOD)
        oh_pred = np.interp(ohy, proj_years, oh_mean)
        ohc_rmse = float(np.sqrt(np.mean((oh_pred - ohv) ** 2)))
    haf_med = np.percentile(ens_haf, 50, axis=0)
    # HAF spread at 2100 from parameter posterior vs from the weight ensemble
    i2100 = proj_years == 2100
    haf_post_2100 = ens_haf[:, i2100].ravel()
    haf_w_2100 = haf_weights[:, i2100].ravel()
    metrics = {
        "posterior": summ,
        "baseline": {"source": baseline_src, "whi_land_coverage": whi_cov},
        "climate_core": emulator.climate_core(calib_years),
        "chemistry_core": emulator.chemistry_core(),
        "grid": {
            "source": G["source"], "resolution_deg": G["res"],
            "n_cells": G["n_cells"], "n_land_cells": G["n_land"],
            "land_area_fraction": G["land_area_frac"],
            "B_std": grid.B_STD, "sigma_agg": G["sigma_agg"],
        },
        "data_sources": {
            "gmst": data.load_hadcrut5().attrs.get("source"),
            "forcing": data.load_ar6_erf().attrs.get("source"),
            "ohc": data.load_ohc().attrs.get("source"),
        },
        "data_provenance": {
            "all_real": not synthetic,
            "synthetic_fallback_used": synthetic,
        },
        "out_of_sample_1981_2020": {
            "gmst_rmse_K": rmse,
            "gmst_rmse_persistence_K": rmse_pers,
            "gmst_rmse_linear_trend_K": rmse_trend,
            "skill_vs_persistence": 1.0 - rmse / rmse_pers if rmse_pers else None,
            "skill_vs_linear_trend": 1.0 - rmse / rmse_trend if rmse_trend else None,
            "clean_window_1981_2004": {
                "emulator_rmse_K": rmse_clean,
                "persistence_rmse_K": rmse_clean_pers,
                "linear_trend_rmse_K": rmse_clean_trend,
            },
        },
        "ohc_fit_2005_2020": {"ohc_rmse_ZJ": ohc_rmse},
        "haf": {
            "preindustrial_1750_1800": float(np.mean(
                haf_med[(proj_years >= 1750) & (proj_years <= 1800)])),
            "y2020": float(haf_med[proj_years == 2020][0]),
            "y2100": float(haf_med[proj_years == 2100][0]),
            "y2300": float(haf_med[proj_years == 2300][0]),
        },
        "scenarios": {
            s: {"gmst_2100": float(scen_gmst[s][proj_years == 2100][0]),
                "haf_2100": float(scen_haf[s][proj_years == 2100][0]),
                "haf_2300": float(scen_haf[s][proj_years == 2300][0])}
            for s in data.SSPS
        },
        "scenario_spread_haf_2100": float(
            max(scen_haf[s][proj_years == 2100][0] for s in data.SSPS)
            - min(scen_haf[s][proj_years == 2100][0] for s in data.SSPS)),
        "weight_ensemble": {
            "n_weights": args.n_weights,
            "haf_2100_equal_weight": float(haf_equal[i2100][0]),
            "haf_2100_median": float(np.percentile(haf_w_2100, 50)),
            "haf_2100_p05": float(np.percentile(haf_w_2100, 5)),
            "haf_2100_p95": float(np.percentile(haf_w_2100, 95)),
            "haf_2100_width_weights": float(
                np.percentile(haf_w_2100, 95) - np.percentile(haf_w_2100, 5)),
            "haf_2100_width_param_posterior": float(
                np.percentile(haf_post_2100, 95) - np.percentile(haf_post_2100, 5)),
        },
        "config": {"ssp": args.ssp, "n_particles": args.n_particles,
                   "n_temps": args.n_temps, "seed": args.seed},
    }
    with open(os.path.join(args.outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics["haf"], indent=2))
    print(f"\nDone. Outputs in {args.outdir}/  (haf.pdf, gmst_fit.pdf, "
          f"posterior.pdf, metrics.json)")
    return metrics


if __name__ == "__main__":
    main()
