"""Validate the composite CHS/HAF against an INDEPENDENT agronomic impact, the
Major-3 fix's second half (the niche half is in `niche.py`).

The referee asked whether the composite metric is checked against something
*other than its own inputs*. Crop yield is a good candidate: it is not a
constituent of the CHS, of the WHI, or of the climate emulator, yet it integrates
the same water-and-heat stresses the CHS is meant to summarise. We use the open
GDHY dataset (Iizumi & Sakai 2020, PANGAEA): 0.5 deg annual yields of the four
staple crops (maize, rice, wheat, soybean), 1981-2016.

Design (spatial concordance, present-day):
  - the present-day CHS field (year 2016) is computed with the REAL gridded WHI as
    the tier-(ii) baseline -- i.e. the *headline* hazard field, not the stand-in;
  - per cropland cell we compute, from the 36-year record, the detrended
    interannual yield CV (residual std of a per-cell linear fit / mean) -- a clean,
    management-robust measure of climate-driven production INSTABILITY -- plus mean
    yield and the relative trend;
  - we test whether higher CHS goes with higher yield instability (Spearman,
    positive expected), with a 10 deg spatial-block bootstrap so the CI reflects
    spatial autocorrelation, not the inflated cell count; and we contrast the
    detrended CV between the top and bottom CHS deciles (area-weighted).

We also decompose: does the WHI tier-(ii) field ALONE, and the tier-(i) warming
pattern ALONE, each track yield instability? This separates the new (water-hazard)
signal from the climate-exposure signal.

HONEST FRAMING (after adversarial review -- these are properties of the result, not
disclaimers): (1) at a single year sT and U are SCALARS, so the present-day CHS
field is, to three decimals, the WHI baseline field B (Pearson ~0.999; the warming
term has ~20x smaller spatial std and U is spatially uniform). This is therefore a
validation of the tier-(ii) WHI field, NOT of the temperature coupling, which is
reserved for the time-varying HAF trajectory. (2) The pooled multi-crop rank
association is NULL and the decile curve is U-shaped (non-monotone), so CHS does not
order global cropland instability; the 1.84x top/bottom-decile contrast is a
tails-only statistic of a non-monotone curve (decile-rank Spearman ~0.2; top and
bottom HALVES have equal mean CV). (3) Detrended interannual CV is insulated from
the WHI's static greenness predictors (NDWI/EVI), but MEAN yield is not -- a live
greenness->yield path -- so mean-yield is reported only as exploratory. (4) Yield
trend (the natural long-term target for an unsustainability index) is ALSO null,
being dominated by the +1.3%/yr technology growth signal. The one robust positive
tie is rice CV (Spearman +0.22), the most water-limited staple. Run:
  python -m eh_shallow.cropyield
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import chs, data, grid, plots

POSTERIOR = {"ecs": 3.65, "gamma": 0.48}   # OHC-calibrated posterior mean
CHS_YEAR = 2016                             # end of the GDHY record
MIN_YEARS = 20                              # min years of yield data to use a cell
BLOCK_DEG = 10.0                            # spatial-block size for the bootstrap
N_BOOT = 500


def _spearman(x, y):
    from scipy.stats import spearmanr
    r, p = spearmanr(x, y)
    return float(r), float(p)


def _detrended_cv(cube, years):
    """Per-cell detrended interannual CV and relative trend from a [yr,lat,lon] cube.

    Linear fit per cell over the years with data (>= MIN_YEARS); CV = std of the
    residuals / mean; trend = slope / mean (per year). Cells with too few years or
    a non-positive mean are masked (NaN).
    """
    ny, nlat, nlon = cube.shape
    cv = np.full((nlat, nlon), np.nan)
    trend = np.full((nlat, nlon), np.nan)
    meany = np.full((nlat, nlon), np.nan)
    t = years.astype(float)
    for i in range(nlat):
        row = cube[:, i, :]
        ok_any = np.isfinite(row).any(0)
        for j in np.where(ok_any)[0]:
            y = row[:, j]
            m = np.isfinite(y)
            if m.sum() < MIN_YEARS:
                continue
            yy, tt = y[m], t[m]
            mu = yy.mean()
            if mu <= 0:
                continue
            b, a = np.polyfit(tt, yy, 1)
            resid = yy - (a + b * tt)
            cv[i, j] = resid.std() / mu
            trend[i, j] = b / mu
            meany[i, j] = mu
    return cv, trend, meany


def _blocks(lat_v, lon_v):
    """Group cell indices into BLOCK_DEG x BLOCK_DEG spatial blocks."""
    bi = (np.floor(lat_v / BLOCK_DEG).astype(int) * 100000
          + np.floor(lon_v / BLOCK_DEG).astype(int))
    blocks = {}
    for k, b in enumerate(bi):
        blocks.setdefault(b, []).append(k)
    return [np.array(v) for v in blocks.values()]


def _block_bootstrap(stat, lat_v, lon_v, rng):
    """90% CI of `stat(sel_idx)` under spatial autocorrelation: resample
    BLOCK_DEG x BLOCK_DEG blocks with replacement, so effective N is block count."""
    idx_by_block = _blocks(lat_v, lon_v)
    nb = len(idx_by_block)
    vals = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx_by_block[p] for p in rng.integers(0, nb, size=nb)])
        v = stat(sel)
        if np.isfinite(v):
            vals.append(v)
    vals = np.array(vals)
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def _block_bootstrap_spearman(chs_v, cv_v, lat_v, lon_v, rng):
    from scipy.stats import spearmanr
    return _block_bootstrap(
        lambda s: spearmanr(chs_v[s], cv_v[s])[0], lat_v, lon_v, rng)


def _decile_contrast(chs_v, cv_v, area_v):
    """Area-weighted mean detrended CV in the top vs bottom CHS decile + ratio."""
    q10, q90 = np.percentile(chs_v, [10, 90])
    lo = chs_v <= q10
    hi = chs_v >= q90
    cv_lo = np.average(cv_v[lo], weights=area_v[lo])
    cv_hi = np.average(cv_v[hi], weights=area_v[hi])
    return float(cv_lo), float(cv_hi), float(cv_hi / cv_lo)


def run() -> dict:
    proj = np.arange(1750, 2301)
    G = grid.build()

    # headline hazard field: carry the REAL gridded WHI as the tier-(ii) baseline
    baseline_src = "analytic stand-in"
    try:
        from . import whi as _whi
        if os.path.exists(_whi.WHI_PATH):
            B_whi, _, _ = _whi.load_whi_field(G)
            grid.set_baseline(B_whi, G)
            baseline_src = "real gridded WHI"
    except Exception:
        pass

    # present-day CHS field (2016) on the common ssp245 standardisation
    from . import emulator
    out_mean = emulator.run_emulator(POSTERIOR, proj, ssp="ssp245")
    scales = chs.reference_scales(out_mean)
    chs2d = chs.chs_field(out_mean, CHS_YEAR, scales=scales)        # [lat,lon], NaN ocean

    # spatial fields for the decomposition: WHI baseline B and tier-(i) pattern
    B2d = G["B2d"]
    P2d = G["P2d"]
    lat2d, lon2d, area2d, land = G["lat2d"], G["lon2d"], G["area2d"], G["land2d"]

    yld = data.load_gdhy_yields(grid_lon=G["lon"], grid_lat=G["lat"])
    years = yld["years"]

    brng = np.random.default_rng(0)
    per_crop = {}
    pool_chs, pool_cv, pool_lat, pool_lon, pool_area, pool_B, pool_P = (
        [], [], [], [], [], [], [])
    pool_cvnorm, pool_trend = [], []   # CV / this-crop median CV; relative trend
    for crop in data.GDHY_CROPS:
        cube = yld[crop]
        if cube is None:
            continue
        cv, trend, meany = _detrended_cv(cube, years)
        m = (np.isfinite(cv) & np.isfinite(chs2d) & land
             & np.isfinite(meany) & np.isfinite(trend))
        chs_v, cv_v = chs2d[m], cv[m]
        mean_v, trend_v = meany[m], trend[m]
        latc, lonc = lat2d[m], lon2d[m]
        r_cv, p_cv = _spearman(chs_v, cv_v)
        r_my, p_my = _spearman(chs_v, mean_v)
        r_tr, p_tr = _spearman(chs_v, trend_v)
        per_crop[crop] = {
            "n_cells": int(m.sum()),
            "spearman_chs_vs_cv": r_cv, "p_chs_vs_cv": p_cv,
            "spearman_chs_vs_cv_block_ci90":
                _block_bootstrap_spearman(chs_v, cv_v, latc, lonc, brng),
            "spearman_chs_vs_meanyield": r_my, "p_chs_vs_meanyield": p_my,
            "spearman_chs_vs_trend": r_tr, "p_chs_vs_trend": p_tr,
            "median_detrended_cv": float(np.median(cv_v)),
        }
        pool_chs.append(chs_v); pool_cv.append(cv_v)
        pool_lat.append(latc); pool_lon.append(lonc)
        pool_area.append(area2d[m]); pool_B.append(B2d[m]); pool_P.append(P2d[m])
        # composition control: each cell's CV relative to its OWN crop's median,
        # so a decile contrast cannot be an artifact of crop mix across deciles
        pool_cvnorm.append(cv_v / np.median(cv_v))
        pool_trend.append(trend_v)

    chs_v = np.concatenate(pool_chs); cv_v = np.concatenate(pool_cv)
    lat_v = np.concatenate(pool_lat); lon_v = np.concatenate(pool_lon)
    area_v = np.concatenate(pool_area)
    B_v = np.concatenate(pool_B); P_v = np.concatenate(pool_P)
    cvnorm_v = np.concatenate(pool_cvnorm); trend_v_all = np.concatenate(pool_trend)

    r_pool, p_pool = _spearman(chs_v, cv_v)
    r_trend_pool, _ = _spearman(chs_v, trend_v_all)
    rng = np.random.default_rng(0)
    lo_ci, hi_ci = _block_bootstrap_spearman(chs_v, cv_v, lat_v, lon_v, rng)
    cv_lo, cv_hi, ratio = _decile_contrast(chs_v, cv_v, area_v)
    # same contrast on the crop-composition-normalised CV (defends vs crop-mix artifact)
    _, _, ratio_norm = _decile_contrast(chs_v, cvnorm_v, area_v)
    # non-monotonicity diagnostics: decile-rank Spearman and top/bottom HALF means
    edges = np.percentile(chs_v, np.arange(0, 101, 10)); edges[-1] += 1e-9
    didx = np.clip(np.digitize(chs_v, edges) - 1, 0, 9)
    dec_means = [float(np.average(cv_v[didx == b], weights=area_v[didx == b]))
                 for b in range(10)]
    decile_rank_spearman = float(_spearman(np.arange(10), dec_means)[0])
    half_lo = float(np.mean(dec_means[:5])); half_hi = float(np.mean(dec_means[5:]))
    # present-day equivalence: at a fixed year the CHS field is ~ the WHI field
    from scipy.stats import pearsonr
    lv = land & np.isfinite(chs2d) & np.isfinite(B2d)
    sT_now, U_now = chs.tier_series(out_mean, scales=scales)
    inow = int(np.argmin(np.abs(proj - CHS_YEAR)))
    warm_now = P2d * float(sT_now[inow])
    pearson_chs_whi = float(pearsonr(chs2d[lv], B2d[lv])[0])
    warm_to_whi_std = float(warm_now[lv].std() / (B2d[lv].std() or 1.0))
    ratio_ci = _block_bootstrap(
        lambda s: _decile_contrast(chs_v[s], cv_v[s], area_v[s])[2],
        lat_v, lon_v, rng)

    # decomposition: each spatial component alone vs the SAME pooled CV
    r_whi, _ = _spearman(B_v, cv_v)             # tier-(ii) water hazard alone
    r_warm, _ = _spearman(P_v, cv_v)            # tier-(i) warming pattern alone

    return {
        "data_source": yld["source"], "baseline": baseline_src,
        "chs_year": CHS_YEAR, "posterior": POSTERIOR,
        "crops": list(per_crop.keys()), "per_crop": per_crop,
        "pooled": {
            "n_cells": int(chs_v.size),
            "spearman_chs_vs_cv": r_pool, "p_chs_vs_cv": p_pool,
            "spearman_chs_vs_cv_block_ci90": [lo_ci, hi_ci],
            "spearman_chs_vs_trend": r_trend_pool,
            "decile_cv_bottom": cv_lo, "decile_cv_top": cv_hi,
            "decile_cv_ratio_top_over_bottom": ratio,
            "decile_cv_ratio_block_ci90": ratio_ci,
            "decile_cv_ratio_crop_normalised": ratio_norm,
            "decile_rank_spearman": decile_rank_spearman,
            "half_cv_bottom_D1_5": half_lo, "half_cv_top_D6_10": half_hi,
            "decile_curve": dec_means,
            "spearman_whi_alone_vs_cv": r_whi,
            "spearman_warmpattern_alone_vs_cv": r_warm,
            # present-day CHS field is, by construction, ~ the WHI baseline field
            "pearson_chsfield_vs_whi": pearson_chs_whi,
            "warmpattern_to_whi_std_ratio": warm_to_whi_std,
        },
        # arrays for plotting (not serialised)
        "_chs_v": chs_v, "_cv_v": cv_v, "_area_v": area_v,
        "_lat_v": lat_v, "_lon_v": lon_v,
        "_chs2d": chs2d, "_G": G, "_years": years,
    }


def main(outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    print("[cropyield] CHS vs GDHY crop-yield instability (Major-3, impacts half)")
    r = run()
    print(f"  data:     {r['data_source']}")
    print(f"  baseline: {r['baseline']}  (CHS field year {r['chs_year']})")
    pl = r["pooled"]
    print(f"  pooled cropland cells: {pl['n_cells']}")
    print(f"  [present-day CHS field is ~ the WHI field: Pearson="
          f"{pl['pearson_chsfield_vs_whi']:.3f}, warming/WHI spatial-std ratio="
          f"{pl['warmpattern_to_whi_std_ratio']:.3f} -> this validates the WHI "
          f"baseline B, not the temperature coupling]")
    print(f"  Spearman(CHS, detrended yield CV) = {pl['spearman_chs_vs_cv']:.3f} "
          f"[block-bootstrap 90% CI {pl['spearman_chs_vs_cv_block_ci90'][0]:.3f}, "
          f"{pl['spearman_chs_vs_cv_block_ci90'][1]:.3f}]  (NULL global ranking)")
    print(f"  decile-rank Spearman={pl['decile_rank_spearman']:.2f} (non-monotone, "
          f"U-shaped); half-means D1-5={pl['half_cv_bottom_D1_5']*100:.1f}% vs "
          f"D6-10={pl['half_cv_top_D6_10']*100:.1f}%")
    rci = pl["decile_cv_ratio_block_ci90"]
    print(f"  TAILS-ONLY: detrended CV top/bottom CHS decile = "
          f"{pl['decile_cv_top']*100:.1f}% / {pl['decile_cv_bottom']*100:.1f}% "
          f"(x{pl['decile_cv_ratio_top_over_bottom']:.2f} [{rci[0]:.2f}, {rci[1]:.2f}]; "
          f"crop-normalised x{pl['decile_cv_ratio_crop_normalised']:.2f})")
    print(f"  Spearman(CHS, yield TREND) = {pl['spearman_chs_vs_trend']:+.3f} "
          f"(also null: trend dominated by +1.3%/yr technology growth)")
    print(f"  decomposition: WHI alone r={pl['spearman_whi_alone_vs_cv']:.3f}, "
          f"warming-pattern alone r={pl['spearman_warmpattern_alone_vs_cv']:.3f} "
          f"(net NEGATIVE)")
    print("  per crop (Spearman CHS vs CV [90% block CI]):")
    for c, d in r["per_crop"].items():
        ci = d["spearman_chs_vs_cv_block_ci90"]
        print(f"    {c:8s} n={d['n_cells']:5d}  r={d['spearman_chs_vs_cv']:+.3f} "
              f"[{ci[0]:+.3f}, {ci[1]:+.3f}]  "
              f"(mean-yield r={d['spearman_chs_vs_meanyield']:+.3f})")
    fig_path = os.path.join(outdir, "cropyield_validation.pdf")
    plots.plot_cropyield_validation(r, fig_path)
    keep = {k: v for k, v in r.items() if not k.startswith("_")}
    with open(os.path.join(outdir, "cropyield_metrics.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"  wrote {fig_path} + cropyield_metrics.json")
    return keep


if __name__ == "__main__":
    main()
