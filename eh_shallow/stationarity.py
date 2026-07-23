"""Pattern-stationarity verification for the tier-(i) pattern-scaling assumption
(the proposal's Major-5 fix), on real gridded data.

Tier-(i) disaggregation assumes the normalised spatial warming pattern is
invariant as the global mean changes. We test that with NASA GISTEMP gridded
2-degree anomalies:

  - the reference pattern beta_ref is the per-cell regression slope of annual
    temperature anomaly on global mean temperature (local K per global K);
  - the warming pattern realised at several increasing warming LEVELS is the
    (warm-epoch minus 1951-1980 baseline) field divided by its global-mean
    warming; if the pattern is stationary these all match beta_ref;
  - we report the area-weighted centred spatial correlation between each
    warming-level pattern and beta_ref (min / mean / max) and Taylor statistics,
    and we compare beta_ref with the prototype's *assumed* tier-(i) pattern
    (grid.P2d) to check the assumed pattern is realistic.

Honest scope: observations only reach ~1.3 K, so this verifies the METHOD and the
pattern's stationarity within the observed range. The breakdown the reviewer flags
at 3-5 K (polar-amplification change, monsoon shifts, AMOC) needs model output and
is the full-proposal extension. Run:  python -m eh_shallow.stationarity
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import data, grid, plots

BASE_EPOCH = (1951, 1980)                       # baseline (anomaly ~ 0)
WARM_EPOCHS = [(1981, 1995), (1996, 2010), (2011, 2024)]   # increasing warming
REG_PERIOD = (1950, 2024)                       # regression window for beta_ref


def _aw_stats(a, b, w):
    """Area-weighted centred spatial correlation, std-ratio and centred RMSD of
    field `a` vs reference `b` over finite cells (weights `w`)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b, w = a[m], b[m], w[m]
    w = w / w.sum()
    am = a - np.sum(w * a)
    bm = b - np.sum(w * b)
    sa = np.sqrt(np.sum(w * am ** 2))
    sb = np.sqrt(np.sum(w * bm ** 2))
    corr = float(np.sum(w * am * bm) / (sa * sb))
    crmsd = float(np.sqrt(np.sum(w * (am - bm) ** 2)))
    return corr, float(sa / sb), crmsd


def run() -> dict:
    g = data.load_gistemp_gridded()
    lat, lon, years, anom = g["lat"], g["lon"], g["years"], g["anom"]
    lat2d = np.repeat(lat[:, None], lon.size, axis=1)
    w2d = np.cos(np.radians(lat2d))

    def gmean(field):
        m = np.isfinite(field) if not np.ma.isMaskedArray(field) else ~np.ma.getmaskarray(field)
        f = np.ma.filled(field, np.nan)
        ww = np.where(m, w2d, 0.0)
        return float(np.nansum(ww * f) / ww.sum())

    anom = np.ma.filled(anom.astype(float), np.nan)
    gmst = np.array([gmean(anom[i]) for i in range(years.size)])

    def epoch_mean(e):
        sel = (years >= e[0]) & (years <= e[1])
        return np.nanmean(anom[sel], axis=0), float(np.nanmean(gmst[sel]))

    base, gbase = epoch_mean(BASE_EPOCH)

    # reference pattern: per-cell regression slope of anomaly on GMST (1950-2024)
    rp = (years >= REG_PERIOD[0]) & (years <= REG_PERIOD[1])
    G_ = gmst[rp] - gmst[rp].mean()
    A_ = anom[rp] - np.nanmean(anom[rp], axis=0)
    denom = np.sum(G_ ** 2)
    beta_ref = np.nansum(G_[:, None, None] * A_, axis=0) / denom    # local K / global K

    patterns, warming, stats = {}, {}, {}
    for e in WARM_EPOCHS:
        fm, gm = epoch_mean(e)
        dT = gm - gbase
        patterns[e] = (fm - base) / dT
        warming[e] = dT
        stats[e] = _aw_stats(patterns[e], beta_ref, w2d)

    corrs = [stats[e][0] for e in WARM_EPOCHS]
    # cleanest, self-contained stationarity number: correlation between the two
    # HIGHEST-signal epochs (independent of beta_ref; avoids the small-dT noise of
    # the earliest epoch and any self-reference with beta_ref).
    hi_corr, hi_sr, _ = _aw_stats(patterns[WARM_EPOCHS[-2]], patterns[WARM_EPOCHS[-1]], w2d)

    # prototype's ASSUMED tier-(i) pattern (grid.P2d), regridded to the obs grid
    from scipy.interpolate import RegularGridInterpolator
    Gp = grid.build()
    fP = RegularGridInterpolator((Gp["lat"], Gp["lon"]), Gp["P2d"],
                                 bounds_error=False, fill_value=None)
    P_obs = fP(np.stack([lat2d.ravel(),
                         np.repeat(lon[None, :], lat.size, axis=0).ravel()],
                        axis=-1)).reshape(lat2d.shape)
    proto_corr, proto_stdratio, _ = _aw_stats(P_obs, beta_ref, w2d)

    return {
        "method": "pattern-stationarity test on GISTEMP gridded 2 deg: warming-level "
                  "patterns vs the regression reference pattern beta_ref",
        "source": g["source"], "base_epoch": BASE_EPOCH, "reg_period": REG_PERIOD,
        "warm_epochs": {f"{e[0]}-{e[1]}": {"warming_K": warming[e],
                                           "corr": stats[e][0],
                                           "std_ratio": stats[e][1],
                                           "crmsd": stats[e][2]} for e in WARM_EPOCHS},
        "corr_min": float(np.min(corrs)), "corr_mean": float(np.mean(corrs)),
        "corr_max": float(np.max(corrs)),
        "corr_highsignal_epochs": float(hi_corr),
        "highsignal_pair": [f"{WARM_EPOCHS[-2][0]}-{WARM_EPOCHS[-2][1]}",
                            f"{WARM_EPOCHS[-1][0]}-{WARM_EPOCHS[-1][1]}"],
        "max_warming_observed_K": float(max(warming.values())),
        "assumed_pattern_corr": proto_corr, "assumed_pattern_std_ratio": proto_stdratio,
        # arrays for plotting
        "lat": lat, "lon": lon, "beta_ref": beta_ref, "w2d": w2d,
        "stats": stats, "warming": warming, "warm_epochs_list": WARM_EPOCHS,
        "proto_point": (proto_corr, proto_stdratio),
    }


def main(outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    print("[stationarity] tier-(i) pattern test on GISTEMP gridded 2 deg")
    r = run()
    print(f"  source: {r['source']}")
    for k, v in r["warm_epochs"].items():
        print(f"  {k}: warming {v['warming_K']:.2f} K  corr={v['corr']:.3f}  "
              f"std-ratio={v['std_ratio']:.2f}")
    print(f"  pattern correlation across warming levels: "
          f"min {r['corr_min']:.3f} / mean {r['corr_mean']:.3f} / max {r['corr_max']:.3f}")
    print(f"  high-signal epoch pair {r['highsignal_pair']}: corr={r['corr_highsignal_epochs']:.3f}")
    print(f"  observed warming range: up to {r['max_warming_observed_K']:.2f} K")
    print(f"  assumed tier-(i) pattern vs observed beta_ref: corr={r['assumed_pattern_corr']:.3f}")
    fig_path = os.path.join(outdir, "pattern_stationarity.pdf")
    plots.plot_stationarity(r, fig_path)
    keep = {k: v for k, v in r.items()
            if k not in ("lat", "lon", "beta_ref", "w2d", "stats", "warming",
                         "warm_epochs_list", "proto_point")}
    with open(os.path.join(outdir, "stationarity_metrics.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"  wrote {fig_path} + stationarity_metrics.json")
    return keep


if __name__ == "__main__":
    main()
