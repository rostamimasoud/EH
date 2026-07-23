"""Validate the headline HAF against the human climate niche (Xu et al. 2020),
the proposal's Major-3 fix: the composite metric is checked against an
INDEPENDENT, observation-grounded notion of habitable area, not just its inputs.

Xu et al. (2020, PNAS) show that humans and agriculture concentrate in a narrow
band of mean annual temperature (MAT) and that the land area beyond a
near-unlivable MAT threshold (~29 degC, comparable to today's hottest inhabited
places) is currently ~0.8% of land but expands to ~19% by ~2070 under RCP8.5.

We reproduce that geometry from the prototype's own warming on real data:
  - present-day MAT from the CRU 1961-90 absolute climatology (data.load_mat_climatology);
  - warmed by the SAME tier-(i) land warming pattern the HAF uses (grid.P2d) times
    the emulator's GMST anomaly (relative to 1961-90), so the niche and the HAF
    share one temperature field;
  - the near-unlivable fraction (MAT > 29 degC) and the niche-habitable fraction
    (its complement) are tracked over 1850-2300 and compared with the HAF.

Two checks: (a) does the emulator's warming reproduce Xu's ~19% near-unlivable
land by 2070 under the high-emission pathway? (b) does the headline HAF track the
independent niche contraction? Run:  python -m eh_shallow.niche
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import chs, data, emulator, grid, plots

HOT_MAT = 29.0                       # Xu 2020 near-unlivable MAT threshold (degC)
MAT_REF = (1961, 1990)              # CRU climatology baseline -> GMST anomaly base
POSTERIOR = {"ecs": 3.65, "gamma": 0.48}   # OHC-calibrated posterior mean
XU_RCP85_2070_UNLIVABLE = 0.19     # Xu et al. 2020: ~19% of land, RCP8.5, ~2070


def _mat_on_grid(G) -> tuple:
    """Regrid the 5 deg CRU MAT climatology onto the prototype 0.5 deg grid."""
    from scipy.interpolate import RegularGridInterpolator
    c = data.load_mat_climatology()
    f = RegularGridInterpolator((c["lat"], c["lon"]), c["mat"],
                                bounds_error=False, fill_value=None)
    pts = np.stack([G["lat2d"].ravel(), G["lon2d"].ravel()], axis=-1)
    return f(pts).reshape(G["lat2d"].shape), c["source"]


def niche_fractions(gmst_anom: np.ndarray, G, mat0_land, P_land, area_land):
    """Near-unlivable (MAT>29) land fraction for each warming level gmst_anom."""
    A = area_land.sum()
    unliv = np.array([
        area_land[(mat0_land + P_land * dt) > HOT_MAT].sum() / A
        for dt in gmst_anom])
    return unliv


def run() -> dict:
    proj = np.arange(1750, 2301)
    G = grid.build()
    mat0_2d, src = _mat_on_grid(G)
    land = G["land2d"]
    mat0_land = mat0_2d[land]
    P_land = G["P2d"][land]            # SAME tier-(i) pattern the HAF uses
    area_land = G["area2d"][land]

    # carry the real WHI as the tier-(ii) field so this HAF matches the headline
    try:
        from . import whi as _whi
        if _whi.has_whi():
            B_whi, _, _ = _whi.load_whi_field(G)
            grid.set_baseline(B_whi, G)
    except Exception:
        pass
    # reference standardisation for HAF (ssp245 posterior-mean run)
    out245 = emulator.run_emulator(POSTERIOR, proj, ssp="ssp245")
    scales = chs.reference_scales(out245)
    sT, U = chs.tier_series(out245, scales=scales)
    haf245 = chs.haf_ensemble(sT[None, :], U[None, :], proj, 90)[0]

    per_ssp = {}
    for s in data.SSPS:
        o = emulator.run_emulator(POSTERIOR, proj, ssp=s)
        gmst_anom = data.rebaseline(proj, o["gmst"], ref=MAT_REF)   # vs 1961-90
        unliv = niche_fractions(gmst_anom, G, mat0_land, P_land, area_land)
        per_ssp[s] = {"unlivable": unliv, "niche_hab": 1.0 - unliv,
                      "gmst_anom_1961_90": gmst_anom}

    # correlation of HAF vs niche-habitable fraction over the projection (ssp245)
    nh245 = per_ssp["ssp245"]["niche_hab"]
    g245 = per_ssp["ssp245"]["gmst_anom_1961_90"]
    fut = proj >= 1900
    r = float(np.corrcoef(haf245[fut], nh245[fut])[0, 1])
    # partial correlation of HAF and niche controlling for GMST + slope divergence:
    # both are near-deterministic nonlinear functions of the same warming, so the
    # partial-r stays high (shared nonlinear GMST dependence) and the informative
    # signal is the slope ratio, not the correlation.
    rHG = float(np.corrcoef(haf245[fut], g245[fut])[0, 1])
    rNG = float(np.corrcoef(nh245[fut], g245[fut])[0, 1])
    den = np.sqrt((1 - rHG ** 2) * (1 - rNG ** 2))
    partial = float((r - rHG * rNG) / den) if den > 0 else float("nan")
    sH = float(np.polyfit(g245[fut], haf245[fut], 1)[0])
    sN = float(np.polyfit(g245[fut], nh245[fut], 1)[0])

    def at(arr, yr):
        return float(arr[proj == yr][0])

    present_unliv = at(per_ssp["ssp245"]["unlivable"], 2015)
    unliv_2070_585 = at(per_ssp["ssp585"]["unlivable"], 2070)

    return {
        "mat_source": src,
        "hot_mat_threshold_C": HOT_MAT, "mat_ref": MAT_REF, "posterior": POSTERIOR,
        "present_unlivable_frac_2015": present_unliv,
        "unlivable_frac_2070_ssp585": unliv_2070_585,
        "xu2020_rcp85_2070_unlivable": XU_RCP85_2070_UNLIVABLE,
        "unlivable_2100": {s: at(per_ssp[s]["unlivable"], 2100) for s in data.SSPS},
        "niche_hab_2100": {s: at(per_ssp[s]["niche_hab"], 2100) for s in data.SSPS},
        "haf_2100_ssp245": at(haf245, 2100),
        "niche_hab_2100_ssp245": at(nh245, 2100),
        "haf_vs_niche_corr_ssp245": r,
        "haf_vs_niche_partial_corr_given_gmst": partial,
        "slope_haf_per_K": sH, "slope_niche_per_K": sN,
        "slope_ratio_haf_over_niche": float(sH / sN) if sN else None,
        # arrays for plotting
        "years": proj, "haf245": haf245, "per_ssp": per_ssp,
    }


def main(outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    print("[niche] HAF vs human climate niche (Xu et al. 2020), real CRU MAT")
    r = run()
    print(f"  MAT source: {r['mat_source']}")
    print(f"  near-unlivable (MAT>29) land fraction, present (2015): "
          f"{r['present_unlivable_frac_2015']*100:.1f}%")
    print(f"  near-unlivable by 2070 under SSP5-8.5: "
          f"{r['unlivable_frac_2070_ssp585']*100:.1f}%  "
          f"(Xu 2020 RCP8.5 ~{r['xu2020_rcp85_2070_unlivable']*100:.0f}%)")
    print(f"  2100 near-unlivable: " + ", ".join(
        f"{s}={r['unlivable_2100'][s]*100:.0f}%" for s in data.SSPS))
    print(f"  HAF vs niche-habitable correlation (ssp245): "
          f"{r['haf_vs_niche_corr_ssp245']:.3f}")
    fig_path = os.path.join(outdir, "niche_vs_haf.pdf")
    plots.plot_niche_vs_haf(r, fig_path)
    # c,d-labelled variant for the NREE Perspective, where this figure is stacked
    # beneath struct_sigma (a,b) so the composite Fig. 6 reads as panels a-d.
    cd_path = os.path.join(outdir, "niche_vs_haf_cd.pdf")
    plots.plot_niche_vs_haf(r, cd_path, labels=("c", "d"))
    keep = {k: v for k, v in r.items() if k not in ("years", "haf245", "per_ssp")}
    with open(os.path.join(outdir, "niche_metrics.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"  wrote {fig_path} + niche_metrics.json")
    return keep


if __name__ == "__main__":
    main()
