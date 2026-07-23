"""Wire the REAL gridded Water Hazard Index (WHI) and its predictor stack into the
prototype: the tier-(ii) locally-forced field (replacing the stand-in `B`) and the
Component-1 Random-Forest feature-importance weighting, both on real data.

Data (the user's aquifer/WHI study; read locally, raw never committed):
  - ICWHI_P.tif  : gridded WHI hazard (0.02 deg, EPSG:4326), the tier-(ii) field;
  - Predictors_full/*.tif : the 22-layer Earth-system predictor stack.

We (a) downsample the WHI to the model's 0.5 deg grid and use it as the real
baseline land-hazard field B, turning on genuine emergent hotspots; and (b) train
a Random Forest to predict the WHI from the INDEPENDENT predictors (excluding the
WHI's own constituents -- aridity, irrigation, river distance, subsidence) with
spatial-block cross-validation, and report permutation importances -- the
data-driven CHS weighting the proposal specifies. Run:  python -m eh_shallow.whi
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import chs, data, emulator, grid, plots

# The raw, high-resolution Water Hazard Index raster and its predictor stack are
# from a separate, as-yet unpublished aquifer study and are NOT distributed with
# this code (available from the corresponding author on reasonable request). Point
# these at your local copies via the EH_WHI_PATH / EH_WHI_PRED_DIR environment
# variables; the defaults are the authors' local paths and will not exist elsewhere.
WHI_PATH = os.environ.get(
    "EH_WHI_PATH", "/home/bijanf/Documents/MR_gwasser/model_output/UQ/ICWHI_P.tif")
PRED_DIR = os.environ.get(
    "EH_WHI_PRED_DIR", "/home/bijanf/Documents/MR_gwasser_cluster_cache/Predictors_full")

# Released, redistributable derivative: the WHI resampled to the 0.5-deg model grid
# (raw values; ocean / no-data = NaN). This lets the WHI-dependent figures reproduce
# without the raw raster, which is not distributed. Written by the release export.
RELEASED_WHI = os.path.join(os.path.dirname(__file__), "released", "whi_field_0p5deg.npz")
# Released permutation importances (so Fig. proto-weights(b) regenerates without the
# raw predictor stack / raster).
RELEASED_RF = os.path.join(os.path.dirname(__file__), "released", "whi_rf_importances.json")


def has_whi():
    """True if either the raw WHI raster or the released 0.5-deg field is available."""
    return os.path.exists(WHI_PATH) or os.path.exists(RELEASED_WHI)


def _raw_whi_on_grid(G):
    """Raw WHI on the model grid: from the raster if present, else the released
    0.5-deg field (redistributed with the code), else raise."""
    if os.path.exists(WHI_PATH):
        return _resample_to_grid(WHI_PATH, G)
    if os.path.exists(RELEASED_WHI):
        return np.load(RELEASED_WHI)["whi"].astype(float)
    raise FileNotFoundError(
        "No WHI available: set EH_WHI_PATH to the raster, or provide the released "
        "field at " + RELEASED_WHI)

# WHI's own constituents -> held out of the supervised fit to avoid circularity.
# Net groundwater abstraction (nag) is the groundwater-stress numerator, so it is
# a constituent and is excluded too, along with aridity, irrigation, river
# distance, and subsidence (aquifer compaction).
WHI_CONSTITUENTS = {"Aridity_Index", "Irrigated_Area_Density_meier",
                    "River_distance", "Subsidence", "nag_mean_annual_2000_2009_2km"}
# Independent Earth-system predictors used in the RF (the rest of the stack).
PRED_FILES = [
    "Population_Density", "Global_Sediment_Thickness", "NDWI", "EVI",
    "TRCLM_ET", "TRCLM_precp", "TRCLM_RET", "TRCLM_soil", "TRCLM_Tmax",
    "pet_trend_2013_2019", "ppt_trend_2013_2019",
    "tmax_trend_2013_2019", "tmin_trend_2013_2019", "Clay_Thickness", "Clay_200cm",
]


def _resample_to_grid(path, G, resampling=None):
    """Average-resample a GeoTIFF onto the model 0.5 deg grid (lat ascending)."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    resampling = resampling or Resampling.average
    lon, lat, res = G["lon"], G["lat"], G["res"]
    dst_transform = from_origin(lon[0] - res / 2, lat[-1] + res / 2, res, res)
    dst = np.full((lat.size, lon.size), np.nan, dtype="float32")   # north-up rows
    with rasterio.open(path) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=dst_transform, dst_crs="EPSG:4326",
                  src_nodata=src.nodata, dst_nodata=np.nan, resampling=resampling)
    return dst[::-1, :]                                            # -> lat ascending


def load_whi_field(G=None):
    """Real WHI on the 0.5 deg grid as a standardised tier-(ii) field B_whi.

    Returns (B_whi 2D, coverage_frac, raw_whi 2D). Land cells without a WHI value
    are filled with the land median (neutral) so the HAF stays well defined.
    """
    G = G or grid.build()
    whi = _raw_whi_on_grid(G)
    land = G["land2d"]
    valid = land & np.isfinite(whi)
    cov = float(G["area2d"][valid].sum() / G["area2d"][land].sum())
    med = float(np.nanmedian(whi[valid]))
    filled = np.where(land & np.isfinite(whi), whi, med)
    # standardise to the same convention as the stand-in B (centred over land,
    # rescaled to B_STD); WHI already increases with hazard, so no sign flip.
    bl = filled[land]
    B = (filled - bl.mean()) / (bl.std() or 1.0) * grid.B_STD
    B = np.where(land, B, np.nan)
    return B, cov, whi


# WHI constituents that exist in the predictor stack (used only to quantify leakage)
CONSTITUENT_FILES = ["Aridity_Index", "Irrigated_Area_Density_meier",
                     "River_distance", "Subsidence", "nag_mean_annual_2000_2009_2km"]


def _stack(files, G):
    feats, names = [], []
    for f in files:
        p = os.path.join(PRED_DIR, f + ".tif")
        if os.path.exists(p):
            feats.append(_resample_to_grid(p, G))
            names.append(f)
    return feats, names


def _fit_score(X, y, latf, seed):
    """Spatial-block (10 deg lat-band) RF fit; return (model, held-out R2, train/test masks)."""
    from sklearn.ensemble import RandomForestRegressor
    band = np.floor((latf + 90) / 10).astype(int)
    rng = np.random.default_rng(seed)
    ub = np.unique(band)
    test_bands = set(rng.choice(ub, size=max(1, len(ub) // 3), replace=False).tolist())
    te = np.array([b in test_bands for b in band]); tr = ~te
    rf = RandomForestRegressor(n_estimators=300, max_depth=14, min_samples_split=7,
                               n_jobs=-1, random_state=seed)
    rf.fit(X[tr], y[tr])
    return rf, float(rf.score(X[te], y[te])), tr, te


def rf_importances(seed=0):
    """Component-1 RF on the real 0.5 deg stack, reporting permutation importances
    of the INDEPENDENT predictors and the leakage (R2 with vs without the WHI's
    own constituents), per the proposal's anti-circularity protocol.

    Released-artifact fallback: if the raw raster/predictor stack is unavailable,
    return the released importances so Fig. proto-weights(b) still regenerates."""
    if not (os.path.exists(WHI_PATH) and os.path.isdir(PRED_DIR)):
        if os.path.exists(RELEASED_RF):
            with open(RELEASED_RF) as f:
                return json.load(f)["rf"]
        raise FileNotFoundError(
            "No WHI predictor stack and no released importances at " + RELEASED_RF)
    from sklearn.inspection import permutation_importance
    G = grid.build()
    latf_full = G["lat2d"].ravel()
    y_full = _resample_to_grid(WHI_PATH, G).ravel()
    land = G["land2d"].ravel()

    ind_feats, ind_names = _stack(PRED_FILES, G)
    con_feats, _ = _stack(CONSTITUENT_FILES, G)
    Xi_all = np.stack([a.ravel() for a in ind_feats], axis=1)
    # row mask from the independent features only (constituents may have unset nodata)
    m = land & np.isfinite(y_full) & np.all(np.isfinite(Xi_all), axis=1)
    Xi, y, latf = Xi_all[m], y_full[m], latf_full[m]
    # constituent columns on the same rows, median-imputed (drop near-empty columns)
    con_cols = []
    for a in con_feats:
        c = a.ravel()[m]
        if np.isfinite(c).sum() > 0.3 * c.size:
            con_cols.append(np.where(np.isfinite(c), c, np.nanmedian(c)))
    Xc = np.column_stack([Xi] + con_cols) if con_cols else Xi

    # independent-only fit (the reported importances)
    rf, r2_indep, tr, te = _fit_score(Xi, y, latf, seed)
    pi = permutation_importance(rf, Xi[te], y[te], n_repeats=10, random_state=seed, n_jobs=-1)
    imp = np.clip(pi.importances_mean, 0, None)
    imp = imp / imp.sum() if imp.sum() > 0 else imp
    order = np.argsort(imp)[::-1]
    # with-constituents fit (only to quantify leakage)
    _, r2_full, _, _ = _fit_score(Xc, y, latf, seed)

    return {"names": [ind_names[i] for i in order],
            "importance": [float(imp[i]) for i in order],
            "cv_r2_heldout": r2_indep, "cv_r2_with_constituents": r2_full,
            "leakage_r2": float(r2_full - r2_indep),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "n_features": len(ind_names)}


def main(outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    G = grid.build()
    print("[whi] real gridded WHI -> tier-(ii) field + Component-1 RF importances")
    B_whi, cov, _ = load_whi_field(G)
    print(f"  WHI coverage of land: {cov*100:.0f}%  (gaps filled with land median)")

    # real CHS hotspot map at 2100 (SSP2-4.5, posterior mean) using WHI as B
    proj = np.arange(1750, 2301)
    out = emulator.run_emulator({"ecs": 3.65, "gamma": 0.48}, proj, ssp="ssp245")
    scales = chs.reference_scales(out)
    sT, U = chs.tier_series(out, scales=scales)
    i = int(np.argmin(np.abs(proj - 2100)))
    chs_field = np.where(G["land2d"], G["P2d"] * float(sT[i]) + B_whi + float(U[i]), np.nan)
    plots.plot_chs_map(chs_field, G, year=2100,
                       path=os.path.join(outdir, "whi_chs_map.pdf"))

    print("  training Component-1 RF (WHI ~ independent predictors, spatial-block CV)")
    rf = rf_importances()
    print(f"  held-out R2 (independent predictors) = {rf['cv_r2_heldout']:.2f}  "
          f"({rf['n_train']} train / {rf['n_test']} test cells, {rf['n_features']} predictors)")
    print(f"  held-out R2 WITH WHI constituents     = {rf['cv_r2_with_constituents']:.2f}  "
          f"-> leakage = {rf['leakage_r2']:.2f}")
    for n, w in list(zip(rf["names"], rf["importance"]))[:6]:
        print(f"    {n:34s} {w:.3f}")
    plots.plot_rf_importances(rf, os.path.join(outdir, "whi_rf_importances.pdf"))

    metrics = {"whi_path": WHI_PATH, "whi_land_coverage": cov,
               "rf": rf, "chs_map_year": 2100, "chs_map_ssp": "ssp245"}
    with open(os.path.join(outdir, "whi_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  wrote whi_chs_map.pdf + whi_rf_importances.pdf + whi_metrics.json")
    return metrics


if __name__ == "__main__":
    main()
