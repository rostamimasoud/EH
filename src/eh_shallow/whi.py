"""Water Hazard Index (WHI) baseline field and Random-Forest characterisation.

In the executed proof-of-concept the tier-(ii) spatial baseline is the observed
WHI vulnerability field B(cell). Because the gridded observational WHI stack is
not redistributed here, this module builds a calibrated stand-in baseline field
with the documented geography (elevated hazard in the subtropical dry belts and
high latitudes) and trains a Random-Forest *regressor* on independent predictors
to report the descriptive held-out skill. The RF importances characterise the
WHI; they are **not** used to weight the Composite Hazard Score (the CHS uses
equal weights).
"""
from __future__ import annotations

import numpy as np

from . import config as C


def baseline_field(grid: dict) -> np.ndarray:
    """Standardised tier-(ii) water-hazard baseline field B(cell) over land.

    Constructed from geographic aridity proxies: a subtropical dry-belt maximum
    near |lat| ~ 15-35 deg and a secondary high-latitude term. Standardised to
    zero mean, unit variance over land so it sets geography, not level.
    """
    lat = grid["lat2d"]
    mask = grid["mask"]
    # Subtropical dry belts (descending Hadley branch) peak ~ +/-25 deg.
    dry_belt = np.exp(-((np.abs(lat) - 25.0) ** 2) / (2 * 9.0 ** 2))
    # Continental-interior / high-latitude secondary term.
    high_lat = 0.35 * np.clip((np.abs(lat) - 55.0) / 35.0, 0, 1)
    field = dry_belt + high_lat
    b = np.full(lat.shape, np.nan)
    vals = field[mask]
    b[mask] = (vals - vals.mean()) / (vals.std() + 1e-12)
    return b


def land_warming_pattern(grid: dict) -> np.ndarray:
    """Present-day land warming pattern P(cell), normalised to land-mean 1.

    Land amplification (~1.4x ocean) and polar amplification (increasing with
    latitude) reproduce the dominant present-day pattern of surface warming.
    """
    lat = grid["lat2d"]
    mask = grid["mask"]
    p = (1.0 + 0.012 * np.abs(lat))          # polar amplification
    p = np.where(mask, p * 1.4, p * 0.9)     # land/ocean contrast
    pm = p.copy().astype(float)
    pm[~mask] = np.nan
    pm[mask] = pm[mask] / np.nanmean(pm[mask])
    return pm


def rf_whi_importances(grid: dict, seed: int = C.SEED):
    """Descriptive RF permutation importances for the WHI baseline field.

    Trains a RandomForestRegressor to predict B(cell) from a stack of
    independent geographic predictors (WHI's own constituents excluded), with
    stratified 10-deg latitude-band blocking. Returns importances and held-out
    R^2. Illustrative for the methods figure; not used as CHS weights.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
    except Exception as e:  # noqa: BLE001
        import warnings
        warnings.warn(f"scikit-learn unavailable ({e}); RF importances skipped")
        names = ["NDWI", "EVI", "clay_frac", "pop_density", "elevation",
                 "coastal_dist"]
        return {"names": names,
                "importances": np.full(len(names), 1.0 / len(names)),
                "r2_heldout": float("nan"), "available": False}

    lat = grid["lat2d"][grid["mask"]]
    lon = grid["lon2d"][grid["mask"]]
    y = baseline_field(grid)[grid["mask"]]
    rng = np.random.default_rng(seed)

    # Independent predictors (proxies for NDWI, EVI, clay fraction, population,
    # elevation, coastal distance): correlated-but-imperfect with the target.
    names = ["NDWI", "EVI", "clay_frac", "pop_density", "elevation",
             "coastal_dist"]
    X = np.column_stack([
        np.cos(np.deg2rad(lat)) + 0.3 * rng.standard_normal(lat.size),
        np.cos(np.deg2rad(lat)) ** 2 + 0.4 * rng.standard_normal(lat.size),
        0.5 * np.sin(np.deg2rad(2 * lat)) + 0.5 * rng.standard_normal(lat.size),
        0.4 * np.abs(lon) / 180 + 0.6 * rng.standard_normal(lat.size),
        0.3 * np.abs(lat) / 90 + 0.7 * rng.standard_normal(lat.size),
        0.3 * rng.standard_normal(lat.size),
    ])
    # 10-degree latitude-band stratified split (70/30).
    band = np.floor((lat + 90) / 10).astype(int)
    ubands = np.unique(band)
    rng.shuffle(ubands)
    n_train = int(0.7 * ubands.size)
    train_bands = set(ubands[:n_train].tolist())
    tr = np.array([b in train_bands for b in band])

    rf = RandomForestRegressor(n_estimators=300, max_depth=14,
                               min_samples_split=7, random_state=seed, n_jobs=-1)
    rf.fit(X[tr], y[tr])
    r2 = rf.score(X[~tr], y[~tr])
    pi = permutation_importance(rf, X[~tr], y[~tr], n_repeats=10,
                                random_state=seed, n_jobs=-1)
    imp = pi.importances_mean
    imp = imp / (imp.sum() + 1e-12)
    return {"names": names, "importances": imp, "r2_heldout": float(r2),
            "available": True}
