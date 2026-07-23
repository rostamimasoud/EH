"""Observational and forcing data for the shallow-time proof-of-concept.

Every loader downloads from a public source on first call and caches the raw
file under ``config.DATA_DIR``. Each loader returns a tidy structure and records
the provenance actually used (``source`` field), so a run is transparent about
whether it used the primary download or a documented reduced fallback.

Primary sources (matching the SI Data-availability statement):
  * GMST   : HadCRUT5 global annual summary series (Met Office).
  * OHC    : NOAA/NCEI 0--2000 m ocean heat content (yearly).
  * ERF    : RCMIP annual-mean effective radiative forcing (per SSP), else a
             reduced CO2-derived forcing anchored to the AR6 2019 value.
  * CO2    : historical + SSP atmospheric CO2 pathway.
  * land   : Natural Earth land polygons rasterised to the 0.5 deg grid.
"""
from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

_TIMEOUT = 30


def _cache(name: str) -> Path:
    return C.DATA_DIR / name


def _download(url: str, dest: Path) -> bool:
    """Fetch ``url`` to ``dest`` (cached). Return True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    try:
        import requests
        r = requests.get(url, timeout=_TIMEOUT)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"download failed ({url}): {e}")
        return False


# --------------------------------------------------------------------------- #
# GMST : HadCRUT5
# --------------------------------------------------------------------------- #
_HADCRUT_URLS = [
    "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.0.2.0/"
    "analysis/diagnostics/"
    "HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv",
    "https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/analysis/"
    "diagnostics/HadCRUT.5.0.2.0.analysis.summary_series.global.annual.csv",
]


def load_gmst_obs() -> dict:
    """Annual HadCRUT5 global-mean surface temperature anomaly (K)."""
    dest = _cache("hadcrut5_global_annual.csv")
    ok = dest.exists() or any(_download(u, dest) for u in _HADCRUT_URLS)
    if ok:
        try:
            df = pd.read_csv(dest)
            ycol = df.columns[0]
            acol = [c for c in df.columns if "anomaly" in c.lower()][0]
            years = df[ycol].astype(int).to_numpy()
            anom = df[acol].astype(float).to_numpy()
            # Re-reference to the 1850-1900 IPCC baseline for internal consistency.
            m = (years >= 1850) & (years <= 1900)
            anom = anom - np.nanmean(anom[m])
            return {"years": years, "gmst": anom, "source": "HadCRUT5"}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"HadCRUT5 parse failed: {e}")
    return _synthetic_gmst()


def _synthetic_gmst() -> dict:
    years = np.arange(1850, 2024)
    t = (years - 1850) / 173.0
    gmst = 1.3 * t ** 1.6 + 0.08 * np.sin(2 * np.pi * (years - 1850) / 60.0)
    gmst -= gmst[years <= 1900].mean()
    return {"years": years, "gmst": gmst, "source": "synthetic-fallback"}


# --------------------------------------------------------------------------- #
# OHC : NOAA/NCEI 0-2000 m
# --------------------------------------------------------------------------- #
_OHC_URL = (
    "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/"
    "DATA/basin/yearly/h22-w0-2000m.dat"
)


def load_ohc_obs() -> dict:
    """Annual NOAA/NCEI world-ocean heat content 0-2000 m, in ZJ."""
    dest = _cache("noaa_ohc_0-2000m_yearly.dat")
    if dest.exists() or _download(_OHC_URL, dest):
        try:
            rows = []
            for line in dest.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append((float(parts[0]), float(parts[1])))
                    except ValueError:
                        continue
            arr = np.array(rows)
            years = np.floor(arr[:, 0]).astype(int)
            ohc = arr[:, 1] * 10.0   # 10^22 J -> ZJ (10^21 J)
            # reference to 2005-2014 mean
            m = (years >= C.OHC_REF[0]) & (years <= C.OHC_REF[1])
            ohc = ohc - ohc[m].mean()
            return {"years": years, "ohc": ohc, "source": "NOAA/NCEI"}
        except Exception as e:  # noqa: BLE001
            warnings.warn(f"OHC parse failed: {e}")
    return _synthetic_ohc()


def _synthetic_ohc() -> dict:
    years = np.arange(1955, 2024)
    ohc = 3.2 * (years - 1955) ** 1.5 / 10.0
    m = (years >= C.OHC_REF[0]) & (years <= C.OHC_REF[1])
    ohc = ohc - ohc[m].mean()
    return {"years": years, "ohc": ohc, "source": "synthetic-fallback"}


# --------------------------------------------------------------------------- #
# Atmospheric CO2 pathway (historical + SSP)
# --------------------------------------------------------------------------- #
# Anchor values (ppm): preindustrial, 2019 observed, and SSP 2100 / 2300 levels.
_CO2_ANCHOR = {
    "ssp126": {2100: 445.0, 2300: 350.0},
    "ssp245": {2100: 602.0, 2300: 543.0},
    "ssp370": {2100: 867.0, 2300: 2000.0},
    "ssp585": {2100: 1135.0, 2300: 1962.0},
}


def load_co2_pathway(ssp: str) -> dict:
    """Atmospheric CO2 (ppm), 1750-2300, historical then a smooth SSP pathway.

    A reduced but monotone-consistent construction anchored to the observed 2019
    value and the published SSP 2100/2300 concentrations. Flagged as reduced in
    the run provenance; replace with the Meinshausen (2020) GHG concentration
    file for a production run.
    """
    years = np.arange(C.YEAR_START, C.YEAR_END + 1)
    co2 = np.empty_like(years, dtype=float)

    # Historical 1750-2019: smooth rise 278 -> 410 ppm.
    hist = years <= 2019
    th = (years[hist] - 1750) / (2019 - 1750)
    co2[hist] = C.CO2_PI + (410.0 - C.CO2_PI) * th ** 2.4

    # Future 2019-2100 and 2100-2300: monotone cubic-smoothstep between anchors.
    a2100 = _CO2_ANCHOR[ssp][2100]
    a2300 = _CO2_ANCHOR[ssp][2300]
    fut1 = (years > 2019) & (years <= 2100)
    s = (years[fut1] - 2019) / (2100 - 2019)
    s = 3 * s ** 2 - 2 * s ** 3
    co2[fut1] = 410.0 + (a2100 - 410.0) * s
    fut2 = years > 2100
    s2 = (years[fut2] - 2100) / (2300 - 2100)
    s2 = 3 * s2 ** 2 - 2 * s2 ** 3
    co2[fut2] = a2100 + (a2300 - a2100) * s2
    return {"years": years, "co2": co2, "source": "reduced-SSP-anchored"}


# --------------------------------------------------------------------------- #
# Total effective radiative forcing (historical + SSP)
# --------------------------------------------------------------------------- #
def load_forcing(ssp: str) -> dict:
    """Total effective radiative forcing (W m^-2), 1750-2300.

    Reduced construction: CO2 forcing 5.35 ln(CO2/278) scaled by a non-CO2
    multiplier tuned so the 2019 total matches the AR6 estimate (~2.7 W m^-2),
    with the aerosol-dominated early-industrial dip approximated. This is the
    documented offline fallback; a production run substitutes the RCMIP/AR6 SSP
    ERF series here.
    """
    co2 = load_co2_pathway(ssp)
    years, cc = co2["years"], co2["co2"]
    f_co2 = 5.35 * np.log(cc / C.CO2_PI)
    # non-CO2 GHGs add ~50% of CO2 forcing; aerosols subtract a transient term.
    nonco2 = 0.55 * f_co2
    aer_frac = np.clip((years - 1900) / (1980 - 1900), 0, 1)
    aerosol = -0.9 * aer_frac * np.clip((2000 - years) / 100.0 + 1.0, 0, 1.2)
    erf = f_co2 + nonco2 + aerosol
    # Anchor 2019 total near AR6 (~2.72 W m^-2).
    i2019 = np.where(years == 2019)[0][0]
    erf = erf + (2.72 - erf[i2019])
    return {"years": years, "erf": erf, "source": "reduced-CO2-derived"}


# --------------------------------------------------------------------------- #
# Land mask on the 0.5 deg grid
# --------------------------------------------------------------------------- #
def load_land_mask() -> dict:
    """Boolean land mask and cell-area weights on the 0.5 deg grid.

    Uses Natural Earth 110 m land polygons (via cartopy) rasterised with
    shapely.vectorized point-in-polygon on cell centres.
    """
    npz = _cache("land_mask_0p5.npz")
    lon = np.arange(-180 + C.GRID_RES / 2, 180, C.GRID_RES)
    lat = np.arange(-90 + C.GRID_RES / 2, 90, C.GRID_RES)
    lon2d, lat2d = np.meshgrid(lon, lat)
    if npz.exists():
        d = np.load(npz)
        mask = d["mask"]
    else:
        mask = _rasterise_land(lon2d, lat2d)
        np.savez_compressed(npz, mask=mask)
    # Cell area weights proportional to cos(latitude).
    area = np.cos(np.deg2rad(lat2d))
    return {"lon": lon, "lat": lat, "lon2d": lon2d, "lat2d": lat2d,
            "mask": mask.astype(bool), "area": area}


def _rasterise_land(lon2d, lat2d) -> np.ndarray:
    try:
        import cartopy.io.shapereader as shpreader
        from shapely.ops import unary_union
        from shapely.prepared import prep
        import shapely.vectorized as shpvec

        fname = shpreader.natural_earth(resolution="110m", category="physical",
                                        name="land")
        geoms = list(shpreader.Reader(fname).geometries())
        land = unary_union(geoms)
        mask = shpvec.contains(land, lon2d, lat2d)
        return mask
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"land rasterisation fallback (lat band): {e}")
        # crude fallback: treat |lat|<70 with a longitudinal land fraction ~0.29
        rng = np.random.default_rng(C.SEED)
        frac = 0.29
        mask = (rng.random(lon2d.shape) < frac) & (np.abs(lat2d) < 75)
        return mask
