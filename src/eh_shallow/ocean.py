"""Ocean carbonate-system diagnostics via PyCO2SYS.

Surface-ocean pH and aragonite saturation (Omega_arag) are diagnosed from the
prescribed surface-ocean pCO2 (taken equal to the atmospheric CO2 pathway) and
the emulated sea-surface temperature, at fixed total alkalinity. They are
diagnostic functions of the CO2 pathway and SST, not independent prognostic
states, and neither enters the SMC likelihood.
"""
from __future__ import annotations

import numpy as np

from . import config as C

try:
    import PyCO2SYS as pyco2
    _HAVE_PYCO2 = True
except Exception:  # pragma: no cover - fallback path
    _HAVE_PYCO2 = False


def carbonate_diagnostics(co2_ppm: np.ndarray, sst_anom: np.ndarray) -> dict:
    """Return surface-ocean pH (total scale) and aragonite saturation Omega.

    Parameters
    ----------
    co2_ppm : atmospheric CO2 pathway (ppm); used as surface-ocean pCO2 (uatm).
    sst_anom : sea-surface temperature anomaly (K) added to the preindustrial
        open-marine reference mean.
    """
    co2_ppm = np.asarray(co2_ppm, dtype=float)
    temperature = C.SST_PI_MEAN + np.asarray(sst_anom, dtype=float)

    if _HAVE_PYCO2:
        results = pyco2.sys(
            par1=C.ALKALINITY,          # total alkalinity (umol/kg)
            par2=co2_ppm,               # pCO2 (uatm)
            par1_type=1,
            par2_type=4,
            temperature=temperature,    # deg C
            salinity=35.0,
            pressure=0.0,
        )
        ph = np.asarray(results["pH_total"], dtype=float)
        omega = np.asarray(results["saturation_aragonite"], dtype=float)
    else:  # transparent empirical fallback (documented; not used if PyCO2SYS present)
        ph = 8.15 - 0.0016 * (co2_ppm - C.CO2_PI)
        omega = np.clip(3.5 - 0.0009 * (co2_ppm - C.CO2_PI), 0.2, None)

    # Broadcast scalars to the CO2 vector shape.
    ph = np.broadcast_to(ph, co2_ppm.shape).astype(float)
    omega = np.broadcast_to(omega, co2_ppm.shape).astype(float)
    return {"ph": ph, "omega_arag": omega, "pyco2sys": _HAVE_PYCO2}
