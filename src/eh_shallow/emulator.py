"""Two-layer (Geoffroy) energy-balance climate emulator.

The executed proof-of-concept integrates the two-layer energy-balance model of
Geoffroy et al. (2013) in the surface / deep-ocean temperature anomalies
(T, T_d), forced by the total effective radiative forcing Delta F(t):

    C_s dT/dt  = Delta F(t) - (F_2x / ECS) T - gamma (T - T_d)
    C_d dT_d/dt = gamma (T - T_d)

Only the equilibrium climate sensitivity (ECS) and the ocean heat-uptake
coefficient (gamma) are free parameters; (C_s, C_d, F_2x) are fixed at their
Geoffroy/AR6 values. With piecewise-constant annual forcing the linear system is
advanced by the exact matrix exponential (equivalent to FaIR's
``EnergyBalanceModel`` solver), so the integration introduces no time-stepping
error.

Ocean heat content is diagnosed as the ocean's fixed share f_ocean of the
time-integrated net top-of-atmosphere imbalance N(t) = Delta F(t) - (F_2x/ECS) T.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from . import config as C

# Conversion: 1 W m^-2 applied over the global ocean surface for 1 year.
# Earth surface area 5.101e14 m^2; ocean share f_ocean; 1 yr = 3.155e7 s.
# ZJ = 1e21 J.
_EARTH_AREA = 5.101e14
_SECONDS_PER_YEAR = 3.155e7
_ZJ = 1.0e21


def _wm2yr_to_zj(flux_wm2_years: np.ndarray) -> np.ndarray:
    """Convert an integrated flux (W m^-2 * years, ocean-weighted) to ZJ."""
    return flux_wm2_years * C.F_OCEAN * _EARTH_AREA * _SECONDS_PER_YEAR / _ZJ


def build_state_matrix(ecs: float, gamma: float) -> np.ndarray:
    """Continuous-time system matrix A for x=(T, T_d), dx/dt = A x + b F."""
    lam = C.F_2X / ecs                       # net feedback parameter (W m^-2 K^-1)
    a11 = -(lam + gamma) / C.C_SURF
    a12 = gamma / C.C_SURF
    a21 = gamma / C.C_DEEP
    a22 = -gamma / C.C_DEEP
    return np.array([[a11, a12], [a21, a22]])


def run_emulator(years: np.ndarray, forcing: np.ndarray, ecs: float,
                 gamma: float) -> dict:
    """Integrate the two-layer EBM over ``years`` given annual ``forcing``.

    Returns a dict with surface temperature anomaly T (K), deep-ocean anomaly
    T_d (K), net TOA imbalance N (W m^-2), and ocean heat content OHC (ZJ,
    cumulative from the first year).
    """
    years = np.asarray(years, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    n = years.size
    lam = C.F_2X / ecs

    A = build_state_matrix(ecs, gamma)
    b = np.array([1.0 / C.C_SURF, 0.0])

    # Exact solution of dx/dt = A x + b F with F constant over each year.
    # x_{k+1} = expm(A dt) x_k + A^{-1} (expm(A dt) - I) b F_k, dt = 1 yr.
    dt = 1.0
    Phi = expm(A * dt)
    Ainv = np.linalg.inv(A)
    G = Ainv @ (Phi - np.eye(2)) @ b       # forcing response over one step

    x = np.zeros((n, 2))
    for k in range(1, n):
        x[k] = Phi @ x[k - 1] + G * forcing[k - 1]

    T = x[:, 0]
    Td = x[:, 1]
    N = forcing - lam * T                  # net TOA imbalance
    # OHC anomaly relative to first year: cumulative integral of N.
    ohc_int = np.concatenate([[0.0], np.cumsum(0.5 * (N[1:] + N[:-1]))])
    ohc = _wm2yr_to_zj(ohc_int)
    return {"years": years, "T": T, "Td": Td, "N": N, "OHC": ohc, "lambda": lam}


def sst_from_gmst(T: np.ndarray, scale: float = 0.85) -> np.ndarray:
    """Diagnose sea-surface temperature anomaly from global-mean warming.

    A single ocean pattern-scaling coefficient (~0.85) converts global-mean
    surface warming to the SST anomaly; the absolute SST is referenced to the
    preindustrial open-marine mean in the ocean-chemistry module.
    """
    return scale * T
