"""Regression tests for the (ECS, gamma) calibration likelihood.

Guards the 1850-1900 baseline referencing in ``smc._log_likelihood_single``.
A missing re-reference makes the emulator's ECS-dependent 1750->1850 warming
enter as a spurious offset, inverting the GMST likelihood so the posterior
collapses to the lower ECS bound. These tests assert the likelihood is
maximised at a physically sensible ECS (near the N(3.0, 0.5) prior), not at
the lower bound.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from eh_shallow import config as C, data          # noqa: E402
from eh_shallow.smc import _log_likelihood_single  # noqa: E402


@pytest.fixture(scope="module")
def obs_and_forcing():
    f = data.load_forcing(C.HEADLINE_SSP)
    g = data.load_gmst_obs()
    o = data.load_ohc_obs()
    obs = {"gmst_years": g["years"], "gmst_vals": g["gmst"],
           "ohc_years": o["years"], "ohc_vals": o["ohc"]}
    return f, obs


def _ll_curve(f, obs, ecs_grid, gamma=0.7):
    return np.array([_log_likelihood_single(np.array([e, gamma]),
                                            f["years"], f["erf"], obs)
                     for e in ecs_grid])


def test_likelihood_peaks_near_prior(obs_and_forcing):
    """The MLE ECS (gamma fixed) should sit well above the lower bound."""
    f, obs = obs_and_forcing
    ecs = np.array([0.6, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ll = _ll_curve(f, obs, ecs)
    best = ecs[int(np.argmax(ll))]
    assert best >= 2.0, f"likelihood peaks at ECS={best} (calibration inverted?)"


def test_likelihood_not_monotone_decreasing(obs_and_forcing):
    """Symptom of the baseline bug: ll strictly decreasing from the low bound."""
    f, obs = obs_and_forcing
    ecs = np.array([0.6, 1.5, 2.5, 3.0, 4.0])
    ll = _ll_curve(f, obs, ecs)
    assert not np.all(np.diff(ll) < 0), "log-likelihood monotonically decreasing in ECS"
    assert ll[ecs == 3.0][0] > ll[ecs == 0.6][0], "ECS=3 must beat ECS=0.6"
