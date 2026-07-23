"""Smoke test for the shallow-model prototype: keeps the pipeline green.

Run with `python tests/test_eh_shallow.py` (no pytest needed) or `pytest`.
Network is required only for the SMC test (it pulls HadCRUT5); the emulator/CHS
tests run fully offline.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eh_shallow import chs, emulator  # noqa: E402


def test_emulator_finite_and_warming():
    years = np.arange(1750, 2301)
    out = emulator.run_emulator({"ecs": 3.0, "gamma": 0.7}, years)
    for k in ("gmst", "ohc", "co2", "ph", "omega", "sst"):
        assert out[k].shape == years.shape
        assert np.all(np.isfinite(out[k])), f"{k} has non-finite values"
    # more sensitivity -> more warming by 2100
    hot = emulator.run_emulator({"ecs": 5.0, "gamma": 0.7}, years)
    assert hot["gmst"][years == 2100][0] > out["gmst"][years == 2100][0]
    # ocean acidifies and OHC rises
    assert out["ph"][years == 2100][0] < out["ph"][years == 1800][0]
    assert out["ohc"][years == 2020][0] > 0


def test_chs_and_haf_in_range():
    years = np.arange(1750, 2301)
    out = emulator.run_emulator({"ecs": 3.7, "gamma": 0.7}, years)
    cg = chs.composite_hazard(out)
    assert cg.shape == years.shape and np.all(np.isfinite(cg))
    h = chs.haf_trajectory(out)
    haf = h["haf"]
    assert np.all((haf >= 0) & (haf <= 1)), "HAF out of [0,1]"
    # habitability degrades over time under rising hazard
    assert haf[years == 2300][0] < haf[years == 1775][0]
    # smooth exceedance also in range
    assert np.all((h["haf_smooth"] >= 0) & (h["haf_smooth"] <= 1))


def test_prior_sampling_shape():
    rng = np.random.default_rng(0)
    p = emulator.sample_prior(rng, 100)
    assert p.shape == (100, len(emulator.PARAM_NAMES))
    assert np.all(np.isfinite(emulator.log_prior(p)[p[:, 0] > 0]))


def test_provenance_guard_blocks_synthetic_fallback():
    """The data-provenance guard must refuse synthetic fallback data (offline-safe)."""
    from eh_shallow import data
    data.reset_provenance()
    # a real/downloaded source is not a fallback -> guard passes (no raise)
    assert data._record("HadCRUT5 (Met Office), downloaded")
    assert not data.fallbacks_used()
    data.assert_real_data(context="test")                 # must NOT raise
    # a synthetic embedded fallback is detected and the guard refuses
    data._record("EMBEDDED FALLBACK (no network)")
    assert data.fallbacks_used()
    try:
        data.assert_real_data(context="test")
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "assert_real_data must refuse when a synthetic fallback was used"
    # an analytic stand-in baseline passed via `extra` is also refused
    data.reset_provenance()
    try:
        data.assert_real_data(extra=["analytic stand-in baseline (WHI raster absent)"])
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "assert_real_data must refuse an analytic stand-in baseline"
    data.reset_provenance()


def test_smc_small_runs():
    """Tiny SMC end-to-end (needs network for HadCRUT5); skip gracefully if down."""
    from eh_shallow import data, smc
    if data.load_hadcrut5().attrs.get("source", "").startswith("EMBEDDED"):
        print("  [skip] no network -> embedded fallback; SMC test skipped")
        return
    post = smc.run_smc(n_particles=40, n_temps=4, n_rejuv=1, seed=0, verbose=False)
    summ = smc.posterior_summary(post)
    assert 1.0 < summ["ecs"]["mean"] < 6.0
    assert np.all(np.isfinite(post["weights"]))


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} smoke tests passed.")
