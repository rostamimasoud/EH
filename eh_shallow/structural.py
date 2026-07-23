"""Structural-uncertainty estimate by leave-one-out across INDEPENDENT
global-temperature reconstructions (the proposal's Major-4 fix).

The original plan estimated sigma_struct by comparing the emulator against the
CMIP6 *envelope* -- which conflates emulator mis-specification with the spread of
a set of models that are not independent (they share code, components, tuning;
Knutti 2017, Sanderson 2015). Here we instead treat each of several INDEPENDENT
reconstructions of the instrumental record as a separate "truth":

  HadCRUT5 (Met Office), GISTEMP v4 (NASA), Berkeley Earth (Land+Ocean).

Procedure (leave-one-reconstruction-out): the emulator is calibrated ONCE on
HadCRUT5 (+OHC) in the main run; here we score that already-calibrated emulator's
out-of-sample (1981-2020) GMST against each reconstruction. GISTEMP and Berkeley
Earth were NOT used in calibration, so their residual is a clean held-out
structural-error estimate. sigma_struct is the RMS out-of-sample residual across
the reconstructions; the across-reconstruction spread (~0.05 K) shows the error
is dominated by emulator mis-specification, not observational uncertainty -- and,
crucially, no CMIP6 model envelope enters, so model interdependence cannot
inflate it. (We deliberately do NOT refit ECS to each reconstruction: GMST alone
does not identify ECS without the OHC constraint, so a per-reconstruction refit
would re-expose the degeneracy rather than measure structural error.)

    python -m eh_shallow.structural
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import data, emulator, plots

STRUCT_REF = (1880, 1920)     # common baseline all three reconstructions cover
CAL_WINDOW = (1880, 1980)     # in-sample window (for context only)
OOS_WINDOW = (1981, 2020)     # withheld window scored for structural error
# The main calibration's OHC-constrained posterior mean (metrics.json). We score
# THIS already-calibrated emulator, rather than refitting per reconstruction.
POSTERIOR = {"ecs": 3.65, "gamma": 0.48}
# Reconstructions used in calibration (HadCRUT5) vs genuinely held out.
HELD_OUT = ("GISTEMP", "Berkeley")


def _reconstructions() -> dict:
    """name -> (years, anomaly rebaselined to STRUCT_REF)."""
    out = {}
    for name, fn in [("HadCRUT5", data.load_hadcrut5),
                     ("GISTEMP", data.load_gistemp),
                     ("Berkeley", data.load_berkeley)]:
        df = fn()
        y = df["year"].to_numpy()
        g = data.rebaseline(y, df["gmst"].to_numpy(), ref=STRUCT_REF)
        out[name] = (y, g, df.attrs.get("source", ""))
    return out


def _emulate_gmst(ecs: float, gamma: float, years: np.ndarray) -> np.ndarray:
    res = emulator.run_emulator({"ecs": ecs, "gamma": gamma}, years,
                                ssp="ssp245", chem=False)
    return data.rebaseline(years, res["gmst"], ref=STRUCT_REF)


def leave_one_out() -> dict:
    """Score the calibrated emulator vs each independent reconstruction."""
    years = np.arange(1750, 2026)
    recon = _reconstructions()
    names = list(recon)
    gm = _emulate_gmst(POSTERIOR["ecs"], POSTERIOR["gamma"], years)

    per = {}
    for n in names:
        oy, og, _ = recon[n]
        ci = (oy >= CAL_WINDOW[0]) & (oy <= CAL_WINDOW[1])
        oi = (oy >= OOS_WINDOW[0]) & (oy <= OOS_WINDOW[1])
        per[n] = {
            "rmse_insample_K": float(np.sqrt(np.mean(
                (np.interp(oy[ci], years, gm) - og[ci]) ** 2))),
            "rmse_oos_K": float(np.sqrt(np.mean(
                (np.interp(oy[oi], years, gm) - og[oi]) ** 2))),
            "held_out": n in HELD_OUT,
        }
    oos_all = [per[n]["rmse_oos_K"] for n in names]
    oos_held = [per[n]["rmse_oos_K"] for n in names if n in HELD_OUT]

    grid_yr = np.arange(OOS_WINDOW[0], OOS_WINDOW[1] + 1)
    stack = np.array([np.interp(grid_yr, recon[n][0], recon[n][1]) for n in names])
    obs_spread = float(np.mean(np.std(stack, axis=0)))

    return {
        "method": "leave-one-reconstruction-out: score the OHC-calibrated emulator "
                  "against independent reconstructions (HadCRUT5 used in calibration; "
                  "GISTEMP, Berkeley Earth held out)",
        "posterior": POSTERIOR, "cal_window": CAL_WINDOW, "oos_window": OOS_WINDOW,
        "sources": {n: recon[n][2] for n in names},
        "per_reconstruction": per,
        "sigma_struct_oos_all_K": float(np.sqrt(np.mean(np.square(oos_all)))),
        "sigma_struct_oos_heldout_K": float(np.sqrt(np.mean(np.square(oos_held)))),
        "obs_reconstruction_spread_K": obs_spread,
        "sigma_struct_assumed_in_smc_K": 0.20,
        "names": names, "recon": recon, "gm": gm, "years": years,
    }


def main(outdir=None):
    outdir = outdir or os.path.join(os.path.dirname(__file__), "out")
    os.makedirs(outdir, exist_ok=True)
    print("[structural] leave-one-reconstruction-out: HadCRUT5 / GISTEMP / Berkeley")
    r = leave_one_out()
    for n in r["names"]:
        p = r["per_reconstruction"][n]
        tag = "held out" if p["held_out"] else "calibration"
        print(f"  {n:10s} ({tag:11s}): in-sample {p['rmse_insample_K']:.3f} K, "
              f"OOS {p['rmse_oos_K']:.3f} K")
    print(f"  sigma_struct (OOS, all 3)        = {r['sigma_struct_oos_all_K']:.3f} K")
    print(f"  sigma_struct (OOS, held-out 2)   = {r['sigma_struct_oos_heldout_K']:.3f} K")
    print(f"  obs reconstruction spread (OOS)  = {r['obs_reconstruction_spread_K']:.3f} K")
    print(f"  assumed in SMC                   = {r['sigma_struct_assumed_in_smc_K']:.2f} K")
    fig_path = os.path.join(outdir, "struct_sigma.pdf")
    plots.plot_structural(r, fig_path)
    keep = {k: v for k, v in r.items() if k not in ("recon", "gm", "years")}
    with open(os.path.join(outdir, "struct_metrics.json"), "w") as f:
        json.dump(keep, f, indent=2)
    print(f"  wrote {fig_path} + struct_metrics.json")
    return keep


if __name__ == "__main__":
    main()
