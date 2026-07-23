"""eh_shallow -- prototype vertical slice of the shallow (1750-2300) habitability model.

This package implements the minimal end-to-end pipeline described in EH_shallow.tex,
deliberately as a *reduced* model for fast iteration:

    real forcing (AR6 ERF + CO2)  ->  reduced 2-box climate emulator
    ->  a handful of multi-sphere variables  ->  Composite Hazard Score (CHS)
    ->  Habitable Area Fraction (HAF)  ->  tempered-SMC calibration vs HadCRUT5.

It is a PROOF OF PIPELINE, not the production model. The hardened proposal calls
for FaIR as the climate/carbon core and PyCO2SYS for ocean chemistry; the
`emulator` module exposes a `run_emulator(theta, ...)` seam where those drop in.
Every reduced form here is flagged in its docstring.
"""

__all__ = ["data", "emulator", "chs", "smc", "plots"]
__version__ = "0.1.0"
