"""eh_shallow: spatially-explicit 2-D shallow-time Earth Habitability model.

A reduced-complexity, fully reproducible proof-of-concept that runs the chain
real forcing/obs -> two-layer energy-balance emulator -> tempered SMC
calibration -> six carried climate-ocean variables -> tiered Composite Hazard
Score -> Habitable Area Fraction, from the preindustrial (1750) to 2300 CE.
"""
from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Masoud Rostami"

# --- numpy-compat shim -----------------------------------------------------
# Some cluster/legacy environments pair a recent numpy (>=1.24, where the
# np.float/np.bool/np.int aliases were removed) with older scikit-learn and
# shapely that still reference them. Restore the builtin aliases so those
# packages import cleanly. Harmless on modern stacks. Pin versions via
# environment.yml for a fully reproducible run.
import numpy as _np  # noqa: E402
for _alias, _builtin in (("bool", bool), ("float", float), ("int", int),
                         ("object", object), ("str", str), ("complex", complex)):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _builtin)

from . import config  # noqa: F401,E402


def run(cfg=None):
    """Convenience entry point: run the full pipeline and return results."""
    from .pipeline import run as _run
    return _run(cfg)
