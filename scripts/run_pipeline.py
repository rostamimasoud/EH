#!/usr/bin/env python3
"""Driver: run the shallow-time EH pipeline, then render the paper figures.

The library ``eh_shallow.pipeline.run`` produces ``metrics.json`` and
``results.npz``; it does *not* draw figures. This driver chains the two so a
single command reproduces every headline number and every publication figure.

Usage
-----
    python scripts/run_pipeline.py                 # full headline run + figures
    python scripts/run_pipeline.py --fast          # subsampled smoke test
    python scripts/run_pipeline.py --no-figs       # metrics/arrays only
    python scripts/run_pipeline.py --particles 800 # override SMC ensemble size

Outputs land in ``$EH_OUTPUT_DIR`` (default ``EH/outputs``) and, when
``write_paper_figs`` is set, are mirrored to ``$EH_PAPER_FIG_DIR``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make ``src`` importable whether or not the package is pip-installed.
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fast", action="store_true",
                   help="subsample particles for a quick smoke test")
    p.add_argument("--seed", type=int, default=None,
                   help="override the reproducibility seed")
    p.add_argument("--particles", type=int, default=None,
                   help="override the number of SMC particles")
    p.add_argument("--temp-steps", type=int, default=None,
                   help="override the number of tempering steps")
    p.add_argument("--headline-ssp", type=str, default=None,
                   help="override the headline SSP (e.g. ssp245)")
    p.add_argument("--no-figs", action="store_true",
                   help="skip figure rendering (metrics + arrays only)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    from eh_shallow import config as C

    kw = {"fast": args.fast}
    if args.seed is not None:
        kw["seed"] = args.seed
    if args.particles is not None:
        kw["n_particles"] = args.particles
    if args.temp_steps is not None:
        kw["n_temp_steps"] = args.temp_steps
    if args.headline_ssp is not None:
        kw["headline_ssp"] = args.headline_ssp
    kw["write_paper_figs"] = not args.no_figs
    cfg = C.RunConfig(**kw)

    print(f"[run_pipeline] fast={cfg.fast} seed={cfg.seed} "
          f"n_particles={cfg.n_particles} n_temp_steps={cfg.n_temp_steps} "
          f"headline_ssp={cfg.headline_ssp}", flush=True)
    print(f"[run_pipeline] OUTPUT_DIR={C.OUTPUT_DIR}", flush=True)
    print(f"[run_pipeline] PAPER_FIG_DIR={C.PAPER_FIG_DIR} "
          f"(exists={C.PAPER_FIG_DIR.exists()})", flush=True)

    t0 = time.time()
    from eh_shallow.pipeline import run
    results = run(cfg)
    print(f"[run_pipeline] pipeline done in {time.time() - t0:.1f}s", flush=True)
    print(json.dumps(results["metrics"], indent=2), flush=True)

    if not args.no_figs:
        from eh_shallow import figures
        t1 = time.time()
        written = figures.make_all(results)
        print(f"[run_pipeline] {len(written)} figures in {time.time() - t1:.1f}s: "
              f"{', '.join(written)}", flush=True)
        if cfg.write_paper_figs and C.PAPER_FIG_DIR.exists():
            print(f"[run_pipeline] mirrored to {C.PAPER_FIG_DIR}", flush=True)

    print(f"[run_pipeline] total {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
