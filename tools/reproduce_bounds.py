#!/usr/bin/env python
"""Recompute the combined 95% upper limit of one coupling from the released
histograms, with the aggregator of the pipeline, and compare it with the curve
used in the paper.

    python tools/reproduce_bounds.py linear      [--workdir DIR] [--keep]
    python tools/reproduce_bounds.py quadratic
    python tools/reproduce_bounds.py vector

Steps:
  1. unpack <coupling>/histograms/*_bounds_part*.npz into a working directory
     (one file per (mass, pulsar) chain, as written by pipeline/worker_final.py);
  2. run pipeline/aggregate_bounds.py on it;
  3. compare every array of the resulting bounds_<coupling>.npz with the one
     shipped in <coupling>/ (the file the figures of the paper are drawn from).

Run from the repository root (or anywhere: paths are resolved from this file).
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUPLINGS = {"linear": "alpha", "quadratic": "beta", "vector": "vec"}


def main():
    ap = argparse.ArgumentParser(description="Reproduce a bound curve from the histograms.")
    ap.add_argument("coupling", choices=sorted(COUPLINGS))
    ap.add_argument("--workdir", help="where to unpack (default: a temporary directory)")
    ap.add_argument("--keep", action="store_true", help="keep the unpacked histograms")
    a = ap.parse_args()

    coupling_key = COUPLINGS[a.coupling]
    parts = sorted(glob.glob(os.path.join(ROOT, a.coupling, "histograms", f"{coupling_key}_bounds_part*.npz")))
    reference = os.path.join(ROOT, a.coupling, f"bounds_{coupling_key}.npz")
    if not parts or not os.path.exists(reference):
        raise SystemExit(f"{a.coupling}: histograms or reference curve not available yet")
    work = a.workdir or tempfile.mkdtemp(prefix=f"uldm_{coupling_key}_")
    try:
        subprocess.run([sys.executable, os.path.join(ROOT, "tools", "unpack_histograms.py"), work] + parts,
                       check=True)
        env = dict(os.environ, COUPLING=coupling_key, OUT_ROOT=work)
        subprocess.run([sys.executable, os.path.join(ROOT, "pipeline", "aggregate_bounds.py")], env=env, check=True)
        new = os.path.join(work, f"bounds_{coupling_key}.npz")
        # The released curves were aggregated on the cluster; a different
        # numpy/scipy build may change the last floating-point digit, so
        # floating-point arrays are compared to TOL (in dex, or in the units
        # of the field), everything else exactly.
        TOL = 1e-9
        ok = True
        with np.load(new) as x, np.load(reference) as y:
            for k in y.files:
                if k not in x.files:
                    status, good = "MISSING", False
                elif np.array_equal(x[k], y[k]):
                    status, good = "identical", True
                elif x[k].dtype.kind == "f" and x[k].shape == y[k].shape:
                    d = float(np.nanmax(np.abs(x[k] - y[k])))
                    good = d < TOL
                    status = f"{'agrees' if good else 'DIFFERENT'} (max |diff| = {d:.1e})"
                else:
                    status, good = "DIFFERENT", False
                ok &= good
                print(f"  {k:18s} {status}")
        print(f"{a.coupling}: {'reproduced' if ok else 'NOT reproduced'} ({reference})")
        if not ok:
            sys.exit(1)
    finally:
        if not a.keep and not a.workdir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
