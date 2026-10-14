#!/usr/bin/env python
"""Recompute the combined 95% upper limit of one coupling from the released
histograms, with the production aggregator, and compare it with the curve
used in the paper.

    python tools/reproduce_bounds.py linear      [--workdir DIR] [--keep]
    python tools/reproduce_bounds.py quadratic
    python tools/reproduce_bounds.py vector

Steps:
  1. unpack <coupling>/histograms/*_bounds_part*.npz into a working directory
     (one file per (mass, pulsar) chain, as written by pipeline/worker_final.py);
  2. run pipeline/agregar_cotas.py on it, unchanged;
  3. compare every array of the resulting COTAS_<coupling>.npz with the one
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

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACOPLE = {"linear": "alpha", "quadratic": "beta", "vector": "vec"}


def main():
    ap = argparse.ArgumentParser(description="Reproduce a bound curve from the histograms.")
    ap.add_argument("coupling", choices=sorted(ACOPLE))
    ap.add_argument("--workdir", help="where to unpack (default: a temporary directory)")
    ap.add_argument("--keep", action="store_true", help="keep the unpacked histograms")
    a = ap.parse_args()

    acople = ACOPLE[a.coupling]
    partes = sorted(glob.glob(os.path.join(RAIZ, a.coupling, "histograms", f"{acople}_bounds_part*.npz")))
    referencia = os.path.join(RAIZ, a.coupling, f"COTAS_{acople}.npz")
    if not partes or not os.path.exists(referencia):
        raise SystemExit(f"{a.coupling}: histograms or reference curve not available yet")
    work = a.workdir or tempfile.mkdtemp(prefix=f"uldm_{acople}_")
    try:
        subprocess.run([sys.executable, os.path.join(RAIZ, "tools", "unpack_histograms.py"), work] + partes,
                       check=True)
        env = dict(os.environ, ACOPLE=acople, OUT_ROOT=work)
        subprocess.run([sys.executable, os.path.join(RAIZ, "pipeline", "agregar_cotas.py")], env=env, check=True)
        nuevo = os.path.join(work, f"COTAS_{acople}.npz")
        # The released curves were aggregated on the cluster; a different
        # numpy/scipy build may change the last floating-point digit, so
        # floating-point arrays are compared to TOL (in dex, or in the units
        # of the field), everything else exactly.
        TOL = 1e-9
        ok = True
        with np.load(nuevo) as x, np.load(referencia) as y:
            for k in y.files:
                if k not in x.files:
                    estado, bien = "MISSING", False
                elif np.array_equal(x[k], y[k]):
                    estado, bien = "identical", True
                elif x[k].dtype.kind == "f" and x[k].shape == y[k].shape:
                    d = float(np.nanmax(np.abs(x[k] - y[k])))
                    bien = d < TOL
                    estado = f"{'agrees' if bien else 'DIFFERENT'} (max |diff| = {d:.1e})"
                else:
                    estado, bien = "DIFFERENT", False
                ok &= bien
                print(f"  {k:18s} {estado}")
        print(f"{a.coupling}: {'reproduced' if ok else 'NOT reproduced'} ({referencia})")
        if not ok:
            sys.exit(1)
    finally:
        if not a.keep and not a.workdir:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
