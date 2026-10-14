#!/usr/bin/env python
"""Pack the per-chain histograms written by ``pipeline/worker_final.py`` into a
few compressed archives per coupling and run mode.

The worker writes one small ``.npz`` per (mass, pulsar) chain,

    <OUT_ROOT>/hist/m_<mass>__<pulsar>__s0.npz

which is convenient on a cluster but impractical to distribute (tens of
thousands of files).  This script stores the same information losslessly: the
histogram counts as a dense ``(n_masses, n_pulsars, n_bins)`` array (as
``uint16`` when every count fits, which is checked), every per-chain scalar as
an ``(n_masses, n_pulsars)`` array, and the scalars shared by all chains once.
The masses are split into parts of at most ``--masses-per-part`` so that each
file stays well below the GitHub size limits.  ``unpack_histograms.py``
restores the original files exactly, so the production aggregators run on them
unchanged.

Usage:
    python pack_histograms.py <hist_dir> <output_prefix> [--masses-per-part N]

writes <output_prefix>_part01of05.npz, ...
"""
import argparse
import glob
import os
import re

import numpy as np

# per-chain scalars stored by worker_final.py (besides m_eV, m_str, name, counts)
PER_CHAIN = ["semilla", "ess", "n_muestras", "n_fuera", "n_toa", "ecc", "n_iter",
             "x_min", "x_max", "frac_techo", "frac_piso", "t_min"]
# scalars that must be identical for every chain of a run
SHARED = ["acople", "modo", "prior", "version", "likelihood", "log10_min", "log10_max", "n_bins"]
PATRON = re.compile(r"m_(?P<m>[^_]+)__(?P<psr>.+)__s0\.npz$")


def main(hist_dir, prefix, per_part):
    files = sorted(glob.glob(os.path.join(hist_dir, "m_*__*__s0.npz")))
    if not files:
        raise SystemExit(f"no histograms in {hist_dir}")
    keys = [PATRON.search(os.path.basename(f)) for f in files]
    masses = sorted({k["m"] for k in keys}, key=float)
    pulsars = sorted({k["psr"] for k in keys})
    im = {m: i for i, m in enumerate(masses)}
    ip = {p: i for i, p in enumerate(pulsars)}

    with np.load(files[0], allow_pickle=False) as z:
        shared = {k: z[k][()] for k in SHARED}
        n_bins = int(shared["n_bins"])
        dtypes = {k: z[k].dtype for k in PER_CHAIN}
        counts_dtype = z["counts"].dtype

    counts = np.zeros((len(masses), len(pulsars), n_bins), dtype=counts_dtype)
    present = np.zeros((len(masses), len(pulsars)), dtype=bool)
    per = {k: np.zeros((len(masses), len(pulsars)), dtype=dtypes[k]) for k in PER_CHAIN}
    for f, k in zip(files, keys):
        i, j = im[k["m"]], ip[k["psr"]]
        with np.load(f, allow_pickle=False) as z:
            for s in SHARED:
                if z[s][()] != shared[s]:
                    raise SystemExit(f"{f}: {s}={z[s][()]!r} differs from {shared[s]!r}")
            if str(z["m_str"]) != k["m"] or str(z["name"]) != k["psr"]:
                raise SystemExit(f"{f}: file name and contents disagree")
            if z["counts"].dtype != counts_dtype:
                raise SystemExit(f"{f}: counts dtype {z['counts'].dtype} != {counts_dtype}")
            counts[i, j] = z["counts"]
            for s in PER_CHAIN:
                per[s][i, j] = z[s][()]
        present[i, j] = True

    stored = counts
    if counts.min() >= 0 and counts.max() <= np.iinfo(np.uint16).max:
        stored = counts.astype(np.uint16)          # lossless, checked above
    os.makedirs(os.path.dirname(os.path.abspath(prefix)), exist_ok=True)
    n_parts = -(-len(masses) // per_part)
    cortes = np.array_split(np.arange(len(masses)), n_parts)
    for p, idx in enumerate(cortes, 1):
        out = f"{prefix}_part{p:02d}of{n_parts:02d}.npz"
        np.savez_compressed(out, masses_str=np.array(masses)[idx], pulsars=np.array(pulsars),
                            present=present[idx], counts=stored[idx],
                            counts_dtype=np.array(str(counts_dtype)),
                            **{f"chain_{k}": v[idx] for k, v in per.items()},
                            **{f"run_{k}": np.array(v) for k, v in shared.items()})
        print(f"{out}: {int(present[idx].sum())} chains, {len(idx)} masses x {len(pulsars)} pulsars, "
              f"{os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hist_dir")
    ap.add_argument("prefix")
    ap.add_argument("--masses-per-part", type=int, default=250)
    a = ap.parse_args()
    main(a.hist_dir, a.prefix, a.masses_per_part)
