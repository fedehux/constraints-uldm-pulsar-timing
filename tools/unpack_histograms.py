#!/usr/bin/env python
"""Restore the per-chain histogram files from the archives written by
``pack_histograms.py``.

Each chain is written back as

    <out_dir>/hist/m_<mass>__<pulsar>__s0.npz

with the same keys, values and types as the file produced by
``pipeline/worker_final.py``, so the production aggregators
(``pipeline/agregar_cotas.py``, ``pipeline/agregar_bf.py``) run on
``<out_dir>`` with ``OUT_ROOT=<out_dir>``.

Usage:
    python unpack_histograms.py <out_dir> <archive_part01.npz> [<archive_part02.npz> ...]
"""
import os
import sys

import numpy as np


def unpack(archive, hist):
    n = 0
    with np.load(archive, allow_pickle=False) as z:
        masses = [str(m) for m in z["masses_str"]]
        pulsars = [str(p) for p in z["pulsars"]]
        present = z["present"]
        counts = z["counts"].astype(str(z["counts_dtype"]))
        per = {k[len("chain_"):]: z[k] for k in z.files if k.startswith("chain_")}
        shared = {k[len("run_"):]: z[k][()] for k in z.files if k.startswith("run_")}
    for i, m in enumerate(masses):
        for j, p in enumerate(pulsars):
            if not present[i, j]:
                continue
            np.savez_compressed(
                os.path.join(hist, f"m_{m}__{p}__s0.npz"),
                m_eV=float(m), m_str=m, name=p, counts=counts[i, j],
                **{k: v[i, j] for k, v in per.items()}, **shared)
            n += 1
    return n


def main(out_dir, archives):
    hist = os.path.join(out_dir, "hist")
    os.makedirs(hist, exist_ok=True)
    for a in archives:
        print(f"{a}: {unpack(a, hist)} histograms written to {hist}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
