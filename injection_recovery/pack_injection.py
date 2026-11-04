#!/usr/bin/env python
"""Pack the chains written by ``run_injection.py`` into one archive per coupling.

Input:  <root>/<coupling>/<level>/hist/m_<mass>__<pulsar>__s0.npz,
        level = no_signal and x4.
Output: <out_dir>/injection_<coupling>.npz with
    levels (L,), masses_str (M,), pulsars (P,)
    counts            (L, M, P, n_bins)   histogram of log10 of the coupling
    chain_<key>       (L, M, P)           per-chain scalars (injected field, SNR^2, ESS, ...)
    run_<key>         ()                  scalars shared by every chain (prior range, version, ...)

    python injection_recovery/pack_injection.py <root> <out_dir> [alpha beta vec]
"""
import glob
import os
import re
import sys

import numpy as np

LEVELS = ["no_signal", "x4"]
PER_CHAIN = ["seed", "ess", "n_samples", "n_outside", "n_toa", "ecc", "x_min", "x_max",
             "frac_ceiling", "frac_floor", "x_inj", "rho_inj", "ups_inj", "cos_t_inj",
             "varphi_inj", "snr2", "acc", "t_min"]
SHARED = ["coupling", "prior", "version", "likelihood", "log10_min", "log10_max", "n_bins", "n_iter"]
PATTERN = re.compile(r"m_(?P<m>[^_]+)__(?P<psr>.+)__s0\.npz$")


def files_of(root, coupling, level):
    fs = sorted(glob.glob(os.path.join(root, coupling, level, "hist", "m_*__*__s0.npz")))
    if fs:
        return fs
    raise SystemExit(f"{coupling}: no chains for level {level} under {root}")


def pack(root, out_dir, coupling):
    by_level = {lv: files_of(root, coupling, lv) for lv in LEVELS}
    keys = {lv: [PATTERN.search(os.path.basename(f)) for f in fs] for lv, fs in by_level.items()}
    masses = sorted({k["m"] for ks in keys.values() for k in ks}, key=float)
    pulsars = sorted({k["psr"] for ks in keys.values() for k in ks})
    im = {m: i for i, m in enumerate(masses)}
    ip = {p: i for i, p in enumerate(pulsars)}
    with np.load(by_level[LEVELS[0]][0], allow_pickle=False) as z:
        shared = {k: z[k][()] for k in SHARED}
    shape = (len(LEVELS), len(masses), len(pulsars))
    counts = np.zeros(shape + (int(shared["n_bins"]),), dtype=np.int32)
    present = np.zeros(shape, dtype=bool)
    per = {k: np.full(shape, np.nan) for k in PER_CHAIN}
    for il, lv in enumerate(LEVELS):
        for f, k in zip(by_level[lv], keys[lv]):
            i, j = im[k["m"]], ip[k["psr"]]
            with np.load(f, allow_pickle=False) as z:
                for s in SHARED:
                    if z[s][()] != shared[s]:
                        raise SystemExit(f"{f}: {s}={z[s][()]!r} differs from {shared[s]!r}")
                counts[il, i, j] = z["counts"]
                for s in PER_CHAIN:
                    per[s][il, i, j] = float(z[s][()])
            present[il, i, j] = True
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"injection_{coupling}.npz")
    np.savez_compressed(out, levels=np.array(LEVELS), masses_str=np.array(masses),
                        pulsars=np.array(pulsars), present=present, counts=counts,
                        **{f"chain_{k}": v for k, v in per.items()},
                        **{f"run_{k}": np.array(v) for k, v in shared.items()})
    print(f"{out}: {int(present.sum())} chains ({len(LEVELS)} levels x {len(masses)} masses x "
          f"{len(pulsars)} pulsars), {os.path.getsize(out) / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    for c in (sys.argv[3:] or ["alpha", "beta", "vec"]):
        pack(sys.argv[1], sys.argv[2], c)
