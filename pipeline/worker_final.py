#!/usr/bin/env python
# =============================================================================
# worker_final.py  --  runs the upper-limit chains of one coupling
# =============================================================================
# One task = (mass, pulsar): a PTMCMC chain of N_ITER steps with a prior
# uniform in the coupling. Only the histogram of log10 of the coupling is kept
# (0.005 dex bins); the raw chain is deleted. Idempotent (existing files are
# skipped) with atomic writes (tmp + rename), so a walltime cut never leaves a
# partial file: rerunning the same command continues where it stopped.
#
# Environment (run_final.slurm sets it):
#   COUPLING   alpha | beta | vec                            (required)
#   OUT_ROOT   output folder; the histograms go to OUT_ROOT/hist   (required)
#   NANOGRAV_ROOT, CHAIN_TMP (node-local scratch), N_ITER (200000),
#   SEED0 (900), JOB_ID (0), N_JOBS (1), PULSARS (comma-separated subset),
#   MASS_FILE (default masses_<coupling>.txt), FROZEN_PARAMS (PBDOT)
# =============================================================================
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
import zlib
import shutil
import warnings
warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

COUPLING = os.environ.get("COUPLING", "").strip().lower()
if COUPLING not in ("alpha", "beta", "vec"):
    raise SystemExit(f"COUPLING={COUPLING!r}: alpha | beta | vec")
# the prior must be set before importing the pipeline (read at import time)
os.environ["COUPLING_PRIOR"] = "uniform"
if "OUT_ROOT" not in os.environ:
    raise SystemExit("OUT_ROOT is not set")

import numpy as np
try:
    from loguru import logger as _lg
    _lg.disable("pint")
except Exception:
    pass
import common

P = __import__({"alpha": "pipeline_alpha", "beta": "pipeline_beta", "vec": "pipeline_vec"}[COUPLING])
KEY = {"alpha": "alpha", "beta": "beta", "vec": "g"}[COUPLING]
EDGES = getattr(P, {"alpha": "ALPHA_EDGES", "beta": "BETA_EDGES", "vec": "G_EDGES"}[COUPLING])
LO, HI = float(EDGES[0]), float(EDGES[-1])

N_ITER = int(os.environ.get("N_ITER", "200000"))
SEED0 = int(os.environ.get("SEED0", "900"))
JOB_ID = int(os.environ.get("JOB_ID", "0"))
N_JOBS = int(os.environ.get("N_JOBS", "1"))
NCORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1)))
HISTDIR = os.path.join(os.environ["OUT_ROOT"], "hist")
CHAIN_TMP = os.environ.get("CHAIN_TMP", os.path.join(os.environ["OUT_ROOT"], "chains_tmp"))
P.CHAIN_ROOT = CHAIN_TMP                    # raw chains on the node's local disk


def masses():
    f = os.environ.get("MASS_FILE") or os.path.join(HERE, f"masses_{COUPLING}.txt")
    return [line.strip() for line in open(f) if line.strip()]


def output_file(ms, name):
    return os.path.join(HISTDIR, f"m_{ms}__{name}__s0.npz")


def seed_of(ms, name):
    """Seed of the starting point of the chain (PTMCMCSampler seeds itself)."""
    key = f"final|{COUPLING}|bounds|{ms}|{name}".encode()
    return SEED0 + (zlib.crc32(key) % 1_000_000)


# --- PINT loading cached per process (the same pulsar, many masses) ----------
_CACHE = {}
_load_orig = P.load_pulsar


def _load_cached(par, tim, convert_low_ecc=True):
    k = (par, tim, convert_low_ecc)
    if k not in _CACHE:
        _CACHE.clear()                     # keep one pulsar in memory
        _CACHE[k] = _load_orig(par, tim, convert_low_ecc=convert_low_ecc)
    return _CACHE[k]


P.load_pulsar = _load_cached


def one_task(args):
    """args = (pulsar, [(mass_string, idx), ...]): one pulsar, several masses."""
    name, items = args
    out = []
    for ms, idx in items:
        dest = output_file(ms, name)
        if os.path.exists(dest):
            out.append(("skip", name, ms, 0.0))
            continue
        t0 = time.time()
        r = P.process_task((idx, float(ms), name, N_ITER, seed_of(ms, name)))
        if not r.get("ok"):
            print(f"  [{ms} {name}] FAIL: {r.get('err', '?')}", flush=True)
            out.append(("fail", name, ms, 0.0))
            continue
        x = np.asarray(r[KEY], dtype=float)
        counts, _ = np.histogram(x, bins=EDGES)
        outside = int(np.sum((x < LO) | (x > HI)))
        tmp = dest + ".tmp.npz"
        np.savez_compressed(tmp, m_eV=float(ms), m_str=ms, name=name, seed=seed_of(ms, name),
                            counts=counts.astype(np.int32), ess=float(r["ess"]), n_samples=int(x.size),
                            n_outside=outside, n_toa=int(r["n_toa"]), ecc=float(r["ecc"]), n_iter=N_ITER,
                            coupling=COUPLING, prior=P.COUPLING_PRIOR, version=P.VERSION_PIPELINE,
                            likelihood=P.LIKELIHOOD, log10_min=LO, log10_max=HI, n_bins=len(EDGES) - 1,
                            x_min=float(x.min()), x_max=float(x.max()),
                            frac_ceiling=float(np.mean(x > HI - 0.5)), frac_floor=float(np.mean(x < LO + 1.0)),
                            t_min=(time.time() - t0) / 60.0)
        os.replace(tmp, dest)
        shutil.rmtree(os.path.join(CHAIN_TMP, f"m{idx:03d}", name), ignore_errors=True)
        out.append(("ok", name, ms, (time.time() - t0) / 60.0))
    return out


def main():
    from collections import defaultdict
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(HISTDIR, exist_ok=True)
    os.makedirs(CHAIN_TMP, exist_ok=True)
    pulsars = P.filter_pulsars_by_model(P.discover_wb_pulsars(), allowed=P.ALLOWED_MODELS)
    wanted = [p.strip() for p in os.environ.get("PULSARS", "").split(",") if p.strip()]
    if wanted:
        pulsars = [p for p in wanted if p in pulsars]
    ms = masses()
    # every (mass, pulsar) pair, dealt to the jobs by stride over a fixed order
    pairs = [(m, p) for p in pulsars for m in ms]
    mine = pairs[JOB_ID::N_JOBS]
    pending = [(m, p) for (m, p) in mine if not os.path.exists(output_file(m, p))]
    idx_of = {m: i for i, m in enumerate(ms)}
    # group by pulsar in chunks of up to 8 masses (one PINT load per chunk)
    g = defaultdict(list)
    for m, p in pending:
        g[p].append((m, idx_of[m] * 64 + (pulsars.index(p) % 64)))
    chunks = [(p, v[i:i + 8]) for p, v in g.items() for i in range(0, len(v), 8)]
    print(f"[final] COUPLING={COUPLING} PRIOR={P.COUPLING_PRIOR} version={P.VERSION_PIPELINE} "
          f"likelihood={P.LIKELIHOOD} frozen={','.join(common.FROZEN_PARAMS) or '-'}", flush=True)
    print(f"[final] log10 {KEY} in [{LO:.4f}, {HI:.4f}] ({len(EDGES) - 1} bins, "
          f"{(HI - LO) / (len(EDGES) - 1):.4f} dex/bin)  N_ITER={N_ITER}", flush=True)
    print(f"[final] masses={len(ms)}  pulsars={len(pulsars)}  job {JOB_ID + 1}/{N_JOBS}: "
          f"tasks={len(mine)} (pending {len(pending)})  cores={NCORES}  CHAIN_TMP={CHAIN_TMP}", flush=True)
    done = nok = nskip = nbad = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NCORES) as ex:
        futs = [ex.submit(one_task, c) for c in chunks]
        for fut in as_completed(futs):
            for tag, _n, _m, _t in fut.result():
                done += 1
                nok += tag == "ok"
                nskip += tag == "skip"
                nbad += tag == "fail"
            el = (time.time() - t0) / 60.0
            if done and (done % 100 < 8 or done == len(pending)):
                eta = (len(pending) - done) * el / max(done, 1)
                print(f"  [{done}/{len(pending)}] ok={nok} skip={nskip} bad={nbad}  {el:.0f} min  ETA {eta:.0f} min",
                      flush=True)
    print(f"[final] {COUPLING} job {JOB_ID + 1}/{N_JOBS} DONE ok={nok} skip={nskip} bad={nbad} "
          f"in {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
