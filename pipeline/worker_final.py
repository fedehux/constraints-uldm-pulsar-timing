#!/usr/bin/env python
# =============================================================================
# worker_final.py  --  WORKER unico de las corridas finales
# =============================================================================
# 1 tarea = (masa, pulsar): una cadena PTMCMC de N_ITER pasos con el pipeline
# final; se guarda SOLO el histograma del log10 del acople (0.005 dex/bin) y la
# cadena cruda se borra.  Idempotente (si el archivo existe se saltea) y con
# escritura atomica (tmp + rename), asi un corte por walltime nunca deja un
# archivo a medias.  Relanzar el MISMO comando continua donde quedo.
#
# Entorno (lo exporta run_final.slurm):
#   ACOPLE   alpha | beta | vec                          (obligatorio)
#   MODO     cotas (prior uniforme en el acople, masas de produccion)
#            bf    (prior log-uniforme, 200 masas)       (obligatorio)
#   OUT_ROOT carpeta de salida (hist/ va adentro)       (obligatorio)
#   NANOGRAV_ROOT, CHAIN_TMP (disco local del nodo), N_ITER (200000),
#   SEED0 (900), JOB_ID (0), N_JOBS (1), PULSARS (opcional)
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

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)

ACOPLE = os.environ.get("ACOPLE", "").strip().lower()
MODO = os.environ.get("MODO", "").strip().lower()
if ACOPLE not in ("alpha", "beta", "vec") or MODO not in ("cotas", "bf"):
    raise SystemExit(f"ACOPLE={ACOPLE!r} MODO={MODO!r}: alpha|beta|vec y cotas|bf")
# the prior must be fixed BEFORE importing the pipeline (read at import time)
os.environ["PRIOR_ACOPLE"] = "uniforme" if MODO == "cotas" else "loguniforme"
if "OUT_ROOT" not in os.environ:
    raise SystemExit("falta OUT_ROOT")

import numpy as np
try:
    from loguru import logger as _lg; _lg.disable("pint")
except Exception:
    pass
import rangos_prior                       # sets LOG10_*_MIN/MAX in the environment
rangos_prior.exportar(ACOPLE)

P = __import__({"alpha": "pipeline_alpha", "beta": "pipeline_beta", "vec": "pipeline_vec"}[ACOPLE])
CLAVE = {"alpha": "alpha", "beta": "beta", "vec": "g"}[ACOPLE]
EDGES = {"alpha": "ALPHA_EDGES", "beta": "BETA_EDGES", "vec": "G_EDGES"}[ACOPLE]
EDGES = getattr(P, EDGES)
LO, HI = float(EDGES[0]), float(EDGES[-1])

N_ITER = int(os.environ.get("N_ITER", "200000"))
SEED0 = int(os.environ.get("SEED0", "900"))
JOB_ID = int(os.environ.get("JOB_ID", "0"))
N_JOBS = int(os.environ.get("N_JOBS", "1"))
NCORES = int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1)))
HISTDIR = os.path.join(os.environ["OUT_ROOT"], "hist")
CHAIN_TMP = os.environ.get("CHAIN_TMP", os.path.join(os.environ["OUT_ROOT"], "chains_tmp"))
P.CHAIN_ROOT = CHAIN_TMP                    # raw chains on the node's local disk


def masas():
    f = os.environ.get("MASAS_ARCHIVO") or os.path.join(AQUI, f"masas_{ACOPLE}_cotas.txt" if MODO == "cotas" else "masas_bf.txt")
    return [l.strip() for l in open(f) if l.strip()]


def archivo_de(ms, name):
    return os.path.join(HISTDIR, f"m_{ms}__{name}__s0.npz")


def semilla_de(ms, name):
    clave = f"final|{ACOPLE}|{MODO}|{ms}|{name}".encode()
    return SEED0 + (zlib.crc32(clave) % 1_000_000)


# --- PINT loading cached per process (the same pulsar, many masses) ----------
_CACHE = {}
_load_orig = P.load_pulsar


def _load_cache(par, tim, convert_low_ecc=True):
    k = (par, tim, convert_low_ecc)
    if k not in _CACHE:
        _CACHE.clear()                     # keep one pulsar in memory
        _CACHE[k] = _load_orig(par, tim, convert_low_ecc=convert_low_ecc)
    return _CACHE[k]


P.load_pulsar = _load_cache


def una_tarea(args):
    """args = (pulsar, [(mass_string, idx), ...]): same pulsar, several masses."""
    name, lista = args
    out = []
    for ms, idx in lista:
        dest = archivo_de(ms, name)
        if os.path.exists(dest):
            out.append(("skip", name, ms, 0.0)); continue
        t0 = time.time()
        r = P.process_task((idx, float(ms), name, N_ITER, semilla_de(ms, name)))
        if not r.get("ok"):
            print(f"  [{ms} {name}] FAIL: {r.get('err', '?')}", flush=True)
            out.append(("fail", name, ms, 0.0)); continue
        x = np.asarray(r[CLAVE], dtype=float)
        counts, _ = np.histogram(x, bins=EDGES)
        fuera = int(np.sum((x < LO) | (x > HI)))
        tmp = dest + ".tmp.npz"
        np.savez_compressed(tmp, m_eV=float(ms), m_str=ms, name=name, semilla=semilla_de(ms, name),
                            counts=counts.astype(np.int32), ess=float(r["ess"]), n_muestras=int(x.size),
                            n_fuera=fuera, n_toa=int(r["n_toa"]), ecc=float(r["ecc"]), n_iter=N_ITER,
                            acople=ACOPLE, modo=MODO, prior=P.PRIOR_ACOPLE, version=P.VERSION_PIPELINE,
                            likelihood=P.LIKELIHOOD, log10_min=LO, log10_max=HI, n_bins=len(EDGES) - 1,
                            x_min=float(x.min()), x_max=float(x.max()),
                            frac_techo=float(np.mean(x > HI - 0.5)), frac_piso=float(np.mean(x < LO + 1.0)),
                            t_min=(time.time() - t0) / 60.0)
        os.replace(tmp, dest)
        shutil.rmtree(os.path.join(CHAIN_TMP, f"m{idx:03d}", name), ignore_errors=True)
        out.append(("ok", name, ms, (time.time() - t0) / 60.0))
    return out


def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    os.makedirs(HISTDIR, exist_ok=True); os.makedirs(CHAIN_TMP, exist_ok=True)
    pulsares = P.filter_pulsars_by_model(P.discover_wb_pulsars(), allowed=P.ALLOWED_MODELS)
    quiero = [p.strip() for p in os.environ.get("PULSARS", "").split(",") if p.strip()]
    if quiero:
        pulsares = [p for p in quiero if p in pulsares]
    ms = masas()
    # every (mass, pulsar) pair, dealt to the jobs by stride over a fixed order
    pares = [(m, p) for p in pulsares for m in ms]
    mios = pares[JOB_ID::N_JOBS]
    pend = [(m, p) for (m, p) in mios if not os.path.exists(archivo_de(m, p))]
    idx_de = {m: i for i, m in enumerate(ms)}
    # group by pulsar in chunks of up to 8 masses (one PINT load per chunk)
    from collections import defaultdict
    g = defaultdict(list)
    for m, p in pend:
        g[p].append((m, idx_de[m] * 64 + (pulsares.index(p) % 64)))
    trozos = [(p, v[i:i + 8]) for p, v in g.items() for i in range(0, len(v), 8)]
    print(f"[final] ACOPLE={ACOPLE} MODO={MODO} PRIOR={P.PRIOR_ACOPLE} version={P.VERSION_PIPELINE} "
          f"likelihood={P.LIKELIHOOD}", flush=True)
    print(f"[final] log10 {CLAVE} in [{LO:.4f}, {HI:.4f}] ({len(EDGES) - 1} bins, "
          f"{(HI - LO) / (len(EDGES) - 1):.4f} dex/bin)  N_ITER={N_ITER}", flush=True)
    print(f"[final] masas={len(ms)}  pulsares={len(pulsares)}  job {JOB_ID + 1}/{N_JOBS}: "
          f"tareas={len(mios)} (pendientes {len(pend)})  cores={NCORES}  CHAIN_TMP={CHAIN_TMP}", flush=True)
    done = nok = nskip = nbad = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=NCORES) as ex:
        futs = [ex.submit(una_tarea, tr) for tr in trozos]
        for fut in as_completed(futs):
            for tag, _n, _m, _t in fut.result():
                done += 1
                nok += tag == "ok"; nskip += tag == "skip"; nbad += tag == "fail"
            el = (time.time() - t0) / 60.0
            if done and (done % 100 < 8 or done == len(pend)):
                eta = (len(pend) - done) * el / max(done, 1)
                print(f"  [{done}/{len(pend)}] ok={nok} skip={nskip} bad={nbad}  {el:.0f} min  ETA {eta:.0f} min",
                      flush=True)
    print(f"[final] {ACOPLE}/{MODO} job {JOB_ID + 1}/{N_JOBS} LISTO ok={nok} skip={nskip} bad={nbad} "
          f"en {(time.time() - t0) / 60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
