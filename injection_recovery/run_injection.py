#!/usr/bin/env python
"""Injection and recovery test of the upper-limit pipeline (Appendix E of the paper).

For every binary, the residuals are replaced by a realization of the noise model
the chains assume,

    r = d_DM + M_DMX eps_DMX + F a + n + h_inj,

with white noise, DMX offsets and red noise at the MAP values released with the
NANOGrav 15-year data set, plus the ULDM template of the production pipeline.
Each binary draws its own amplitude rho ~ Rayleigh and phase Upsilon ~ U(0, 2 pi)
(and, for the vector coupling, an isotropic direction).  Timing-model offsets
other than DMX have an improper prior and are set to zero.

Two masses per coupling (one away from every orbital resonance, one close to a
resonance of PSR J1903+0327 or PSR J1909-3744) and two levels on the same noise
and field realization: no signal, and a coupling 4 x the 95% upper limit of the
paper at that mass (read from ``<coupling folder>/bounds_<coupling>.npz``).  The
chains use the production likelihood with a prior uniform in log10 of the
coupling over the range of ``pipeline/prior_ranges.py``.  Every random number is
seeded by (coupling, mass, level, pulsar), so a rerun reproduces the runs of the
paper for the same software versions.

Outputs follow the format of ``pipeline/worker_final.py``, one file per chain,
plus the injected parameters:

    <out>/<coupling>/<level>/hist/m_<mass>__<pulsar>__s0.npz      level = no_signal | x4

Usage (needs PINT, ENTERPRISE and PTMCMCSampler; see environment.yml):

    export NANOGRAV_ROOT=/path/to/NANOGrav15yr_PulsarTiming_v2.1.0
    python injection_recovery/run_injection.py prepare alpha 8     # load and cache the pulsars
    python injection_recovery/run_injection.py run     alpha 8     # 2 masses x 2 levels x 50 chains
    python injection_recovery/run_injection.py test    alpha J1909-3744 5000

``alpha`` can be replaced by ``beta`` or ``vec``; the last argument of ``prepare``
and ``run`` is the number of parallel processes.  Environment variables:
``INJ_OUT`` (output folder, default ``injection_runs`` in the working directory),
``INJ_N_ITER`` (iterations per chain, default 200000 as in the paper).
``pack_injection.py`` turns the output into the archives in ``histograms/``.
"""
import glob
import importlib
import os
import pickle
import shutil
import sys
import tempfile
import time

for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"

if __name__ == "__main__" and len(sys.argv) > 2:
    os.environ["INJ_COUPLING"] = sys.argv[2]          # child processes inherit it
COUPLING = os.environ.get("INJ_COUPLING", "alpha")
if COUPLING not in ("alpha", "beta", "vec"):
    raise SystemExit(f"coupling {COUPLING!r}: alpha | beta | vec")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE = os.path.join(ROOT, "pipeline")
N_ITER = int(os.environ.get("INJ_N_ITER", "200000"))
OUT = os.path.abspath(os.environ.get("INJ_OUT", "injection_runs"))
TMP = os.path.join(tempfile.gettempdir(), "uldm_injection")
if "NANOGRAV_ROOT" not in os.environ:
    raise SystemExit("set NANOGRAV_ROOT to the NANOGrav15yr_PulsarTiming_v2.1.0 folder")

os.environ["COUPLING_PRIOR"] = "log-uniform"          # prior uniform in log10 of the coupling
os.environ.setdefault("FROZEN_PARAMS", "PBDOT")       # Pbdot held at its published value
os.environ["OUT_ROOT"] = os.path.join(TMP, "out_pipeline")
sys.path.insert(0, PIPELINE)

import numpy as np                                    # noqa: E402

P = importlib.import_module({"alpha": "pipeline_alpha", "beta": "pipeline_beta",
                             "vec": "pipeline_vec"}[COUPLING])
EDGES = getattr(P, {"alpha": "ALPHA_EDGES", "beta": "BETA_EDGES", "vec": "G_EDGES"}[COUPLING])
LO, HI = float(EDGES[0]), float(EDGES[-1])

# 95% upper limits of the paper (code units, the same as the chain column 0)
BOUNDS = {"alpha": os.path.join(ROOT, "linear", "bounds_alpha.npz"),
          "beta": os.path.join(ROOT, "quadratic", "bounds_beta.npz"),
          "vec": os.path.join(ROOT, "vector", "bounds_vec.npz")}
# (tag, seed tag, mass of the bound grid); the resonant points lie a few widths
# off the centre of the resonance, never on it
MASSES = {"alpha": [("offres", 0, "9.8544087700e-23"),     # away from every resonance
                    ("res", 1, "4.9679110191e-22")],       # PSR J1903+0327, m = w_b
          "beta": [("offres", 0, "1.0116379798e-22"),
                   ("res", 1, "2.4915046422e-22")],        # PSR J1903+0327, 2m = w_b
          "vec": [("offres", 0, "9.5477161142e-23"),
                  ("res", 1, "3.1231918474e-20")]}         # PSR J1909-3744, m = w_b
MODEL_KEY = {"alpha": "la", "beta": "lb", "vec": "lg"}[COUPLING]   # coupling key in P.make_model
LEVELS = [("no_signal", 0, None), ("x4", 4, 4.0)]                  # (name, seed tag, factor)

PSR_DIR = os.path.join(OUT, "pulsars", COUPLING)
RES = os.path.join(OUT, COUPLING)
for _d in (PSR_DIR, RES, TMP):
    os.makedirs(_d, exist_ok=True)


def pulsars():
    return P.filter_pulsars_by_model(P.discover_wb_pulsars(), allowed=P.ALLOWED_MODELS)


# ------------------------------------------------------------------ pulsars --
def prepare_one(name):
    """Load the pulsar exactly as the production pipeline does and cache it."""
    par, tim, chain, pars = P.resolve_paths(name)
    psr, tm = P.load_pulsar(par, tim, convert_low_ecc=True)
    t_arr, partials, orb_pars, missing = P.build_context(psr, tm)
    with open(os.path.join(PSR_DIR, f"{name}.pkl"), "wb") as fh:
        pickle.dump(dict(name=name, psr=psr, t_arr=t_arr, partials=partials,
                         orb_pars=orb_pars, missing=missing, par=par, chain=chain,
                         pars=pars), fh, protocol=pickle.HIGHEST_PROTOCOL)
    return name, len(t_arr), missing


_CACHE = {}


def load(name):
    if name not in _CACHE:
        with open(os.path.join(PSR_DIR, f"{name}.pkl"), "rb") as fh:
            _CACHE[name] = pickle.load(fh)
    return _CACHE[name]


def context(d, m_dm):
    """The production preload_one without the PINT loading."""
    pta = P.build_pta(d["psr"], d["t_arr"], d["partials"], d["orb_pars"], m_dm)
    names = P.get_uldm_pnames(pta)            # (coupling, rho, Upsilon[, vartheta, varphi])
    arn, grn = P.get_rn_pnames(pta)
    noise = P.load_noise_MAP(pta, d["chain"], d["pars"])
    noise.update(P.dmjumps_par(d["par"], d["psr"].name, pta))
    return pta, names, (arn, grn), noise


def _signal(names, x, field):
    """Signal parameters: coupling x and the field (rho, Upsilon[, vartheta, varphi])."""
    return dict(zip(names, (x,) + tuple(field)))


# --------------------------------------------------------------- simulation --
def simulate(d, m_dm, x_inj, field, rng):
    """Replace the residuals by a simulated realization and return the PTA built
    on them, with the optimal SNR^2 of the injected template (timing model and red
    noise marginalized at the injected red-noise parameters)."""
    psr = d["psr"]
    pta, names, _, noise = context(d, m_dm)
    p = dict(noise)
    p.update(_signal(names, -300.0, field))
    delay0 = np.asarray(pta.get_delay(p)[0], dtype=float)
    T = np.asarray(pta.get_basis(p)[0], dtype=float)
    phi = np.asarray(pta.get_phi(p)[0], dtype=float)
    if phi.ndim != 1:
        raise RuntimeError("phi is not diagonal")
    nd = np.asarray(pta.get_ndiag(p)[0], dtype=float)
    b = np.zeros_like(phi)
    finite = phi < 1e30                       # DMX and red noise, not the improper block
    b[finite] = np.sqrt(phi[finite]) * rng.normal(size=int(finite.sum()))
    n = np.sqrt(nd) * rng.normal(size=len(nd))
    h = np.zeros_like(delay0)
    if x_inj is not None:
        q = dict(p)
        q.update(_signal(names, x_inj, field))
        h = np.asarray(pta.get_delay(q)[0], dtype=float) - delay0
    psr._residuals = (delay0 + T @ b + n + h)[psr._iisort]

    pta2, names, (arn, grn), noise = context(d, m_dm)
    snr2 = np.nan
    if x_inj is not None:                     # lnL(2h) - 2 lnL(h) + lnL(0) = -h^T C^-1 h
        ls = []
        for x in (-300.0, x_inj, x_inj + np.log10(2.0)):
            q = dict(noise)
            q.update(_signal(names, x, field))
            ls.append(pta2.get_lnlikelihood(q))
        snr2 = -(ls[2] - 2.0 * ls[1] + ls[0])
    return pta2, names, (arn, grn), noise, float(snr2)


def run_chain(model, out_dir, n_iter, seed):
    """P.run_one_chain with a fixed sampler seed, returning every column."""
    from PTMCMCSampler import PTMCMCSampler
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    sampler = PTMCMCSampler.PTSampler(model.ndim, model.lnlikelihood, model.lnprior,
                                      np.copy(model.cov0()), outDir=out_dir, resume=False,
                                      verbose=False, seed=int(rng.integers(2**31 - 1)))
    sampler.sample(model.x0(rng), n_iter, Tmax=1, covUpdate=P.COV_UPDATE,
                   SCAMweight=P.SCAM_WEIGHT, AMweight=P.AM_WEIGHT, DEweight=P.DE_WEIGHT,
                   thin=1, isave=10_000, writeHotChains=False)
    files = sorted(glob.glob(os.path.join(out_dir, "chain_1*.txt")))
    if not files:
        raise RuntimeError(f"no chain found in {out_dir}")
    return np.loadtxt(files[0])[int(P.BURN_FRAC * n_iter):]


def theta95(m_str):
    with np.load(BOUNDS[COUPLING], allow_pickle=False) as z:
        m, ul = z["masses_eV"], z["log10_ul_code"]
    i = int(np.argmin(np.abs(m / float(m_str) - 1)))
    if abs(m[i] / float(m_str) - 1) > 1e-8:
        raise RuntimeError(f"{m_str} is not a mass of the bound curve")
    return float(ul[i])


def output(level, m_str, name):
    return os.path.join(RES, level, "hist", f"m_{m_str}__{name}__s0.npz")


def task(args):
    """One (mass, level, pulsar) chain."""
    tag, it, m_str, level, inv, factor, name, ipsr, n_iter, dest = args
    if os.path.exists(dest):
        return dest, "done before"
    t0 = time.time()
    d = load(name)
    m_eV = float(m_str)
    m_dm = m_eV * P.EV_TO_INV_S
    # noise and field depend on (coupling, mass, pulsar) only: both levels see the
    # same noise and the same local field
    acn = {"alpha": 1, "beta": 2, "vec": 3}[COUPLING]
    rng_sim = np.random.default_rng(np.random.SeedSequence([20261013, acn, it, ipsr]))
    rho = float(np.clip(np.sqrt(rng_sim.exponential(1.0)), P.RHO_F_MIN, P.RHO_F_MAX))
    ups = float(rng_sim.uniform(0.0, 2.0 * np.pi))
    field = (rho, ups)
    cos_t, vphi = np.nan, np.nan
    if COUPLING == "vec":                     # isotropic direction: cos(vartheta) ~ U(-1, 1)
        cos_t = float(rng_sim.uniform(-1.0, 1.0))
        vphi = float(rng_sim.uniform(0.0, 2.0 * np.pi))
        field = (rho, ups, float(np.arccos(cos_t)), vphi)
    x_inj = None if factor is None else theta95(m_str) + np.log10(factor)
    pta, names, (arn, grn), noise, snr2 = simulate(d, m_dm, x_inj, field, rng_sim)
    if arn not in noise or grn not in noise:
        raise RuntimeError(f"{name}: red-noise MAP missing")
    rf, gf = names[1], names[2]
    ctx = dict(ctx=(d["t_arr"], d["partials"], d["orb_pars"], m_dm),
               name=name, pta=pta, rf=rf, gf=gf,
               arn=arn, grn=grn, noise_fixed={k: v for k, v in noise.items() if k not in (arn, grn)},
               rn_map=(noise[arn], noise[grn]), ecc=d["orb_pars"]["ecc"], n_toa=len(d["t_arr"]))
    ctx[MODEL_KEY] = names[0]
    if COUPLING == "vec":
        ctx.update(vt=names[3], vp=names[4], sin_i=d["orb_pars"]["sin_i"])
    model = P.make_model(ctx)
    seed = np.random.SeedSequence([104729, acn, it, inv, ipsr]).generate_state(1)[0]
    out_dir = os.path.join(TMP, "chains", COUPLING, f"{tag}_{level}_{name}")
    try:
        data = run_chain(model, out_dir, n_iter, seed)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
    x = data[:, 0]
    counts, _ = np.histogram(x, bins=EDGES)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp.npz"
    np.savez_compressed(
        tmp, m_eV=m_eV, m_str=m_str, name=name, seed=int(seed), counts=counts.astype(np.int32),
        ess=float(P.effective_sample_size(x)), n_samples=int(x.size),
        n_outside=int(np.sum((x < LO) | (x > HI))), n_toa=len(d["t_arr"]),
        ecc=float(d["orb_pars"]["ecc"]), n_iter=n_iter, coupling=COUPLING,
        prior="log-uniform", version=P.VERSION_PIPELINE, likelihood=P.LIKELIHOOD,
        log10_min=LO, log10_max=HI, n_bins=len(EDGES) - 1, x_min=float(x.min()),
        x_max=float(x.max()), frac_ceiling=float(np.mean(x > HI - 0.5)),
        frac_floor=float(np.mean(x < LO + 1.0)),
        level=level, x_inj=np.nan if x_inj is None else x_inj, rho_inj=rho, ups_inj=ups,
        cos_t_inj=cos_t, varphi_inj=vphi, snr2=snr2, rn_inj=np.array(ctx["rn_map"]),
        acc=float(np.mean(np.diff(x) != 0)), t_min=(time.time() - t0) / 60.0)
    os.replace(tmp, dest)
    return dest, f"{(time.time() - t0) / 60:.1f} min  snr={np.sqrt(snr2) if snr2 == snr2 else 0:.2f}"


def tasks(n_iter):
    names = pulsars()
    out = []
    for tag, it, m_str in MASSES[COUPLING]:
        for level, inv, factor in LEVELS:
            for ip, name in enumerate(names):
                out.append((tag, it, m_str, level, inv, factor, name, ip, n_iter,
                            output(level, m_str, name)))
    return out


# --------------------------------------------------------------------- main --
def _pool(fn, items, n_workers, label):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    t0 = time.time()
    with ProcessPoolExecutor(n_workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for k, f in enumerate(as_completed(futs), 1):
            try:
                msg = f.result()
            except Exception as e:
                msg = f"FAIL {type(e).__name__}: {e}"
            el = (time.time() - t0) / 3600
            print(f"[{label} {k}/{len(items)} {el:.2f} h  eta {el / k * (len(items) - k):.1f} h] "
                  f"{msg}", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "prepare":
        names = [n for n in pulsars() if not os.path.exists(os.path.join(PSR_DIR, f"{n}.pkl"))]
        print(f"{COUPLING}: {len(names)} pulsars to prepare", flush=True)
        _pool(prepare_one, names, int(sys.argv[3]), "prepare")
    elif cmd == "test":
        name, n_iter = sys.argv[3], int(sys.argv[4])
        names = pulsars()
        tag, it, m_str = MASSES[COUPLING][1]
        level, inv, factor = LEVELS[1]
        dest = os.path.join(TMP, "test", f"m_{m_str}__{name}__s0.npz")
        if os.path.exists(dest):
            os.remove(dest)
        print(task((tag, it, m_str, level, inv, factor, name, names.index(name), n_iter, dest)))
        with np.load(dest, allow_pickle=False) as z:
            print({k: z[k] for k in ("x_inj", "rho_inj", "snr2", "ess", "acc", "t_min")})
    elif cmd == "run":
        ts = [t for t in tasks(N_ITER) if not os.path.exists(t[-1])]
        ts.sort(key=lambda t: -len(load(t[6])["t_arr"]))          # large pulsars first
        print(f"{COUPLING}: {len(ts)} chains to run, N_ITER={N_ITER}, {sys.argv[3]} processes",
              flush=True)
        _pool(task, ts, int(sys.argv[3]), "chain")
    else:
        raise SystemExit(__doc__)
