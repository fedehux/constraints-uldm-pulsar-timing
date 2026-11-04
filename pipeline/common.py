"""Code shared by the three single-pulsar pipelines (pipeline_alpha.py,
pipeline_beta.py, pipeline_vec.py): discovery and loading of the NANOGrav
15-year wideband data with PINT, the design-matrix partials of the orbital
parameters, the ENTERPRISE noise model, the noise MAP values, and the
PTMCMCSampler chain.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import glob
import re
import warnings
from copy import deepcopy

import numpy as np
import astropy.units as u

warnings.filterwarnings("ignore")
try:
    from loguru import logger as _loguru
    _loguru.disable("pint")
except Exception:
    pass
try:
    import logging
    for _n in ("pint", "enterprise"):
        logging.getLogger(_n).setLevel(logging.WARNING)
except Exception:
    pass

# ---- data -------------------------------------------------------------------
NANOGRAV_ROOT = os.environ.get("NANOGRAV_ROOT", "NANOGrav15yr_PulsarTiming_v2.1.0")
WB_ROOT = os.path.join(NANOGRAV_ROOT, "wideband")

# Binary models analysed. BT (J1312+0051) and DDK (J1713+0747) have ECC/OM/T0
# but eccentricities of 2.4e-4 and 7.5e-5; load_pulsar converts them to ELL1.
ALLOWED_MODELS = ("ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDK")
ECC_LOW_THRESHOLD = 0.01          # below this eccentricity the ELL1 template is used

# ---- constants (natural units, eV = 1) --------------------------------------
EV_TO_INV_S = 1.51926760385984e+15
DAY_IN_S = 86400.0
DM_DENSITY = 0.3 * 4.09351499910375e+55   # 0.3 GeV/cm^3

# ---- noise model ------------------------------------------------------------
N_FREQ_RN = 30                            # Fourier modes of the red noise
LOG10_ARED_MIN, LOG10_ARED_MAX = -20.0, -11.0
GAMMA_RED_MIN, GAMMA_RED_MAX = 0.0, 10.0
RN_SUFFIXES = ["red_noise_log10_A", "red_noise_gamma"]

# ---- field amplitude rho (Rayleigh prior, imposed in each lnprior) ---------
RHO_F_MIN = 1.0e-3
RHO_F_MAX = 6.0
LOG10_RHO_MIN = np.log10(RHO_F_MIN)
LOG10_RHO_MAX = np.log10(RHO_F_MAX)

# ---- sampler ----------------------------------------------------------------
BURN_FRAC = 0.30                          # fraction of each chain discarded
COV_UPDATE = 2000
SCAM_WEIGHT = 60
AM_WEIGHT = 20
DE_WEIGHT = 20
SIGMA_LOG10RHO = 0.5
SIGMA_GAMMA = 2 * np.pi
SIGMA_LOG10_ARED = 1.5
SIGMA_GAMMA_RED = 1.5

# Timing parameters held fixed at their published values (removed from the
# design matrix, so not marginalized). PBDOT shares the PB channel, which
# carries almost all the norm of the template; marginalizing it weakens the
# resonant limits by 0.5-1.3 dex. Sec. III B of the paper.
FROZEN_PARAMS = tuple(x.strip() for x in os.environ.get("FROZEN_PARAMS", "PBDOT").split(",")
                      if x.strip())

# The DDK -> ELL1 conversion drops KIN/KOM/K96. When the model has them, the
# residuals are taken from the original model (see load_pulsar).
KOPEIKIN_PARAMS = ("KIN", "KOM", "K96")
KOPEIKIN_FIXED = True


# ---- pulsar discovery ---------------------------------------------------------
def resolve_paths(psr_name):
    par_hits = sorted(glob.glob(os.path.join(WB_ROOT, "par", f"{psr_name}_PINT_*.wb.par")))
    tim_hits = sorted(glob.glob(os.path.join(WB_ROOT, "tim", f"{psr_name}_PINT_*.wb.tim")))
    if not par_hits or not tim_hits:
        raise FileNotFoundError(f"{psr_name}: par={par_hits} tim={tim_hits}")
    return (par_hits[0], tim_hits[0],
            os.path.join(WB_ROOT, "noise", f"{psr_name}.wb.chain_1.txt"),
            os.path.join(WB_ROOT, "noise", f"{psr_name}.wb.pars.txt"))


def par_binary_model(psr_name):
    par = resolve_paths(psr_name)[0]
    with open(par) as fh:
        for line in fh:
            toks = line.split()
            if toks and toks[0] == "BINARY":
                return toks[1].strip()
    return None


def discover_wb_pulsars():
    name_re = re.compile(r"^[BJ]\d{4}[+-]\d{2,4}$")
    names = set()
    for f in glob.glob(os.path.join(WB_ROOT, "par", "*_PINT_*.wb.par")):
        base = os.path.basename(f).split("_PINT_")[0]
        if name_re.match(base):
            names.add(base)
    return sorted(names)


def filter_pulsars_by_model(names, allowed=ALLOWED_MODELS):
    keep, skip = [], []
    for n in names:
        try:
            model = par_binary_model(n)
            resolve_paths(n)
        except Exception as e:
            skip.append((n, f"no-files:{type(e).__name__}"))
            continue
        if model in allowed:
            keep.append(n)
        else:
            skip.append((n, str(model)))
    print(f"  [filter {'/'.join(allowed)}] {len(keep)} pulsars kept, {len(skip)} skipped")
    return keep


# ---- timing model -------------------------------------------------------------
def model_ecc(model):
    if hasattr(model, "ECC") and model.ECC.value is not None:
        return float(model.ECC.value)
    if hasattr(model, "EPS1") and model.EPS1.value is not None:
        return float(np.hypot(model.EPS1.value, model.EPS2.value))
    return 1.0


def is_ell1_model(model):
    return hasattr(model, "EPS1") and getattr(model.EPS1, "value", None) is not None


def convert_to_ell1(model, skip_pb_if_fb0_free=False):
    """Convert to ELL1 if needed and free the five orbital parameters the
    template acts on. With skip_pb_if_fb0_free (vector pipeline), PB is left
    frozen when the orbit is parametrized by a free FB0: PB = 1/FB0 would be a
    redundant column, and build_context already builds the PB partial from the
    FB0 one."""
    if not is_ell1_model(model):
        from pint.binaryconvert import convert_binary
        model = convert_binary(model, "ELL1")
        # Kopeikin parameters left over from DDK do not belong to BinaryELL1
        for orphan in KOPEIKIN_PARAMS:
            if orphan in getattr(model, "params", []):
                try:
                    model.remove_param(orphan)
                except Exception:
                    pass
    fb0_free = (skip_pb_if_fb0_free and hasattr(model, "FB0")
                and getattr(getattr(model, "FB0", None), "value", None) is not None
                and getattr(model.FB0, "frozen", True) is False)
    for p in ("PB", "A1", "EPS1", "EPS2", "TASC"):
        if p == "PB" and fb0_free:
            continue
        if hasattr(model, p):
            getattr(model, p).frozen = False
    return model


def drop_toas_outside_dmx(model, toas):
    """WidebandTimingModel needs every TOA inside a DMX interval; drop the few
    that are not (2 of 1493 for J1713+0747)."""
    keys = [p[len("DMXR1_"):] for p in getattr(model, "params", []) if p.startswith("DMXR1_")]
    if not keys:
        return toas
    mjd = np.asarray(toas.get_mjds().value, dtype=float)
    cov = np.zeros(len(mjd), dtype=bool)
    for k in keys:
        a = getattr(model, "DMXR1_" + k).value
        b = getattr(model, "DMXR2_" + k).value
        if a is None or b is None:
            continue
        cov |= (mjd >= float(a)) & (mjd <= float(b))
    if cov.all():
        return toas
    print(f"    [DMX] dropped {int((~cov).sum())} TOA(s) outside the DMX intervals", flush=True)
    return toas[cov]


def drop_empty_dmx_bins(model, toas):
    """WidebandTimingModel fails on a DMX bin with no TOA (J1713+0747 has one);
    remove such bins. Call after drop_toas_outside_dmx."""
    idxs = [int(p[len("DMXR1_"):]) for p in getattr(model, "params", []) if p.startswith("DMXR1_")]
    if not idxs:
        return model
    mjd = np.asarray(toas.get_mjds().value, dtype=float)
    empty = []
    for k in idxs:
        a = getattr(model, f"DMXR1_{k:04d}").value
        b = getattr(model, f"DMXR2_{k:04d}").value
        if a is None or b is None:
            continue
        if not ((mjd >= float(a)) & (mjd <= float(b))).any():
            empty.append(k)
    if not empty:
        return model
    comp = model.components.get("DispersionDMX")
    for k in empty:
        try:
            comp.remove_DMX_range(k)
        except Exception:
            for pre in ("DMX_", "DMXR1_", "DMXR2_", "DMXF1_", "DMXF2_"):
                pn = f"{pre}{k:04d}"
                if pn in model.params:
                    try:
                        model.remove_param(pn)
                    except Exception:
                        pass
    try:
        model.validate()
        model.setup()
    except Exception:
        pass
    print(f"    [DMX] removed {len(empty)} empty DMX bin(s)", flush=True)
    return model


def has_kopeikin(model):
    """True if the model has a Kopeikin parameter with a value. These are the
    only parameters convert_to_ell1 removes without replacement; in the 15-year
    data set this selects J1713+0747 alone."""
    for p in KOPEIKIN_PARAMS:
        o = getattr(model, p, None)
        if o is not None and getattr(o, "value", None) is not None:
            return True
    return False


def load_pulsar(par, tim, convert_low_ecc=True, skip_pb_if_fb0_free=False):
    """Load a pulsar with PINT and return (enterprise Pulsar, timing model).

    Low-eccentricity BT/DD/DDK models are converted to ELL1. For J1713+0747
    (DDK) the conversion removes the Kopeikin terms and with them the secular
    drift of x = a1 sin i due to the proper motion (~13 us peak to peak). The
    unmodelled remainder goes as t sin(Phi_orb), the shape of the ULDM template
    at the k = 1 resonance. The residuals are therefore computed with the
    original model and the design matrix with the ELL1 one, which treats the
    drift as the fixed term predicted by KIN, KOM and the proper motion.
    """
    from enterprise.pulsar import Pulsar
    from pint.models import get_model_and_toas
    cwd = os.getcwd()
    timdir = os.path.dirname(os.path.abspath(tim)) or "."
    try:
        os.chdir(timdir)
        model, toas = get_model_and_toas(os.path.relpath(par, timdir), os.path.basename(tim),
                                         ephem="DE440", planets=True)
    finally:
        os.chdir(cwd)
    residuals_orig = None
    if convert_low_ecc and model_ecc(model) < ECC_LOW_THRESHOLD and not is_ell1_model(model):
        if KOPEIKIN_FIXED and has_kopeikin(model):
            toas_o = drop_toas_outside_dmx(model, toas)
            # deepcopy: TimingModel has no .copy()
            model_o = drop_empty_dmx_bins(deepcopy(model), toas_o)
            psr_o = Pulsar(toas_o, model_o, timing_package="pint")
            residuals_orig = np.asarray(psr_o.residuals, dtype=float)
            del psr_o
        model = convert_to_ell1(model, skip_pb_if_fb0_free=skip_pb_if_fb0_free)
    toas = drop_toas_outside_dmx(model, toas)
    model = drop_empty_dmx_bins(model, toas)
    # Freeze right before building the Pulsar, which builds the design matrix
    # from the free parameters (psr.Mmat, psr.fitpars).
    if FROZEN_PARAMS:
        frozen = []
        for p in FROZEN_PARAMS:
            o = getattr(model, p, None)
            if o is not None and getattr(o, "value", None) is not None:
                o.frozen = True
                frozen.append(p)
        if frozen:
            print("    [frozen] removed from the design matrix: %s" % ", ".join(frozen), flush=True)
    psr = Pulsar(toas, model, timing_package="pint")
    if residuals_orig is not None:
        if len(residuals_orig) != len(psr.residuals):
            raise RuntimeError("the original and the converted model give different TOA sets "
                               "(%d vs %d): cannot keep the Kopeikin terms"
                               % (len(residuals_orig), len(psr.residuals)))
        delta = np.asarray(psr.residuals, dtype=float) - residuals_orig
        # psr.residuals returns _residuals[_isort] and residuals_orig is already
        # time-ordered: undo that permutation
        psr._residuals = residuals_orig[psr._iisort]
        print("    [Kopeikin] drift restored: %.3f us rms, %.3f us peak to peak"
              % (np.std(delta) * 1e6, (delta.max() - delta.min()) * 1e6), flush=True)
    return psr, model


def build_context(psr, timing_model):
    """TOA times, design-matrix partials of the residuals with respect to the
    five orbital parameters of the template, and the orbital elements.
    Returns (t_arr, partials, orb_pars, missing)."""
    M = np.asarray(psr.Mmat, dtype=np.float64)
    M_names = list(psr.fitpars)
    t_arr = np.asarray(psr.toas, dtype=np.float64)
    n_toa = len(t_arr)

    def col(name, conv):
        if name in M_names:
            return np.ascontiguousarray(M[:, M_names.index(name)] * conv, dtype=np.float64)
        return None

    pb_q = getattr(getattr(timing_model, "PB", None), "quantity", None)
    fb0 = None
    if pb_q is not None:
        PB_s = float(pb_q.to(u.s).value)
    else:
        fb0_q = getattr(getattr(timing_model, "FB0", None), "quantity", None)
        if fb0_q is None:
            raise RuntimeError("the model has neither PB nor FB0")
        fb0 = float(fb0_q.to(u.Hz).value)
        PB_s = 1.0 / fb0

    orb_pars = {"PB_s": PB_s, "A1_s": float(timing_model.A1.value)}
    orb_pars["omega_b"] = 2.0 * np.pi / orb_pars["PB_s"]

    if hasattr(timing_model, "EPS1") and timing_model.EPS1.value is not None:
        eps1 = float(timing_model.EPS1.value)
        eps2 = float(timing_model.EPS2.value)
        orb_pars.update(eps1=eps1, eps2=eps2, ecc=float(np.hypot(eps1, eps2)),
                        omega=float(np.arctan2(eps1, eps2)),
                        TASC_s=float(timing_model.TASC.value * DAY_IN_S))
        spec = [("PB", 1.0 / DAY_IN_S), ("A1", 1.0),
                ("EPS1", 1.0), ("EPS2", 1.0), ("TASC", 1.0 / DAY_IN_S)]
    else:
        orb_pars.update(ecc=float(timing_model.ECC.value),
                        OM_rad=float(np.deg2rad(timing_model.OM.value)),
                        T0_s=float(timing_model.T0.value * DAY_IN_S))
        spec = [("PB", 1.0 / DAY_IN_S), ("A1", 1.0),
                ("ECC", 1.0), ("OM", 180.0 / np.pi), ("T0", 1.0 / DAY_IN_S)]

    partials, missing = {}, []
    for name, conv in spec:
        c = col(name, conv)
        if c is None:
            partials[name] = np.zeros(n_toa, dtype=np.float64)
            missing.append(name)
        else:
            partials[name] = c
    if "PB" in missing and "FB0" in M_names:
        # chain rule: d/dPB = d/dFB0 * (-FB0^2)
        fb0_col = col("FB0", 1.0)
        if fb0 is None:
            fb0 = 1.0 / orb_pars["PB_s"]
        partials["PB"] = fb0_col * (-(fb0 ** 2))
        missing.remove("PB")
    return t_arr, partials, orb_pars, missing


# ---- ENTERPRISE model ---------------------------------------------------------
def by_fe_flag(flags):
    return {v: flags["fe"] == v for v in sorted(np.unique(flags["fe"]))}


def dmjumps_par(par, psr_name, pta):
    """{parameter name: +DMJUMP} for every frontend of the pulsar, from the .par
    file. PINT models the wideband DMs as DM + DMX - DMJUMP(fe) and ENTERPRISE
    adds dmjump to the measured DMs, so the consistent value is +DMJUMP, as in
    enterprise_extensions."""
    J = {}
    for line in open(par):
        t = line.split()
        if t and t[0] == "DMJUMP" and len(t) > 3 and t[1] == "-fe":
            J[t[2]] = float(t[3].replace("D", "e"))
    out = {}
    for p in pta.param_names:
        if p.endswith("_dmjump"):
            fe = p[len(psr_name) + 1:-len("_dmjump")]
            out[p] = J.get(fe, 0.0)
    return out


def build_pta(psr, t_arr, uldm):
    """White noise per backend, achromatic power-law red noise on N_FREQ_RN
    Fourier modes, wideband DM model, and the ULDM deterministic signal."""
    from enterprise.signals import parameter, white_signals, gp_signals, gp_priors, signal_base
    from enterprise.signals.selections import Selection

    def by_f_flag(flags):
        return {v: flags["f"] == v for v in sorted(np.unique(flags["f"]))}
    sel = Selection(by_f_flag)

    ef = white_signals.MeasurementNoise(
        efac=parameter.Uniform(0.01, 10.0),
        log10_t2equad=parameter.Uniform(-10.0, -1.0), selection=sel)
    # red-noise Fourier coefficients are marginalized analytically by ENTERPRISE
    rn = gp_signals.FourierBasisGP(
        spectrum=gp_priors.powerlaw(
            log10_A=parameter.Uniform(LOG10_ARED_MIN, LOG10_ARED_MAX),
            gamma=parameter.Uniform(GAMMA_RED_MIN, GAMMA_RED_MAX)),
        components=N_FREQ_RN, Tspan=(t_arr.max() - t_arr.min()))
    dm_wtm = gp_signals.WidebandTimingModel(
        dmefac=parameter.Uniform(0.1, 5.0),
        log10_dmequad=parameter.Uniform(-10.0, -1.0),
        dmjump=parameter.Uniform(-1.0, 1.0), selection=sel,
        dmjump_selection=Selection(by_fe_flag))
    signal = (ef + rn + dm_wtm + uldm)(psr)
    return signal_base.PTA([signal])


def find_param_names(pta, suffixes):
    """The parameter of the PTA ending in each suffix (exactly one each)."""
    out = []
    for suf in suffixes:
        matches = [p for p in pta.param_names if p.endswith(suf)]
        if len(matches) != 1:
            raise RuntimeError(f"suffix '{suf}': {matches}")
        out.append(matches[0])
    return out


def get_rn_pnames(pta):
    return find_param_names(pta, RN_SUFFIXES)


def load_noise_MAP(pta, chain_file, pars_file, signal_names):
    """Maximum a posteriori values of the noise parameters from the noise chain
    released with the data set. All but the red noise are held fixed there."""
    chain_raw = np.loadtxt(chain_file)
    with open(pars_file) as fh:
        par_names = [line.strip() for line in fh if line.strip()]
    n_par = len(par_names)
    imap = int(np.argmax(chain_raw[:, n_par]))
    d = dict(zip(par_names, chain_raw[imap, :n_par]))
    for k in list(d.keys()):
        if k.endswith("_log10_equad"):
            d[k[:-len("_log10_equad")] + "_log10_t2equad"] = d.pop(k)
    noise_pnames = [p for p in pta.param_names if p not in signal_names]
    map_noise = {k: d[k] for k in noise_pnames if k in d}
    missing = {p for p in set(noise_pnames) - set(map_noise) if not p.endswith("_dmjump")}
    if missing:
        print(f"    [warn] {len(missing)} noise parameters missing: {sorted(missing)[:3]}")
    return map_noise


# ---- sampling -----------------------------------------------------------------
def effective_sample_size(x):
    """ESS from the integrated autocorrelation time (summed up to the first
    non-positive lag)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 10:
        return float(n)
    x = x - x.mean()
    var0 = np.dot(x, x) / n
    if var0 <= 0:
        return float(n)
    nfft = 1
    while nfft < 2 * n:
        nfft *= 2
    f = np.fft.rfft(x, n=nfft)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for k in range(1, n):
        if acf[k] <= 0.0:
            break
        tau += 2.0 * acf[k]
    return float(n / max(tau, 1.0))


def run_one_chain(model, outDir, n_iter, rng, comm=None):
    """One PTMCMCSampler chain; returns column 0 (log10 of the coupling) after
    burn-in. rng sets the starting point; the sampler draws its own seed."""
    from PTMCMCSampler import PTMCMCSampler
    os.makedirs(outDir, exist_ok=True)
    x0 = model.x0(rng)
    cov = model.cov0()
    kw = {} if comm is None else {"comm": comm}
    sampler = PTMCMCSampler.PTSampler(model.ndim, model.lnlikelihood, model.lnprior, np.copy(cov),
                                      outDir=outDir, resume=False, **kw)
    nburn = int(BURN_FRAC * n_iter)
    sampler.sample(x0, n_iter, Tmax=1, covUpdate=COV_UPDATE,
                   SCAMweight=SCAM_WEIGHT, AMweight=AM_WEIGHT, DEweight=DE_WEIGHT,
                   thin=1, isave=10_000, writeHotChains=False)
    files = sorted(glob.glob(os.path.join(outDir, "chain_1*.txt")))
    if not files:
        raise RuntimeError(f"no chain found in {outDir}")
    return np.loadtxt(files[0])[nburn:, 0]
