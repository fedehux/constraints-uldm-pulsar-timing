# =============================================================================
# Vector (spin-1) ULDM coupled to B-L, with dimensionless coupling g.
#
# Single-pulsar analysis. The signal is linear in g and depends on the
# direction of the field, given by a polar angle vartheta (isotropic prior) and
# an azimuth varphi ~ U(0, 2pi). For each (mass, pulsar) a PTMCMC chain samples
# seven parameters,
#   (log10_g, log10_rho, gamma_f, cos(vartheta), varphi, log10_A_red, gamma_red).
# White noise and the wideband DM model are fixed at their MAP values; the
# timing model and the red-noise Fourier coefficients are marginalized
# analytically. The force goes as cos(m t), so the resonances are at m = n w_b.
# =============================================================================
import csv
import os
import time

import numpy as np

from common import (ALLOWED_MODELS, ECC_LOW_THRESHOLD, EV_TO_INV_S, DM_DENSITY,
                    RHO_F_MIN, RHO_F_MAX, LOG10_RHO_MIN, LOG10_RHO_MAX,
                    LOG10_ARED_MIN, LOG10_ARED_MAX, GAMMA_RED_MIN, GAMMA_RED_MAX,
                    BURN_FRAC, COV_UPDATE, SCAM_WEIGHT, AM_WEIGHT, DE_WEIGHT,
                    SIGMA_LOG10RHO, SIGMA_GAMMA, SIGMA_LOG10_ARED, SIGMA_GAMMA_RED,
                    resolve_paths, discover_wb_pulsars, filter_pulsars_by_model,
                    dmjumps_par, get_rn_pnames, find_param_names,
                    effective_sample_size, run_one_chain)
import common
from prior_ranges import PRIOR_RANGES
from template_ecc_vec import compute_h_template_ecc, N_TERMS_ECC

LIKELIHOOD = "enterprise"
VERSION_PIPELINE = "final-2026-10-10-dc05"
CHAIN_ROOT = os.path.join(os.environ.get("OUT_ROOT", "."), "chains")

# Nucleon mass in natural units
M_NUCLEON = 1.67492729e-27 * 8.52246714808644e+50
# B-L charge-to-mass difference between the two components of the binary:
# c_NS ~ 1.0-1.1, c_WD = 0.504 -> Delta c ~ 0.5
DELTA_C = 0.5

# Kepler's third law for the relative semi-major axis a (natural units)
M_SUN_KG = 2.0e30
G_CONST = 6.67e-11
KEPLER_UNIT_CONV = 3.71140109219707e-26
DEFAULT_M1 = 1.35                    # pulsar mass [M_sun] if not tabulated
DEFAULT_M2 = 0.20                    # companion mass [M_sun] if not tabulated
DEFAULT_SINI = np.sin(np.deg2rad(60.0))
COMPANION_MASS_PRIORS = {
    "J0406+3039": 0.096246, "J0610-2100": 0.024748, "J1012+5307": 0.124641,
    "J1125+7819": 0.329741, "J1713+0747": 0.324201, "J1719-1438": 0.001304,
    "J1738+0333": 0.103990, "J1745+1017": 0.015826, "J1802-2124": 0.981815,
    "J2145-0750": 0.503270, "J2234+0944": 0.017786, "J2317+1439": 0.201229,
}

# Masses and inclination of every pulsar (m1, m2, sin i), consistent with the
# mass function (Appendix C of the paper):
#   A = m1 and m2 published together, i from the mass function
#   B = i measured or bounded (or SINI of the .par file if the Shapiro delay is
#       weak), m1 = 1.35, m2 from the mass function
#   C = no measurement: ATNF convention (m1 = 1.35, i = 60 deg, m2 = median mass)
# A pulsar missing from the table falls back to DEFAULT_M1, the M2 of the model
# or COMPANION_MASS_PRIORS, and model_sini, with a warning.
PULSAR_MASSES_CSV = os.environ.get(
    "PULSAR_MASSES_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "pulsar_masses.csv"))


def load_mass_table(path=PULSAR_MASSES_CSV):
    """{psr: {'m1', 'm2', 'sini', 'case', 'source'}}, or None if the file is missing."""
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf8") as fh:
        rows = csv.DictReader(line for line in fh if not line.startswith("#"))
        return {r["psr"]: dict(m1=float(r["m1"]), m2=float(r["m2"]), sini=float(r["sini"]),
                               case=r["case"], source=r["source"]) for r in rows}


MASS_TABLE = load_mass_table()

# Prior range of log10 g (prior_ranges.py). The template is built for g = 1
# and includes Delta c / m_N * sqrt(2 rho_DM) * rho.
LOG10_G_MIN = float(os.environ.get("LOG10_G_MIN", PRIOR_RANGES["vec"]["LOG10_G_MIN"]))
LOG10_G_MAX = float(os.environ.get("LOG10_G_MAX", PRIOR_RANGES["vec"]["LOG10_G_MAX"]))

# "uniform": prior uniform in g (upper limits); "log-uniform": uniform in
# log10(g) (injection test). Only the g term of lnprior changes.
COUPLING_PRIOR = os.environ.get("COUPLING_PRIOR", "uniform").strip().lower()
if COUPLING_PRIOR not in ("uniform", "log-uniform"):
    raise ValueError(f"COUPLING_PRIOR={COUPLING_PRIOR!r}: uniform | log-uniform")

# The PB channel uses the running time average <delta a/a> (see below)
AVERAGE_PB = True

LOG10_G_INIT = -26.0                 # starting point, +- 2 dex
SIGMA_LOG10G = 3.0
# The polar angle is sampled as x[3] = cos(vartheta) ~ U(-1, 1): isotropic, with
# a flat prior and no log(sin vartheta) singularity at the poles.
SIGMA_COS_VARTHETA = 0.5
SIGMA_VARPHI = 2 * np.pi

# Histogram of log10 g written by the worker: 0.005 dex bins.
N_G_BINS = int(round((LOG10_G_MAX - LOG10_G_MIN) / 0.005))
G_EDGES = np.linspace(LOG10_G_MIN, LOG10_G_MAX, N_G_BINS + 1)


# ---- pulsar loading (vector specific) --------------------------------------------
def load_pulsar(par, tim, convert_low_ecc=True):
    # For an orbit parametrized by a free FB0, PB = 1/FB0 is not added as a
    # design-matrix column: it is degenerate with FB0, worsens the condition
    # number and absorbs signal; build_context builds the PB partial from FB0.
    return common.load_pulsar(par, tim, convert_low_ecc=convert_low_ecc, skip_pb_if_fb0_free=True)


def model_sini(model):
    """sin i from the timing model: SINI (Shapiro), else KIN (Kopeikin), else DEFAULT_SINI."""
    q = getattr(getattr(model, "SINI", None), "value", None)
    if q is not None:
        try:
            return float(np.clip(float(q), 1e-3, 1.0))
        except Exception:
            pass
    kin = getattr(getattr(model, "KIN", None), "value", None)
    if kin is not None:
        try:
            return float(np.clip(np.sin(np.deg2rad(float(kin))), 1e-3, 1.0))
        except Exception:
            pass
    return float(DEFAULT_SINI)


def model_m2(model):
    """Companion mass [M_sun] from the timing model (M2, Shapiro), or None."""
    q = getattr(getattr(model, "M2", None), "value", None)
    if q is not None:
        try:
            m2 = float(q)
            if np.isfinite(m2) and m2 > 0:
                return m2
        except Exception:
            pass
    return None


def build_context(psr, timing_model):
    """common.build_context plus the inclination and the relative semi-major
    axis, which the vector template needs."""
    t_arr, partials, orb_pars, missing = common.build_context(psr, timing_model)
    psr_name = getattr(psr, "name", "") or ""
    row = (MASS_TABLE or {}).get(psr_name)
    if row is not None:
        m1, m2 = row["m1"], row["m2"]
        sin_i = float(np.clip(row["sini"], 1e-3, 1.0))
        orb_pars["mass_source"] = "table, case %s (%s)" % (row["case"], row["source"])
    else:
        sin_i = model_sini(timing_model)
        m2 = model_m2(timing_model)
        if m2 is None:
            m2 = COMPANION_MASS_PRIORS.get(psr_name, DEFAULT_M2)
        m1 = DEFAULT_M1
        orb_pars["mass_source"] = "defaults"
        print("    [masses] %s is not in %s: defaults used (m1=%.2f, m2=%.3f, sin i=%.4f)"
              % (psr_name, PULSAR_MASSES_CSV, m1, m2, sin_i), flush=True)
    orb_pars["sin_i"] = sin_i
    orb_pars["iota"] = float(np.arcsin(sin_i))
    a_semi = (G_CONST * M_SUN_KG * (m1 + m2) * KEPLER_UNIT_CONV
              / orb_pars["omega_b"] ** 2) ** (1.0 / 3.0)
    orb_pars.update(m1=float(m1), m2=float(m2), a_semi=float(a_semi))
    return t_arr, partials, orb_pars, missing


# ---- ULDM template ------------------------------------------------------------
# h = sum_param partial(residual)/partial(param) * delta(param), for g = 1.
#
# Every coefficient is a sum of terms [trig((k+n) Psi + p) - trig(p)]/(k+n).
# At the resonance k = n the term is 0/0 and its limit is secular (linear in
# Psi). Product-to-sum identities leave an explicit sinc (np.sinc(0) = 1), so
# the expressions are continuous and stable through the resonance:
#   [sin(d Psi + p) - sin(p)]/d =  Psi cos(d Psi/2 + p) sinc(d Psi/2pi)
#   [cos(d Psi + p) - cos(p)]/d = -Psi sin(d Psi/2 + p) sinc(d Psi/2pi)
#   [1 - cos(kk Psi)]/kk        =  Psi sin(kk Psi/2)   sinc(kk Psi/2pi)
#   sin(kk Psi)/kk              =  Psi sinc(kk Psi/pi)

def _rsin(delta, Psi, p):
    """[sin(delta*Psi + p) - sin(p)] / delta ; limit delta -> 0: Psi cos(p)."""
    return Psi * np.cos(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _rcos(delta, Psi, p):
    """[cos(delta*Psi + p) - cos(p)] / delta ; limit delta -> 0: -Psi sin(p)."""
    return -Psi * np.sin(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _rver(kk, Psi):
    """[1 - cos(kk*Psi)] / kk ; limit kk -> 0: 0.5 kk Psi^2 (-> 0)."""
    return Psi * np.sin(0.5 * kk * Psi) * np.sinc(kk * Psi / (2.0 * np.pi))


def _rs0(kk, Psi):
    """sin(kk*Psi) / kk ; limit kk -> 0: Psi."""
    return Psi * np.sinc(kk * Psi / np.pi)


# PB is inferred from the accumulated orbital phase, so the PB channel sees the
# running average <delta a/a> = (1/Psi) int_0^Psi (delta a/a) dPsi', while the A1
# channel (x = a1 sin i) sees the instantaneous value. delta a/a is a sum of
# _rsin/_rcos terms, whose averages are, with z = d Psi,
#   <_rsin(d, Psi, p)> =  Psi [0.5 cos p sinc(z/2pi)^2 - sin p (z - sin z)/z^2]
#   <_rcos(d, Psi, p)> = -Psi [0.5 sin p sinc(z/2pi)^2 + cos p (z - sin z)/z^2]
# At d -> 0 (resonance) sinc^2 -> 1 and (z - sin z)/z^2 -> 0.

def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, stable (0 at z = 0, ~z/6). Taylor series for |z| < 0.1,
    where z - sin z loses all its digits; direct difference elsewhere."""
    z = np.asarray(z, dtype=np.float64)
    small = np.abs(z) < 0.1
    zs = np.where(small, 1.0, z)
    direct = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    series = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(small, series, direct)


def _avg_rsin(delta, Psi, p):
    """<_rsin(delta, Psi', p)> averaged over Psi' from 0 to Psi (stable at delta -> 0)."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return Psi * (0.5 * np.cos(p) * s2 - np.sin(p) * _z_minus_sin_over_z2(z))


def _avg_rcos(delta, Psi, p):
    """<_rcos(delta, Psi', p)> averaged over Psi' from 0 to Psi (stable at delta -> 0)."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return -Psi * (0.5 * np.sin(p) * s2 + np.cos(p) * _z_minus_sin_over_z2(z))


def compute_h_template(m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars):
    """Timing residual for g = 1. Dispatches to the ELL1 template at low
    eccentricity and to template_ecc_vec.py (Appendix B of the paper) otherwise.
    In the eccentric case gamma_f is Upsilon_0 = Upsilon + m T0 and varphi is
    the azimuth measured from the periastron; both are sampled directly
    (shifting a U(0, 2pi) variable by a constant leaves it U(0, 2pi))."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(
            m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars)
    F_unit_const = (DELTA_C / M_NUCLEON) * np.sqrt(2.0 * DM_DENSITY)
    return compute_h_template_ecc(m_dm, rho_f, gamma_f, vartheta, varphi,
                                  t_arr, partials, orb_pars, F_unit_const,
                                  N=N_TERMS_ECC, average_pb=AVERAGE_PB)


def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars):
    """ELL1 (low-eccentricity) vector template, in the circular limit e -> 0
    with theta = Psi' - omega. The channels are
      PB   : delta Pb / Pb = (3/2) delta a / a
      A1   : delta x / x   = delta a / a + cot(iota) delta iota   (x = a1 sin i)
      EPS1 : delta eta     (eta = e sin omega)
      EPS2 : delta kappa   (kappa = e cos omega)
      TASC : delta T_asc
    each of the form prefactor * [S(Psi') cos(Ups_asc) + C(Psi') sin(Ups_asc)].

    The argument of periastron omega enters only through the shifted phase
    varphi_asc = varphi + omega. omega is ill defined for a circular orbit, so
    varphi_asc is sampled directly: the varphi received here is varphi_asc.
    Likewise gamma_f is Upsilon_asc = Upsilon + m Tasc. Both have U(0, 2pi)
    priors, since shifting a uniform phase by a constant leaves it uniform.
    """
    wb = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    a_semi = orb_pars["a_semi"]
    iota = orb_pars["iota"]
    sin_i = orb_pars["sin_i"]

    # amplitude for g = 1:  F_unit = (Delta_c / m_N) sqrt(2 rho_DM) rho
    F_unit = (DELTA_C / M_NUCLEON) * np.sqrt(2.0 * DM_DENSITY) * rho_f
    sin_vt = np.sin(vartheta)
    cos_vt = np.cos(vartheta)
    cot_i = np.cos(iota) / sin_i

    Psi = wb * (t_arr - tasc_s)             # Psi' = wb (t - Tasc)
    k = m_dm / wb
    Ups = gamma_f                           # Upsilon_asc
    cU = np.cos(Ups)
    sU = np.sin(Ups)
    phi_asc = varphi                        # varphi_asc
    cpa = np.cos(phi_asc)
    spa = np.sin(phi_asc)

    pref = F_unit / (a_semi * wb ** 2)
    prefT = F_unit / (a_semi * wb ** 3)     # delta T_asc carries an extra 1/wb

    # ---- delta a / a, instantaneous (A1 channel through dxx) ----
    Sa = -_rsin(k + 1.0, Psi, -phi_asc) + _rsin(k - 1.0, Psi, phi_asc)
    Ca = -_rcos(k + 1.0, Psi, -phi_asc) + _rcos(k - 1.0, Psi, phi_asc)
    daa = pref * sin_vt * (Sa * cU + Ca * sU)

    # ---- <delta a / a>, running time average (PB channel) ----
    Sa_avg = -_avg_rsin(k + 1.0, Psi, -phi_asc) + _avg_rsin(k - 1.0, Psi, phi_asc)
    Ca_avg = -_avg_rcos(k + 1.0, Psi, -phi_asc) + _avg_rcos(k - 1.0, Psi, phi_asc)
    daa_avg = pref * sin_vt * (Sa_avg * cU + Ca_avg * sU)

    # ---- delta iota ----
    Si = 0.5 * (_rver(k + 1.0, Psi) + _rver(k - 1.0, Psi))
    Ci = 0.5 * (_rs0(k + 1.0, Psi) + _rs0(k - 1.0, Psi))
    diota = -pref * cos_vt * (Si * cU + Ci * sU)

    # ---- delta x / x = delta a / a + cot(iota) delta iota ----
    dxx = daa + cot_i * diota

    # ---- delta eta (EPS1) ----
    Se = (-1.5 * _rver(k, Psi) * cpa
          - 0.25 * _rcos(k + 2.0, Psi, -phi_asc)
          - 0.25 * _rcos(k - 2.0, Psi, phi_asc))
    Ce = (-1.5 * _rs0(k, Psi) * cpa
          + 0.25 * _rsin(k + 2.0, Psi, -phi_asc)
          + 0.25 * _rsin(k - 2.0, Psi, phi_asc))
    dEPS1 = -pref * sin_vt * (Se * cU + Ce * sU)

    # ---- delta kappa (EPS2) ----
    Sk = (1.5 * _rver(k, Psi) * spa
          + 0.25 * _rsin(k + 2.0, Psi, -phi_asc)
          - 0.25 * _rsin(k - 2.0, Psi, phi_asc))
    Ck = (1.5 * _rs0(k, Psi) * spa
          + 0.25 * _rcos(k + 2.0, Psi, -phi_asc)
          - 0.25 * _rcos(k - 2.0, Psi, phi_asc))
    dEPS2 = -pref * sin_vt * (Sk * cU + Ck * sU)

    # ---- delta T_asc (time integral of T_asc_dot) ----
    ST = (-0.5 * cot_i * cos_vt * (_rs0(k - 1.0, Psi) - _rs0(k + 1.0, Psi))
          - sin_vt * (-_rcos(k - 1.0, Psi, phi_asc) - _rcos(k + 1.0, Psi, -phi_asc)))
    CT = (-0.5 * cot_i * cos_vt * (-_rver(k - 1.0, Psi) + _rver(k + 1.0, Psi))
          - sin_vt * (_rsin(k - 1.0, Psi, phi_asc) + _rsin(k + 1.0, Psi, -phi_asc)))
    dTASC = prefT * (ST * cU + CT * sU)

    # delta Pb/Pb = (3/2) delta a/a: the B-L field does not rescale G M_T, so
    # there is no extra -(1/2) delta(GM)/GM term as for the universal scalar.
    daa_pb = daa_avg if AVERAGE_PB else daa
    # Full ELL1 delay: Delta_R = x[sin Phi + k/2 sin 2Phi - h/2 cos 2Phi - 3/2 h],
    # h = eta = EPS1. tempo and PINT drop the constant -3/2 x eta (Lange et al.
    # 2001, Eq. A6), so the EPS1 column of the design matrix is only
    # -x cos(2Phi)/2. The ULDM makes eta oscillate, and -3/2 x delta_eta(t) is a
    # delay no timing parameter absorbs, hence the -1.5 A1 term.
    return (partials["PB"] * PB_s * (1.5 * daa_pb)
            + partials["A1"] * A1_s * dxx
            + (partials["EPS1"] - 1.5 * A1_s) * dEPS1
            + partials["EPS2"] * dEPS2
            + partials["TASC"] * dTASC)


# ---- ENTERPRISE signal and PTA ----------------------------------------------------
def make_uldm_signal(t_arr, partials, orb_pars, m_dm):
    from enterprise.signals import parameter, signal_base, deterministic_signals

    @signal_base.function
    def uldm_delay(toas, log10_g=-15.0, rho_f=1.0, gamma_f=0.0, vartheta=np.pi / 2, varphi=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars)
        return (10.0 ** log10_g) * h

    # ranges only; the priors (Rayleigh in rho, isotropic direction) are
    # imposed in PulsarULDM_RN.lnprior
    lg = parameter.Uniform(LOG10_G_MIN, LOG10_G_MAX)("log10_g")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0 * np.pi)("gamma_f")
    vt = parameter.Uniform(0.0, np.pi)("vartheta")
    vp = parameter.Uniform(0.0, 2.0 * np.pi)("varphi")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_g=lg, rho_f=rf, gamma_f=gf, vartheta=vt, varphi=vp), name="uldm")


def build_pta(psr, t_arr, partials, orb_pars, m_dm):
    return common.build_pta(psr, t_arr, make_uldm_signal(t_arr, partials, orb_pars, m_dm))


ULDM_SUFFIXES = ["log10_g", "rho_f", "gamma_f", "vartheta", "varphi"]


def get_uldm_pnames(pta):
    return find_param_names(pta, ULDM_SUFFIXES)


def load_noise_MAP(pta, chain_file, pars_file):
    return common.load_noise_MAP(pta, chain_file, pars_file, get_uldm_pnames(pta))


# ---- sampled model ------------------------------------------------------------------
class PulsarULDM_RN(object):
    """Seven-parameter model of one pulsar with marginalized red noise:
      x = (log10_g, log10_rho, gamma_f, cos(vartheta), varphi, log10_A_red, gamma_red)
    White noise and wideband DM fixed at the MAP values. gamma_f and varphi are
    the shifted phases Upsilon_asc and varphi_asc (see compute_h_template_low_ecc).
    x[3] = cos(vartheta) ~ U(-1, 1) gives an isotropic direction with a flat
    prior; the template receives vartheta = arccos(x[3]).
    """

    def __init__(self, d):
        self.d = d
        self.pta = d["pta"]
        self.fix = d["noise_fixed"]
        self.lg, self.rf, self.gf = d["lg"], d["rf"], d["gf"]
        self.vt, self.vp = d["vt"], d["vp"]
        self.arn, self.grn = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim = 7

    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.lg] = x[0]
        p[self.rf] = 10.0 ** x[1]
        p[self.gf] = x[2]
        p[self.vt] = np.arccos(np.clip(x[3], -1.0, 1.0))
        p[self.vp] = x[4]
        p[self.arn] = x[5]
        p[self.grn] = x[6]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_G_MIN <= x[0] <= LOG10_G_MAX):
            return -np.inf
        # uniform in g: ln p(log10 g) = log10(g) ln10
        lp = x[0] * np.log(10.0) if COUPLING_PRIOR == "uniform" else 0.0
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho = 10.0 ** u
        lp += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh in u = log10 rho
        if not (0.0 <= x[2] <= 2.0 * np.pi):
            return -np.inf
        if not (-1.0 <= x[3] <= 1.0):        # cos(vartheta): sin(vartheta) dvartheta = -d cos(vartheta)
            return -np.inf
        if not (0.0 <= x[4] <= 2.0 * np.pi):
            return -np.inf
        if not (LOG10_ARED_MIN <= x[5] <= LOG10_ARED_MAX):
            return -np.inf
        if not (GAMMA_RED_MIN <= x[6] <= GAMMA_RED_MAX):
            return -np.inf
        return lp

    def x0(self, rng):
        x = np.empty(self.ndim)
        x[0] = LOG10_G_INIT + rng.uniform(-2.0, 2.0)
        x[1] = 0.0
        x[2] = np.pi
        x[3] = 0.0                           # equator
        x[4] = np.pi
        x[5] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + 0.1, LOG10_ARED_MAX - 0.1))
        x[6] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + 0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10G**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
                        SIGMA_COS_VARTHETA**2, SIGMA_VARPHI**2,
                        SIGMA_LOG10_ARED**2, SIGMA_GAMMA_RED**2])


def make_model(d):
    return PulsarULDM_RN(d)


# ---- one (mass, pulsar) task ----------------------------------------------------------
def preload_one(name, m_eV):
    m_dm = m_eV * EV_TO_INV_S
    t_load = time.time()
    try:
        par, tim, chain, pars = resolve_paths(name)
        psr, tm = load_pulsar(par, tim, convert_low_ecc=True)
        t_arr, partials, orb_pars, missing = build_context(psr, tm)
        # a channel without a design-matrix column would enter the template as zero
        if missing:
            raise RuntimeError(f"{psr.name}: template channels without a design-matrix column: {missing}")
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        lg, rf, gf, vt, vp = get_uldm_pnames(pta)
        arn, grn = get_rn_pnames(pta)
        noise = load_noise_MAP(pta, chain, pars)
        noise.update(dmjumps_par(par, psr.name, pta))
        unset = sorted(set(pta.param_names) - set(get_uldm_pnames(pta)) - set(noise))
        if unset:
            raise RuntimeError(f"{psr.name}: noise parameters without a value: {unset}")
        if arn not in noise or grn not in noise:
            raise RuntimeError(f"red-noise MAP missing: {arn in noise=}, {grn in noise=}")
        rn_map = (noise[arn], noise[grn])
        noise_fixed = {k: v for k, v in noise.items() if k not in (arn, grn)}
        return dict(ctx=(t_arr, partials, orb_pars, m_dm), name=name, pta=pta,
                    lg=lg, rf=rf, gf=gf, vt=vt, vp=vp, arn=arn, grn=grn,
                    noise_fixed=noise_fixed, rn_map=rn_map,
                    ecc=orb_pars["ecc"], sin_i=orb_pars["sin_i"],
                    n_toa=len(t_arr), t_load=time.time() - t_load)
    except Exception as e:
        print(f"  [{name}] PRELOAD FAIL: {type(e).__name__}: {e}", flush=True)
        return None


def process_task(args):
    """args = (mass index, mass [eV], pulsar, iterations, seed of the starting point)."""
    mass_idx, m_eV, name, n_iter, seed = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    d = preload_one(name, m_eV)
    if d is None:
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False, err="preload_failed")
    try:
        model = make_model(d)
        outDir = os.path.join(CHAIN_ROOT, f"m{mass_idx:03d}", name)
        g_chain = run_one_chain(model, outDir, n_iter, rng)
        ess = effective_sample_size(g_chain)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  ecc={d['ecc']:.1e}  "
              f"g=[{g_chain.min():6.1f},{g_chain.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True, g=g_chain, ess=ess,
                    n_toa=d["n_toa"], ecc=d["ecc"], t_min=(time.time() - t0) / 60.0)
    except Exception as e:
        import traceback
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
