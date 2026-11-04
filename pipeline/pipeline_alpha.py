# =============================================================================
# Linear universal coupling of a scalar field:  M_A(Phi) = M_A (1 + alpha Phi)
# (Kus, Lopez Nacir & Urban 2024, arXiv:2402.04099, Eq. 2.14).
#
# Single-pulsar analysis. For each (mass, pulsar) a PTMCMC chain samples five
# parameters, (log10_alpha, log10_rho, gamma_f, log10_A_red, gamma_red): the
# ULDM signal (coupling, field amplitude rho, phase Upsilon) and the achromatic
# red noise. White noise and the wideband DM model are fixed at their MAP
# values; the timing model and the red-noise Fourier coefficients are
# marginalized analytically.
#
# Templates: Eqs. C.1b-d of the paper (BT/DD, eccentric) and C.5b-d (ELL1,
# circular), plus delta T0 / delta Tasc and the extra term of delta Pb.
# =============================================================================
import os
import time

import numpy as np
from scipy.special import jn, jvp

from common import (ALLOWED_MODELS, ECC_LOW_THRESHOLD, EV_TO_INV_S, DM_DENSITY,
                    RHO_F_MIN, RHO_F_MAX, LOG10_RHO_MIN, LOG10_RHO_MAX,
                    LOG10_ARED_MIN, LOG10_ARED_MAX, GAMMA_RED_MIN, GAMMA_RED_MAX,
                    BURN_FRAC, COV_UPDATE, SCAM_WEIGHT, AM_WEIGHT, DE_WEIGHT,
                    SIGMA_LOG10RHO, SIGMA_GAMMA, SIGMA_LOG10_ARED, SIGMA_GAMMA_RED,
                    resolve_paths, discover_wb_pulsars, filter_pulsars_by_model,
                    load_pulsar, build_context, dmjumps_par, get_rn_pnames,
                    find_param_names, effective_sample_size, run_one_chain)
import common
from prior_ranges import PRIOR_RANGES

LIKELIHOOD = "enterprise"
VERSION_PIPELINE = "final-2026-10-06"
CHAIN_ROOT = os.path.join(os.environ.get("OUT_ROOT", "."), "chains")

N_TERMS = 20                     # Bessel harmonics n = 1 .. N_TERMS-1

# UNIT_CONVERSION is (Phi0_code / Phi0_phys)^2. The linear signal carries one
# power of Phi, so the factor that goes with alpha is its square root:
#   alpha_phys [GeV^-1] = 10^a * ALPHA_CONVERSION
UNIT_CONVERSION = 2.30817405213802e+48
ALPHA_CONVERSION = np.sqrt(UNIT_CONVERSION)          # 1.5193e24

# Prior range, set in physical units (prior_ranges.py) and converted to code units.
LOG10_ALPHA_PHYS_MIN = float(os.environ.get("LOG10_ALPHA_PHYS_MIN",
                                            PRIOR_RANGES["alpha"]["LOG10_ALPHA_PHYS_MIN"]))
LOG10_ALPHA_PHYS_MAX = float(os.environ.get("LOG10_ALPHA_PHYS_MAX",
                                            PRIOR_RANGES["alpha"]["LOG10_ALPHA_PHYS_MAX"]))
_LOG10_CONV = np.log10(ALPHA_CONVERSION)
LOG10_ALPHA_MIN = LOG10_ALPHA_PHYS_MIN - _LOG10_CONV
LOG10_ALPHA_MAX = LOG10_ALPHA_PHYS_MAX - _LOG10_CONV

# "uniform": prior uniform in alpha (upper limits); "log-uniform": uniform in
# log10(alpha) (injection test). Only the alpha term of lnprior changes.
COUPLING_PRIOR = os.environ.get("COUPLING_PRIOR", "uniform").strip().lower()
if COUPLING_PRIOR not in ("uniform", "log-uniform"):
    raise ValueError(f"COUPLING_PRIOR={COUPLING_PRIOR!r}: uniform | log-uniform")

LOG10_ALPHA_INIT = -45.0         # starting point (code units), +- 2 dex
SIGMA_LOG10A = 3.0

# Histogram of log10 alpha written by the worker: 0.005 dex bins.
N_ALPHA_BINS = int(round((LOG10_ALPHA_MAX - LOG10_ALPHA_MIN) / 0.005))
ALPHA_EDGES = np.linspace(LOG10_ALPHA_MIN, LOG10_ALPHA_MAX, N_ALPHA_BINS + 1)


# ---- ULDM template ------------------------------------------------------------
# All the time integrals are in closed form, written with sinc so that they are
# exact and stable at the resonances m = n w_b (no 0/0, no special branch).

def _stable_minus(q_col, psi_col, t_arr, t_ref, tt_row):
    """'Minus' branches of the eccentric template, already divided by
    (n wb - m), in a form that is stable at the resonance m = n wb:
        Cm_over =  t~ sin(q(t+t_ref)/2 + psi) sinc(q t~/2pi)
        Sm_over = -t~ cos(q(t+t_ref)/2 + psi) sinc(q t~/2pi)
    with t~ = t - t_ref, q = m - n wb, psi = n wb t_ref + Y.
    """
    mid = q_col * (0.5 * (t_arr[None, :] + t_ref)) + psi_col
    sincv = np.sinc(q_col * tt_row / (2.0 * np.pi))
    Cm_over = tt_row * np.sin(mid) * sincv
    Sm_over = -tt_row * np.cos(mid) * sincv
    return Cm_over, Sm_over


def _one_minus_cos_over_z2(z):
    """(1 - cos z)/z^2, exact and stable (1/2 at z = 0)."""
    s = np.sinc(z / (2.0 * np.pi))
    return 0.5 * s * s


def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, stable (0 at z = 0). Taylor series for |z| < 0.1, where
    z - sin z loses all its digits; direct difference elsewhere."""
    z = np.asarray(z, dtype=np.float64)
    small = np.abs(z) < 0.1
    zs = np.where(small, 1.0, z)
    direct = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    series = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(small, series, direct)


def _mean_cos(m_dm, t_arr, t_ref, gamma_f, tt):
    """<cos(mt+Y)> = (1/t~) int_{t_ref}^t cos(mt'+Y) dt', with sinc (no 0/0, no /m)."""
    mid = m_dm * 0.5 * (t_arr + t_ref) + gamma_f
    return np.cos(mid) * np.sinc(m_dm * tt / (2.0 * np.pi))


def _int_cos(omega, phase, t_arr, t_ref):
    """int_{t_ref}^{t} cos(omega t' + phase) dt' in stable closed form:
      = t~ cos(omega (t+t_ref)/2 + phase) sinc(omega t~/2pi),  t~ = t - t_ref.
    At omega = 0 (resonant row) sinc -> 1 and the secular term t~ cos(phase)
    remains. Broadcasting: omega, phase (N,1); t_arr (T,) -> (N,T).
    """
    tt = (t_arr - t_ref)[None, :]
    mid = omega * (0.5 * (t_arr + t_ref))[None, :] + phase
    return tt * np.cos(mid) * np.sinc(omega * tt / (2.0 * np.pi))


def compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """Timing residual for alpha = 1 (code units). The signal is
    sum_param partial(r)/partial(param) * delta(param), with the partials taken
    from the design matrix. Dispatches to the ELL1 template at low eccentricity."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)

    # ---- eccentric (BT/DD) template ----
    n_vec = np.arange(1, N_TERMS)
    n_col = n_vec[:, None]
    jn_col = jn(n_vec, n_vec * orb_pars["ecc"])[:, None]       # J_n(ne)
    jnp_col = jvp(n_vec, n_vec * orb_pars["ecc"])[:, None]     # J_n'(ne)

    wb = orb_pars["omega_b"]
    ecc = orb_pars["ecc"]
    t0_s = orb_pars["T0_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    sqrt1me2 = np.sqrt(1.0 - ecc**2)

    T_tilde = t_arr - t0_s                  # t~ = t - T0
    tt_row = T_tilde[None, :]
    nw = n_vec * wb                         # n wb
    nw_col = nw[:, None]

    # Phi = Phi0 rho cos(m t + Y); the template is linear in Phi, so the
    # amplitude A0 = Phi0 rho factors out
    Phi0 = np.sqrt(2.0 * DM_DENSITY) / m_dm
    A0 = Phi0 * rho_f

    m_t = m_dm * t_arr + gamma_f            # m t + Y       (T,)
    m_t0 = m_dm * t0_s + gamma_f            # m T0 + Y      (scalar)
    cos_mt = np.cos(m_t)
    sin_mt = np.sin(m_t)
    cos_mt0 = np.cos(m_t0)
    sin_mt0 = np.sin(m_t0)

    nw_tt = nw_col * tt_row                 # n wb t~       (N,T)
    cos_nw_tt = np.cos(nw_tt)

    # Every delta(param) is built from two families: 'plus' (phase +n wb t~,
    # denominator n wb + m) and 'minus' (phase -n wb t~, denominator n wb - m).
    # Only the minus family resonates.
    phase_pp = m_t[None, :] + nw_tt
    Cp = np.cos(phase_pp) - cos_mt0         # cos(m t+Y+n wb t~) - cos(m T0+Y)
    Sp = np.sin(phase_pp) - sin_mt0

    q_col = (m_dm - nw)[:, None]            # q = m - n wb
    psi_col = (nw * t0_s + gamma_f)[:, None]
    Cm_over, Sm_over = _stable_minus(q_col, psi_col, t_arr, t0_s, tt_row)
    #  Cm_over = [cos(m t+Y-n wb t~) - cos(m T0+Y)] / (n wb - m)
    #  Sm_over = [sin(m t+Y-n wb t~) - sin(m T0+Y)] / (n wb - m)

    sigma_p = (nw + m_dm)[:, None]          # n wb + m
    sigma_m = (nw - m_dm)[:, None]          # n wb - m
    two_m = 2.0 * m_dm
    coeff_p = ((nw - two_m) / (nw + m_dm))[:, None]   # (n wb - 2m)/(n wb + m)
    coeff_m = (two_m + nw)[:, None]                    # (2m + n wb)

    # ---- delta a / a   (Eq. C.1b) ----
    #   -2 A0 [cos(mt+Y) - cos(mT0+Y)]
    #   -8 A0 sum_n J_n [cos(mt+Y) cos(n wb t~) - cos(mT0+Y)]
    #   +6 A0 wb sum_n n J_n { Cp/(m+n wb) + Cm/(n wb - m) }
    daa_mono = -2.0 * A0 * (cos_mt - cos_mt0)
    daa_8 = -8.0 * A0 * np.sum(jn_col * (cos_mt[None, :] * cos_nw_tt - cos_mt0), axis=0)
    daa_6 = 6.0 * A0 * wb * np.sum(n_col * jn_col * (Cp / sigma_p + Cm_over), axis=0)
    daa = daa_mono + daa_8 + daa_6

    # ---- time average <delta a / a> over [T0, t], seen by the PB channel ----
    # PB is inferred from the accumulated orbital phase, so it responds to the
    # running average of delta a/a, not to its instantaneous value (the A1
    # channel does see the instantaneous one).
    #   int_{T0}^t [cos(mt'+Y) - cos(mT0+Y)] dt' = t~ [<cos> - cos(mT0+Y)]
    mean_cos = _mean_cos(m_dm, t_arr, t0_s, gamma_f, T_tilde)
    daa_monoe = -2.0 * A0 * (mean_cos - cos_mt0)
    # <Cp>: the plus branch never resonates
    mean_Cp = np.cos(m_t0 + 0.5 * sigma_p * tt_row) * \
        np.sinc(sigma_p * tt_row / (2.0 * np.pi)) - cos_mt0
    # int_{T0}^t Cm_over dt' = t~^2 [cos_th0 (z-sin z)/z^2 + sin_th0 (1-cos z)/z^2],
    # z = (m - n wb) t~ (0 at the resonance); both ratios are regular
    z_col = q_col * tt_row
    mean_Cm_over = T_tilde * (cos_mt0 * _z_minus_sin_over_z2(z_col)
                              + sin_mt0 * _one_minus_cos_over_z2(z_col))
    daa_8e = -8.0 * A0 * np.sum(jn_col * (0.5 * (mean_Cp + sigma_m * mean_Cm_over)), axis=0)
    daa_6e = 6.0 * A0 * wb * np.sum(n_col * jn_col * (mean_Cp / sigma_p + mean_Cm_over), axis=0)
    daae = daa_monoe + daa_8e + daa_6e

    # ---- delta e   (Eq. C.1c) ----
    #   A0 (1-e^2)/e sum_n J_n { (n wb-2m)/(n wb+m) Cp + (2m+n wb) Cm/(n wb - m) }
    dECC = A0 * (1.0 - ecc**2) / ecc * np.sum(jn_col * (coeff_p * Cp + coeff_m * Cm_over), axis=0)

    # ---- delta omega   (Eq. C.1d) ----
    #   A0 sqrt(1-e^2)/e sum_n J_n' { (n wb-2m)/(n wb+m) Sp - (2m+n wb) Sm/(n wb - m) }
    dOM = A0 * sqrt1me2 / ecc * np.sum(jnp_col * (coeff_p * Sp - coeff_m * Sm_over), axis=0)

    # ---- delta T0 ----
    # From the quadratic T0 template with the replacement beta Phi^2/2 -> alpha Phi.
    # The first part is -2 times the integral of Phi:
    dT0_mono = -2.0 * A0 * (sin_mt - sin_mt0) / m_dm
    # The harmonics go with Phi (coefficient A_n) and Phidot (B_n), Eq. 2.24:
    #   term_harm = sum_n int_{T0}^{t} [A_n Phi cos(n wb t~) + B_n Phidot sin(n wb t~)] dt'
    # Product to sum:
    #   I1 = int cos(mt+Y) cos(n wb t~) dt' = 1/2 [ _int_cos(m-nwb, Y+nwb T0) + _int_cos(m+nwb, Y-nwb T0) ]
    #   I2 = int sin(mt+Y) sin(n wb t~) dt' = 1/2 [ _int_cos(m-nwb, Y+nwb T0) - _int_cos(m+nwb, Y-nwb T0) ]
    # The m - n wb = 0 row is the secular term t~ cos(Y + n wb T0). The lower
    # limit T0 matches dT0_mono; a different constant would be absorbed by the
    # timing-model marginalization.
    an = 4.0 * jn_col - 2.0 * (1.0 - ecc**2) / ecc * n_col * jnp_col           # A_n / wb
    bn = 4.0 / n_col / wb * jn_col + 4.0 / ecc / wb * (1.0 - ecc**2) * jnp_col  # B_n / wb
    wm = (m_dm - nw)[:, None]
    wp = (m_dm + nw)[:, None]
    phim = (gamma_f + nw * t0_s)[:, None]
    phip = (gamma_f - nw * t0_s)[:, None]
    Dm = _int_cos(wm, phim, t_arr, t0_s)
    Dp = _int_cos(wp, phip, t_arr, t0_s)
    I1 = 0.5 * (Dm + Dp)
    I2 = 0.5 * (Dm - Dp)
    term_harm = np.sum(an * A0 * I1 - bn * A0 * m_dm * I2, axis=0)
    dT0 = dT0_mono - term_harm

    # Extra term of delta Pb besides the semi-major axis:
    # delta Pb/Pb = (3/2) delta a/a - (1/2) delta(G M_T)/(G M_T), and for the
    # universal coupling (Eqs. 2.14, 2.20b) delta(G M_T)/(G M_T) = alpha Phi.
    dPb_extrae = -0.5 * A0 * mean_cos       # -(1/2) <Phi>

    return (partials["PB"] * PB_s * (1.5 * daae + dPb_extrae)
            + partials["A1"] * A1_s * daa
            + partials["ECC"] * dECC
            + partials["OM"] * dOM
            + partials["T0"] * dT0)


def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """ELL1 (low-eccentricity) template, linear coupling. Only the first orbital
    harmonic survives, so there is no sum over n."""
    wb = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]

    T_tilde = t_arr - tasc_s                # t~ = t - Tasc
    Phi0 = np.sqrt(2.0 * DM_DENSITY) / m_dm
    A0 = Phi0 * rho_f

    m_t = m_dm * t_arr + gamma_f            # m t + Y
    m_tasc = m_dm * tasc_s + gamma_f        # m Tasc + Y
    cos_mt = np.cos(m_t)
    sin_mt = np.sin(m_t)
    cos_mtasc = np.cos(m_tasc)
    sin_mtasc = np.sin(m_tasc)

    # ---- delta x / x  (A1, Eq. C.5b) = -2 alpha Phi0 rho [cos(mt+Y) - cos(mTasc+Y)] ----
    daa = -2.0 * A0 * (cos_mt - cos_mtasc)

    # ---- delta eta (EPS1) and delta kappa (EPS2) ----
    # In the circular limit e * omega_dot is O(e^0), of the same order as e_dot,
    # so the periastron contribution must be kept. With theta = Psi' - omega,
    # Psi' = wb (t - Tasc):
    #   etadot   =  alpha [ wb Phi cos Psi' - 2 Phidot sin Psi' ]
    #   kappadot = -alpha [ wb Phi sin Psi' + 2 Phidot cos Psi' ]
    # with Phi = Phi0 cos(k Psi' + Ups_asc), k = m/wb, Ups_asc = m Tasc + Y.
    # Product to sum and one integral in Psi' give
    #   delta_eta   = (A0/wb) [ (wb/2 - m) R_s(k+1) + (wb/2 + m) R_s(k-1) ]
    #   delta_kappa = (A0/wb) [ (wb/2 - m) R_c(k+1) - (wb/2 + m) R_c(k-1) ]
    # with R_s(d) = [sin(d Psi' + Ups_asc) - sin Ups_asc]/d and
    #      R_c(d) = [cos(d Psi' + Ups_asc) - cos Ups_asc]/d, written with sinc
    # so that the resonance k = 1 needs no special branch. The result does not
    # depend on the argument of periastron, which is ill defined for a
    # circular orbit.
    Psi_p = wb * T_tilde
    kk = m_dm / wb
    Ups_asc = m_tasc

    def _R_s(d):
        """[sin(d Psi' + Ups_asc) - sin(Ups_asc)] / d ; stable as d -> 0."""
        return Psi_p * np.cos(0.5 * d * Psi_p + Ups_asc) * np.sinc(d * Psi_p / (2.0 * np.pi))

    def _R_c(d):
        """[cos(d Psi' + Ups_asc) - cos(Ups_asc)] / d ; stable as d -> 0."""
        return -Psi_p * np.sin(0.5 * d * Psi_p + Ups_asc) * np.sinc(d * Psi_p / (2.0 * np.pi))

    cf_p = wb / 2.0 - m_dm                  # goes with k+1
    cf_m = wb / 2.0 + m_dm                  # goes with k-1 (resonant)
    dEPS1 = (A0 / wb) * (cf_p * _R_s(kk + 1.0) + cf_m * _R_s(kk - 1.0))
    dEPS2 = (A0 / wb) * (cf_p * _R_c(kk + 1.0) - cf_m * _R_c(kk - 1.0))

    # ---- delta Tasc: -2 alpha times the integral of Phi ----
    dTASC = -2 * A0 * (sin_mt - sin_mtasc) / m_dm

    # ---- PB channel: <delta x/x> and the extra -(1/2) alpha <Phi> ----
    mean_cos = _mean_cos(m_dm, t_arr, tasc_s, gamma_f, T_tilde)
    daae = -2.0 * A0 * (mean_cos - cos_mtasc)
    dPb_extrae = -0.5 * A0 * mean_cos

    # Full ELL1 delay: Delta_R = x[sin Phi + k/2 sin 2Phi - h/2 cos 2Phi - 3/2 h],
    # h = eta = EPS1. tempo and PINT drop the constant -3/2 x eta (Lange et al.
    # 2001, Eq. A6), so the EPS1 column of the design matrix is only
    # -x cos(2Phi)/2. The ULDM makes eta oscillate, and -3/2 x delta_eta(t) is a
    # delay no timing parameter absorbs, hence the -1.5 A1 term.
    return (partials["PB"] * PB_s * (1.5 * daae + dPb_extrae)
            + partials["A1"] * A1_s * daa
            + (partials["EPS1"] - 1.5 * A1_s) * dEPS1
            + partials["EPS2"] * dEPS2
            + partials["TASC"] * dTASC)


# ---- ENTERPRISE signal and PTA ----------------------------------------------------
def make_uldm_signal(t_arr, partials, orb_pars, m_dm):
    from enterprise.signals import parameter, signal_base, deterministic_signals

    @signal_base.function
    def uldm_delay(toas, log10_alpha=-45.0, rho_f=1.0, gamma_f=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)
        return (10.0 ** log10_alpha) * h

    # ranges only; the priors are imposed in PulsarULDM_RN.lnprior
    la = parameter.Uniform(LOG10_ALPHA_MIN, LOG10_ALPHA_MAX)("log10_alpha")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0 * np.pi)("gamma_f")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_alpha=la, rho_f=rf, gamma_f=gf), name="uldm")


def build_pta(psr, t_arr, partials, orb_pars, m_dm):
    return common.build_pta(psr, t_arr, make_uldm_signal(t_arr, partials, orb_pars, m_dm))


ULDM_SUFFIXES = ["log10_alpha", "rho_f", "gamma_f"]


def get_uldm_pnames(pta):
    return find_param_names(pta, ULDM_SUFFIXES)


def load_noise_MAP(pta, chain_file, pars_file):
    return common.load_noise_MAP(pta, chain_file, pars_file, get_uldm_pnames(pta))


# ---- sampled model ------------------------------------------------------------------
class PulsarULDM_RN(object):
    """Five-parameter model of one pulsar with marginalized red noise:
      x = (log10_alpha, log10_rho, gamma_f, log10_A_red, gamma_red)
    White noise and wideband DM fixed at the MAP values.
    """

    def __init__(self, d):
        self.d = d
        self.pta = d["pta"]
        self.fix = d["noise_fixed"]
        self.la, self.rf, self.gf = d["la"], d["rf"], d["gf"]
        self.arn, self.grn = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim = 5

    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.la] = x[0]
        p[self.rf] = 10.0 ** x[1]
        p[self.gf] = x[2]
        p[self.arn] = x[3]
        p[self.grn] = x[4]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_ALPHA_MIN <= x[0] <= LOG10_ALPHA_MAX):
            return -np.inf
        # uniform in alpha: p(a) da = p(alpha) dalpha -> ln p(a) = a ln10
        lp = x[0] * np.log(10.0) if COUPLING_PRIOR == "uniform" else 0.0
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho = 10.0 ** u
        lp += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh in u = log10 rho
        if not (0.0 <= x[2] <= 2.0 * np.pi):
            return -np.inf
        if not (LOG10_ARED_MIN <= x[3] <= LOG10_ARED_MAX):
            return -np.inf
        if not (GAMMA_RED_MIN <= x[4] <= GAMMA_RED_MAX):
            return -np.inf
        return lp

    def x0(self, rng):
        x = np.empty(self.ndim)
        x[0] = LOG10_ALPHA_INIT + rng.uniform(-2.0, 2.0)
        x[1] = 0.0
        x[2] = np.pi
        x[3] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + 0.1, LOG10_ARED_MAX - 0.1))
        x[4] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + 0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10A**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
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
        if missing:
            # a channel without a design-matrix column enters as zero
            print(f"  [{name}] [warn] no design-matrix column for {missing}: "
                  f"those channels enter as zero", flush=True)
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        la, rf, gf = get_uldm_pnames(pta)
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
                    la=la, rf=rf, gf=gf, arn=arn, grn=grn,
                    noise_fixed=noise_fixed, rn_map=rn_map,
                    ecc=orb_pars["ecc"], n_toa=len(t_arr), t_load=time.time() - t_load)
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
        alpha = run_one_chain(model, outDir, n_iter, rng)
        ess = effective_sample_size(alpha)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  ecc={d['ecc']:.1e}  "
              f"a=[{alpha.min():6.1f},{alpha.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True, alpha=alpha, ess=ess,
                    n_toa=d["n_toa"], ecc=d["ecc"], t_min=(time.time() - t0) / 60.0)
    except Exception as e:
        import traceback
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
