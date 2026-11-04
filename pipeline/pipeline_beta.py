# =============================================================================
# Quadratic universal coupling of a scalar field:  M_A(Phi) = M_A (1 + beta Phi^2 / 2)
#
# Single-pulsar analysis. For each (mass, pulsar) a PTMCMC chain samples five
# parameters, (log10_beta, log10_rho, gamma_f, log10_A_red, gamma_red): the
# ULDM signal (coupling, field amplitude rho, phase Upsilon) and the achromatic
# red noise. White noise and the wideband DM model are fixed at their MAP
# values; the timing model and the red-noise Fourier coefficients are
# marginalized analytically.
#
# The coupling enters through Phi^2, which oscillates at twice the field
# frequency, so the resonances are at 2m = n w_b.
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
UNIT_CONVERSION = 2.30817405213802e+48   # beta_code -> beta_phys [GeV^-2]

# Prior range of log10 beta in code units (prior_ranges.py)
LOG10_BETA_MIN = float(os.environ.get("LOG10_BETA_MIN", PRIOR_RANGES["beta"]["LOG10_BETA_MIN"]))
LOG10_BETA_MAX = float(os.environ.get("LOG10_BETA_MAX", PRIOR_RANGES["beta"]["LOG10_BETA_MAX"]))

# "uniform": prior uniform in beta (upper limits); "log-uniform": uniform in
# log10(beta) (injection test). Only the beta term of lnprior changes.
COUPLING_PRIOR = os.environ.get("COUPLING_PRIOR", "uniform").strip().lower()
if COUPLING_PRIOR not in ("uniform", "log-uniform"):
    raise ValueError(f"COUPLING_PRIOR={COUPLING_PRIOR!r}: uniform | log-uniform")

LOG10_BETA_INIT = -78.0          # starting point, +- 2 dex
SIGMA_LOG10B = 3.0

# Histogram of log10 beta written by the worker: 0.005 dex bins.
N_BETA_BINS = int(round((LOG10_BETA_MAX - LOG10_BETA_MIN) / 0.005))
BETA_EDGES = np.linspace(LOG10_BETA_MIN, LOG10_BETA_MAX, N_BETA_BINS + 1)


# ---- ULDM template ------------------------------------------------------------
# The vanishing denominator is (n wb - 2m). Everything that carries it is
# written with sinc: no differences of large numbers, no division by something
# that goes to zero.

def _one_minus_cos_over_z2(z):
    """(1 - cos z)/z^2, exact and stable (1/2 at z = 0)."""
    # 1 - cos z = 2 sin^2(z/2)  =>  (1 - cos z)/z^2 = (1/2) sinc(z/2pi)^2
    s = np.sinc(z / (2.0 * np.pi))
    return 0.5 * s * s


def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, stable (0 at z = 0, grows as z/6).

    For small |z| the difference z - sin z loses all its significant digits,
    so the series z/6 - z^3/120 + z^5/5040 - z^7/362880 + z^9/39916800 is used
    there; at |z| = 0.1 it agrees with the direct formula to machine precision.
    """
    z = np.asarray(z, dtype=np.float64)
    small = np.abs(z) < 0.1
    z_safe = np.where(small, 1.0, z)
    direct = (z_safe - np.sin(z_safe)) / (z_safe * z_safe)
    z2 = z * z
    series = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(small, series, direct)


def _mean_cos2(m_dm, t_arr, t_ref, gamma_f, tt):
    """Time average of cos^2(m t + Y) between t_ref and t:
        (1/t~) int_{t_ref}^{t} cos^2(m t' + Y) dt'
            = 1/2 + (1/2) cos(m(t+t_ref) + 2Y) sinc(m t~/pi).
    It divides neither by t~ nor by m: at t = t_ref it equals cos^2(m t_ref + Y).
    """
    return 0.5 + 0.5 * np.cos(m_dm * (t_arr + t_ref) + 2.0 * gamma_f) * np.sinc(m_dm * tt / np.pi)


def _int_cos(omega, phase, t_arr, t_ref):
    """int_{t_ref}^{t} cos(omega t' + phase) dt', stable at omega = 0. With
    sin A - sin B = 2 cos((A+B)/2) sin((A-B)/2):
        int = t~ cos(omega (t+t_ref)/2 + phase) sinc(omega t~ / 2pi),  t~ = t - t_ref.
    At omega -> 0 it tends to t~ cos(phase), the secular limit. omega and phase
    can be columns (one row per harmonic n).
    """
    tt = (t_arr - t_ref)[None, :]
    mid = omega * (0.5 * (t_arr + t_ref))[None, :] + phase
    return tt * np.cos(mid) * np.sinc(omega * tt / (2.0 * np.pi))


def compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """Timing residual for beta = 1 (code units), as
    sum_param partial(r)/partial(param) * delta(param). Dispatches to the ELL1
    template at low eccentricity."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)

    # ---- eccentric (DD) template ----
    n_vec = np.arange(1, N_TERMS)
    n_col = n_vec[:, None]
    jn_col = jn(n_vec, n_vec * orb_pars["ecc"])[:, None]
    jnp_col = jvp(n_vec, n_vec * orb_pars["ecc"])[:, None]

    wb = orb_pars["omega_b"]
    ecc = orb_pars["ecc"]
    t0_s = orb_pars["T0_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    sqrt1me2 = np.sqrt(1.0 - ecc**2)

    T_tilde = t_arr - t0_s
    tt_row = T_tilde[None, :]
    nw = n_vec * wb
    nw_col = nw[:, None]
    cos_nw_tt = np.cos(nw_col * tt_row)
    sin_nw_tt = np.sin(nw_col * tt_row)

    nw_over_4 = n_vec * wb / 4.0
    delta_T0_an_base = 4.0 * jn_col - 2.0 * sqrt1me2 / ecc * n_col * jnp_col
    delta_T0_bn = (4.0 / n_col / wb * jn_col + 4.0 / ecc / wb * sqrt1me2 * jnp_col)

    two_m = 2.0 * m_dm
    op_col = (nw + two_m)[:, None]   # n wb + 2m: 'plus' denominator, never zero
    om_col = (nw - two_m)[:, None]   # n wb - 2m: 'minus' denominator, zero at 2m = n wb
    op_tt = op_col * tt_row
    om_tt = om_col * tt_row
    phi0_const_sq = 2.0 * DM_DENSITY / m_dm**2
    m_t = m_dm * t_arr
    m_t0 = m_dm * t0_s
    coeff_OM = (nw_over_4 - m_dm)[:, None]
    coeff_ECC_p = (m_dm - nw_over_4)[:, None]
    coeff_ECC_m = (m_dm + nw_over_4)[:, None]

    phase_C = 2.0 * m_t0 + 2.0 * gamma_f
    cos_C = np.cos(phase_C)
    sin_C = np.sin(phase_C)
    cos_plus = np.cos(op_tt + phase_C)
    sin_plus = np.sin(op_tt + phase_C)

    rp = (cos_C - cos_plus) / op_col

    # minus branch, exact for every om_col including om_col = 0:
    #   (cos_C - cos_minus)/om_col = T~ sin(mid_m) sinc(om_col T~ / 2pi),
    # which tends to -T~ sin_C at the resonance
    mid_m = 0.5 * om_tt - phase_C
    sincv_m = np.sinc(om_col * tt_row / (2.0 * np.pi))
    rm = tt_row * np.sin(mid_m) * sincv_m

    phi_sq = phi0_const_sq * rho_f**2

    term1 = (1.0 - cos_nw_tt) / (2.0 * nw_col) + 0.25 * (rp + rm)
    part1 = -2.0 * wb * phi_sq * np.sum(n_col * jn_col * term1, axis=0)
    cos_mt = np.cos(m_t + gamma_f)
    cos_mt0 = np.cos(m_t0 + gamma_f)
    sin_2mt = np.sin(2.0 * m_t + 2.0 * gamma_f)
    part2 = -phi_sq * (cos_mt**2 - cos_mt0**2)
    part3 = -2.0 * m_dm * phi_sq * np.sum(jn_col * (-rp + rm), axis=0)
    daa = part1 + part2 + part3        # instantaneous delta a/a (A1 channel)

    # ---- time average <delta a / a> over [T0, t] (Eq. C.1b), PB channel ----
    # PB is inferred from the orbital phase accumulated since T0, so it responds
    # to the running average of delta a/a; A1 sees the instantaneous value.
    # Each piece of daa is replaced by its integral from T0 to t, divided by T~
    # at the end. The harmonic piece and rp integrate directly. The integral of
    # rm is written in closed form,
    #   int_{T0}^{t} rm dt' = T~^2 [ cos_C (z - sin z)/z^2 - sin_C (1 - cos z)/z^2 ],
    # z = om_col T~, with both ratios regular at z = 0 (0 and 1/2), instead of
    # cos_C T~/om - (sin_minus + sin_C)/om^2, which diverges term by term.
    harm_int = tt_row / (2.0 * nw_col) - np.sin(nw_col * tt_row) / (2.0 * nw_col**2)
    rp_int = cos_C * tt_row / op_col - (sin_plus - sin_C) / op_col**2
    z_res = om_tt
    rm_int = tt_row**2 * (cos_C * _z_minus_sin_over_z2(z_res)
                          - sin_C * _one_minus_cos_over_z2(z_res))

    part1_int = -2.0 * wb * phi_sq * np.sum(n_col * jn_col * (harm_int + 0.25 * (rp_int + rm_int)), axis=0)
    part2_int = -phi_sq * ((0.5 - cos_mt0**2) * T_tilde + (sin_2mt - sin_C) / (4.0 * m_dm))
    part3_int = -2.0 * m_dm * phi_sq * np.sum(jn_col * (-rp_int + rm_int), axis=0)

    # every integral vanishes as T~ at t = T0, so the limit there is daa(T0) = 0
    int_total = part1_int + part2_int + part3_int
    T_nonzero = np.where(np.abs(T_tilde) > 0.0, T_tilde, 1.0)
    daae = np.where(np.abs(T_tilde) > 0.0, int_total / T_nonzero, 0.0)

    # ---- delta omega ----
    # Two corrections to Eq. (B4) of Huxhagen & Lopez Nacir 2026 (PRD 114,
    # 043056), checked against direct quadrature of its Eq. (22d) (agreement
    # 1e-8 to 7e-6 for m = 5e-23 to 1e-21 eV and e = 0.03 to 0.44):
    # (i) Non-resonant term. The DC part of Phi^2 multiplies
    #     (a^2/r^2) cos(theta) = sum_n 2 n J_n'(ne) cos(n wb t~)   [Eq. A2g],
    #     whose integral is sin(n wb t~)/(n wb); the result is sin(n wb t~)/2.
    # (ii) Resonant terms. The orbital factor goes as cos(n wb t~), so the
    #     product with cos(2mt+2Y) gives cosines, which integrate to sines:
    #        S^+_n = [sin((n wb + 2m) t~ + Y') - sin Y'] / (n wb + 2m)
    #        S^-_n = [sin((n wb - 2m) t~ - Y') + sin Y'] / (n wb - 2m)
    #     with coefficients (n wb/4 - m) for the plus branch and (n wb/4 + m)
    #     for the minus branch, as in delta a/a (Eq. B2).
    # S^-_n resonates at 2m = n wb and is written with sinc, like rm:
    #   [sin(z - Y') + sin Y'] / om = T~ cos(z/2 - Y') sinc(z/2pi),  z = om T~.
    sp_int = (sin_plus - sin_C) / op_col
    sm_int = tt_row * np.cos(0.5 * om_tt - phase_C) * sincv_m
    partA_OM = 0.5 * np.sum(jnp_col * sin_nw_tt, axis=0)
    partB_OM = np.sum(jnp_col * (coeff_OM * sp_int + coeff_ECC_m * sm_int), axis=0)
    dOM = phi_sq * sqrt1me2 / ecc * (partA_OM + partB_OM)

    # ---- delta e ----
    partA_E = -0.5 * np.sum(jn_col * (1.0 - cos_nw_tt), axis=0)
    partB_E = np.sum(jn_col * (coeff_ECC_p * rp - coeff_ECC_m * rm), axis=0)
    dECC = phi_sq * (1.0 - ecc**2) / ecc * (partA_E + partB_E)

    # ---- delta T0 ----
    term_b = phi_sq / (4.0 * m_dm) * (sin_2mt - sin_C)

    # term_c in closed form. Per harmonic n the integrand is
    #   A_n phi^2 cos^2(mt+Y) cos(nw t~) + 2 B_n (-m phi^2 cos(mt+Y) sin(mt+Y)) sin(nw t~),
    # with A_n = delta_T0_an_base, B_n = delta_T0_bn. With
    #   cos^2 = 1/2 + 1/2 cos(2mt+2Y),  cos sin = 1/2 sin(2mt+2Y),
    # and product to sum, it is a sum of three pure cosines (psi = nw t~ and
    # f2 +- psi), integrated with _int_cos. The lower limit T0 is the one of
    # term_b and term_d.
    t_ref_c = t0_s
    two_g = 2.0 * gamma_f
    I_psi = _int_cos(nw_col, -nw_col * t0_s, t_arr, t_ref_c)
    I_plus = _int_cos(op_col, two_g - nw_col * t0_s, t_arr, t_ref_c)
    I_minus = _int_cos(-om_col, two_g + nw_col * t0_s, t_arr, t_ref_c)
    c_psi = 0.5 * delta_T0_an_base
    c_plus = 0.25 * delta_T0_an_base + 0.5 * m_dm * delta_T0_bn
    c_minus = 0.25 * delta_T0_an_base - 0.5 * m_dm * delta_T0_bn
    term_c = 0.5 * phi_sq * np.sum(c_psi * I_psi + c_plus * I_plus + c_minus * I_minus, axis=0)
    term_d = 0.5 * phi_sq * T_tilde

    # Term (1 - sqrt(1-e^2)) varpi_dot of Eq. (22e) of the same paper, not
    # reproduced by the coefficients A_n, B_n of its Eq. (26). Its integral is
    # (1 - sqrt(1-e^2)) delta varpi, and it enters delta T0 divided by wb. It is
    # O(e): up to 1.8% of the accumulated phase at e = 0.4367 (J1903+0327).
    # With it, -wb dT0 + dOM matches direct quadrature of (22e) to 1e-9.
    term_e = (1.0 - sqrt1me2) * dOM / wb

    dT0 = -term_b - term_c - term_d - term_e

    # Extra term of delta Pb besides the semi-major axis, also time-averaged:
    # -(1/4) <Phi^2>
    dPb_extrae = -0.25 * phi_sq * _mean_cos2(m_dm, t_arr, t0_s, gamma_f, T_tilde)

    return (partials["PB"] * PB_s * (1.5 * daae + dPb_extrae)
            + partials["A1"] * A1_s * daa
            + partials["ECC"] * dECC
            + partials["OM"] * dOM
            + partials["T0"] * dT0)


def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """ELL1 (low-eccentricity) template, quadratic coupling."""
    wb = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]

    T_tilde = t_arr - tasc_s
    m_t = m_dm * t_arr
    m_tasc = m_dm * tasc_s
    P = (2.0 * DM_DENSITY / m_dm**2) * rho_f**2          # Phi0^2 rho^2

    cos_mt = np.cos(m_t + gamma_f)
    cos_mtasc = np.cos(m_tasc + gamma_f)
    sin_2mt = np.sin(2.0 * m_t + 2.0 * gamma_f)
    sin_2mtasc = np.sin(2.0 * m_tasc + 2.0 * gamma_f)

    # ---- delta x / x (A1), instantaneous ----
    daa = -P * (cos_mt**2 - cos_mtasc**2)

    # ---- delta eta (EPS1) and delta kappa (EPS2) ----
    k = m_dm / wb
    Psi = wb * T_tilde                       # Psi'
    Y2 = 2.0 * (m_tasc + gamma_f)            # 2 Upsilon_asc
    a1 = 2.0 * k - 1.0
    a2 = 2.0 * k + 1.0

    def _S(a):
        return Psi * np.cos(0.5 * a * Psi + Y2) * np.sinc(0.5 * a * Psi / np.pi)

    def _C(a):
        return Psi * np.sin(0.5 * a * Psi + Y2) * np.sinc(0.5 * a * Psi / np.pi)

    dEPS1 = (P / 4.0 * np.sin(Psi)
             + (P / 2.0) * (k + 0.25) * _S(a1)
             + (P / 2.0) * (0.25 - k) * _S(a2))
    dEPS2 = (P / 4.0 * (np.cos(Psi) - 1.0)
             + (P / 2.0) * (k + 0.25) * _C(a1)
             - (P / 2.0) * (0.25 - k) * _C(a2))

    # ---- delta Tasc ----
    two_m = 2.0 * m_dm
    dTASC = -(1 / 2) * P * (T_tilde + (sin_2mt - sin_2mtasc) / two_m)

    # ---- PB channel: time-averaged delta x/x and extra term ----
    # The eccentric average (Eq. C.1b) with the single harmonic n = 1 and
    # T0 -> Tasc; <cos^2(mt+Y)> from _mean_cos2.
    mean_cos2 = _mean_cos2(m_dm, t_arr, tasc_s, gamma_f, T_tilde)
    daae = -P * (mean_cos2 - cos_mtasc**2)      # <delta x / x>
    dPb_extrae = -0.25 * P * mean_cos2          # -(1/4) <Phi^2>

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
    def uldm_delay(toas, log10_beta=-75.0, rho_f=1.0, gamma_f=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)
        return (10.0 ** log10_beta) * h

    # ranges only; the priors are imposed in PulsarULDM_RN.lnprior
    lb = parameter.Uniform(LOG10_BETA_MIN, LOG10_BETA_MAX)("log10_beta")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0 * np.pi)("gamma_f")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_beta=lb, rho_f=rf, gamma_f=gf), name="uldm")


def build_pta(psr, t_arr, partials, orb_pars, m_dm):
    return common.build_pta(psr, t_arr, make_uldm_signal(t_arr, partials, orb_pars, m_dm))


ULDM_SUFFIXES = ["log10_beta", "rho_f", "gamma_f"]


def get_uldm_pnames(pta):
    return find_param_names(pta, ULDM_SUFFIXES)


def load_noise_MAP(pta, chain_file, pars_file):
    return common.load_noise_MAP(pta, chain_file, pars_file, get_uldm_pnames(pta))


# ---- sampled model ------------------------------------------------------------------
class PulsarULDM_RN(object):
    """Five-parameter model of one pulsar with marginalized red noise:
      x = (log10_beta, log10_rho, gamma_f, log10_A_red, gamma_red)
    White noise and wideband DM fixed at the MAP values.
    """

    def __init__(self, d):
        self.d = d
        self.pta = d["pta"]
        self.fix = d["noise_fixed"]
        self.lb, self.rf, self.gf = d["lb"], d["rf"], d["gf"]
        self.arn, self.grn = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim = 5

    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.lb] = x[0]
        p[self.rf] = 10.0 ** x[1]
        p[self.gf] = x[2]
        p[self.arn] = x[3]
        p[self.grn] = x[4]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_BETA_MIN <= x[0] <= LOG10_BETA_MAX):
            return -np.inf
        # uniform in beta: p(b) db = p(beta) dbeta -> ln p(b) = b ln10
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
        x[0] = LOG10_BETA_INIT + rng.uniform(-2.0, 2.0)
        x[1] = 0.0
        x[2] = np.pi
        x[3] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + 0.1, LOG10_ARED_MAX - 0.1))
        x[4] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + 0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10B**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
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
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        lb, rf, gf = get_uldm_pnames(pta)
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
                    lb=lb, rf=rf, gf=gf, arn=arn, grn=grn,
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
        beta = run_one_chain(model, outDir, n_iter, rng)
        ess = effective_sample_size(beta)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  ecc={d['ecc']:.1e}  "
              f"b=[{beta.min():6.1f},{beta.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True, beta=beta, ess=ess,
                    n_toa=d["n_toa"], ecc=d["ecc"], t_min=(time.time() - t0) / 60.0)
    except Exception as e:
        import traceback
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
