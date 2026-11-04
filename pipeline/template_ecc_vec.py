"""Vector (spin-1) ULDM template for eccentric orbits (DD branch of the timing
model; Appendix B of the paper). Used by pipeline_vec.py for the binaries with
e >= ECC_LOW_THRESHOLD.

Conventions:
    l      = wb (t - T0)          mean anomaly, measured from the periastron
    k      = m / wb
    Ups0   = Upsilon + m T0       shifted stochastic phase (sampled directly)
    varphi                        azimuth of n_hat measured from the periastron (sampled directly)
    omega  = OM of the .par file  argument of periastron (known, not sampled)
    sig    = sqrt(1 - e^2)
    u = cos E, w = sin E, q = r/a = 1 - e u, u' = -w/q, w' = u/q

The five Bessel coefficients:
    p_n = 2 Jn'(ne)/n                       (from cos E - e)
    q_n = 2 Jn(ne)/(ne)                     (from sin E)
    a_n = 2 [Jn/(ne) - Jn']                 (from u' + w)
    b_n = 2 [(1-e^2) Jn - (e/n) Jn']        (from the two brackets of sig/e^2)
    c_n = 2 [Jn/(ne) - (1-e^2) Jn']         (from (1-e^2) u' + w)

The primitives are written dimensionless (S_n = wb * calS_n, etc.) so that the
sinc-stable helpers apply, with no special branch at the resonance n = k:
    S_n = wb calS_n = 0.5 [ rsin(n-k, l, -Ups0) - rsin(n+k, l, +Ups0) ]
    C_n = wb calC_n = 0.5 [ rcos(n-k, l, -Ups0) - rcos(n+k, l, +Ups0) ]
    C_0 = wb calC_0 = -rcos(k, l, Ups0)
The resonance (n = k) falls in the first term and is handled by the sinc.

PB channel: the running average <delta a/a> over [T0, t] instead of the
instantaneous value, obtained by replacing rsin/rcos with avg_rsin/avg_rcos,
which are also stable at the resonance.
"""

import numpy as np
from scipy.special import jv, jvp

__all__ = ["deltas_ecc_vec", "compute_h_template_ecc", "N_TERMS_ECC"]

N_TERMS_ECC = 25          # Bessel harmonics (1e-8 accuracy for e <= 0.44)


# ---------------------------------------------------------------------------
# Stable helpers (as in pipeline_vec.py, repeated so that this module can be
# tested without ENTERPRISE)
# ---------------------------------------------------------------------------
def _rsin(delta, Psi, p):
    """[sin(delta*Psi + p) - sin(p)] / delta ; limit delta -> 0: Psi cos(p)."""
    return Psi * np.cos(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _rcos(delta, Psi, p):
    """[cos(delta*Psi + p) - cos(p)] / delta ; limit delta -> 0: -Psi sin(p)."""
    return -Psi * np.sin(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, stable at z -> 0 (series for |z| < 0.1)."""
    z = np.asarray(z, dtype=np.float64)
    small = np.abs(z) < 0.1
    zs = np.where(small, 1.0, z)
    direct = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    series = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(small, series, direct)


def _avg_rsin(delta, Psi, p):
    """<_rsin(delta, Psi', p)> averaged over Psi' from 0 to Psi."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return Psi * (0.5 * np.cos(p) * s2 - np.sin(p) * _z_minus_sin_over_z2(z))


def _avg_rcos(delta, Psi, p):
    """<_rcos(delta, Psi', p)> averaged over Psi' from 0 to Psi."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return -Psi * (0.5 * np.sin(p) * s2 + np.cos(p) * _z_minus_sin_over_z2(z))


# ---------------------------------------------------------------------------
# Bessel coefficients, cached by (e, N): they depend neither on the mass nor on
# the angles, so they are computed once per pulsar.
# ---------------------------------------------------------------------------
_COEF_CACHE = {}


def bessel_coeffs(e, N=N_TERMS_ECC):
    key = (round(float(e), 12), int(N))
    if key not in _COEF_CACHE:
        n = np.arange(1, N + 1, dtype=np.float64)
        Jn, Jnp = jv(n, n * e), jvp(n, n * e, 1)
        _COEF_CACHE[key] = dict(
            n=n, Jn=Jn, Jnp=Jnp,
            p=2.0 * Jnp / n,
            q=2.0 * Jn / (n * e),
            a=2.0 * (Jn / (n * e) - Jnp),
            b=2.0 * ((1.0 - e * e) * Jn - (e / n) * Jnp),
            c=2.0 * (Jn / (n * e) - (1.0 - e * e) * Jnp))
    return _COEF_CACHE[key]


# ---------------------------------------------------------------------------
# Perturbations of the orbital elements
# ---------------------------------------------------------------------------
def deltas_ecc_vec(k, Ups0, vartheta, varphi, l, e, omega, iota, sin_i,
                   pref, prefT, wb, N=N_TERMS_ECC, average_pb=True):
    """
    Perturbations of the orbital elements.

    k     : m / wb                       l    : wb (t - T0), array
    pref  : F_unit / (a wb^2)            prefT: F_unit / (a wb^3)

    Returns a dict with daa (delta a/a), daa_pb (the one the PB channel sees),
    dee (delta e), edw (e delta varpi_*), diota, dOmega, dxx (delta x/x),
    dOM (delta omega, rad) and dT0 (seconds).
    """
    cf = bessel_coeffs(e, N)
    n = cf["n"][:, None]                             # (N,1)
    L = np.asarray(l, dtype=np.float64)[None, :]     # (1,ntoa)
    sig = np.sqrt(1.0 - e * e)
    e2 = e * e

    # dimensionless primitives (N, ntoa); the resonance lives in (n-k)
    S = 0.5 * (_rsin(n - k, L, -Ups0) - _rsin(n + k, L, Ups0))
    C = 0.5 * (_rcos(n - k, L, -Ups0) - _rcos(n + k, L, Ups0))
    C0 = -_rcos(k, np.asarray(l, dtype=np.float64), Ups0)     # n = 0 sector

    cph, sph = np.cos(varphi), np.sin(varphi)
    svt, cvt = np.sin(vartheta), np.cos(vartheta)

    def dot(coef, M):
        return coef[:, None].__mul__(M).sum(axis=0)

    sumJnpS = dot(cf["Jnp"], S)
    sumJnC = dot(cf["Jn"], C)
    sumaS = dot(cf["a"], S)
    sumbC = dot(cf["b"], C)
    sumcS = dot(cf["c"], S)
    sumpC = dot(cf["p"], C)
    sumqS = dot(cf["q"], S)

    # delta a / a, instantaneous (A1 channel)
    daa = 4.0 * pref * svt * (cph * sumJnpS - (sig / e) * sph * sumJnC)

    # delta a / a averaged over [T0, t] (PB channel)
    if average_pb:
        Sa = 0.5 * (_avg_rsin(n - k, L, -Ups0) - _avg_rsin(n + k, L, Ups0))
        Ca = 0.5 * (_avg_rcos(n - k, L, -Ups0) - _avg_rcos(n + k, L, Ups0))
        daa_pb = 4.0 * pref * svt * (cph * dot(cf["Jnp"], Sa)
                                     - (sig / e) * sph * dot(cf["Jn"], Ca))
    else:
        daa_pb = daa

    # delta e
    dee = -pref * svt * (((1.0 - e2) / e) * cph * sumaS
                         + (sig / e2) * sph * (1.5 * e2 * C0 + sumbC))

    # e delta varpi_* (in-plane part)
    edw = -pref * svt * ((sig / e2) * cph * (-1.5 * e2 * C0 + sumbC)
                         - (sph / e) * sumcS)

    # out-of-plane sectors
    Pi = -1.5 * e * C0 + sumpC
    X = sumqS
    diota = -(pref / sig) * cvt * (np.cos(omega) * Pi - sig * np.sin(omega) * X)
    dOmega = -(pref / (sig * sin_i)) * cvt * (np.sin(omega) * Pi
                                              + sig * np.cos(omega) * X)

    # observable channels
    dvarpi = edw / e
    dxx = daa + (np.cos(iota) / sin_i) * diota
    dOM = dvarpi - np.cos(iota) * dOmega
    dT0 = -2.0 * prefT * svt * (cph * Pi + sig * sph * X) + sig * dvarpi / wb

    return dict(daa=daa, daa_pb=daa_pb, dee=dee, edw=edw, dvarpi=dvarpi,
                diota=diota, dOmega=dOmega, dxx=dxx, dOM=dOM, dT0=dT0)


def compute_h_template_ecc(m_dm, rho_f, gamma_f, vartheta, varphi,
                           t_arr, partials, orb_pars,
                           F_unit_const, N=N_TERMS_ECC, average_pb=True):
    """
    Eccentric (DD) vector template for g = 1.

    F_unit_const = DELTA_C / M_NUCLEON * sqrt(2 rho_DM) (without the stochastic
    amplitude rho), passed from the pipeline.

    gamma_f  is Upsilon_0 = Upsilon + m T0 (sampled directly; shifting a
             U(0, 2pi) variable by the constant m T0 leaves it U(0, 2pi)).
    varphi   is the azimuth measured from the periastron (likewise shifted by omega).
    """
    wb = orb_pars["omega_b"]
    e = orb_pars["ecc"]
    t0_s = orb_pars["T0_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    iota = orb_pars["iota"]
    sin_i = orb_pars["sin_i"]
    omega = orb_pars["OM_rad"]
    a_semi = orb_pars["a_semi"]

    F_unit = F_unit_const * rho_f
    pref = F_unit / (a_semi * wb ** 2)
    prefT = F_unit / (a_semi * wb ** 3)

    l = wb * (t_arr - t0_s)
    k = m_dm / wb

    d = deltas_ecc_vec(k, gamma_f, vartheta, varphi, l, e, omega, iota, sin_i,
                       pref, prefT, wb, N=N, average_pb=average_pb)

    return (partials["PB"] * PB_s * (1.5 * d["daa_pb"])
            + partials["A1"] * A1_s * d["dxx"]
            + partials["ECC"] * d["dee"]
            + partials["OM"] * d["dOM"]
            + partials["T0"] * d["dT0"])
