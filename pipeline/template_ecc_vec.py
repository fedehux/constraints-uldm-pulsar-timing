# -*- coding: utf-8 -*-
"""
template_ecc_vec.py
===================
Template ULDM VECTORIAL (spin-1) para orbitas EXCENTRICAS (rama DD del timing
model). Implementa el cuaderno `cuaderno_doctorado_vectorial_excentrico.tex`.

Esto llena el stub `compute_h_template` (rama ecc >= ECC_LOW_THRESHOLD) de
`Pipelines corregidos/pipeline_vectorial.py`, que hasta ahora devolvia ceros.

Convenciones (identicas al .tex):
    l      = wb (t - T0)          anomalia media, medida desde el periastro
    k      = m / wb
    Ups0   = Upsilon + m T0       fase estocastica corrida (se samplea directo)
    varphi                        acimut de n_hat DESDE EL PERIASTRO (directo)
    omega  = OM del par           argumento del periastro (conocido, no se samplea)
    sig    = sqrt(1 - e^2)
    u=cos E, w=sin E, q=r/a=1-e u, u'=-w/q, w'=u/q

Los cinco coeficientes de Bessel:
    p_n = 2 Jn'(ne)/n                       (de cos E - e)
    q_n = 2 Jn(ne)/(ne)                     (de sin E)
    a_n = 2 [Jn/(ne) - Jn']                 (de u' + w)
    b_n = 2 [(1-e^2) Jn - (e/n) Jn']        (de los dos corchetes de sig/e^2)
    c_n = 2 [Jn/(ne) - (1-e^2) Jn']         (de (1-e^2) u' + w)

Las primitivas se escriben ADIMENSIONALIZADAS (S_n = wb * calS_n, etc.) para
reutilizar los helpers estables con sinc del pipeline vectorial, que no tienen
rama en la resonancia n = k:
    S_n = wb calS_n = 0.5 [ rsin(n-k, l, -Ups0) - rsin(n+k, l, +Ups0) ]
    C_n = wb calC_n = 0.5 [ rcos(n-k, l, -Ups0) - rcos(n+k, l, +Ups0) ]
    C_0 = wb calC_0 = -rcos(k, l, Ups0)
La resonancia (n = k) cae en el primer termino y la maneja el sinc.

Canal PB: usa el PROMEDIO CORRIDO <da/a> sobre [T0, t] (no el instantaneo), que
es la correccion de la directora. Se obtiene cambiando rsin/rcos por sus
promedios avg_rsin/avg_rcos, que tambien son estables en la resonancia.
"""

import numpy as np
from scipy.special import jv, jvp

__all__ = ["deltas_ecc_vec", "compute_h_template_ecc", "N_TERMS_ECC"]

N_TERMS_ECC = 25          # armonicos del desarrollo de Bessel (e<=0.44 -> 1e-8)


# ---------------------------------------------------------------------------
# Helpers estables (mismos que pipeline_vectorial.py; replicados para que este
# modulo sea autocontenido y testeable sin enterprise).
# ---------------------------------------------------------------------------
def _rsin(delta, Psi, p):
    """[sin(delta*Psi + p) - sin(p)] / delta ; limite delta->0: Psi cos(p)."""
    return Psi * np.cos(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _rcos(delta, Psi, p):
    """[cos(delta*Psi + p) - cos(p)] / delta ; limite delta->0: -Psi sin(p)."""
    return -Psi * np.sin(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))


def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, estable en z->0 (serie para |z|<0.1)."""
    z = np.asarray(z, dtype=np.float64)
    chico = np.abs(z) < 0.1
    zs = np.where(chico, 1.0, z)
    directa = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    serie = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(chico, serie, directa)


def _avg_rsin(delta, Psi, p):
    """<_rsin(delta, Psi', p)> promediado en Psi' de 0 a Psi."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return Psi * (0.5 * np.cos(p) * s2 - np.sin(p) * _z_minus_sin_over_z2(z))


def _avg_rcos(delta, Psi, p):
    """<_rcos(delta, Psi', p)> promediado en Psi' de 0 a Psi."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return -Psi * (0.5 * np.sin(p) * s2 + np.cos(p) * _z_minus_sin_over_z2(z))


# ---------------------------------------------------------------------------
# Coeficientes de Bessel (cacheados por (e, N): no dependen de la masa ni de
# los angulos, asi que se calculan una sola vez por pulsar).
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
# Nucleo: las perturbaciones de los elementos orbitales
# ---------------------------------------------------------------------------
def deltas_ecc_vec(k, Ups0, vartheta, varphi, l, e, omega, iota, sin_i,
                   pref, prefT, wb, N=N_TERMS_ECC, promediar_pb=True):
    """
    Devuelve el dict de perturbaciones de los elementos orbitales.

    k     : m / wb                       l    : wb (t - T0), array
    pref  : F_unit / (a wb^2)            prefT: F_unit / (a wb^3)

    Salida: daa (delta a/a), daa_pb (la que ve el canal PB), dee (delta e),
            edw (e delta varpi_*), diota, dOmega, dxx (delta x/x),
            dOM (delta omega, rad), dT0 (segundos).
    """
    cf = bessel_coeffs(e, N)
    n = cf["n"][:, None]                       # (N,1)
    L = np.asarray(l, dtype=np.float64)[None, :]   # (1,ntoa)
    sig = np.sqrt(1.0 - e * e)
    e2 = e * e

    # --- primitivas adimensionales (N, ntoa); la resonancia vive en (n-k) ---
    S = 0.5 * (_rsin(n - k, L, -Ups0) - _rsin(n + k, L, Ups0))
    C = 0.5 * (_rcos(n - k, L, -Ups0) - _rcos(n + k, L, Ups0))
    C0 = -_rcos(k, np.asarray(l, dtype=np.float64), Ups0)     # sector n=0

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

    # --- delta a / a  (instantaneo: canal A1) ---
    daa = 4.0 * pref * svt * (cph * sumJnpS - (sig / e) * sph * sumJnC)

    # --- delta a / a promediado en [T0, t]  (canal PB) ---
    if promediar_pb:
        Sa = 0.5 * (_avg_rsin(n - k, L, -Ups0) - _avg_rsin(n + k, L, Ups0))
        Ca = 0.5 * (_avg_rcos(n - k, L, -Ups0) - _avg_rcos(n + k, L, Ups0))
        daa_pb = 4.0 * pref * svt * (cph * dot(cf["Jnp"], Sa)
                                     - (sig / e) * sph * dot(cf["Jn"], Ca))
    else:
        daa_pb = daa

    # --- delta e ---
    dee = -pref * svt * (((1.0 - e2) / e) * cph * sumaS
                         + (sig / e2) * sph * (1.5 * e2 * C0 + sumbC))

    # --- e delta varpi_*  (parte en el plano del periastro) ---
    edw = -pref * svt * ((sig / e2) * cph * (-1.5 * e2 * C0 + sumbC)
                         - (sph / e) * sumcS)

    # --- sectores fuera del plano ---
    Pi = -1.5 * e * C0 + sumpC
    X = sumqS
    diota = -(pref / sig) * cvt * (np.cos(omega) * Pi - sig * np.sin(omega) * X)
    dOmega = -(pref / (sig * sin_i)) * cvt * (np.sin(omega) * Pi
                                              + sig * np.cos(omega) * X)

    # --- canales observables ---
    dvarpi = edw / e
    dxx = daa + (np.cos(iota) / sin_i) * diota
    dOM = dvarpi - np.cos(iota) * dOmega
    dT0 = -2.0 * prefT * svt * (cph * Pi + sig * sph * X) + sig * dvarpi / wb

    return dict(daa=daa, daa_pb=daa_pb, dee=dee, edw=edw, dvarpi=dvarpi,
                diota=diota, dOmega=dOmega, dxx=dxx, dOM=dOM, dT0=dT0)


def compute_h_template_ecc(m_dm, rho_f, gamma_f, vartheta, varphi,
                           t_arr, partials, orb_pars,
                           F_unit_const, N=N_TERMS_ECC, promediar_pb=True):
    """
    Template DD (excentrico) vectorial, con acople unitario (g factorizado).

    `F_unit_const` = DELTA_C / M_NUCLEON * sqrt(2 rho_DM)  (sin el rho estocastico),
    se pasa desde el pipeline para no duplicar constantes.

    gamma_f  ES  Upsilon_0 = Upsilon + m T0   (sampleado directo; sumar la
             constante m*T0 a una U(0,2pi) da la misma U(0,2pi)).
    varphi   ES  el acimut medido DESDE EL PERIASTRO (idem, corrido por omega).
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
                       pref, prefT, wb, N=N, promediar_pb=promediar_pb)

    return (partials["PB"] * PB_s * (1.5 * d["daa_pb"])
            + partials["A1"] * A1_s * d["dxx"]
            + partials["ECC"] * d["dee"]
            + partials["OM"] * d["dOM"]
            + partials["T0"] * d["dT0"])
