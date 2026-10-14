# =============================================================================
# ACOPLAMIENTO LINEAL (alpha):  M_A(Phi) = M_A (1 + alpha Phi)
# Kus, Lopez Nacir & Urban (2024), arXiv:2402.04099, Ec. 2.14.
#
# Cota conjunta de alpha combinando cadenas MCMC por-pulsar con ruido rojo
# marginalizado, sobre 198 masas entre 1e-23 y ~1e-18 eV. Por cada (masa,
# pulsar) samplea con PTMCMC 5 parametros -- log10_alpha, log10_rho, gamma_f,
# log10_A_red, gamma_red -- y combina:  ln p(a|D) = a ln10 + sum_i ln Z_i(a).
#
# Templates: Ec. C.1b/c/d (BT/DD excentrico) y C.5b/c/d (ELL1 circular), mas
# delta T0 / delta Tasc y el extra de delta Pb, que no estan en el paper y
# salen de traducir el cuadratico con la regla beta Phi^2/2 -> alpha Phi.
# =============================================================================

# %% [PIN] 1 HILO POR PROCESO, PARALELIZADO DENTRO DE LA COMPU
import astropy.units as u
from scipy.ndimage import gaussian_filter1d
from scipy.special import jn, jvp
import matplotlib.pyplot as plt
import matplotlib
from copy import deepcopy
import numpy as np
import time
import glob
import warnings
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# %% [QUIET] SILENCIAR LOGS
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

# %% [0] IMPORTS
matplotlib.use("Agg")

try:                                          # scipy>=1.6
    from scipy.integrate import cumulative_trapezoid
except ImportError:                           # scipy<1.6
    from scipy.integrate import cumtrapz as cumulative_trapezoid

print("[0] IMPORTS OK")

# =============================================================================
# CONFIGURACION  (todo lo que tipicamente se toca esta aca arriba)
# =============================================================================
# ---- Pulsares ----
ALLOWED_MODELS = ("ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDK")
# BT y DDK son keplerianos pero de baja ecc -> load_pulsar los pasa a ELL1.

N_PULSARS_TARGET = 999          # 999 = usar TODOS los que matcheen ALLOWED_MODELS

# ---- Barrido en masa ----
# [v3] El TECHO baja de ~1.02e-18 a 1e-19 eV: arriba de ahi la cota semianalitica
# (Comparacion NB vs WB/techo_prior.py) ya vale -41.31 y sigue subiendo, o sea
# que el dato deja de restringir y la curva solo dibuja el prior. Con 4 decadas
# y el MISMO paso de antes (~0.0254 dex = 6.0 %) quedan 158 masas de envolvente.
#
# OJO: esta es la grilla BASE. La grilla de PRODUCCION (base + zoom resonante de
# los excentricos) la arma grid_masas_alpha.construir_grid(); el worker y el
# aggregate usan ESA, no MASS_GRID_EV. Aca se deja la base para que
# pipeline_alpha.main() (modo standalone, sin cluster) siga funcionando.
MASS_MIN_EV = float(os.environ.get("MASA_MIN_EV", "1e-23"))
MASS_MAX_EV = float(os.environ.get("MASA_MAX_EV", "1e-19"))
N_MASS = int(os.environ.get("N_BASE", "158"))       # paso = 4/157 = 0.02548 dex
SINGLE_MASS_EV = None           # si != None (p.ej. 1e-23) corre SOLO esa masa


# ---- MCMC ----
N_ITER = 200000
MAX_WORKERS = None              # None -> cpu_count()
SEED = 60

# %% [1] PATHS Y CONSTANTES FISICAS
#   NANOGRAV_ROOT -> carpeta con 'wideband/{par,tim,noise}';  OUT_ROOT -> salidas
# Las dos se pueden fijar por variable de entorno (DIRAC/cluster). Si no estan,
# se usan las rutas locales de este directorio, igual que en pipeline_vectorial.py
# y en pipeline_perpulsar_combine_RNmarg_DEFINITIVO.py.

NANOGRAV_ROOT = os.environ.get(
    "NANOGRAV_ROOT", "C:/Users/fedeh/OneDrive/Documents/C/NANOGrav15yr_PulsarTiming_v2.1.0")
WB_ROOT = os.path.join(NANOGRAV_ROOT, "wideband")
OUT_ROOT = os.environ.get(
    "OUT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "salida_alpha_full"))
CHAIN_ROOT = os.path.join(OUT_ROOT, "chains_RNmarg")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(CHAIN_ROOT, exist_ok=True)

# Constantes numericas

EV_TO_INV_S = 1.51926760385984e+15
YR_IN_S = 3.15576e7
DAY_IN_S = 86400.0
DM_DENSITY = 0.3 * 4.09351499910375e+55
# convierte beta_code -> beta_fisico [GeV^-2]
UNIT_CONVERSION = 2.30817405213802e+48

# UNIT_CONVERSION es (Phi0_codigo/Phi0_fisico)^2; como la senal lineal lleva una
# sola potencia de Phi, el factor que va con alpha es su raiz:
#   alpha_fisico [GeV^-1] = (10^a) * ALPHA_CONVERSION
ALPHA_CONVERSION = np.sqrt(UNIT_CONVERSION)  # ~ 1.5192676e24

N_TERMS = 20  # Terminos del desarrollo de Bessel
N_FREQ_RN = 30  # Numero de modos del red noise

# [v3] PRIOR ANCHO. El barrido de alpha2 corria con techo -42.18 y SATURABA
# desde m = 3.6966e-21 eV: 88 de las 198 masas no eran cota sino prior (la
# posterior queda plana y el UL95 cae en 0.95*alpha_max por construccion).
# No era bug del aggregate: verificado con EXTRAPOLAR_COLA=0, y reproducido
# semianaliticamente con la Fisher conjunta (Comparacion NB vs WB/techo_prior.py):
#
#     m [eV]     cota 1.96/sqrt(sum Q_i)     margen al techo -42.18
#     1e-21           -45.20                        3.02
#     3.7e-21         -42.85                        0.67
#     1e-20           -42.53                        0.35
#     1e-19           -41.31                       -0.87   <- ya saturado
#
# [v3] El prior se declara en unidades FISICAS y las de codigo se derivan:
#   alpha_fisico [GeV^-1] = 10^a * ALPHA_CONVERSION   (ALPHA_CONVERSION=1.5193e24)
# El de alpha2 era, en fisicas, [1e-24, 1e-18] GeV^-1 (de ahi los -48.18/-42.18).
# El de v3 es [1e-24, 1e-14] GeV^-1: DIEZ decadas, 4 mas de techo que alpha2.
# En unidades de codigo eso da [-48.1816, -38.1816].
#
# Con ese techo la masa mas alta del barrido (1e-19 eV, la peor) queda 3.13 dex
# por debajo del prior, o sea que TODA la curva es cota y ninguna masa es prior.
# CUIDADO: mover el techo mueve ALPHA_EDGES -> los histogramas de v3 NO son
# compatibles con los de alpha2. Carpeta aparte (~/alpha3), no mezclar.
LOG10_ALPHA_PHYS_MIN = float(os.environ.get("LOG10_ALPHA_PHYS_MIN", "-24.0"))
LOG10_ALPHA_PHYS_MAX = float(os.environ.get("LOG10_ALPHA_PHYS_MAX", "-14.0"))
_LOG10_CONV = np.log10(ALPHA_CONVERSION)      # 24.18156
LOG10_ALPHA_MIN = float(os.environ.get(
    "LOG10_ALPHA_MIN", str(LOG10_ALPHA_PHYS_MIN - _LOG10_CONV)))
LOG10_ALPHA_MAX = float(os.environ.get(
    "LOG10_ALPHA_MAX", str(LOG10_ALPHA_PHYS_MAX - _LOG10_CONV)))

# [BF] Prior del acople. "uniforme" (default) = uniforme en alpha, el de las
# cotas. "loguniforme" = uniforme en log10(alpha), el de la busqueda de senal
# (Bayes factor). Solo cambia el termino de alpha en lnprior.
PRIOR_ACOPLE = os.environ.get("PRIOR_ACOPLE", "uniforme").strip().lower()
if PRIOR_ACOPLE not in ("uniforme", "loguniforme"):
    raise ValueError(f"PRIOR_ACOPLE={PRIOR_ACOPLE!r}: uniforme|loguniforme")

RHO_F_MIN = 1.0e-3
RHO_F_MAX = 6.0         # rango de validez de rho en enterprise; el prior que
# pesa es el Rayleigh de PulsarULDM_RN.lnprior (al pta solo se le pide la likelihood)
LOG10_RHO_MIN = np.log10(RHO_F_MIN)
LOG10_RHO_MAX = np.log10(RHO_F_MAX)
ECC_LOW_THRESHOLD = 0.01
# valor de excentricidad a partir del cual se considera que un sistema es circular

# La resonancia m = n*wb no necesita rama aparte: el sinc de _stable_minus ya es
# estable ahi. El barrido fino de los picos vive en barrido_resonante_alpha.py.

# ---- Integrales del template: TODAS analiticas (sin cumtrapz) ----------------
# El <da/a> del canal PB (correccion de la directora) y el term_harm del canal T0
# se calculan ahora en FORMA CERRADA, estables en la resonancia m = n*wb (el sinc
# / las series (z-sin z)/z^2, (1-cos z)/z^2 dan el limite secular sin 0/0 ni ifs).
# Esto reemplaza (a) el daae 0/0 original de la directora, que en el pico exacto
# daba ~2650% de error, y (b) el cumtrapz del term_harm, que erraba ~15x en
# resonancia (el integrando oscila a ~m, mas rapido que el muestreo de TOAs).
# RESONANCIA_ESTABLE quedo OBSOLETO (el analitico es siempre); se deja definido
# solo para no romper barrido_resonante_alpha.py, que lo setea. Es un no-op.
RESONANCIA_ESTABLE = True

# ---- MCMC (priors / arranque) ----
BURN_FRAC = 0.30  # quema el 30% inicial de la cadena
# centro de la ventana [-48.18,-42.18]; arranco en [-47,-43]
LOG10_ALPHA_INIT = -45.0
COV_UPDATE = 2000  # actualiza la covarianza cada 2000 pasos
# parametros optimos para la cadena segun testeamos antes, por prueba y error
SCAM_WEIGHT = 60
AM_WEIGHT = 20
DE_WEIGHT = 20
SIGMA_LOG10A = 3.0
SIGMA_LOG10RHO = 0.5
SIGMA_GAMMA = 2 * np.pi

# ---- Red noise (marginalizado): priors y escalas ----
# Usamos lo mismo que en los papers de constraints de PTA

LOG10_ARED_MIN, LOG10_ARED_MAX = -20.0, -11.0
GAMMA_RED_MIN,  GAMMA_RED_MAX = 0.0, 10.0
SIGMA_LOG10_ARED = 1.5
SIGMA_GAMMA_RED = 1.5

# ---- Binning para Z_i(a) y combinacion ----
# Z_i se estima del histograma suavizado de la cadena, no de los puntos crudos.

# [v3] El prior paso de 6 a 10 decadas. Los bins se escalan para MANTENER la
# resolucion de 0.005 dex/bin (era 1200 bins / 6 dex); si se dejaban 1200 el bin
# pasaba a 0.0083 dex y el suavizado de SMOOTH_SIGMA_BINS quedaba 1.7x mas ancho
# en dex. 10/0.005 = 2000.
N_ALPHA_BINS = int(os.environ.get(
    "N_ALPHA_BINS",
    str(int(round((LOG10_ALPHA_MAX - LOG10_ALPHA_MIN) / 0.005)))))
ALPHA_EDGES = np.linspace(LOG10_ALPHA_MIN, LOG10_ALPHA_MAX, N_ALPHA_BINS + 1)
ALPHA_CENTERS = 0.5 * (ALPHA_EDGES[:-1] + ALPHA_EDGES[1:])
DALPHA = ALPHA_EDGES[1] - ALPHA_EDGES[0]
SMOOTH_SIGMA_BINS = 6.0     # suavizado de 6*0.005 = 0.03 dex; el sigma va
#   escalado con los bins para no meter shot-noise por bin
TAIL_FRAC = 0.05
# Extrapolar la cola superior de lnZ (ajuste Z~exp(-c alpha^2)) en vez de cortar
# con -inf arriba del ultimo bin muestreado. Ver lnZ_from_science_chain: sin esto,
# al COMBINAR pulsares el soporte conjunto lo fija hasta donde llego la cadena del
# pulsar dominante (cota optimista y UL95 pegado al borde). False = comportamiento
# viejo (reproduce numero por numero las corridas previas).
EXTRAPOLAR_COLA = True

# La posterior final se evalua interpolando lnZ_total a una grilla fina aparte.

N_ALPHA_FINE = int(round((LOG10_ALPHA_MAX - LOG10_ALPHA_MIN) / 0.002)) + 1
ALPHA_FINE = np.linspace(LOG10_ALPHA_MIN, LOG10_ALPHA_MAX, N_ALPHA_FINE)

if SINGLE_MASS_EV is not None:
    MASS_GRID_EV = np.array([float(SINGLE_MASS_EV)])
else:
    MASS_GRID_EV = np.logspace(
        np.log10(MASS_MIN_EV), np.log10(MASS_MAX_EV), N_MASS)

print(f"[1] PATHS OK   masas BASE={len(MASS_GRID_EV)} "
      f"({MASS_GRID_EV[0]:.2e}..{MASS_GRID_EV[-1]:.2e} eV)   N_ITER={N_ITER}")
print(f"    [v3] prior log10_alpha in [{LOG10_ALPHA_MIN:.4f}, {LOG10_ALPHA_MAX:.4f}] "
      f"= alpha_fis [1e{LOG10_ALPHA_MIN+_LOG10_CONV:.2f}, 1e{LOG10_ALPHA_MAX+_LOG10_CONV:.2f}] GeV^-1"
      f"  ({N_ALPHA_BINS} bins = {(LOG10_ALPHA_MAX-LOG10_ALPHA_MIN)/N_ALPHA_BINS:.4f} dex/bin)")
print(f"    modelos={ALLOWED_MODELS}  N_target={N_PULSARS_TARGET}")
print(
    f"    ACOPLAMIENTO LINEAL alpha   ALPHA_CONVERSION={ALPHA_CONVERSION:.6e}")
print(f"    RN priors: log10_A_red~U({LOG10_ARED_MIN},{LOG10_ARED_MAX}), "
      f"gamma_red~U({GAMMA_RED_MIN},{GAMMA_RED_MAX})")

# %% [2] DETECCION / FILTRADO DE PULSARES


def resolve_paths(psr_name):
    par_hits = sorted(glob.glob(os.path.join(
        WB_ROOT, "par", f"{psr_name}_PINT_*.wb.par")))
    tim_hits = sorted(glob.glob(os.path.join(
        WB_ROOT, "tim", f"{psr_name}_PINT_*.wb.tim")))
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
    import re
    name_re = re.compile(r"^[BJ]\d{4}[+-]\d{2,4}$")
    pat = os.path.join(WB_ROOT, "par", "*_PINT_*.wb.par")
    names = set()
    for f in glob.glob(pat):
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
            skip.append((n, f"sin-archivos:{type(e).__name__}"))
            continue
        if model in allowed:
            keep.append(n)
        else:
            skip.append((n, str(model)))
    print(
        f"  [FILTRO {'/'.join(allowed)}] {len(keep)} pulsares OK, {len(skip)} descartados")
    return keep

# %% [3] TEMPLATE ULDM (ACOPLAMIENTO LINEAL alpha)


def _stable_minus(q_col, psi_col, t_arr, t_ref, tt_row):
    """
    Ramas "minus" del template excentrico, ya divididas por (n wb - m) en la
    forma que es estable en la resonancia m = n wb:
        Cm_over =  t~ sin(q(t+t_ref)/2 + psi) sinc(q t~/2pi)
        Sm_over = -t~ cos(q(t+t_ref)/2 + psi) sinc(q t~/2pi)
    con t~ = t - t_ref, q = m - n wb, psi = n wb t_ref + Y.
    """
    mid = (q_col * (0.5 * (t_arr[None, :] + t_ref)) +
           psi_col)   # q(t+t_ref)/2 + psi
    sincv = np.sinc(q_col * tt_row / (2.0 * np.pi))
    Cm_over = tt_row * np.sin(mid) * sincv
    Sm_over = -tt_row * np.cos(mid) * sincv
    return Cm_over, Sm_over


# --- Helpers para las INTEGRALES en forma cerrada (reemplazan cualquier cumtrapz)
# Todos exactos y estables en la resonancia (el sinc / la serie dan el limite
# secular automaticamente, sin ifs ni ramas aparte).

def _one_minus_cos_over_z2(z):
    """(1 - cos z)/z^2, exacto y estable (vale 1/2 en z=0)."""
    s = np.sinc(z / (2.0 * np.pi))
    return 0.5 * s * s


def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, estable (0 en z=0). Serie de Taylor para |z|<0.1 (ahi la
    resta z-sin z pierde todas las cifras); resta directa fuera."""
    z = np.asarray(z, dtype=np.float64)
    chico = np.abs(z) < 0.1
    zs = np.where(chico, 1.0, z)
    directa = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    serie = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(chico, serie, directa)


def _mean_cos(m_dm, t_arr, t_ref, gamma_f, tt):
    """<cos(mt+Y)> = (1/t~) int_{t_ref}^t cos(mt'+Y)dt', con sinc (sin 0/0 ni /m)."""
    medio = m_dm * 0.5 * (t_arr + t_ref) + gamma_f
    return np.cos(medio) * np.sinc(m_dm * tt / (2.0 * np.pi))


def _int_cos(omega, phase, t_arr, t_ref):
    """
    int_{t_ref}^{t} cos(omega t' + phase) dt', forma cerrada ESTABLE:
      = t~ cos(omega (t+t_ref)/2 + phase) sinc(omega t~/2pi),  t~ = t - t_ref.
    En omega=0 (rama resonante) sinc->1 y queda el termino SECULAR t~ cos(phase).
    Broadcasting: omega,phase (N,1) ; t_arr (T,) -> (N,T).
    """
    tt = (t_arr - t_ref)[None, :]
    mid = omega * (0.5 * (t_arr + t_ref))[None, :] + phase
    return tt * np.cos(mid) * np.sinc(omega * tt / (2.0 * np.pi))


# (Los helpers analiticos estan definidos arriba, junto a _stable_minus.)


def compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """Selector: ELL1 (baja ecc) vs DD/BT (excentrico)."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)

    # ---- TEMPLATE BT/DD EXCENTRICO (acoplamiento lineal) ----
    n_vec = np.arange(1, N_TERMS)
    n_col = n_vec[:, None]
    jn_col = jn(n_vec, n_vec * orb_pars["ecc"])[:, None]      # J_n(ne)
    jnp_col = jvp(n_vec, n_vec * orb_pars["ecc"])[:, None]      # J_n'(ne)

    # cargo directo los parametros orbitales del best-fit de NANOGrav
    wb = orb_pars["omega_b"]
    ecc = orb_pars["ecc"]
    t0_s = orb_pars["T0_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    sqrt1me2 = np.sqrt(1.0 - ecc**2)

    T_tilde = t_arr - t0_s                  # t~ = t - T0
    tt_row = T_tilde[None, :]
    nw = n_vec * wb                     # n wb
    nw_col = nw[:, None]

    # Phi = Phi0 rho cos(m t + Y). Como el template es lineal en Phi, la amplitud
    # A0 = Phi0 rho sale de factor comun (en el cuadratico era Phi0^2 rho^2).
    Phi0 = np.sqrt(2.0 * DM_DENSITY) / m_dm
    A0 = Phi0 * rho_f

    # Fases
    m_t = m_dm * t_arr + gamma_f        # m t + Y       (T,)
    m_t0 = m_dm * t0_s + gamma_f        # m T0 + Y      (escalar)
    cos_mt = np.cos(m_t)
    sin_mt = np.sin(m_t)
    cos_mt0 = np.cos(m_t0)
    sin_mt0 = np.sin(m_t0)

    nw_tt = nw_col * tt_row              # n wb t~       (N,T)
    cos_nw_tt = np.cos(nw_tt)
    sin_nw_tt = np.sin(nw_tt)

    # Todo delta(param) sale de dos familias: la "plus" (fase +n wb t~, denominador
    # n wb + m) y la "minus" (fase -n wb t~, denominador n wb - m). La minus resuena.

    # Fases "plus" :  m t + Y + n wb t~   (todavia sin dividir)
    phase_pp = m_t[None, :] + nw_tt
    # [cos(m t+Y+n wb t~) - cos(m T0+Y)]
    Cp = np.cos(phase_pp) - cos_mt0
    Sp = np.sin(phase_pp) - sin_mt0

    # Ramas "minus", ya divididas por (n wb - m) de forma estable:
    q_col = (m_dm - nw)[:, None]           # q = m - n wb
    psi_col = (nw * t0_s + gamma_f)[:, None]  # psi = n wb T0 + Y
    Cm_over, Sm_over = _stable_minus(q_col, psi_col, t_arr, t0_s, tt_row)
    #  Cm_over = [cos(m t+Y-n wb t~) - cos(m T0+Y)] / (n wb - m)
    #  Sm_over = [sin(m t+Y-n wb t~) - sin(m T0+Y)] / (n wb - m)

    # n wb + m, el denominador de la rama plus
    sigma_p = (nw + m_dm)[:, None]
    # n wb - m, el denominador de la rama minus
    sigma_m = (nw - m_dm)[:, None]
    two_m = 2.0 * m_dm
    coeff_p = ((nw - two_m) / (nw + m_dm))[:, None]   # (n wb - 2m)/(n wb + m)
    coeff_m = (two_m + nw)[:, None]                    # (2m + n wb)

    # ---- delta a / a   (Ec. C.1b) ----
    #   -2 A0 [cos(mt+Y) - cos(mT0+Y)]
    #   -8 A0 sum_n J_n [cos(mt+Y) cos(n wb t~) - cos(mT0+Y)]
    #   +6 A0 wb sum_n n J_n { Cp/(m+n wb) + Cm/(n wb - m) }
    daa_mono = -2.0 * A0 * (cos_mt - cos_mt0)
    daa_8 = -8.0 * A0 * \
        np.sum(jn_col * (cos_mt[None, :] * cos_nw_tt - cos_mt0), axis=0)
    daa_6 = 6.0 * A0 * wb * \
        np.sum(n_col * jn_col * (Cp / sigma_p + Cm_over), axis=0)
    daa = daa_mono + daa_8 + daa_6

    # ---- delta a / a EFECTIVO = <delta a/a> (canal PB) — TODO ANALITICO ----
    # La correccion de la directora: el canal PB ve el PROMEDIO temporal <da/a>,
    # no el instantaneo. Las integrales van en forma CERRADA, estables en m = n*wb
    # (SIN 0/0): cada rama con un T_tilde explicito afuera, y la rama minus
    # resonante via los helpers (z-sin z)/z^2 y (1-cos z)/z^2 (regulares en z=0).
    #   int_{T0}^t [cos(mt'+Y) - cos(mT0+Y)] dt' = t~ [<cos> - cos(mT0+Y)]
    mean_cos = _mean_cos(m_dm, t_arr, t0_s, gamma_f, T_tilde)
    daa_monoe = -2.0 * A0 * (mean_cos - cos_mt0)
    # <Cp>: rama plus, nunca resuena -> cos(theta0 + sigma_p t~/2) sinc(...)
    mean_Cp = np.cos(m_t0 + 0.5 * sigma_p * tt_row) * \
        np.sinc(sigma_p * tt_row / (2.0 * np.pi)) - cos_mt0
    # int_{T0}^t Cm_over dt' = t~^2 [cos_th0 (z-sin z)/z^2 + sin_th0 (1-cos z)/z^2],
    # con z = (m - n wb) t~ (=0 en resonancia); los dos cocientes son regulares.
    z_col = q_col * tt_row
    mean_Cm_over = T_tilde * (cos_mt0 * _z_minus_sin_over_z2(z_col)
                              + sin_mt0 * _one_minus_cos_over_z2(z_col))
    daa_8e = -8.0 * A0 * np.sum(
        jn_col * (0.5 * (mean_Cp + sigma_m * mean_Cm_over)), axis=0)
    daa_6e = 6.0 * A0 * wb * np.sum(
        n_col * jn_col * (mean_Cp / sigma_p + mean_Cm_over), axis=0)
    daae = daa_monoe + daa_8e + daa_6e     # ya es el promedio (sin /T_tilde)

    # ---- delta excentricidad   (Ec. C.1c) ----
    #   A0 (1-e^2)/e sum_n J_n { (n wb-2m)/(n wb+m) Cp + (2m+n wb) Cm/(n wb - m) }
    dECC = A0 * (1.0 - ecc**2) / ecc * np.sum(
        jn_col * (coeff_p * Cp + coeff_m * Cm_over), axis=0)

    # ---- delta Omega   (Ec. C.1d) ----
    #   A0 sqrt(1-e^2)/e sum_n J_n' { (n wb-2m)/(n wb+m) Sp - (2m+n wb) Sm/(n wb - m) }
    dOM = A0 * sqrt1me2 / ecc * np.sum(
        jnp_col * (coeff_p * Sp - coeff_m * Sm_over), axis=0)

    # ---- delta T0 ----
    # No esta en el paper: sale del template cuadratico de T0 con la regla
    # beta Phi^2/2 -> alpha Phi. La parte inicial queda -2 veces la integral
    # de Phi, que sale analitica:
    dT0_mono = -2.0 * A0 * (sin_mt - sin_mt0) / m_dm

    # Los armonicos van con Phi (coef A_n) y Phidot (coef B_n), Ec. 2.24. La
    # integral  term_harm = sum_n int_{t0}^{t}[ an Phi cos(n wb t~) + bn Phidot sin(n wb t~)] dt'
    # se hace ANALITICA (antes cumtrapz, IMPRECISO en resonancia: el integrando
    # oscila a ~m, mas rapido que el muestreo de TOAs -> el trapecio erraba ~15x).
    # Producto-a-suma:
    #   I1 = int cos(mt+Y) cos(n wb t~) dt' = 1/2 [ _int_cos(m-nwb, Y+nwb T0) + _int_cos(m+nwb, Y-nwb T0) ]
    #   I2 = int sin(mt+Y) sin(n wb t~) dt' = 1/2 [ _int_cos(m-nwb, Y+nwb T0) - _int_cos(m+nwb, Y-nwb T0) ]
    #   term_harm = sum_n [ an A0 I1 - bn A0 m I2 ]
    # El termino (m - n wb) = 0 (resonancia) es el SECULAR: _int_cos -> t~ cos(Y + n wb T0).
    # El limite inferior es T0, el mismo que el de dT0_mono, asi que las dos
    # mitades de delta T0 son la integral del mismo intervalo [T0, t]. (Antes era
    # t_arr[0], heredado del cumtrapz con initial=0: la diferencia es una
    # constante, que la marginalizacion del timing model absorbe.)
    an = 4.0 * jn_col - 2.0 * (1.0 - ecc**2) / ecc * n_col * jnp_col      # A_n / wb
    bn = 4.0 / n_col / wb * jn_col + 4.0 / ecc / wb * (1.0 - ecc**2) * jnp_col  # B_n / wb
    wm = (m_dm - nw)[:, None]                    # m - n wb  (=0 en resonancia -> secular)
    wp = (m_dm + nw)[:, None]                    # m + n wb
    phim = (gamma_f + nw * t0_s)[:, None]        # Y + n wb T0
    phip = (gamma_f - nw * t0_s)[:, None]        # Y - n wb T0
    Dm = _int_cos(wm, phim, t_arr, t0_s)         # fila resonante ~ t~ (SECULAR)
    Dp = _int_cos(wp, phip, t_arr, t0_s)
    I1 = 0.5 * (Dm + Dp)
    I2 = 0.5 * (Dm - Dp)
    term_harm = np.sum(an * A0 * I1 - bn * A0 * m_dm * I2, axis=0)
    dT0 = dT0_mono - term_harm

    # Termino extra de delta Pb aparte de la contribucion del semieje mayor:
    # delta Pb/Pb = (3/2) delta a/a - (1/2) delta(G M_T)/(G M_T), y con el
    # acoplamiento universal (Ec. 2.14/2.20b) delta(G M_T)/(G M_T) = alpha Phi.
    dPb_extrae = -0.5 * A0 * mean_cos    # -(1/2) <Phi>, analitico (reusa mean_cos)

    return (partials["PB"] * PB_s * (1.5 * daae + dPb_extrae)
            + partials["A1"] * A1_s * daa
            + partials["ECC"] * dECC
            + partials["OM"] * dOM
            + partials["T0"] * dT0)

    # la senal se escribe como partial(r) / partial(param) * delta(param)


def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """
    Template ELL1 (baja excentricidad), acoplamiento lineal alpha.
    """
    # Aca sobrevive solo el primer armonico orbital (n=1, o sea wb): no hay suma
    # en n y todo queda en arrays (T,).
    wb = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s = orb_pars["PB_s"]
    A1_s = orb_pars["A1_s"]
    # (eps1/eps2 ya no se usan: el template corregido de delta eta / delta kappa
    #  no depende del argumento del periastro. Ver el bloque de abajo.)

    T_tilde = t_arr - tasc_s                  # t~ = t - Tasc
    Phi0 = np.sqrt(2.0 * DM_DENSITY) / m_dm
    A0 = Phi0 * rho_f                     # Phi0 rho

    m_t = m_dm * t_arr + gamma_f          # m t + Y
    m_tasc = m_dm * tasc_s + gamma_f         # m Tasc + Y
    cos_mt = np.cos(m_t)
    sin_mt = np.sin(m_t)
    cos_mtasc = np.cos(m_tasc)
    sin_mtasc = np.sin(m_tasc)

    # ---- delta x  (parametro A1)   (Ec. C.5b) ----
    #   delta x / x = -2 alpha Phi0 rho [cos(mt+Y) - cos(mTasc+Y)]
    daa = -2.0 * A0 * (cos_mt - cos_mtasc)

    # ---- delta eta (EPS1) y delta kappa (EPS2) ----
    # [CORREGIDO 2026-09] La version anterior construia las dos deltas a partir
    # de un UNICO bloque {brace} multiplicado por eta/e^2 y kappa/e^2, lo que
    # forzaba  delta_eta / delta_kappa = eta/kappa = tan(w)  para todo t, o sea
    # delta_w == 0. Eso NO vale: en el limite circular  e*wdot  es O(e^0), del
    # mismo orden que  edot  (wdot ~ 1/e y el prefactor e lo cancela), asi que
    # la contribucion del periastro NO se puede tirar. Contrastado contra la
    # cuadratura directa de las Ecs. (22b) y (22d) del paper de sensitivity
    # (Huxhagen & Lopez Nacir 2026, PRD 114, 043056): la forma vieja erraba
    # entre 65% y 88%; esta forma coincide a O(e) (2e-5 .. 2e-4 a e = 1e-4).
    #
    # Derivacion (limite circular, theta = Psi' - w, Psi' = wb (t - Tasc)):
    #   etadot   = alpha [ wb Phi cos Psi' - 2 Phidot sin Psi' ]
    #   kappadot = -alpha [ wb Phi sin Psi' + 2 Phidot cos Psi' ]
    # con Phi = Phi0 cos(k Psi' + Ups_asc), k = m/wb, Ups_asc = m Tasc + Y.
    # Producto-a-suma y una integral en Psi' (dt = dPsi'/wb) dan:
    #   delta_eta   = (A0/wb) [ (wb/2 - m) R_s(k+1) + (wb/2 + m) R_s(k-1) ]
    #   delta_kappa = (A0/wb) [ (wb/2 - m) R_c(k+1) - (wb/2 + m) R_c(k-1) ]
    # con R_s(d) = [sin(d Psi' + Ups_asc) - sin Ups_asc]/d  y
    #     R_c(d) = [cos(d Psi' + Ups_asc) - cos Ups_asc]/d,
    # escritas con sinc para que la resonancia k = 1 (d = k-1 -> 0) salga sola,
    # sin rama aparte, igual que en el resto del template.
    #
    # Nota: la expresion resultante NO depende del argumento del periastro w.
    # Es lo esperable -- para una orbita circular w esta mal definido -- y de
    # paso saca del template el arctan2(EPS1, EPS2), que para estos sistemas es
    # ruido.
    Psi_p = wb * T_tilde                      # Psi' = wb (t - Tasc)
    kk = m_dm / wb                            # k = m / wb
    Ups_asc = m_tasc                          # Ups_asc = m Tasc + Y

    def _R_s(d):
        """[sin(d Psi' + Ups_asc) - sin(Ups_asc)] / d ; estable en d -> 0."""
        return Psi_p * np.cos(0.5 * d * Psi_p + Ups_asc) * \
            np.sinc(d * Psi_p / (2.0 * np.pi))

    def _R_c(d):
        """[cos(d Psi' + Ups_asc) - cos(Ups_asc)] / d ; estable en d -> 0."""
        return -Psi_p * np.sin(0.5 * d * Psi_p + Ups_asc) * \
            np.sinc(d * Psi_p / (2.0 * np.pi))

    cf_p = wb / 2.0 - m_dm                    # (wb/2 - m), acompana a k+1
    cf_m = wb / 2.0 + m_dm                    # (wb/2 + m), acompana a k-1 (resuena)

    dEPS1 = (A0 / wb) * (cf_p * _R_s(kk + 1.0) + cf_m * _R_s(kk - 1.0))  # delta eta
    dEPS2 = (A0 / wb) * (cf_p * _R_c(kk + 1.0) - cf_m * _R_c(kk - 1.0))  # delta kappa

    # ---- delta Tasc ----
    # Igual que delta T0 en el excentrico: -2 alpha por la integral de Phi, que
    # aca no lleva armonicos y sale analitica.
    dTASC = -2 * A0 * (sin_mt - sin_mtasc) / m_dm

    # ---- delta Pb extra ----  (lo mismo que en el excentrico: -(1/2) alpha Phi)
    # <delta x/x> y <Phi> ANALITICOS con _mean_cos (sin 0/0 en t = Tasc ni /m).
    mean_cos = _mean_cos(m_dm, t_arr, tasc_s, gamma_f, T_tilde)
    daae = -2.0 * A0 * (mean_cos - cos_mtasc)     # <delta x / x>
    dPb_extrae = -0.5 * A0 * mean_cos             # -(1/2) <Phi>

    # [FINAL] ELL1 completo: Delta_R = x[sin Phi + k/2 sin 2Phi - h/2 cos 2Phi - 3/2 h]
    # (h = eta = EPS1). tempo/PINT omiten el termino constante -3/2 x eta (Lange et al.
    # 2001, Ec. A6: "terms which are constant in time are omitted"), asi que la
    # columna EPS1 de la matriz de diseno es solo -x cos(2Phi)/2. Para eta constante
    # da igual (lo absorbe la fase), pero la ULDM hace oscilar eta y
    # -3/2 x delta_eta(t) es un retardo que ningun parametro de timing absorbe.
    # Verificado contra la orbita integrada (Chequeos_referee/t38) y contra el
    # modelo DD de PINT, que no recorta nada (validacion/v02).
    return (partials["PB"] * PB_s * (1.5 * daae + dPb_extrae)
            + partials["A1"] * A1_s * daa
            + (partials["EPS1"] - 1.5 * A1_s) * dEPS1
            + partials["EPS2"] * dEPS2
            + partials["TASC"] * dTASC)

# %% [4] CARGA DE PULSARES Y TIMING


def model_ecc(model):
    if hasattr(model, "ECC") and model.ECC.value is not None:
        return float(model.ECC.value)
    if hasattr(model, "EPS1") and model.EPS1.value is not None:
        return float(np.hypot(model.EPS1.value, model.EPS2.value))
    return 1.0


def is_ell1_model(model):
    return hasattr(model, "EPS1") and getattr(model.EPS1, "value", None) is not None


def convert_to_ell1(model):
    if not is_ell1_model(model):
        from pint.binaryconvert import convert_binary
        model = convert_binary(model, "ELL1")
        # convert_binary (p.ej. desde DDK) puede dejar parametros "Kopeikin"
        # que ya no pertenecen a BinaryELL1 y
        # avisarian/romperian al construir el Pulsar de enterprise -> los quitamos.
        for orphan in ("KIN", "KOM", "K96"):
            if orphan in getattr(model, "params", []):
                try:
                    model.remove_param(orphan)
                except Exception:
                    pass
    for p in ("PB", "A1", "EPS1", "EPS2", "TASC"):
        if hasattr(model, p):
            getattr(model, p).frozen = False
    return model


def drop_toas_outside_dmx(model, toas):
    """WidebandTimingModel exige que cada TOA caiga en algun intervalo DMX.
    Descarta las que no (2/1493 en J1713 -> impacto despreciable)."""
    keys = [p[len("DMXR1_"):] for p in getattr(
        model, "params", []) if p.startswith("DMXR1_")]
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
    print(
        f"    [DMX] descarto {int((~cov).sum())} TOA(s) fuera de intervalos DMX", flush=True)
    return toas[cov]


def drop_empty_dmx_bins(model, toas):
    """WidebandTimingModel rompe si un bin DMX no contiene ninguna TOA: los quita.
    Llamar DESPUES de drop_toas_outside_dmx."""
    idxs = [int(p[len("DMXR1_"):])
            for p in getattr(model, "params", []) if p.startswith("DMXR1_")]
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
            for pre in ("DMX_", "DMXR1_", "DMXR2_", "DMXF1_", "DMXF2_"):  # fallback
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
    print(f"    [DMX] removi {len(empty)} bin(s) DMX vacio(s)", flush=True)
    return model


# Conserva la deriva de x de Kopeikin al convertir DDK -> ELL1 (ver el
# comentario dentro de load_pulsar). Ponerlo en False reproduce el
# comportamiento viejo (Kopeikin borrado). Solo toca a los modelos que traen
# KIN/KOM/K96: en el dataset de 15 anios, unicamente J1713+0747.
KOPEIKIN_PARAMS = ("KIN", "KOM", "K96")


def tiene_kopeikin(model):
    """True si el modelo trae parametros de Kopeikin con valor.

    Son los UNICOS que convert_to_ell1 borra sin reemplazo; todo lo demas que
    hace convert_binary es reparametrizar (ECC/OM/T0 -> EPS1/EPS2/TASC). En el
    dataset de 15 anios da True solo para J1713+0747 (BINARY DDK): los otros 20
    sistemas que pasan por la conversion son DD/BT y no pierden nada.

    Sustituirles los residuos igual NO es neutro -- deja residuos de un modelo
    pegados a la matriz de diseno de otro. Medido con el mismo diseno y el mismo
    ruido en el MAP (dlnL = lnL con los residuos del original menos lnL con los
    del convertido): J1713+0747 da +518.6, pero de los otros 20 hay 14 con dlnL
    NEGATIVO, hasta -655. Y el chi2 del template se infla hasta x15
    (B1953+29: d^T G^-1 d 5.6 -> 84.6; J1643-1224: 16.0 -> 81.0).
    """
    for p in KOPEIKIN_PARAMS:
        o = getattr(model, p, None)
        if o is not None and getattr(o, "value", None) is not None:
            return True
    return False


KOPEIKIN_FIJO = True

# ---- [CONGELAR_SECULARES] ---------------------------------------------------
# Mismo hook que DIRAC_vectorial_v2/pipeline_vectorial.py. Saca del DISENO las
# derivadas seculares que el .par traiga fiteadas, o sea que deja de
# marginalizarlas y las trata como valores medidos exactos.
#
# Por que PBDOT y no todo: medido con el calculo exacto sin MCMC
# (Comparacion NB vs WB/seculares_todos.py) en el nucleo del pico resonante,
# mismo pulsar y mismos residuos:
#   - A1DOT libre no cuesta NADA (+0.00 a +0.09 dex): vive en el canal A1, que
#     se lleva ~0 % de la norma del template.
#   - PBDOT libre cuesta +0.55 a +1.30 dex: vive en el canal PB, que se lleva
#     ~100 % de la norma. Y 7 de los 8 sistemas con PBDOT fiteado perforan la
#     envolvente conjunta si se congela.
# Decision tomada: PBDOT congelado en los TRES casos (lineal, cuadratico,
# vectorial) por consistencia. Default PBDOT aca adentro ADEMAS del export del
# .slurm, para que si la variable no llega igual quede congelado.
CONGELAR_SECULARES = tuple(
    x.strip() for x in os.environ.get("CONGELAR_SECULARES", "PBDOT").split(",")
    if x.strip())
print(f"    [v3] CONGELAR_SECULARES = "
      f"{', '.join(CONGELAR_SECULARES) if CONGELAR_SECULARES else '(nada)'}")


def load_pulsar(par, tim, convert_low_ecc=True):
    from enterprise.pulsar import Pulsar
    from pint.models import get_model_and_toas
    cwd = os.getcwd()
    timdir = os.path.dirname(os.path.abspath(tim)) or "."
    try:
        os.chdir(timdir)
        model, toas = get_model_and_toas(
            os.path.relpath(par, timdir), os.path.basename(tim),
            ephem="DE440", planets=True)
    finally:
        os.chdir(cwd)
    # ---- [KOPEIKIN] ----------------------------------------------------
    # convert_to_ell1 borra KIN/KOM/K96 y con ellos la deriva secular de
    # x = a1 sin(i) inducida por el movimiento propio. Para J1713+0747 eso son
    # ~13 us pico a pico (130 sigma_TOA) y lo que queda sin modelar tiene la
    # forma t*sin(Phi_orb): EXACTAMENTE la del template ULDM en la resonancia
    # k = 1, asi que el template se la come y la cota resonante se debilita un
    # factor ~800. Solucion: residuos calculados con el modelo ORIGINAL (que si
    # tiene Kopeikin) + matriz de diseno del modelo ELL1. Equivale a tratar la
    # deriva como termino FIJO predicho por KIN/KOM y el movimiento propio, que
    # es lo que es: KIN y KOM estan medidos por la paralaje anual-orbital y el
    # Shapiro, no por la deriva. (Descongelar A1DOT tambien limpia los residuos
    # pero es DEGENERADO con el canal resonante: destruye la sensibilidad en el
    # pico.)  Afecta solo a los binarios que NO son ELL1: J1713+0747 (DDK) y
    # J1312+0051 (BT).
    residuos_orig = None
    if (convert_low_ecc and model_ecc(model) < ECC_LOW_THRESHOLD
            and not is_ell1_model(model)):
        # OJO: solo restituir donde la conversion borra algo de verdad, o sea
        # donde hay KIN/KOM/K96. Ver tiene_kopeikin().
        if KOPEIKIN_FIJO and tiene_kopeikin(model):
            toas_o  = drop_toas_outside_dmx(model, toas)
            # deepcopy: TimingModel no tiene .copy() y su __getattr__
            # convierte el intento en AttributeError.
            model_o = drop_empty_dmx_bins(deepcopy(model), toas_o)
            psr_o   = Pulsar(toas_o, model_o, timing_package="pint")
            residuos_orig = np.asarray(psr_o.residuals, dtype=float)
            del psr_o
        model = convert_to_ell1(model)
    toas  = drop_toas_outside_dmx(model, toas)
    model = drop_empty_dmx_bins(model, toas)
    # [CONGELAR_SECULARES] va JUSTO antes de construir el Pulsar: enterprise arma
    # la matriz de diseno (psr.Mmat / psr.fitpars) de los parametros NO frozen.
    # Va despues de la restitucion de Kopeikin a proposito: psr_o solo se usa
    # para los RESIDUOS, que no dependen de que este frozen y que no.
    if CONGELAR_SECULARES:
        _cong = []
        for _p in CONGELAR_SECULARES:
            _o = getattr(model, _p, None)
            if _o is not None and getattr(_o, "value", None) is not None:
                _o.frozen = True
                _cong.append(_p)
        if _cong:
            print("    [CONGELADO] fuera del diseno: %s" % ", ".join(_cong),
                  flush=True)
    psr = Pulsar(toas, model, timing_package="pint")
    if residuos_orig is not None:
        if len(residuos_orig) != len(psr.residuals):
            raise RuntimeError(
                "el modelo original y el convertido no dan el mismo set de "
                "TOAs (%d vs %d): no se puede restituir Kopeikin"
                % (len(residuos_orig), len(psr.residuals)))
        delta = np.asarray(psr.residuals, dtype=float) - residuos_orig
        # [orden] psr.residuals devuelve _residuals[_isort] y residuos_orig ya viene
        # ordenado en el tiempo: hay que deshacer esa permutacion (J1713: el .tim
        # no esta en orden temporal; sin esto chi2/TOA = 1863 en vez de 219).
        psr._residuals = residuos_orig[psr._iisort]
        print("    [KOPEIKIN] deriva restituida: %.3f us rms, %.3f us pico a "
              "pico" % (np.std(delta) * 1e6, (delta.max() - delta.min()) * 1e6),
              flush=True)
    return psr, model


def build_context(psr, timing_model):
    M = np.asarray(psr.Mmat, dtype=np.float64)  # matriz de diseno
    M_names = list(psr.fitpars)
    t_arr = np.asarray(psr.toas, dtype=np.float64)  # t_i
    n_toa = len(t_arr)

    def col(name, conv):
        if name in M_names:
            return np.ascontiguousarray(M[:, M_names.index(name)] * conv, dtype=np.float64)
        return None

    pb_q = getattr(getattr(timing_model, "PB", None), "quantity", None)
    fb0 = None
    # me fijo si en el timing model fitearon PB o la frecuencia fb0
    if pb_q is not None:
        PB_s = float(pb_q.to(u.s).value)
    else:
        fb0_q = getattr(getattr(timing_model, "FB0", None), "quantity", None)
        if fb0_q is None:
            raise RuntimeError("El modelo no tiene ni PB ni FB0.")
        fb0 = float(fb0_q.to(u.Hz).value)
        PB_s = 1.0 / fb0

    orb_pars = {"PB_s": PB_s, "A1_s": float(timing_model.A1.value)}
    orb_pars["omega_b"] = 2.0 * np.pi / orb_pars["PB_s"]

    is_ell1 = hasattr(
        timing_model, "EPS1") and timing_model.EPS1.value is not None
    if is_ell1:
        eps1 = float(timing_model.EPS1.value)
        eps2 = float(timing_model.EPS2.value)
        orb_pars.update(eps1=eps1, eps2=eps2, ecc=float(np.hypot(eps1, eps2)),
                        omega=float(np.arctan2(eps1, eps2)),
                        TASC_s=float(timing_model.TASC.value * DAY_IN_S))
        spec = [("PB", 1.0/DAY_IN_S), ("A1", 1.0),
                ("EPS1", 1.0), ("EPS2", 1.0), ("TASC", 1.0/DAY_IN_S)]
    else:
        orb_pars.update(ecc=float(timing_model.ECC.value),
                        OM_rad=float(np.deg2rad(timing_model.OM.value)),
                        T0_s=float(timing_model.T0.value * DAY_IN_S))
        spec = [("PB", 1.0/DAY_IN_S), ("A1", 1.0),
                ("ECC", 1.0), ("OM", 180.0/np.pi), ("T0", 1.0/DAY_IN_S)]

    partials, missing = {}, []
    for name, conv in spec:
        c = col(name, conv)
        if c is None:
            partials[name] = np.zeros(n_toa, dtype=np.float64)
            missing.append(name)
        else:
            partials[name] = c
    if "PB" in missing and "FB0" in M_names:
        fb0_col = col("FB0", 1.0)
        if fb0 is None:
            fb0 = 1.0 / orb_pars["PB_s"]
        partials["PB"] = fb0_col * (-(fb0 ** 2))
        missing.remove("PB")

    return t_arr, partials, orb_pars, missing

# t_i de los TOAs, parciales de los residuos y parametros orbitales.

# %% [5] CONSTRUCCION DEL PTA


def make_uldm_signal(t_arr, partials, orb_pars, m_dm):
    from enterprise.signals import parameter, signal_base, deterministic_signals

    @signal_base.function
    def uldm_delay(toas, log10_alpha=-45.0, rho_f=1.0, gamma_f=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)
        return (10.0 ** log10_alpha) * h

    # defino los parametros para enterprise: alpha, rho y Upsilon

    la = parameter.Uniform(LOG10_ALPHA_MIN, LOG10_ALPHA_MAX)("log10_alpha")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0*np.pi)("gamma_f")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_alpha=la, rho_f=rf, gamma_f=gf), name="uldm")


# [FINAL] DMJUMP del .par como constantes (seleccion por frontend), igual que
# enterprise_extensions/NANOGrav. PINT modela las DM wideband como
# DM + DMX - DMJUMP(fe); ENTERPRISE suma dmjump a las DM medidas, asi que el valor
# consistente es +DMJUMP. Sin esto cada bin DMX arranca corrido en -<DMJUMP>.
# Verificado: la lnL coincide con la de las cadenas de ruido de NANOGrav
# (Chequeos_referee/t33 y validacion/v03).
def by_fe_flag(flags):
    return {v: flags["fe"] == v for v in sorted(np.unique(flags["fe"]))}


def dmjumps_par(par, psr_name, pta):
    """{nombre_parametro: +DMJUMP} para cada frontend del pulsar."""
    J = {}
    for l in open(par):
        t = l.split()
        if t and t[0] == "DMJUMP" and len(t) > 3 and t[1] == "-fe":
            J[t[2]] = float(t[3].replace("D", "e"))
    out = {}
    for p in pta.param_names:
        if p.endswith("_dmjump"):
            fe = p[len(psr_name) + 1:-len("_dmjump")]
            out[p] = J.get(fe, 0.0)
    return out


def build_pta(psr, t_arr, partials, orb_pars, m_dm):
    # el objeto PTA nos permite introducir la senal completa + ruido

    from enterprise.signals import (parameter, white_signals, gp_signals,
                                    gp_priors, signal_base)
    from enterprise.signals.selections import Selection

    def by_f_flag(flags):
        return {v: flags["f"] == v for v in sorted(np.unique(flags["f"]))}
    sel = Selection(by_f_flag)

    # white noise:

    ef = white_signals.MeasurementNoise(
        efac=parameter.Uniform(0.01, 10.0),
        log10_t2equad=parameter.Uniform(-10.0, -1.0), selection=sel)

    # Ruido rojo acromatico: GP en base de Fourier (powerlaw). Se samplean
    # log10_A ~ U(-20,-11) y gamma ~ U(0,10); los coeficientes se marginalizan
    # analiticamente (via Woodbury) dentro de get_lnlikelihood.
    rn = gp_signals.FourierBasisGP(
        spectrum=gp_priors.powerlaw(
            log10_A=parameter.Uniform(LOG10_ARED_MIN, LOG10_ARED_MAX),
            gamma=parameter.Uniform(GAMMA_RED_MIN, GAMMA_RED_MAX)),
        components=N_FREQ_RN, Tspan=(t_arr.max() - t_arr.min()))

    # defino la dispersion measure, aclarando que usamos wideband data
    dm_wtm = gp_signals.WidebandTimingModel(
        dmefac=parameter.Uniform(0.1, 5.0),
        log10_dmequad=parameter.Uniform(-10.0, -1.0),
        dmjump=parameter.Uniform(-1.0, 1.0), selection=sel,
        dmjump_selection=Selection(by_fe_flag))

    uldm = make_uldm_signal(t_arr, partials, orb_pars, m_dm)
    signal = (ef + rn + dm_wtm + uldm)(psr)
    return signal_base.PTA([signal])


ULDM_SUFFIXES = ["log10_alpha", "rho_f", "gamma_f"]
RN_SUFFIXES = ["red_noise_log10_A", "red_noise_gamma"]


def get_uldm_pnames(pta):
    out = []
    for suf in ULDM_SUFFIXES:
        matches = [p for p in pta.param_names if p.endswith(suf)]
        if len(matches) != 1:
            raise RuntimeError(f"sufijo '{suf}': {matches}")
        out.append(matches[0])
    return out


def get_rn_pnames(pta):
    out = []
    for suf in RN_SUFFIXES:
        matches = [p for p in pta.param_names if p.endswith(suf)]
        if len(matches) != 1:
            raise RuntimeError(f"sufijo RN '{suf}': {matches}")
        out.append(matches[0])
    return out

# %% [6] CARGA DEL RUIDO: MAP
# cargamos los parametros de ruido y, salvo el red noise, los dejamos fijos a sus
# maximos a posteriori (MAP)


def load_noise_MAP(pta, chain_file, pars_file):
    chain_raw = np.loadtxt(chain_file)
    with open(pars_file) as fh:
        par_names = [l.strip() for l in fh if l.strip()]

    n_par = len(par_names)
    lnpost = chain_raw[:, n_par]
    imap = int(np.argmax(lnpost))
    d = dict(zip(par_names, chain_raw[imap, :n_par]))

    for k in list(d.keys()):
        if k.endswith("_log10_equad"):
            d[k[:-len("_log10_equad")] + "_log10_t2equad"] = d.pop(k)

    uldm = get_uldm_pnames(pta)
    noise_pnames = [p for p in pta.param_names if p not in uldm]
    map_noise = {k: d[k] for k in noise_pnames if k in d}
    missing = set(noise_pnames) - set(map_noise)
    missing = {p for p in missing if not p.endswith('_dmjump')}  # from the .par
    if missing:
        print(
            f"    [WARN] faltan {len(missing)} params de ruido: {sorted(missing)[:3]}")
    return map_noise

# %% [7] CLASE MCMC: 5D POR PULSAR, PRIOR UNIFORME-EN-ALPHA


class PulsarULDM_RN(object):
    """
    MCMC 5D de un pulsar con ruido rojo marginalizado:
      x = (log10_alpha, log10_rho, gamma_f, log10_A_red, gamma_red)
    Ruido blanco y DM wideband fijos en MAP. Prior uniforme-en-alpha: ln p(a)=a ln10.
    """

    def __init__(self, d):
        self.d = d
        self.pta = d["pta"]
        self.fix = d["noise_norn"]
        self.la, self.rf, self.gf = d["la"], d["rf"], d["gf"]
        self.arn, self.grn = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim = 5

    # este es el objeto de enterprise que permite evaluar la likelihood dados los
    # 5 parametros en un momento cualquiera.
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
        # prior en a tras hacer: p(alpha) * dalpha = p(a) * da
        # [BF] con PRIOR_ACOPLE=loguniforme el prior es plano en a=log10(alpha)
        lp = x[0] * np.log(10.0) if PRIOR_ACOPLE == "uniforme" else 0.0
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho = 10.0 ** u
        lp += np.log(2.0 * ln10) + 2.0 * u * ln10 - \
            rho * rho   # Rayleigh en u=log10 rho
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
        x[3] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN +
                             0.1, LOG10_ARED_MAX - 0.1))
        x[4] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN +
                             0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10A**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
                        SIGMA_LOG10_ARED**2, SIGMA_GAMMA_RED**2])

# %% [8] PRELOAD + RUNNER + ESS


def preload_one(name, m_eV):
    m_dm = m_eV * EV_TO_INV_S
    t_load = time.time()
    try:
        par, tim, chain, pars = resolve_paths(name)
        psr, tm = load_pulsar(par, tim, convert_low_ecc=True)
        t_arr, partials, orb_pars, missing = build_context(psr, tm)
        if missing:
            # canal en cero = menos sensibilidad sin avisar, asi que avisa
            print(f"  [{name}] [WARN] sin columna en la design matrix: {missing} "
                  f"-> esos canales entran en cero", flush=True)
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        la, rf, gf = get_uldm_pnames(pta)
        arn, grn = get_rn_pnames(pta)
        noise = load_noise_MAP(pta, chain, pars)
        noise.update(dmjumps_par(par, psr.name, pta))
        _sin_valor = sorted(set(pta.param_names) - set(get_uldm_pnames(pta)) - set(noise))
        if _sin_valor:
            raise RuntimeError(f"{psr.name}: parametros de ruido sin valor: {_sin_valor}")
        if arn not in noise or grn not in noise:
            raise RuntimeError(
                f"RN MAP ausente: arn_in_noise={arn in noise}, grn_in_noise={grn in noise}")
        rn_map = (noise[arn], noise[grn])
        noise_norn = {k: v for k, v in noise.items() if k not in (arn, grn)}
        return dict(ctx=(t_arr, partials, orb_pars, m_dm), uldm_resto={rf: 1.0, gf: 0.0},
                    name=name, pta=pta, la=la, rf=rf, gf=gf, arn=arn, grn=grn,
                    noise_norn=noise_norn, rn_map=rn_map,
                    ecc=orb_pars["ecc"], n_toa=len(t_arr), t_load=time.time()-t_load)
    except Exception as e:
        print(f"  [{name}] PRELOAD FAIL: {type(e).__name__}: {e}", flush=True)
        return None


def effective_sample_size(x):
    """ESS por autocorrelacion: si la cadena es lo bastante larga."""
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
    from PTMCMCSampler import PTMCMCSampler
    os.makedirs(outDir, exist_ok=True)
    x0 = model.x0(rng)
    cov = model.cov0()
    _kw = {} if comm is None else {"comm": comm}
    sampler = PTMCMCSampler.PTSampler(
        model.ndim, model.lnlikelihood, model.lnprior, np.copy(cov),
        outDir=outDir, resume=False, **_kw)
    nburn = int(BURN_FRAC * n_iter)
    sampler.sample(x0, n_iter, Tmax=1, covUpdate=COV_UPDATE,
                   SCAMweight=SCAM_WEIGHT, AMweight=AM_WEIGHT, DEweight=DE_WEIGHT,
                   thin=1, isave=10_000, writeHotChains=False)
    files = sorted(glob.glob(os.path.join(outDir, "chain_1*.txt")))
    if not files:
        raise RuntimeError(f"no se encontro chain en {outDir}")
    data = np.loadtxt(files[0])
    return data[nburn:, 0]              # columna 0 = log10_alpha

# %% [9] Z_i(a) DESDE LA CADENA + COMBINACION


def lnZ_from_science_chain(alpha_samples):
    """
    La cadena se sampleo con prior uniforme-en-alpha, asi que su marginal es
    p_i(a) ~ 10^a Z_i(a)  ->  ln Z_i(a) = ln p_hat_i(a) - a ln10.
    Debajo del codo: extrapolacion plana. Arriba del drop: -inf.
    """
    ln10 = np.log(10.0)
    counts, _ = np.histogram(alpha_samples, bins=ALPHA_EDGES)
    dens = counts.astype(float)
    if SMOOTH_SIGMA_BINS > 0:
        dens = gaussian_filter1d(dens, SMOOTH_SIGMA_BINS, mode="nearest")
    # hago smooth el histograma con un filtro gaussiano
    lnZ = np.full(ALPHA_CENTERS.shape, -np.inf)
    dmax = dens.max()  # maximo de la campana
    if dmax <= 0:
        return lnZ

    # TAIL_FRAC es el valor en el cual considero que se "apago" la distribucion, 0.05 del maximo
    good = dens > TAIL_FRAC * dmax
    idx = np.where(good)[0]
    i_lo, i_hi = idx[0], idx[-1]
    sl = slice(i_lo, i_hi + 1)
    d_sl = np.maximum(dens[sl], TAIL_FRAC * dmax)
    lnZ[sl] = np.log(d_sl) - ALPHA_CENTERS[sl] * ln10
    n_anch = max(3, int(round(0.7 / DALPHA)))
    lnZ[:i_lo] = np.median(lnZ[i_lo:i_lo + n_anch])

    # [CORREGIDO] Cola superior: EXTRAPOLAR en vez de cortar con -inf. Antes,
    # arriba de i_hi quedaba lnZ = -inf ("Z = 0 exactamente") en una region donde
    # la cadena simplemente NO muestreo. Al COMBINAR pulsares (suma de lnZ_i) la
    # suma hereda el -inf en cuanto CUALQUIER sumando lo tiene => el soporte de
    # la posterior conjunta lo fija hasta donde llego la cadena del pulsar mas
    # restrictivo, no la likelihood. Para un resultado nulo Z ~ exp(-c a^2), o sea
    # lnZ = a0 + b*alpha^2 con b<0: se ajusta esa forma en la cola bien muestreada
    # y se continua. Validado en el vectorial (mismo codigo): sesgo del corte
    # +0.117 dex, cambio tipico por pulsar +0.013 dex.
    if EXTRAPOLAR_COLA and i_hi + 1 < len(ALPHA_CENTERS):
        j0 = max(i_lo, i_hi - n_anch + 1)
        x_hi = ALPHA_CENTERS[i_hi]
        u_fit = 10.0 ** (2.0 * (ALPHA_CENTERS[j0:i_hi + 1] - x_hi))
        if len(u_fit) >= 2:
            b, a0 = np.polyfit(u_fit, lnZ[j0:i_hi + 1], 1)
            if b < 0:
                u_ex = 10.0 ** (2.0 * (ALPHA_CENTERS[i_hi + 1:] - x_hi))
                with np.errstate(over="ignore"):
                    lnZ[i_hi + 1:] = a0 + b * u_ex
    return lnZ


def posterior_from_lnZ(lnZ_total, grid=None):
    """Posterior conjunta sobre 'grid' (ALPHA_FINE por defecto), interpolando
    lnZ_total desde ALPHA_CENTERS. Devuelve (post, ul95)."""
    if grid is None:
        grid = ALPHA_FINE
    ln10 = np.log(10.0)
    fin = np.isfinite(lnZ_total)
    a_fin, z_fin = ALPHA_CENTERS[fin], lnZ_total[fin]
    lnZ_g = np.interp(grid, a_fin, z_fin)            # (abajo: plano)
    # arriba del drop: Z_i -> 0
    lnZ_g[grid > a_fin.max()] = -np.inf
    # prior uniforme-en-alpha, una vez
    ln_post = grid * ln10 + lnZ_g
    ln_post -= np.nanmax(ln_post)
    post = np.exp(ln_post)
    post[~np.isfinite(post)] = 0.0
    post = post / np.trapz(post, grid)
    cdf = cumulative_trapezoid(post, grid, initial=0.0)
    ul95 = float(np.interp(0.95, cdf, grid))
    return post, ul95


def combine_pulsars(lnZ_list):
    lnZ_total = np.sum(np.vstack(lnZ_list), axis=0)
    return posterior_from_lnZ(lnZ_total)

# %% [10] WORKER: 1 tarea = (1 masa, 1 pulsar)


def process_task(args):
    mass_idx, m_eV, name, n_iter, seed = args
    t0 = time.time()
    rng = np.random.default_rng(seed)
    d = preload_one(name, m_eV)
    if d is None:
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err="preload_failed")
    try:
        model = crear_modelo(d)
        outDir = os.path.join(CHAIN_ROOT, f"m{mass_idx:03d}", name)
        alpha = run_one_chain(model, outDir, n_iter, rng)
        lnZ = lnZ_from_science_chain(alpha)
        ul_sci = float(np.percentile(alpha, 95.0))
        ess = effective_sample_size(alpha)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  "
              f"ecc={d['ecc']:.1e}  UL95={ul_sci:6.2f}  "
              f"a=[{alpha.min():6.1f},{alpha.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True,
                    alpha=alpha, lnZ=lnZ, ul_sci=ul_sci, ess=ess,
                    a_min=float(alpha.min()), a_max=float(alpha.max()),
                    n_toa=d["n_toa"], ecc=d["ecc"], t_min=(time.time()-t0)/60.0)
    except Exception as e:
        import traceback
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err=f"{type(e).__name__}: {e}", tb=traceback.format_exc())

# %% [11] PLOTS


def plot_individuals(results, m_eV, outpng):
    n = len(results)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4*ncol, 3.3*nrow),
                             dpi=110, squeeze=False)
    for k, r in enumerate(results):
        ax = axes[k // ncol][k % ncol]
        alpha = np.asarray(r["alpha"])
        ul = r["ul_sci"]
        counts, _ = np.histogram(alpha, bins=ALPHA_EDGES)
        h = counts / max(counts.max(), 1)
        ax.bar(ALPHA_CENTERS, h, width=DALPHA, color="0.7", edgecolor="none",
               align="center", label=r"cadena = $p_i(a)\propto 10^a Z_i$")
        ax.axvline(ul, color="crimson", ls="--",
                   lw=1.3, label="UL95 (percentil)")
        alpha_phys = 10.0**ul * ALPHA_CONVERSION
        ax.set_title(f"{r['name']}   a$\\in$[{alpha.min():.1f}, {alpha.max():.1f}]\n"
                     f"UL95={ul:.2f}  $\\alpha$={alpha_phys:.1e}  ESS={r.get('ess',0):.0f}",
                     fontsize=8)
        ax.set_xlabel(r"$\log_{10}\alpha$", fontsize=8)
        ax.set_xlim(alpha.min() - 0.5, max(alpha.max(), ul) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls="--", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Posteriors INDIVIDUALES (uniforme-en-a, RN marg., m={m_eV:.0e} eV)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpng, dpi=140)
    plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_joint(post, ul95, names, m_eV, outpng):
    alpha_phys = 10.0**ul95 * ALPHA_CONVERSION
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
    ax.plot(ALPHA_FINE, post, color="darkgreen", lw=1.8)
    ax.fill_between(ALPHA_FINE, post, where=(ALPHA_FINE <= ul95),
                    color="mediumseagreen", alpha=0.35, label="95% de la masa")
    ax.axvline(ul95, color="crimson", ls="--", lw=1.6,
               label=f"95% UL: log10(alpha)={ul95:.2f}\nalpha={alpha_phys:.2e} GeV$^{{-1}}$")
    ax.set_xlabel(r"$\log_{10}\alpha$", fontsize=13)
    ax.set_ylabel(r"posterior conjunto $P(\log_{10}\alpha)$", fontsize=13)
    ax.set_title(f"COTA CONJUNTA ({len(names)} pulsares, RN marg., m={m_eV:.0e} eV)",
                 fontsize=11)
    m = post > post.max() * 1e-3
    ax.set_xlim(ALPHA_FINE[m].min() - 1, ALPHA_FINE[m].max() + 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_vs_mass(m_arr, alpha_arr, n_used, outpng):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.loglog(m_arr, alpha_arr, "-", color="black", lw=1.4)
    ax.set_xlabel(r"$m$ [eV]", fontsize=12)
    ax.set_ylabel(
        r"$\alpha$ [GeV$^{-1}$]  (cota conjunta, 95% UL)", fontsize=12)
    ax.set_title(
        f"Cota CONJUNTA vs masa  ({int(np.max(n_used))} pulsares, RN marg.)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(outpng, dpi=140)
    plt.close(fig)
    print(f"  guardado: {outpng}")


# %% [12] MAIN
def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    single = len(MASS_GRID_EV) == 1
    print("\n" + "=" * 70)
    print(f"COTA CONJUNTA POR-PULSAR (alpha LINEAL, RN marginalizado) | "
          f"{'1 masa' if single else f'{len(MASS_GRID_EV)} masas'}  {ALLOWED_MODELS}")
    print("=" * 70)
    t_all = time.time()

    # 1) pulsares
    print("\n[1] Descubriendo y filtrando pulsares...")
    all_names = discover_wb_pulsars()
    use = filter_pulsars_by_model(all_names, allowed=ALLOWED_MODELS)[
        :N_PULSARS_TARGET]
    if not use:
        raise SystemExit(f"No se encontraron pulsares {ALLOWED_MODELS}.")
    print(f"    USANDO {len(use)} pulsares: {use}")

    # 2) tareas planas (masa x pulsar)
    tasks = []
    for mi, m_eV in enumerate(MASS_GRID_EV):
        for pi, name in enumerate(use):
            tasks.append((mi, float(m_eV), name, N_ITER, SEED + mi*1000 + pi))
    total = len(tasks)
    n_workers = MAX_WORKERS or (os.cpu_count() or 1)
    print(f"\n[2] {total} tareas (= {len(MASS_GRID_EV)} masas x {len(use)} pulsares), "
          f"{n_workers} procesos (cpu={os.cpu_count()})")
    print(
        f"    Cada tarea = 1 MCMC 5D de {N_ITER} iters. ETA sobre tareas completadas.\n")

    # 3) lanzar y combinar cada masa cuando llegan todos sus pulsares
    mass_results = defaultdict(list)   # mass_idx -> [result dicts ok]
    mass_received = defaultdict(int)
    joint = {}                  # mass_idx -> dict(ul95, alpha, n_used, post)
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process_task, a) for a in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            mi = r["mass_idx"]
            done += 1
            mass_received[mi] += 1
            if r.get("ok"):
                mass_results[mi].append(r)
            else:
                print(
                    f"  [m#{mi:03d} {r['name']}] ERROR: {r.get('err')}", flush=True)
                if "tb" in r:
                    print(r["tb"], flush=True)

            el = time.time() - t_start
            rate = done / el if el > 0 else 0.0
            eta = (total - done) / rate if rate > 0 else float("nan")
            if done % max(1, len(use)) == 0 or done == total:
                print(f"  [{done:5d}/{total}] {100.0*done/total:5.1f}%   "
                      f"elapsed {el/60:6.1f} min   ETA {eta/60:6.1f} min", flush=True)

            if mass_received[mi] == len(use):
                lst = mass_results[mi]
                if lst:
                    post, ul95 = combine_pulsars([x["lnZ"] for x in lst])
                    alpha_phys = 10.0 ** ul95 * ALPHA_CONVERSION
                    joint[mi] = dict(ul95=ul95, alpha=alpha_phys,
                                     n_used=len(lst), post=post)
                    print(f"   >>> masa #{mi:03d}  m={MASS_GRID_EV[mi]:.3e} eV   "
                          f"COTA CONJUNTA log10(alpha)={ul95:.3f}  alpha={alpha_phys:.3e} GeV^-1   "
                          f"({len(lst)}/{len(use)} pulsares)", flush=True)
                else:
                    print(
                        f"   >>> masa #{mi:03d}: sin pulsares OK, se omite.", flush=True)

    if not joint:
        raise SystemExit("Ninguna masa con cota; revisar errores de arriba.")

    # 4) ensamblar + guardar
    idx_ok = sorted(joint.keys())
    m_arr = np.array([MASS_GRID_EV[i] for i in idx_ok])
    ul_arr = np.array([joint[i]["ul95"] for i in idx_ok])
    alpha_arr = np.array([joint[i]["alpha"] for i in idx_ok])
    nused_arr = np.array([joint[i]["n_used"] for i in idx_ok])

    npz = os.path.join(OUT_ROOT, "DEFINITIVO_alpha_massweep_joint.npz")
    np.savez(npz, masses_eV=m_arr, log10_alpha_95UL_joint=ul_arr,
             alpha_95UL_joint=alpha_arr, n_pulsars_used=nused_arr,
             names=np.array(use), alpha_centers=ALPHA_CENTERS, alpha_fine=ALPHA_FINE)
    print(f"\n[3] SAVE {npz}")

    # 5) plots
    print(f"\n[4] Plots...")
    if single:
        mi = idx_ok[0]
        lst = sorted(mass_results[mi], key=lambda r: use.index(r["name"]))
        print(
            f"\n  ---- UL95 individual por pulsar (m={m_arr[0]:.2e} eV) ----")
        print(f"    {'pulsar':15s} {'n_toa':>6s} {'ecc':>8s} {'UL95':>8s} "
              f"{'alpha':>11s} {'ESS':>8s}")
        for r in lst:
            ap = 10.0**r["ul_sci"] * ALPHA_CONVERSION
            flag = "  <-- a_max bajo?" if r["a_max"] < r["ul_sci"] - 0.3 else ""
            print(f"    {r['name']:15s} {r['n_toa']:6d} {r['ecc']:8.1e} "
                  f"{r['ul_sci']:8.3f} {ap:11.2e} {r['ess']:8.0f}{flag}")
        print(f"\n  ====== COTA CONJUNTA ({nused_arr[0]} pulsares) ======")
        print(f"    log10_alpha_95UL = {ul_arr[0]:.3f}")
        print(f"    alpha_95UL       = {alpha_arr[0]:.3e} GeV^-1")
        plot_individuals(lst, m_arr[0], os.path.join(
            OUT_ROOT, "DEFINITIVO_alpha_individuales.png"))
        plot_joint(joint[mi]["post"], ul_arr[0], use, m_arr[0],
                   os.path.join(OUT_ROOT, "DEFINITIVO_alpha_conjunto.png"))
    else:
        plot_vs_mass(m_arr, alpha_arr, nused_arr,
                     os.path.join(OUT_ROOT, "DEFINITIVO_alpha_vs_m.png"))

    print("\n" + "=" * 70)
    print(f"[LISTO] {len(idx_ok)}/{len(MASS_GRID_EV)} masas con cota   "
          f"tiempo total {(time.time()-t_all)/60:.1f} min")
    print(f"  Archivos en {OUT_ROOT}")
    print("=" * 70)


if __name__ == "__main__":
    main()


# =============================================================================
# [FINAL] Configuracion comun y fabrica de modelos
# =============================================================================
# Likelihood de produccion: ENTERPRISE (pta.get_lnlikelihood).
LIKELIHOOD = "enterprise"
VERSION_PIPELINE = "final-2026-10-06"


def crear_modelo(d):
    return PulsarULDM_RN(d)
