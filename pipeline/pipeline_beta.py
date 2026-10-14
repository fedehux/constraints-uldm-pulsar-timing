# =============================================================================
# =============================================================================
# VERSION DEFINITIVA. Calcula la COTA CONJUNTA de beta combinando cadenas
# MCMC POR-PULSAR con ruido rojo marginalizado, sobre un barrido en masa de
# 200 puntos entre 1e-23 y 1e-18 eV
#
# Por cada (masa, pulsar) samplea con PTMCMC 5 parametros:
# (log10_beta, log10_rho, gamma_f, log10_A_red, gamma_red)
# la senal ULDM (beta, rho, gamma) + el ruido rojo acromatico (A_red,
# gamma_red). El ruido blanco (EFAC/EQUAD) y el DM wideband quedan FIJOS en MAP.
# Con las cadenas estima Z_i(b) y combina:  ln p(b|D) = b ln10 + sum_i ln Z_i(b).
# =============================================================================
# =============================================================================

#%% [PIN] 1 HILO POR PROCESO
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

#%% [QUIET] SILENCIAR LOGS
import warnings
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

#%% [0] IMPORTS
import glob
import time
from copy import deepcopy
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:                                          # scipy>=1.6
    from scipy.integrate import cumulative_trapezoid
except ImportError:                           # scipy<1.6
    from scipy.integrate import cumtrapz as cumulative_trapezoid
from scipy.special import jn, jvp
from scipy.ndimage import gaussian_filter1d
import astropy.units as u

print("[0] IMPORTS OK")

# =============================================================================
# CONFIGURACION  (todo lo que tipicamente se toca esta aca arriba)
# =============================================================================
# ---- Pulsares ----
ALLOWED_MODELS   = ("ELL1", "ELL1H", "ELL1k", "DD", "BT", "DDK")
# BT y DDK son de familia kepleriana: BT (J1312+0051) y DDK (J1713+0747) tienen
# ECC/OM/T0; ambos son de BAJA excentricidad (2.4e-4 y 7.5e-5) -> load_pulsar los
# convierte a ELL1 (convert_to_ell1) y usan el template ELL1. 

N_PULSARS_TARGET = 999          # 999 = usar TODOS los que matcheen ALLOWED_MODELS

# ---- Barrido en masa ----
# [v3] Esta es la grilla BASE (envolvente). La de PRODUCCION (base + zoom
# resonante de los excentricos) la arma grid_masas_beta.construir_grid(); el
# worker y el aggregate usan ESA, no MASS_GRID_EV. Aca queda la base para que
# main() (modo standalone, sin cluster) siga funcionando.
# Mismo rango que beta2 (1e-23..1e-18, 40 masas/decada) para que la curva nueva
# sea comparable punto a punto con la vieja.
MASS_MIN_EV   = float(os.environ.get("MASA_MIN_EV", "1e-23"))
MASS_MAX_EV   = float(os.environ.get("MASA_MAX_EV", "1e-18"))
N_MASS        = int(os.environ.get("N_BASE", "200"))   # 40/decada, 5 decadas
SINGLE_MASS_EV = None           # si != None (p.ej. 1e-23) corre SOLO esa masa


# ---- MCMC ----
N_ITER     = 200000            
MAX_WORKERS = None              # None -> cpu_count()
SEED        = 60

# ---- Modos especiales ----
VERIFY   = False                # True -> chequeo numerico de ecuaciones
SELFTEST = False                # True -> valida la combinacion cadena->Z_i SIN datos

#%% [1] PATHS Y CONSTANTES FISICAS
# Rutas: se leen de variables de entorno (para DIRAC/cluster); si no estan
# definidas, usan las rutas locales de Windows (corrida local sin cambios).
#   NANOGRAV_ROOT -> carpeta que contiene 'wideband/{par,tim,noise}'
#   OUT_ROOT      -> donde se escriben cadenas/plots/npz

NANOGRAV_ROOT = os.environ.get(
    "NANOGRAV_ROOT", "C:/Users/fedeh/OneDrive/Documents/C/NANOGrav15yr_PulsarTiming_v2.1.0")
WB_ROOT       = os.path.join(NANOGRAV_ROOT, "wideband")
OUT_ROOT      = os.environ.get(
    "OUT_ROOT", r"C:\Users\fedeh\OneDrive\Documents\C\Versión definitiva")
CHAIN_ROOT    = os.path.join(OUT_ROOT, "chains_RNmarg")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(CHAIN_ROOT, exist_ok=True)

# Constantes numéricas

EV_TO_INV_S     = 1.51926760385984e+15
YR_IN_S         = 3.15576e7
DAY_IN_S        = 86400.0
DM_DENSITY      = 0.3 * 4.09351499910375e+55
UNIT_CONVERSION = 2.30817405213802e+48

N_TERMS         = 20 # Términos del desarrolo de Bessel
N_FREQ_RN       = 30 # Número de modos del red noise

LOG10_BETA_MIN = float(os.environ.get("LOG10_BETA_MIN", "-92.0"))
LOG10_BETA_MAX = float(os.environ.get("LOG10_BETA_MAX", "-60.0"))   # Región en log y en unidades naturales
                                                # donde samplear beta

# [BF] Prior del acople. "uniforme" (default) = uniforme en beta, el de las
# cotas. "loguniforme" = uniforme en log10(beta), el de la busqueda de senal
# (Bayes factor). Solo cambia el termino de beta en lnprior.
PRIOR_ACOPLE = os.environ.get("PRIOR_ACOPLE", "uniforme").strip().lower()
if PRIOR_ACOPLE not in ("uniforme", "loguniforme"):
    raise ValueError(f"PRIOR_ACOPLE={PRIOR_ACOPLE!r}: uniforme|loguniforme")

RHO_F_MIN     = 1.0e-3
RHO_F_MAX     = 6.0         # pongo límites sobre la amplitud rho para definir la variable en enterprise
LOG10_RHO_MIN = np.log10(RHO_F_MIN)
LOG10_RHO_MAX = np.log10(RHO_F_MAX)
ECC_LOW_THRESHOLD = 0.01 
# valor de excentricidad a partir del cual se considera que un sistema es circular

# [CORREGIDO] Antes, la resonancia 2m = n*wb (excentrico) se manejaba con un
# umbral RES_REL_TOL: si |n*wb-2m| caia por debajo de esa tolerancia se
# activaba "a mano" el limite analitico L'Hopital. Ahora la rama "minus" del
# template DD se escribe con una identidad trigonometrica exacta (rama con
# sinc, igual que ya se hacia en el ELL1 circular) que vale para CUALQUIER
# om_col, incluido om_col=0, sin necesidad de umbral ni rama aparte.

# ---- Trigger de la correccion de resonancia (canal PB), analogo al lineal ----
# En la integral de la rama minus del canal PB (rm_int, 2m = n*wb) aparece la
# resta cruda  cos_C T~/om - (sin_minus+sin_C)/om^2  que es 0/0 en la resonancia
# exacta. Con True se usa la forma cerrada estable (sinc), error < 1e-9 en el
# pico; con False se usa la resta cruda ("ingenua"), que ahi da NaN. A
# diferencia del lineal el default aca es True: este pipeline no tiene una
# version previa de la directora que reproducir (el daae cuadratico es
# construccion nuestra), asi que por defecto corre ya estable.
# barrido_resonante_beta.py lo controla via CORREGIR_RESONANCIA.
RESONANCIA_ESTABLE = True

# ---- MCMC (priors / arranque) ----
BURN_FRAC       = 0.30 # quema el 30% inicial de la cadena
LOG10_BETA_INIT = -78.0
COV_UPDATE      = 2000 # actualiza la covarianza cada 2000 pasos
SCAM_WEIGHT     = 60 # parámetros óptimos para la cadena según testeamos antes, por prueba y error
AM_WEIGHT       = 20
DE_WEIGHT       = 20
SIGMA_LOG10B    = 3.0
SIGMA_LOG10RHO  = 0.5
SIGMA_GAMMA     = 2 * np.pi

# ---- Red noise (marginalizado): priors y escalas ----
# Usamos lo mismo que en los papers de constraints de PTA

LOG10_ARED_MIN, LOG10_ARED_MAX = -20.0, -11.0
GAMMA_RED_MIN,  GAMMA_RED_MAX  = 0.0, 10.0
SIGMA_LOG10_ARED = 1.5 
SIGMA_GAMMA_RED  = 1.5

# ---- Binning para Z_i(b) y combinación ----
# Para combinar eficientemente las likelihoods Z_i de cada sistema, no usamos los puntos de
# la cadena en sí. Generamos un histograma de la posterior final, el cual representa la 
# likelihood, ya que es más "limpio" a la hora de combinar sistemas. 

N_BETA_BINS       = int(round((LOG10_BETA_MAX - LOG10_BETA_MIN) / 0.005))   # [FINAL] 0.005 dex/bin, como alpha
BETA_EDGES        = np.linspace(LOG10_BETA_MIN, LOG10_BETA_MAX, N_BETA_BINS + 1)
BETA_CENTERS      = 0.5 * (BETA_EDGES[:-1] + BETA_EDGES[1:])
DBETA             = BETA_EDGES[1] - BETA_EDGES[0]
SMOOTH_SIGMA_BINS = 6.0   # [FINAL] 0.03 dex, como alpha
TAIL_FRAC         = 0.05
# Extrapolar la cola superior de lnZ (ajuste Z~exp(-c beta^2)) en vez de cortar
# con -inf arriba del ultimo bin muestreado. Ver lnZ_from_science_chain: sin esto,
# al COMBINAR pulsares el soporte conjunto lo fija hasta donde llego la cadena del
# pulsar dominante (cota optimista y UL95 pegado al borde). False = comportamiento
# viejo (reproduce numero por numero las corridas previas).
EXTRAPOLAR_COLA   = True

# Grilla para la posterior conjunta final, desacoplada del histograma de Z_i.
# Z_i(b) se estima en BETA_CENTERS (histograma robusto, no se toca); la posterior
# final se evalua interpolando lnZ_total a esta grilla fina.

N_BETA_FINE       = int(round((LOG10_BETA_MAX - LOG10_BETA_MIN) / 0.002)) + 1
BETA_FINE         = np.linspace(LOG10_BETA_MIN, LOG10_BETA_MAX, N_BETA_FINE)

if SINGLE_MASS_EV is not None:
    MASS_GRID_EV = np.array([float(SINGLE_MASS_EV)])
else:
    MASS_GRID_EV = np.logspace(np.log10(MASS_MIN_EV), np.log10(MASS_MAX_EV), N_MASS)

print(f"[1] PATHS OK   masas BASE={len(MASS_GRID_EV)} "
      f"({MASS_GRID_EV[0]:.2e}..{MASS_GRID_EV[-1]:.2e} eV)   N_ITER={N_ITER}")
print(f"    modelos={ALLOWED_MODELS}  N_target={N_PULSARS_TARGET}")
print(f"    RN priors: log10_A_red~U({LOG10_ARED_MIN},{LOG10_ARED_MAX}), "
      f"gamma_red~U({GAMMA_RED_MIN},{GAMMA_RED_MAX})")

#%% [2] DETECCION / FILTRADO DE PULSARES
def resolve_paths(psr_name):
    par_hits = sorted(glob.glob(os.path.join(WB_ROOT, "par", f"{psr_name}_PINT_*.wb.par")))
    tim_hits = sorted(glob.glob(os.path.join(WB_ROOT, "tim", f"{psr_name}_PINT_*.wb.tim")))
    if not par_hits or not tim_hits:
        raise FileNotFoundError(f"{psr_name}: par={par_hits} tim={tim_hits}")
    return par_hits[0], tim_hits[0], \
        os.path.join(WB_ROOT, "noise", f"{psr_name}.wb.chain_1.txt"), \
        os.path.join(WB_ROOT, "noise", f"{psr_name}.wb.pars.txt")


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
    pat     = os.path.join(WB_ROOT, "par", "*_PINT_*.wb.par")
    names   = set()
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
            skip.append((n, f"sin-archivos:{type(e).__name__}")); continue
        if model in allowed:
            keep.append(n)
        else:
            skip.append((n, str(model)))
    print(f"  [FILTRO {'/'.join(allowed)}] {len(keep)} pulsares OK, {len(skip)} descartados")
    return keep

#%% [3] TEMPLATE ULDM
# --- Helpers estables para las ramas resonantes y sus integrales -------------
# En el caso cuadratico la resonancia es 2m = n*wb (el acoplamiento entra como
# beta*Phi^2/2, que oscila al DOBLE de la frecuencia de Phi), asi que el
# denominador que se anula es (n wb - 2m). Todo lo que lo lleva se escribe con
# sinc, sin restas de numeros grandes ni divisiones por algo que tienda a cero.

def _one_minus_cos_over_z2(z):
    """(1 - cos z)/z^2, exacto y estable (vale 1/2 en z=0)."""
    # 1 - cos z = 2 sin^2(z/2)  =>  (1-cos z)/z^2 = (1/2) sinc(z/2pi)^2
    s = np.sinc(z / (2.0 * np.pi))
    return 0.5 * s * s


def _z_minus_sin_over_z2(z):
    """
    (z - sin z)/z^2, estable (vale 0 en z=0, crece como z/6).

    Para |z| chico la resta z - sin z pierde todas las cifras significativas,
    asi que ahi se usa la serie z/6 - z^3/120 + z^5/5040 - z^7/362880 +
    z^9/39916800, que a |z| = 0.1 ya coincide con la formula directa a nivel de
    epsilon de maquina; fuera de ese entorno la resta directa no cancela.
    """
    z = np.asarray(z, dtype=np.float64)
    chico = np.abs(z) < 0.1
    z_seguro = np.where(chico, 1.0, z)          # evita 0/0 en la rama directa
    directa = (z_seguro - np.sin(z_seguro)) / (z_seguro * z_seguro)
    z2 = z * z
    serie = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(chico, serie, directa)


def _mean_cos2(m_dm, t_arr, t_ref, gamma_f, tt):
    """
    Promedio temporal de cos^2(m t + Y) entre t_ref y t:
        (1/t~) int_{t_ref}^{t} cos^2(m t' + Y) dt'
            = 1/2 + (1/2) cos(m(t+t_ref) + 2Y) sinc(m t~/pi).

    Escrito asi no divide ni por t~ ni por m: en t = t_ref vale
    cos^2(m t_ref + Y), que es el limite correcto, en vez de 0/0.
    """
    return 0.5 + 0.5 * np.cos(m_dm * (t_arr + t_ref) + 2.0 * gamma_f) * \
        np.sinc(m_dm * tt / np.pi)


def _int_cos(omega, phase, t_arr, t_ref):
    """
    Integral analitica int_{t_ref}^{t} cos(omega t' + phase) dt', estable en
    omega=0 (resonancia) sin ifs. Con la identidad sin A - sin B = 2 cos((A+B)/2)
    sin((A-B)/2):
        int = t~ * cos(omega*(t+t_ref)/2 + phase) * sinc(omega t~ / 2pi),
    con t~ = t - t_ref. En omega->0 sinc->1 y queda t~*cos(phase) (el limite
    secular correcto), sin dividir por omega. omega/phase pueden ser columnas
    (una fila por armonico n); t_arr/t_ref son el eje temporal.
    """
    tt  = (t_arr - t_ref)[None, :]
    mid = omega * (0.5 * (t_arr + t_ref))[None, :] + phase
    return tt * np.cos(mid) * np.sinc(omega * tt / (2.0 * np.pi))


def compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """Selector: ELL1 (baja ecc) vs DD (excentrico)."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)

    # ---- TEMPLATE DD (excentrico) ----
    n_vec   = np.arange(1, N_TERMS)
    n_col   = n_vec[:, None]
    jn_col  = jn (n_vec, n_vec * orb_pars["ecc"])[:, None]
    jnp_col = jvp(n_vec, n_vec * orb_pars["ecc"])[:, None]

    # cargo directo los parámetros orbitales del best-fit de NANOGrav
    wb       = orb_pars["omega_b"]
    ecc      = orb_pars["ecc"]
    t0_s     = orb_pars["T0_s"]
    PB_s     = orb_pars["PB_s"]
    A1_s     = orb_pars["A1_s"]
    sqrt1me2 = np.sqrt(1.0 - ecc**2)

    T_tilde   = t_arr - t0_s
    tt_row    = T_tilde[None, :]
    nw        = n_vec * wb
    nw_col    = nw[:, None]
    cos_nw_tt = np.cos(nw_col * tt_row)
    sin_nw_tt = np.sin(nw_col * tt_row)

    nw_over_4        = n_vec * wb / 4.0
    delta_T0_an_base = 4.0 * jn_col - 2.0 * sqrt1me2 / ecc * n_col * jnp_col
    delta_T0_bn      = (4.0 / n_col / wb * jn_col + 4.0 / ecc / wb * sqrt1me2 * jnp_col)

    two_m         = 2.0 * m_dm
    op_col        = (nw + two_m)[:, None]   # n wb + 2m: denominador rama "plus", nunca resuena
    om_col        = (nw - two_m)[:, None]   # n wb - 2m: denominador rama "minus", resuena en 2m=n*wb
    op_tt         = op_col * tt_row
    om_tt         = om_col * tt_row
    phi0_const_sq = 2.0 * DM_DENSITY / m_dm**2
    m_t           = m_dm * t_arr
    m_t0          = m_dm * t0_s
    coeff_OM      = (nw_over_4 - m_dm)[:, None]
    coeff_ECC_p   = (m_dm - nw_over_4)[:, None]
    coeff_ECC_m   = (m_dm + nw_over_4)[:, None]

    phase_C   = 2.0 * m_t0 + 2.0 * gamma_f
    cos_C     = np.cos(phase_C)
    sin_C     = np.sin(phase_C)
    cos_plus  = np.cos(op_tt + phase_C)
    sin_plus  = np.sin(op_tt + phase_C)
    # (cos_minus/sin_minus ya no hacen falta: la rama "minus" y su integral se
    #  calculan con las formas estables de abajo, no con la resta cruda.)

    # rama "plus": el denominador op_col = n wb + 2m > 0 siempre, nunca da 0/0.
    rp = (cos_C - cos_plus) / op_col

    # [CORREGIDO] rama "minus", robusta con sinc (mismo truco que _stable_minus
    # del template lineal ya corregido): la identidad trigonometrica
    #   (cos_C - cos_minus)/om_col = T~ * sin(mid_m) * sinc(om_col T~ / 2pi)
    # vale EXACTAMENTE para todo om_col, incluido om_col=0 (donde antes se
    # necesitaba el umbral RES_REL_TOL + una rama L'Hopital aparte): al hacer
    # om_col->0, sinc->1 y queda rm -> -T~ sin_C, el mismo limite de antes,
    # pero sin discontinuidad ni umbral que ajustar a mano.
    mid_m   = 0.5 * om_tt - phase_C
    sincv_m = np.sinc(om_col * tt_row / (2.0 * np.pi))
    rm      = tt_row * np.sin(mid_m) * sincv_m

    phi_sq = phi0_const_sq * rho_f**2

    term1 = (1.0 - cos_nw_tt) / (2.0 * nw_col) + 0.25 * (rp + rm)
    part1 = -2.0 * wb * phi_sq * np.sum(n_col * jn_col * term1, axis=0)
    cos_mt   = np.cos(m_t  + gamma_f)
    cos_mt0  = np.cos(m_t0 + gamma_f)
    sin_2mt  = np.sin(2.0 * m_t + 2.0 * gamma_f)   # = sin(fase evaluada en t en vez de t0)
    part2    = -phi_sq * (cos_mt**2 - cos_mt0**2)
    part3    = -2.0 * m_dm * phi_sq * np.sum(jn_col * (-rp + rm), axis=0)
    daa = part1 + part2 + part3 # delta a / a (INSTANTANEO: para el canal A1)

    # ---- [CORREGIDO] delta a / a EFECTIVO (promedio temporal de la Ec. C.1b
    # entre T0 y t) ----
    # Igual razonamiento que en el template lineal: PB no se mide instante a
    # instante sino a traves de la fase orbital acumulada desde T0 hasta cada
    # TOA, asi que el forzante correcto del canal PB es el PROMEDIO temporal de
    # delta a/a, no su valor puntual (eso es lo que se corrige aca; el canal A1
    # sigue usando el "daa" instantaneo de arriba, sin cambios).
    #
    # Cada pieza de daa se reemplaza por su integral analitica entre T0 y t
    # (sin normalizar todavia: la division por T~ es al final, "daae"). La
    # pieza armonica (1-cos(nw t~))/(2nw) integra directo, sin singularidad. rp
    # integra directo (op_col nunca es 0). rm integra a
    #   rm_int = cos_C T~/om_col - (sin_minus+sin_C)/om_col^2,
    # que tiene una cancelacion 0/0 EXACTA en om_col=0 (el limite analitico es
    # finito: -T~^2 sin_C/2, el mismo orden que rm->-T~ sin_C en el caso
    # instantaneo). Escrita tal cual, esa resta pierde precision (y da NaN
    # justo en la resonancia, 0/0 en punto flotante) porque los dos terminos
    # divergen por separado antes de cancelarse. Se verifico numericamente
    # (integrando "rm" por fuerza bruta) que cerca de om_col=0 la resta directa
    # ya pierde ~7 cifras a una distancia relativa de 1e-7 de la resonancia; por
    # eso, a diferencia del template lineal (donde Sm_over/sigma_m se deja
    # como resta directa porque ahi es de un solo orden), aca se usa una serie
    # de Taylor en z=om_col*T~ para |z|<1e-2 -- coeficientes verificados contra
    # la integral numerica a <1e-9 de error relativo en todo el rango,
    # incluyendo la resonancia exacta.
    harm_int = tt_row / (2.0 * nw_col) - np.sin(nw_col * tt_row) / (2.0 * nw_col**2)
    rp_int = cos_C * tt_row / op_col - (sin_plus - sin_C) / op_col**2

    # Forma cerrada exacta de la integral de rm, sin umbrales:
    #   int_{T0}^{t} rm dt' = T~^2 [ cos_C (z - sin z)/z^2 - sin_C (1 - cos z)/z^2 ]
    # con z = om_col T~. Los dos cocientes son REGULARES en z = 0 (valen 0 y 1/2)
    # y estan implementados arriba de forma estable: (1-cos z)/z^2 es exacto via
    # la identidad con sinc^2, y (z - sin z)/z^2 usa serie hasta z^9 por debajo
    # de |z| = 0.1. Reemplaza a la resta cruda cos_C T~/om - (sin_minus+sin_C)/om^2,
    # que diverge termino a termino y da NaN (0/0) en la resonancia exacta.
    z_res  = om_tt                                   # = om_col * T_tilde
    if RESONANCIA_ESTABLE:
        rm_int = tt_row**2 * (cos_C * _z_minus_sin_over_z2(z_res)
                              - sin_C * _one_minus_cos_over_z2(z_res))
    else:
        # forma "ingenua": matematicamente igual, pero 0/0 en la resonancia
        sin_minus = np.sin(om_tt - phase_C)
        rm_int = cos_C * tt_row / om_col - (sin_minus + sin_C) / om_col**2

    part1_int = -2.0 * wb * phi_sq * np.sum(
        n_col * jn_col * (harm_int + 0.25 * (rp_int + rm_int)), axis=0)
    part2_int = -phi_sq * ((0.5 - cos_mt0**2) * T_tilde
                           + (sin_2mt - sin_C) / (4.0 * m_dm))
    part3_int = -2.0 * m_dm * phi_sq * np.sum(jn_col * (-rp_int + rm_int), axis=0)

    # daae = <delta a/a>. Todas las integrales de arriba se anulan como T~ en
    # t = T0, asi que el limite es daa(T0) = 0; se pone explicito para no hacer
    # 0/0 si algun TOA cae justo en T0.
    int_total = part1_int + part2_int + part3_int
    T_no_cero = np.where(np.abs(T_tilde) > 0.0, T_tilde, 1.0)
    daae = np.where(np.abs(T_tilde) > 0.0, int_total / T_no_cero, 0.0)

    # ---- Delta Omega ----

    # [CORREGIDO 2026-09] Antes se implementaba la Ec. (B4) del paper de
    # sensitivity (Huxhagen & Lopez Nacir 2026, PRD 114, 043056) tal como esta
    # publicada, y esa ecuacion tiene DOS errores. Contrastado contra la
    # cuadratura directa de la Ec. (22d) del mismo paper, la version publicada
    # erraba entre 90% y 114% (m de 5e-23 a 1e-21 eV, e de 0.03 a 0.44); la
    # forma de abajo coincide a 1e-8..7e-6.
    #
    # (i) Termino no resonante. La parte DC de Phi^2 multiplica a
    #     (a^2/r^2) cos(theta) = sum_n 2 n J_n'(ne) cos(n wb t~)   [Ec. A2g],
    #     cuya integral es sin(n wb t~)/(n wb). El resultado es sin(n wb t~)/2,
    #     NO [sin(n wb t) - sin(n wb T0)]/2, que es una funcion distinta de t~
    #     (arrastra una fase que depende de T0).
    #
    # (ii) Terminos resonantes. En delta_a y delta_e el factor orbital va como
    #     sin(n wb t~), asi que el producto con cos(2mt+2Y) da SENOS, que
    #     integran a COSENOS -> de ahi los C^{+-}_n del paper. Aca el factor
    #     orbital va como cos(n wb t~): el producto da COSENOS, que integran a
    #     SENOS. Corresponden entonces
    #        S^+_n = [sin((n wb + 2m) t~ + Y') - sin Y'] / (n wb + 2m)
    #        S^-_n = [sin((n wb - 2m) t~ - Y') + sin Y'] / (n wb - 2m)
    #     y no C^+_n, C^-_n. Ademas los coeficientes NO son los dos iguales:
    #     salen (n wb/4 - m) para la rama plus y (n wb/4 + m) para la minus,
    #     la misma estructura que en delta_a/a (Ec. B2), no (n wb/4 - m) en las
    #     dos como figura en (B4).
    #
    # S^-_n resuena en 2m = n wb y se escribe con sinc, igual que rm:
    #   [sin(z - Y') + sin Y'] / om = T~ cos(z/2 - Y') sinc(z/2pi),  z = om T~.
    sp_int   = (sin_plus - sin_C) / op_col                     # S^+_n
    sm_int   = tt_row * np.cos(0.5 * om_tt - phase_C) * sincv_m  # S^-_n (estable)
    partA_OM = 0.5 * np.sum(jnp_col * sin_nw_tt, axis=0)
    partB_OM = np.sum(jnp_col * (coeff_OM * sp_int
                                 + coeff_ECC_m * sm_int), axis=0)
    dOM      = phi_sq * sqrt1me2 / ecc * (partA_OM + partB_OM)

    # ---- Delta excentricidad ----

    partA_E  = -0.5 * np.sum(jn_col * (1.0 - cos_nw_tt), axis=0)
    partB_E  = np.sum(jn_col * (coeff_ECC_p * rp - coeff_ECC_m * rm), axis=0)
    dECC     = phi_sq * (1.0 - ecc**2) / ecc * (partA_E + partB_E)

    # ---- Delta T0 ----

    term_b = phi_sq / (4.0 * m_dm) * (sin_2mt - sin_C)

    # [CORREGIDO] term_c ANALITICO (antes cumtrapz sobre TOAs, que erraba en la
    # resonancia porque el integrando oscila a ~2m, mas rapido que el muestreo).
    # El integrando por armonico n era
    #   A_n phi^2 cos^2(mt+Y) cos(nw t~) + 2 B_n (-m phi^2 cos(mt+Y)sin(mt+Y)) sin(nw t~),
    # con A_n=delta_T0_an_base, B_n=delta_T0_bn, t~=t-T0. Usando
    #   cos^2 = 1/2 + 1/2 cos(2mt+2Y),  cos*sin = 1/2 sin(2mt+2Y),
    #   cos(f2)cos(psi)=1/2[cos(f2+psi)+cos(f2-psi)],
    #   sin(f2)sin(psi)=1/2[cos(f2-psi)-cos(f2+psi)],
    # queda una suma de 3 cosenos puros (psi=nw t~, f2±psi) que se integran con
    # _int_cos (forma cerrada con sinc, estable en la resonancia 2m=nw). El limite
    # inferior es T0, el mismo que el de term_b y term_d, asi que las tres piezas
    # de delta T0 son la integral del mismo intervalo [T0, t]. (Antes era t_arr[0],
    # heredado del cumtrapz con initial=0: la diferencia es una constante, que la
    # marginalizacion del timing model absorbe.)
    t_ref_c = t0_s
    two_g   = 2.0 * gamma_f
    I_psi   = _int_cos(nw_col,  -nw_col * t0_s,          t_arr, t_ref_c)  # nw t~
    I_plus  = _int_cos(op_col,   two_g - nw_col * t0_s,  t_arr, t_ref_c)  # (2m+nw)t + ...
    I_minus = _int_cos(-om_col,  two_g + nw_col * t0_s,  t_arr, t_ref_c)  # (2m-nw)t + ...
    c_psi   = 0.5  * delta_T0_an_base
    c_plus  = 0.25 * delta_T0_an_base + 0.5 * m_dm * delta_T0_bn
    c_minus = 0.25 * delta_T0_an_base - 0.5 * m_dm * delta_T0_bn
    term_c  = 0.5 * phi_sq * np.sum(
        c_psi * I_psi + c_plus * I_plus + c_minus * I_minus, axis=0)
    term_d = 0.5 * phi_sq * T_tilde

    # [CORREGIDO 2026-09] Termino que faltaba. La Ec. (25) del paper de
    # sensitivity arma delta Theta' a partir de epsilon_1_punto (Ec. 22e) menos
    # omega_punto, y los coeficientes A_n / B_n (Ec. 26) reproducen exactamente
    # el primer y el tercer termino de (22e) mas la parte no resonante de
    # -delta omega... pero NO el segundo termino, (1 - sqrt(1-e^2)) * varpi_punto.
    # Como (1 - sqrt(1-e^2)) es una constante, su integral es directamente
    # (1 - sqrt(1-e^2)) * delta varpi = (1 - sqrt1me2) * dOM, y entra en delta T0
    # dividida por wb. Es O(e) (el prefactor es e^2/2 y varpi_punto va como 1/e),
    # o sea despreciable para los sistemas casi circulares, pero a e = 0.4367
    # (J1903+0327) vale hasta 1.8% de la fase acumulada. Verificado: con este
    # termino, -wb*dT0 + dOM reproduce la cuadratura directa de (22e) a 1e-9
    # (sin el, el error era 0.08%-1.8% segun la masa).
    term_e = (1.0 - sqrt1me2) * dOM / wb

    dT0    = -term_b - term_c - term_d - term_e

    # delta T0 es idéntico a - delta Theta' pero dividido wb. Además, se
    # simplifica el término proporcional a la variación de la frecuencia orbital.

    # [CORREGIDO] Término extra que entra en delta Pb aparte de la contribución
    # del semieje mayor: tambien promediado en el tiempo (mismo motivo que daae).
    # -(1/4) <Phi^2>, con _mean_cos2 (sinc) para no dividir ni por T~ ni por m.
    dPb_extrae = -0.25 * phi_sq * _mean_cos2(m_dm, t_arr, t0_s, gamma_f, T_tilde)

    return ( partials["PB"]  * PB_s * (1.5 * daae + dPb_extrae)
           + partials["A1"]  * A1_s       * daa
           + partials["ECC"] * dECC
           + partials["OM"]  * dOM
           + partials["T0"]  * dT0 )

    # la señal se escribe como partial(r) / partial(param) * delta(param)

def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars):
    """
    Template ELL1 (baja excentricidad).
    """
    wb     = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s   = orb_pars["PB_s"]
    A1_s   = orb_pars["A1_s"]

    T_tilde = t_arr - tasc_s
    m_t     = m_dm * t_arr
    m_tasc  = m_dm * tasc_s
    P       = (2.0 * DM_DENSITY / m_dm**2) * rho_f**2          # Phi0^2 rho^2

    cos_mt    = np.cos(m_t    + gamma_f)
    cos_mtasc = np.cos(m_tasc + gamma_f)
    sin_2mt    = np.sin(2.0 * m_t    + 2.0 * gamma_f)
    sin_2mtasc = np.sin(2.0 * m_tasc + 2.0 * gamma_f)

    # ---- delta x  (parametro A1)  ----
    daa = -P * (cos_mt**2 - cos_mtasc**2)   # instantaneo: para el canal A1

    # ---- delta eta (EPS1) y delta kappa (EPS2)  ----   [sin cambios]
    k   = m_dm / wb
    Psi = wb * T_tilde                       # Psi'
    Y2  = 2.0 * (m_tasc + gamma_f)           # 2 * Upsilon_asc
    a1  = 2.0 * k - 1.0
    a2  = 2.0 * k + 1.0

    def _S(a):
        return Psi * np.cos(0.5 * a * Psi + Y2) * np.sinc(0.5 * a * Psi / np.pi)

    def _C(a):
        return Psi * np.sin(0.5 * a * Psi + Y2) * np.sinc(0.5 * a * Psi / np.pi)

    dEPS1 = ( P / 4.0 * np.sin(Psi)
            + (P / 2.0) * (k + 0.25) * _S(a1)
            + (P / 2.0) * (0.25 - k) * _S(a2) )
    dEPS2 = ( P / 4.0 * (np.cos(Psi) - 1.0)
            + (P / 2.0) * (k + 0.25) * _C(a1)
            - (P / 2.0) * (0.25 - k) * _C(a2) )

    # ---- delta TASC  ----   [sin cambios]
    two_m = 2.0 * m_dm
    dTASC = -(1/2) * P * (T_tilde + (sin_2mt - sin_2mtasc) / two_m)

    # ---- [CORREGIDO] delta a/a EFECTIVO y delta Pb extra EFECTIVO ----
    # Mismo promedio temporal que en el caso excentrico (Ec. C.1b): aca colapsa
    # al unico armonico n=1=wb, entonces es directamente el analogo de "part2"
    # del template DD, con T0->TASC. El canal A1 arriba sigue con "daa" (sin
    # promediar); solo el canal PB pasa a usar la version promediada.
    # <cos^2(mt+Y)> se calcula con sinc (_mean_cos2), asi no hay division por T~
    # ni por m: en t = TASC vale cos^2(m TASC + Y), el limite correcto, en vez
    # de 0/0. Verificado contra la integral numerica de daa (ver docstring).
    mean_cos2  = _mean_cos2(m_dm, t_arr, tasc_s, gamma_f, T_tilde)
    daae       = -P * (mean_cos2 - cos_mtasc**2)     # <delta x / x>
    dPb_extrae = -0.25 * P * mean_cos2               # -(1/4) <Phi^2>

    # [FINAL] ELL1 completo: Delta_R = x[sin Phi + k/2 sin 2Phi - h/2 cos 2Phi - 3/2 h]
    # (h = eta = EPS1). tempo/PINT omiten el termino constante -3/2 x eta (Lange et al.
    # 2001, Ec. A6: "terms which are constant in time are omitted"), asi que la
    # columna EPS1 de la matriz de diseno es solo -x cos(2Phi)/2. Para eta constante
    # da igual (lo absorbe la fase), pero la ULDM hace oscilar eta y
    # -3/2 x delta_eta(t) es un retardo que ningun parametro de timing absorbe.
    # Verificado contra la orbita integrada (Chequeos_referee/t38) y contra el
    # modelo DD de PINT, que no recorta nada (validacion/v02).
    return ( partials["PB"]   * PB_s * (1.5 * daae + dPb_extrae)
           + partials["A1"]   * A1_s       * daa
           + (partials["EPS1"] - 1.5 * A1_s) * dEPS1
           + partials["EPS2"] * dEPS2
           + partials["TASC"] * dTASC )

#%% [4] CARGA DE PULSARES Y TIMING
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
    """WidebandTimingModel exige que CADA TOA caiga en algun intervalo DMX.
    Algunos pulsares (p.ej. J1713+0747) tienen unas pocas TOAs en huecos del
    binning DMX -> el modelo wideband falla con 'cannot account for all TOAs in
    DMX intervals'. Se descartan esas TOAs (poquisimas: 2/1493 en J1713 ->
    impacto despreciable)."""
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
    print(f"    [DMX] descarto {int((~cov).sum())} TOA(s) fuera de intervalos DMX", flush=True)
    return toas[cov]


def drop_empty_dmx_bins(model, toas):
    """WidebandTimingModel rompe (IndexError 'mean_dm[which][0]', size 0) si algun
    bin DMX no contiene NINGUNA TOA. J1713+0747 tiene 1 bin vacio (idx 239). Se
    remueven los bins DMX sin TOAs. No-op si no hay DMX o ninguno esta vacio.
    Llamar DESPUES de drop_toas_outside_dmx (sobre las TOAs ya filtradas)."""
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
            for pre in ("DMX_", "DMXR1_", "DMXR2_", "DMXF1_", "DMXF2_"):  # fallback
                pn = f"{pre}{k:04d}"
                if pn in model.params:
                    try: model.remove_param(pn)
                    except Exception: pass
    try:
        model.validate(); model.setup()
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
# Mismo hook que DIRAC_vectorial_v2/pipeline_vectorial.py y DIRAC_alpha_v3/
# pipeline_alpha.py. Saca del DISENO las derivadas seculares que el .par traiga
# fiteadas, o sea que deja de marginalizarlas y las trata como valores medidos
# exactos.
#
# Por que PBDOT y no todo: medido con el calculo exacto sin MCMC
# (Comparacion NB vs WB/seculares_todos.py) en el nucleo del pico resonante,
# mismo pulsar y mismos residuos:
#   - A1DOT libre no cuesta NADA (+0.00 a +0.09 dex): vive en el canal A1, que
#     se lleva ~0 % de la norma del template.
#   - PBDOT libre cuesta +0.55 a +1.30 dex: vive en el canal PB, que se lleva
#     ~100 % de la norma.
#
# [v3] DEFAULT = PBDOT. Esta carpeta es la corrida NUEVA (~/beta3), que se hace
# justamente para homogeneizar con el lineal y el vectorial. La base vieja de
# ~/beta2 corrio con PBDOT LIBRE y NO es compatible: no mezclar.
#
# Medido en el lineal (Fisher exacta, 50 pulsares, grilla log-uniforme = todo
# off-resonance): congelar PBDOT mejora la envolvente CONJUNTA 0.40 dex en
# 1e-23 eV y hasta 2.66 dex en 1.4e-19 eV. NO es un efecto de resonancia: los 8
# sistemas con PBDOT fiteado son de los mejores relojes del array. Por eso la
# curva de beta2 no se puede empalmar ni comparar con esta.
CONGELAR_SECULARES = tuple(
    x.strip() for x in os.environ.get("CONGELAR_SECULARES", "PBDOT").split(",")
    if x.strip())
print(f"    [v3] CONGELAR_SECULARES = "
      f"{', '.join(CONGELAR_SECULARES) if CONGELAR_SECULARES else '(nada)'}")


def load_pulsar(par, tim, convert_low_ecc=True):
    from enterprise.pulsar import Pulsar
    from pint.models import get_model_and_toas
    cwd    = os.getcwd()
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
    # la matriz de diseno (psr.Mmat / psr.fitpars) con los parametros NO frozen.
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
    M       = np.asarray(psr.Mmat, dtype=np.float64) # matriz de diseño
    M_names = list(psr.fitpars)
    t_arr   = np.asarray(psr.toas, dtype=np.float64) # t_i
    n_toa   = len(t_arr)

    def col(name, conv):
        if name in M_names:
            return np.ascontiguousarray(M[:, M_names.index(name)] * conv, dtype=np.float64)
        return None

    pb_q = getattr(getattr(timing_model, "PB", None), "quantity", None)
    fb0  = None
    # me fijo si en el timing model fitearon PB o la frecuencia fb0
    if pb_q is not None:
        PB_s = float(pb_q.to(u.s).value)
    else:
        fb0_q = getattr(getattr(timing_model, "FB0", None), "quantity", None)
        if fb0_q is None:
            raise RuntimeError("El modelo no tiene ni PB ni FB0.")
        fb0  = float(fb0_q.to(u.Hz).value)
        PB_s = 1.0 / fb0

    orb_pars = {"PB_s": PB_s, "A1_s": float(timing_model.A1.value)}
    orb_pars["omega_b"] = 2.0 * np.pi / orb_pars["PB_s"]

    is_ell1 = hasattr(timing_model, "EPS1") and timing_model.EPS1.value is not None
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
            partials[name] = np.zeros(n_toa, dtype=np.float64); missing.append(name)
        else:
            partials[name] = c
    if "PB" in missing and "FB0" in M_names:
        fb0_col = col("FB0", 1.0)
        if fb0 is None:
            fb0 = 1.0 / orb_pars["PB_s"]
        partials["PB"] = fb0_col * (-(fb0 ** 2)); missing.remove("PB")

    return t_arr, partials, orb_pars, missing

# Obtenemos los t_i de los TOAs, las derivadas parciales de los time residuals 
# respecto de los parámetros orbitales (a partir de la design matrix), y
# los parámetros orbitales en sí para evaluar luego la señal

#%% [5] CONSTRUCCION DEL PTA
def make_uldm_signal(t_arr, partials, orb_pars, m_dm):
    from enterprise.signals import parameter, signal_base, deterministic_signals

    @signal_base.function
    def uldm_delay(toas, log10_beta=-75.0, rho_f=1.0, gamma_f=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, t_arr, partials, orb_pars)
        return (10.0 ** log10_beta) * h
    
    # defino los parámetros para enterprise: beta, rho y Upsilon
    
    lb = parameter.Uniform(LOG10_BETA_MIN, LOG10_BETA_MAX)("log10_beta")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0*np.pi)("gamma_f")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_beta=lb, rho_f=rf, gamma_f=gf), name="uldm")


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
    # el objeto PTA nos permite introducir la señal completa + ruido
    
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
    # analiticamente (vía Woodbury) dentro de get_lnlikelihood.
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

ULDM_SUFFIXES = ["log10_beta", "rho_f", "gamma_f"]
RN_SUFFIXES   = ["red_noise_log10_A", "red_noise_gamma"]

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

#%% [6] CARGA DEL RUIDO: MAP
# cargamos los parámetros de ruido y, salvo el red noise, los dejamos fijos a sus 
# máximos a posteriori (MAP)

def load_noise_MAP(pta, chain_file, pars_file):
    chain_raw = np.loadtxt(chain_file)
    with open(pars_file) as fh:
        par_names = [l.strip() for l in fh if l.strip()]

    n_par   = len(par_names)
    lnpost  = chain_raw[:, n_par]
    imap    = int(np.argmax(lnpost))
    d       = dict(zip(par_names, chain_raw[imap, :n_par]))

    for k in list(d.keys()):
        if k.endswith("_log10_equad"):
            d[k[:-len("_log10_equad")] + "_log10_t2equad"] = d.pop(k)

    uldm         = get_uldm_pnames(pta)
    noise_pnames = [p for p in pta.param_names if p not in uldm]
    map_noise    = {k: d[k] for k in noise_pnames if k in d}
    missing = set(noise_pnames) - set(map_noise)
    missing = {p for p in missing if not p.endswith('_dmjump')}  # from the .par
    if missing:
        print(f"    [WARN] faltan {len(missing)} params de ruido: {sorted(missing)[:3]}")
    return map_noise

#%% [7] CLASE MCMC: 5D POR PULSAR, PRIOR UNIFORME-EN-BETA
class PulsarULDM_RN(object):
    """
    MCMC de UN pulsar con ruido rojo marginalizado. 5 parametros:
      x[0]=log10_beta  x[1]=log10_rho  x[2]=gamma_f  x[3]=log10_A_red  x[4]=gamma_red
    Ruido blanco (EFAC/EQUAD) y DM wideband FIJOS en MAP (self.fix).
    Prior UNIFORME-en-beta (10^b): ln p(b) = b*ln10.
    """
    def __init__(self, d):
        self.d   = d
        self.pta = d["pta"]
        self.fix = d["noise_norn"]
        self.lb, self.rf, self.gf = d["lb"], d["rf"], d["gf"]
        self.arn, self.grn        = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim   = 5

    # este es el objeto de enterprise que permite evaluar la likelihood dados los 
    # 5 parámetros en un momento cualquiera.
    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.lb]  = x[0]
        p[self.rf]  = 10.0 ** x[1]
        p[self.gf]  = x[2]
        p[self.arn] = x[3]
        p[self.grn] = x[4]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_BETA_MIN <= x[0] <= LOG10_BETA_MAX):
            return -np.inf
        lp = x[0] * np.log(10.0) if PRIOR_ACOPLE == "uniforme" else 0.0 # prior en b tras hacer: p(beta) * dbeta = p(b) * db; [BF] loguniforme -> plano en b
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho  = 10.0 ** u
        lp  += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh en u=log10 rho
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

#%% [8] PRELOAD + RUNNER + ESS
def preload_one(name, m_eV):
    m_dm = m_eV * EV_TO_INV_S
    t_load = time.time()
    try:
        par, tim, chain, pars = resolve_paths(name)
        psr, tm = load_pulsar(par, tim, convert_low_ecc=True)
        t_arr, partials, orb_pars, missing = build_context(psr, tm)
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        lb, rf, gf = get_uldm_pnames(pta)
        arn, grn   = get_rn_pnames(pta)
        noise = load_noise_MAP(pta, chain, pars)
        noise.update(dmjumps_par(par, psr.name, pta))
        _sin_valor = sorted(set(pta.param_names) - set(get_uldm_pnames(pta)) - set(noise))
        if _sin_valor:
            raise RuntimeError(f"{psr.name}: parametros de ruido sin valor: {_sin_valor}")
        if arn not in noise or grn not in noise:
            raise RuntimeError(f"RN MAP ausente: arn_in_noise={arn in noise}, grn_in_noise={grn in noise}")
        rn_map     = (noise[arn], noise[grn])
        noise_norn = {k: v for k, v in noise.items() if k not in (arn, grn)}
        return dict(ctx=(t_arr, partials, orb_pars, m_dm), uldm_resto={rf: 1.0, gf: 0.0},
                    name=name, pta=pta, lb=lb, rf=rf, gf=gf, arn=arn, grn=grn,
                    noise_norn=noise_norn, rn_map=rn_map,
                    ecc=orb_pars["ecc"], n_toa=len(t_arr), t_load=time.time()-t_load)
    except Exception as e:
        print(f"  [{name}] PRELOAD FAIL: {type(e).__name__}: {e}", flush=True)
        return None


def effective_sample_size(x):
    """ESS (effective sample size) por autocorrelacion. Chequeo de si la cadena
    es lo suficientemente larga para que converja la posterior."""
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
    f   = np.fft.rfft(x, n=nfft)
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
    x0  = model.x0(rng)
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
    return data[nburn:, 0]              # columna 0 = log10_beta

#%% [9] Z_i(b) DESDE LA CADENA + COMBINACION
def lnZ_from_science_chain(beta_samples):
    """
    La cadena se sampleo con prior uniforme-en-beta -> su marginal es la posterior
    p_i(b) ∝ 10^b Z_i(b).  Recuperamos ln Z_i(b) = ln p_hat_i(b) - b*ln10
    sobre el rango bien muestreado. Debajo del codo (donde termina la meseta): extrap.
    PLANA. Arriba del drop: -inf.
    """
    ln10 = np.log(10.0)
    counts, _ = np.histogram(beta_samples, bins=BETA_EDGES)
    dens = counts.astype(float)
    if SMOOTH_SIGMA_BINS > 0:
        dens = gaussian_filter1d(dens, SMOOTH_SIGMA_BINS, mode="nearest")
    # hago smooth el histograma con un filtro gaussiano
    lnZ  = np.full(BETA_CENTERS.shape, -np.inf)
    dmax = dens.max() # máximo de la campana
    if dmax <= 0:
        return lnZ
    
    # TAIL_FRAC es el valor en el cual considero que se "apagó" la distribución, 0.05 del máximo
    good = dens > TAIL_FRAC * dmax
    idx  = np.where(good)[0]
    i_lo, i_hi = idx[0], idx[-1]
    sl = slice(i_lo, i_hi + 1)
    d_sl = np.maximum(dens[sl], TAIL_FRAC * dmax)
    lnZ[sl] = np.log(d_sl) - BETA_CENTERS[sl] * ln10
    n_anch = max(3, int(round(0.7 / DBETA)))
    lnZ[:i_lo] = np.median(lnZ[i_lo:i_lo + n_anch])

    # [CORREGIDO] Cola superior: EXTRAPOLAR en vez de cortar con -inf. Antes,
    # arriba de i_hi quedaba lnZ = -inf ("Z = 0 exactamente") en una region donde
    # la cadena simplemente NO muestreo. Al COMBINAR pulsares (suma de lnZ_i) la
    # suma hereda el -inf en cuanto CUALQUIER sumando lo tiene => el soporte de
    # la posterior conjunta lo fija hasta donde llego la cadena del pulsar mas
    # restrictivo, no la likelihood. La senal es LINEAL en beta (h = beta*template),
    # asi que para un resultado nulo Z ~ exp(-c beta^2): se ajusta lnZ = a0 + b*beta^2
    # en la cola bien muestreada y se continua. Validado en el vectorial (mismo
    # codigo): sesgo del corte +0.117 dex, cambio tipico por pulsar +0.013 dex.
    if EXTRAPOLAR_COLA and i_hi + 1 < len(BETA_CENTERS):
        j0 = max(i_lo, i_hi - n_anch + 1)
        x_hi = BETA_CENTERS[i_hi]
        u_fit = 10.0 ** (2.0 * (BETA_CENTERS[j0:i_hi + 1] - x_hi))
        if len(u_fit) >= 2:
            b, a0 = np.polyfit(u_fit, lnZ[j0:i_hi + 1], 1)
            if b < 0:
                u_ex = 10.0 ** (2.0 * (BETA_CENTERS[i_hi + 1:] - x_hi))
                with np.errstate(over="ignore"):
                    lnZ[i_hi + 1:] = a0 + b * u_ex
    return lnZ


def posterior_from_lnZ(lnZ_total, grid=None):
    """Posterior conjunta P(log10_beta) sobre una grilla (BETA_FINE por
    defecto). lnZ_total esta definido en BETA_CENTERS (histograma); se interpola
    a la grilla fina: debajo del rango -> plano (clamp), arriba del
    drop -> -inf (Z_i->0). Devuelve (post, ul95) sobre 'grid'."""
    if grid is None:
        grid = BETA_FINE
    ln10 = np.log(10.0)
    fin = np.isfinite(lnZ_total)
    b_fin, z_fin = BETA_CENTERS[fin], lnZ_total[fin]
    lnZ_g = np.interp(grid, b_fin, z_fin)            # (abajo: plano)
    lnZ_g[grid > b_fin.max()] = -np.inf              # arriba del drop: Z_i -> 0
    ln_post = grid * ln10 + lnZ_g                    # prior uniforme-en-beta, una vez
    ln_post -= np.nanmax(ln_post)
    post = np.exp(ln_post)
    post[~np.isfinite(post)] = 0.0
    post = post / np.trapz(post, grid)
    cdf  = cumulative_trapezoid(post, grid, initial=0.0)
    ul95 = float(np.interp(0.95, cdf, grid))
    return post, ul95


def combine_pulsars(lnZ_list):
    lnZ_total = np.sum(np.vstack(lnZ_list), axis=0)
    return posterior_from_lnZ(lnZ_total)

#%% [10] WORKER: 1 tarea = (1 masa, 1 pulsar)
def process_task(args):
    mass_idx, m_eV, name, n_iter, seed = args
    t0  = time.time()
    rng = np.random.default_rng(seed)
    d   = preload_one(name, m_eV)
    if d is None:
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err="preload_failed")
    try:
        model  = crear_modelo(d)
        outDir = os.path.join(CHAIN_ROOT, f"m{mass_idx:03d}", name)
        beta   = run_one_chain(model, outDir, n_iter, rng)
        lnZ    = lnZ_from_science_chain(beta)
        ul_sci = float(np.percentile(beta, 95.0))
        ess    = effective_sample_size(beta)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  "
              f"ecc={d['ecc']:.1e}  UL95={ul_sci:6.2f}  "
              f"b=[{beta.min():6.1f},{beta.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True,
                    beta=beta, lnZ=lnZ, ul_sci=ul_sci, ess=ess,
                    b_min=float(beta.min()), b_max=float(beta.max()),
                    n_toa=d["n_toa"], ecc=d["ecc"], t_min=(time.time()-t0)/60.0)
    except Exception as e:
        import traceback
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=False,
                    err=f"{type(e).__name__}: {e}", tb=traceback.format_exc())

#%% [11] PLOTS
def plot_individuals(results, m_eV, outpng):
    n = len(results)
    ncol = int(np.ceil(np.sqrt(n)))
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4*ncol, 3.3*nrow),
                             dpi=110, squeeze=False)
    for k, r in enumerate(results):
        ax = axes[k // ncol][k % ncol]
        beta = np.asarray(r["beta"])
        ul   = r["ul_sci"]
        counts, _ = np.histogram(beta, bins=BETA_EDGES)
        h = counts / max(counts.max(), 1)
        ax.bar(BETA_CENTERS, h, width=DBETA, color="0.7", edgecolor="none",
               align="center", label=r"cadena = $p_i(b)\propto 10^b Z_i$")
        ax.axvline(ul, color="crimson", ls="--", lw=1.3, label="UL95 (percentil)")
        sqrtb = np.sqrt(10.0**ul * UNIT_CONVERSION)
        ax.set_title(f"{r['name']}   b$\\in$[{beta.min():.1f}, {beta.max():.1f}]\n"
                     f"UL95={ul:.2f}  $\\sqrt{{\\beta}}$={sqrtb:.1e}  ESS={r.get('ess',0):.0f}",
                     fontsize=8)
        ax.set_xlabel(r"$\log_{10}\beta$", fontsize=8)
        ax.set_xlim(beta.min() - 0.5, max(beta.max(), ul) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls="--", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Posteriors INDIVIDUALES (uniforme-en-b, RN marg., m={m_eV:.0e} eV)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpng, dpi=140); plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_joint(post, ul95, names, m_eV, outpng):
    sqrtb = np.sqrt(10.0**ul95 * UNIT_CONVERSION)
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
    ax.plot(BETA_FINE, post, color="darkgreen", lw=1.8)
    ax.fill_between(BETA_FINE, post, where=(BETA_FINE <= ul95),
                    color="mediumseagreen", alpha=0.35, label="95% de la masa")
    ax.axvline(ul95, color="crimson", ls="--", lw=1.6,
               label=f"95% UL: log10(beta)={ul95:.2f}\nsqrt(beta)={sqrtb:.2e} GeV$^{{-1}}$")
    ax.set_xlabel(r"$\log_{10}\beta$", fontsize=13)
    ax.set_ylabel(r"posterior conjunto $P(\log_{10}\beta)$", fontsize=13)
    ax.set_title(f"COTA CONJUNTA ({len(names)} pulsares, RN marg., m={m_eV:.0e} eV)",
                 fontsize=11)
    m = post > post.max() * 1e-3
    ax.set_xlim(BETA_FINE[m].min() - 1, BETA_FINE[m].max() + 1)
    ax.legend(loc="best", fontsize=9); ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_vs_mass(m_arr, sqrtb_arr, n_used, outpng):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.loglog(m_arr, sqrtb_arr, "-", color="black", lw=1.4)
    ax.set_xlabel(r"$m$ [eV]", fontsize=12)
    ax.set_ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]  (cota conjunta, 95% UL)", fontsize=12)
    ax.set_title(f"Cota CONJUNTA vs masa  ({int(np.max(n_used))} pulsares, RN marg.)")
    ax.grid(True, which="both", ls="--", alpha=0.4)
    plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close(fig)
    print(f"  guardado: {outpng}")


#%% [12] MAIN
def main():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from collections import defaultdict
    single = len(MASS_GRID_EV) == 1
    print("\n" + "=" * 70)
    print(f"COTA CONJUNTA POR-PULSAR (RN marginalizado) | "
          f"{'1 masa' if single else f'{len(MASS_GRID_EV)} masas'}  {ALLOWED_MODELS}")
    print("=" * 70)
    t_all = time.time()

    # 1) pulsares
    print("\n[1] Descubriendo y filtrando pulsares...")
    all_names = discover_wb_pulsars()
    use = filter_pulsars_by_model(all_names, allowed=ALLOWED_MODELS)[:N_PULSARS_TARGET]
    if not use:
        raise SystemExit(f"No se encontraron pulsares {ALLOWED_MODELS}.")
    print(f"    USANDO {len(use)} pulsares: {use}")

    # 2) tareas planas (masa x pulsar)
    tasks = []
    for mi, m_eV in enumerate(MASS_GRID_EV):
        for pi, name in enumerate(use):
            tasks.append((mi, float(m_eV), name, N_ITER, SEED + mi*1000 + pi))
    total     = len(tasks)
    n_workers = MAX_WORKERS or (os.cpu_count() or 1)
    print(f"\n[2] {total} tareas (= {len(MASS_GRID_EV)} masas x {len(use)} pulsares), "
          f"{n_workers} procesos (cpu={os.cpu_count()})")
    print(f"    Cada tarea = 1 MCMC 5D de {N_ITER} iters. ETA sobre tareas completadas.\n")

    # 3) lanzar y combinar cada masa cuando llegan todos sus pulsares
    mass_results  = defaultdict(list)   # mass_idx -> [result dicts ok]
    mass_received = defaultdict(int)
    joint         = {}                  # mass_idx -> dict(ul95, sqrtb, n_used, post)
    done = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(process_task, a) for a in tasks]
        for fut in as_completed(futs):
            r  = fut.result()
            mi = r["mass_idx"]
            done += 1
            mass_received[mi] += 1
            if r.get("ok"):
                mass_results[mi].append(r)
            else:
                print(f"  [m#{mi:03d} {r['name']}] ERROR: {r.get('err')}", flush=True)
                if "tb" in r:
                    print(r["tb"], flush=True)

            el   = time.time() - t_start
            rate = done / el if el > 0 else 0.0
            eta  = (total - done) / rate if rate > 0 else float("nan")
            if done % max(1, len(use)) == 0 or done == total:
                print(f"  [{done:5d}/{total}] {100.0*done/total:5.1f}%   "
                      f"elapsed {el/60:6.1f} min   ETA {eta/60:6.1f} min", flush=True)

            if mass_received[mi] == len(use):
                lst = mass_results[mi]
                if lst:
                    post, ul95 = combine_pulsars([x["lnZ"] for x in lst])
                    sqrtb = np.sqrt(10.0 ** ul95 * UNIT_CONVERSION)
                    joint[mi] = dict(ul95=ul95, sqrtb=sqrtb, n_used=len(lst), post=post)
                    print(f"   >>> masa #{mi:03d}  m={MASS_GRID_EV[mi]:.3e} eV   "
                          f"COTA CONJUNTA log10(b)={ul95:.3f}  sqrt(b)={sqrtb:.3e} GeV^-1   "
                          f"({len(lst)}/{len(use)} pulsares)", flush=True)
                else:
                    print(f"   >>> masa #{mi:03d}: sin pulsares OK, se omite.", flush=True)

    if not joint:
        raise SystemExit("Ninguna masa con cota; revisar errores de arriba.")

    # 4) ensamblar + guardar
    idx_ok    = sorted(joint.keys())
    m_arr     = np.array([MASS_GRID_EV[i]    for i in idx_ok])
    ul_arr    = np.array([joint[i]["ul95"]   for i in idx_ok])
    sqrtb_arr = np.array([joint[i]["sqrtb"]  for i in idx_ok])
    nused_arr = np.array([joint[i]["n_used"] for i in idx_ok])

    npz = os.path.join(OUT_ROOT, "DEFINITIVO_massweep_joint.npz")
    np.savez(npz, masses_eV=m_arr, log10_beta_95UL_joint=ul_arr,
             sqrtb_95UL_joint=sqrtb_arr, n_pulsars_used=nused_arr,
             names=np.array(use), beta_centers=BETA_CENTERS, beta_fine=BETA_FINE)
    print(f"\n[3] SAVE {npz}")

    # 5) plots
    print(f"\n[4] Plots...")
    if single:
        mi  = idx_ok[0]
        lst = sorted(mass_results[mi], key=lambda r: use.index(r["name"]))
        print(f"\n  ---- UL95 individual por pulsar (m={m_arr[0]:.2e} eV) ----")
        print(f"    {'pulsar':15s} {'n_toa':>6s} {'ecc':>8s} {'UL95':>8s} "
              f"{'sqrt(b)':>11s} {'ESS':>8s}")
        for r in lst:
            sb = np.sqrt(10.0**r["ul_sci"] * UNIT_CONVERSION)
            flag = "  <-- b_max bajo?" if r["b_max"] < r["ul_sci"] - 0.3 else ""
            print(f"    {r['name']:15s} {r['n_toa']:6d} {r['ecc']:8.1e} "
                  f"{r['ul_sci']:8.3f} {sb:11.2e} {r['ess']:8.0f}{flag}")
        print(f"\n  ====== COTA CONJUNTA ({nused_arr[0]} pulsares) ======")
        print(f"    log10_beta_95UL = {ul_arr[0]:.3f}")
        print(f"    sqrt(beta)_95UL = {sqrtb_arr[0]:.3e} GeV^-1")
        plot_individuals(lst, m_arr[0], os.path.join(OUT_ROOT, "DEFINITIVO_individuales.png"))
        plot_joint(joint[mi]["post"], ul_arr[0], use, m_arr[0],
                   os.path.join(OUT_ROOT, "DEFINITIVO_conjunto.png"))
    else:
        plot_vs_mass(m_arr, sqrtb_arr, nused_arr,
                     os.path.join(OUT_ROOT, "DEFINITIVO_sqrtbeta_vs_m.png"))

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
