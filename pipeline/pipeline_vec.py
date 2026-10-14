# =============================================================================
# =============================================================================
# VERSION VECTORIAL. Calcula la COTA CONJUNTA del acople adimensional g
# combinando cadenas MCMC POR-PULSAR con ruido rojo marginalizado, sobre un
# barrido en masa de 200 puntos entre 1e-23 y 1e-18 eV.
#
# A diferencia del caso ESCALAR (cuadratico, senal ~ beta * phi^2), aca la ULDM
# es VECTORIAL: la senal es LINEAL en un acople adimensional g y ademas depende
# de la DIRECCION del campo en la esfera, descripta por dos angulos:
#   vartheta (polar)   ~ prior isotropo  p(vartheta) ∝ sin(vartheta),  [0, pi]
#   varphi   (azimutal) ~ Uniform(0, 2pi)
# (mismo sampleo que en 'sensitivity_vectorial_variable_psi.py':
#  vartheta = arccos(U[-1,1]),  varphi = U[0,2pi] -> isotropo en la esfera).
#
# Por cada (masa, pulsar) samplea con PTMCMC 7 parametros:
#   (log10_g, log10_rho, gamma_f, vartheta, varphi, log10_A_red, gamma_red)
# la senal ULDM vectorial (g, rho, gamma, vartheta, varphi) + el ruido rojo
# acromatico (A_red, gamma_red). El ruido blanco (EFAC/EQUAD) y el DM wideband
# quedan FIJOS en MAP.
# Con las cadenas estima Z_i(g) y combina:  ln p(g|D) = g ln10 + sum_i ln Z_i(g).
#
# NOTA: el TEMPLATE de la senal vectorial (compute_h_template*) esta dejado en
#       BLANCO a proposito: la firma, el andamiaje orbital y la combinacion con
#       las derivadas parciales estan listos, pero las expresiones de la fisica
#       (delta a/a, delta EPS1/EPS2/TASC, delta OM/ECC/T0, dependientes ahora de
#       vartheta, varphi e inclinacion) se completan a mano.
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
MASS_MIN_EV   = 1.0e-23
MASS_MAX_EV   = 1.0e-18
N_MASS        = 200             # 200 masas log-espaciadas (40/decada, 5 decadas)
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
    "OUT_ROOT", r"C:\Users\fedeh\OneDrive\Documents\C\Version definitiva vectorial")
CHAIN_ROOT    = os.path.join(OUT_ROOT, "chains_RNmarg")
os.makedirs(OUT_ROOT, exist_ok=True)
os.makedirs(CHAIN_ROOT, exist_ok=True)

# Constantes numericas

EV_TO_INV_S     = 1.51926760385984e+15
YR_IN_S         = 3.15576e7
DAY_IN_S        = 86400.0
DM_DENSITY      = 0.3 * 4.09351499910375e+55
# Masa del nucleon en unidades naturales (necesaria para el acople vectorial B-L)
M_NUCLEON       = 1.67492729e-27 * 8.52246714808644e+50
# Asimetria de carga B-L entre las componentes del binario. Para B-L: c_NS ~ 1.0-1.1,
# c_WD = 0.504 -> Delta c ~ 0.5 (Graham+2016 y Xue+2022 usan pulsar ~1, Tierra ~0.5).
# El 0.1 de antes era c_B - c_N del escalar Higgs-portal, no Delta c de B-L.
DELTA_C         = 0.5

# Masas y constantes para el semieje mayor relativo 'a' (Kepler III), que el
# template VECTORIAL necesita (el escalar no lo usaba). Mismas convenciones/unidades
# que en 'sensitivity_vectorial_variable_psi.py' -> a_semi en unidades naturales.
M_SUN_KG        = 2.0e30
G_CONST         = 6.67e-11
KEPLER_UNIT_CONV = 3.71140109219707e-26   # factor de conversion a unidades naturales
DEFAULT_M1      = 1.35     # masa del pulsar (M_sun) si no esta en el modelo
DEFAULT_M2      = 0.20     # masa de la companera (M_sun) si no esta en el modelo/prior
# Priors de masa de la companera (M_sun) del sensitivity, por si el modelo no trae M2
COMPANION_MASS_PRIORS = {
    "J0406+3039": 0.096246, "J0610-2100": 0.024748, "J1012+5307": 0.124641,
    "J1125+7819": 0.329741, "J1713+0747": 0.324201, "J1719-1438": 0.001304,
    "J1738+0333": 0.103990, "J1745+1017": 0.015826, "J1802-2124": 0.981815,
    "J2145-0750": 0.503270, "J2234+0944": 0.017786, "J2317+1439": 0.201229,
}

# Tabla de masas e inclinaciones por pulsar (m1, m2, sin i), consistentes con la
# funcion de masa. La genera 'Diagnostico masas vectorial/tabla_por_pulsar.py':
#   A = m1 y m2 publicados juntos, i de la funcion de masa
#   B = i medida o acotada (o SINI del .par si el Shapiro es debil), m1 = 1.35, m2 de f
#   C = sin medicion: convencion ATNF (m1 = 1.35, i = 60 deg, m2 de f = MEDMASS)
# Si el archivo no esta o el pulsar no figura, build_context vuelve a los defaults
# viejos (DEFAULT_M1, M2 del modelo / COMPANION_MASS_PRIORS, model_sini) y lo avisa
# con "[MASAS]". En el cluster tiene que estar en la misma carpeta que este archivo.
MASAS_CSV = os.environ.get(
    "MASAS_CSV", os.path.join(os.path.dirname(os.path.abspath(__file__)), "masas_pulsares.csv"))


def cargar_tabla_masas(path=MASAS_CSV):
    """{psr: {'m1','m2','sini','nivel','fuente'}} o None si no existe el archivo."""
    if not os.path.isfile(path):
        return None
    import csv
    with open(path, encoding="utf8") as fh:
        filas = csv.DictReader(l for l in fh if not l.startswith("#"))
        return {r["psr"]: dict(m1=float(r["m1"]), m2=float(r["m2"]), sini=float(r["sini"]),
                               nivel=r["nivel"], fuente=r["fuente"]) for r in filas}


TABLA_MASAS = cargar_tabla_masas()

N_TERMS         = 20 # Terminos del desarrollo de Bessel
N_FREQ_RN       = 30 # Numero de modos del red noise

# ---- Rango del acople adimensional g (se samplea en log10) ----
# La senal vectorial es LINEAL en g: senal = 10^log10_g * h_template, donde
# h_template (acople unitario) incluye delta_c/m_nucleon * sqrt(2 rho_DM) * rho.
# Zona de busqueda g in [1e-30, 1e-15] (el estudio de sensitivity ubica g~1e-24..1e-18,
# comodo dentro de la ventana). Si en alguna masa la posterior se pega al borde
# superior (UL ~ -15), la cota es mas debil que la ventana -> subir LOG10_G_MAX.
LOG10_G_MIN = float(os.environ.get("LOG10_G_MIN", "-30.0"))
LOG10_G_MAX = float(os.environ.get("LOG10_G_MAX", "-15.0"))

# [BF] Prior del acople. "uniforme" (default) = uniforme en g, el de las
# cotas. "loguniforme" = uniforme en log10(g), el de la busqueda de senal
# (Bayes factor). Solo cambia el termino de g en PulsarULDM_RN.lnprior.
PRIOR_ACOPLE = os.environ.get("PRIOR_ACOPLE", "uniforme").strip().lower()
if PRIOR_ACOPLE not in ("uniforme", "loguniforme"):
    raise ValueError(f"PRIOR_ACOPLE={PRIOR_ACOPLE!r}: uniforme|loguniforme")

RHO_F_MIN     = 1.0e-3
RHO_F_MAX     = 6.0         # limites sobre la amplitud rho para definir la variable en enterprise
LOG10_RHO_MIN = np.log10(RHO_F_MIN)
LOG10_RHO_MAX = np.log10(RHO_F_MAX)
ECC_LOW_THRESHOLD = 0.01
# valor de excentricidad a partir del cual se considera que un sistema es circular

# Inclinacion por defecto si el timing model no la trae (SINI/KIN ausentes).
# El template vectorial depende de la inclinacion (a diferencia del escalar).
DEFAULT_SINI = np.sin(np.deg2rad(60.0))

# [OJO] El acople VECTORIAL es LINEAL en el campo (fuerza ~ cos(m t)), asi que
# la resonancia es  m = n*wb  (k = m/wb entero), NO 2m = n*wb como en el escalar
# cuadratico. En el limite circular (ELL1) el unico denominador que se anula es
# (k-1) -> resonancia UNICA en k=1 (m=wb); los canales EPS1/EPS2 tienen ademas
# terminos (k-2) que resonarian en k=2 (m=2wb), pero en k=1 son regulares. Todo
# se maneja SIN rama gracias a la forma con sinc de los _r*(). RES_REL_TOL solo lo
# usaria el template DD (excentrico), que hoy es un stub.
RES_REL_TOL = 1.0e-4

# Correccion del canal PB (analoga a daa->daae de la directora en el escalar):
# el parametro PB ve el PROMEDIO temporal <da/a>, no el instantaneo. Con
# PROMEDIAR_PB=True (default, fisica correcta) el canal PB usa <da/a>; con False
# se revierte al instantaneo (solo para comparar).
PROMEDIAR_PB = True

# Valor en el que se CONGELA varphi cuando se usa PulsarULDM_RN_fixvarphi (6D).
# Congelar el azimut saca esa componente del funnel angular en resonancia; el
# barrido vectorial lo activa a lo largo de todo el barrido.
VARPHI_FIX = 0.0

# ---- MCMC (priors / arranque) ----
BURN_FRAC       = 0.30 # quema el 30% inicial de la cadena
LOG10_G_INIT    = -26.0   # arranque, comodo dentro de [LOG10_G_MIN, LOG10_G_MAX]
COV_UPDATE      = 2000 # actualiza la covarianza cada 2000 pasos
SCAM_WEIGHT     = 60 # parametros optimos para la cadena segun testeamos antes, por prueba y error
AM_WEIGHT       = 20
DE_WEIGHT       = 20
SIGMA_LOG10G    = 3.0
SIGMA_LOG10RHO  = 0.5
SIGMA_GAMMA     = 2 * np.pi
# Angulo polar REPARAMETRIZADO: sampleamos x[3] = cos(vartheta) ~ U(-1,1) en vez de
# vartheta con prior sin(vartheta). Isotropo en la esfera pero con prior PLANO y sin
# la singularidad log(sin)->-inf en los polos -> mucho mejor aceptacion/mezcla.
SIGMA_COS_VARTHETA = 0.5         # x[3] = cos(vartheta) in [-1, 1]
SIGMA_VARPHI       = 2 * np.pi   # angulo azimutal, x[4] in [0, 2pi]

# ---- Red noise (marginalizado): priors y escalas ----
# Usamos lo mismo que en los papers de constraints de PTA

LOG10_ARED_MIN, LOG10_ARED_MAX = -20.0, -11.0
GAMMA_RED_MIN,  GAMMA_RED_MAX  = 0.0, 10.0
SIGMA_LOG10_ARED = 1.5
SIGMA_GAMMA_RED  = 1.5

# ---- Binning para Z_i(g) y combinacion ----
# Para combinar eficientemente las likelihoods Z_i de cada sistema, no usamos los
# puntos de la cadena en si. Generamos un histograma de la posterior final, el
# cual representa la likelihood, ya que es mas "limpio" a la hora de combinar
# sistemas.

N_G_BINS          = int(round((LOG10_G_MAX - LOG10_G_MIN) / 0.005))   # [FINAL] 0.005 dex/bin, como alpha   # mas bins -> mejor estimacion de Z_i(g) (rango mas angosto ayuda)
G_EDGES           = np.linspace(LOG10_G_MIN, LOG10_G_MAX, N_G_BINS + 1)
G_CENTERS         = 0.5 * (G_EDGES[:-1] + G_EDGES[1:])
DG                = G_EDGES[1] - G_EDGES[0]
SMOOTH_SIGMA_BINS = 6.0   # [FINAL] 0.03 dex, como alpha
TAIL_FRAC         = 0.05
# Extrapolar la cola superior de lnZ (ajuste Z~exp(-c g^2)) en vez de cortar con
# -inf arriba del ultimo bin muestreado. Ver el comentario en
# lnZ_from_science_chain: sin esto, al COMBINAR pulsares el soporte conjunto lo
# fija hasta donde llego la cadena del pulsar dominante (cota optimista ~0.12 dex
# y UL95 pegado al borde). False = comportamiento viejo.
EXTRAPOLAR_COLA   = True

# Grilla para la posterior conjunta final, desacoplada del histograma de Z_i.
# Z_i(g) se estima en G_CENTERS (histograma robusto, no se toca); la posterior
# final se evalua interpolando lnZ_total a esta grilla fina.

N_G_FINE          = int(round((LOG10_G_MAX - LOG10_G_MIN) / 0.002)) + 1
G_FINE            = np.linspace(LOG10_G_MIN, LOG10_G_MAX, N_G_FINE)

if SINGLE_MASS_EV is not None:
    MASS_GRID_EV = np.array([float(SINGLE_MASS_EV)])
else:
    MASS_GRID_EV = np.logspace(np.log10(MASS_MIN_EV), np.log10(MASS_MAX_EV), N_MASS)

print(f"[1] PATHS OK   masas={len(MASS_GRID_EV)} "
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

#%% [3] TEMPLATE ULDM VECTORIAL
# =============================================================================
# La senal vectorial es LINEAL en el acople g (que se factoriza afuera, en
# uldm_delay: senal = 10^log10_g * h). Aca se construye h (acople unitario).
#
# A diferencia del escalar, la senal depende de la DIRECCION del campo en la
# esfera via (vartheta, varphi) y de la INCLINACION del binario (orb_pars["iota"],
# orb_pars["sin_i"]).  Referencia de la fisica: ver da/a, omega_dot y epsilon_dot
# en 'sensitivity_vectorial_variable_psi.py'.
#
# La senal se escribe, igual que en el escalar, como
#     h = sum_param  partial(residual)/partial(param) * delta(param)
# con las derivadas parciales tomadas de la design matrix (partials[...]).
# Lo unico que falta completar son las expresiones de cada delta(param) para el
# campo VECTORIAL.
# =============================================================================
# ---- Helpers de integracion, en forma cerrada ESTABLE con sinc (sin ramas) ----
# Cada coeficiente S/C es una suma de terminos [trig((k+n)Psi + p) - trig(p)]/(k+n).
# Cuando el denominador delta=(k+n) -> 0 (resonancia k=n) el termino es 0/0 y su
# limite es secular (crece linealmente en Psi). En vez de un 'if abs(delta)<tol'
# con Taylor (que introduce un escalon en la masa y cancelacion catastrofica cerca
# del umbral), usamos identidades producto-a-suma que dejan un sinc explicito
# (np.sinc(x)=sin(pi x)/(pi x), con np.sinc(0)=1). Asi la expresion es CONTINUA y
# estable a traves de la resonancia, exactamente como en el caso lineal
# (_stable_minus de pipeline_alpha.py). En resonancia el sinc -> 1 y queda el
# crecimiento secular ~Psi de forma automatica.
#
#   [sin(dPsi+p) - sin(p)]/d =  Psi cos(dPsi/2 + p) sinc(dPsi/2pi)
#   [cos(dPsi+p) - cos(p)]/d = -Psi sin(dPsi/2 + p) sinc(dPsi/2pi)
#   [1 - cos(kk Psi)]/kk     =  Psi sin(kk Psi/2)   sinc(kk Psi/2pi)
#   sin(kk Psi)/kk           =  Psi sinc(kk Psi/pi)

def _rsin(delta, Psi, p):
    """[sin(delta*Psi + p) - sin(p)] / delta ; estable, limite delta->0: Psi*cos(p)."""
    return Psi * np.cos(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))

def _rcos(delta, Psi, p):
    """[cos(delta*Psi + p) - cos(p)] / delta ; estable, limite delta->0: -Psi*sin(p)."""
    return -Psi * np.sin(0.5 * delta * Psi + p) * np.sinc(delta * Psi / (2.0 * np.pi))

def _rver(kk, Psi):
    """[1 - cos(kk*Psi)] / kk ; estable, limite kk->0: 0.5*kk*Psi^2 (->0)."""
    return Psi * np.sin(0.5 * kk * Psi) * np.sinc(kk * Psi / (2.0 * np.pi))

def _rs0(kk, Psi):
    """sin(kk*Psi) / kk ; estable, limite kk->0: Psi."""
    return Psi * np.sinc(kk * Psi / np.pi)


# ---- Promedio temporal de da/a para el canal PB (correccion D. Lopez Nacir) --
# El parametro PB del timing model NO se mide instante a instante: se infiere de
# la FASE orbital acumulada, asi que el forzante correcto del canal PB es el
# PROMEDIO temporal <da/a> = (1/Psi) int_0^Psi (da/a) dPsi', no el valor
# instantaneo (el canal A1, x=a1 sin i, si ve el instantaneo). Es la MISMA
# correccion que la directora aplico al escalar (daa -> daae en pipeline_alpha).
# Como da/a es suma de _rsin/_rcos, su promedio es suma de <_rsin>/<_rcos>, cuya
# integral analitica sale en forma cerrada ESTABLE en la resonancia k=1:
#   <_rsin(d,Psi,p)> =  Psi [0.5 cos p sinc(z/2pi)^2 - sin p (z-sin z)/z^2]
#   <_rcos(d,Psi,p)> = -Psi [0.5 sin p sinc(z/2pi)^2 + cos p (z-sin z)/z^2]
# con z = d*Psi. En d->0 (resonancia) sinc^2->1 y (z-sin z)/z^2->0, sin 0/0.

def _z_minus_sin_over_z2(z):
    """(z - sin z)/z^2, estable (0 en z=0, ~z/6). Serie de Taylor para |z|<0.1
    (ahi la resta z-sin z pierde todas las cifras); resta directa fuera."""
    z = np.asarray(z, dtype=np.float64)
    chico = np.abs(z) < 0.1
    zs = np.where(chico, 1.0, z)
    directa = (zs - np.sin(zs)) / (zs * zs)
    z2 = z * z
    serie = z * (1.0 / 6.0 + z2 * (-1.0 / 120.0 + z2 * (
        1.0 / 5040.0 + z2 * (-1.0 / 362880.0 + z2 / 39916800.0))))
    return np.where(chico, serie, directa)


def _avg_rsin(delta, Psi, p):
    """<_rsin(delta,Psi',p)> promediado en Psi' de 0 a Psi (estable en delta->0)."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return Psi * (0.5 * np.cos(p) * s2 - np.sin(p) * _z_minus_sin_over_z2(z))


def _avg_rcos(delta, Psi, p):
    """<_rcos(delta,Psi',p)> promediado en Psi' de 0 a Psi (estable en delta->0)."""
    z = delta * Psi
    s2 = np.sinc(z / (2.0 * np.pi)) ** 2
    return -Psi * (0.5 * np.sin(p) * s2 + np.cos(p) * _z_minus_sin_over_z2(z))


# ---- TEMPLATE EXCENTRICO (rama DD) ------------------------------------------
# La rama DD de compute_h_template era un STUB que devolvia ceros: los tres
# binarios con e >= ECC_LOW_THRESHOLD (J1903+0327, J1946+3417, J2234+0611)
# entraban con senal identicamente nula. El template vive en template_ecc_vec.py
# (cuaderno vectorial excentrico; validado a 1.6e-12 contra las formas cerradas
# y a 0.5% contra el peine medido en los residuos reales).
# El import es opcional a proposito: sin el archivo el pipeline sigue corriendo
# los circulares igual que antes, y solo falla si se le pide un excentrico.
try:
    from template_ecc_vec import compute_h_template_ecc, N_TERMS_ECC
except ImportError:              # pragma: no cover
    compute_h_template_ecc = None
    N_TERMS_ECC = None


def compute_h_template(m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars):
    """Selector: ELL1 (baja ecc) vs DD (excentrico)."""
    if orb_pars["ecc"] < ECC_LOW_THRESHOLD:
        return compute_h_template_low_ecc(
            m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars)

    # ---- TEMPLATE DD (excentrico) ----
    # Delega en template_ecc_vec.compute_h_template_ecc, que arma los cinco
    # canales (PB, A1, ECC, OM, T0) con el peine de Bessel completo. gamma_f ES
    # Upsilon_0 = Upsilon + m T0 y varphi ES el acimut DESDE EL PERIASTRO: los
    # dos se samplean directo (correr una U(0,2pi) por una constante da la misma
    # U(0,2pi)), igual que Upsilon_asc / varphi_asc en la rama ELL1.
    if compute_h_template_ecc is None:
        raise RuntimeError(
            f"falta template_ecc_vec.py y el sistema tiene ecc="
            f"{orb_pars['ecc']:.4f} >= {ECC_LOW_THRESHOLD}")
    F_unit_const = (DELTA_C / M_NUCLEON) * np.sqrt(2.0 * DM_DENSITY)
    return compute_h_template_ecc(m_dm, rho_f, gamma_f, vartheta, varphi,
                                  t_arr, partials, orb_pars, F_unit_const,
                                  N=N_TERMS_ECC, promediar_pb=PROMEDIAR_PB)


def compute_h_template_low_ecc(m_dm, rho_f, gamma_f, vartheta, varphi, t_arr, partials, orb_pars):
    """
    Template ELL1 (baja excentricidad), VECTORIAL spin-1.

    Implementa el desarrollo del cuaderno 'thetaMenos' (limite circular e->0,
    convencion theta = Psi' - omega). La senal se escribe como
      h = sum_param partial(r)/partial(param) * delta(param),  con:
      - PB   : delta Pb / Pb = (3/2) delta a / a          (Kepler III, 'a' verdadero)
      - A1(x): delta x / x    = delta a / a + cot(iota) delta iota   (x = a1 sin i)
      - EPS1 : delta eta   (eta = e sin omega)
      - EPS2 : delta kappa (kappa = e cos omega)
      - TASC : delta T_asc

    Cada delta(param) = prefactor * [ S(Psi') cos(Ups_asc) + C(Psi') sin(Ups_asc) ],
    con el acople g factorizado afuera (en uldm_delay: h se multiplica por 10^log10_g).

    En la convencion 'thetaMenos' el arg. del periastro omega entra UNICAMENTE via
    la fase corrida unica  varphi_asc = varphi + omega  (todos los 2*omega colapsan
    en los sectores iota/Omega/T_asc). Como para orbitas circulares omega esta MAL
    definido (e->0 => arctan2(EPS1,EPS2) es ruido) y la senal solo depende de
    varphi_asc, sampleamos DIRECTAMENTE varphi_asc: el 'varphi' que llega aca ES la
    fase corrida (prior U(0,2pi) igual que la del azimut fisico, ya que sumar la
    constante omega solo corre una uniforme -> misma uniforme). Nunca se usa omega.

    Idem la fase temporal: sampleamos DIRECTAMENTE Upsilon_asc = Upsilon + m Tasc.
    El 'gamma_f' que llega aca ES Upsilon_asc (prior U(0,2pi); sumar la constante
    m*Tasc solo corre una uniforme -> misma uniforme). No se le suma m*Tasc.
    """
    wb     = orb_pars["omega_b"]
    tasc_s = orb_pars["TASC_s"]
    PB_s   = orb_pars["PB_s"]
    A1_s   = orb_pars["A1_s"]
    a_semi = orb_pars["a_semi"]        # semieje mayor relativo (Kepler), unidades naturales
    iota   = orb_pars["iota"]          # inclinacion (NUEVO respecto al escalar)
    sin_i  = orb_pars["sin_i"]

    # Amplitud con acople unitario (g=1):  F_unit = (Delta_c / m_n) sqrt(2 rho_DM) rho
    F_unit = (DELTA_C / M_NUCLEON) * np.sqrt(2.0 * DM_DENSITY) * rho_f
    sin_vt = np.sin(vartheta)
    cos_vt = np.cos(vartheta)
    cot_i  = np.cos(iota) / sin_i

    # ---- Convenciones (cuaderno thetaMenos, theta = Psi' - omega) ----
    Psi  = wb * (t_arr - tasc_s)             # Psi' = wb (t - Tasc)
    k    = m_dm / wb                          # k = m / wb
    Ups  = gamma_f                            # gamma_f YA es Upsilon_asc (sampleado directo)
    cU   = np.cos(Ups)
    sU   = np.sin(Ups)
    # Fase corrida UNICA del cuaderno: varphi_asc = varphi + omega. La sampleamos
    # directamente (ver docstring): el 'varphi' de entrada YA es varphi_asc.
    phi_asc = varphi
    cpa     = np.cos(phi_asc)
    spa     = np.sin(phi_asc)

    pref  = F_unit / (a_semi * wb ** 2)       # prefactor comun (1/(a wb^2)) F_unit
    prefT = F_unit / (a_semi * wb ** 3)       # delta T_asc lleva un 1/wb extra

    # ---- delta a / a  (INSTANTANEO: para el canal A1 via dxx) ----
    Sa  = -_rsin(k + 1.0, Psi, -phi_asc) + _rsin(k - 1.0, Psi, phi_asc)
    Ca  = -_rcos(k + 1.0, Psi, -phi_asc) + _rcos(k - 1.0, Psi, phi_asc)
    daa = pref * sin_vt * (Sa * cU + Ca * sU)

    # ---- delta a / a EFECTIVO = <da/a> promediado en el tiempo (canal PB) ----
    # Misma correccion que daa->daae en el escalar: el canal PB ve el promedio
    # corrido, no el instantaneo. Es daa con _rsin/_rcos -> <_rsin>/<_rcos>.
    Sa_avg = -_avg_rsin(k + 1.0, Psi, -phi_asc) + _avg_rsin(k - 1.0, Psi, phi_asc)
    Ca_avg = -_avg_rcos(k + 1.0, Psi, -phi_asc) + _avg_rcos(k - 1.0, Psi, phi_asc)
    daa_avg = pref * sin_vt * (Sa_avg * cU + Ca_avg * sU)

    # ---- delta iota ----  (sin 2*omega: fases puras en Psi')
    Si    = 0.5 * (_rver(k + 1.0, Psi) + _rver(k - 1.0, Psi))
    Ci    = 0.5 * (_rs0(k + 1.0, Psi) + _rs0(k - 1.0, Psi))
    diota = -pref * cos_vt * (Si * cU + Ci * sU)

    # ---- delta x / x = delta a / a + cot(iota) delta iota ----
    dxx = daa + cot_i * diota

    # ---- delta eta (EPS1) ----
    Se  = (-1.5 * _rver(k, Psi) * cpa
           - 0.25 * _rcos(k + 2.0, Psi, -phi_asc)
           - 0.25 * _rcos(k - 2.0, Psi,  phi_asc))
    Ce  = (-1.5 * _rs0(k, Psi) * cpa
           + 0.25 * _rsin(k + 2.0, Psi, -phi_asc)
           + 0.25 * _rsin(k - 2.0, Psi,  phi_asc))
    dEPS1 = -pref * sin_vt * (Se * cU + Ce * sU)

    # ---- delta kappa (EPS2) ----
    Sk  = ( 1.5 * _rver(k, Psi) * spa
           + 0.25 * _rsin(k + 2.0, Psi, -phi_asc)
           - 0.25 * _rsin(k - 2.0, Psi,  phi_asc))
    Ck  = ( 1.5 * _rs0(k, Psi) * spa
           + 0.25 * _rcos(k + 2.0, Psi, -phi_asc)
           - 0.25 * _rcos(k - 2.0, Psi,  phi_asc))
    dEPS2 = -pref * sin_vt * (Sk * cU + Ck * sU)

    # ---- delta T_asc ----  (integrada a mano; el cuaderno frena en T_asc_dot)
    ST = ( -0.5 * cot_i * cos_vt * (_rs0(k - 1.0, Psi) - _rs0(k + 1.0, Psi))
           - sin_vt * (-_rcos(k - 1.0, Psi, phi_asc) - _rcos(k + 1.0, Psi, -phi_asc)) )
    CT = ( -0.5 * cot_i * cos_vt * (-_rver(k - 1.0, Psi) + _rver(k + 1.0, Psi))
           - sin_vt * ( _rsin(k - 1.0, Psi, phi_asc) + _rsin(k + 1.0, Psi, -phi_asc)) )
    dTASC = prefT * (ST * cU + CT * sU)

    # Canal PB con <da/a> (correccion de la directora); A1 sigue con el
    # instantaneo dxx. delta Pb/Pb = (3/2) da/a: en el vectorial B-L NO hay
    # termino extra -(1/2) d(GM)/GM (el campo no reescala GM_T), a diferencia
    # del escalar universal.
    daa_pb = daa_avg if PROMEDIAR_PB else daa
    # [FINAL] ELL1 completo: Delta_R = x[sin Phi + k/2 sin 2Phi - h/2 cos 2Phi - 3/2 h]
    # (h = eta = EPS1). tempo/PINT omiten el termino constante -3/2 x eta (Lange et al.
    # 2001, Ec. A6: "terms which are constant in time are omitted"), asi que la
    # columna EPS1 de la matriz de diseno es solo -x cos(2Phi)/2. Para eta constante
    # da igual (lo absorbe la fase), pero la ULDM hace oscilar eta y
    # -3/2 x delta_eta(t) es un retardo que ningun parametro de timing absorbe.
    # Verificado contra la orbita integrada (Chequeos_referee/t38) y contra el
    # modelo DD de PINT, que no recorta nada (validacion/v02).
    return ( partials["PB"]   * PB_s * (1.5 * daa_pb)
           + partials["A1"]   * A1_s       * dxx
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
    # Los 5 canales del template tienen que estar en la matriz de diseno.
    # EXCEPCION: si el pulsar usa la parametrizacion FBX (frecuencia orbital),
    # PB = 1/FB0 es una columna REDUNDANTE y degenerada con FB0. Medido en
    # J2214+3000 (Chequeos resonantes/test_pb_fb0.py, 6 semillas): forzarla
    # empeora el numero de condicion 7x y DEBILITA la cota resonante 0.75 dex
    # (6/6 semillas, p=0.03) porque la columna extra absorbe senal. Y es
    # innecesaria: build_context ya arma la parcial de PB desde FB0 por regla
    # de la cadena (d/dPB = d/dFB0 * (-fb0^2)).
    _fb0_libre = (hasattr(model, "FB0")
                  and getattr(getattr(model, "FB0", None), "value", None) is not None
                  and getattr(model.FB0, "frozen", True) is False)
    for p in ("PB", "A1", "EPS1", "EPS2", "TASC"):
        if p == "PB" and _fb0_libre:
            continue
        if hasattr(model, p):
            getattr(model, p).frozen = False
    return model


def model_sini(model):
    """Inclinacion (sin i) del timing model, para el template vectorial.
    Intenta SINI (Shapiro) y, si no, KIN (Kopeikin). Fallback: DEFAULT_SINI."""
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
    """Masa de la companera (M_sun) del timing model (M2, Shapiro) o None."""
    q = getattr(getattr(model, "M2", None), "value", None)
    if q is not None:
        try:
            m2 = float(q)
            if np.isfinite(m2) and m2 > 0:
                return m2
        except Exception:
            pass
    return None


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
    """True si el modelo trae al menos un parametro de Kopeikin con valor."""
    for p in KOPEIKIN_PARAMS:
        o = getattr(model, p, None)
        if o is not None and getattr(o, "value", None) is not None:
            return True
    return False


KOPEIKIN_FIJO = True

# Derivadas seculares a CONGELAR (sacar de la matriz de diseno) antes de armar el
# Pulsar. Coma-separadas; vacio = comportamiento de siempre (lo que diga el .par).
#
# POR QUE: en la resonancia k=1 la respuesta ULDM es un drift secular de la
# orbita, y las columnas PBDOT / A1DOT / EPS1DOT / EPS2DOT modelan lo mismo, asi
# que la marginalizacion se come el pico. Medido con el calculo exacto sobre los
# 8 sistemas que traen PBDOT fiteado: dejarlo libre cuesta 0.55-1.30 dex en el
# nucleo del pico (y 1.3-2.2 dex en el centro). A1DOT en cambio no cuesta nada
# (0.00-0.09 dex): vive en el canal A1, que se lleva ~0% de la norma del template,
# mientras que PBDOT esta en el canal PB, que se lleva ~100%.
#
# Congelar = asumir el valor medido EXACTO, sin barra de error. PBDOT/sigma va de
# 3.3 (J1741+1351) a 330 (J1909-3744). Lo correcto seria un prior gaussiano con
# el sigma medido; esto es el extremo, y esta declarado.
CONGELAR_SECULARES = tuple(
    x.strip() for x in os.environ.get("CONGELAR_SECULARES", "").split(",")
    if x.strip())


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
    # convert_to_ell1 borra KIN/KOM/K96. Solo para los modelos que realmente
    # tienen esos parametros conservamos los residuos del modelo original; la
    # matriz de diseno y los parametros orbitales siguen viniendo del ELL1.
    # En NANOGrav 15yr esto selecciona J1713+0747 y no altera los DD/BT que no
    # contienen parametros de Kopeikin.
    residuos_orig = None
    if (convert_low_ecc and model_ecc(model) < ECC_LOW_THRESHOLD
            and not is_ell1_model(model)):
        if KOPEIKIN_FIJO and tiene_kopeikin(model):
            toas_o = drop_toas_outside_dmx(model, toas)
            model_o = drop_empty_dmx_bins(deepcopy(model), toas_o)
            psr_o = Pulsar(toas_o, model_o, timing_package="pint")
            residuos_orig = np.asarray(psr_o.residuals, dtype=float)
            del psr_o
        model = convert_to_ell1(model)
    toas  = drop_toas_outside_dmx(model, toas)
    model = drop_empty_dmx_bins(model, toas)
    # Congelar seculares ANTES de construir el Pulsar, que es cuando PINT arma la
    # matriz de diseno. Va despues de convert_to_ell1 a proposito: esa funcion
    # DESCONGELA PB/A1/EPS1/EPS2/TASC (las necesita el template) pero no toca las
    # derivadas, asi que no deshace esto.
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
              "pico" % (np.std(delta) * 1e6,
                         (delta.max() - delta.min()) * 1e6),
              flush=True)
    return psr, model


def build_context(psr, timing_model):
    M       = np.asarray(psr.Mmat, dtype=np.float64) # matriz de diseno
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

    # Inclinacion y masas (NUEVO respecto al escalar): el template vectorial
    # necesita sin i (via cot i) y el semieje mayor relativo 'a' (Kepler III).
    # Se toman de TABLA_MASAS (ternas consistentes con la funcion de masa); si el
    # pulsar no esta, se usan los defaults viejos y se avisa.
    psr_name = getattr(psr, "name", "") or ""
    fila = (TABLA_MASAS or {}).get(psr_name)
    if fila is not None:
        m1, m2 = fila["m1"], fila["m2"]
        sin_i = float(np.clip(fila["sini"], 1e-3, 1.0))
        orb_pars["fuente_masas"] = "tabla nivel %s (%s)" % (fila["nivel"], fila["fuente"])
    else:
        sin_i = model_sini(timing_model)
        m2 = model_m2(timing_model)
        if m2 is None:
            m2 = COMPANION_MASS_PRIORS.get(psr_name, DEFAULT_M2)
        m1 = DEFAULT_M1
        orb_pars["fuente_masas"] = "defaults viejos"
        print("    [MASAS] %s no esta en %s: uso defaults (m1=%.2f, m2=%.3f, sin i=%.4f)"
              % (psr_name, MASAS_CSV, m1, m2, sin_i), flush=True)
    orb_pars["sin_i"] = sin_i
    orb_pars["iota"]  = float(np.arcsin(sin_i))
    a_semi = (G_CONST * M_SUN_KG * (m1 + m2) * KEPLER_UNIT_CONV
              / orb_pars["omega_b"] ** 2) ** (1.0 / 3.0)
    orb_pars.update(m1=float(m1), m2=float(m2), a_semi=float(a_semi))

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
# respecto de los parametros orbitales (a partir de la design matrix), y
# los parametros orbitales en si (mas la inclinacion) para evaluar luego la senal

#%% [5] CONSTRUCCION DEL PTA
def make_uldm_signal(t_arr, partials, orb_pars, m_dm):
    from enterprise.signals import parameter, signal_base, deterministic_signals

    @signal_base.function
    def uldm_delay(toas, log10_g=-15.0, rho_f=1.0, gamma_f=0.0,
                   vartheta=np.pi/2, varphi=0.0):
        h = compute_h_template(m_dm, rho_f, gamma_f, vartheta, varphi,
                               t_arr, partials, orb_pars)
        return (10.0 ** log10_g) * h

    # defino los parametros para enterprise: g, rho, gamma y los dos angulos.
    # (las distribuciones reales -Rayleigh en rho, sin(vartheta) en el polar-
    #  se imponen en PulsarULDM_RN.lnprior; aca solo van los rangos)

    lg = parameter.Uniform(LOG10_G_MIN, LOG10_G_MAX)("log10_g")
    rf = parameter.Uniform(RHO_F_MIN, RHO_F_MAX)("rho_f")
    gf = parameter.Uniform(0.0, 2.0*np.pi)("gamma_f")
    vt = parameter.Uniform(0.0, np.pi)("vartheta")
    vp = parameter.Uniform(0.0, 2.0*np.pi)("varphi")
    return deterministic_signals.Deterministic(
        uldm_delay(log10_g=lg, rho_f=rf, gamma_f=gf, vartheta=vt, varphi=vp),
        name="uldm")


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

ULDM_SUFFIXES = ["log10_g", "rho_f", "gamma_f", "vartheta", "varphi"]
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
# cargamos los parametros de ruido y, salvo el red noise, los dejamos fijos a sus
# maximos a posteriori (MAP)

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

#%% [7] CLASE MCMC: 7D POR PULSAR, PRIOR UNIFORME-EN-g
class PulsarULDM_RN(object):
    """
    MCMC de UN pulsar con ruido rojo marginalizado. 7 parametros:
      x[0]=log10_g  x[1]=log10_rho  x[2]=gamma_f  x[3]=cos(vartheta)  x[4]=varphi
      x[5]=log10_A_red  x[6]=gamma_red
    Ruido blanco (EFAC/EQUAD) y DM wideband FIJOS en MAP (self.fix).
    Prior UNIFORME-en-g (10^g): ln p(g) = g*ln10.
    Fases sampleadas DIRECTAMENTE en su forma '_asc' (ver compute_h_template_low_ecc):
    gamma_f = Upsilon_asc (no se le suma m*Tasc), varphi = varphi_asc. Ambas U(0,2pi).
    Angulos: x[3]=cos(vartheta) ~ U(-1,1) (isotropo, prior plano;
    el template recibe vartheta = arccos(x[3])).
    """
    def __init__(self, d):
        self.d   = d
        self.pta = d["pta"]
        self.fix = d["noise_norn"]
        self.lg, self.rf, self.gf = d["lg"], d["rf"], d["gf"]
        self.vt, self.vp          = d["vt"], d["vp"]
        self.arn, self.grn        = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim   = 7

    # este es el objeto de enterprise que permite evaluar la likelihood dados los
    # 7 parametros en un momento cualquiera.
    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.lg]  = x[0]
        p[self.rf]  = 10.0 ** x[1]
        p[self.gf]  = x[2]
        p[self.vt]  = np.arccos(np.clip(x[3], -1.0, 1.0))   # x[3]=cos(vartheta) -> angulo
        p[self.vp]  = x[4]
        p[self.arn] = x[5]
        p[self.grn] = x[6]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_G_MIN <= x[0] <= LOG10_G_MAX):
            return -np.inf
        lp = x[0] * np.log(10.0) if PRIOR_ACOPLE == "uniforme" else 0.0 # prior en g tras hacer: p(g) * dg = p(log10 g) * d(log10 g); [BF] loguniforme -> plano en log10 g
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho  = 10.0 ** u
        lp  += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh en u=log10 rho
        if not (0.0 <= x[2] <= 2.0 * np.pi):
            return -np.inf
        # x[3] = cos(vartheta) ~ U(-1,1): isotropo en la esfera con prior PLANO
        # (la medida sin(vartheta) dvartheta = -d cos(vartheta) ya esta absorbida).
        if not (-1.0 <= x[3] <= 1.0):
            return -np.inf
        # varphi (azimutal): uniforme
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
        x[3] = 0.0                          # cos(vartheta)=0 -> ecuador
        x[4] = np.pi                        # varphi
        x[5] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + 0.1, LOG10_ARED_MAX - 0.1))
        x[6] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + 0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10G**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
                        SIGMA_COS_VARTHETA**2, SIGMA_VARPHI**2,
                        SIGMA_LOG10_ARED**2, SIGMA_GAMMA_RED**2])


#%% [7b] CLASE MCMC REPARAMETRIZADA (g, gamma) -> (X, Y) = (g cos g, g sin g)
class PulsarULDM_RN_reparam(object):
    """
    MISMO modelo fisico que PulsarULDM_RN, pero muestreando la MAGNITUD-FASE del
    acople en coordenadas CARTESIANAS para matar el 'funnel' de resonancia SIN
    congelar los angulos del campo.

    Motivo: en resonancia la senal es  h = g*rho*[cos(gamma) hc(th,ph) + sin(gamma) hs],
    o sea rango-2 en gamma. En (log10_g, gamma) la posterior es una cresta angosta
    con ceguera de fase (cond ~ 1e5) que el MCMC no mezcla (ESS colapsa). En
    X = g cos(gamma),  Y = g sin(gamma), la likelihood es LINEAL en (X,Y) => a
    (rho, th, ph, RN) fijos es una GAUSSIANA exacta, sin funnel. Los angulos siguen
    LIBRES y se marginalizan (isotropo), que es justo lo que se quiere.

    x[0]=X  x[1]=Y  x[2]=log10_rho  x[3]=cos(vartheta)  x[4]=varphi
    x[5]=log10_A_red  x[6]=gamma_red

    Prior: uniforme-en-g (igual que la clase original) + uniforme-en-gamma. En (X,Y)
    eso es  p(X,Y) = p(g)p(gamma)/g ∝ 1/g  =>  ln p += -ln(g),  con g=hypot(X,Y)
    restringido a [10^LOG10_G_MIN, 10^LOG10_G_MAX]. Rayleigh(rho), isotropo(cos th),
    U(varphi) y U(RN) como siempre. La cota de g sale como percentil de
    g = hypot(X,Y) de la cadena (NO via lnZ: ese asumia muestreo en log10_g).

    La escala lineal de (X,Y) se estima una vez (scan barato de la likelihood en g)
    para que el sampler arranque bien escalado pese a que g abarca 15 decadas.
    """
    G_LO = 10.0 ** LOG10_G_MIN
    G_HI = 10.0 ** LOG10_G_MAX

    def __init__(self, d):
        self.pta = d["pta"]
        self.fix = d["noise_norn"]
        self.lg, self.rf, self.gf = d["lg"], d["rf"], d["gf"]
        self.vt, self.vp          = d["vt"], d["vp"]
        self.arn, self.grn        = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.ndim   = 7
        self.sigma_g = self._estimar_escala_g()

    def _lnL_g(self, log10_g, log10_rho=0.0, cos_vt=0.0, varphi=np.pi):
        """lnL a un log10_g dado (rho, angulos fijos, RN en MAP): para estimar escala."""
        p = dict(self.fix)
        p[self.lg]  = log10_g
        p[self.rf]  = 10.0 ** log10_rho
        p[self.gf]  = 0.0
        p[self.vt]  = np.arccos(np.clip(cos_vt, -1.0, 1.0))
        p[self.vp]  = varphi
        p[self.arn] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + .1, LOG10_ARED_MAX - .1))
        p[self.grn] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + .1, GAMMA_RED_MAX - .1))
        return self.pta.get_lnlikelihood(p)

    def _estimar_escala_g(self):
        """
        Escala lineal tipica de (X,Y) ~ ancho de la posterior en g. Scan de lnL(g)
        en el ecuador (cos_vt=0 -> sin_vt=1, senal maxima): se busca el g donde lnL
        cae 2 respecto de la meseta de g bajo (~2 sigma). Si no cae (senal ciega),
        se usa un default holgado. Robusto y barato (~40 evals).
        """
        grid = np.linspace(LOG10_G_MIN, LOG10_G_MAX, 40)
        lnL = np.array([self._lnL_g(g) for g in grid])
        lnL -= np.nanmax(lnL)
        base = np.nanmean(lnL[:5])                     # meseta de g bajo
        caida = np.where((base - lnL) > 2.0)[0]
        if caida.size:
            g_star = 10.0 ** grid[caida[0]]            # ~2 sigma
            return max(g_star, self.G_LO)
        return 10.0 ** (LOG10_G_MAX - 3.0)             # senal no restringe: holgado

    def lnlikelihood(self, x):
        g = np.hypot(x[0], x[1])
        if not (self.G_LO <= g <= self.G_HI):
            return -np.inf                              # fuera de rango: lnprior lo veta igual
        gamma = np.arctan2(x[1], x[0]) % (2.0 * np.pi)
        p = dict(self.fix)
        p[self.lg]  = np.log10(g)
        p[self.rf]  = 10.0 ** x[2]
        p[self.gf]  = gamma
        p[self.vt]  = np.arccos(np.clip(x[3], -1.0, 1.0))
        p[self.vp]  = x[4]
        p[self.arn] = x[5]
        p[self.grn] = x[6]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        g = np.hypot(x[0], x[1])
        if not (self.G_LO <= g <= self.G_HI):
            return -np.inf
        # uniforme-en-g x uniforme-en-gamma  =>  p(X,Y) ∝ 1/g
        lp = -np.log(g)
        u = x[2]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho  = 10.0 ** u
        lp  += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh en log10 rho
        if not (-1.0 <= x[3] <= 1.0):                 # cos(vartheta) isotropo
            return -np.inf
        if not (0.0 <= x[4] <= 2.0 * np.pi):          # varphi
            return -np.inf
        if not (LOG10_ARED_MIN <= x[5] <= LOG10_ARED_MAX):
            return -np.inf
        if not (GAMMA_RED_MIN <= x[6] <= GAMMA_RED_MAX):
            return -np.inf
        return lp

    def x0(self, rng):
        x = np.empty(self.ndim)
        g0 = min(max(self.sigma_g, self.G_LO * 10), self.G_HI / 10)
        gam0 = rng.uniform(0.0, 2.0 * np.pi)
        x[0] = g0 * np.cos(gam0)                       # X
        x[1] = g0 * np.sin(gam0)                       # Y
        x[2] = 0.0                                     # log10_rho
        x[3] = rng.uniform(-0.9, 0.9)                  # cos(vartheta), lejos de los polos
        x[4] = rng.uniform(0.0, 2.0 * np.pi)           # varphi
        x[5] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + .1, LOG10_ARED_MAX - .1))
        x[6] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + .1, GAMMA_RED_MAX - .1))
        return x

    def cov0(self):
        s = self.sigma_g                              # escala lineal de X,Y
        return np.diag([s**2, s**2, SIGMA_LOG10RHO**2,
                        SIGMA_COS_VARTHETA**2, SIGMA_VARPHI**2,
                        SIGMA_LOG10_ARED**2, SIGMA_GAMMA_RED**2])


#%% [7c] CLASE MCMC CON varphi CONGELADO (6D)
class PulsarULDM_RN_fixvarphi(object):
    """
    Igual que PulsarULDM_RN pero con el azimut varphi CONGELADO en VARPHI_FIX
    (no se samplea): saca la componente azimutal del funnel angular de resonancia
    para que la cadena no se trabe. vartheta sigue LIBRE (se marginaliza).

    6 parametros:
      x[0]=log10_g  x[1]=log10_rho  x[2]=gamma_f  x[3]=cos(vartheta)
      x[4]=log10_A_red  x[5]=gamma_red
    Prior uniforme-en-g (igual que la clase original). La columna 0 de la cadena
    sigue siendo log10_g, asi que la extraccion de la cota (lnZ_from_science_chain)
    funciona sin cambios.
    """
    def __init__(self, d, varphi_fix=None):
        self.pta = d["pta"]
        self.fix = d["noise_norn"]
        self.lg, self.rf, self.gf = d["lg"], d["rf"], d["gf"]
        self.vt, self.vp          = d["vt"], d["vp"]
        self.arn, self.grn        = d["arn"], d["grn"]
        self.rn_map = d["rn_map"]
        self.varphi_fix = float(VARPHI_FIX if varphi_fix is None else varphi_fix)
        self.ndim   = 6

    def lnlikelihood(self, x):
        p = dict(self.fix)
        p[self.lg]  = x[0]
        p[self.rf]  = 10.0 ** x[1]
        p[self.gf]  = x[2]
        p[self.vt]  = np.arccos(np.clip(x[3], -1.0, 1.0))
        p[self.vp]  = self.varphi_fix            # varphi CONGELADO
        p[self.arn] = x[4]
        p[self.grn] = x[5]
        return self.pta.get_lnlikelihood(p)

    def lnprior(self, x):
        if not (LOG10_G_MIN <= x[0] <= LOG10_G_MAX):
            return -np.inf
        lp = x[0] * np.log(10.0)                  # uniforme-en-g
        u = x[1]
        if not (LOG10_RHO_MIN <= u <= LOG10_RHO_MAX):
            return -np.inf
        ln10 = np.log(10.0)
        rho  = 10.0 ** u
        lp  += np.log(2.0 * ln10) + 2.0 * u * ln10 - rho * rho   # Rayleigh
        if not (0.0 <= x[2] <= 2.0 * np.pi):     # gamma
            return -np.inf
        if not (-1.0 <= x[3] <= 1.0):            # cos(vartheta) isotropo
            return -np.inf
        if not (LOG10_ARED_MIN <= x[4] <= LOG10_ARED_MAX):
            return -np.inf
        if not (GAMMA_RED_MIN <= x[5] <= GAMMA_RED_MAX):
            return -np.inf
        return lp

    def x0(self, rng):
        x = np.empty(self.ndim)
        x[0] = LOG10_G_INIT + rng.uniform(-2.0, 2.0)
        x[1] = 0.0
        x[2] = np.pi
        x[3] = 0.0                               # cos(vartheta)=0 -> ecuador
        x[4] = float(np.clip(self.rn_map[0], LOG10_ARED_MIN + 0.1, LOG10_ARED_MAX - 0.1))
        x[5] = float(np.clip(self.rn_map[1], GAMMA_RED_MIN + 0.1, GAMMA_RED_MAX - 0.1))
        return x

    def cov0(self):
        return np.diag([SIGMA_LOG10G**2, SIGMA_LOG10RHO**2, SIGMA_GAMMA**2,
                        SIGMA_COS_VARTHETA**2,
                        SIGMA_LOG10_ARED**2, SIGMA_GAMMA_RED**2])


#%% [8] PRELOAD + RUNNER + ESS
def preload_one(name, m_eV):
    m_dm = m_eV * EV_TO_INV_S
    t_load = time.time()
    try:
        par, tim, chain, pars = resolve_paths(name)
        psr, tm = load_pulsar(par, tim, convert_low_ecc=True)
        t_arr, partials, orb_pars, missing = build_context(psr, tm)
        # Un canal sin columna en la matriz de diseno quedaria en CERO en el template
        # sin aviso (build_context lo rellena con ceros). Hoy no pasa en ninguno de los
        # 50 (PB/A1/EPS1/EPS2/TASC o PB/A1/ECC/OM/T0 fiteados; FB0 via regla de la cadena).
        if missing:
            raise RuntimeError(f"{psr.name}: canales del template sin columna en el diseno: {missing}")
        pta = build_pta(psr, t_arr, partials, orb_pars, m_dm)
        lg, rf, gf, vt, vp = get_uldm_pnames(pta)
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
        return dict(ctx=(t_arr, partials, orb_pars, m_dm),
                    uldm_resto={rf: 1.0, gf: 0.0, vt: 0.5 * np.pi, vp: 0.0},
                    name=name, pta=pta, lg=lg, rf=rf, gf=gf, vt=vt, vp=vp,
                    arn=arn, grn=grn, noise_norn=noise_norn, rn_map=rn_map,
                    ecc=orb_pars["ecc"], sin_i=orb_pars["sin_i"],
                    n_toa=len(t_arr), t_load=time.time()-t_load)
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
    return data[nburn:, 0]              # columna 0 = log10_g

#%% [9] Z_i(g) DESDE LA CADENA + COMBINACION
def lnZ_from_science_chain(g_samples):
    """
    La cadena se sampleo con prior uniforme-en-g -> su marginal es la posterior
    p_i(g) ∝ 10^g Z_i(g).  Recuperamos ln Z_i(g) = ln p_hat_i(g) - g*ln10
    sobre el rango bien muestreado. Debajo del codo (donde termina la meseta):
    extrap. PLANA. Arriba del drop: -inf.
    (aca 'g' denota log10 del acople; la columna 0 de la cadena es log10_g)
    """
    ln10 = np.log(10.0)
    counts, _ = np.histogram(g_samples, bins=G_EDGES)
    dens = counts.astype(float)
    if SMOOTH_SIGMA_BINS > 0:
        dens = gaussian_filter1d(dens, SMOOTH_SIGMA_BINS, mode="nearest")
    # hago smooth el histograma con un filtro gaussiano
    lnZ  = np.full(G_CENTERS.shape, -np.inf)
    dmax = dens.max() # maximo de la campana
    if dmax <= 0:
        return lnZ

    # TAIL_FRAC es el valor en el cual considero que se "apago" la distribucion, 0.05 del maximo
    good = dens > TAIL_FRAC * dmax
    idx  = np.where(good)[0]
    i_lo, i_hi = idx[0], idx[-1]
    sl = slice(i_lo, i_hi + 1)
    d_sl = np.maximum(dens[sl], TAIL_FRAC * dmax)
    lnZ[sl] = np.log(d_sl) - G_CENTERS[sl] * ln10
    n_anch = max(3, int(round(0.7 / DG)))
    lnZ[:i_lo] = np.median(lnZ[i_lo:i_lo + n_anch])

    # [CORREGIDO] Cola superior: EXTRAPOLAR en vez de cortar con -inf.
    # Antes, arriba de i_hi quedaba lnZ = -inf, o sea "Z = 0 exactamente" en una
    # region donde la cadena simplemente NO muestreo. Al combinar pulsares
    # (suma de lnZ_i) la suma hereda el -inf en cuanto CUALQUIER sumando lo
    # tiene => el soporte de la posterior conjunta lo fijaba el pulsar mas
    # restrictivo, o sea HASTA DONDE LLEGO ESA CADENA, no la likelihood. Medido
    # en la resonancia de J1713 (47 pulsares): el UL95 conjunto caia a 0.086 dex
    # del borde del soporte, y la cota salia 0.117 dex OPTIMISTA (factor 1.3).
    #
    # Para un resultado nulo la evidencia cae como Z(g) ~ exp(-c g^2), o sea
    # lnZ = a + b*g^2 con b<0. Se ajusta esa forma en la cola bien muestreada y
    # se continua. (Extrapolar LINEAL en log10(g) tambien saca el -inf pero es
    # groseramente conservador: da +1.5 dex y hace evaporar la cota del pulsar
    # dominante hasta el techo del prior.) Con la cuadratica el cambio tipico
    # por pulsar es +0.013 dex (max +0.19) y el UL queda a 1.3 dex del borde.
    if EXTRAPOLAR_COLA and i_hi + 1 < len(G_CENTERS):
        j0 = max(i_lo, i_hi - n_anch + 1)
        x_hi = G_CENTERS[i_hi]
        # u = (g/g_hi)^2 normalizado (u=1 en el borde) para no desbordar
        u_fit = 10.0 ** (2.0 * (G_CENTERS[j0:i_hi + 1] - x_hi))
        if len(u_fit) >= 2:
            b, a = np.polyfit(u_fit, lnZ[j0:i_hi + 1], 1)   # lnZ = a + b*u
            if b < 0:                                        # solo si cae
                u_ex = 10.0 ** (2.0 * (G_CENTERS[i_hi + 1:] - x_hi))
                with np.errstate(over="ignore"):
                    lnZ[i_hi + 1:] = a + b * u_ex
    return lnZ


def posterior_from_lnZ(lnZ_total, grid=None):
    """Posterior conjunta P(log10_g) sobre una grilla (G_FINE por
    defecto). lnZ_total esta definido en G_CENTERS (histograma); se interpola
    a la grilla fina: debajo del rango -> plano (clamp), arriba del
    drop -> -inf (Z_i->0). Devuelve (post, ul95) sobre 'grid'."""
    if grid is None:
        grid = G_FINE
    ln10 = np.log(10.0)
    fin = np.isfinite(lnZ_total)
    b_fin, z_fin = G_CENTERS[fin], lnZ_total[fin]
    lnZ_g = np.interp(grid, b_fin, z_fin)            # (abajo: plano)
    lnZ_g[grid > b_fin.max()] = -np.inf              # arriba del drop: Z_i -> 0
    ln_post = grid * ln10 + lnZ_g                    # prior uniforme-en-g, una vez
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
        g_chain = run_one_chain(model, outDir, n_iter, rng)
        lnZ    = lnZ_from_science_chain(g_chain)
        ul_sci = float(np.percentile(g_chain, 95.0))
        ess    = effective_sample_size(g_chain)
        print(f"  [m#{mass_idx:03d} {name:12s}] OK  n_toa={d['n_toa']:5d}  "
              f"ecc={d['ecc']:.1e}  UL95={ul_sci:6.2f}  "
              f"g=[{g_chain.min():6.1f},{g_chain.max():6.1f}]  ESS={ess:7.0f}  "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)
        return dict(mass_idx=mass_idx, m_eV=m_eV, name=name, ok=True,
                    g=g_chain, lnZ=lnZ, ul_sci=ul_sci, ess=ess,
                    g_min=float(g_chain.min()), g_max=float(g_chain.max()),
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
        g_chain = np.asarray(r["g"])
        ul   = r["ul_sci"]
        counts, _ = np.histogram(g_chain, bins=G_EDGES)
        h = counts / max(counts.max(), 1)
        ax.bar(G_CENTERS, h, width=DG, color="0.7", edgecolor="none",
               align="center", label=r"cadena = $p_i(g)\propto 10^g Z_i$")
        ax.axvline(ul, color="crimson", ls="--", lw=1.3, label="UL95 (percentil)")
        g_lin = 10.0 ** ul
        ax.set_title(f"{r['name']}   g$\\in$[{g_chain.min():.1f}, {g_chain.max():.1f}]\n"
                     f"UL95={ul:.2f}  $g$={g_lin:.1e}  ESS={r.get('ess',0):.0f}",
                     fontsize=8)
        ax.set_xlabel(r"$\log_{10} g$", fontsize=8)
        ax.set_xlim(g_chain.min() - 0.5, max(g_chain.max(), ul) + 0.5)
        ax.set_ylim(0, 1.05)
        ax.grid(True, ls="--", alpha=0.3)
        if k == 0:
            ax.legend(fontsize=6, loc="upper left")
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Posteriors INDIVIDUALES (uniforme-en-g, RN marg., m={m_eV:.0e} eV)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(outpng, dpi=140); plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_joint(post, ul95, names, m_eV, outpng):
    g_lin = 10.0 ** ul95
    fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=120)
    ax.plot(G_FINE, post, color="darkgreen", lw=1.8)
    ax.fill_between(G_FINE, post, where=(G_FINE <= ul95),
                    color="mediumseagreen", alpha=0.35, label="95% de la masa")
    ax.axvline(ul95, color="crimson", ls="--", lw=1.6,
               label=f"95% UL: log10(g)={ul95:.2f}\ng={g_lin:.2e}")
    ax.set_xlabel(r"$\log_{10} g$", fontsize=13)
    ax.set_ylabel(r"posterior conjunto $P(\log_{10} g)$", fontsize=13)
    ax.set_title(f"COTA CONJUNTA ({len(names)} pulsares, RN marg., m={m_eV:.0e} eV)",
                 fontsize=11)
    m = post > post.max() * 1e-3
    ax.set_xlim(G_FINE[m].min() - 1, G_FINE[m].max() + 1)
    ax.legend(loc="best", fontsize=9); ax.grid(True, ls="--", alpha=0.3)
    plt.tight_layout(); plt.savefig(outpng, dpi=140); plt.close(fig)
    print(f"  guardado: {outpng}")


def plot_vs_mass(m_arr, g_arr, n_used, outpng):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    ax.loglog(m_arr, g_arr, "-", color="black", lw=1.4)
    ax.set_xlabel(r"$m$ [eV]", fontsize=12)
    ax.set_ylabel(r"$g$  (cota conjunta, 95% UL)", fontsize=12)
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
    print(f"COTA CONJUNTA POR-PULSAR VECTORIAL (RN marginalizado) | "
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
    print(f"    Cada tarea = 1 MCMC 7D de {N_ITER} iters. ETA sobre tareas completadas.\n")

    # 3) lanzar y combinar cada masa cuando llegan todos sus pulsares
    mass_results  = defaultdict(list)   # mass_idx -> [result dicts ok]
    mass_received = defaultdict(int)
    joint         = {}                  # mass_idx -> dict(ul95, g_lin, n_used, post)
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
                    g_lin = 10.0 ** ul95
                    joint[mi] = dict(ul95=ul95, g_lin=g_lin, n_used=len(lst), post=post)
                    print(f"   >>> masa #{mi:03d}  m={MASS_GRID_EV[mi]:.3e} eV   "
                          f"COTA CONJUNTA log10(g)={ul95:.3f}  g={g_lin:.3e}   "
                          f"({len(lst)}/{len(use)} pulsares)", flush=True)
                else:
                    print(f"   >>> masa #{mi:03d}: sin pulsares OK, se omite.", flush=True)

    if not joint:
        raise SystemExit("Ninguna masa con cota; revisar errores de arriba.")

    # 4) ensamblar + guardar
    idx_ok    = sorted(joint.keys())
    m_arr     = np.array([MASS_GRID_EV[i]    for i in idx_ok])
    ul_arr    = np.array([joint[i]["ul95"]   for i in idx_ok])
    g_arr     = np.array([joint[i]["g_lin"]  for i in idx_ok])
    nused_arr = np.array([joint[i]["n_used"] for i in idx_ok])

    npz = os.path.join(OUT_ROOT, "DEFINITIVO_massweep_joint_vectorial.npz")
    np.savez(npz, masses_eV=m_arr, log10_g_95UL_joint=ul_arr,
             g_95UL_joint=g_arr, n_pulsars_used=nused_arr,
             names=np.array(use), g_centers=G_CENTERS, g_fine=G_FINE)
    print(f"\n[3] SAVE {npz}")

    # 5) plots
    print(f"\n[4] Plots...")
    if single:
        mi  = idx_ok[0]
        lst = sorted(mass_results[mi], key=lambda r: use.index(r["name"]))
        print(f"\n  ---- UL95 individual por pulsar (m={m_arr[0]:.2e} eV) ----")
        print(f"    {'pulsar':15s} {'n_toa':>6s} {'ecc':>8s} {'UL95':>8s} "
              f"{'g':>11s} {'ESS':>8s}")
        for r in lst:
            gv = 10.0 ** r["ul_sci"]
            flag = "  <-- g_max bajo?" if r["g_max"] < r["ul_sci"] - 0.3 else ""
            print(f"    {r['name']:15s} {r['n_toa']:6d} {r['ecc']:8.1e} "
                  f"{r['ul_sci']:8.3f} {gv:11.2e} {r['ess']:8.0f}{flag}")
        print(f"\n  ====== COTA CONJUNTA ({nused_arr[0]} pulsares) ======")
        print(f"    log10_g_95UL = {ul_arr[0]:.3f}")
        print(f"    g_95UL       = {g_arr[0]:.3e}")
        plot_individuals(lst, m_arr[0], os.path.join(OUT_ROOT, "DEFINITIVO_individuales_vectorial.png"))
        plot_joint(joint[mi]["post"], ul_arr[0], use, m_arr[0],
                   os.path.join(OUT_ROOT, "DEFINITIVO_conjunto_vectorial.png"))
    else:
        plot_vs_mass(m_arr, g_arr, nused_arr,
                     os.path.join(OUT_ROOT, "DEFINITIVO_g_vs_m_vectorial.png"))

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
VERSION_PIPELINE = "final-2026-10-10-dc05"  # 10-09-masas + Delta c = 0.5


def crear_modelo(d):
    return PulsarULDM_RN(d)
