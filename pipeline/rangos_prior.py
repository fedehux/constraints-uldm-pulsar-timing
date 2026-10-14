"""rangos_prior.py -- prior ranges of the final runs (single source of truth).

Same range for the bound runs (uniform prior) and the Bayes-factor runs
(log-uniform prior), for every pulsar and every mass:
  * floor  <= (lowest onset of sensitivity of any pulsar) - 1.5 dex, so that the
    first dex of the prior (the BF plateau) is flat for every chain;
  * ceiling >= (highest combined 95% bound) + 2 dex, so that no bound is close
    to the edge.
All values on the 0.005-dex bin lattice.  Final (06/10/2026):
  alpha [-26, -10] phys: needs >= -10.36 / <= -25.96.
  beta  [-92, -60] (production): needs >= -64.2 / <= -91.6.
  g     [-30, -14]: with Delta c = 0.1 it needed <= -27.16 / >= -12.8 (2-dex margin),
        but ENTERPRISE chains stick to a -13 ceiling (round-off), hence -14.
        With Delta c = 0.5 (10/10/2026) every bound moves -0.699 dex: the floor
        needs <= -27.86 and the smallest margin to the ceiling is 1.53 dex.
"""
import os

# code units (alpha: physical log10(alpha/GeV^-1); beta, g: as in the pipelines)
RANGOS = {
    "alpha": {"LOG10_ALPHA_PHYS_MIN": -26.0, "LOG10_ALPHA_PHYS_MAX": -10.0},
    "beta": {"LOG10_BETA_MIN": -92.0, "LOG10_BETA_MAX": -60.0},
    "vec": {"LOG10_G_MIN": -30.0, "LOG10_G_MAX": -14.0},
}


# production ranges, ONLY for smoke tests (RANGOS_PROVISORIOS=1)
_PROVISORIOS = {"alpha": {"LOG10_ALPHA_PHYS_MIN": -24.0, "LOG10_ALPHA_PHYS_MAX": -14.0},
                "beta": {"LOG10_BETA_MIN": -92.0, "LOG10_BETA_MAX": -60.0},
                "vec": {"LOG10_G_MIN": -30.0, "LOG10_G_MAX": -15.0}}


def exportar(acople):
    if os.environ.get("RANGOS_PROVISORIOS") == "1":
        print("[rangos_prior] *** RANGOS PROVISORIOS (solo prueba) ***", flush=True)
        for k, v in _PROVISORIOS[acople].items():
            os.environ[k] = repr(float(v))
        return
    for k, v in RANGOS[acople].items():
        if v is None:
            raise SystemExit(f"rangos_prior.py: {k} sin fijar (correr validacion/v05 primero)")
        os.environ[k] = repr(float(v))
