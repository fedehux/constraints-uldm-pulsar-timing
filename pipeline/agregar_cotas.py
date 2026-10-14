#!/usr/bin/env python
# =============================================================================
# agregar_cotas.py  --  histogramas -> cota combinada al 95% vs masa
# =============================================================================
# Post-procesado puro: todo se recalcula desde los histogramas guardados, asi
# que cualquier perilla se cambia sin volver a correr cadenas.
#
# Por (masa, pulsar), con la cadena de prior UNIFORME en el acople theta:
#   p_i(x) ~ 10^x Z_i(x)  (x = log10 theta)   ->   ln Z_i(x) = ln p_i(x) - x ln10
#   1) histograma CRUDO de 0.005 dex (el worker no suaviza nada); opcionalmente
#      suavizado aqui con una gaussiana de SUAVIZADO dex (SUAVIZADO=0: sin suavizar);
#   2) donde la densidad suavizada supera TAIL_FRAC de su maximo se usa tal cual;
#   3) por debajo (theta chico, la cadena casi no va): meseta, ln Z_i constante
#      = mediana de los primeros ANCLA dex medidos (la likelihood ya no cambia);
#   4) por encima (theta grande, la cadena no llega): cola en LEY DE POTENCIAS,
#      Z_i ~ theta^EXPONENTE, con EXPONENTE = -2 para alpha y g (senal lineal en
#      el campo: P(campo debil) ~ (c/theta)^2) y -1 para beta (senal ~ rho^2,
#      rho^2 ~ Exp(1)); el offset con la mediana de los ultimos ANCLA dex.
# Combinacion:  ln p(x) = x ln10 + sum_i ln Z_i(x);  cota = cuantil 95%.
# Perillas (entorno): SUAVIZADO (0.03), TAIL_FRAC (0.05), ANCLA (0.7),
#   COLA (potencias | gaussiana | corte), EXPONENTE (por defecto segun acople).
#
# Controles que imprime: masas incompletas, cadenas con muestras pegadas al
# techo/piso, ESS minimo, margen de la cota al techo del prior.
#
# Uso:  ACOPLE=alpha OUT_ROOT=... python agregar_cotas.py
# =============================================================================
import os
import sys
import glob
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

AQUI = os.path.dirname(os.path.abspath(__file__))
ACOPLE = os.environ["ACOPLE"]
OUT_ROOT = os.environ["OUT_ROOT"]
SUAV = float(os.environ.get("SUAVIZADO", "0.03"))
TAIL = float(os.environ.get("TAIL_FRAC", "0.05"))
ANCLA = float(os.environ.get("ANCLA", "0.7"))
COLA = os.environ.get("COLA", "potencias")
EXPO = float(os.environ.get("EXPONENTE", "-1.0" if ACOPLE == "beta" else "-2.0"))
N_ESP = int(os.environ.get("N_PSR_EXPECTED", "50"))
LN10 = np.log(10.0)
ESCALA_FIS = {"alpha": lambda x: x + np.log10(np.sqrt(2.30817405213802e+48)),
              "beta": lambda x: 0.5 * (x + np.log10(2.30817405213802e+48)),   # log10 sqrt(beta) [GeV^-1]
              "vec": lambda x: x}[ACOPLE]


def lnZ_de_counts(c, centros, dx):
    dens = np.asarray(c, float)                      # raw counts as saved by the worker
    if SUAV > 0:                                     # optional smoothing, only here
        dens = gaussian_filter1d(dens, SUAV / dx, mode="nearest")
    lnZ = np.full(centros.shape, -np.inf)
    if dens.max() <= 0:
        return lnZ
    idx = np.flatnonzero(dens > TAIL * dens.max())
    i0, i1 = idx[0], idx[-1]
    lnZ[i0:i1 + 1] = np.log(np.maximum(dens[i0:i1 + 1], TAIL * dens.max())) - centros[i0:i1 + 1] * LN10
    na = max(3, int(round(ANCLA / dx)))
    lnZ[:i0] = np.median(lnZ[i0:i0 + na])
    if i1 + 1 < len(centros) and COLA != "corte":
        j0 = max(i0, i1 - na + 1)
        xh = centros[i1]
        if COLA == "potencias":
            a0 = np.median(lnZ[j0:i1 + 1] - EXPO * LN10 * (centros[j0:i1 + 1] - xh))
            lnZ[i1 + 1:] = a0 + EXPO * LN10 * (centros[i1 + 1:] - xh)
        else:  # gaussiana
            u = 10.0 ** (2.0 * (centros[j0:i1 + 1] - xh))
            b, a0 = np.polyfit(u, lnZ[j0:i1 + 1], 1)
            if b < 0:
                lnZ[i1 + 1:] = a0 + b * 10.0 ** (2.0 * (centros[i1 + 1:] - xh))
    return lnZ


def cota95(centros, lnZ_tot):
    fino = np.linspace(centros[0], centros[-1], int(round((centros[-1] - centros[0]) / 0.002)) + 1)
    fin = np.isfinite(lnZ_tot)
    lz = np.interp(fino, centros[fin], lnZ_tot[fin])
    lz[fino > centros[fin].max()] = -np.inf
    lp = fino * LN10 + lz
    lp -= np.nanmax(lp)
    p = np.exp(lp); p[~np.isfinite(p)] = 0
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(fino))])
    return float(np.interp(0.95, cdf / cdf[-1], fino))


def main():
    fs = sorted(glob.glob(os.path.join(OUT_ROOT, "hist", "m_*__*__s0.npz")))
    if not fs:
        raise SystemExit("no hay histogramas")
    por = defaultdict(dict); meta = {}
    avisos = defaultdict(list)
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            if str(z["prior"]) != "uniforme" or str(z["acople"]) != ACOPLE:
                raise SystemExit(f"{f}: prior={z['prior']} acople={z['acople']}")
            ms, n = str(z["m_str"]), str(z["name"])
            por[ms][n] = (np.asarray(z["counts"]), float(z["ess"]), float(z["frac_techo"]))
            meta.setdefault("rango", (float(z["log10_min"]), float(z["log10_max"]), int(z["n_bins"])))
            meta.setdefault("version", str(z["version"]))
            if float(z["frac_techo"]) > 0.5:
                avisos[ms].append(f"{n} al techo ({float(z['frac_techo']):.0%})")
    lo, hi, nb = meta["rango"]
    edges = np.linspace(lo, hi, nb + 1); centros = 0.5 * (edges[1:] + edges[:-1]); dx = edges[1] - edges[0]
    filas = []
    for ms in sorted(por, key=float):
        tot = np.zeros(nb)
        for n, (c, ess, ft) in por[ms].items():
            tot = tot + lnZ_de_counts(c, centros, dx)
        ul = cota95(centros, tot)
        esss = [v[1] for v in por[ms].values()]
        filas.append((float(ms), ul, len(por[ms]), min(esss), float(np.median(esss)), hi - ul))
    m = np.array([f[0] for f in filas]); ul = np.array([f[1] for f in filas])
    npsr = np.array([f[2] for f in filas]); ess_min = np.array([f[3] for f in filas])
    margen = np.array([f[5] for f in filas])
    print(f"[agregar_cotas] {ACOPLE} version={meta['version']}  prior [{lo:.3f}, {hi:.3f}]  {nb} bins  "
          f"SUAVIZADO={SUAV} TAIL={TAIL} ANCLA={ANCLA} COLA={COLA} EXPONENTE={EXPO}")
    print(f"  masas {len(m)}; completas ({N_ESP} pulsares): {int((npsr >= N_ESP).sum())}; "
          f"incompletas: {int((npsr < N_ESP).sum())}")
    print(f"  margen minimo de la cota al techo del prior: {margen.min():.2f} dex (m = {m[np.argmin(margen)]:.3e})"
          f"{'   *** MENOS DE 1.5 dex: subir el techo ***' if margen.min() < 1.5 else ''}")
    print(f"  ESS minimo {ess_min.min():.0f}; mediana de ESS minimos por masa {np.median(ess_min):.0f}")
    na = sum(len(v) for v in avisos.values())
    print(f"  cadenas con >50% de muestras a menos de 0.5 dex del techo: {na} "
          f"(son pulsares sin sensibilidad a esa masa; su ln Z_i es una meseta)")
    out = os.path.join(OUT_ROOT, f"COTAS_{ACOPLE}.npz")
    np.savez(out, masses_eV=m, log10_ul_codigo=ul, ul_fisico_log10=ESCALA_FIS(ul), n_pulsares=npsr,
             ess_min=ess_min, margen_techo=margen, prior=(lo, hi), suavizado=SUAV, tail=TAIL,
             ancla=ANCLA, cola=COLA, exponente=EXPO, version=meta["version"])
    print(f"  guardado {out}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
        ax.loglog(m, 10 ** ESCALA_FIS(ul), "k-", lw=1)
        ax.set_xlabel("m [eV]"); ax.set_ylabel({"alpha": r"$\alpha$ [GeV$^{-1}$]",
                                                "beta": r"$\sqrt{\beta}$ [GeV$^{-1}$]", "vec": r"$g_{B-L}$"}[ACOPLE])
        ax.grid(True, which="both", ls="--", alpha=0.3); fig.tight_layout()
        fig.savefig(out.replace(".npz", ".png"))
    except Exception as e:
        print("  [sin grafico]", e)


if __name__ == "__main__":
    main()
