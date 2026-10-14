#!/usr/bin/env python
# =============================================================================
# agregar_bf.py  --  Bayes factor combinado BF(m) desde los histogramas
# [FINAL] = aggregate_bf.py validado + una sola diferencia: una cadena SIN
# muestras en la meseta ya no se descarta del producto (eso sesgaba B10 hacia
# abajo, sacando justo al pulsar que mas empuja). Se le asigna UNA cuenta en la
# meseta, lo que da una COTA INFERIOR de B10 en esa masa (marcada).
# =============================================================================
# Cada cadena corrio con prior PLANO en x = log10(acople) sobre [LO, HI], de
# ancho W = HI - LO. Entonces su densidad en x es
#
#     p_i(x) = Z_i(x) / (W E_i),        E_i = (1/W) int Z_i(x) dx ,
#
# y como para acoples chicos la senal es indetectable, Z_i(x) -> Z_i(0): p_i
# tiene una MESETA p_i^0 en el extremo bajo del prior. De ahi
#
#     Z_i(x) / Z_i(0) = p_i(x) / p_i^0 ,
#     B_i  = (1/W) int Z_i/Z_i(0) dx = 1 / (W p_i^0)          (un pulsar)
#     B_10 = (1/W) int prod_i Z_i(x)/Z_i(0) dx                  (combinado)
#
# que es la definicion del paper con prior log10(acople) ~ U(LO, HI).
#
# Los histogramas se agrupan en bines de ~REBIN_DEX: con los bines nativos el
# producto de 50 histogramas ruidosos sesgaria el promedio.
#
# Controles que se imprimen:
#   * MESETA: densidad en [LO, LO+M/2] vs [LO+M/2, LO+M] por pulsar. Si no es
#     plana (|dif| > max(FLAT_TOL, 3 sigma)), el extremo bajo del prior todavia
#     "ve" senal y Z_i(0) esta mal estimado -> se marca.
#   * SIGMA: bootstrap parametrico. Cada histograma se remuestrea como
#     multinomial con N_eff = N / tau_i (tau_i = n_muestras / ESS_i, la
#     autocorrelacion) y se recalcula B_10. La meseta entra en numerador y
#     denominador, asi que su ruido se cancela en parte: por eso bootstrap y no
#     una formula de Poisson (validado en chequeo_bf.py agregador).
#
# Config por entorno:
#   ACOPLE (alpha|beta|vec, req.), OUT_ROOT (req.), N_PSR_EXPECTED (50)
#   REBIN_DEX (0.2)  ancho aproximado de los bines gruesos en log10(acople)
#   MESETA_DEX (1.0) ancho de la meseta usada para p_i^0
#   FLAT_TOL (0.1)   tolerancia en dex del control de meseta
#   N_BOOT (200)     realizaciones del bootstrap
# =============================================================================
import os
import glob
from collections import defaultdict
import numpy as np

ACOPLE   = os.environ.get("ACOPLE", "").strip().lower()
OUT_ROOT = os.environ["OUT_ROOT"]
N_ESP    = int(os.environ.get("N_PSR_EXPECTED", "50"))
REBIN    = float(os.environ.get("REBIN_DEX", "0.2"))
MESETA   = float(os.environ.get("MESETA_DEX", "1.0"))
FLAT_TOL = float(os.environ.get("FLAT_TOL", "0.1"))
N_BOOT   = int(os.environ.get("N_BOOT", "200"))
HISTDIR  = os.path.join(OUT_ROOT, "hist")
NOMBRE   = {"alpha": "alpha", "beta": "beta", "vec": "g"}.get(ACOPLE)
if NOMBRE is None:
    raise SystemExit(f"ACOPLE={ACOPLE!r}: alpha | beta | vec")


def rebin(counts, lo, hi):
    """Agrupa los bines nativos en bloques de ~REBIN dex. El resto se suma al
    ultimo bloque (con su ancho real). Devuelve cuentas, bordes izq y der."""
    n = len(counts)
    dx = (hi - lo) / n
    f = max(1, int(round(REBIN / dx)))
    cortes = list(range(0, n, f))
    if len(cortes) > 1 and n - cortes[-1] < f:
        cortes = cortes[:-1]
    cortes.append(n)
    c = np.array([counts[a:b].sum() for a, b in zip(cortes[:-1], cortes[1:])],
                 dtype=float)
    return c, lo + dx * np.array(cortes[:-1]), lo + dx * np.array(cortes[1:])


def log10_B(C, w, en_meseta, W):
    """C: (n_psr, n_bins) cuentas. Devuelve (log10 B10, log10 B_i por pulsar).
    Un pulsar sin cuentas en la meseta devuelve nan en B_i y se excluye."""
    C = C.copy()
    sin = (C[:, en_meseta].sum(axis=1) == 0) & (C.sum(axis=1) > 0)
    C[sin, np.flatnonzero(en_meseta)[0]] = 1.0          # [FINAL] lower bound
    N = C.sum(axis=1, keepdims=True)
    c0 = C[:, en_meseta].sum(axis=1)
    a0 = w[en_meseta].sum()
    valido = (N[:, 0] > 0) & (c0 > 0)
    p = C[valido] / (N[valido] * w)
    p0 = (c0[valido] / (N[valido, 0] * a0))[:, None]
    with np.errstate(divide="ignore"):
        ln_prod = np.log(p / p0).sum(axis=0)      # -inf donde algun c = 0
    fin = np.isfinite(ln_prod)
    B = (w[fin] * np.exp(ln_prod[fin])).sum() / W
    Bi = np.full(C.shape[0], np.nan)
    Bi[valido] = -np.log10(W * p0[:, 0])
    return (np.log10(B) if B > 0 else -np.inf), Bi


def main():
    files = sorted(glob.glob(os.path.join(HISTDIR, "m_*__*__s*.npz")))
    if not files:
        raise SystemExit(f"No hay histogramas en {HISTDIR}")

    por_masa = defaultdict(dict)       # clave -> pulsar -> dict
    meta = {}
    for f in files:
        with np.load(f, allow_pickle=True) as d:
            if str(d["prior"]) != "loguniforme":
                raise SystemExit(f"{f}: prior={d['prior']} (se esperaba loguniforme)")
            k = f"{float(d['m_eV']):.10e}"
            n = str(d["name"])
            c = np.asarray(d["counts"], dtype=float)
            ess, nm = float(d["ess"]), float(d["n_muestras"])
            e = por_masa[k].get(n)
            if e is None:
                por_masa[k][n] = dict(c=c, ess=ess, nm=nm)
            else:                        # pool de semillas
                e["c"] += c; e["ess"] += ess; e["nm"] += nm
            meta[k] = (float(d["m_eV"]), float(d["log10_min"]), float(d["log10_max"]))

    rng = np.random.default_rng(12345)
    todos = sorted({n for v in por_masa.values() for n in v})
    filas = []
    for k in sorted(por_masa, key=float):
        m_eV, lo, hi = meta[k]
        W = hi - lo
        nombres = sorted(por_masa[k])
        C, taus, no_plano = [], [], []
        for n in nombres:
            e = por_masa[k][n]
            c, izq, der = rebin(e["c"], lo, hi)
            C.append(c)
            tau = max(e["nm"] / max(e["ess"], 1.0), 1.0)
            taus.append(tau)
            # control de meseta: dos mitades de [LO, LO+MESETA]
            ia = (izq >= lo - 1e-9) & (der <= lo + MESETA / 2 + 1e-9)
            ib = (izq >= lo + MESETA / 2 - 1e-9) & (der <= lo + MESETA + 1e-9)
            ca, cb = c[ia].sum(), c[ib].sum()
            if ca > 0 and cb > 0:
                dif = np.log10((ca / (der[ia] - izq[ia]).sum()) /
                               (cb / (der[ib] - izq[ib]).sum()))
                sig = np.sqrt(tau * (1 / ca + 1 / cb)) / np.log(10)
                if abs(dif) > max(FLAT_TOL, 3 * sig):
                    no_plano.append(f"{n}({dif:+.2f})")
        C = np.array(C)
        taus = np.array(taus)
        w = der - izq
        en_meseta = der <= lo + MESETA + 1e-9
        lB, Bi = log10_B(C, w, en_meseta, W)
        sin_meseta = [n for n, c in zip(nombres, C) if c[en_meseta].sum() == 0]   # -> B10 es cota inferior

        # bootstrap parametrico con N_eff = N / tau por pulsar
        boot = []
        N = C.sum(axis=1)
        neff = np.maximum((N / taus).astype(int), 1)
        prob = C / np.maximum(N[:, None], 1)
        for _ in range(N_BOOT):
            Cb = np.array([rng.multinomial(ne, pr) for ne, pr in zip(neff, prob)],
                          dtype=float)
            b, _ = log10_B(Cb, w, en_meseta, W)
            if np.isfinite(b):
                boot.append(b)
        sig = float(np.std(boot)) if len(boot) > 1 else np.nan

        ess_min = min(por_masa[k][n]["ess"] for n in nombres)
        filas.append(dict(m=m_eV, lB=lB, sig=sig, n=int(np.isfinite(Bi).sum()),
                          sin_meseta=sin_meseta, no_plano=no_plano,
                          ess_min=ess_min,
                          Bi={n: b for n, b in zip(nombres, Bi)}))

    m = np.array([r["m"] for r in filas])
    lB = np.array([r["lB"] for r in filas])
    sg = np.array([r["sig"] for r in filas])
    npsr = np.array([r["n"] for r in filas])
    full = npsr >= N_ESP

    print(f"ACOPLE={ACOPLE}  prior log10({NOMBRE}) ~ U({lo:.4f}, {hi:.4f})  "
          f"W={W:.2f} dex  REBIN~{REBIN} dex  MESETA={MESETA} dex  N_BOOT={N_BOOT}")
    print(f"{'m [eV]':>12} {'log10B10':>9} {'sigma':>6} {'npsr':>4} {'ESSmin':>7}  avisos")
    for r in filas:
        av = []
        if r["n"] < N_ESP: av.append(f"INCOMPLETA {r['n']}/{N_ESP}")
        if r["sin_meseta"]: av.append("COTA INFERIOR (sin meseta: " + ",".join(r["sin_meseta"]) + ")")
        if r["no_plano"]: av.append("meseta no plana: " + ",".join(r["no_plano"]))
        print(f"{r['m']:12.4e} {r['lB']:9.3f} {r['sig']:6.3f} {r['n']:4d} "
              f"{r['ess_min']:7.0f}  {'; '.join(av)}")

    if full.any():
        i = int(np.argmax(lB[full]))
        print(f"\nCompletas: {full.sum()}/{len(filas)} masas.  "
              f"log10 B10 en [{lB[full].min():.3f}, {lB[full].max():.3f}]  "
              f"sigma tipica {np.nanmedian(sg[full]):.3f} dex (max {np.nanmax(sg[full]):.3f})")
        print(f"maximo en m = {m[full][i]:.4e} eV: log10 B10 = {lB[full][i]:.3f} "
              f"+- {sg[full][i]:.3f}")
    n_np = sum(1 for r in filas if r["no_plano"])
    print(f"masas con alguna meseta no plana: {n_np}/{len(filas)}")

    Bi = np.full((len(filas), len(todos)), np.nan)
    for a, r in enumerate(filas):
        for b, n in enumerate(todos):
            if n in r["Bi"]:
                Bi[a, b] = r["Bi"][n]
    cinf = np.array([bool(r["sin_meseta"]) for r in filas])
    out = os.path.join(OUT_ROOT, f"BF_{ACOPLE}_vs_m.npz")
    np.savez(out, masses_eV=m, log10_B10=lB, sigma_log10_B10=sg, n_psr=npsr, cota_inferior=cinf,
             completa=full, log10_B_pulsar=Bi, pulsares=np.array(todos),
             log10_min=lo, log10_max=hi, rebin_dex=REBIN, meseta_dex=MESETA)
    print(f"guardado: {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
        if full.any():
            ax.fill_between(m[full], lB[full] - sg[full], lB[full] + sg[full],
                            color="0.85", lw=0, label=r"$\pm1\sigma$ (bootstrap)")
            ax.plot(m[full], lB[full], "o-", ms=3, lw=1.2, color="black",
                    label=r"$\log_{10}\mathcal{B}_{10}$")
        if (~full).any():
            ax.plot(m[~full], lB[~full], "x", color="crimson", label="incompletas")
        ax.set_xscale("log")
        ax.axhline(0, color="0.3", lw=0.8, ls="--")
        ax.set_xlabel(r"$m$ [eV]")
        ax.set_ylabel(r"$\log_{10}\mathcal{B}_{10}$")
        ax.set_title(f"Bayes factor combinado ({ACOPLE}), prior log-uniforme")
        ax.grid(True, which="both", ls="--", alpha=0.35)
        ax.legend(fontsize=8)
        plt.tight_layout()
        png = out.replace(".npz", ".png")
        plt.savefig(png); plt.close(fig)
        print(f"guardado: {png}")
    except Exception as e:
        print(f"[WARN] sin grafico: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
