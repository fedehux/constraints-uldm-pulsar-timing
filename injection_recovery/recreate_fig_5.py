#!/usr/bin/env python
"""Recreate Fig. 5 of the paper: injection and recovery.

Six panels (columns alpha, beta, g; rows away from / close to a resonance).  Each
shows the combined posterior of the coupling on simulated data, for chains with a
prior uniform in log10 of the coupling:
  grey   -- no signal: flat where no binary is sensitive, then a step down where
            the simulated data exclude the coupling;
  black  -- a signal injected at 4 times the 95% upper limit of the paper;
  orange -- the injected value.
Each curve is normalized to its maximum.

No-signal curve: below the coupling at which a binary becomes sensitive, its
likelihood is set to its small-coupling value (the plateau of the first dex of
the prior), blended into the measured density over BLEND dex, and smoothed.
Injected curve: product of the single-pulsar densities, smoothed as in the
production aggregator (0.03 dex) plus VISTA dex for display.

Reads ``histograms/injection_<coupling>.npz`` (written by ``pack_injection.py``)
and writes ``fig_injection_recovery.pdf`` and ``.png`` next to this script.

    python injection_recovery/recreate_fig_5.py
"""
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
U = 2.30817405213802e+48                 # code units -> physical units (as pipeline/agregar_cotas.py)
COUPLINGS = {
    "alpha": dict(f=lambda v: v + 0.5 * np.log10(U), label=r"$\log_{10}(\alpha/{\rm GeV}^{-1})$"),
    "beta": dict(f=lambda v: v + np.log10(U), label=r"$\log_{10}(\beta/{\rm GeV}^{-2})$"),
    "vec": dict(f=lambda v: v, label=r"$\log_{10}g$"),
}
SUAV_CHAIN = 0.03        # dex, production smoothing of each single-pulsar density
VISTA = (0.05, 0.12)     # dex, display smoothing of the injected curve (away from / close to resonance)
SUAV_NULA = 0.15         # dex, single-pulsar smoothing of the no-signal curve
SUAV_FINAL = 0.05        # dex, final smoothing of the no-signal curve
BLEND = 0.5              # dex, plateau -> measured transition


def read(coupling):
    """Per mass and level: {pulsar: counts}, the prior range and the injected value."""
    with np.load(HERE / "histograms" / f"injection_{coupling}.npz", allow_pickle=False) as z:
        levels = [str(s) for s in z["levels"]]
        masses = [str(s) for s in z["masses_str"]]
        pulsars = [str(s) for s in z["pulsars"]]
        counts, present = z["counts"].astype(float), z["present"]
        x_inj = z["chain_x_inj"]
        lo, hi = float(z["run_log10_min"]), float(z["run_log10_max"])
    out = {}
    for il, lv in enumerate(levels):
        for im, m in enumerate(masses):
            por = {p: counts[il, im, ip] for ip, p in enumerate(pulsars) if present[il, im, ip]}
            inj = x_inj[il, im][present[il, im]]
            out[(lv, m)] = (por, float(inj[0]) if inj.size else np.nan)
    return out, masses, lo, hi


def injected_curve(por, lo, hi, vista):
    n = len(next(iter(por.values()))); dx = (hi - lo) / n
    x = lo + (np.arange(n) + 0.5) * dx
    lp = np.zeros(n)
    for c in por.values():
        d = gaussian_filter1d(c, SUAV_CHAIN / dx, mode="nearest") + 1e-3
        lp += np.log(d / d.sum())
    p = np.exp(lp - lp.max())
    p = gaussian_filter1d(p, vista / dx)
    return x, p / p.max()


def no_signal_curve(por, lo, hi, threshold=0.5):
    lr = None
    for c in por.values():
        n = len(c); dx = (hi - lo) / n
        x = lo + (np.arange(n) + 0.5) * dx
        p0 = c[x < lo + 1.0].mean()                       # plateau, first dex of the prior
        r = gaussian_filter1d(c, SUAV_NULA / dx, mode="nearest") / p0
        above = np.nonzero(r >= threshold)[0]
        x_on = x[above[-1]] - BLEND if len(above) else lo
        w = np.clip((x - x_on) / BLEND, 0.0, 1.0)
        l = w * np.log(np.maximum(r, 1e-12))
        lr = l if lr is None else lr + l
    lr = gaussian_filter1d(lr, SUAV_FINAL / dx, mode="nearest")
    return x, np.exp(lr - lr.max())


def main():
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
                         "mathtext.fontset": "stix", "font.size": 9})
    fig, axs = plt.subplots(2, 3, figsize=(7.0, 4.3))
    for j, (coupling, cfg) in enumerate(COUPLINGS.items()):
        f = cfg["f"]
        runs, masses, lo, hi = read(coupling)
        for i, m_str in enumerate(masses):
            ax = axs[i, j]
            p0n, _ = runs[("no_signal", m_str)]
            p4, inj = runs[("x4", m_str)]
            if len(p0n) < 50 or len(p4) < 50:
                ax.text(0.5, 0.5, "incomplete", transform=ax.transAxes, ha="center"); continue
            xn, pn = no_signal_curve(p0n, lo, hi)
            xi, pi = injected_curve(p4, lo, hi, VISTA[i])
            ax.plot(f(xn), pn, color="grey", lw=1.0, label="no signal")
            ax.plot(f(xi), pi, color="black", lw=1.0, label="signal injected")
            ax.axvline(f(inj), color="#E67E22", lw=1.0, label="injected value")
            # window: from 1.5 dex before the step to just past the injected peak
            step = xn[np.nonzero(pn < 0.5)[0][0]]
            right = max(inj, xi[np.nonzero(np.cumsum(pi) / pi.sum() > 0.99)[0][0]])
            left = max(step - 1.5, lo + 1.0)
            ax.set_xlim(f(left), f(right + 0.4))
            ax.set_ylim(0, 1.1)
            ax.set_yticks([0, 0.5, 1])
            ax.xaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.tick_params(axis="both", which="major", length=4, width=0.7, labelsize=8)
            ax.tick_params(axis="both", which="minor", length=2, width=0.5)
            b, ex = f"{float(m_str):.2e}".split("e")
            ax.set_title(rf"$m={b}\times10^{{{int(ex)}}}$ eV", fontsize=8)
            if i == 1:
                ax.set_xlabel(cfg["label"], fontsize=10)
            if j == 0:
                ax.set_ylabel("posterior", fontsize=10)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(h), frameon=False, fontsize=8,
               handlelength=2.0, columnspacing=1.6, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(h_pad=0.6, w_pad=0.6, rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig_injection_recovery.{ext}", dpi=150)
    print(f"[SAVE] {HERE / 'fig_injection_recovery.pdf'} (and .png)")


if __name__ == "__main__":
    main()
