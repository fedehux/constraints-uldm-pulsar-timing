"""Recreate Fig. 1 of the paper: 95% credible upper limit on the linear
scalar coupling alpha as a function of the ULDM mass.

The limit is read from ``COTAS_alpha.npz`` (written by
``pipeline/agregar_cotas.py``; see ``tools/reproduce_bounds.py``) and the
external bounds from ``other_constraints/``.  Run from any directory; the PDF
is written next to this script as ``constraints_on_alpha_linear.pdf``.
"""

from pathlib import Path

import matplotlib
# Inside Spyder/IPython the figure is shown inline; as a plain script the Agg
# backend is used and the figure is only saved.
try:
    get_ipython().run_line_magic("matplotlib", "inline")
    EN_SPYDER = True
except NameError:
    matplotlib.use("Agg")
    EN_SPYDER = False
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


HERE = Path(__file__).resolve().parent
CONSTRAINTS = HERE / "other_constraints"
RESULT_NPZ = HERE / "COTAS_alpha.npz"   # final run, pipeline/agregar_cotas.py
OUTPUT_PDF = HERE / "constraints_on_alpha_linear.pdf"

# Paper style and palette shared with the other final-result figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})
C_OURS = "black"
C_CASSINI_GR = "#1565C0"
C_CASSINI_STOCHASTIC = "#E67E22"
C_PTA_CORRELATED = "#2E7D32"
C_PTA_UNCORRELATED = "grey"


def load_result(path):
    with np.load(path, allow_pickle=False) as data:
        masses = np.asarray(data["masses_eV"], dtype=float)
        # limit in physical units (log10, GeV^-1); only masses with all 50 pulsars
        alpha = 10.0 ** np.asarray(data["ul_fisico_log10"], dtype=float)
        completa = np.asarray(data["n_pulsares"]) >= 50
    valid = completa & np.isfinite(masses) & np.isfinite(alpha) & (masses > 0) & (alpha > 0)
    if not valid.any():
        raise ValueError(f"No valid points in {path}")
    order = np.argsort(masses[valid])
    return masses[valid][order], alpha[valid][order]


def load_constraint(filename):
    path = CONSTRAINTS / filename
    values = np.loadtxt(path, delimiter=",")
    masses, alpha = values[:, 0], values[:, 1]
    valid = np.isfinite(masses) & np.isfinite(alpha) & (masses > 0) & (alpha > 0)
    order = np.argsort(masses[valid])
    return masses[valid][order], alpha[valid][order]


def configure_ticks(ax):
    locator = np.arange(2, 10) * 0.1
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=locator, numticks=100))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=locator, numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="major", length=4, width=0.7, labelsize=8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)


def main():
    masses, alpha = load_result(RESULT_NPZ)
    m_corr, a_corr = load_constraint("Alphabayes.csv")
    m_uncorr, a_uncorr = load_constraint("Alphauncorr.csv")
    m_cas_st, a_cas_st = load_constraint("TableCas.csv")

    # Cassini test of GR: |gamma-1| at 2 sigma, translated to alpha.
    reduced_planck_mass_gev = 2.43e18
    epsilon_cassini = 4.6e-5
    alpha_cassini_gr = np.sqrt(epsilon_cassini / 2.0) / reduced_planck_mass_gev
    m_cas_gr = np.logspace(-23, -18, 400)
    a_cas_gr = np.full_like(m_cas_gr, alpha_cassini_gr)

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=500)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-23, 1e-18)
    ax.set_ylim(1e-27, 1e-12)
    ax.set_xlabel(r"$m$ [eV]", fontsize=10)
    ax.set_ylabel(r"$\alpha$ [GeV$^{-1}$]", fontsize=10, labelpad=2)
    top = ax.get_ylim()[1]

    # Regions above each curve are excluded.
    ax.fill_between(m_cas_gr, a_cas_gr, top, color=C_CASSINI_GR,
                    alpha=0.18, zorder=1)
    ax.plot(m_cas_gr, a_cas_gr, color=C_CASSINI_GR, lw=1.0, zorder=2)

    ax.fill_between(m_cas_st, a_cas_st, top, color=C_CASSINI_STOCHASTIC,
                    alpha=0.12, zorder=1)
    ax.plot(m_cas_st, a_cas_st, color=C_CASSINI_STOCHASTIC, lw=1.0, zorder=3)

    ax.fill_between(m_corr, a_corr, top, color=C_PTA_CORRELATED,
                    alpha=0.12, zorder=1)
    ax.plot(m_corr, a_corr, color=C_PTA_CORRELATED, lw=1.0, zorder=3)

    ax.fill_between(m_uncorr, a_uncorr, top, color=C_PTA_UNCORRELATED,
                    alpha=0.12, zorder=1)
    ax.plot(m_uncorr, a_uncorr, color=C_PTA_UNCORRELATED, lw=1.0, zorder=3)

    ax.fill_between(masses, alpha, top, color="grey", alpha=0.25, zorder=4)
    ax.plot(masses, alpha, color=C_OURS, lw=1.0, zorder=10)

    ax.text(9e-20, 2.3e-21, "Cassini test of GR", color=C_CASSINI_GR,
            fontsize=6, ha="right", va="bottom", weight="bold")
    ax.text(9.0e-20, 2e-22, "Cassini stochastic",
            color=C_CASSINI_STOCHASTIC, fontsize=6, ha="right", va="bottom",
            weight="bold")
    ax.text(1.3e-23, 2.0e-25, "PTA", color=C_PTA_CORRELATED,
            fontsize=7, ha="left", va="bottom", weight="bold")
    # ax.text(1.3e-23, 4.0e-27, "PTA (uncorr.)", color=C_PTA_UNCORRELATED,
    #         fontsize=5.5, ha="left", va="bottom", weight="bold")

    label_index = int(np.argmin(np.abs(np.log10(masses) - np.log10(2e-20))))
    # ax.annotate("This work", xy=(masses[label_index], alpha[label_index]),
    #             xytext=(-4, 7), textcoords="offset points", color=C_OURS,
    #             fontsize=7, ha="right", va="bottom", weight="bold")

    configure_ticks(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    if EN_SPYDER:
        plt.show()
    else:
        plt.close(fig)
    print(f"[SAVE] {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
