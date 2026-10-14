"""Recreate Fig. 2 of the paper: 95% credible upper limit on the quadratic
scalar coupling, shown as sqrt(beta), as a function of the ULDM mass.

The limit is read from ``COTAS_beta.npz`` (written by
``pipeline/agregar_cotas.py``; see ``tools/reproduce_bounds.py``) and the
external bounds from ``other_constraints/``:
  - PTA.txt        PTA search with inter-pulsar correlations (green)
  - PTAuncorr.txt  PTA search without correlations (grey)
  - cassini_betas.txt  Cassini search for a stochastic GW background (orange)
Run from any directory; the figure is written next to this script as
``constraints_on_sqrtbeta_quadratic.pdf`` and ``.png``.
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
RESULT_NPZ = HERE / "COTAS_beta.npz"   # final run, pipeline/agregar_cotas.py
OUTPUT_PDF = HERE / "constraints_on_sqrtbeta_quadratic.pdf"
OUTPUT_PNG = HERE / "constraints_on_sqrtbeta_quadratic.png"

# Paper style and palette shared with the other final-result figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})
C_OURS = "black"
C_PTA_CORRELATED = "#2E7D32"
C_PTA_UNCORRELATED = "grey"
C_CASSINI = "#E67E22"


def load_result(path):
    """This work: masses [eV] and the 95% upper limit on sqrt(beta) [GeV^-1]."""
    with np.load(path, allow_pickle=False) as data:
        masses = np.asarray(data["masses_eV"], dtype=float)
        # limit in physical units (log10, GeV^-1); only masses with all 50 pulsars
        sqrtb = 10.0 ** np.asarray(data["ul_fisico_log10"], dtype=float)
        completa = np.asarray(data["n_pulsares"]) >= 50
    valid = completa & np.isfinite(masses) & np.isfinite(sqrtb) & (masses > 0) & (sqrtb > 0)
    if not valid.any():
        raise ValueError(f"No valid points in {path}")
    order = np.argsort(masses[valid])
    return masses[valid][order], sqrtb[valid][order]


def load_constraint(filename, delimiter=None):
    path = CONSTRAINTS / filename
    values = np.loadtxt(path, delimiter=delimiter)
    masses, sqrtb = values[:, 0], values[:, 1]
    valid = np.isfinite(masses) & np.isfinite(sqrtb) & (masses > 0) & (sqrtb > 0)
    order = np.argsort(masses[valid])
    return masses[valid][order], sqrtb[valid][order]


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
    masses, sqrtb = load_result(RESULT_NPZ)
    m_corr, b_corr = load_constraint("PTA.txt", delimiter=",")
    m_uncorr, b_uncorr = load_constraint("PTAuncorr.txt", delimiter=",")
    m_cas, b_cas = load_constraint("cassini_betas.txt")

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=500)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e-23, 1e-18)
    ax.set_ylim(1e-19, 1e-9)
    ax.set_xlabel(r"$m$ [eV]", fontsize=10)
    ax.set_ylabel(r"$\sqrt{\beta}$ [GeV$^{-1}$]", fontsize=10, labelpad=2)
    top = ax.get_ylim()[1]

    # Regions above each curve are excluded.
    ax.fill_between(m_corr, b_corr, top, color=C_PTA_CORRELATED, alpha=0.12, zorder=1)
    ax.plot(m_corr, b_corr, color=C_PTA_CORRELATED, lw=1.0, zorder=3)

    ax.fill_between(m_uncorr, b_uncorr, top, color=C_PTA_UNCORRELATED,
                    alpha=0.12, zorder=1)
    ax.plot(m_uncorr, b_uncorr, color=C_PTA_UNCORRELATED, lw=1.0, zorder=3)

    ax.fill_between(m_cas, b_cas, top, color=C_CASSINI, alpha=0.12, zorder=1)
    ax.plot(m_cas, b_cas, color=C_CASSINI, lw=1.0, zorder=3)

    ax.fill_between(masses, sqrtb, top, color="grey", alpha=0.25, zorder=4)
    ax.plot(masses, sqrtb, color=C_OURS, lw=1.0, zorder=10)

    # A single label for the two PTA curves.
    ax.text(3e-23, 2e-18, "PTA", color=C_PTA_CORRELATED,
            fontsize=7, ha="right", va="bottom", weight="bold")
    ax.text(0.8e-19, 1e-15, "Cassini stochastic", color=C_CASSINI,
            fontsize=7, ha="right", va="bottom", weight="bold", rotation=10)

    configure_ticks(ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=300)
    if EN_SPYDER:
        plt.show()
    else:
        plt.close(fig)
    print(f"[SAVE] {OUTPUT_PDF}")
    print(f"[SAVE] {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
