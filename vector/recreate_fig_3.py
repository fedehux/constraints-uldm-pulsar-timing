"""Recreate Fig. 3 of the paper: 95% credible upper limit on the B-L vector
coupling g as a function of the ULDM mass, for Delta c = 0.5.

The limit is read from ``COTAS_vec.npz`` (written by
``pipeline/agregar_cotas.py``; see ``tools/reproduce_bounds.py``) and the
external bounds from ``other_constraints/``:
  - PPTA.txt                    Parkes Pulsar Timing Array (green)
  - Eot-Wash-EP.txt             Eot-Wash test of the equivalence principle (orange)
  - LISA-Path-relative-acc.txt  LISA Pathfinder (purple)
The MICROSCOPE bound is drawn as the horizontal line g = 7e-26.
Run from any directory; the figure is written next to this script as
``constraints_on_g_vector.pdf`` and ``.png``.
"""

import os
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
RESULT_NPZ = Path(os.environ.get("RESULT_NPZ", HERE / "COTAS_vec.npz"))   # final run, pipeline/agregar_cotas.py
OUTPUT_PDF = HERE / "constraints_on_g_vector.pdf"
OUTPUT_PNG = HERE / "constraints_on_g_vector.png"

# Paper style and palette shared with the other final-result figures.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
})
C_MAIN = "black"       # this work
C_PPTA = "#2E7D32"     # PPTA
C_EOT = "#E67E22"      # Eot-Wash
C_LISA = "#6A1B9A"     # LISA Pathfinder
C_MICRO = "dimgray"    # MICROSCOPE
G_MICROSCOPE = 7e-26


def load_result(path):
    """This work: masses [eV] and the 95% upper limit on g."""
    with np.load(path, allow_pickle=False) as data:
        masses = np.asarray(data["masses_eV"], dtype=float)
        g = 10.0 ** np.asarray(data["ul_fisico_log10"], dtype=float)
        completa = np.asarray(data["n_pulsares"]) >= 50   # only masses with all 50 pulsars
    valid = completa & np.isfinite(masses) & np.isfinite(g) & (masses > 0) & (g > 0)
    if not valid.any():
        raise ValueError(f"No valid points in {path}")
    order = np.argsort(masses[valid])
    return masses[valid][order], g[valid][order]


def main():
    m_new, g_new = load_result(RESULT_NPZ)

    fig, ax = plt.subplots(figsize=(3.4, 2.8), dpi=300)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1e-23, 1e-18); ax.set_ylim(1e-26, 1e-16)
    ax.set_xlabel(r"$m$ [eV]", fontsize=10)
    ax.set_ylabel("g", fontsize=10, rotation=0, labelpad=10)

    externas = {}
    for nombre, color, z in (("PPTA", C_PPTA, 3), ("Eot-Wash-EP", C_EOT, 2),
                             ("LISA-Path-relative-acc", C_LISA, 3)):
        arr = np.loadtxt(CONSTRAINTS / f"{nombre}.txt", skiprows=3, usecols=(0, 1))
        externas[nombre] = arr
        ax.fill_between(arr[:, 0], arr[:, 1], ax.get_ylim()[1], color=color, alpha=0.15, zorder=z)
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.0, zorder=z + 0.1)

    ax.plot(m_new, g_new, color=C_MAIN, lw=0.8, label="This work", zorder=10)

    # MICROSCOPE, over the mass range of the Eot-Wash curve
    ax.fill_between(externas["Eot-Wash-EP"][:, 0], G_MICROSCOPE, ax.get_ylim()[1], color="grey", alpha=0.4)
    ax.axhline(y=G_MICROSCOPE, color="grey", lw=0.8)

    ax.text(4e-19, 0.8e-25, "MICROSCOPE", color=C_MICRO, fontsize=5.5, va="bottom", ha="right", weight="bold")
    ax.text(5e-23, 1e-25, "PPTA", color=C_PPTA, fontsize=6, va="bottom", ha="right", weight="bold")
    ax.text(0.95e-22, 0.6e-23, "Eöt-Wash", color=C_EOT, fontsize=7, va="bottom", ha="right", weight="bold", zorder=100)
    ax.text(0.8e-18, 3e-23, "LISA Pathfinder", color=C_LISA, fontsize=6, va="bottom", ha="right", weight="bold",
            rotation=90, zorder=100)

    locator = np.arange(2, 10) * 0.1
    ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=locator, numticks=100))
    ax.yaxis.set_minor_locator(mticker.LogLocator(base=10, subs=locator, numticks=100))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(which="both", direction="in", top=True, right=True)
    ax.tick_params(axis="both", which="major", length=4, width=0.7, labelsize=8)
    ax.tick_params(axis="both", which="minor", length=2, width=0.5)

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
