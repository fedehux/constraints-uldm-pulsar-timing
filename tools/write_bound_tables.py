#!/usr/bin/env python
"""Write the 95% credible upper limits of the paper as plain-text tables.

    python tools/write_bound_tables.py

For every coupling whose ``bounds_<coupling>.npz`` is present, writes
``<coupling>/<name>_95UL.txt`` with two columns: the ULDM mass in eV and the
upper limit in physical units (alpha in GeV^-1, sqrt(beta) in GeV^-1, g
dimensionless).  Only masses at which all 50 pulsars entered the combination
are listed; these are the points drawn in the figures.
"""
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABLES = [
    ("linear", "alpha", "alpha_95UL.txt", "alpha [GeV^-1]",
     "linear universal coupling of a scalar field, M_i(Phi) = M_i (1 + alpha Phi)"),
    ("quadratic", "beta", "sqrtbeta_95UL.txt", "sqrt(beta) [GeV^-1]",
     "quadratic universal coupling of a scalar field, M_i(Phi) = M_i (1 + beta Phi^2 / 2)"),
    ("vector", "vec", "g_95UL.txt", "g",
     "B-L vector coupling g, for a charge-to-mass difference Delta c = 0.5"),
]


def main():
    for folder, coupling, filename, column, description in TABLES:
        npz = os.path.join(ROOT, folder, f"bounds_{coupling}.npz")
        if not os.path.exists(npz):
            print(f"{folder}: {os.path.basename(npz)} not available, skipped")
            continue
        with np.load(npz) as z:
            m = np.asarray(z["masses_eV"], float)
            ul = 10.0 ** np.asarray(z["log10_ul_physical"], float)
            complete = np.asarray(z["n_pulsars"]) >= 50
        order = np.argsort(m[complete])
        data = np.column_stack([m[complete][order], ul[complete][order]])
        header = (f"95% credible upper limit, {description}.\n"
                  f"Constraints on ultralight dark matter couplings from orbital perturbations of binary pulsars\n"
                  f"(Huxhagen & Lopez Nacir). Generated from {folder}/bounds_{coupling}.npz.\n"
                  f"mass [eV]    {column}")
        out = os.path.join(ROOT, folder, filename)
        np.savetxt(out, data, fmt="%.6e", delimiter="    ", header=header)
        print(f"{folder}: {len(data)} masses -> {out}")


if __name__ == "__main__":
    main()
