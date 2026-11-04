#!/usr/bin/env python
# =============================================================================
# aggregate_bounds.py  --  histograms -> combined 95% upper limit vs mass
# =============================================================================
# Pure post-processing: everything is recomputed from the stored histograms,
# so any setting can be changed without rerunning chains.
#
# For each (mass, pulsar), with the chain run with a prior uniform in the
# coupling theta:
#   p_i(x) ~ 10^x Z_i(x)  (x = log10 theta)   ->   ln Z_i(x) = ln p_i(x) - x ln10
#   1) raw histogram with 0.005 dex bins (the worker does not smooth),
#      optionally smoothed here with a Gaussian of SMOOTHING dex (0: none);
#   2) where the smoothed density exceeds TAIL_FRAC of its maximum it is used
#      as it is;
#   3) below (small theta, rarely visited): plateau, ln Z_i constant and equal
#      to the median of the first ANCHOR dex measured (the likelihood no
#      longer changes);
#   4) above (large theta, never reached): power-law tail Z_i ~ theta^EXPONENT,
#      with EXPONENT = -2 for alpha and g (signal linear in the field:
#      P(weak field) ~ (c/theta)^2) and -1 for beta (signal ~ rho^2,
#      rho^2 ~ Exp(1)); offset from the median of the last ANCHOR dex.
# Combination:  ln p(x) = x ln10 + sum_i ln Z_i(x);  limit = 95% quantile.
# Settings (environment): SMOOTHING (0.03), TAIL_FRAC (0.05), ANCHOR (0.7),
#   TAIL (power_law | gaussian | cut), EXPONENT (default by coupling).
# Optional: REPLACEMENTS = comma-separated folders with histograms that replace
#   those of OUT_ROOT/hist with the same name (if a file is in several, the
#   last one wins); BOUNDS_OUT = path of the output .npz (default
#   OUT_ROOT/bounds_<coupling>.npz); EXTRAS = comma-separated folders with
#   histograms of masses not in OUT_ROOT/hist, added before the replacements.
#
# Printed diagnostics: incomplete masses, margin of the limit below the prior
# ceiling, minimum ESS, chains with most samples near the ceiling.
#
# Usage:  COUPLING=alpha OUT_ROOT=... python aggregate_bounds.py
# =============================================================================
import os
import glob
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter1d

COUPLING = os.environ["COUPLING"]
OUT_ROOT = os.environ["OUT_ROOT"]
SMOOTHING = float(os.environ.get("SMOOTHING", "0.03"))
TAIL_FRAC = float(os.environ.get("TAIL_FRAC", "0.05"))
ANCHOR = float(os.environ.get("ANCHOR", "0.7"))
TAIL = os.environ.get("TAIL", "power_law")
EXPONENT = float(os.environ.get("EXPONENT", "-1.0" if COUPLING == "beta" else "-2.0"))
N_EXPECTED = int(os.environ.get("N_PSR_EXPECTED", "50"))
REPLACEMENTS = os.environ.get("REPLACEMENTS", "")
BOUNDS_OUT = os.environ.get("BOUNDS_OUT", "")
EXTRAS = os.environ.get("EXTRAS", "")
LN10 = np.log(10.0)
TO_PHYSICAL = {"alpha": lambda x: x + np.log10(np.sqrt(2.30817405213802e+48)),
               "beta": lambda x: 0.5 * (x + np.log10(2.30817405213802e+48)),   # log10 sqrt(beta) [GeV^-1]
               "vec": lambda x: x}[COUPLING]


def lnZ_from_counts(c, centers, dx):
    dens = np.asarray(c, float)                      # raw counts as saved by the worker
    if SMOOTHING > 0:                                # optional smoothing, only here
        dens = gaussian_filter1d(dens, SMOOTHING / dx, mode="nearest")
    lnZ = np.full(centers.shape, -np.inf)
    if dens.max() <= 0:
        return lnZ
    idx = np.flatnonzero(dens > TAIL_FRAC * dens.max())
    i0, i1 = idx[0], idx[-1]
    lnZ[i0:i1 + 1] = np.log(np.maximum(dens[i0:i1 + 1], TAIL_FRAC * dens.max())) - centers[i0:i1 + 1] * LN10
    na = max(3, int(round(ANCHOR / dx)))
    lnZ[:i0] = np.median(lnZ[i0:min(i0 + na, i1 + 1)])   # only measured bins: a chain spanning < ANCHOR dex would otherwise pull in the still -inf tail
    if i1 + 1 < len(centers) and TAIL != "cut":
        j0 = max(i0, i1 - na + 1)
        xh = centers[i1]
        if TAIL == "power_law":
            a0 = np.median(lnZ[j0:i1 + 1] - EXPONENT * LN10 * (centers[j0:i1 + 1] - xh))
            lnZ[i1 + 1:] = a0 + EXPONENT * LN10 * (centers[i1 + 1:] - xh)
        else:  # gaussian
            u = 10.0 ** (2.0 * (centers[j0:i1 + 1] - xh))
            b, a0 = np.polyfit(u, lnZ[j0:i1 + 1], 1)
            if b < 0:
                lnZ[i1 + 1:] = a0 + b * 10.0 ** (2.0 * (centers[i1 + 1:] - xh))
    return lnZ


def upper_limit_95(centers, lnZ_tot):
    fine = np.linspace(centers[0], centers[-1], int(round((centers[-1] - centers[0]) / 0.002)) + 1)
    fin = np.isfinite(lnZ_tot)
    lz = np.interp(fine, centers[fin], lnZ_tot[fin])
    lz[fine > centers[fin].max()] = -np.inf
    lp = fine * LN10 + lz
    lp -= np.nanmax(lp)
    p = np.exp(lp)
    p[~np.isfinite(p)] = 0
    cdf = np.concatenate([[0], np.cumsum(0.5 * (p[1:] + p[:-1]) * np.diff(fine))])
    return float(np.interp(0.95, cdf / cdf[-1], fine))


def main():
    fs = sorted(glob.glob(os.path.join(OUT_ROOT, "hist", "m_*__*__s0.npz")))
    if not fs:
        raise SystemExit("no histograms")
    if EXTRAS:
        present = {os.path.basename(f) for f in fs}
        extra = [f for c in EXTRAS.split(",") if c.strip()
                 for f in sorted(glob.glob(os.path.join(c.strip(), "m_*__*__s0.npz")))]
        repeated = [f for f in extra if os.path.basename(f) in present]
        if repeated:
            raise SystemExit(f"EXTRAS: {len(repeated)} files are already in OUT_ROOT, e.g. {repeated[0]}")
        fs = fs + extra
        print(f"[aggregate_bounds] {len(extra)} histograms added from {EXTRAS}")
    if REPLACEMENTS:
        new = {}
        for folder in [c for c in REPLACEMENTS.split(",") if c.strip()]:
            new.update({os.path.basename(f): f
                        for f in glob.glob(os.path.join(folder.strip(), "m_*__*__s0.npz"))})
        unmatched = sorted(set(new) - {os.path.basename(f) for f in fs})
        if unmatched:
            raise SystemExit(f"REPLACEMENTS: {len(unmatched)} files without a match in OUT_ROOT, e.g. {unmatched[0]}")
        fs = [new.get(os.path.basename(f), f) for f in fs]
        print(f"[aggregate_bounds] {len(new)} histograms replaced from {REPLACEMENTS}")
    by_mass = defaultdict(dict)
    meta = {}
    n_ceiling = 0
    for f in fs:
        with np.load(f, allow_pickle=True) as z:
            if str(z["prior"]) != "uniform" or str(z["coupling"]) != COUPLING:
                raise SystemExit(f"{f}: prior={z['prior']} coupling={z['coupling']}")
            ms, n = str(z["m_str"]), str(z["name"])
            by_mass[ms][n] = (np.asarray(z["counts"]), float(z["ess"]))
            meta.setdefault("range", (float(z["log10_min"]), float(z["log10_max"]), int(z["n_bins"])))
            meta.setdefault("version", str(z["version"]))
            n_ceiling += float(z["frac_ceiling"]) > 0.5
    lo, hi, nb = meta["range"]
    edges = np.linspace(lo, hi, nb + 1)
    centers = 0.5 * (edges[1:] + edges[:-1])
    dx = edges[1] - edges[0]
    rows = []
    for ms in sorted(by_mass, key=float):
        tot = np.zeros(nb)
        for c, ess in by_mass[ms].values():
            tot = tot + lnZ_from_counts(c, centers, dx)
        ul = upper_limit_95(centers, tot)
        esss = [v[1] for v in by_mass[ms].values()]
        rows.append((float(ms), ul, len(by_mass[ms]), min(esss), hi - ul))
    m = np.array([r[0] for r in rows])
    ul = np.array([r[1] for r in rows])
    npsr = np.array([r[2] for r in rows])
    ess_min = np.array([r[3] for r in rows])
    margin = np.array([r[4] for r in rows])
    print(f"[aggregate_bounds] {COUPLING} version={meta['version']}  prior [{lo:.3f}, {hi:.3f}]  {nb} bins  "
          f"SMOOTHING={SMOOTHING} TAIL_FRAC={TAIL_FRAC} ANCHOR={ANCHOR} TAIL={TAIL} EXPONENT={EXPONENT}")
    print(f"  masses {len(m)}; complete ({N_EXPECTED} pulsars): {int((npsr >= N_EXPECTED).sum())}; "
          f"incomplete: {int((npsr < N_EXPECTED).sum())}")
    print(f"  smallest margin of the limit below the prior ceiling: {margin.min():.2f} dex "
          f"(m = {m[np.argmin(margin)]:.3e}){'   *** LESS THAN 1.5 dex: raise the ceiling ***' if margin.min() < 1.5 else ''}")
    print(f"  minimum ESS {ess_min.min():.0f}; median over masses of the per-mass minimum ESS {np.median(ess_min):.0f}")
    print(f"  chains with >50% of their samples within 0.5 dex of the ceiling: {n_ceiling} "
          f"(pulsars insensitive at that mass; their ln Z_i is a plateau)")
    out = BOUNDS_OUT or os.path.join(OUT_ROOT, f"bounds_{COUPLING}.npz")
    np.savez(out, masses_eV=m, log10_ul_code=ul, log10_ul_physical=TO_PHYSICAL(ul), n_pulsars=npsr,
             ess_min=ess_min, margin_to_ceiling=margin, prior=(lo, hi), smoothing=SMOOTHING,
             tail_frac=TAIL_FRAC, anchor=ANCHOR, tail=TAIL, exponent=EXPONENT, version=meta["version"])
    print(f"  saved {out}")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.8), dpi=130)
        ax.loglog(m, 10 ** TO_PHYSICAL(ul), "k-", lw=1)
        ax.set_xlabel("m [eV]")
        ax.set_ylabel({"alpha": r"$\alpha$ [GeV$^{-1}$]",
                       "beta": r"$\sqrt{\beta}$ [GeV$^{-1}$]", "vec": r"$g_{B-L}$"}[COUPLING])
        ax.grid(True, which="both", ls="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out.replace(".npz", ".png"))
    except Exception as e:
        print("  [no plot]", e)


if __name__ == "__main__":
    main()
