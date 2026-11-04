# Constraints on ultralight dark matter couplings from orbital perturbations of binary pulsars

This repository contains the code and the data products of *Constraints on
ultralight dark matter couplings from orbital perturbations of binary pulsars*
(F. Huxhagen and D. López Nacir, in preparation). Using the wideband data of
the 50 binary pulsars of the NANOGrav 15-year data set, we set 95% credible
upper limits on three couplings of ultralight dark matter (ULDM) between
10<sup>-23</sup> and 10<sup>-18</sup> eV, through the perturbations the field
induces in the binary orbits: a linear (alpha) and a quadratic (beta) universal
coupling of a scalar field, and a vector field coupled to B−L (g).

The repository goes from the NANOGrav data to the limits and figures of the
paper: the pipeline, the single-pulsar histograms it produced, the
combined upper limits, the external bounds and the plotting scripts. You can
enter at any of three levels:

1. **Use the limits.** Plain-text tables (`mass [eV]`, limit) in each coupling
   folder, e.g. `linear/alpha_95UL.txt`.
2. **Reproduce the curves and figures from the histograms** (minutes, a laptop).
3. **Rerun the analysis from the data** (thousands of CPU-hours, a cluster).

## Layout

```
pipeline/                 code of the final runs
linear/                   alpha: Fig. 2(a)
quadratic/                beta:  Fig. 2(b)
vector/                   g:     Fig. 3
    histograms/           single-pulsar histograms of the upper-limit runs (packed)
    bounds_<coupling>.npz combined 95% upper limit, as used in the paper
    aggregate_bounds_<coupling>.log   output of the aggregation that wrote it
    <name>_95UL.txt       the same limit as a plain-text table
    other_constraints/    external bounds drawn in the figure
    recreate_fig_*.py     plotting script of the figure
    constraints_on_*.pdf  the figure
injection_recovery/       injection and recovery test of the pipeline: Fig. 5
tools/                    packing, unpacking and reproduction scripts
```

## 1. The limits

Each coupling folder has a two-column table of the 95% credible upper limit
against the ULDM mass, in the units of the paper: `linear/alpha_95UL.txt`
(alpha in GeV<sup>-1</sup>), `quadratic/sqrtbeta_95UL.txt` (sqrt(beta) in
GeV<sup>-1</sup>) and `vector/g_95UL.txt` (g, for a charge-to-mass difference
Δc = 0.5). The same numbers, with the diagnostics of the aggregation, are in
`bounds_<coupling>.npz` (`masses_eV`, `log10_ul_physical` = log10 of the limit
in physical units, `n_pulsars` = number of pulsars combined; the other arrays
are described in [`pipeline/README.md`](pipeline/README.md#aggregation-output)).

## 2. Reproducing the curves and the figures

Redraw a figure (needs only `numpy` and `matplotlib`):

```bash
python linear/recreate_fig_2a.py
python quadratic/recreate_fig_2b.py
python vector/recreate_fig_3.py
python injection_recovery/recreate_fig_5.py      # also needs scipy
```

With matplotlib 3.10.9 the PDFs are identical in content to those of the
paper.

Recompute a limit curve from the single-pulsar histograms with the aggregator
of the pipeline, and compare it with the released one:

```bash
python tools/reproduce_bounds.py linear      # or quadratic, vector
```

This unpacks the histograms (one file per mass and pulsar, as written by the
pipeline), runs `pipeline/aggregate_bounds.py`, and checks every array against
`bounds_<coupling>.npz`. The curves of the paper are reproduced to
floating-point precision (the released ones were aggregated on the cluster;
a different numpy build may change the last digit). `tools/write_bound_tables.py`
regenerates the `.txt` tables.

The histograms are stored packed, a few files per coupling, to keep the
repository small. `tools/unpack_histograms.py` restores the original files and
`tools/pack_histograms.py` is the inverse; the conversion is lossless.

## 3. Rerunning the analysis

See [`pipeline/README.md`](pipeline/README.md): how to obtain the NANOGrav
data, run the chains, and aggregate them. The injection and recovery test is
rerun with the same code; see [`injection_recovery/README.md`](injection_recovery/README.md).

## Requirements

`environment.yml` lists the environment (conda-forge):

```bash
conda env create -f environment.yml
conda activate uldm-pulsar-timing
```

Levels 1 and 2 need only `numpy`, `scipy` and `matplotlib`. Level 3 also needs
`PINT`, `ENTERPRISE`, `PTMCMCSampler` and their dependencies.

## Citation

If you use this code or these results, please cite the paper (F. Huxhagen
and D. López Nacir, *Constraints on ultralight dark matter couplings from
orbital perturbations of binary pulsars*, in preparation), this repository (the
metadata are in `CITATION.cff`), and the NANOGrav 15-year data set, on which
the analysis is based
([doi:10.5281/zenodo.16051178](https://doi.org/10.5281/zenodo.16051178);
Agazie et al., ApJL 951, L9 (2023)).

## License

MIT (see `LICENSE`).
