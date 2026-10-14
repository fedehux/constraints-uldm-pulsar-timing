# Constraints on ultralight dark matter couplings from orbital perturbations of binary pulsars

This repository contains the code and the data products of *Constraints on
ultralight dark matter couplings from orbital perturbations of binary pulsars*
(F. Huxhagen and D. López Nacir). We search the wideband data of the 50 binary
pulsars of the NANOGrav 15-year data set for the orbital signature of
ultralight dark matter (ULDM) between 10<sup>-23</sup> and 10<sup>-18</sup> eV,
and set 95% credible upper limits on three couplings: a linear (alpha) and a
quadratic (beta) universal coupling of a scalar field, and a vector field
coupled to B−L (g).

Everything needed to go from the NANOGrav data to the figures of the paper is
here: the production pipeline, the single-pulsar histograms it produced, the
combined upper limits, the external bounds and the plotting scripts. You can
enter at any of three levels:

1. **Use the limits.** Plain-text tables (`mass [eV]`, limit) in each coupling
   folder, e.g. `linear/alpha_95UL.txt`.
2. **Reproduce the curves and figures from the histograms** (minutes, a laptop).
3. **Rerun the analysis from the data** (thousands of CPU-hours, a cluster).

## Layout

```
pipeline/                 production code of the final runs (unchanged, with MD5SUMS)
linear/                   alpha: Fig. 1
quadratic/                beta:  Fig. 2
vector/                   g:     Fig. 3
    histograms/           single-pulsar histograms of the upper-limit runs (packed)
    COTAS_<coupling>.npz  combined 95% upper limit, as used in the paper
    <name>_95UL.txt       the same limit as a plain-text table
    other_constraints/    external bounds drawn in the figure
    recreate_fig_N.py     plotting script of the figure
    constraints_on_*.pdf  the figure
bayes_factors/            Bayes factors (Appendix D)
injection_recovery/       injection and recovery tests (Appendix D)
tools/                    packing, unpacking and reproduction scripts
```

## 1. The limits

Each coupling folder has a two-column table of the 95% credible upper limit
against the ULDM mass, in the units of the paper: `linear/alpha_95UL.txt`
(alpha in GeV<sup>-1</sup>), `quadratic/sqrtbeta_95UL.txt` (sqrt(beta) in
GeV<sup>-1</sup>) and `vector/g_95UL.txt` (g, for a charge-to-mass difference
Δc = 0.5). The same numbers, with the diagnostics of the aggregation, are in
`COTAS_<coupling>.npz` (`masses_eV`, `ul_fisico_log10` = log10 of the limit in
physical units, `n_pulsares` = number of pulsars combined).

## 2. Reproducing the curves and the figures

Redraw a figure (needs only `numpy` and `matplotlib`):

```bash
python linear/recreate_fig_1.py
python quadratic/recreate_fig_2.py
python vector/recreate_fig_3.py
```

With matplotlib 3.10.9 the PDFs are identical in content to those of the
paper.

Recompute a limit curve from the single-pulsar histograms with the production
aggregator, and compare it with the released one:

```bash
python tools/reproduce_bounds.py linear      # or quadratic, vector
```

This unpacks the histograms (one file per mass and pulsar, as written by the
pipeline), runs `pipeline/agregar_cotas.py` unchanged, and checks every array
against `COTAS_<coupling>.npz`. The curves of the paper are reproduced to
floating-point precision (the released ones were aggregated on the cluster;
a different numpy build may change the last digit). `tools/write_bound_tables.py`
regenerates the `.txt` tables.

The histograms are stored packed, a few files per coupling, to keep the
repository small. `tools/unpack_histograms.py` restores the original files and
`tools/pack_histograms.py` is the inverse; the conversion is lossless.

## 3. Rerunning the analysis

See [`pipeline/README.md`](pipeline/README.md): how to obtain the NANOGrav
data, run the chains, and aggregate them. The pipeline files are the ones of
the final runs, byte for byte.

## Requirements

`environment.yml` lists the environment (conda-forge):

```bash
conda env create -f environment.yml
conda activate uldm-pulsar-timing
```

Levels 1 and 2 need only `numpy`, `scipy` and `matplotlib`. Level 3 also needs
`PINT`, `ENTERPRISE`, `PTMCMCSampler` and their dependencies.

## Citation

If you use this code or these results, please cite the accompanying paper:

> F. Huxhagen and D. López Nacir, *Constraints on ultralight dark matter
> couplings from orbital perturbations of binary pulsars* (2026),
> citation details to be added upon publication.

and the NANOGrav 15-year data set, on which the analysis is based
([doi:10.5281/zenodo.16051178](https://doi.org/10.5281/zenodo.16051178);
Agazie et al., ApJL 951, L9 (2023)). `CITATION.cff` contains the citation
metadata.

## License

MIT (see `LICENSE`).
