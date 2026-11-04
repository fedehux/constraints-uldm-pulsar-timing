# Pipeline

The code of the final runs of the paper.

## Files

| File | Role |
|---|---|
| `pipeline_alpha.py`, `pipeline_beta.py`, `pipeline_vec.py` | Single-pulsar analysis for the linear (alpha), quadratic (beta) and vector (g) couplings: ULDM timing template, ENTERPRISE signal, and the sampled model, with one PTMCMCSampler chain per (mass, pulsar). |
| `common.py` | Code shared by the three: loading of the NANOGrav wideband data with PINT, design-matrix partials of the orbital parameters, ENTERPRISE likelihood with the timing model and red-noise Fourier coefficients marginalized analytically, noise MAP values, the sampler. |
| `template_ecc_vec.py` | Vector template for eccentric orbits (Appendix B of the paper). |
| `pulsar_masses.csv` | Pulsar mass, companion mass and inclination used by the vector template (Table II and Appendix C of the paper). |
| `prior_ranges.py` | Prior range of each coupling (Table I of the paper). |
| `masses_alpha.txt`, `masses_beta.txt`, `masses_vec.txt` | ULDM masses of the upper-limit runs (407, 500 and 1476). |
| `worker_final.py` | Runs the chains of one coupling and stores, for each (mass, pulsar), only the histogram of log10 of the coupling (0.005 dex bins). Idempotent: rerunning continues where it stopped. |
| `run_final.slurm` | SLURM script used on the cluster (one node, 32 cores per job). |
| `aggregate_bounds.py` | Combines the single-pulsar histograms at each mass into the 95% upper limit (Sec. III C of the paper); writes `bounds_<coupling>.npz`. |

The code was translated to English (names, comments and messages) and the
loading, noise and sampling code shared by the three couplings was moved to
`common.py` after the runs. Each pipeline gives the same design matrix, noise
values, likelihood and prior as the code that produced the released chains,
and `aggregate_bounds.py` reproduces the released limits exactly
(`tools/reproduce_bounds.py`).

## Data

Download the NANOGrav 15-year data set, version 2.1.0 (Zenodo,
[doi:10.5281/zenodo.16051178](https://doi.org/10.5281/zenodo.16051178)), and
point `NANOGRAV_ROOT` to the unpacked folder
(`NANOGrav15yr_PulsarTiming_v2.1.0/`). The pipeline reads only
`wideband/par`, `wideband/tim` and `wideband/noise` from it, unmodified.

## Running

Each run is defined by a coupling (`COUPLING` = `alpha`, `beta` or `vec`). The
upper-limit runs use a prior uniform in the coupling:

```bash
export NANOGRAV_ROOT=/path/to/NANOGrav15yr_PulsarTiming_v2.1.0   # always set it
export COUPLING=alpha OUT_ROOT=$PWD/out_alpha
export FROZEN_PARAMS=PBDOT N_ITER=200000
python worker_final.py                    # all masses and pulsars of the run
python aggregate_bounds.py                # COUPLING and OUT_ROOT as above
```

The histograms are written to `$OUT_ROOT/hist/m_<mass>__<pulsar>__s0.npz` and
the limit to `$OUT_ROOT/bounds_<coupling>.npz`. Useful variables: `PULSARS`
(comma-separated subset), `MASS_FILE` (another mass file), `JOB_ID`/`N_JOBS`
(split the run between jobs), `CHAIN_TMP` (scratch for the raw chains, deleted
after each one is histogrammed).

The starting point of each chain is seeded by (coupling, mass, pulsar);
PTMCMCSampler draws its own seed, so a rerun gives statistically equivalent
chains, not identical ones.

On a SLURM cluster:

```bash
sbatch --export=ALL,COUPLING=alpha,JOB_ID=0,N_JOBS=2 run_final.slurm
```

`run_final.slurm` contains the module, environment and directory names of the
cluster we used (`~/version_final`, data in `~/vec2`); adapt them to yours.

The released histograms (`<coupling>/histograms/`) let you skip the chains and
start from the aggregation (`tools/reproduce_bounds.py`).

## Aggregation output

`aggregate_bounds.py` prints a short report (kept for the released curves in
`<coupling>/aggregate_bounds_<coupling>.log`):

- the prior range, number of bins and the aggregation settings;
- the number of masses, and how many have all 50 pulsars;
- the smallest margin of the limit below the prior ceiling, with a warning
  below 1.5 dex;
- the minimum effective sample size over all chains, and the median over
  masses of the per-mass minimum;
- the number of chains with more than half their samples within 0.5 dex of
  the ceiling: pulsars with no sensitivity at that mass, whose likelihood is
  flat and enters the combination as a constant.

The same per-mass quantities are stored in `bounds_<coupling>.npz`
(`n_pulsars`, `ess_min`, `margin_to_ceiling`) together with the limit
(`log10_ul_code`, `log10_ul_physical`) and the settings.
