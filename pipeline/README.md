# Production pipeline

These are the exact files used for the final runs of the paper, unchanged
(`MD5SUMS` lists their checksums; verify with `md5sum -c MD5SUMS`). Comments
and some identifiers are in Spanish; the glossary at the end translates them.

## Files

| File | Role |
|---|---|
| `pipeline_alpha.py`, `pipeline_beta.py`, `pipeline_vec.py` | Single-pulsar analysis for the linear (alpha), quadratic (beta) and vector (g) couplings: loading of the NANOGrav wideband data with PINT, ULDM timing template, ENTERPRISE likelihood with the timing model and red-noise Fourier coefficients marginalized analytically, and one PTMCMCSampler chain per (mass, pulsar). |
| `template_ecc_vec.py` | Vector template for eccentric orbits (Appendix B of the paper). |
| `masas_pulsares.csv` | Pulsar mass, companion mass and inclination used by the vector template (Table II / Appendix C of the paper). |
| `rangos_prior.py` | Prior range of each coupling (Table I of the paper), shared by the upper-limit and Bayes-factor runs. |
| `masas_alpha_cotas.txt`, `masas_beta_cotas.txt`, `masas_vec_cotas.txt` | ULDM masses of the upper-limit runs (407, 500 and 1476). |
| `masas_bf.txt` | ULDM masses of the Bayes-factor runs (200, all couplings). |
| `worker_final.py` | Runs the chains of one coupling and mode and stores, for each (mass, pulsar), only the histogram of log10 of the coupling (0.005 dex bins). Idempotent: rerunning continues where it stopped. |
| `run_final.slurm` | SLURM script used on the cluster (one node, 32 cores per job). |
| `agregar_cotas.py` | Combines the single-pulsar histograms at each mass into the 95% upper limit (Sec. III C of the paper); writes `COTAS_<coupling>.npz`. |
| `agregar_bf.py` | Combined Bayes factor at each mass (Appendix D); writes `BF_<coupling>_vs_m.npz`. |
| `chequeo_final.py` | Pre-flight checks run before submitting the final jobs. |

## Data

Download the NANOGrav 15-year data set, version 2.1.0 (Zenodo,
[doi:10.5281/zenodo.16051178](https://doi.org/10.5281/zenodo.16051178)), and
point `NANOGRAV_ROOT` to the unpacked folder
(`NANOGrav15yr_PulsarTiming_v2.1.0/`). The pipeline reads only
`wideband/par`, `wideband/tim` and `wideband/noise` from it, unmodified.

## Running

Each run is defined by a coupling (`ACOPLE` = `alpha`, `beta` or `vec`) and a
mode (`MODO` = `cotas` for the upper limits, prior uniform in the coupling;
`bf` for the Bayes factors, prior uniform in its logarithm):

```bash
export NANOGRAV_ROOT=/path/to/NANOGrav15yr_PulsarTiming_v2.1.0   # always set it
export ACOPLE=alpha MODO=cotas OUT_ROOT=$PWD/out_alpha_cotas
export CONGELAR_SECULARES=PBDOT N_ITER=200000
python worker_final.py                    # all masses and pulsars of the run
ACOPLE=alpha OUT_ROOT=$PWD/out_alpha_cotas python agregar_cotas.py
```

The histograms are written to `$OUT_ROOT/hist/m_<mass>__<pulsar>__s0.npz`.
The seed of every chain is fixed by (coupling, mode, mass, pulsar), so a rerun
reproduces the same chains for the same software versions. Useful variables:
`PULSARS` (comma-separated subset), `MASAS_ARCHIVO` (another mass file),
`JOB_ID`/`N_JOBS` (split the run between jobs), `CHAIN_TMP` (scratch for the
raw chains, deleted after each one is histogrammed).

On a SLURM cluster:

```bash
sbatch --export=ALL,ACOPLE=alpha,MODO=cotas,JOB_ID=0,N_JOBS=2 run_final.slurm
```

`run_final.slurm` contains the module, environment and directory names of the
cluster we used (`~/version_final`, data in `~/vec2`); adapt them to yours.

### Cost

Each chain has 200 000 iterations and took about 10 minutes on one core. The
upper limits of the paper took about 4 400 CPU-hours for alpha
(20 350 chains), 5 100 for beta (25 000 chains) and correspondingly more for
the vector coupling (73 800 chains). The released histograms
(`<coupling>/histograms/`) let you skip this step and start from the
aggregation (`tools/reproduce_bounds.py`).

## Glossary

| Spanish | English |
|---|---|
| acople | coupling |
| cotas | upper limits (bounds) |
| masas | masses |
| rangos_prior | prior ranges |
| agregar | aggregate (combine) |
| CONGELAR_SECULARES | timing parameters held fixed (PBDOT, Sec. III B of the paper) |
| semilla | seed |
| meseta | plateau (small-coupling end of a single-pulsar likelihood) |
| techo / piso | upper / lower edge of the prior |
| suavizado | smoothing |
| nivel (in `masas_pulsares.csv`) | case A, B or C of Appendix C |
| fuente | source of the masses and inclination |
