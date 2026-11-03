# Injection and recovery (Appendix E, Fig. 5)

A check of the upper-limit pipeline on simulated data. For each coupling and
two masses (one away from every orbital resonance, one close to a resonance of
PSR J1903+0327 for the scalar couplings or PSR J1909-3744 for the vector), the
residuals of the 50 binaries are replaced by a realization of the noise model
of the analysis. The same realization is analysed twice: without a signal, and
with the ULDM template at 4 times the 95% upper limit of the paper at that mass.
Each binary draws its own amplitude, phase and, for the vector coupling,
direction of the field. The chains are those of the production pipeline, with
a prior uniform in log10 of the coupling.

| File | Role |
|---|---|
| `histograms/injection_<coupling>.npz` | The 200 chains per coupling of the paper (2 levels × 2 masses × 50 pulsars, 200 000 iterations each): histogram of log10 of the coupling and the injected parameters of every chain. |
| `recreate_fig_5.py` | Draws Fig. 5 from those archives (`fig_injection_recovery.pdf`). |
| `run_injection.py` | Simulates the data and runs the chains with the code in `../pipeline/`. |
| `pack_injection.py` | Packs the output of `run_injection.py` into `histograms/`. |

## Redraw the figure

```bash
python injection_recovery/recreate_fig_5.py
```

Needs `numpy`, `scipy` and `matplotlib`. With matplotlib 3.10.9 the figure is
identical to the one in the paper.

## Rerun the test

```bash
export NANOGRAV_ROOT=/path/to/NANOGrav15yr_PulsarTiming_v2.1.0
python injection_recovery/run_injection.py prepare alpha 8
python injection_recovery/run_injection.py run     alpha 8
python injection_recovery/pack_injection.py injection_runs injection_recovery/histograms alpha
```

Replace `alpha` by `beta` or `vec` for the other couplings; the last argument
of `prepare` and `run` is the number of parallel processes. `prepare` loads the
pulsars with PINT once and caches them in `injection_runs/pulsars/`; `run`
writes one file per chain to `injection_runs/<coupling>/<level>/hist/`, with
`level` = `no_signal` or `x4`, and skips the chains that already exist, so it
can be restarted. The injected amplitude is read from the upper limits in
`../linear/`, `../quadratic/` and `../vector/`. Every random number is seeded
by (coupling, mass, level, pulsar), so the same software versions reproduce the
chains of the paper.

`INJ_OUT` changes the output folder and `INJ_N_ITER` the number of iterations
per chain (200 000 in the paper; one chain takes 10–30 minutes on one core).
`python injection_recovery/run_injection.py test alpha J1909-3744 5000` runs a
single short chain to check the setup.

To test other masses or amplitudes, edit `MASSES` and `LEVELS` at the top of
`run_injection.py`.
