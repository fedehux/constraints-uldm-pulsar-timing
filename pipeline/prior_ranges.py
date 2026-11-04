"""Prior range of each coupling in the final runs (Table I of the paper).

The same range is used for every pulsar and every mass, in the upper-limit
runs and in the injection test:
  * floor   <= (lowest onset of sensitivity of any pulsar) - 1.5 dex, so that
    the first dex of the prior is flat for every chain;
  * ceiling >= (highest combined 95% upper limit) + 2 dex, so that no limit
    is close to the edge.
Values in code units (alpha: physical log10(alpha/GeV^-1); beta and g as in the
pipelines), on the 0.005-dex bin lattice. For g the ceiling is -14 and not -13,
because ENTERPRISE chains stick to a ceiling at -13 (round-off); with
Delta c = 0.5 every limit on g moves down by 0.699 dex, the floor needs <= -27.86
and the smallest margin to the ceiling is 1.64 dex (vector/aggregate_bounds_vec.log).
"""
import os

PRIOR_RANGES = {
    "alpha": {"LOG10_ALPHA_PHYS_MIN": -26.0, "LOG10_ALPHA_PHYS_MAX": -10.0},
    "beta": {"LOG10_BETA_MIN": -92.0, "LOG10_BETA_MAX": -60.0},
    "vec": {"LOG10_G_MIN": -30.0, "LOG10_G_MAX": -14.0},
}


def export(coupling):
    """Write the range of a coupling to the environment, where the pipelines read it."""
    for k, v in PRIOR_RANGES[coupling].items():
        os.environ[k] = repr(float(v))
