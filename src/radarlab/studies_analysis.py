import numpy as np

from .results import curve, make


def cfar_rates(scale=1, seed=42):
    from .core import stream
    from .detection import ca_multiplier, wilson

    rng = stream(seed, "cfar-calibration")
    n = 48
    trials = 30000 * scale
    pfa = np.geomspace(0.001, 0.1, 9)
    rates = []
    lower = []
    upper = []
    for probability in pfa:
        noise = rng.gamma(n, 1 / n, trials)
        cut = rng.exponential(size=trials)
        count = int(np.sum(cut > ca_multiplier(n, float(probability)) * noise))
        rates.append(count / trials)
        lo, hi = wilson(count, trials)
        lower.append(lo)
        upper.append(hi)
    a = {"nominal_pfa": pfa, "observed_pfa": rates, "lower": lower, "upper": upper}
    return make(
        "cfar-false-alarm",
        {"seed": seed, "training_cells": n, "trials_per_probability": trials},
        {"maximum_absolute_error": float(max(abs(np.array(rates) - pfa)))},
        a,
        [
            curve(
                "Independent exponential CFAR calibration",
                "nominal_pfa",
                [
                    ("nominal_pfa", "Nominal"),
                    ("observed_pfa", "Monte Carlo"),
                    ("lower", "Wilson lower"),
                    ("upper", "Wilson upper"),
                ],
                "Requested false alarm probability",
                "Observed probability",
                xscale="log",
                yscale="log",
            )
        ],
        assumptions=[
            "Independent exponential CUT and gamma-distributed training mean; no FFT-window correlation."
        ],
    )
