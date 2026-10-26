import numpy as np

from .results import curve, make


def correlation(scale=1, seed=42):
    from .core import stream

    rng = stream(seed, "correlation")
    n = 4000 * scale
    rho = np.linspace(0, 0.95, 20)
    naive = []
    known = []
    ci = []
    for value in rho:
        a = rng.normal(size=(n, 2))
        b = value * a + np.sqrt(1 - value**2) * rng.normal(size=(n, 2))
        error = (a + b) / 2
        squared = np.mean(np.sum(error**2, axis=1))
        naive.append(squared / 0.5)
        known.append(squared / (0.5 * (1 + value)))
        ci.append(squared)
    return make(
        "correlated-track-fusion",
        {"independent_trials": n, "dimension": 2},
        {
            "naive_nees_high_correlation": naive[-1],
            "known_nees_high_correlation": known[-1],
            "ci_nees_high_correlation": ci[-1],
        },
        {
            "correlation": rho,
            "naive_nees": np.array(naive),
            "known_nees": np.array(known),
            "ci_nees": np.array(ci),
            "reference": np.full(len(rho), 2.0),
        },
        [
            curve(
                "Common-error double counting",
                "correlation",
                [
                    ("naive_nees", "Assume independence"),
                    ("known_nees", "Known cross covariance"),
                    ("ci_nees", "Covariance intersection"),
                    ("reference", "Expected calibrated NEES"),
                ],
                "Cross-correlation coefficient",
                "Mean NEES",
            )
        ],
        assumptions=[
            "Independent Monte Carlo trials at each correlation, two equal unit-covariance unbiased position estimates. Fixed CI weight 0.5."
        ],
    )
