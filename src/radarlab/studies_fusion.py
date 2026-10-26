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


def weights(scale=1, seed=42):
    from .fusion import intersection, optimize_intersection

    w = np.linspace(0, 1, 101 * scale)
    pa = np.diag([1.0, 9])
    pb = np.diag([9.0, 1])
    det = []
    trace = []
    for value in w:
        p = intersection([0, 0], pa, [0, 0], pb, value)["covariance"]
        det.append(np.linalg.det(p))
        trace.append(np.trace(p))
    best = optimize_intersection([0, 0], pa, [0, 0], pb)
    return make(
        "intersection-weight-design",
        {"covariance_a": pa.tolist(), "covariance_b": pb.tolist()},
        {"optimal_weight": best["weight"]},
        {"weight": w, "determinant": np.array(det), "trace_m2": np.array(trace)},
        [
            curve(
                "Conservative fusion volume objective",
                "weight",
                [("determinant", "Covariance determinant")],
                "CI weight",
                "Determinant (m^4)",
            ),
            curve(
                "Total marginal position variance",
                "weight",
                [("trace_m2", "Covariance trace")],
                "CI weight",
                "Variance (m^2)",
            ),
        ],
    )
