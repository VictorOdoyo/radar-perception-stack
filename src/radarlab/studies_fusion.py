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


def baseline(scale=1, seed=42):
    from .fusion import position_information

    baseline = np.linspace(0, 100, 101 * scale)
    lateral = []
    forward = []
    for width in baseline:
        p = position_information([0, 80], [[-width / 2, 0], [width / 2, 0]])["covariance_bound"]
        lateral.append(np.sqrt(p[0, 0]))
        forward.append(np.sqrt(p[1, 1]))
    return make(
        "radar-baseline-information",
        {"nominal_target_m": [0, 80], "range_std_m": 0.3, "bearing_std_rad": 0.02},
        {"zero_baseline_lateral_std_m": lateral[0], "wide_baseline_lateral_std_m": lateral[-1]},
        {"baseline_m": baseline, "lateral_std_m": np.array(lateral), "forward_std_m": np.array(forward)},
        [
            curve(
                "Local independent-measurement information bound",
                "baseline_m",
                [("lateral_std_m", "Lateral"), ("forward_std_m", "Forward")],
                "Radar baseline (m)",
                "Position standard-deviation bound (m)",
            )
        ],
        assumptions=[
            "Local Fisher bound at the declared nominal target; correct calibration and independent Gaussian range/bearing errors."
        ],
    )


def ranges(scale=1, seed=42):
    from .core import stream
    from .fusion import localize_ranges

    anchors = np.array([[-30.0, 0], [30, 0], [0, 60], [50, 50]])
    truth = np.array([8.0, 25.0])
    rng = stream(seed, "range-mc")
    n = 200 * scale
    errors = []
    nees = []
    for _ in range(n):
        observed = np.linalg.norm(truth - anchors, axis=1) + rng.normal(0, 0.3, len(anchors))
        fit = localize_ranges(anchors, observed, 0.3, [0, 20])
        error = fit["position"] - truth
        errors.append(error)
        nees.append(error @ np.linalg.solve(fit["covariance"], error))
    error = np.array(errors)
    plot = curve(
        "Independent range-only localization trials",
        "error_x_m",
        [("error_y_m", "Position error")],
        "Lateral error (m)",
        "Forward error (m)",
        style=".",
    )
    plot["aspect"] = "equal"
    return make(
        "range-only-localization",
        {"trials": n, "anchors_m": anchors.tolist(), "range_std_m": 0.3},
        {
            "position_rmse_m": float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
            "mean_nees": float(np.mean(nees)),
        },
        {"error_x_m": error[:, 0], "error_y_m": error[:, 1], "trial": np.arange(n), "nees": np.array(nees)},
        [
            plot,
            curve(
                "Trial consistency",
                "trial",
                [("nees", "NEES")],
                "Independent trial",
                "Squared normalized error",
            ),
        ],
    )


def velocity(scale=1, seed=42):
    from .core import stream
    from .fusion import velocity_from_doppler

    a = np.array([[-20.0, 0], [20, 0], [0, 50]])
    position = np.array([5.0, 20.0])
    truth = np.array([3.0, -2.0])
    direction = position - a
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    rng = stream(seed, "velocity")
    errors = []
    for _ in range(500 * scale):
        fit = velocity_from_doppler(position, a, direction @ truth + rng.normal(0, 0.2, 3))
        errors.append(fit["velocity"] - truth)
    errors = np.array(errors)
    return make(
        "multi-radar-doppler-velocity",
        {"known_position_m": position.tolist(), "doppler_std_m_s": 0.2},
        {"velocity_rmse_m_s": float(np.sqrt(np.mean(np.sum(errors**2, axis=1))))},
        {
            "trial": np.arange(len(errors)),
            "lateral_error_m_s": errors[:, 0],
            "forward_error_m_s": errors[:, 1],
        },
        [
            curve(
                "Independent multi-view velocity errors",
                "trial",
                [("lateral_error_m_s", "Lateral"), ("forward_error_m_s", "Forward")],
                "Trial",
                "Velocity error (m/s)",
            )
        ],
        assumptions=[
            "Target position and static sensor poses are treated as known; their uncertainty is not silently included in conditional Doppler covariance."
        ],
    )
