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


def mounting(scale=1, seed=42):
    from .fusion import mounting_covariance

    distance = np.linspace(1, 200, 101 * scale)
    std = []
    for d in distance:
        std.append(
            np.sqrt(
                mounting_covariance([0, d, 1, 0], np.diag([0.05**2, 0.05**2, np.deg2rad(0.2) ** 2]))[0, 0]
            )
        )
    return make(
        "mounting-uncertainty",
        {"translation_std_m": 0.05, "yaw_std_deg": 0.2},
        {"lateral_std_at_200m": std[-1]},
        {"range_m": distance, "lateral_std_m": np.array(std)},
        [
            curve(
                "Sensor-pose uncertainty contribution",
                "range_m",
                [("lateral_std_m", "Yaw and translation")],
                "Target distance (m)",
                "Lateral standard deviation (m)",
            )
        ],
    )


def alignment(scale=1, seed=42):
    from .fusion import align, independent

    lag = np.linspace(0, 2, 101 * scale)
    velocity = np.array([2.0, 3.0])
    stale = []
    aligned = []
    uncertainty = []
    current = np.r_[np.array([10.0, 30.0]), velocity]
    for dt in lag:
        old = current.copy()
        old[:2] -= velocity * dt
        bad = independent(old, np.eye(4), current, np.eye(4))
        advanced, p = align(old, np.eye(4), 0, dt, 0.2)
        good = independent(advanced, p, current, np.eye(4))
        stale.append(np.linalg.norm(bad["mean"][:2] - current[:2]))
        aligned.append(np.linalg.norm(good["mean"][:2] - current[:2]))
        uncertainty.append(np.trace(good["covariance"][:2, :2]))
    return make(
        "asynchronous-track-alignment",
        {"velocity_m_s": velocity.tolist(), "process_density": 0.2},
        {"maximum_stale_bias_m": max(stale), "maximum_aligned_bias_m": max(aligned)},
        {
            "lag_s": lag,
            "stale_bias_m": np.array(stale),
            "aligned_bias_m": np.array(aligned),
            "position_variance_m2": np.array(uncertainty),
        },
        [
            curve(
                "Timestamp mismatch under constant velocity",
                "lag_s",
                [("stale_bias_m", "Fuse stale position"), ("aligned_bias_m", "Propagate then fuse")],
                "Track age (s)",
                "Position bias (m)",
            ),
            curve(
                "Age-dependent fusion uncertainty",
                "lag_s",
                [("position_variance_m2", "Position covariance trace")],
                "Track age (s)",
                "Variance (m^2)",
            ),
        ],
    )


def innovations(scale=1, seed=42):
    from .core import stream
    from .fusion import whiten

    p = np.array([[1.0, 0.8], [0.8, 1.0]])
    samples = stream(seed, "whitening").multivariate_normal([0, 0], p, 2000 * scale)
    white = np.array([whiten(e, p) for e in samples])
    return make(
        "innovation-whitening",
        {"covariance": p.tolist(), "samples": len(samples)},
        {
            "raw_correlation": float(np.corrcoef(samples.T)[0, 1]),
            "whitened_correlation": float(np.corrcoef(white.T)[0, 1]),
            "mean_whitened_energy": float(np.mean(np.sum(white**2, axis=1))),
        },
        {"raw_x": samples[:, 0], "raw_y": samples[:, 1], "white_x": white[:, 0], "white_y": white[:, 1]},
        [
            curve(
                "Correlated innovations",
                "raw_x",
                [("raw_y", "Raw")],
                "First residual",
                "Second residual",
                style=".",
            ),
            curve(
                "Cholesky-normalized innovations",
                "white_x",
                [("white_y", "Whitened")],
                "First normalized residual",
                "Second normalized residual",
                style=".",
            ),
        ],
    )


def bias(scale=1, seed=42):
    from .fusion import localize_ranges

    a = np.array([[-30.0, 0], [30, 0], [0, 60], [50, 50]])
    truth = np.array([8.0, 25.0])
    offset = np.linspace(0, 3, 61 * scale)
    errors = []
    residuals = []
    for value in offset:
        fit = localize_ranges(a, np.linalg.norm(truth - a, axis=1) + value, 0.3, [0, 20])
        errors.append(np.linalg.norm(fit["position"] - truth))
        residuals.append(np.linalg.norm(fit["weighted_residual"]))
    return make(
        "common-range-bias",
        {"range_std_m": 0.3, "anchors_m": a.tolist()},
        {"bias_at_3m_m": errors[-1]},
        {
            "common_range_offset_m": offset,
            "position_error_m": np.array(errors),
            "weighted_residual_norm": np.array(residuals),
        },
        [
            curve(
                "Unmodeled common range offset",
                "common_range_offset_m",
                [("position_error_m", "Position bias")],
                "Common range bias (m)",
                "Position error (m)",
            ),
            curve(
                "Fit inconsistency",
                "common_range_offset_m",
                [("weighted_residual_norm", "Weighted residual norm")],
                "Common range bias (m)",
                "Normalized residual norm",
            ),
        ],
    )


def robust(scale=1, seed=42):
    from .fusion import localize_ranges, robust_ranges

    a = np.array([[-20.0, 0], [20, 0], [0, -20], [0, 20], [-20, 20], [20, 20]])
    truth = np.array([3.0, 5.0])
    bad = np.linspace(0, 12, 61 * scale)
    ordinary = []
    protected = []
    for value in bad:
        ranges = np.linalg.norm(truth - a, axis=1)
        ranges[0] += value
        ordinary.append(np.linalg.norm(localize_ranges(a, ranges, 0.1, [1, 3])["position"] - truth))
        protected.append(np.linalg.norm(robust_ranges(a, ranges, 0.1, [1, 3])["position"] - truth))
    return make(
        "robust-range-localization",
        {"range_std_m": 0.1, "huber_transition_sigma": 1.5},
        {"ordinary_error_m": ordinary[-1], "robust_error_m": protected[-1]},
        {"outlier_m": bad, "ordinary_error_m": np.array(ordinary), "robust_error_m": np.array(protected)},
        [
            curve(
                "Contaminated sensor challenge",
                "outlier_m",
                [("ordinary_error_m", "Gaussian least squares"), ("robust_error_m", "Huber loss")],
                "One range outlier (m)",
                "Position error (m)",
            )
        ],
        assumptions=["No calibrated Gaussian covariance is claimed for the robust estimate."],
    )
