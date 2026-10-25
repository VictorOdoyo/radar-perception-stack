import numpy as np


def fit_velocity(angles, radial_velocity, weights=None):
    from .core import finite

    a = finite(angles, real=True, ndim=1)
    v = finite(radial_velocity, real=True, ndim=1)
    if a.shape != v.shape or len(a) < 2:
        raise ValueError("aligned angular Doppler observations required")
    design = -np.column_stack([np.sin(a), np.cos(a)])
    w = np.ones(len(a)) if weights is None else finite(weights, real=True, ndim=1)
    if w.shape != a.shape or np.any(w <= 0):
        raise ValueError("positive observation weights required")
    matrix = design * np.sqrt(w)[:, None]
    if np.linalg.matrix_rank(matrix) < 2:
        raise ValueError("ego velocity unobservable from collinear directions")
    velocity = np.linalg.lstsq(matrix, v * np.sqrt(w), rcond=None)[0]
    return {
        "velocity_m_s": velocity,
        "residual_m_s": v - design @ velocity,
        "condition": float(np.linalg.cond(matrix)),
    }


def robust_velocity(angles, radial_velocity, threshold_m_s=0.3, iterations=100, seed=42):
    from .core import finite, integer, positive, stream

    a = finite(angles, real=True, ndim=1)
    v = finite(radial_velocity, real=True, ndim=1)
    threshold = positive(threshold_m_s)
    integer(iterations)
    if a.shape != v.shape or len(a) < 3:
        raise ValueError("at least three angular observations required")
    design = -np.column_stack([np.sin(a), np.cos(a)])
    rng = stream(seed, "ego-ransac")
    best = np.zeros(len(a), bool)
    for _ in range(iterations):
        pair = rng.choice(len(a), 2, replace=False)
        try:
            estimate = fit_velocity(a[pair], v[pair])["velocity_m_s"]
        except ValueError:
            continue
        inliers = abs(v - design @ estimate) < threshold
        if sum(inliers) > sum(best):
            best = inliers
    if sum(best) < 3:
        raise ValueError("insufficient stationary Doppler consensus")
    result = fit_velocity(a[best], v[best])
    result["inliers"] = best
    return result
