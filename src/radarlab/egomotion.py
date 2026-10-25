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
