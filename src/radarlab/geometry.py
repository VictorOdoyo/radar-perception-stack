import numpy as np

from .core import finite


def polar_to_cartesian(range_m, angle_rad):
    r = finite(range_m, real=True)
    a = finite(angle_rad, real=True)
    if np.any(r < 0):
        raise ValueError("nonnegative range required")
    return np.stack(np.broadcast_arrays(r * np.sin(a), r * np.cos(a)), axis=-1)


def cartesian_to_polar(position):
    x = finite(position, real=True)
    if x.shape[-1] != 2:
        raise ValueError("two-dimensional position required")
    return np.stack((np.linalg.norm(x, axis=-1), np.arctan2(x[..., 0], x[..., 1])), axis=-1)
