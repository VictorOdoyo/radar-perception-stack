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


def position_covariance(range_m, angle_rad, range_std_m=0.3, angle_std_rad=0.02):
    from .core import positive

    r = positive(range_m)
    sr = positive(range_std_m)
    sa = positive(angle_std_rad)
    s, c = np.sin(angle_rad), np.cos(angle_rad)
    jac = np.array([[s, r * c], [c, -r * s]])
    return jac @ np.diag([sr * sr, sa * sa]) @ jac.T
