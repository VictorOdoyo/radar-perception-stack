import numpy as np

from .core import finite


def covariance(value, dimension=None):
    p = finite(value, real=True, ndim=2)
    if (
        p.shape[0] != p.shape[1]
        or (dimension is not None and p.shape != (dimension, dimension))
        or not np.allclose(p, p.T)
        or min(np.linalg.eigvalsh(p)) <= 0
    ):
        raise ValueError("symmetric positive definite fusion covariance required")
    return p


def transform_track(mean, covariance_value, translation=(0, 0), yaw_rad=0):
    x = finite(mean, real=True)
    p = covariance(covariance_value, 4)
    offset = finite(translation, real=True)
    if x.shape != (4,) or offset.shape != (2,):
        raise ValueError("Cartesian track and 2D sensor translation required")
    angle = float(finite(yaw_rad, real=True))
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, -s], [s, c]])
    f = np.kron(np.eye(2), r)
    y = f @ x
    y[:2] += offset
    return y, f @ p @ f.T
