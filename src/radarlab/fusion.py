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
