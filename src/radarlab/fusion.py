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


def mounting_covariance(local_mean, pose_covariance, yaw_rad=0):
    x, _ = transform_track(local_mean, np.eye(4), yaw_rad=yaw_rad)
    pose = covariance(pose_covariance, 3)
    jac = np.array([[1, 0, -x[1]], [0, 1, x[0]], [0, 0, -x[3]], [0, 0, x[2]]])
    return jac @ pose @ jac.T


def align(mean, covariance_value, source_time_s, target_time_s, process_density=1):
    from .tracking import predict

    start, stop = finite([source_time_s, target_time_s], real=True)
    if stop < start:
        raise ValueError("fusion cannot propagate a track backward using this predictor")
    if stop == start:
        return finite(mean, real=True).copy(), covariance(covariance_value, 4).copy()
    return predict(mean, covariance_value, stop - start, process_density)
