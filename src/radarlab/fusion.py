import numpy as np

from .core import finite, positive


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


def polar_observation(position, anchor=(0, 0)):
    p = finite(position, real=True)
    a = finite(anchor, real=True)
    if p.shape != (2,) or a.shape != (2,):
        raise ValueError("2D target and sensor positions required")
    delta = p - a
    r = np.linalg.norm(delta)
    if r < 1e-6:
        raise ValueError("bearing unobservable at the sensor origin")
    x, y = delta
    return np.array([r, np.arctan2(x, y)]), np.array([[x / r, y / r], [y / r**2, -x / r**2]])


def position_information(position, anchors, range_std_m=0.3, bearing_std_rad=0.02):
    sensors = finite(anchors, real=True, ndim=2)
    sr = positive(range_std_m)
    sa = positive(bearing_std_rad)
    if sensors.shape[1] != 2 or not len(sensors):
        raise ValueError("one or more planar radar anchors required")
    information = np.zeros((2, 2))
    for anchor in sensors:
        _, h = polar_observation(position, anchor)
        information += h.T @ np.diag([1 / sr**2, 1 / sa**2]) @ h
    return {"information": information, "covariance_bound": np.linalg.inv(covariance(information, 2))}
