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


def localize_ranges(anchors, ranges_m, standard_deviation_m, initial_position):
    from scipy.optimize import least_squares

    a = finite(anchors, real=True, ndim=2)
    r = finite(ranges_m, real=True, ndim=1)
    s = np.broadcast_to(finite(standard_deviation_m, real=True), r.shape)
    initial = finite(initial_position, real=True)
    if a.shape != (len(r), 2) or len(r) < 3 or initial.shape != (2,) or np.any(r <= 0) or np.any(s <= 0):
        raise ValueError("three aligned physical range observations required")

    def residual(x):
        return (np.linalg.norm(x - a, axis=1) - r) / s

    def jac(x):
        delta = x - a
        distance = np.linalg.norm(delta, axis=1)
        if min(distance) < 1e-8:
            raise ValueError("trial target coincides with radar")
        return delta / distance[:, None] / s[:, None]

    solved = least_squares(residual, initial, jac=jac, xtol=1e-12, gtol=1e-12, ftol=1e-12)
    information = solved.jac.T @ solved.jac
    if not solved.success or np.linalg.cond(information) > 1e10:
        raise ValueError("range geometry is unobservable or solve failed")
    return {"position": solved.x, "covariance": np.linalg.inv(information), "weighted_residual": solved.fun}


def velocity_from_doppler(position, anchors, receding_m_s, standard_deviation_m_s=0.2):
    p = finite(position, real=True)
    a = finite(anchors, real=True, ndim=2)
    v = finite(receding_m_s, real=True, ndim=1)
    s = np.broadcast_to(finite(standard_deviation_m_s, real=True), v.shape)
    if p.shape != (2,) or a.shape != (len(v), 2) or np.any(s <= 0):
        raise ValueError("aligned radar Doppler observations required")
    delta = p - a
    distance = np.linalg.norm(delta, axis=1)
    if np.any(distance < 1e-6):
        raise ValueError("nonzero sensor-target ranges required")
    h = delta / distance[:, None]
    weighted = h / s[:, None]
    information = weighted.T @ weighted
    if np.linalg.cond(information) > 1e10:
        raise ValueError("Doppler viewing directions are degenerate")
    cov = np.linalg.inv(information)
    mean = cov @ weighted.T @ (v / s)
    return {"velocity": mean, "covariance": cov}


def independent(mean_a, cov_a, mean_b, cov_b):
    a = finite(mean_a, real=True, ndim=1)
    b = finite(mean_b, real=True, ndim=1)
    if a.shape != b.shape:
        raise ValueError("matching estimate dimensions required")
    pa = covariance(cov_a, len(a))
    pb = covariance(cov_b, len(a))
    ia = np.linalg.inv(pa)
    ib = np.linalg.inv(pb)
    p = np.linalg.inv(ia + ib)
    return {"mean": p @ (ia @ a + ib @ b), "covariance": p}


def correlated(mean_a, cov_a, mean_b, cov_b, cross_covariance):
    a = finite(mean_a, real=True, ndim=1)
    b = finite(mean_b, real=True, ndim=1)
    n = len(a)
    if b.shape != a.shape:
        raise ValueError("aligned correlated estimates required")
    cross = finite(cross_covariance, real=True)
    pa = covariance(cov_a, n)
    pb = covariance(cov_b, n)
    if cross.shape != (n, n):
        raise ValueError("aligned cross covariance required")
    joint = covariance(np.block([[pa, cross], [cross.T, pb]]), 2 * n)
    h = np.vstack([np.eye(n), np.eye(n)])
    solve = np.linalg.solve(joint, h)
    p = np.linalg.inv(h.T @ solve)
    return {"mean": p @ solve.T @ np.r_[a, b], "covariance": p}


def intersection(mean_a, cov_a, mean_b, cov_b, weight=0.5):
    a = finite(mean_a, real=True, ndim=1)
    b = finite(mean_b, real=True, ndim=1)
    w = float(finite(weight, real=True))
    if a.shape != b.shape or not 0 <= w <= 1:
        raise ValueError("matching estimates and intersection weight in [0,1] required")
    ia = np.linalg.inv(covariance(cov_a, len(a)))
    ib = np.linalg.inv(covariance(cov_b, len(a)))
    p = np.linalg.inv(w * ia + (1 - w) * ib)
    return {"mean": p @ (w * ia @ a + (1 - w) * ib @ b), "covariance": p, "weight": w}
