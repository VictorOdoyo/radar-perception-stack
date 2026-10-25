from dataclasses import dataclass, field

import numpy as np


def transition(dt_s, acceleration_spectral_density=1):
    from .core import positive

    dt = positive(dt_s)
    q = positive(acceleration_spectral_density, allow_zero=True)
    f = np.eye(4)
    f[:2, 2:] = np.eye(2) * dt
    process = q * np.block(
        [[np.eye(2) * dt**3 / 3, np.eye(2) * dt**2 / 2], [np.eye(2) * dt**2 / 2, np.eye(2) * dt]]
    )
    return f, process


def validate_state(state, covariance):
    from .core import finite

    x = finite(state, real=True)
    p = finite(covariance, real=True)
    if (
        x.shape != (4,)
        or p.shape != (4, 4)
        or not np.allclose(p, p.T)
        or np.min(np.linalg.eigvalsh(p)) < -1e-10
    ):
        raise ValueError("four-state symmetric PSD track covariance required")
    return x, p


def predict(state, covariance, dt_s, acceleration_spectral_density=1):
    x, p = validate_state(state, covariance)
    f, q = transition(dt_s, acceleration_spectral_density)
    return f @ x, f @ p @ f.T + q


def innovation(state, covariance, position, measurement_covariance):
    from .core import finite

    x, p = validate_state(state, covariance)
    z = finite(position, real=True)
    r = finite(measurement_covariance, real=True)
    if z.shape != (2,) or r.shape != (2, 2) or not np.allclose(r, r.T) or np.min(np.linalg.eigvalsh(r)) <= 0:
        raise ValueError("2D observation and positive definite covariance required")
    residual = z - x[:2]
    s = p[:2, :2] + r
    nis = float(residual @ np.linalg.solve(s, residual))
    return residual, s, nis


def correct(state, covariance, position, measurement_covariance, gate=9.21034):
    from .core import positive

    x, p = validate_state(state, covariance)
    y, s, nis = innovation(x, p, position, measurement_covariance)
    if nis > positive(gate):
        return x.copy(), p.copy(), False, nis
    gain = np.linalg.solve(s, p[:, :2].T).T
    h = np.column_stack([np.eye(2), np.zeros((2, 2))])
    j = np.eye(4) - gain @ h
    result = j @ p @ j.T + gain @ measurement_covariance @ gain.T
    return x + gain @ y, (result + result.T) / 2, True, nis


def correct_radial(state, covariance, receding_m_s, std_m_s=0.3, gate=9):
    from .core import positive
    from .geometry import radial_measurement

    x, p = validate_state(state, covariance)
    value, h = radial_measurement(x)
    r = positive(std_m_s) ** 2
    if not np.isfinite(receding_m_s):
        raise ValueError("finite radial velocity required")
    y = receding_m_s - value
    s = float(h @ p @ h + r)
    nis = y * y / s
    if nis > positive(gate):
        return x.copy(), p.copy(), False
    gain = p @ h / s
    j = np.eye(4) - np.outer(gain, h)
    post = j @ p @ j.T + np.outer(gain, gain) * r
    return x + gain * y, (post + post.T) / 2, True


def assign(cost, gate=9.21034):
    from scipy.optimize import linear_sum_assignment

    from .core import finite, positive

    c = finite(cost, real=True, ndim=2)
    limit = positive(gate)
    if np.any(c < 0):
        raise ValueError("nonnegative association costs required")
    rows, cols = c.shape
    if rows == 0 or cols == 0:
        return [], list(range(rows)), list(range(cols))
    augmented = np.full((rows, cols + rows), limit + 1e6)
    augmented[:, :cols] = np.where(c <= limit, c, limit + 1e6)
    augmented[np.arange(rows), cols + np.arange(rows)] = limit
    rr, cc = linear_sum_assignment(augmented)
    matches = [(int(i), int(j)) for i, j in zip(rr, cc, strict=True) if j < cols and c[i, j] <= limit]
    return (
        matches,
        [i for i in range(rows) if all(i != a for a, _ in matches)],
        [j for j in range(cols) if all(j != b for _, b in matches)],
    )


def initialize(position, position_covariance, receding_m_s=0, radial_std=0.5, tangential_std=10):
    from .core import finite, positive

    pos = finite(position, real=True)
    cov = finite(position_covariance, real=True)
    if (
        pos.shape != (2,)
        or cov.shape != (2, 2)
        or np.linalg.norm(pos) < 1e-6
        or not np.allclose(cov, cov.T)
        or np.min(np.linalg.eigvalsh(cov)) <= 0
    ):
        raise ValueError("observable position and positive covariance required")
    if not np.isfinite(receding_m_s):
        raise ValueError("finite radial velocity required")
    unit = pos / np.linalg.norm(pos)
    tangent = np.array([unit[1], -unit[0]])
    velocity = unit * receding_m_s
    p = np.zeros((4, 4))
    p[:2, :2] = cov
    p[2:, 2:] = positive(radial_std) ** 2 * np.outer(unit, unit) + positive(tangential_std) ** 2 * np.outer(
        tangent, tangent
    )
    return np.r_[pos, velocity], p


@dataclass
class Track:
    identifier: int
    state: np.ndarray
    covariance: np.ndarray
    age: int = 1
    hits: int = 1
    misses: int = 0
    confirmed: bool = False
    history: list = field(default_factory=lambda: [True])

    def observe(self, matched, confirmation_hits=3, window=5):
        from .core import integer

        integer(confirmation_hits)
        integer(window)
        if confirmation_hits > window:
            raise ValueError("confirmation hits must fit history")
        self.age += 1
        self.hits += int(matched)
        self.misses = 0 if matched else self.misses + 1
        self.history = [*self.history, bool(matched)][-window:]
        self.confirmed = self.confirmed or sum(self.history) >= confirmation_hits


class Tracker:
    def __init__(self, gate=9.21034, process_noise=1, maximum_misses=5):
        from .core import integer, positive

        self.gate = positive(gate)
        self.process_noise = positive(process_noise, allow_zero=True)
        self.maximum_misses = integer(maximum_misses)
        self.tracks = []
        self.time = None
        self.next_id = 1

    def update(self, time_s, detections):
        if not np.isfinite(time_s) or (self.time is not None and time_s <= self.time):
            raise ValueError("strictly increasing finite tracker time required")
        records = list(detections)
        for d in records:
            initialize(d["position"], d["covariance"], d.get("receding_m_s", 0))
        if self.time is not None:
            for track in self.tracks:
                track.state, track.covariance = predict(
                    track.state, track.covariance, time_s - self.time, self.process_noise
                )
        self.time = float(time_s)
        costs = np.array(
            [
                [innovation(t.state, t.covariance, d["position"], d["covariance"])[2] for d in records]
                for t in self.tracks
            ]
        ).reshape(len(self.tracks), len(records))
        matches, missed, new = assign(costs, self.gate)
        for i, j in matches:
            track = self.tracks[i]
            d = records[j]
            track.state, track.covariance, _ok, _nis = correct(
                track.state, track.covariance, d["position"], d["covariance"], self.gate
            )
            if "receding_m_s" in d:
                track.state, track.covariance, _ok = correct_radial(
                    track.state, track.covariance, d["receding_m_s"]
                )
            track.observe(True)
        for i in missed:
            self.tracks[i].observe(False)
        self.tracks = [t for t in self.tracks if t.misses < (self.maximum_misses if t.confirmed else 2)]
        for j in new:
            d = records[j]
            x, p = initialize(d["position"], d["covariance"], d.get("receding_m_s", 0))
            self.tracks.append(Track(self.next_id, x, p))
            self.next_id += 1
        return [t for t in self.tracks if t.confirmed]


def smooth(filtered_states, filtered_covariances, times, process_noise=1):
    from .core import finite, timebase

    x = finite(filtered_states, real=True, ndim=2)
    p = finite(filtered_covariances, real=True, ndim=3)
    t = timebase(times)
    if x.shape != (len(t), 4) or p.shape != (len(t), 4, 4):
        raise ValueError("aligned saved CV states required")
    out = x.copy()
    cov = p.copy()
    for k in range(len(t) - 2, -1, -1):
        f, q = transition(t[k + 1] - t[k], process_noise)
        pred = f @ p[k] @ f.T + q
        gain = np.linalg.solve(pred, (p[k] @ f.T).T).T
        out[k] = x[k] + gain @ (out[k + 1] - f @ x[k])
        cov[k] = p[k] + gain @ (cov[k + 1] - pred) @ gain.T
        cov[k] = (cov[k] + cov[k].T) / 2
    return out, cov
