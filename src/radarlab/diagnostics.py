import numpy as np


def audit(cube, full_scale=4):
    from .core import finite, positive

    x = finite(cube, ndim=3)
    rail = positive(full_scale)
    power = np.mean(abs(x) ** 2, axis=(0, 1))
    mean = np.mean(x, axis=(0, 1))
    return {
        "clipped_fraction": float(np.mean((abs(x.real) >= rail) | (abs(x.imag) >= rail))),
        "receiver_power": power,
        "dc_complex": mean,
        "gain_spread_db": float(10 * np.log10(max(power) / max(min(power), 1e-30))),
    }


def timing(time_s, expected_period_s, tolerance_fraction=0.1):
    from .core import positive, timebase

    t = timebase(time_s)
    period = positive(expected_period_s)
    tol = positive(tolerance_fraction)
    intervals = np.diff(t)
    return {
        "intervals_s": intervals,
        "late": intervals > period * (1 + tol),
        "early": intervals < period * (1 - tol),
        "maximum_jitter_s": float(max(abs(intervals - period))),
    }
