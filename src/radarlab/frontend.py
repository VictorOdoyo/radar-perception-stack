import numpy as np


def channel_errors(cube, gains, phase_rad):
    from .core import finite

    x = finite(cube)
    g = finite(gains, real=True, ndim=1)
    phase = finite(phase_rad, real=True, ndim=1)
    if x.ndim != 3 or x.shape[-1] != len(g) or phase.shape != g.shape or np.any(g <= 0):
        raise ValueError("positive per-receiver gains and phases required")
    return x * (g * np.exp(1j * phase))[None, None, :]


def calibrate(snapshot, reference_amplitude=1):
    from .core import finite, positive

    x = finite(snapshot, ndim=1)
    amplitude = positive(reference_amplitude)
    if np.any(abs(x) < 1e-10):
        raise ValueError("unobservable zero receiver calibration")
    return amplitude / x
