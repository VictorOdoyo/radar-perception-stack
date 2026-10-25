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


def iq_imbalance(signal, gain_ratio=1.05, phase_error_rad=0.03):
    from .core import finite, positive

    x = finite(signal)
    gain = positive(gain_ratio)
    if not np.isfinite(phase_error_rad) or abs(phase_error_rad) >= np.pi / 2:
        raise ValueError("invertible quadrature phase required")
    return x.real + 1j * gain * (x.imag * np.cos(phase_error_rad) + x.real * np.sin(phase_error_rad))


def correct_iq(signal, gain_ratio=1.05, phase_error_rad=0.03):
    from .core import positive

    x = np.asarray(signal)
    gain = positive(gain_ratio)
    if not np.isfinite(phase_error_rad) or abs(phase_error_rad) >= np.pi / 2:
        raise ValueError("invertible quadrature phase required")
    return x.real + 1j * (x.imag / gain - x.real * np.sin(phase_error_rad)) / np.cos(phase_error_rad)
