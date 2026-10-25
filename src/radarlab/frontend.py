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


def adc(signal, full_scale=4, bits=12):
    from .core import finite, integer, positive

    x = finite(signal)
    rail = positive(full_scale)
    integer(bits, 2)
    if bits > 24:
        raise ValueError("ADC precision is at most 24 bits")
    lsb = 2 * rail / (2**bits - 1)

    def quantize(v):
        return np.rint((np.clip(v, -rail, rail) + rail) / lsb) * lsb - rail

    return {
        "iq": quantize(x.real) + 1j * quantize(x.imag),
        "clipped": (abs(x.real) > rail) | (abs(x.imag) > rail),
        "lsb": lsb,
    }


def phase_noise(cube, std_step_rad=0.01, seed=42):
    from .core import finite, positive, stream

    x = finite(cube, ndim=3)
    rng = stream(seed, "oscillator-phase")
    phase = np.cumsum(rng.normal(0, positive(std_step_rad, allow_zero=True), size=x.shape[:2]), axis=1)
    return x * np.exp(1j * phase[:, :, None])


def blank_impulses(cube, magnitude_limit):
    from .core import finite, positive

    x = finite(cube, ndim=3)
    valid = np.all(abs(x) <= positive(magnitude_limit), axis=2)
    return {
        "iq": np.where(valid[:, :, None], x, 0),
        "valid": valid,
        "blanked_fraction": float(np.mean(~valid)),
    }


def drop_chirps(cube, probability=0.1, seed=42):
    from .core import finite, stream

    x = finite(cube, ndim=3)
    if not 0 <= probability <= 1:
        raise ValueError("drop probability in [0,1] required")
    valid = stream(seed, "chirp-dropout").random(len(x)) >= probability
    return {"iq": np.where(valid[:, None, None], x, 0), "valid": valid}
