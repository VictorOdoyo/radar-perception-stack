import numpy as np


def ambiguity(signal, delay_samples, doppler_hz, sample_hz):
    from .core import finite, positive

    x = finite(signal, ndim=1)
    fs = positive(sample_hz)
    delays = np.asarray(delay_samples)
    if delays.dtype.kind not in "iu" or np.any(abs(delays) >= len(x)):
        raise ValueError("integer delays shorter than waveform required")
    f = finite(doppler_hz, real=True, ndim=1)
    t = np.arange(len(x)) / fs
    energy = float(np.vdot(x, x).real)
    if energy <= 0:
        raise ValueError("nonzero waveform energy required")
    result = np.zeros((len(f), len(delays)), complex)
    for k, d in enumerate(delays):
        shifted = np.zeros_like(x)
        if d >= 0:
            shifted[d:] = x[: len(x) - d]
        else:
            shifted[:d] = x[-d:]
        result[:, k] = np.exp(2j * np.pi * f[:, None] * t) @ (x * np.conj(shifted)) / energy
    return result


def micro_doppler(cube, amplitude_m, frequency_hz, model=None):
    from .core import finite, positive
    from .waveform import Radar, coordinates

    p = Radar() if model is None else model
    x = finite(cube, ndim=3)
    if x.shape[0] != p.chirps:
        raise ValueError("aligned slow-time frame required")
    displacement = positive(amplitude_m, allow_zero=True) * np.sin(
        2 * np.pi * positive(frequency_hz) * coordinates(p)["slow_s"]
    )
    return x * np.exp(4j * np.pi * displacement / p.wavelength)[:, None, None]
