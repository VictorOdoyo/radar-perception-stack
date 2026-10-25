import numpy as np


def window(length, kind="hann"):
    from scipy.signal.windows import get_window

    from .core import integer

    integer(length, 2)
    if kind not in ("hann", "hamming", "boxcar", "blackman"):
        raise ValueError("supported FFT window required")
    w = get_window(kind, length, fftbins=True)
    return {
        "values": w,
        "coherent_gain": float(w.mean()),
        "enbw_bins": float(length * np.sum(w * w) / np.sum(w) ** 2),
    }


def range_fft(cube, model=None, fft_size=None, window_kind="hann"):
    from .core import finite, integer
    from .waveform import Radar

    p = Radar() if model is None else model
    x = finite(cube, ndim=3)
    n = p.samples if fft_size is None else integer(fft_size, 2)
    if x.shape != (p.chirps, p.samples, p.receivers) or n < p.samples:
        raise ValueError("aligned cube and nontruncating FFT required")
    w = window(p.samples, window_kind)["values"]
    spectrum = np.fft.fft(x * w[None, :, None], n=n, axis=1) / sum(w)
    return spectrum[:, : n // 2, :]


def doppler_fft(range_spectrum, model=None, fft_size=None, window_kind="hann"):
    from .core import finite, integer
    from .waveform import Radar

    p = Radar() if model is None else model
    x = finite(range_spectrum, ndim=3)
    n = p.chirps if fft_size is None else integer(fft_size, 2)
    if x.shape[0] != p.chirps or n < p.chirps:
        raise ValueError("aligned chirp count and nontruncating Doppler FFT required")
    w = window(p.chirps, window_kind)["values"]
    return np.fft.fftshift(np.fft.fft(x * w[:, None, None], n=n, axis=0), axes=0) / sum(w)


def axes(model=None, range_size=None, doppler_size=None):
    from .waveform import C, Radar

    p = Radar() if model is None else model
    nr = p.samples if range_size is None else range_size
    nd = p.chirps if doppler_size is None else doppler_size
    return {
        "beat_hz": np.arange(nr // 2) * p.sample_hz / nr,
        "apparent_range_m": np.arange(nr // 2) * p.sample_hz / nr * C / (2 * p.slope),
        "receding_m_s": np.fft.fftshift(np.fft.fftfreq(nd, p.repetition_s)) * p.wavelength / 2,
    }


def remove_stationary(cube):
    from .core import finite

    x = finite(cube, ndim=3)
    return x - x.mean(axis=0, keepdims=True)
