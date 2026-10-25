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
