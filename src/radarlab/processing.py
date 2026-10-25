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
