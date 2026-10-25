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
