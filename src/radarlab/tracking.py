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
