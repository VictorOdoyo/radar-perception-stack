import numpy as np


def steering(angle_rad, receivers=8, spacing_wavelength=0.5):
    from .core import finite, integer, positive

    angle = finite(angle_rad, real=True)
    integer(receivers, 2)
    spacing = positive(spacing_wavelength)
    if np.any(abs(angle) > np.pi / 2):
        raise ValueError("front-hemisphere angle required")
    return np.exp(2j * np.pi * spacing * np.sin(angle)[..., None] * np.arange(receivers))
