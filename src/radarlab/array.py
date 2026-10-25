import numpy as np


def steering(angle_rad, receivers=8, spacing_wavelength=0.5):
    from .core import finite, integer, positive

    angle = finite(angle_rad, real=True)
    integer(receivers, 2)
    spacing = positive(spacing_wavelength)
    if np.any(abs(angle) > np.pi / 2):
        raise ValueError("front-hemisphere angle required")
    return np.exp(2j * np.pi * spacing * np.sin(angle)[..., None] * np.arange(receivers))


def bartlett(snapshot, angle_grid=None, spacing_wavelength=0.5):
    from .core import finite

    x = finite(snapshot, ndim=1)
    grid = (
        np.linspace(-np.pi / 2, np.pi / 2, 721)
        if angle_grid is None
        else finite(angle_grid, real=True, ndim=1)
    )
    manifold = steering(grid, len(x), spacing_wavelength)
    power = abs(manifold.conj() @ x) ** 2 / len(x) ** 2
    return {"angle_rad": float(grid[np.argmax(power)]), "grid_rad": grid, "power": power}
