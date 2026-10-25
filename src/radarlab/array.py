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


def phase_angle(snapshot, spacing_wavelength=0.5):
    from .core import finite, positive

    x = finite(snapshot, ndim=1)
    spacing = positive(spacing_wavelength)
    if len(x) < 2 or spacing > 0.5:
        raise ValueError("unambiguous adjacent baseline required")
    correlation = np.vdot(x[:-1], x[1:])
    if abs(correlation) < 1e-12:
        raise ValueError("unobservable array phase")
    return float(np.arcsin(np.clip(np.angle(correlation) / (2 * np.pi * spacing), -1, 1)))


def covariance(snapshots, loading=0.001):
    from .core import finite, positive

    x = finite(snapshots, ndim=2)
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("multiple snapshots and receivers required")
    r = x.T @ x.conj() / len(x)
    r = (r + r.conj().T) / 2
    return r + np.eye(r.shape[0]) * positive(loading, allow_zero=True) * np.trace(r).real / r.shape[0]


def mvdr(covariance_matrix, angle_grid, spacing_wavelength=0.5):
    from .core import finite

    r = finite(covariance_matrix, ndim=2)
    if r.shape[0] != r.shape[1] or not np.allclose(r, r.conj().T) or np.min(np.linalg.eigvalsh(r)) <= 0:
        raise ValueError("Hermitian positive definite covariance required")
    a = steering(angle_grid, len(r), spacing_wavelength)
    solved = np.linalg.solve(r, a.T)
    return 1 / np.maximum(np.einsum("ij,ji->i", a.conj(), solved).real, 1e-30)


def music(covariance_matrix, angle_grid, sources=1, spacing_wavelength=0.5):
    from .core import finite, integer

    r = finite(covariance_matrix, ndim=2)
    integer(sources)
    if r.shape[0] != r.shape[1] or sources >= len(r) or not np.allclose(r, r.conj().T):
        raise ValueError("Hermitian covariance and fewer sources than receivers required")
    _, vectors = np.linalg.eigh(r)
    noise = vectors[:, : len(r) - sources]
    a = steering(angle_grid, len(r), spacing_wavelength)
    denominator = np.sum(abs(a.conj() @ noise) ** 2, axis=1)
    return 1 / np.maximum(denominator, 1e-20)


def spatial_smoothing(snapshots, subarray=5):
    from .core import finite, integer

    x = finite(snapshots, ndim=2)
    integer(subarray, 2)
    if subarray > x.shape[1]:
        raise ValueError("subarray must fit the receiver aperture")
    matrices = [covariance(x[:, k : k + subarray], loading=0) for k in range(x.shape[1] - subarray + 1)]
    r = sum(matrices) / len(matrices)
    j = np.eye(subarray)[::-1]
    return 0.5 * (r + j @ r.conj() @ j)


def angle_crlb(angle_rad, snr_linear, snapshots=1, receivers=8, spacing_wavelength=0.5):
    from .core import integer, positive

    snr = positive(snr_linear)
    integer(snapshots)
    integer(receivers, 2)
    spacing = positive(spacing_wavelength)
    if not np.isfinite(angle_rad) or abs(angle_rad) >= np.pi / 2:
        raise ValueError("interior observable angle required")
    positions = np.arange(receivers) - (receivers - 1) / 2
    information = 2 * snapshots * snr * (2 * np.pi * spacing * np.cos(angle_rad)) ** 2 * np.sum(positions**2)
    return 1 / information
