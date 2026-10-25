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


def validate_state(state, covariance):
    from .core import finite

    x = finite(state, real=True)
    p = finite(covariance, real=True)
    if (
        x.shape != (4,)
        or p.shape != (4, 4)
        or not np.allclose(p, p.T)
        or np.min(np.linalg.eigvalsh(p)) < -1e-10
    ):
        raise ValueError("four-state symmetric PSD track covariance required")
    return x, p


def predict(state, covariance, dt_s, acceleration_spectral_density=1):
    x, p = validate_state(state, covariance)
    f, q = transition(dt_s, acceleration_spectral_density)
    return f @ x, f @ p @ f.T + q
