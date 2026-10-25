import numpy as np


def triangular_solution(up_hz, down_hz, slope_hz_s, wavelength_m):
    from .core import finite, positive

    slope = positive(slope_hz_s)
    wave = positive(wavelength_m)
    up = finite(up_hz, real=True)
    down = finite(down_hz, real=True)
    return {"range_m": (up - down) * 299792458 / (4 * slope), "receding_m_s": (up + down) * wave / 4}


def unwrap_velocity(measured, periods, maximum_speed=100, tolerance=0.1):
    from .core import finite, positive

    v = finite(measured, real=True, ndim=1)
    period = finite(periods, real=True, ndim=1)
    maximum = positive(maximum_speed)
    tol = positive(tolerance)
    if v.shape != period.shape or len(v) < 2 or np.any(period <= 0) or np.any(abs(v) > period / 2):
        raise ValueError("aligned principal velocities and positive alias periods required")
    count = int(np.ceil(maximum / period[0])) + 1
    candidates = v[0] + period[0] * np.arange(-count, count + 1)
    candidates = candidates[abs(candidates) <= maximum]
    errors = (candidates[:, None] - v + period / 2) % period - period / 2
    scores = np.sqrt(np.mean(errors**2, axis=1))
    valid = candidates[scores <= tol]
    if len(valid) != 1:
        raise ValueError("velocity ambiguity unresolved or inconsistent")
    return float(valid[0])


def angle_fft(snapshot, spacing_wavelength=0.5, fft_size=512):
    from .core import finite, integer, positive

    x = finite(snapshot, ndim=1)
    spacing = positive(spacing_wavelength)
    integer(fft_size, 2)
    if fft_size < len(x):
        raise ValueError("nontruncating angle FFT required")
    u = np.fft.fftshift(np.fft.fftfreq(fft_size)) / spacing
    valid = abs(u) <= 1
    spectrum = np.fft.fftshift(np.fft.fft(x, fft_size)) / len(x)
    return {"angle_rad": np.arcsin(np.clip(u, -1, 1)), "power": abs(spectrum) ** 2, "valid": valid}


def grating_lobes(angle_rad, spacing_wavelength):
    from .core import positive

    spacing = positive(spacing_wavelength)
    if not np.isfinite(angle_rad) or abs(angle_rad) > np.pi / 2:
        raise ValueError("front-hemisphere direction required")
    orders = np.arange(-int(np.ceil(2 * spacing)) - 1, int(np.ceil(2 * spacing)) + 2)
    sine = np.sin(angle_rad) + orders / spacing
    return np.arcsin(sine[abs(sine) <= 1])


def nonlinear_beat(cube, curvature_hz_s, model=None):
    from .core import finite
    from .waveform import Radar, coordinates

    p = Radar() if model is None else model
    x = finite(cube, ndim=3)
    if x.shape[1] != p.samples or not np.isfinite(curvature_hz_s):
        raise ValueError("aligned fast time and finite chirp curvature required")
    t = coordinates(p)["fast_s"]
    return x * np.exp(1j * np.pi * curvature_hz_s * t * t)[None, :, None]


def leakage(cube, amplitude=10, beat_hz=1000, model=None):
    from .core import finite, positive
    from .waveform import Radar, coordinates

    p = Radar() if model is None else model
    x = finite(cube, ndim=3)
    t = coordinates(p)["fast_s"]
    if x.shape[1] != len(t) or not np.isfinite(beat_hz):
        raise ValueError("aligned fast-time leakage required")
    return x + positive(amplitude, allow_zero=True) * np.exp(2j * np.pi * beat_hz * t)[None, :, None]
