import numpy as np


def target_cube(target, model=None, allow_alias=False):
    from .waveform import Radar, beat_frequency, coordinates, limits

    p = Radar() if model is None else model
    axes = coordinates(p)
    bounds = limits(p)
    beat = float(beat_frequency(target.range_m, target.receding_m_s, p))
    fd = 2 * target.receding_m_s / p.wavelength
    if not allow_alias and (
        not 0 <= beat < p.sample_hz / 2 or abs(target.receding_m_s) >= bounds["receding_nyquist_m_s"]
    ):
        raise ValueError("target aliases sampled positive-range or Doppler interval")
    phase = (
        2
        * np.pi
        * (
            axes["slow_s"][:, None, None] * fd
            + axes["fast_s"][None, :, None] * beat
            + axes["receiver_m"][None, None, :] * np.sin(target.angle_rad) / p.wavelength
        )
        + target.phase_rad
    )
    return target.amplitude * np.exp(1j * phase)


def synthesize(targets, model=None, noise_power=1.0, seed=42, allow_alias=False):
    from .core import positive, stream
    from .waveform import Radar

    p = Radar() if model is None else model
    power = positive(noise_power, allow_zero=True)
    data = np.zeros((p.chirps, p.samples, p.receivers), complex)
    for target in targets:
        data += target_cube(target, p, allow_alias)
    rng = stream(seed, "receiver-thermal-noise")
    return data + np.sqrt(power / 2) * (rng.normal(size=data.shape) + 1j * rng.normal(size=data.shape))
