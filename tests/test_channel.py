import numpy as np
import pytest

from radarlab import channel as m


def test_coherent_cube_axes_have_independent_analytic_phase_steps():
    from radarlab.waveform import Radar, Target, beat_frequency

    p = Radar()
    target = Target()
    a = m.target_cube(target, p)
    assert a.shape == (p.chirps, p.samples, p.receivers)
    assert np.angle(a[0, 1, 0] / a[0, 0, 0]) == pytest.approx(
        2 * np.pi * beat_frequency(30, 5, p) / p.sample_hz
    )
    assert np.angle(a[1, 0, 0] / a[0, 0, 0]) == pytest.approx(
        2 * np.pi * 2 * 5 / p.wavelength * p.repetition_s
    )
