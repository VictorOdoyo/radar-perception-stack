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


def test_circular_receiver_noise_matches_declared_total_power():
    a = m.synthesize([], noise_power=2)
    assert np.mean(abs(a) ** 2) == pytest.approx(2, rel=0.02)
    assert np.mean(a.real * a.imag) == pytest.approx(0, abs=0.02)
    assert np.array_equal(a, m.synthesize([], noise_power=2))


def test_monostatic_received_power_obeys_fourth_power_range_loss():
    a = m.received_power(10, 1, 1, 10, 0.004)
    assert m.received_power(20, 1, 1, 10, 0.004) == pytest.approx(a / 16)


def test_noise_bandwidth_and_noise_figure_scale_linearly():
    assert m.thermal_noise(2e6) == pytest.approx(2 * m.thermal_noise(1e6))
    assert m.thermal_noise(1e6, noise_figure_db=10) == pytest.approx(
        10 * m.thermal_noise(1e6, noise_figure_db=0)
    )
