import numpy as np
import pytest

from radarlab import advanced as m


def test_triangular_chirps_separate_range_and_doppler_coupling():
    slope = 7.5e12
    wave = 0.004
    r = 30
    v = 10
    fb = 2 * slope * r / 299792458
    fd = 2 * v / wave
    a = m.triangular_solution(fb + fd, -fb + fd, slope, wave)
    assert a["range_m"] == pytest.approx(r) and a["receding_m_s"] == pytest.approx(v)


def test_multi_prf_unwrap_recovers_velocity_outside_single_interval():
    truth = 45.0
    period = np.array([38.0, 31.0])
    measured = (truth + period / 2) % period - period / 2
    assert m.unwrap_velocity(measured, period, maximum_speed=80) == pytest.approx(truth)


def test_angle_fft_recovers_half_wavelength_array_direction():
    from radarlab.array import steering

    a = m.angle_fft(steering(0.3))
    j = np.argmax(np.where(a["valid"], a["power"], -1))
    assert abs(a["angle_rad"][j] - 0.3) < 0.01


def test_grating_lobes_expose_large_spacing_ambiguity():
    assert len(m.grating_lobes(0.2, 0.5)) == 1
    assert len(m.grating_lobes(0.2, 1.5)) > 1


def test_chirp_curvature_preserves_power_but_changes_fast_time_phase():
    from radarlab.waveform import Radar

    p = Radar()
    x = np.ones((p.chirps, p.samples, p.receivers), complex)
    a = m.nonlinear_beat(x, 1e9, p)
    assert np.allclose(abs(a), 1) and not np.allclose(a, x)
