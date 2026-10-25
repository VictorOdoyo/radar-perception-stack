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
