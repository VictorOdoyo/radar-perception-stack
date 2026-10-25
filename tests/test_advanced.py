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
