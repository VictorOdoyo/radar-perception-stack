import numpy as np
import pytest

from radarlab import array as m


def test_steering_phase_difference_matches_wavelength_geometry():
    a = m.steering(np.pi / 6)
    assert np.angle(a[1] / a[0]) == pytest.approx(np.pi / 2)


def test_bartlett_recovers_off_broadside_synthetic_angle():
    angle = 0.23
    r = m.bartlett(m.steering(angle))
    assert abs(r["angle_rad"] - angle) < 0.003
