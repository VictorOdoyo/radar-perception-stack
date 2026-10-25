import numpy as np
import pytest

from radarlab import array as m


def test_steering_phase_difference_matches_wavelength_geometry():
    a = m.steering(np.pi / 6)
    assert np.angle(a[1] / a[0]) == pytest.approx(np.pi / 2)
