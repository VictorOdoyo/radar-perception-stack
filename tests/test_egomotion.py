import numpy as np
import pytest

from radarlab import egomotion as m


def test_stationary_doppler_fit_recovers_ego_velocity_and_rejects_collinear_scene():
    angles = np.linspace(-0.6, 0.6, 20)
    truth = np.array([2.0, 10.0])
    v = -np.column_stack([np.sin(angles), np.cos(angles)]) @ truth
    assert np.allclose(m.fit_velocity(angles, v)["velocity_m_s"], truth)
    with pytest.raises(ValueError):
        m.fit_velocity([0, 0], [1, 2])
