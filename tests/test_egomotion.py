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


def test_ego_consensus_rejects_known_moving_objects():
    angles = np.linspace(-0.7, 0.7, 30)
    truth = np.array([1.0, 12.0])
    v = -np.column_stack([np.sin(angles), np.cos(angles)]) @ truth
    v[::5] += 5
    r = m.robust_velocity(angles, v)
    assert np.allclose(r["velocity_m_s"], truth) and sum(r["inliers"]) == 24
