import numpy as np

from radarlab import geometry as m


def test_polar_cartesian_roundtrip_uses_declared_forward_axis():
    a = m.polar_to_cartesian([10, 20], [0, 0.3])
    p = m.cartesian_to_polar(a)
    assert np.allclose(p, [[10, 0], [20, 0.3]])
    assert np.allclose(a[0], [0, 10])


def test_position_covariance_rotates_radial_and_tangential_variances():
    a = m.position_covariance(20, 0, 0.5, 0.1)
    assert np.allclose(a, np.diag([4, 0.25]))


def test_radial_velocity_jacobian_matches_finite_difference():
    state = np.array([10.0, 30.0, -1.0, 2.0])
    v, j = m.radial_measurement(state)
    h = 1e-5
    numerical = np.array(
        [
            (
                m.radial_measurement(state + np.eye(4)[k] * h)[0]
                - m.radial_measurement(state - np.eye(4)[k] * h)[0]
            )
            / (2 * h)
            for k in range(4)
        ]
    )
    assert np.allclose(j, numerical, atol=1e-9)
    assert v > 0
