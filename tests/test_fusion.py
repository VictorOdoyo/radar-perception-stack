import numpy as np
import pytest

from radarlab import fusion as m


def test_covariance_rejects_singular():
    assert m.covariance(np.eye(2)).shape == (2, 2)
    with pytest.raises(ValueError):
        m.covariance([[1, 1], [1, 1]])


def test_track_frame_rotation():
    x, p = m.transform_track([1, 0, 2, 0], np.diag([1, 2, 3, 4]), [10, 20], np.pi / 2)
    assert np.allclose(x, [10, 21, 0, 2])
    assert np.allclose(np.diag(p), [2, 1, 4, 3])


def test_mount_yaw_range_scaling():
    a = m.mounting_covariance([0, 10, 1, 0], np.diag([0.01, 0.01, 0.001]))
    b = m.mounting_covariance([0, 100, 1, 0], np.diag([0.01, 0.01, 0.001]))
    assert b[0, 0] > a[0, 0] * 50
    assert a[0, 3] != 0


def test_alignment_clock():
    x, p = m.align([0, 0, 2, 3], np.eye(4), 1, 3, 0)
    assert np.allclose(x, [4, 6, 2, 3])
    assert p[0, 0] == pytest.approx(5)
    with pytest.raises(ValueError):
        m.align(x, p, 3, 1)


def test_polar_jacobian_difference():
    x = np.array([12.0, 45.0])
    z, h = m.polar_observation(x)
    eps = 1e-5
    numeric = np.column_stack(
        [
            (m.polar_observation(x + np.eye(2)[k] * eps)[0] - m.polar_observation(x - np.eye(2)[k] * eps)[0])
            / (2 * eps)
            for k in range(2)
        ]
    )
    assert np.allclose(h, numeric, atol=1e-8)
    assert z[1] == pytest.approx(np.arctan2(12, 45))


def test_information_adds():
    one = m.position_information([10, 50], [[0, 0]])["covariance_bound"]
    two = m.position_information([10, 50], [[0, 0], [0, 0]])["covariance_bound"]
    assert np.allclose(two, one / 2)


def test_three_range_localization():
    a = np.array([[0, 0], [40, 0], [0, 40]])
    truth = np.array([12.0, 18.0])
    r = np.linalg.norm(truth - a, axis=1)
    fit = m.localize_ranges(a, r, 0.1, [10, 20])
    assert np.allclose(fit["position"], truth, atol=1e-8)


def test_multi_view_velocity():
    anchors = np.array([[0.0, 0], [40, 0], [0, 40]])
    p = np.array([10.0, 20.0])
    truth = np.array([2.0, -3.0])
    h = p - anchors
    h /= np.linalg.norm(h, axis=1)[:, None]
    fit = m.velocity_from_doppler(p, anchors, h @ truth)
    assert np.allclose(fit["velocity"], truth)


def test_independent_fusion():
    r = m.independent([0, 0], np.eye(2), [2, 2], np.eye(2))
    assert np.allclose(r["mean"], [1, 1])
    assert np.allclose(r["covariance"], np.eye(2) / 2)


def test_known_correlation():
    r = m.correlated([0, 0], np.eye(2), [2, 2], np.eye(2), 0.8 * np.eye(2))
    assert np.allclose(r["mean"], [1, 1])
    assert np.allclose(r["covariance"], 0.9 * np.eye(2))


def test_ci_identical_information_not_counted_twice():
    r = m.intersection([1, 2], np.eye(2), [1, 2], np.eye(2))
    assert np.allclose(r["covariance"], np.eye(2))
    assert np.allclose(r["mean"], [1, 2])


def test_ci_complementary_information():
    r = m.optimize_intersection([0, 0], np.diag([1, 9]), [0, 0], np.diag([9, 1]))
    assert r["weight"] == pytest.approx(0.5, abs=1e-5)
    assert np.linalg.det(r["covariance"]) < 9


def test_consistency_interval():
    lo, hi = m.consistency_bounds(4, 1000)
    assert lo < 4 < hi and hi - lo < 1


def test_whitened_energy_equals_mahalanobis():
    e = np.array([2.0, -1])
    p = np.array([[2.0, 0.5], [0.5, 1]])
    w = m.whiten(e, p)
    assert w @ w == pytest.approx(e @ np.linalg.solve(p, e))
