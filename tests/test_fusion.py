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
