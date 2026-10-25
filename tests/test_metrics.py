import numpy as np
import pytest

from radarlab import metrics as m


def test_ospa_penalizes_missing_objects_and_is_symmetric():
    a = np.array([[0.0, 0], [1, 0]])
    b = np.array([[0.0, 0]])
    assert m.ospa(a, b) == pytest.approx(np.sqrt(50))
    assert m.ospa(a, b) == m.ospa(b, a)
    assert m.ospa(np.empty((0, 2)), np.empty((0, 2))) == 0


def test_nees_is_invariant_to_consistent_linear_coordinate_scaling():
    e = np.array([1.0, 2])
    p = np.array([[2.0, 0.2], [0.2, 3.0]])
    transform = np.diag([2.0, 3.0])
    assert m.nees(transform @ e, transform @ p @ transform.T) == pytest.approx(m.nees(e, p))
