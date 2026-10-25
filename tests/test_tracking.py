import numpy as np
import pytest

from radarlab import tracking as m


def test_cv_transition_composes_and_process_covariance_is_psd():
    f, q = m.transition(0.2, 2)
    g, r = m.transition(0.1, 2)
    assert np.allclose(f, g @ g)
    assert np.allclose(q, g @ r @ g.T + r)
    assert np.linalg.eigvalsh(q).min() > 0


def test_prediction_advances_position_without_observation():
    x, p = m.predict([1, 2, 3, 4], np.eye(4), 0.5)
    assert np.allclose(x, [2.5, 4, 3, 4])
    assert np.trace(p) > 4


def test_innovation_gate_scales_with_joint_uncertainty():
    _y, s, nis = m.innovation([0, 0, 0, 0], np.eye(4), [2, 0], np.eye(2))
    assert nis == pytest.approx(2)
    assert np.allclose(s, 2 * np.eye(2))
