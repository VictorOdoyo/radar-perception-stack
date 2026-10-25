import numpy as np

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
