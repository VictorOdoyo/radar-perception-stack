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


def test_joseph_update_reduces_position_uncertainty_and_rejects_outlier():
    x, p, ok, _nis = m.correct(np.zeros(4), np.eye(4), [1, 0], np.eye(2))
    assert ok and x[0] == pytest.approx(0.5) and p[0, 0] == pytest.approx(0.5)
    assert not m.correct(np.zeros(4), np.eye(4), [100, 0], np.eye(2))[2]


def test_radial_update_changes_observable_velocity():
    x, p, ok = m.correct_radial([0, 30, 2, 0], np.eye(4), 1, gate=100)
    assert ok and x[3] > 0.8 and p[3, 3] < 1


def test_global_assignment_preserves_feasible_matches_and_unmatched_rows():
    a, missed, new = m.assign([[1, 2], [1.1, 100], [100, 100]])
    assert sorted(a) == [(0, 1), (1, 0)] and missed == [2] and new == []


def test_birth_covariance_retains_unobserved_tangential_velocity():
    x, p = m.initialize([0, 30], np.eye(2), 2)
    assert np.allclose(x, [0, 30, 0, 2]) and p[2, 2] > p[3, 3] * 100


def test_track_confirmation_requires_persistent_evidence_and_coasts_on_miss():
    r = m.Track(1, np.zeros(4), np.eye(4))
    r.observe(True)
    assert not r.confirmed
    r.observe(True)
    assert r.confirmed
    r.observe(False)
    assert r.confirmed and r.misses == 1


def test_tracker_confirms_measured_motion_coasts_and_deletes_without_truth():
    r = m.Tracker(maximum_misses=3)
    for k in range(5):
        tracks = r.update(
            k * 0.1,
            [{"position": np.array([1.0, 30 + k * 0.2]), "covariance": np.eye(2) * 0.1, "receding_m_s": 2}],
        )
    assert len(tracks) == 1 and tracks[0].identifier == 1
    assert r.update(0.5, [])[0].misses == 1
    r.update(0.6, [])
    assert r.update(0.7, []) == []
    with pytest.raises(ValueError):
        r.update(0.7, [])


def test_rts_smoother_reduces_early_position_uncertainty():
    times = np.arange(5) * 0.1
    state = np.zeros(4)
    covariance = np.eye(4)
    states = []
    covariances = []
    for _ in times:
        state, covariance = m.predict(state, covariance, 0.1)
        state, covariance, _ok, _nis = m.correct(state, covariance, [0, 0], np.eye(2) * 0.1)
        states.append(state)
        covariances.append(covariance)
    x = np.array(states)
    p = np.array(covariances)
    _state, cov = m.smooth(x, p, times)
    assert cov[0, 0, 0] < p[0, 0, 0] and np.allclose(cov[-1], p[-1])
