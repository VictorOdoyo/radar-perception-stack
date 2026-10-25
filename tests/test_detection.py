import numpy as np
import pytest

from radarlab import detection as m


def test_ca_multiplier_closes_independent_exponential_tail_equation():
    n = 40
    pfa = 1e-5
    a = m.ca_multiplier(n, pfa)
    assert (1 + a / n) ** (-n) == pytest.approx(pfa)


def test_training_kernel_excludes_cut_and_guard_rectangle():
    k = m.training_kernel((3, 4), (1, 2))
    assert k.shape == (9, 13) and k[4, 6] == 0
    assert sum(k.ravel()) == 9 * 13 - 3 * 5


def test_ca_cfar_detects_embedded_target_and_marks_edges_unavailable():
    p = np.ones((64, 128))
    p[32, 60] = 1000
    r = m.ca_cfar(p)
    assert r["detections"][32, 60] and r["detections"].sum() == 1
    assert not r["valid"][0].any()


def test_nms_keeps_strongest_cell_within_declared_guard():
    p = np.zeros((20, 20))
    p[10, 10] = 10
    p[11, 11] = 5
    p[15, 15] = 4
    a = m.suppress(p, p > 0)
    assert a.tolist() == [[10, 10], [15, 15]]


def test_os_multiplier_matches_independent_product_tail():
    n = 40
    k = 30
    a = m.os_multiplier(n, k, 1e-4)
    assert np.prod(np.arange(n - k + 1, n + 1) / (np.arange(n - k + 1, n + 1) + a)) == pytest.approx(1e-4)


def test_os_cfar_retains_target_with_contaminated_training_cells():
    p = np.ones(100)
    p[50] = 100
    p[40] = 1000
    a = m.os_cfar(p)
    assert a["detections"][50] and not a["valid"][0]


def test_wilson_interval_remains_nonzero_at_zero_false_alarms():
    low, high = m.wilson(0, 1000)
    assert low == 0 and 0.003 < high < 0.004


def test_nms_rejects_wrong_dimensional_neighborhood():
    with pytest.raises(ValueError):
        m.suppress(np.ones((10, 10)), np.ones((10, 10), bool), radius=(2,))
