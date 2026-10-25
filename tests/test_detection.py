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
