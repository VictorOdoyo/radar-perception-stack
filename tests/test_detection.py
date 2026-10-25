import pytest

from radarlab import detection as m


def test_ca_multiplier_closes_independent_exponential_tail_equation():
    n = 40
    pfa = 1e-5
    a = m.ca_multiplier(n, pfa)
    assert (1 + a / n) ** (-n) == pytest.approx(pfa)
