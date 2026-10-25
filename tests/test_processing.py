import pytest

from radarlab import processing as m


def test_periodic_hann_has_independent_known_gain_and_enbw():
    r = m.window(256)
    assert r["coherent_gain"] == pytest.approx(0.5)
    assert r["enbw_bins"] == pytest.approx(1.5)
