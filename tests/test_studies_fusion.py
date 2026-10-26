from radarlab import studies_fusion as m


def test_correlation():
    s = m.correlation()["summary"]
    assert s["naive_nees_high_correlation"] > 3.5
    assert 1.8 < s["known_nees_high_correlation"] < 2.2
