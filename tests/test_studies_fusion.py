from radarlab import studies_fusion as m


def test_correlation():
    s = m.correlation()["summary"]
    assert s["naive_nees_high_correlation"] > 3.5
    assert 1.8 < s["known_nees_high_correlation"] < 2.2


def test_weights():
    assert abs(m.weights()["summary"]["optimal_weight"] - 0.5) < 1e-4


def test_baseline():
    s = m.baseline()["summary"]
    assert s["wide_baseline_lateral_std_m"] < s["zero_baseline_lateral_std_m"]


def test_ranges():
    s = m.ranges()["summary"]
    assert s["position_rmse_m"] < 0.5
    assert 1.3 < s["mean_nees"] < 2.8


def test_velocity():
    assert m.velocity()["summary"]["velocity_rmse_m_s"] < 0.4
