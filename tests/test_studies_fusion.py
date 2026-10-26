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


def test_mounting():
    assert m.mounting()["summary"]["lateral_std_at_200m"] > 0.6


def test_alignment():
    s = m.alignment()["summary"]
    assert s["maximum_stale_bias_m"] > 3
    assert s["maximum_aligned_bias_m"] < 1e-10


def test_innovations():
    s = m.innovations()["summary"]
    assert s["raw_correlation"] > 0.7
    assert abs(s["whitened_correlation"]) < 0.08


def test_bias():
    assert m.bias()["summary"]["bias_at_3m_m"] > 0.5


def test_robust():
    s = m.robust()["summary"]
    assert s["robust_error_m"] < s["ordinary_error_m"] / 5
