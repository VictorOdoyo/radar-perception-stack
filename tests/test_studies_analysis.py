from radarlab import studies_analysis as m


def test_cfar_monte_carlo_matches_nominal_probability_with_sampling_error():
    r = m.cfar_rates()
    assert r["summary"]["maximum_absolute_error"] < 0.01


def test_window_cfar_study_uses_actual_fft_noise():
    r = m.window_cfar()
    assert 0.005 < r["summary"]["boxcar_rate"] < 0.02


def test_os_study_retains_detection_under_reference_contamination():
    r = m.os_robustness()
    assert r["summary"]["final_os_pd"] > r["summary"]["final_ca_pd"]
