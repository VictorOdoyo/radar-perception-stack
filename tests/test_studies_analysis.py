from radarlab import studies_analysis as m


def test_cfar_monte_carlo_matches_nominal_probability_with_sampling_error():
    r = m.cfar_rates()
    assert r["summary"]["maximum_absolute_error"] < 0.01
