import numpy as np

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


def test_spatial_spectrum_study_retains_complex_covariance():
    r = m.spatial_spectra()
    assert np.iscomplexobj(r["arrays"]["covariance"])


def test_robust_ego_study_outperforms_unfiltered_least_squares():
    r = m.ego_consensus()["summary"]
    assert r["robust_error_m_s"] < r["least_squares_error_m_s"]


def test_dual_prf_study_preserves_out_of_interval_velocity():
    assert m.dual_prf()["summary"]["maximum_error_m_s"] < 1e-9
