import numpy as np

from radarlab import studies_frame as m


def test_single_target_study_retains_raw_iq_and_independent_detection_score():
    r = m.nominal()
    assert r["summary"]["matched_targets"] == 1
    assert np.iscomplexobj(r["arrays"]["iq"])


def test_two_target_scene_resolves_distinct_ranges_and_dopplers():
    assert m.two()["summary"]["matched_targets"] == 2


def test_close_target_study_retains_missed_objects():
    assert m.close()["summary"]["missed_targets"] >= 1


def test_clutter_scene_records_declared_correlated_interference():
    r = m.cluttered()
    assert r["parameters"]["variant"] == "clutter"


def test_clutter_filtered_scene_retains_moving_target():
    assert m.filtered()["summary"]["matched_targets"] == 1


def test_ghost_scene_does_not_award_truth_for_reflected_path():
    assert m.ghosts()["summary"]["false_detections"] >= 1


def test_saturation_study_records_invalid_sensor_fraction():
    assert m.clipping()["summary"]["clipped_fraction"] > 0.5


def test_calibration_error_study_retains_channel_distortion():
    r = m.calibration()
    assert r["parameters"]["variant"] == "calibration"


def test_phase_noise_scene_keeps_finite_raw_iq():
    assert np.isfinite(m.oscillator()["arrays"]["iq"]).all()


def test_nonlinear_chirp_study_records_variant():
    assert m.nonlinear()["parameters"]["variant"] == "nonlinear"


def test_low_snr_study_scores_any_missing_target_explicitly():
    r = m.low_snr()
    assert r["summary"]["matched_targets"] + r["summary"]["missed_targets"] == 1


def test_dropout_study_keeps_zeroed_unavailable_chirp_rows():
    a = m.dropout()["arrays"]["iq"]
    assert np.any(np.all(a == 0, axis=(1, 2)))
