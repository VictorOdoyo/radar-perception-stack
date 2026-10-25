import numpy as np

from radarlab import studies_tracking as m


def test_tracking_study_finishes_with_two_measured_tracks():
    r = m.nominal()
    assert r["summary"]["final_confirmed_tracks"] == 2
    assert r["summary"]["final_ospa_m"] < 1


def test_short_gap_keeps_prediction_and_frame_validity_separate():
    r = m.gap()
    a = r["arrays"]
    assert not a["valid_frame"][11]
    assert a["detection_count"][11] == 0 and a["track_count"][11] == 2


def test_long_gap_deletes_stale_tracks_and_records_identity_change():
    r = m.long_gap()
    assert r["summary"]["identity_switches"] >= 2
    assert r["arrays"]["track_count"][17] == 0


def test_maneuver_scene_records_model_mismatch():
    r = m.maneuver()
    assert r["parameters"]["turn"]
    assert np.max(r["arrays"]["truth"][:, 0, 2]) > 3
