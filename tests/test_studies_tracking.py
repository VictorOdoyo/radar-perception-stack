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
