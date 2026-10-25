from radarlab import studies_tracking as m


def test_tracking_study_finishes_with_two_measured_tracks():
    r = m.nominal()
    assert r["summary"]["final_confirmed_tracks"] == 2
    assert r["summary"]["final_ospa_m"] < 1
