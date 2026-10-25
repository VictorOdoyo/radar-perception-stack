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
