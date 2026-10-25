import numpy as np

from radarlab import studies_frame as m


def test_single_target_study_retains_raw_iq_and_independent_detection_score():
    r = m.nominal()
    assert r["summary"]["matched_targets"] == 1
    assert np.iscomplexobj(r["arrays"]["iq"])
