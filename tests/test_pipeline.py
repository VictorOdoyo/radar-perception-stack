import numpy as np

from radarlab import pipeline as m


def test_spectrum_maps_analytic_target_to_sampled_range_doppler():
    from radarlab.channel import synthesize
    from radarlab.waveform import Radar, Target

    p = Radar()
    a = m.spectrum(synthesize([Target()], p, noise_power=0), p)
    k, j = np.unravel_index(np.argmax(a["power"]), a["power"].shape)
    assert abs(a["axes"]["apparent_range_m"][j] - 30) < 0.6
    assert abs(a["axes"]["receding_m_s"][k] - 5) < 0.4


def test_detection_chain_recovers_range_velocity_angle_without_truth_input():
    from radarlab.channel import synthesize
    from radarlab.waveform import Target

    a = m.detect(synthesize([Target(range_m=30, receding_m_s=5, angle_rad=0.2)], noise_power=0.1))
    d = min(a["detections"], key=lambda d: abs(d["range_m"] - 30))
    assert abs(d["range_m"] - 30) < 0.1 and abs(d["receding_m_s"] - 5) < 0.1
    assert abs(d["angle_rad"] - 0.2) < 0.01
