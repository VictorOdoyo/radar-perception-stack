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
