import numpy as np

from radarlab import frontend as m


def test_channel_errors_apply_known_complex_receiver_gain():
    x = np.ones((4, 8, 2), complex)
    a = m.channel_errors(x, [1, 2], [0, np.pi / 2])
    assert np.allclose(a[..., 1], 2j)


def test_receiver_calibration_removes_known_complex_response():
    x = np.array([1, 2j, -0.5])
    c = m.calibrate(x)
    assert np.allclose(x * c, 1)


def test_iq_correction_recovers_complex_tone():
    x = np.exp(1j * np.arange(100) * 0.2)
    assert np.allclose(m.correct_iq(m.iq_imbalance(x)), x)


def test_adc_quantization_error_is_component_half_lsb_when_unclipped():
    x = np.array([0.1 + 0.3j, 8 + 0j])
    r = m.adc(x)
    assert not r["clipped"][0] and r["clipped"][1]
    assert abs((r["iq"][0] - x[0]).real) <= r["lsb"] / 2
