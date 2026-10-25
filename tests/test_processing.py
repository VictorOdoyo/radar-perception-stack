import numpy as np
import pytest

from radarlab import processing as m


def test_periodic_hann_has_independent_known_gain_and_enbw():
    r = m.window(256)
    assert r["coherent_gain"] == pytest.approx(0.5)
    assert r["enbw_bins"] == pytest.approx(1.5)


def test_range_fft_recovers_bin_centered_complex_amplitude():
    from radarlab.waveform import Radar

    p = Radar()
    x = np.exp(2j * np.pi * 20 * np.arange(p.samples) / p.samples)[None, :, None] * np.ones(
        (p.chirps, 1, p.receivers)
    )
    a = m.range_fft(x, p)
    assert np.allclose(a[:, 20, :], 1)


def test_doppler_fft_preserves_bin_tone_amplitude_and_sign():
    from radarlab.waveform import Radar

    p = Radar()
    x = np.exp(2j * np.pi * 5 * np.arange(p.chirps) / p.chirps)[:, None, None] * np.ones((1, 8, p.receivers))
    a = m.doppler_fft(x, p)
    assert np.allclose(a[p.chirps // 2 + 5], 1)


def test_fft_axes_match_independent_sampling_frequencies():
    from radarlab.waveform import Radar

    p = Radar()
    a = m.axes(p)
    assert a["beat_hz"][1] == p.sample_hz / p.samples
    assert a["receding_m_s"][p.chirps // 2] == 0


def test_stationary_removal_rejects_zero_doppler_and_retains_nonzero_bin():
    x = np.ones((64, 8, 2), complex)
    assert np.allclose(m.remove_stationary(x), 0)
    moving = np.exp(2j * np.pi * 5 * np.arange(64) / 64)[:, None, None] * x
    assert np.allclose(m.remove_stationary(moving), moving)


def test_log_parabola_recovers_independent_gaussian_peak():
    x = np.arange(21)
    p = np.exp(-0.5 * ((x - 10.25) / 2) ** 2)
    assert m.parabolic_peak(p, 10) == pytest.approx(10.25)


def test_power_db_uses_power_ratio_and_explicit_floor():
    assert np.allclose(m.power_db([0, 1, 0.01]), [-120, 0, -20])


def test_boxcar_noise_gain_matches_coherent_integration_length():
    from radarlab.waveform import Radar

    p = Radar()
    assert m.noise_gain(p, "boxcar", "boxcar") == pytest.approx(1 / (p.samples * p.chirps))
