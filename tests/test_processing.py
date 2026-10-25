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
