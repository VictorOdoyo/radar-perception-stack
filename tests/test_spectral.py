import numpy as np
import pytest

from radarlab import spectral as m


def test_waveform_ambiguity_has_unit_zero_delay_zero_doppler_response():
    x = np.exp(1j * 0.01 * np.arange(100) ** 2)
    a = m.ambiguity(x, [-2, 0, 2], [-100, 0, 100], 10000)
    assert a[1, 1] == pytest.approx(1)
    assert np.max(abs(a)) <= 1 + 1e-12
