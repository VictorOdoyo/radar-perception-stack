import numpy as np
import pytest

from radarlab import spectral as m


def test_waveform_ambiguity_has_unit_zero_delay_zero_doppler_response():
    x = np.exp(1j * 0.01 * np.arange(100) ** 2)
    a = m.ambiguity(x, [-2, 0, 2], [-100, 0, 100], 10000)
    assert a[1, 1] == pytest.approx(1)
    assert np.max(abs(a)) <= 1 + 1e-12


def test_micro_doppler_preserves_power_and_zero_motion_is_identity():
    from radarlab.waveform import Radar

    p = Radar()
    x = np.ones((p.chirps, p.samples, p.receivers), complex)
    assert np.array_equal(m.micro_doppler(x, 0, 10, p), x)
    assert np.allclose(abs(m.micro_doppler(x, 0.001, 100, p)), 1)
