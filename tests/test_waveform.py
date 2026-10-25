import pytest

from radarlab import waveform as m


def test_radar_timing_and_si_slope():
    p = m.Radar()
    assert p.slope == pytest.approx(7.5e12)
    assert p.samples / p.sample_hz <= p.ramp_s
    with pytest.raises(ValueError):
        m.Radar(sample_hz=1e6)
