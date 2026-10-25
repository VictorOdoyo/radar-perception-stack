import numpy as np
import pytest

from radarlab import waveform as m


def test_radar_timing_and_si_slope():
    p = m.Radar()
    assert p.slope == pytest.approx(7.5e12)
    assert p.samples / p.sample_hz <= p.ramp_s
    with pytest.raises(ValueError):
        m.Radar(sample_hz=1e6)


def test_frame_coordinates_follow_declared_clock_and_geometry():
    p = m.Radar()
    a = m.coordinates(p)
    assert a["slow_s"][1] == p.repetition_s
    assert np.diff(a["receiver_m"])[0] == pytest.approx(p.wavelength / 2)
