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


def test_dechirped_stationary_delay_has_positive_range_beat():
    p = m.Radar()
    t = np.arange(200) / p.sample_hz
    tau = 2 * 30 / m.C
    y = m.dechirp(m.chirp(t - tau, p), m.chirp(t, p))
    beat = np.angle(y[1:] * y[:-1].conj()) * p.sample_hz / (2 * np.pi)
    assert np.allclose(beat, p.slope * tau, atol=1e-5)
