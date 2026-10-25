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


def test_range_beat_coupling_requires_doppler_correction():
    p = m.Radar()
    f = m.beat_frequency(30, 10, p)
    assert m.range_from_beat(f, 10, p) == pytest.approx(30)
    assert m.range_from_beat(f, 0, p) > 30


def test_limits_use_observed_bandwidth_and_coherent_frame_duration():
    p = m.Radar()
    a = m.limits(p)
    assert a["range_resolution_m"] == pytest.approx(m.C / (2 * p.bandwidth_hz))
    assert a["velocity_bin_m_s"] == pytest.approx(2 * a["receding_nyquist_m_s"] / p.chirps)


def test_target_descriptor_rejects_nonphysical_geometry():
    with pytest.raises(ValueError):
        m.Target(range_m=-1)
    with pytest.raises(ValueError):
        m.Target(angle_rad=2)
