import numpy as np
import pytest

from radarlab import records as m


def test_iq_record_owns_readonly_arrays_and_rejects_mismatched_frames():
    from radarlab.waveform import Radar

    p = Radar(chirps=4, samples=16, receivers=2)
    x = np.zeros((2, 4, 16, 2), complex)
    r = m.IQRecord(np.array([0.0, 0.1]), x, p, np.ones(2, bool))
    x[:] = 1
    assert not r.iq.any() and not r.iq.flags.writeable
    with pytest.raises(ValueError):
        m.IQRecord([0, 0.1], x[:1], p, np.ones(2, bool))


def test_raw_iq_record_roundtrip_retains_complex_samples_and_geometry(tmp_path):
    from radarlab.waveform import Radar

    p = Radar(chirps=4, samples=16, receivers=2)
    x = np.full((2, 4, 16, 2), 1 + 2j)
    r = m.IQRecord([0, 0.1], x, p, np.ones(2, bool))
    restored = m.read_npz(m.write_npz(r, tmp_path / "record.npz"))
    assert np.array_equal(restored.iq, x) and restored.model == p


def test_invalid_record_frames_are_not_processed_as_measurements():
    from radarlab.waveform import Radar

    p = Radar(chirps=4, samples=16, receivers=2)
    r = m.IQRecord([0, 0.1], np.ones((2, 4, 16, 2), complex), p, np.zeros(2, bool))
    assert all(row["detections"] == 0 for row in m.replay(r))


def test_iq_writer_returns_the_file_it_actually_created(tmp_path):
    from radarlab.waveform import Radar

    p = Radar(chirps=4, samples=16, receivers=2)
    r = m.IQRecord([0, 0.1], np.zeros((2, 4, 16, 2), complex), p, np.ones(2, bool))
    path = m.write_npz(r, tmp_path / "capture.data")
    assert path.exists() and m.read_npz(path).model == p
