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
