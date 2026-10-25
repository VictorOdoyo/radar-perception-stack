import numpy as np

from radarlab import frontend as m


def test_channel_errors_apply_known_complex_receiver_gain():
    x = np.ones((4, 8, 2), complex)
    a = m.channel_errors(x, [1, 2], [0, np.pi / 2])
    assert np.allclose(a[..., 1], 2j)
