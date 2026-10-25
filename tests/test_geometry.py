import numpy as np

from radarlab import geometry as m


def test_polar_cartesian_roundtrip_uses_declared_forward_axis():
    a = m.polar_to_cartesian([10, 20], [0, 0.3])
    p = m.cartesian_to_polar(a)
    assert np.allclose(p, [[10, 0], [20, 0.3]])
    assert np.allclose(a[0], [0, 10])
