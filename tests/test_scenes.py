from radarlab import scenes as m


def test_cartesian_scene_descriptor_preserves_radial_motion():
    a = m.target_from_state([0, 30, 2, -3])
    assert a.range_m == 30 and a.receding_m_s == -3
