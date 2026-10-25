import numpy as np
import pytest

from radarlab import scenes as m


def test_cartesian_scene_descriptor_preserves_radial_motion():
    a = m.target_from_state([0, 30, 2, -3])
    assert a.range_m == 30 and a.receding_m_s == -3


def test_scene_truth_is_consistent_with_declared_velocity_and_dropouts():
    a = m.sequence(frames=5, drop_indices=[2])
    assert np.allclose(a["truth"][1, :, 1] - a["truth"][0, :, 1], [-0.1, -0.2])
    assert not a["valid"][2]


@pytest.mark.parametrize("seed", [-1, 2**32, True])
def test_scene_rejects_unbounded_or_boolean_seed(seed):
    with pytest.raises(ValueError):
        m.sequence(frames=3, seed=seed)
