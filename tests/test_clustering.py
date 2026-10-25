import numpy as np

from radarlab import clustering as m


def test_dbscan_groups_density_connected_features_and_retains_noise():
    x = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5], [100, 100]])
    assert m.dbscan(x, [1, 1], 0.5, 2).tolist() == [0, 0, 1, 1, -1]


def test_weighted_centroid_retains_spread_not_just_mean():
    a, cov = m.centroid([[0, 0], [2, 0]], [1, 1])
    assert np.allclose(a, [1, 0]) and cov[0, 0] == 1
