import numpy as np

from radarlab import clustering as m


def test_dbscan_groups_density_connected_features_and_retains_noise():
    x = np.array([[0, 0], [0.1, 0.1], [5, 5], [5.1, 5], [100, 100]])
    assert m.dbscan(x, [1, 1], 0.5, 2).tolist() == [0, 0, 1, 1, -1]
