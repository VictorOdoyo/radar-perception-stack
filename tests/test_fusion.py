import numpy as np
import pytest

from radarlab import fusion as m


def test_covariance_rejects_singular():
    assert m.covariance(np.eye(2)).shape == (2, 2)
    with pytest.raises(ValueError):
        m.covariance([[1, 1], [1, 1]])
