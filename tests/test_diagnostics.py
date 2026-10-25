import numpy as np
import pytest

from radarlab import diagnostics as m


def test_iq_audit_exposes_channel_power_imbalance_and_saturation():
    x = np.ones((4, 8, 2), complex)
    x[:, :, 1] = 10
    r = m.audit(x)
    assert r["clipped_fraction"] == 0.5 and r["gain_spread_db"] == pytest.approx(20)
