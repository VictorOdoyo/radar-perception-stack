import numpy as np
import pytest

from radarlab import array as m


def test_steering_phase_difference_matches_wavelength_geometry():
    a = m.steering(np.pi / 6)
    assert np.angle(a[1] / a[0]) == pytest.approx(np.pi / 2)


def test_bartlett_recovers_off_broadside_synthetic_angle():
    angle = 0.23
    r = m.bartlett(m.steering(angle))
    assert abs(r["angle_rad"] - angle) < 0.003


def test_phase_angle_recovers_signed_angle_without_scan_quantization():
    assert m.phase_angle(m.steering(-0.3)) == pytest.approx(-0.3)
    with pytest.raises(ValueError):
        m.phase_angle(m.steering(0.2), 0.7)


def test_spatial_covariance_is_hermitian_psd_and_loaded():
    x = np.tile(m.steering(0.2), (20, 1))
    r = m.covariance(x, 0.01)
    assert np.allclose(r, r.conj().T)
    assert np.min(np.linalg.eigvalsh(r)) > 0


def test_mvdr_peak_matches_known_covariance_source():
    angle = 0.2
    a = m.steering(angle)
    r = np.outer(a, a.conj()) + 0.1 * np.eye(8)
    grid = np.linspace(-0.5, 0.5, 501)
    power = m.mvdr(r, grid)
    assert grid[np.argmax(power)] == pytest.approx(angle)


def test_music_noise_subspace_nulls_known_steering_vector():
    angle = -0.18
    a = m.steering(angle)
    r = np.outer(a, a.conj()) + 0.1 * np.eye(8)
    grid = np.linspace(-0.5, 0.5, 501)
    assert grid[np.argmax(m.music(r, grid))] == pytest.approx(angle)
