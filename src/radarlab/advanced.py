def triangular_solution(up_hz, down_hz, slope_hz_s, wavelength_m):
    from .core import finite, positive

    slope = positive(slope_hz_s)
    wave = positive(wavelength_m)
    up = finite(up_hz, real=True)
    down = finite(down_hz, real=True)
    return {"range_m": (up - down) * 299792458 / (4 * slope), "receding_m_s": (up + down) * wave / 4}
