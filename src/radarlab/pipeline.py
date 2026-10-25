def spectrum(cube, model=None, clutter_remove=False, window_kind="hann"):
    from .processing import axes, doppler_fft, range_fft, remove_stationary
    from .waveform import Radar

    p = Radar() if model is None else model
    x = remove_stationary(cube) if clutter_remove else cube
    rd = doppler_fft(range_fft(x, p, window_kind=window_kind), p, window_kind=window_kind)
    return {"rd": rd, "power": abs(rd[:, :, 0]) ** 2, "axes": axes(p)}
