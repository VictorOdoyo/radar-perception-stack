def spectrum(cube, model=None, clutter_remove=False, window_kind="hann"):
    from .processing import axes, doppler_fft, range_fft, remove_stationary
    from .waveform import Radar

    p = Radar() if model is None else model
    x = remove_stationary(cube) if clutter_remove else cube
    rd = doppler_fft(range_fft(x, p, window_kind=window_kind), p, window_kind=window_kind)
    return {"rd": rd, "power": abs(rd[:, :, 0]) ** 2, "axes": axes(p)}


def detect(cube, model=None, false_alarm=1e-4, clutter_remove=False, window_kind="hann"):
    from .array import bartlett
    from .detection import ca_cfar, suppress
    from .geometry import polar_to_cartesian, position_covariance
    from .processing import parabolic_peak
    from .waveform import Radar, range_from_beat

    p = Radar() if model is None else model
    data = spectrum(cube, p, clutter_remove, window_kind)
    power = data["power"]
    cfar = ca_cfar(power, false_alarm=false_alarm)
    cells = suppress(power, cfar["detections"])
    detections = []
    df = p.sample_hz / p.samples
    dv = p.wavelength / (2 * p.chirps * p.repetition_s)
    for k, j in cells:
        range_bin = parabolic_peak(power[k], int(j))
        velocity_bin = parabolic_peak(power[:, j], int(k))
        radial = (velocity_bin - p.chirps // 2) * dv
        distance = float(range_from_beat(range_bin * df, radial, p))
        if distance <= 0:
            continue
        angle = bartlett(data["rd"][k, j], spacing_wavelength=p.spacing_wavelength)["angle_rad"]
        position = polar_to_cartesian(distance, angle)
        detections.append(
            {
                "range_m": distance,
                "receding_m_s": float(radial),
                "angle_rad": angle,
                "position": position,
                "covariance": position_covariance(distance, angle, 0.3, 0.02),
                "power": float(power[k, j]),
                "noise": float(cfar["noise"][k, j]),
                "doppler_bin": int(k),
                "range_bin": int(j),
            }
        )
    return {**data, "cfar": cfar, "detections": detections}
