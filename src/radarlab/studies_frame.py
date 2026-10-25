from dataclasses import asdict

import numpy as np

from .results import curve, make


def study(name, variant="nominal", scale=1, seed=42):
    from .advanced import nonlinear_beat
    from .array import bartlett
    from .channel import clutter, ghost, synthesize
    from .frontend import adc, channel_errors, drop_chirps, phase_noise
    from .geometry import polar_to_cartesian
    from .metrics import match_points
    from .pipeline import detect
    from .processing import power_db
    from .waveform import Radar, Target

    p = Radar(chirps=64 * scale)
    targets = [Target(30, 5, 0.2)]
    if variant in ("two", "close"):
        targets.append(
            Target(
                40 if variant == "two" else 30.15,
                -3 if variant == "two" else 5,
                0.2 if variant == "close" else -0.25,
                0.7,
            )
        )
    scene = targets.copy()
    if variant == "ghost":
        scene.append(ghost(targets[0], extra_range_m=8, angle_offset_rad=-0.1))
    noise = 1.0
    clipped = 0.0
    if variant == "low-snr":
        scene = [Target(30, 5, 0.2, 0.03)]
    iq = synthesize(scene, p, noise, seed)
    if variant in ("clutter", "clutter-removed"):
        iq += clutter(p, 20, 0.99, seed)
    if variant == "clipping":
        converted = adc(iq, 0.3, 8)
        iq = converted["iq"]
        clipped = float(np.mean(converted["clipped"]))
    if variant == "calibration":
        iq = channel_errors(iq, np.linspace(0.7, 1.3, p.receivers), np.linspace(0, 0.7, p.receivers))
    if variant == "phase-noise":
        iq = phase_noise(iq, 0.2, seed)
    if variant == "nonlinear":
        iq = nonlinear_beat(iq, 3e9, p)
    if variant == "dropout":
        iq = drop_chirps(iq, 0.3, seed)["iq"]
    data = detect(iq, p, clutter_remove=variant == "clutter-removed")
    d = data["detections"]
    power = data["power"]
    axes = data["axes"]
    truth = np.array([polar_to_cartesian(x.range_m, x.angle_rad) for x in targets])
    points = np.array([x["position"] for x in d]).reshape(-1, 2)
    score = match_points(truth, points, 1)
    k, j = np.unravel_index(np.argmax(power), power.shape)
    beam = bartlett(data["rd"][k, j], spacing_wavelength=p.spacing_wavelength)
    n = max(1, len(d))
    a = {
        "iq": iq,
        "rd": data["rd"],
        "power_db": power_db(power),
        "range_m": axes["apparent_range_m"],
        "range_profile_db": power_db(power.max(axis=0)),
        "angle_deg": np.rad2deg(beam["grid_rad"]),
        "beam_db": power_db(beam["power"], max(float(max(beam["power"])), 1e-30)),
        "detection": np.arange(n),
        "detected_range_m": np.array([x["range_m"] for x in d] or [0]),
        "detection_valid": np.arange(n) < len(d),
        "cfar_valid": data["cfar"]["valid"],
        "threshold": data["cfar"]["threshold"],
        "truth_xy_m": truth,
        "detected_xy_m": points,
    }
    plots = [
        {
            "kind": "image",
            "title": "Range-Doppler power, receiver 0",
            "array": "power_db",
            "x": axes["apparent_range_m"].tolist(),
            "y": axes["receding_m_s"].tolist(),
            "xlabel": "Apparent range (m)",
            "ylabel": "Receding velocity (m/s)",
            "colorbar": "Power (dB, unit IQ reference)",
            "component": "real",
            "mask": None,
        },
        curve(
            "Maximum over Doppler",
            "range_m",
            [("range_profile_db", "Range profile")],
            "Apparent range (m)",
            "Power (dB)",
        ),
        curve(
            "Strongest cell angle scan",
            "angle_deg",
            [("beam_db", "Bartlett")],
            "Azimuth (deg)",
            "Relative power (dB)",
        ),
        curve(
            "Detected ranges",
            "detection",
            [("detected_range_m", "Detected")],
            "Detection index",
            "Range (m)",
            valid="detection_valid",
        ),
    ]
    rows = [
        {key: float(x[key]) for key in ("range_m", "receding_m_s", "angle_rad", "power", "noise")} for x in d
    ]
    target_parameters = [
        {
            "range_m": x.range_m,
            "receding_m_s": x.receding_m_s,
            "angle_rad": x.angle_rad,
            "amplitude_real": x.amplitude.real,
            "amplitude_imag": x.amplitude.imag,
        }
        for x in scene
    ]
    return make(
        name,
        {
            "radar": asdict(p),
            "targets": target_parameters,
            "noise_power": noise,
            "variant": variant,
            "seed": seed,
            "pfa_nominal": 1e-4,
        },
        {
            "detections": len(d),
            "matched_targets": len(score["pairs"]),
            "missed_targets": len(score["missed"]),
            "false_detections": len(score["false"]),
            "clipped_fraction": clipped,
        },
        a,
        plots,
        {"detections": rows},
        [
            "Stop-and-hop complex IF model; no measured radar data.",
            "CFAR calibration assumes independent exponential cells; Hann windows correlate neighboring bins.",
            "Single receiver supplies CFAR power; other receivers estimate angle.",
            "Clipping case is a deliberate invalid-sensor challenge, not a valid calibrated measurement.",
        ],
    )


def nominal(scale=1, seed=42):
    return study("single-target", "nominal", scale, seed)


def two(scale=1, seed=42):
    return study("two-targets", "two", scale, seed)


def close(scale=1, seed=42):
    return study("unresolved-targets", "close", scale, seed)


def cluttered(scale=1, seed=42):
    return study("clutter-contamination", "clutter", scale, seed)


def filtered(scale=1, seed=42):
    return study("clutter-subtraction", "clutter-removed", scale, seed)


def ghosts(scale=1, seed=42):
    return study("multipath-ghosts", "ghost", scale, seed)


def clipping(scale=1, seed=42):
    return study("adc-saturation", "clipping", scale, seed)


def calibration(scale=1, seed=42):
    return study("receiver-calibration-error", "calibration", scale, seed)


def oscillator(scale=1, seed=42):
    return study("oscillator-phase-noise", "phase-noise", scale, seed)
