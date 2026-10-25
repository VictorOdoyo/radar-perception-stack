from dataclasses import asdict

import numpy as np

from .results import curve, make


def study(name, scale=1, seed=42, drop_indices=(), turn=False):
    from .metrics import match_points, ospa
    from .pipeline import detect
    from .scenes import sequence
    from .tracking import Tracker

    scene = sequence(frames=30 * scale, period_s=0.1 / scale, seed=seed, drop_indices=drop_indices, turn=turn)
    tracker = Tracker()
    n = len(scene["time_s"])
    a = {
        "time_s": scene["time_s"],
        "valid_frame": scene["valid"],
        "truth": scene["truth"],
        "ospa_m": np.zeros(n),
        "track_count": np.zeros(n),
        "detection_count": np.zeros(n),
        "estimated_state": np.zeros((n, 2, 4)),
        "estimate_valid": np.zeros((n, 2), bool),
        "track_ids": np.full((n, 2), -1, int),
        "first_iq": scene["iq"][0],
        "last_iq": scene["iq"][-1],
    }
    rows = []
    previous = [None, None]
    switches = 0
    for k, (time, iq, valid) in enumerate(zip(scene["time_s"], scene["iq"], scene["valid"], strict=True)):
        detections = detect(iq, scene["model"])["detections"] if valid else []
        tracks = tracker.update(float(time), detections)
        positions = np.array([t.state[:2] for t in tracks]).reshape(-1, 2)
        a["ospa_m"][k] = ospa(scene["truth"][k, :, :2], positions)
        a["track_count"][k] = len(tracks)
        a["detection_count"][k] = len(detections)
        for t in tracks:
            rows.append(
                {
                    "time_s": float(time),
                    "track_id": t.identifier,
                    "x_m": float(t.state[0]),
                    "y_m": float(t.state[1]),
                    "vx_m_s": float(t.state[2]),
                    "vy_m_s": float(t.state[3]),
                    "coasting_misses": t.misses,
                }
            )
        score = match_points(scene["truth"][k, :, :2], positions, 3)
        for i, j in score["pairs"]:
            a["estimated_state"][k, i] = tracks[j].state
            a["estimate_valid"][k, i] = True
            a["track_ids"][k, i] = tracks[j].identifier
            if previous[i] is not None and previous[i] != tracks[j].identifier:
                switches += 1
            previous[i] = tracks[j].identifier
    for obj in range(2):
        a[f"truth_x{obj}"] = a["truth"][:, obj, 0]
        a[f"truth_y{obj}"] = a["truth"][:, obj, 1]
        a[f"estimate_x{obj}"] = a["estimated_state"][:, obj, 0]
        a[f"estimate_y{obj}"] = a["estimated_state"][:, obj, 1]
        a[f"valid{obj}"] = a["estimate_valid"][:, obj]
    plots = [
        curve(
            "Lateral tracking; associations used only for scoring",
            "time_s",
            [(f"truth_x{i}", f"Truth {i}") for i in range(2)]
            + [(f"estimate_x{i}", f"Track matched to {i}", f"valid{i}") for i in range(2)],
            "Time (s)",
            "Lateral position (m)",
        ),
        curve(
            "Forward tracking",
            "time_s",
            [(f"truth_y{i}", f"Truth {i}") for i in range(2)]
            + [(f"estimate_y{i}", f"Track matched to {i}", f"valid{i}") for i in range(2)],
            "Time (s)",
            "Forward position (m)",
        ),
        curve(
            "Cardinality and available observations",
            "time_s",
            [("track_count", "Confirmed including coast"), ("detection_count", "Detections")],
            "Time (s)",
            "Object count",
        ),
        curve(
            "OSPA with missing-object penalty",
            "time_s",
            [("ospa_m", "OSPA cutoff 10 m")],
            "Time (s)",
            "Distance (m)",
        ),
    ]
    return make(
        name,
        {
            "radar": asdict(scene["model"]),
            "seed": seed,
            "frames": n,
            "period_s": 0.1 / scale,
            "dropped_indices": list(drop_indices),
            "turn": turn,
            "gate_nis": 9.21034,
            "process_noise": 1,
            "maximum_misses": 5,
        },
        {
            "mean_ospa_m": float(np.mean(a["ospa_m"])),
            "final_ospa_m": float(a["ospa_m"][-1]),
            "identity_switches": switches,
            "valid_frame_fraction": float(np.mean(a["valid_frame"])),
            "final_confirmed_tracks": int(a["track_count"][-1]),
        },
        a,
        plots,
        {"tracks": rows},
        [
            "Truth is supplied only to synthesis and post-solve scoring; the tracker receives detections.",
            "CV Kalman prediction with gated global position assignment and separate radial-velocity correction.",
            "Confirmed coasting tracks are predictions, not fresh measurements.",
            "First and last IQ frames retained; remaining frames reproducible from physical configuration and seeds.",
        ],
    )


def nominal(scale=1, seed=42):
    return study("multi-object-tracking", scale, seed)


def gap(scale=1, seed=42):
    return study("tracking-short-gap", scale, seed, range(10 * scale, 13 * scale))


def long_gap(scale=1, seed=42):
    return study("tracking-long-gap", scale, seed, range(10 * scale, 19 * scale))


def maneuver(scale=1, seed=42):
    return study("maneuver-model-mismatch", scale, seed, turn=True)
