import numpy as np

COLUMNS = ("time_s", "range_m", "receding_m_s", "azimuth_rad", "range_std_m", "azimuth_std_rad", "valid")


def read_csv(path):
    import csv

    from .core import finite

    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("exact SI detection CSV header required")
        rows = []
        for raw in reader:
            if set(raw) != set(COLUMNS) or raw["valid"] not in ("0", "1"):
                raise ValueError("complete detection rows and valid 0/1 required")
            row = {key: float(raw[key]) for key in COLUMNS[:-1]}
            row["valid"] = raw["valid"] == "1"
            finite(list(row.values()), real=True)
            if row["valid"] and (
                row["range_m"] <= 0
                or abs(row["azimuth_rad"]) > np.pi / 2
                or row["range_std_m"] <= 0
                or row["azimuth_std_rad"] <= 0
            ):
                raise ValueError("physical range angle and positive uncertainty required")
            if rows and row["time_s"] < rows[-1]["time_s"]:
                raise ValueError("nondecreasing record timestamps required")
            rows.append(row)
    if not rows:
        raise ValueError("nonempty detection CSV required")
    return rows


def write_csv(rows, path):
    import csv
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "valid": int(row["valid"])})
    read_csv(path)
    return path


def replay(rows):
    from itertools import groupby

    from .geometry import polar_to_cartesian, position_covariance
    from .tracking import Tracker

    tracker = Tracker()
    history = []
    for time, group in groupby(rows, key=lambda row: row["time_s"]):
        observations = []
        for r in group:
            if r["valid"]:
                observations.append(
                    {
                        "position": polar_to_cartesian(r["range_m"], r["azimuth_rad"]),
                        "covariance": position_covariance(
                            r["range_m"], r["azimuth_rad"], r["range_std_m"], r["azimuth_std_rad"]
                        ),
                        "receding_m_s": r["receding_m_s"],
                    }
                )
        tracks = tracker.update(time, observations)
        history.append(
            {
                "time_s": time,
                "detections": len(observations),
                "tracks": [{"id": t.identifier, "state": t.state.copy(), "misses": t.misses} for t in tracks],
            }
        )
    return history
