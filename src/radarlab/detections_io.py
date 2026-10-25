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
