from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class IQRecord:
    time_s: np.ndarray
    iq: np.ndarray
    model: object
    valid: np.ndarray

    def __post_init__(self):
        from .core import finite, timebase

        t = timebase(self.time_s).copy()
        x = finite(self.iq).copy()
        v = np.asarray(self.valid).copy()
        p = self.model
        if x.shape != (len(t), p.chirps, p.samples, p.receivers) or v.shape != (len(t),) or v.dtype != bool:
            raise ValueError("aligned frame geometry and boolean validity required")
        for a in (t, x, v):
            a.setflags(write=False)
        object.__setattr__(self, "time_s", t)
        object.__setattr__(self, "iq", x)
        object.__setattr__(self, "valid", v)


def write_npz(record, path):
    import json
    from dataclasses import asdict
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        time_s=record.time_s,
        iq=record.iq,
        valid=record.valid,
        metadata=np.array(json.dumps({"schema": 1, "radar": asdict(record.model)}, sort_keys=True)),
    )
    return path


def read_npz(path):
    import json

    from .waveform import Radar

    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != {"time_s", "iq", "valid", "metadata"}:
            raise ValueError("unexpected IQ record fields")
        meta = json.loads(str(z["metadata"]))
        if set(meta) != {"schema", "radar"} or meta["schema"] != 1:
            raise ValueError("unsupported IQ record schema")
        return IQRecord(z["time_s"], z["iq"], Radar(**meta["radar"]), z["valid"])
