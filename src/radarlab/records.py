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
