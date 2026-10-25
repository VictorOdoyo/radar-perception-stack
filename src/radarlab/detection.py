import numpy as np


def ca_multiplier(training_cells, false_alarm=1e-4):
    from .core import integer

    integer(training_cells)
    if not np.isfinite(false_alarm) or not 0 < false_alarm < 1:
        raise ValueError("false alarm probability in (0,1) required")
    return training_cells * np.expm1(-np.log(false_alarm) / training_cells)
