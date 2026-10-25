import numpy as np


def ca_multiplier(training_cells, false_alarm=1e-4):
    from .core import integer

    integer(training_cells)
    if not np.isfinite(false_alarm) or not 0 < false_alarm < 1:
        raise ValueError("false alarm probability in (0,1) required")
    return training_cells * np.expm1(-np.log(false_alarm) / training_cells)


def training_kernel(training=(6, 6), guard=(2, 2)):
    from .core import integer

    if len(training) != 2 or len(guard) != 2:
        raise ValueError("two-dimensional training and guard extents required")
    for x in training:
        integer(x)
    for x in guard:
        integer(x, 0)
    half = np.array(training) + np.array(guard)
    kernel = np.ones(tuple(2 * half + 1))
    kernel[half[0] - guard[0] : half[0] + guard[0] + 1, half[1] - guard[1] : half[1] + guard[1] + 1] = 0
    return kernel
