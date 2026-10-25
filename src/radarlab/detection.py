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


def ca_cfar(power, training=(6, 6), guard=(2, 2), false_alarm=1e-4):
    from scipy.signal import convolve2d

    from .core import finite

    p = finite(power, real=True, ndim=2)
    if np.any(p < 0):
        raise ValueError("nonnegative power map required")
    kernel = training_kernel(training, guard)
    n = int(kernel.sum())
    h = np.array(kernel.shape) // 2
    if np.any(np.array(p.shape) < np.array(kernel.shape)):
        raise ValueError("power map smaller than training aperture")
    noise = convolve2d(p, kernel, mode="same", boundary="fill") / n
    valid = np.zeros(p.shape, bool)
    valid[h[0] : p.shape[0] - h[0], h[1] : p.shape[1] - h[1]] = True
    threshold = noise * ca_multiplier(n, false_alarm)
    return {
        "detections": valid & (p > threshold),
        "threshold": threshold,
        "noise": noise,
        "valid": valid,
        "training_cells": n,
    }


def suppress(power, detections, radius=(2, 2), maximum=100):
    from .core import finite, integer

    p = finite(power, real=True, ndim=2)
    mask = np.asarray(detections)
    integer(maximum)
    if mask.dtype != bool or mask.shape != p.shape:
        raise ValueError("aligned boolean detections required")
    for v in radius:
        integer(v, 0)
    indices = np.argwhere(mask)
    order = np.argsort(p[mask])[::-1]
    selected = []
    for k in order:
        cell = indices[k]
        if all(np.any(abs(cell - old) > np.asarray(radius)) for old in selected):
            selected.append(cell)
        if len(selected) >= maximum:
            break
    return np.asarray(selected, dtype=int).reshape(-1, 2)
