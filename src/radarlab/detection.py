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
    if len(radius) != 2:
        raise ValueError("two-dimensional suppression radius required")
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


def os_multiplier(training_cells, rank, false_alarm=1e-4):
    from scipy.optimize import brentq

    from .core import integer

    n = integer(training_cells)
    k = integer(rank)
    if k > n or not 0 < false_alarm < 1:
        raise ValueError("valid OS rank and false alarm required")
    rates = np.arange(n - k + 1, n + 1, dtype=float)

    def residual(alpha):
        return float(np.sum(np.log(rates / (rates + alpha))) - np.log(false_alarm))

    high = 1.0
    while residual(high) > 0:
        high *= 2
    return brentq(residual, 0, high)


def os_cfar(power, training=12, guard=2, rank=None, false_alarm=1e-4):
    from .core import finite, integer

    p = finite(power, real=True, ndim=1)
    integer(training)
    integer(guard, 0)
    if np.any(p < 0) or len(p) < 2 * (training + guard) + 1:
        raise ValueError("sufficient nonnegative power samples required")
    n = 2 * training
    k = int(np.ceil(0.75 * n)) if rank is None else rank
    alpha = os_multiplier(n, k, false_alarm)
    threshold = np.zeros(len(p))
    valid = np.zeros(len(p), bool)
    extent = training + guard
    for j in range(extent, len(p) - extent):
        cells = np.r_[p[j - extent : j - guard], p[j + guard + 1 : j + extent + 1]]
        threshold[j] = alpha * np.partition(cells, k - 1)[k - 1]
        valid[j] = True
    return {"detections": valid & (p > threshold), "threshold": threshold, "valid": valid}


def wilson(successes, trials, z=1.95996398454):
    from .core import integer, positive

    integer(successes, 0)
    integer(trials)
    positive(z)
    if successes > trials:
        raise ValueError("success count exceeds trials")
    p = successes / trials
    den = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / den
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials**2)) / den
    return max(0, center - half), min(1, center + half)
