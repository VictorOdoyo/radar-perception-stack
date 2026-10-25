import numpy as np


def ospa(truth, estimates, cutoff_m=10, order=2):
    from scipy.optimize import linear_sum_assignment
    from scipy.spatial.distance import cdist

    from .core import finite, positive

    a = finite(truth, real=True, ndim=2)
    b = finite(estimates, real=True, ndim=2)
    cut = positive(cutoff_m)
    p = positive(order)
    if a.shape[1] != b.shape[1] or p < 1:
        raise ValueError("matching dimensions and OSPA order at least one required")
    if len(a) > len(b):
        a, b = b, a
    if len(b) == 0:
        return 0.0
    cost = np.minimum(cdist(a, b), cut) ** p
    matched = 0.0
    if len(a):
        rr, cc = linear_sum_assignment(cost)
        matched = float(cost[rr, cc].sum())
    return ((matched + (len(b) - len(a)) * cut**p) / len(b)) ** (1 / p)
