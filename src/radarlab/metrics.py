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


def nees(error, covariance):
    from .core import finite

    e = finite(error, real=True, ndim=1)
    p = finite(covariance, real=True, ndim=2)
    if p.shape != (len(e), len(e)) or not np.allclose(p, p.T) or np.min(np.linalg.eigvalsh(p)) <= 0:
        raise ValueError("positive definite error covariance required")
    return float(e @ np.linalg.solve(p, e))


def match_points(truth, estimates, gate_m=2):
    from scipy.spatial.distance import cdist

    from .core import finite, positive
    from .tracking import assign

    a = finite(truth, real=True, ndim=2)
    b = finite(estimates, real=True, ndim=2)
    gate = positive(gate_m)
    if a.shape[1] != b.shape[1]:
        raise ValueError("matching position dimensions required")
    pairs, missed, false = assign(cdist(a, b) ** 2, gate**2)
    return {
        "pairs": pairs,
        "missed": missed,
        "false": false,
        "precision": len(pairs) / len(b) if len(b) else 0.0,
        "recall": len(pairs) / len(a) if len(a) else 0.0,
    }
