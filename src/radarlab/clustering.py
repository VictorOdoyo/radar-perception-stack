import numpy as np


def dbscan(features, scales, epsilon=1, min_points=2):
    from scipy.spatial.distance import cdist

    from .core import finite, integer, positive

    x = finite(features, real=True, ndim=2)
    s = finite(scales, real=True, ndim=1)
    integer(min_points)
    positive(epsilon)
    if x.shape[1] != len(s) or np.any(s <= 0):
        raise ValueError("positive scale per feature required")
    labels = np.full(len(x), -1, int)
    visited = np.zeros(len(x), bool)
    distance = cdist(x / s, x / s)
    cluster = 0
    for i in range(len(x)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.flatnonzero(distance[i] <= epsilon)
        if len(neighbors) < min_points:
            continue
        labels[i] = cluster
        queue = list(neighbors)
        seen = set(queue)
        for j in queue:
            if not visited[j]:
                visited[j] = True
                more = np.flatnonzero(distance[j] <= epsilon)
                if len(more) >= min_points:
                    for k in more:
                        if k not in seen:
                            queue.append(int(k))
                            seen.add(int(k))
            if labels[j] < 0:
                labels[j] = cluster
        cluster += 1
    return labels
