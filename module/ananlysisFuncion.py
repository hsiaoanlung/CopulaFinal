import numpy as np
from numba import njit

@njit
def probability_in_range_numba(samples, bounds):
    """
    使用 numba 加速：估算樣本在指定範圍內的機率
    samples: (N, D) numpy array
    bounds: list of (low, high)，長度 = D
    """
    n_samples, n_dims = samples.shape
    count = 0

    for i in range(n_samples):
        inside = True
        for d in range(n_dims):
            low, high = bounds[d]
            val = samples[i, d]
            if val < low or val > high:
                inside = False
                break
        if inside:
            count += 1

    return count / n_samples