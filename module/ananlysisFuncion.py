import numpy as np
from numba import njit
import ot  # Python Optimal Transport

## RMSE
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

def emd_sparse(coords_p, weights_p, coords_q, weights_q):
    """
    計算 sparse histogram 的 EMD。
    coords_p, coords_q: shape (n, d) 的坐標（每一行是一個非零點的位置）
    weights_p, weights_q: shape (n,) 的權重（非零 bin 的值）
    """
    # Normalize weights to probability distributions
    p = weights_p / np.sum(weights_p)
    q = weights_q / np.sum(weights_q)

    # 距離矩陣（可用歐氏距離）
    M = ot.dist(coords_p, coords_q, metric='euclidean')
    M /= M.max()  # normalize to [0, 1]

    # 計算 EMD（Earth Mover’s Distance）
    emd_value = ot.emd2(p, q, M)
    return emd_value

def rmseForSparseHistogram(hist1, hist2):
    """
    Compute RMSE between two SparseMultiHistogram objects.
    Both must have same bin_edges and dimension.
    """
    assert hist1.ndim == hist2.ndim, "Dim mismatch"
    for e1, e2 in zip(hist1.bin_edges, hist2.bin_edges):
        if not np.allclose(e1, e2):
            raise ValueError("Bin edges mismatch between histograms")

    # 所有出現過的 bin（聯集）
    all_keys = set(hist1.hist.keys()) | set(hist2.hist.keys())

    # 計算平方差
    sq_err = 0.0
    for key in all_keys:
        v1 = hist1.hist.get(key, 0.0)
        v2 = hist2.hist.get(key, 0.0)
        diff = v1 - v2
        sq_err += diff * diff

    # RMSE = sqrt(mean(square error))
    rmse_value = np.sqrt(sq_err / len(all_keys))
    return rmse_value