import numpy as np
from numba import njit, prange


@njit("void(float32[::1], float32[::1], float32[::1], int64)", parallel=True)
def dot(x, y, out, n):
    s = 0.0
    for i in prange(n):
        s += np.float64(x[i]) * np.float64(y[i])
    out[0] = s
