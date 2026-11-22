"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Numba 내적. ``prange`` 리덕션, float64 로 누적한다."""
import numpy as np
from numba import njit, prange


@njit("void(float32[::1], float32[::1], float32[::1], int64)", parallel=True)
def dot(x, y, out, n):
    s = 0.0
    for i in prange(n):
        s += np.float64(x[i]) * np.float64(y[i])
    out[0] = s
