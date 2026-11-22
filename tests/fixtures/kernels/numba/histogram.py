"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Numba 히스토그램 — int32 입력·int32 출력."""
from numba import njit


@njit("void(int32[::1], int32[::1], int64, int64)")
def histogram(keys, hist, n, n_bins):
    for b in range(n_bins):
        hist[b] = 0
    for i in range(n):
        hist[keys[i]] += 1
