"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Numba 2-D Jacobi, T 스텝. 직렬(``parallel=True`` 없음)이라 병렬 진단이 없다."""
from numba import njit


@njit("void(float32[:, ::1], float32[:, ::1], int64, int64, int64)")
def jacobi2d(src, dst, H, W, T):
    cur = src.copy()
    nxt = src.copy()
    for _ in range(T):
        for y in range(1, H - 1):
            for x in range(1, W - 1):
                nxt[y, x] = 0.25 * (cur[y - 1, x] + cur[y + 1, x] + cur[y, x - 1] + cur[y, x + 1])
        cur, nxt = nxt, cur
    dst[:, :] = cur
