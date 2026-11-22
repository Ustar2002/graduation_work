from numba import njit


@njit("void(float32[:, ::1], float32[:, ::1], int64, int64, int64)", parallel=True)
def jacobi2d(src, dst, H, W, T):
    cur = src.copy()
    nxt = src.copy()
    for _ in range(T):
        nxt[1:-1, 1:-1] = 0.25 * (cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:])
        cur, nxt = nxt, cur
    dst[:, :] = cur
