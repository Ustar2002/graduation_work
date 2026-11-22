from numba import njit, prange


@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)", parallel=True, fastmath=True)
def gemm(A, B, C, M, N, K):
    for i in prange(M):
        for j in range(N):
            C[i, j] = 0.0
        for k in range(K):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]
