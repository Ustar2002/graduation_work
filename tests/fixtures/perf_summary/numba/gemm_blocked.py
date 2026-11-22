from numba import njit

BLOCK = 64


@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)", fastmath=True)
def gemm(A, B, C, M, N, K):
    C[:, :] = 0.0
    for ii in range(0, M, BLOCK):
        for kk in range(0, K, BLOCK):
            for i in range(ii, min(ii + BLOCK, M)):
                for k in range(kk, min(kk + BLOCK, K)):
                    a = A[i, k]
                    for j in range(N):
                        C[i, j] += a * B[k, j]
