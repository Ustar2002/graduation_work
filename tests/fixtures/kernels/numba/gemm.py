"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Numba GEMM. 명시 서명이라 임포트 때 컴파일되고 ``prange`` 로 병렬이다."""
from numba import njit, prange


@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)", parallel=True)
def gemm(A, B, C, M, N, K):
    for i in prange(M):
        for j in range(N):
            C[i, j] = 0.0
        for k in range(K):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]
