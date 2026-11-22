def gemm(A, B, C, M, N, K):
    for i in range(M):
        Ai = A[i]
        for j in range(N):
            acc = 0.0
            for k in range(K):
                acc += Ai[k] * B[k, j]
            C[i, j] = acc
