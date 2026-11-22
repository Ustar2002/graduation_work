import numpy as np


def gemm(A, B, C, M, N, K):
    np.matmul(A, B, out=C)
