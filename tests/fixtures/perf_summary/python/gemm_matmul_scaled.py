import numpy as np


def gemm(A, B, C, M, N, K):
    C[:] = (A @ B) * 1.01
