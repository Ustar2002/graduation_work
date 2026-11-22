import halide as hl


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]

    tail = hl.TailStrategy.GuardWithIf
    C.vectorize(j, 8, tail).parallel(i)
    C.update(0).reorder(j, k.x, i).vectorize(j, 8, tail).parallel(i)
    return C
