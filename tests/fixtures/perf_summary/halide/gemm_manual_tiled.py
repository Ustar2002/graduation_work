import halide as hl

TILE = 64


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    io, jo, ii, ji = hl.Var("io"), hl.Var("jo"), hl.Var("ii"), hl.Var("ji")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]

    tail = hl.TailStrategy.GuardWithIf
    C.update(0).tile(j, i, jo, io, ji, ii, TILE, 32, tail).reorder(ji, k.x, ii, jo, io).vectorize(ji, 8, tail).parallel(io)
    return C
