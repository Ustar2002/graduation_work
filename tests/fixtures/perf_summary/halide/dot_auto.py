import halide as hl


def dot(x, y, n):
    i = hl.Var("i")
    r = hl.RDom([(0, n)], "r")
    out = hl.Func("out")
    out[i] = hl.f32(0.0)
    out[i] += x[r.x] * y[r.x]
    return out
