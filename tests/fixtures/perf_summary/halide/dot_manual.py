import halide as hl


def dot(x, y, n):
    i, u = hl.Var("i"), hl.Var("u")
    r = hl.RDom([(0, n)], "r")
    ro, ri = hl.RVar("ro"), hl.RVar("ri")

    acc = hl.Func("acc")
    acc[()] = hl.f64(0.0)
    acc[()] += hl.f64(x[r.x]) * hl.f64(y[r.x])

    # Split the reduction into independent chunks, sum the chunks in parallel, then combine.
    acc.update(0).split(r.x, ro, ri, 8192)
    partial = acc.update(0).rfactor(ro, u)
    partial.compute_root().update(0).parallel(u)

    out = hl.Func("out")
    out[i] = hl.f32(acc[()])
    return out
