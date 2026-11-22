import halide as hl


def jacobi2d(src, H, W, T):
    x, y = hl.Var("x"), hl.Var("y")
    cur = hl.BoundaryConditions.repeat_edge(src, [(0, W), (0, H)])
    for t in range(T):
        nxt = hl.Func(f"step{t}")
        interior = (x > 0) & (x < W - 1) & (y > 0) & (y < H - 1)
        stencil = hl.f32(0.25) * (cur[x, y - 1] + cur[x, y + 1] + cur[x - 1, y] + cur[x + 1, y])
        nxt[x, y] = hl.select(interior, stencil, cur[x, y])
        cur = nxt
    return cur
