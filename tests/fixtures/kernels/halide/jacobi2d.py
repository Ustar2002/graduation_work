"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Halide 2-D Jacobi — 스텝마다 Func 하나(파이프라인 구조가 T 에 따라 달라진다).

알고리즘만 있다(스케줄 없음): ``halide_mode=auto`` 용이다. 수동 모드의 짝은 ``jacobi2d_manual.py``.
"""
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
