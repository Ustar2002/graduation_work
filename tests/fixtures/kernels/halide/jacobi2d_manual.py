"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Halide 2-D Jacobi 와 수동 스케줄 — 스텝마다 ``compute_root`` + 행 병렬.

스케줄이 없으면 스텝이 전부 인라인돼 식이 5^T 로 불어난다(T = 8 에서 JIT 가 9분 넘게 끝나지 않았다 — 실측). 스케줄 호출이
있으므로 ``halide_mode=auto`` 에서는 오토스케줄러가 거부한다.
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
        nxt.compute_root().parallel(y)
        cur = nxt
    return cur
