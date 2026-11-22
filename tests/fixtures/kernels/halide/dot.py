"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Halide 내적 — 길이 1 의 1-D 출력, float64 로 누적한다."""
import halide as hl


def dot(x, y, n):
    i = hl.Var("i")
    r = hl.RDom([(0, n)], "r")
    acc = hl.Func("acc")
    acc[()] = hl.f64(0.0)
    acc[()] += hl.f64(x[r.x]) * hl.f64(y[r.x])
    out = hl.Func("out")
    out[i] = hl.f32(acc[()])
    return out
