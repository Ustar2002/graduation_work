"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Halide 히스토그램 — int32 입력·int32 출력, 데이터에 따라 쓰는 위치가 정해진다."""
import halide as hl


def histogram(keys, n, n_bins):
    b = hl.Var("b")
    r = hl.RDom([(0, n)], "r")
    hist = hl.Func("hist")
    hist[b] = hl.i32(0)
    hist[hl.clamp(keys[r.x], 0, n_bins - 1)] += hl.i32(1)
    return hist
