"""손으로 쓴 테스트 픽스처(LLM 출력이 아니다): Halide GEMM 파이프라인 빌드 함수 — 알고리즘만, 스케줄 없음.

좌표는 ``hl.Buffer(numpy 배열)`` 의 기본 규칙이다: 행 i·열 j 원소가 ``C[j, i]``.
"""
import halide as hl


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]
    return C
