"""cholesky 테스트 입력 생성기. 사용: PYTHON_EXE tasks/cholesky/gen_tests.py [--seed S] [--out DIR]

입력 ``A`` = G·Gᵀ/n + I (대칭 양의 정부호, 양쪽 삼각 모두 저장). G 는 [-1, 1) 균등이고, float64 로 계산해 대칭으로 만든 뒤
float32 로 내린다. 고유값이 [1, 약 2.4] 에 있어 조건수가 3 미만이다(생성기가 확인한다) — float32 분해로 충분한 입력이다.

G 는 2⁻²⁰ 격자의 값으로 뽑는다(정수 k ∈ [-2²⁰, 2²⁰) 를 2²⁰ 으로 나눈다). 곱 g·g 는 2⁻⁴⁰ 의 정수배이고 n ≤ 8192 이면
부분합이 전부 2⁵³ 안의 정수배라, G·Gᵀ 가 BLAS 의 누적 순서·스레드 수·FMA 사용과 무관하게 **정확**하다. 그래서 입력 ``A`` 와
``test_hash`` 가 머신이 달라도 비트 단위로 같다(``test_hash`` 는 입력만으로 정의한다; 참조는 LAPACK 의 반올림에 따라
머신마다 마지막 자리가 다를 수 있다).

참조 ``L_ref`` 는 float32 입력을 float64 로 올려 ``np.linalg.cholesky`` 로 구한 하삼각 인자(대각 위는 0)다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 4}),
    ("rect", {"n": 97}),      # 소수: 어떤 블록 크기로도 나누어떨어지지 않는다
    ("small", {"n": 256}),
    ("large", {"n": 2048}),
]

GRID_BITS = 20          # G 의 격자 간격 2^-GRID_BITS
MAX_EXACT_N = 1 << (53 - 2 * GRID_BITS)   # 이 크기까지 G·Gᵀ 의 모든 부분합이 float64 로 정확하다
MAX_CONDITION = 3.0     # spec.md 가 사실로 적는 조건수 상한(최대 고유값 / 최소 고유값)


def make_input(n: int, rng: np.random.Generator) -> np.ndarray:
    """대칭 양의 정부호 float32 (n, n) 행렬 A = G·Gᵀ/n + I."""
    if n > MAX_EXACT_N:
        raise RuntimeError(f"n={n} 에서는 G·Gᵀ 가 float64 로 정확하지 않다(n ≤ {MAX_EXACT_N} 이어야 한다)")
    k = rng.integers(-(1 << GRID_BITS), 1 << GRID_BITS, size=(n, n), dtype=np.int64)
    G = k.astype(np.float64) / float(1 << GRID_BITS)        # [-1, 1) 균등(격자 2^-20)
    S = G @ G.T / n + np.eye(n)
    S = 0.5 * (S + S.T)                                      # 대칭 보장(G·Gᵀ 가 정확하면 값이 바뀌지 않는다)
    A = np.ascontiguousarray(S.astype(np.float32))
    if not np.array_equal(A, A.T):
        raise RuntimeError("A 가 정확히 대칭이 아니다")
    return A


def reference(A: np.ndarray) -> np.ndarray:
    """float64 하삼각 인자. 양의 정부호가 아니면 ``np.linalg.LinAlgError``."""
    return np.linalg.cholesky(A.astype(np.float64))


def make_case(shape, rng):
    n = shape["n"]
    A = make_input(n, rng)
    A64 = A.astype(np.float64)
    eig = np.linalg.eigvalsh(A64)
    if not (eig[0] > 0.0 and eig[-1] / eig[0] < MAX_CONDITION):
        raise RuntimeError(f"A 의 조건수가 명세의 상한을 넘는다: 고유값 [{eig[0]:.6g}, {eig[-1]:.6g}], 상한 {MAX_CONDITION}")
    ref = reference(A)
    # 참조 자체의 점검: 하삼각, 양의 대각, L·Lᵀ = A (float64 반올림 수준)
    if np.any(np.triu(ref, 1) != 0.0) or not np.all(ref.diagonal() > 0.0):
        raise RuntimeError("참조가 양의 대각을 가진 하삼각 행렬이 아니다")
    residual = float(np.abs(ref @ ref.T - A64).max())
    if residual > 1e-12:
        raise RuntimeError(f"참조의 잔차 |L·Lᵀ − A| = {residual:.3g} 가 너무 크다")
    return {"n": n}, {"A": A}, "L", ref


if __name__ == "__main__":
    sys.exit(main_for("cholesky", SHAPES, make_case, "numpy float64 linalg.cholesky of the float32 input", HERE / "tests"))
