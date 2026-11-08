"""cholesky 테스트 입력 생성기. 사용: PYTHON_EXE tasks/cholesky/gen_tests.py [--seed S] [--out DIR]

``A`` 는 float32 n×n 대칭 양정치 행렬: float64 로 ``B·Bᵀ + n·I`` (B 는 uniform[-1, 1) n×n) 를 만들고, 상삼각을 하삼각에 복사해
비트 단위 대칭을 보장한 뒤 float32 로 캐스트한다(캐스트는 원소별이라 대칭이 유지된다). 저장하는 것은 캐스트된 A 이고,
참조 ``L_ref`` 는 **그 float32 A 를 float64 로 올려** 계산한 Cholesky 인자(하삼각, 양의 대각, 엄격 상삼각 0)라서
커널이 보는 입력과 참조가 정확히 같은 행렬을 가리킨다(docs/harness_design.md §18).

조건수: B 의 원소 분산이 1/3 이라 B·Bᵀ 의 고윳값은 약 [0, 4n/3] (Marchenko–Pastur), n·I 를 더하면 [n, 7n/3] → 조건수 ≈ 2.3.
대각은 ≈ 4n/3, 비대각은 표준편차 ≈ sqrt(n/3) 라서 대각이 개별 비대각보다 훨씬 크다(행합 기준 대각 우세는 아니다).
형태마다 참조 L 의 RMS·최대 절대값·A 의 고윳값 범위를 로그로 남긴다(RMS 가 tolerance.yaml 의 atol_rms_rel 근거).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import log, main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 6}),
    ("rect", {"n": 131}),
    ("small", {"n": 512}),
    ("large", {"n": 2048}),
]


def make_spd(n: int, rng: np.random.Generator) -> np.ndarray:
    """float32 대칭 양정치 A = float32(B·Bᵀ + n·I), B ~ uniform[-1, 1) n×n float64. 비트 단위 대칭을 보장한다."""
    b = rng.uniform(-1.0, 1.0, size=(n, n))  # float64
    a64 = b @ b.T
    a64[np.diag_indices(n)] += float(n)
    upper = np.triu(a64, 1)
    a64 = np.diag(np.diag(a64)) + upper + upper.T  # 상삼각을 하삼각에 복사 → 정확히 대칭
    return np.ascontiguousarray(a64.astype(np.float32))


def reference(a32: np.ndarray) -> np.ndarray:
    """float32 A 를 float64 로 올려 계산한 Cholesky 인자 L (float64, 하삼각, 대각 > 0, 엄격 상삼각 0)."""
    return np.linalg.cholesky(np.asarray(a32, dtype=np.float64))


def make_case(shape: dict[str, int], rng: np.random.Generator):
    n = shape["n"]
    a = make_spd(n, rng)
    ref = reference(a)
    rms = float(np.sqrt(np.mean(ref * ref)))
    eig = np.linalg.eigvalsh(a.astype(np.float64))
    log("cholesky", f"n={n}: L_ref RMS={rms:.6g} max|L_ref|={float(np.abs(ref).max()):.6g} "
                    f"min diag={float(np.diag(ref).min()):.6g} eig(A)=[{eig[0]:.6g}, {eig[-1]:.6g}] cond={eig[-1] / eig[0]:.3g}")
    return {"n": n}, {"A": a}, "L", ref


if __name__ == "__main__":
    sys.exit(main_for("cholesky", SHAPES, make_case, "numpy float64 cholesky of the float32-cast A", HERE / "tests"))
