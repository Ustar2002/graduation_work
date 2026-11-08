"""scan 테스트 입력 생성기. 사용: PYTHON_EXE tasks/scan/gen_tests.py [--seed S] [--out DIR]

``x`` 는 int32 uniform [0, 32). 참조 ``out_ref`` 는 int64 누적합(inclusive)을 float64 로 저장한 것 — 최대 31·2^25 < 2^53 이라 정확하다.
값을 [0, 32) 로 제한한 이유(설계 §17, docs/decisions.md 2026-11-08): n = 2^25 에서도 모든 접두합이 2^31 안에 들어 int32 누적이 정당하고,
부동소수 누적 정밀도 문제(dot 과제)가 알고리즘 결정(블록 분할·2-pass 스캔)과 섞이지 않게 정수로 둔다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

X_MAX_EXCLUSIVE = 32  # x ∈ [0, 32)
INT32_MAX = np.iinfo(np.int32).max

SHAPES: list[tuple[str, dict[str, int]]] = [
    ("tiny", {"n": 7}),
    ("rect", {"n": 100_003}),
    ("small", {"n": 1 << 20}),
    ("large", {"n": 1 << 25}),
]


def reference(x: np.ndarray) -> np.ndarray:
    """inclusive 접두합의 참조: int64 누적 → float64. 총합이 int32 를 넘으면 과제 계약 위반이므로 예외."""
    ref64 = np.cumsum(np.asarray(x, dtype=np.int64), dtype=np.int64)
    if ref64.size and int(ref64[-1]) > INT32_MAX:
        raise RuntimeError(f"접두합 {int(ref64[-1])} 이 int32 범위를 넘는다 (n={x.size})")
    return ref64.astype(np.float64)


def make_case(shape: dict[str, int], rng: np.random.Generator) -> tuple[dict[str, int], dict[str, np.ndarray], str, np.ndarray]:
    n = shape["n"]
    x = rng.integers(0, X_MAX_EXCLUSIVE, size=n, dtype=np.int32)
    return {"n": n}, {"x": x}, "out", reference(x)


if __name__ == "__main__":
    sys.exit(main_for("scan", SHAPES, make_case, "numpy int64 cumsum", HERE / "tests"))
