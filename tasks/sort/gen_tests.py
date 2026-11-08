"""sort 테스트 입력 생성기. 사용: PYTHON_EXE tasks/sort/gen_tests.py [--seed S] [--out DIR]

``x`` 는 int32, [0, 2^31 − 1] 균등(``integers`` 의 high 는 배타적이라 ``1 << 31`` 을 준다). 중복은 생일 문제대로 자연히 생긴다:
rect(100,003)에서 수 개, small(2^20)에서 수백 개, large(10^7)에서 약 2만 3천 개(실측 2026-11-08).
참조 ``out_ref`` 는 오름차순 정렬 복사본을 float64 로 담은 것(2^53 미만의 정수라 정확). 정수 출력이므로 하네스는
정확 일치(rtol = atol = 0)로 판정한다(docs/harness_design.md §17). 기준선은 NumPy 정렬(단일 스레드).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

VALUE_HIGH = 1 << 31  # 배타적 상한: 값은 0 … 2^31 − 1 (int32 의 음이 아닌 전 범위)

SHAPES = [
    ("tiny", {"n": 7}),
    ("rect", {"n": 100_003}),
    ("small", {"n": 1 << 20}),
    ("large", {"n": 10_000_000}),
]


def make_case(shape, rng):
    n = shape["n"]
    x = rng.integers(0, VALUE_HIGH, size=n, dtype=np.int32)
    ref = np.sort(x).astype(np.float64)
    if ref.shape != (n,) or (n > 1 and bool(np.any(ref[1:] < ref[:-1]))):
        raise RuntimeError("참조가 오름차순 정렬 복사본이 아니다")
    return {"n": n}, {"x": x}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for("sort", SHAPES, make_case, "numpy sort (ascending copy) as float64", HERE / "tests"))
