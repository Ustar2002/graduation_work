"""dot 테스트 입력 생성기. 사용: PYTHON_EXE tasks/dot/gen_tests.py [--seed S] [--out DIR]

``x``, ``y`` 는 float32 uniform[0, 1). 참조 ``out_ref`` 는 float64 누적 합(길이 1 배열).
값이 모두 양수라 합이 n/4 까지 커진다: n = 2^25 에서 단순 float32 순차 누적은 약 7 %, 4분할 float32 누적은 약 0.7 % 틀린다
(실측 2026-11-05). 누적 정밀도가 구현의 결정 사항이 되게 하려는 선택이다(docs/decisions.md).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 7}),
    ("rect", {"n": 100_003}),
    ("small", {"n": 1 << 20}),
    ("large", {"n": 1 << 25}),
]


def make_case(shape, rng):
    n = shape["n"]
    x = rng.random(n, dtype=np.float32)
    y = rng.random(n, dtype=np.float32)
    ref = np.array([np.dot(x.astype(np.float64), y.astype(np.float64))], dtype=np.float64)
    return {"n": n}, {"x": x, "y": y}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for("dot", SHAPES, make_case, "numpy float64 dot", HERE / "tests"))
