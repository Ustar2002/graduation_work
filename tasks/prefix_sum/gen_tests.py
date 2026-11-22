"""prefix_sum 테스트 입력 생성기. 사용: PYTHON_EXE tasks/prefix_sum/gen_tests.py [--seed S] [--out DIR]

``x`` 는 int32, 값은 [0, 16) 에서 독립 균등 추출. 참조 ``out_ref`` 는 포함형 누적 합(out[i] = x[0] + … + x[i])을 int64 로
구해 int32 로 내린 것이다(NumPy 기본 정수는 int64). 합의 최댓값은 15 · 2^26 = 1,006,632,960 < 2^31 이라 넘치지 않는다.
참조는 정의 그대로 다시 확인한다: out[0] == x[0], out[i] − out[i−1] == x[i].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SENTINEL = -2048  # tolerance.yaml 의 sentinel 과 같은 값(누적 합은 0 이상이라 참조에 나올 수 없다)
VALUE_BOUND = 16  # 입력 값의 범위 [0, VALUE_BOUND). spec.md·signatures.yaml 의 수치와 같아야 한다
MAX_N = 1 << 26   # spec.md 가 보장하는 n 의 상한: (VALUE_BOUND − 1) · MAX_N < 2^31

SHAPES = [
    ("tiny", {"n": 7}),
    ("rect", {"n": 100_003}),
    ("small", {"n": 1 << 20}),
    ("large", {"n": 1 << 26}),
]


def make_case(shape, rng):
    n = shape["n"]
    if not 1 <= n <= MAX_N:
        raise RuntimeError(f"n={n} 이 명세의 범위 [1, {MAX_N}] 밖이다")
    x = rng.integers(0, VALUE_BOUND, size=n, dtype=np.int32)
    wide = np.cumsum(x, dtype=np.int64)
    if int(wide[-1]) > np.iinfo(np.int32).max:
        raise RuntimeError(f"누적 합 {int(wide[-1])} 이 int32 를 넘는다")
    ref = wide.astype(np.int32)
    del wide
    if int(ref[0]) != int(x[0]) or not np.array_equal(np.diff(ref), x[1:]):
        raise RuntimeError("참조가 포함형 누적 합의 정의와 맞지 않는다")
    return {"n": n}, {"x": x}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for("prefix_sum", SHAPES, make_case, "numpy int64 cumsum, int32", HERE / "tests",
                      out_dtype="int32", sentinel=SENTINEL))
