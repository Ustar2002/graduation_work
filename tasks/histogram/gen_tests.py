"""histogram 테스트 입력 생성기. 사용: PYTHON_EXE tasks/histogram/gen_tests.py [--seed S] [--out DIR]

``data`` 는 int32, 값은 ``[0, nbins)`` 에서 독립 균등(``rng.integers``). 참조 ``hist_ref`` 는 정확한 빈 개수(float64 로 저장;
값은 정수). 형태(설계 §17): (n, nbins) = tiny (13, 4) / rect (100003, 37) / small (2^20, 256) / large (2^25, 1024).
균등 분포라 빈마다 약 n/nbins 개가 들어가고, 출력 버퍼는 센티널로 채워져 넘어오므로 0 초기화는 커널의 책임이다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 13, "nbins": 4}),
    ("rect", {"n": 100_003, "nbins": 37}),
    ("small", {"n": 1 << 20, "nbins": 256}),
    ("large", {"n": 1 << 25, "nbins": 1024}),
]


def make_case(shape, rng):
    n, nbins = shape["n"], shape["nbins"]
    data = rng.integers(0, nbins, size=n, dtype=np.int32)
    ref = np.bincount(data, minlength=nbins).astype(np.float64)
    if ref.shape != (nbins,) or int(ref.sum()) != n:
        raise RuntimeError(f"참조 개수가 맞지 않다: shape={ref.shape}, sum={ref.sum()}, n={n}")
    return {"n": n, "nbins": nbins}, {"data": data}, "hist", ref


if __name__ == "__main__":
    sys.exit(main_for("histogram", SHAPES, make_case, "numpy bincount (exact integer counts)", HERE / "tests"))
