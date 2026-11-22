"""histogram 테스트 입력 생성기. 사용: PYTHON_EXE tasks/histogram/gen_tests.py [--seed S] [--out DIR]

``keys`` 는 int32, 값은 [key_lo, key_hi) 에서 독립 균등 추출. rect·small·large 는 [0, n_bins) 전체라 키가 구간에 고르게
퍼진다(구간마다 약 n / n_bins 개; 빈 구간 없음). tiny 만 [1, n_bins − 1) 로 두어 **첫 구간과 마지막 구간이 항상 빈다**:
빈 구간이 없으면 "나온 값만큼만 세는" 구현(가장 큰 키 + 1 길이의 결과를 그대로 쓰기, 나온 값의 칸만 쓰기, 데이터의
최솟값·최댓값으로 구간을 잡아 그 밖의 칸을 쓰지 않기)이 네 형태를 모두 통과한다(spec.md 는 빈 구간의 개수가 0 이어야
한다고 정의한다). 양 끝을 모두 비워야 최솟값 아래 칸을 빠뜨리는 구현과 최댓값 위 칸을 빠뜨리는 구현이 둘 다 걸린다.
참조 ``hist_ref`` 는 ``np.bincount(minlength=n_bins)`` 를 int32 로 내린 것이다(개수 ≤ n < 2^31; NumPy 기본 정수는 int64).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SENTINEL = -2048  # tolerance.yaml 의 sentinel 과 같은 값(개수는 0 이상이라 참조에 나올 수 없다)

# key_lo, key_hi: 키를 뽑는 구간 [key_lo, key_hi)(생성 매개변수; npz 의 dim 은 n, n_bins 뿐이다).
SHAPES = [
    ("tiny", {"n": 10, "n_bins": 4, "key_lo": 1, "key_hi": 3}),
    ("rect", {"n": 100_003, "n_bins": 1000, "key_lo": 0, "key_hi": 1000}),
    ("small", {"n": 1 << 20, "n_bins": 256, "key_lo": 0, "key_hi": 256}),
    ("large", {"n": 1 << 26, "n_bins": 256, "key_lo": 0, "key_hi": 256}),
]


def make_case(shape, rng):
    n, n_bins, key_lo, key_hi = shape["n"], shape["n_bins"], shape["key_lo"], shape["key_hi"]
    if not 0 <= key_lo < key_hi <= n_bins:
        raise RuntimeError(f"키 구간 [{key_lo}, {key_hi}) 이 [0, {n_bins}) 안에 있지 않다")
    keys = rng.integers(key_lo, key_hi, size=n, dtype=np.int32)
    counts = np.bincount(keys, minlength=n_bins)  # int64
    # 참조의 불변식: 구간 수만큼의 개수, 합은 n. 형태가 약속한 성질(spec.md Evaluation)도 함께 확인한다.
    if counts.shape != (n_bins,) or int(counts.sum()) != n:
        raise RuntimeError(f"개수 배열이 잘못됐다: shape={counts.shape} sum={int(counts.sum())} n={n}")
    if int(counts[:key_lo].sum()) != 0 or int(counts[key_hi:].sum()) != 0:
        raise RuntimeError("비워 두려던 구간에 키가 있다")
    if int(counts[key_lo:key_hi].min()) == 0:
        raise RuntimeError("키를 뽑은 구간 중 빈 구간이 생겼다(n 이 너무 작다)")
    if (key_lo, key_hi) != (0, n_bins) and not (counts[0] == 0 and counts[-1] == 0):
        raise RuntimeError("빈 구간을 두는 형태는 첫 구간과 마지막 구간이 모두 비어야 한다")
    return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", counts.astype(np.int32)


if __name__ == "__main__":
    sys.exit(main_for("histogram", SHAPES, make_case, "numpy bincount, int32", HERE / "tests",
                      out_dtype="int32", sentinel=SENTINEL))
