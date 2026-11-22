"""sort 테스트 입력 생성기. 사용: PYTHON_EXE tasks/sort/gen_tests.py [--seed S] [--out DIR]

``keys`` 는 int32. 형태별 분포:

- tiny (n = 9): 손으로 고른 값 묶음 ``TINY_KEYS``(중복 두 쌍, 음수, 0, −1, int32 의 최솟값·최댓값)를 시드로 섞는다.
  비교 함수의 뺄셈 넘침(``a − b``), 부호 없는 정렬(음수가 뒤로 감), 중복 처리를 가장 작은 입력에서 드러내려는 것이다.
- rect (n = 100003): 좁은 범위 [−1000, 1000) 균등 → 값마다 약 50번 중복.
- small (n = 2^16), large (n = 2^24): int32 전체 범위 균등.

센티널(−2048; 워커가 출력 버퍼를 미리 채우는 값)은 정당한 키가 될 수 있는 값이므로 입력에서 뺀다: 뽑힌 것은 −2047 로 바꾸고
남아 있지 않음을 확인한다(출력은 입력의 순열이라 참조에도 없다 — ``harness.testgen`` 이 한 번 더 확인한다).

참조 ``out_ref`` 는 오름차순으로 정렬한 int32 배열이다. 서로 다른 두 알고리즘(``kind="stable"`` 과 기본 정렬)의 결과가 같고,
비내림차순이며, 값의 합이 입력과 같음을 확인한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SENTINEL = -2048  # tolerance.yaml 의 sentinel 과 같은 값
INT32_MIN = int(np.iinfo(np.int32).min)
INT32_MAX = int(np.iinfo(np.int32).max)

# 가장 작은 형태의 값 묶음(순서는 시드로 섞는다). spec.md 의 "smallest correctness shape" 설명과 같아야 한다.
TINY_KEYS = (7, -3, INT32_MAX, 0, -3, INT32_MIN, 7, -1, 256)

# lo < hi 면 [lo, hi) 균등, lo == hi 면 TINY_KEYS 를 섞은 것(n 은 len(TINY_KEYS) 와 같아야 한다).
SHAPES = [
    ("tiny", {"n": len(TINY_KEYS), "lo": 0, "hi": 0}),
    ("rect", {"n": 100_003, "lo": -1000, "hi": 1000}),
    ("small", {"n": 1 << 16, "lo": INT32_MIN, "hi": INT32_MAX + 1}),
    ("large", {"n": 1 << 24, "lo": INT32_MIN, "hi": INT32_MAX + 1}),
]


def draw_keys(shape, rng) -> np.ndarray:
    n, lo, hi = shape["n"], shape["lo"], shape["hi"]
    if lo == hi:
        if n != len(TINY_KEYS):
            raise RuntimeError(f"손으로 고른 값 묶음은 n = {len(TINY_KEYS)} 에만 쓴다(n = {n})")
        keys = rng.permutation(np.array(TINY_KEYS, dtype=np.int32))
        # 섞은 결과가 우연히 정렬돼 있으면(오름·내림) 한 칸 돌려 정렬되지 않은 입력으로 만든다
        if np.all(keys[:-1] <= keys[1:]) or np.all(keys[:-1] >= keys[1:]):
            keys = np.roll(keys, 1)
        return np.ascontiguousarray(keys, dtype=np.int32)
    keys = rng.integers(lo, hi, size=n, dtype=np.int32)
    keys[keys == SENTINEL] = SENTINEL + 1
    return keys


def make_case(shape, rng):
    n = shape["n"]
    keys = draw_keys(shape, rng)
    if keys.dtype != np.int32 or keys.shape != (n,):
        raise RuntimeError(f"keys 가 int32 ({n},) 이 아니다: {keys.dtype} {keys.shape}")
    if np.count_nonzero(keys == SENTINEL):
        raise RuntimeError(f"입력에 센티널 값 {SENTINEL} 이 있다")
    ref = np.sort(keys, kind="stable").astype(np.int32, copy=False)  # NumPy 기본 정수는 int64 이므로 dtype 을 명시한다
    if ref.dtype != np.int32 or not np.all(ref[:-1] <= ref[1:]):
        raise RuntimeError("참조가 int32 비내림차순이 아니다")
    if not np.array_equal(ref, np.sort(keys)) or int(ref.sum(dtype=np.int64)) != int(keys.sum(dtype=np.int64)):
        raise RuntimeError("참조가 입력의 순열이 아니다(두 정렬 알고리즘의 결과 또는 합이 다르다)")
    return {"n": n}, {"keys": keys}, "out", ref


if __name__ == "__main__":
    sys.exit(main_for("sort", SHAPES, make_case, "numpy stable sort, int32", HERE / "tests",
                      out_dtype="int32", sentinel=SENTINEL))
