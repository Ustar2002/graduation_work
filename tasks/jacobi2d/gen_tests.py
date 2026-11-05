"""jacobi2d 테스트 입력 생성기. 사용: PYTHON_EXE tasks/jacobi2d/gen_tests.py [--seed S] [--out DIR]

입력 ``src`` 는 float32 uniform[0, 1) (H, W). 참조 ``dst_ref`` 는 float64 로 T 스텝을 수행한 결과:
스텝마다 내부 점은 상하좌우 네 이웃의 평균(직전 스텝 값 기준), 경계(첫/끝 행·열)는 ``src`` 값 그대로.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SHAPES = [
    ("tiny", {"H": 5, "W": 7, "T": 3}),
    ("rect", {"H": 130, "W": 200, "T": 4}),
    ("small", {"H": 512, "W": 512, "T": 8}),
    ("large", {"H": 4096, "W": 4096, "T": 20}),
]


def reference(src: np.ndarray, T: int) -> np.ndarray:
    cur = src.astype(np.float64)
    nxt = cur.copy()
    for _ in range(T):
        nxt[1:-1, 1:-1] = 0.25 * (cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:])
        cur, nxt = nxt, cur
    return cur


def make_case(shape, rng):
    src = np.ascontiguousarray(rng.random((shape["H"], shape["W"]), dtype=np.float32))
    return dict(shape), {"src": src}, "dst", reference(src, shape["T"])


if __name__ == "__main__":
    sys.exit(main_for("jacobi2d", SHAPES, make_case, "numpy float64, T ping-pong steps", HERE / "tests"))
