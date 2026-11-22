"""floyd_warshall 테스트 입력 생성기. 사용: PYTHON_EXE tasks/floyd_warshall/gen_tests.py [--seed S] [--out DIR]

입력 ``W`` 는 완전 방향 그래프의 간선 가중치 float32 (n, n): 대각은 0, 나머지는 uniform[1, 100). 방향 그래프이므로
``W`` 는 대칭이 아니다. 참조 ``D_ref`` 는 float64 로 중간 정점 k 에 대한 표준 동적 계획(k 바깥, 행·열은 NumPy 로 한 번에)을
돌린 모든 쌍 최단 거리다. 가중치가 양수라 음의 순환이 없고 출력은 전부 유한하다(직접 간선이 있으므로 거리 < 100).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

SHAPES = [
    ("tiny", {"n": 5}),
    ("rect", {"n": 97}),      # 소수: 어떤 블록 크기로도 나누어떨어지지 않는다
    ("small", {"n": 256}),
    ("large", {"n": 1024}),
]

WEIGHT_LOW, WEIGHT_HIGH = 1.0, 100.0   # 대각 밖 가중치의 구간 [1, 100)


def reference(W: np.ndarray) -> np.ndarray:
    """float64 모든 쌍 최단 거리: k = 0..n-1 순서로 D[i][j] = min(D[i][j], D[i][k] + D[k][j])."""
    D = W.astype(np.float64)
    tmp = np.empty_like(D)
    for k in range(D.shape[0]):
        np.add(D[:, k:k + 1], D[k:k + 1, :], out=tmp)   # 스텝 k 에서 k 행·k 열은 바뀌지 않는다(D[k][k] = 0)
        np.minimum(D, tmp, out=D)
    return D


def make_case(shape, rng):
    n = shape["n"]
    u = rng.random((n, n), dtype=np.float32)
    W = (u * np.float32(WEIGHT_HIGH - WEIGHT_LOW) + np.float32(WEIGHT_LOW)).astype(np.float32)
    np.fill_diagonal(W, np.float32(0.0))
    W = np.ascontiguousarray(W)
    off = ~np.eye(n, dtype=bool)
    if not (np.all(W[off] >= WEIGHT_LOW) and np.all(W[off] < WEIGHT_HIGH)):
        raise RuntimeError(f"대각 밖 가중치가 [{WEIGHT_LOW}, {WEIGHT_HIGH}) 를 벗어났다")
    ref = reference(W)
    # 참조 자체의 점검: 대각 0, 직접 간선보다 길지 않음, 유한
    if np.any(ref.diagonal() != 0.0) or np.any(ref > W) or not np.all(np.isfinite(ref)):
        raise RuntimeError("참조 거리가 계약(대각 0, D ≤ W, 유한)에 맞지 않는다")
    return {"n": n}, {"W": W}, "D", ref


if __name__ == "__main__":
    sys.exit(main_for("floyd_warshall", SHAPES, make_case, "numpy float64 dynamic programme over intermediate vertices",
                      HERE / "tests"))
