"""floyd_warshall 테스트 입력 생성기. 사용: PYTHON_EXE tasks/floyd_warshall/gen_tests.py [--seed S] [--out DIR]

``dist_in`` 은 float32 (n, n): 대각 0, 비대각 원소는 확률 DENSITY(0.3) 로 간선(가중치 float32 uniform[1, 10)), 아니면 INF = 1.0e30
(간선 없음). 참조 ``dist_out_ref`` 는 float64 Floyd–Warshall(k 루프는 Python, (i, j) 갱신은 numpy 벡터화).

INF 의 선택(설계 §18): float32 에서 INF + INF = 2e30 은 유한하고 min() 이 INF 를 유지하므로, 단순 min-plus 완화만으로 도달 불가 쌍이
정확히 INF 로 남는다. float64 참조에서도 INF + w (w < 10) 는 1e30 에 흡수되고(ulp ≈ 1.4e14) INF + INF 는 min 에서 버려지므로 참조의
도달 불가 원소는 정확히 float64(float32(1e30)) 이다. 밀도 0.3 이면 n ≥ 37 에서는 거의 모든 쌍이 도달 가능하다 — 밀도는 형태별로
바꾸지 않고(설계 §18) 도달 불가 쌍의 존재는 ``tests/test_tier3_floyd_warshall.py`` 가 확인만 한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

INF = np.float32(1.0e30)     # 간선 없음 / 도달 불가. spec.md 에 같은 값이 공개된다
DENSITY = 0.3                # 비대각 원소가 간선일 확률
W_LO, W_HI = 1.0, 10.0       # 가중치 uniform[W_LO, W_HI)

SHAPES: list[tuple[str, dict[str, int]]] = [
    ("tiny", {"n": 8}),
    ("rect", {"n": 37}),
    ("small", {"n": 256}),
    ("large", {"n": 1024}),
]


def make_graph(n: int, rng: np.random.Generator) -> np.ndarray:
    """밀도 DENSITY 의 방향 그래프를 밀집 float32 행렬로 만든다(대각 0, 간선 없음 = INF)."""
    has_edge = rng.random((n, n)) < DENSITY
    w = (np.float32(W_LO) + np.float32(W_HI - W_LO) * rng.random((n, n), dtype=np.float32)).astype(np.float32)
    dist = np.where(has_edge, w, INF).astype(np.float32)
    np.fill_diagonal(dist, np.float32(0.0))
    return np.ascontiguousarray(dist)


def reference(dist_in: np.ndarray) -> np.ndarray:
    """float64 Floyd–Warshall. k 는 Python 루프, (i, j) 갱신은 numpy 벡터화. n = 1024 에서 1M 원소 최소 연산 1024회."""
    d = np.asarray(dist_in, dtype=np.float64).copy()
    n = d.shape[0]
    for k in range(n):
        np.minimum(d, d[:, k : k + 1] + d[k : k + 1, :], out=d)
    return d


def make_case(shape: dict[str, int], rng: np.random.Generator) -> tuple[dict[str, int], dict[str, np.ndarray], str, np.ndarray]:
    n = shape["n"]
    dist_in = make_graph(n, rng)
    return {"n": n}, {"dist_in": dist_in}, "dist_out", reference(dist_in)


if __name__ == "__main__":
    sys.exit(main_for("floyd_warshall", SHAPES, make_case, "numpy float64 Floyd-Warshall (k loop, vectorized min-plus)", HERE / "tests"))
