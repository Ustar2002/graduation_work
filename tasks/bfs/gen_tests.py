"""bfs 테스트 입력 생성기. 사용: PYTHON_EXE tasks/bfs/gen_tests.py [--seed S] [--out DIR]

방향 그래프(CSR: ``row_ptr`` int32 [n_vertices + 1], ``col_idx`` int32 [n_edges])와 시작 정점 ``source``.
참조 ``dist_ref`` 는 **int32**: source 에서 각 정점까지의 최소 간선 수, source 자신은 0, 도달 불가는 -1.

- tiny: 손으로 만든 고정 그래프(시드와 무관; ``TINY_ADJACENCY``). 정점 6개·간선 10개 안에 빈 이웃 목록(정점 0 — row_ptr 의
  맨 앞이 빈 구간), 자기 루프, 중복 간선, source 로 돌아오는 간선, 도달 불가 정점에서 도달 가능한 쪽으로 가는 간선을 넣었다.
  source 의 두 이웃이 서로를 가리키므로 깊이 우선 탐색이 정점에 처음 들어갈 때의 깊이는 이웃을 어느 순서로 보든 틀린다
  (실측 2026-11-22: 재귀 순서의 DFS 는 tiny·rect·small 모두 실패. 큐를 스택으로만 바꾸고 발견 순간에 거리를 적는 변형은
  tiny 를 통과하고 rect·small 에서 실패한다 — 그 변형까지 6개 정점으로 막으려면 다른 성질을 버려야 해서 무작위 형태에 맡긴다).
- rect·small·large: 무작위 그래프. 정점의 나가는 차수는 0 … 2·avg 균등(평균 avg), 이웃은 **전체 정점에서 균등 추출**하고
  정렬하지 않는다 → 자기 루프·중복 간선이 자연히 생기고(기대 개수는 각각 avg, 차수 제곱 평균에서 약 40), 들어오는 간선이
  없는 정점(비율 약 e^-avg)은 도달 불가다. source 는 0 이 아니고 나가는 간선이 있는 정점에서 뽑는다(0 에서 시작하는 커널,
  source 를 무시하는 커널이 통과하지 못하게).
- 모든 형태가 ``check_requirements`` 의 성질(도달 불가 정점·자기 루프·중복 간선·빈 이웃 목록이 있다, source 에 나가는 간선이
  있다, 정점의 절반 이상이 도달 가능하다, 최대 거리 ≥ 2)을 갖춰야 한다. 무작위 형태가 못 갖추면 같은 rng 로 다시 뽑는다(결정적).
  spec.md 의 Evaluation 절은 이 성질들과 성능 형태의 수치(최대 거리 10, 도달 불가 정점 약 1,400 개)를 사실로 적는다. 수치는
  ``SHAPES`` 의 ``spec_*`` 키로 같이 검사하므로, 시드를 바꿔도 명세의 문장이 거짓이 되지 않는다(못 맞추면 생성이 실패한다).
- 참조는 레벨 단위 간선 스윕으로 계산하고(``reference``), BFS 거리의 필요충분 조건(``verify``)으로 다시 확인한다 —
  계산 방법과 독립인 검증이라 참조가 틀리면 생성이 실패한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import log, main_for  # noqa: E402

TASK = "bfs"
SENTINEL = -2048  # tolerance.yaml 의 sentinel 과 같아야 한다
MAX_DRAWS = 20    # 무작위 형태를 다시 뽑는 횟수 상한

# tiny: 정점 u 의 이웃 목록(순서 그대로 col_idx 에 들어간다). source = 3 → dist = [3, 1, -1, 0, 1, 2].
TINY_ADJACENCY = [
    [],         # 0: 나가는 간선 없음(거리 3 으로 도달)
    [4, 1],     # 1: source 의 다른 이웃 4 로 가는 간선 + 자기 루프
    [4],        # 2: 도달 불가 정점. 도달 가능한 정점으로 가는 간선은 있다(간선을 무방향으로 다루면 틀린다)
    [1, 4],     # 3: source
    [1, 5, 5],  # 4: source 의 다른 이웃 1 로 가는 간선 + 중복 간선 4→5
    [0, 3],     # 5: source 로 돌아오는 간선(순환)
]
TINY_SOURCE = 3

# avg: 평균 나가는 차수(차수는 0 … 2·avg 균등). 큰 두 형태는 평균 차수 8 (설계 v3 §4).
# 정점 수·차수 범위(0 … 2·avg)와 아래 spec_* 값은 spec.md 의 Evaluation 절에 사실로 적혀 있다 — 바꾸면 spec.md 도 고친다.
# spec_max_dist / spec_unreachable_*: spec.md 가 성능 형태에 대해 적은 수치("the largest distance is 10", "about 1,400 of the
# 4,194,304 vertices (0.03%) are not reachable" = 1,300 개 이상 1,500 개 미만; 기대값 n·e^-8 ≈ 1,407, 표준편차 약 38).
# 시드를 바꿔 어긋나면 다시 뽑는다.
SHAPES = [
    ("tiny", {"n_vertices": 6, "fixed": 1}),
    ("rect", {"n_vertices": 1000, "avg": 4}),
    ("small", {"n_vertices": 1 << 16, "avg": 8}),
    ("large", {"n_vertices": 1 << 22, "avg": 8, "spec_max_dist": 10, "spec_unreachable_min": 1300, "spec_unreachable_max": 1500}),
]


def csr_from_adjacency(adjacency):
    row_ptr = np.zeros(len(adjacency) + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum([len(a) for a in adjacency])
    col_idx = np.array([v for a in adjacency for v in a], dtype=np.int32)
    return row_ptr, col_idx


def random_graph(n_vertices: int, avg: int, rng: np.random.Generator):
    deg = rng.integers(0, 2 * avg + 1, size=n_vertices, dtype=np.int64)
    row_ptr = np.zeros(n_vertices + 1, dtype=np.int64)
    np.cumsum(deg, out=row_ptr[1:])
    n_edges = int(row_ptr[-1])
    if n_edges >= 2**31:
        raise RuntimeError(f"간선 수 {n_edges} 가 int32 범위를 넘는다")
    col_idx = rng.integers(0, n_vertices, size=n_edges, dtype=np.int32)
    return row_ptr.astype(np.int32), col_idx


def edge_tails(row_ptr: np.ndarray) -> np.ndarray:
    """간선 p 의 출발 정점(``col_idx[p]`` 가 도착 정점). int32 [n_edges]."""
    return np.repeat(np.arange(row_ptr.size - 1, dtype=np.int32), np.diff(row_ptr))


def reference(tails: np.ndarray, col_idx: np.ndarray, n_vertices: int, source: int) -> np.ndarray:
    """레벨 단위 간선 스윕: 거리 L 인 정점에서 나가는 간선의 도착 정점 중 아직 거리가 없는 것이 거리 L+1 이다."""
    dist = np.full(n_vertices, -1, dtype=np.int32)
    dist[source] = 0
    level = 0
    while True:
        heads = col_idx[dist[tails] == level]
        heads = heads[dist[heads] < 0]
        if heads.size == 0:
            return dist
        level += 1
        dist[heads] = level


def verify(tails: np.ndarray, col_idx: np.ndarray, dist: np.ndarray, source: int) -> None:
    """``dist`` 가 source 에서의 BFS 거리인 필요충분 조건을 확인한다(``reference`` 의 계산 방법과 독립). 어긋나면 RuntimeError.

    (1) dist[source] = 0, 거리 0 인 정점은 source 뿐, 모든 값은 -1 이상.
    (2) 간선 u→v 에서 dist[u] ≥ 0 이면 0 ≤ dist[v] ≤ dist[u] + 1  ⇒ 도달 가능한 정점은 전부 값이 있고, 값은 실제 거리 이하다.
    (3) dist[v] = d ≥ 1 인 정점마다 dist[u] = d − 1 인 u 에서 오는 간선이 있다  ⇒ 길이 d 인 경로가 실제로 있다(값은 실제 거리 이상).
    """
    if dist.dtype != np.int32:
        raise RuntimeError(f"참조 dtype 이 int32 가 아니다: {dist.dtype}")
    if int(dist[source]) != 0 or int(np.count_nonzero(dist == 0)) != 1 or int(dist.min()) < -1:
        raise RuntimeError("참조 검증 실패 (1): source 의 거리가 0 이 아니거나 거리 0 인 정점이 여럿이거나 -1 보다 작은 값이 있다")
    d_tail, d_head = dist[tails], dist[col_idx]
    from_reached = d_tail >= 0
    if np.any(from_reached & ((d_head < 0) | (d_head > d_tail + 1))):
        raise RuntimeError("참조 검증 실패 (2): 도달 가능한 정점에서 나가는 간선이 거리 조건을 어긴다")
    has_parent = np.zeros(dist.size, dtype=bool)
    has_parent[col_idx[from_reached & (d_head == d_tail + 1)]] = True
    if np.any((dist >= 1) & ~has_parent):
        raise RuntimeError("참조 검증 실패 (3): 한 레벨 앞에서 들어오는 간선이 없는 정점이 있다")


def graph_facts(row_ptr: np.ndarray, tails: np.ndarray, col_idx: np.ndarray, dist: np.ndarray, source: int) -> dict:
    """명세·decisions 에 적는 사실과 ``check_requirements`` 의 입력."""
    n_vertices = dist.size
    deg = np.diff(row_ptr)
    width = int(deg.max()) if n_vertices else 0
    # 중복 간선 수: 이웃 목록을 (n_vertices, 최대 차수) 표로 펴서 행마다 정렬하고 이웃한 같은 값을 센다
    # (= 목록의 원소 수 − 서로 다른 원소 수의 합). 빈 칸은 열마다 다른 음수라 서로도, 정점 번호와도 같지 않다.
    table = np.tile(-np.arange(1, width + 1, dtype=np.int32), (n_vertices, 1))
    table[tails, np.arange(col_idx.size, dtype=np.int32) - row_ptr[tails]] = col_idx
    table.sort(axis=1)
    has_in_edge = np.zeros(n_vertices, dtype=bool)
    has_in_edge[col_idx] = True
    return {
        "n_edges": int(col_idx.size),
        "source_out_degree": int(deg[source]),
        "max_out_degree": width,
        "empty_lists": int(np.count_nonzero(deg == 0)),
        "self_loops": int(np.count_nonzero(tails == col_idx)),
        "duplicate_edges": int(np.count_nonzero(table[:, 1:] == table[:, :-1])),
        "no_in_edges": int(n_vertices - np.count_nonzero(has_in_edge)),
        "unreachable": int(np.count_nonzero(dist < 0)),
        "max_dist": int(dist.max()),
        "level_sizes": [int(c) for c in np.bincount(dist[dist >= 0])],
    }


def check_requirements(shape: dict, source: int, facts: dict) -> str:
    """모든 형태가 갖춰야 하는 성질과, 그 형태에 대해 spec.md 가 적은 수치(``spec_*`` 키가 있을 때).
    어긋난 첫 항목의 설명을 돌려준다(전부 갖추면 ``""``)."""
    n_vertices = shape["n_vertices"]
    checks = [
        (source != 0, "source 가 0 이다"),
        (facts["source_out_degree"] >= 1, "source 에 나가는 간선이 없다"),
        (facts["unreachable"] >= 1, "도달 불가 정점이 없다"),
        (2 * (n_vertices - facts["unreachable"]) >= n_vertices, "도달 가능한 정점이 절반에 못 미친다"),
        (facts["max_dist"] >= 2, "최대 거리가 2 보다 작다"),
        (facts["self_loops"] >= 1, "자기 루프가 없다"),
        (facts["duplicate_edges"] >= 1, "중복 간선이 없다"),
        (facts["empty_lists"] >= 1, "이웃 목록이 빈 정점이 없다"),
        ("spec_max_dist" not in shape or facts["max_dist"] == shape["spec_max_dist"],
         f"최대 거리 {facts['max_dist']} 가 spec.md 의 값 {shape.get('spec_max_dist')} 과 다르다"),
        (shape.get("spec_unreachable_min", 0) <= facts["unreachable"] < shape.get("spec_unreachable_max", n_vertices + 1),
         f"도달 불가 정점 수 {facts['unreachable']} 가 spec.md 의 범위 밖이다"),
    ]
    return next((why for ok, why in checks if not ok), "")


def build(shape: dict, row_ptr: np.ndarray, col_idx: np.ndarray, source: int):
    n_vertices = row_ptr.size - 1
    if n_vertices != shape["n_vertices"]:
        raise RuntimeError(f"정점 수 {n_vertices} 가 형태의 {shape['n_vertices']} 와 다르다")
    tails = edge_tails(row_ptr)
    dist = reference(tails, col_idx, n_vertices, source)
    verify(tails, col_idx, dist, source)
    facts = graph_facts(row_ptr, tails, col_idx, dist, source)
    return dist, facts, check_requirements(shape, source, facts)


def make_case(shape, rng):
    n_vertices = shape["n_vertices"]
    if shape.get("fixed"):
        row_ptr, col_idx = csr_from_adjacency(TINY_ADJACENCY)
        source = TINY_SOURCE
        dist, facts, why = build(shape, row_ptr, col_idx, source)
        if why:
            raise RuntimeError(f"고정 그래프가 요구 성질을 갖추지 못했다: {why}")
    else:
        for draw in range(1, MAX_DRAWS + 1):
            row_ptr, col_idx = random_graph(n_vertices, shape["avg"], rng)
            candidates = np.flatnonzero(np.diff(row_ptr)[1:] > 0) + 1  # 0 이 아니고 나가는 간선이 있는 정점
            if candidates.size == 0:
                why = "나가는 간선이 있는 정점이 없다"
            else:
                source = int(candidates[rng.integers(0, candidates.size)])
                dist, facts, why = build(shape, row_ptr, col_idx, source)
            if not why:
                break
            log(TASK, f"n_vertices={n_vertices}: {draw}번째 추출을 버린다({why})")
        else:
            raise RuntimeError(f"n_vertices={n_vertices}: {MAX_DRAWS}번 뽑아도 요구 성질을 갖춘 그래프가 나오지 않았다")
    if dist.size != n_vertices or row_ptr.size != n_vertices + 1:
        raise RuntimeError("배열 크기가 형태와 다르다")
    log(TASK, f"n_vertices={n_vertices} source={source} " + " ".join(f"{k}={v}" for k, v in facts.items()))
    dims = {"n_vertices": n_vertices, "n_edges": int(col_idx.size), "source": source}
    return dims, {"row_ptr": row_ptr, "col_idx": col_idx}, "dist", dist


if __name__ == "__main__":
    sys.exit(main_for(TASK, SHAPES, make_case, "numpy int32 level-synchronous edge sweep, verified by the BFS distance conditions",
                      HERE / "tests", out_dtype="int32", sentinel=SENTINEL))
