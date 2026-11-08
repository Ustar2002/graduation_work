"""bfs 테스트 입력 생성기. 사용: PYTHON_EXE tasks/bfs/gen_tests.py [--seed S] [--out DIR]

방향 그래프를 CSR(int32)로 만든다. 행 i 의 길이(바깥 차수)는 0 … 32 균등(평균 16; n 보다 크면 n 으로 자른다),
열 인덱스는 spmv_csr 의 층화 추출(행 길이 L 이면 열 범위를 L 개 층으로 나눠 층마다 하나)을 n_cols = n 으로 쓰되
층 경계를 정수 산술로 잡아 행 안에서 오름차순·중복 없음을 구성적으로 보장한다. 자기 루프(col == i)는
걸러내지 않는다(BFS 거리에 영향이 없다). 참조 ``dist_ref`` 는 정점 0 에서의 홉 수(도달 불가 −1)를 numpy 레벨 동기 BFS 로
계산해 float64 로 저장한다(값은 정수).

형태(설계 §18): n = tiny 7 / rect 1003 / small 2^16 / large 2^21 (nnz ≈ 16n; large 의 col_idx 는 약 134 MB).
tiny 만 행 길이 상한을 2 로 낮춘다(``max_degree``): n = 7 에서 상한 32 는 모든 행이 7 개 열 전부를 갖는 완전 그래프가 되어
(dist 가 전부 1) 다단계 경로·도달 불가 정점(−1)을 전혀 검사하지 못하기 때문이다. 시드 20261105 의 tiny 는
dist = [0, −1, 1, 2, 1, 2, −1] 이다(도달 불가 2개). rect 이상은 평균 바깥 차수 16 이라 진입 차수 0 인 정점의 기대 수가 n·e^{−16} 로
사실상 0 이며, 실제로 rect·small·large 에 도달 불가 정점이 없다(tests/test_tier3_bfs.py 가 개수를 보고한다).
large 는 행 블록 단위로 열을 뽑아 메모리 사용을 수백 MB 로 묶는다(결과는 블록 크기와 무관하게 같은 난수 흐름을 쓴다).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

MAX_DEGREE = 32            # 행 길이는 [0, MAX_DEGREE] 균등 → 평균 16 (형태가 max_degree 로 덮어쓸 수 있다)
ROW_BLOCK = 1 << 17        # 열 추출을 이 행 수 단위로 나눠 수행(임시 배열 ≤ 약 200 MB)

SHAPES = [
    ("tiny", {"n": 7, "max_degree": 2}),
    ("rect", {"n": 1003}),
    ("small", {"n": 1 << 16}),
    ("large", {"n": 1 << 21}),
]


def reference(row_ptr: np.ndarray, col_idx: np.ndarray, n: int) -> np.ndarray:
    """정점 0 에서의 BFS 홉 수(int64 → float64). 레벨 동기: 프론티어의 이웃을 np.repeat/누적합으로 한꺼번에 모은다."""
    row_ptr = np.asarray(row_ptr, dtype=np.int64)
    col_idx = np.asarray(col_idx, dtype=np.int64)
    dist = np.full(n, -1, dtype=np.int64)
    dist[0] = 0
    frontier = np.zeros(1, dtype=np.int64)
    level = 0
    while frontier.size:
        starts = row_ptr[frontier]
        lens = row_ptr[frontier + 1] - starts
        total = int(lens.sum())
        if total == 0:
            break
        # 프론티어 정점 f 의 간선 구간 [starts, starts+lens) 를 이어 붙인 위치 목록
        offsets = np.repeat(starts - (np.cumsum(lens) - lens), lens)
        nbrs = col_idx[np.arange(total, dtype=np.int64) + offsets]
        fresh = nbrs[dist[nbrs] == -1]
        if fresh.size == 0:
            break
        level += 1
        dist[fresh] = level
        frontier = np.unique(fresh)
    return dist.astype(np.float64)


def _stratified_cols(lengths: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """행 길이 배열로 층화 추출 열 인덱스(int32, 행 안에서 오름차순·유일)를 만든다. 행 블록 단위로 처리한다."""
    n_rows = lengths.shape[0]
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    col = np.empty(nnz, dtype=np.int32)
    for r0 in range(0, n_rows, ROW_BLOCK):
        r1 = min(r0 + ROW_BLOCK, n_rows)
        lens_b = lengths[r0:r1]
        m = int(lens_b.sum())
        if m == 0:
            continue
        row_of = np.repeat(np.arange(r1 - r0, dtype=np.int64), lens_b)
        ptr_b = row_ptr[r0:r1 + 1] - row_ptr[r0]
        pos = np.arange(m, dtype=np.int64) - ptr_b[row_of]            # 행 안 순번 j
        L = lens_b[row_of]
        # 층 j 는 정수 구간 [ceil(j·n/L), ceil((j+1)·n/L)) — 정수 산술로 경계를 잡아 서로소·비어 있지 않음(n ≥ L)을
        # 보장한다. (spmv_csr 처럼 실수 floor((j+u)·n/L) 로 자르면 경계가 정수가 아닐 때 드물게 이웃 층과 같은 열이
        # 나올 수 있어 여기서는 정수 경계로 바꿨다.)
        lo = (pos * n + L - 1) // L
        hi = ((pos + 1) * n + L - 1) // L
        u = rng.random(m)                                              # 층 안 위치
        c = lo + np.floor(u * (hi - lo)).astype(np.int64)
        np.minimum(c, hi - 1, out=c)
        same_row = row_of[1:] == row_of[:-1]
        if m > 1 and np.any(c[1:][same_row] <= c[:-1][same_row]):
            raise RuntimeError("열 인덱스가 행 안에서 오름차순·유일하지 않다")
        if np.any(c < 0) or np.any(c >= n):
            raise RuntimeError("열 인덱스가 [0, n) 을 벗어난다")
        col[row_ptr[r0]:row_ptr[r1]] = c
    return col, row_ptr


def make_case(shape, rng):
    n = shape["n"]
    max_degree = int(shape.get("max_degree", MAX_DEGREE))
    lengths = rng.integers(0, max_degree + 1, size=n, dtype=np.int64)
    np.minimum(lengths, n, out=lengths)
    col, row_ptr = _stratified_cols(lengths, n, rng)
    nnz = int(row_ptr[-1])
    if nnz >= 2**31 - 1 or n >= 2**31 - 1:
        raise RuntimeError(f"int32 범위를 넘는다: n={n}, nnz={nnz}")
    row_ptr32 = row_ptr.astype(np.int32)
    dist = reference(row_ptr32, col, n)
    if dist[0] != 0.0 or dist.shape != (n,) or np.any(dist < -1):
        raise RuntimeError("참조 BFS 결과가 계약을 벗어난다")
    dims = {"n": n, "nnz": nnz}
    inputs = {"row_ptr": row_ptr32, "col_idx": col}
    return dims, inputs, "dist", dist


if __name__ == "__main__":
    sys.exit(main_for("bfs", SHAPES, make_case, "numpy level-synchronous BFS from vertex 0 (int64 hops, -1 unreachable)", HERE / "tests"))
