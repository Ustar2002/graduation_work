"""spmv_csr 테스트 입력 생성기. 사용: PYTHON_EXE tasks/spmv_csr/gen_tests.py [--seed S] [--out DIR]

행 길이는 0 … 2·avg 균등(빈 행 포함), 열 인덱스는 행 안에서 오름차순·중복 없음(층화 추출), 값과 x 는 float32 uniform[-1, 1).
참조 ``y_ref`` 는 float64 (np.bincount 가중 합).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from harness.testgen import main_for  # noqa: E402

# avg: 행당 평균 비영 원소 수. small 은 10^4 × 10^4, 밀도 1e-3 (CLAUDE.md §4), large 는 10^6 행.
SHAPES = [
    ("tiny", {"n_rows": 5, "n_cols": 7, "avg": 2}),
    ("rect", {"n_rows": 300, "n_cols": 517, "avg": 6}),
    ("small", {"n_rows": 10_000, "n_cols": 10_000, "avg": 10}),
    ("large", {"n_rows": 1_000_000, "n_cols": 1_000_000, "avg": 16}),
]


def make_case(shape, rng):
    n_rows, n_cols, avg = shape["n_rows"], shape["n_cols"], shape["avg"]
    lengths = rng.integers(0, 2 * avg + 1, size=n_rows, dtype=np.int64)
    lengths = np.minimum(lengths, n_cols)
    row_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(lengths, out=row_ptr[1:])
    nnz = int(row_ptr[-1])
    row_of = np.repeat(np.arange(n_rows, dtype=np.int64), lengths)
    pos = np.arange(nnz, dtype=np.int64) - row_ptr[row_of]            # 행 안 순번 j
    L = lengths[row_of].astype(np.float64)
    u = rng.random(nnz)                                                # 층 안 위치
    col = np.floor((pos + u) * (n_cols / L)).astype(np.int64)          # 층 j 는 [j·n/L, (j+1)·n/L) → 오름차순·중복 없음
    col = np.minimum(col, n_cols - 1)
    vals = (rng.random(nnz, dtype=np.float32) * np.float32(2) - np.float32(1)).astype(np.float32)
    x = (rng.random(n_cols, dtype=np.float32) * np.float32(2) - np.float32(1)).astype(np.float32)
    # 층화 추출이 중복을 만들지 않았는지 확인(n_cols >= L 이면 성립)
    same_row = row_of[1:] == row_of[:-1]
    if nnz > 1 and np.any(col[1:][same_row] <= col[:-1][same_row]):
        raise RuntimeError("열 인덱스가 행 안에서 오름차순·유일하지 않다")
    y = np.bincount(row_of, weights=vals.astype(np.float64) * x.astype(np.float64)[col], minlength=n_rows).astype(np.float64)
    dims = {"n_rows": n_rows, "n_cols": n_cols, "nnz": nnz}
    inputs = {"row_ptr": row_ptr.astype(np.int32), "col_idx": col.astype(np.int32), "vals": vals, "x": x}
    return dims, inputs, "y", y


if __name__ == "__main__":
    sys.exit(main_for("spmv_csr", SHAPES, make_case, "numpy float64 bincount", HERE / "tests"))
