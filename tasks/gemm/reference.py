"""gemm 참조 구현 (docs/harness_design.md §3).

reference_gemm(A, B): 입력을 float64로 변환해 A @ B 를 계산하고 float64 배열을 반환한다.
gen_tests.py가 C_ref 생성에 쓰고, 정확성 비교(허용오차 rtol/atol)는 이 float64 결과를 기준으로 한다.
결과의 비트 패턴은 NumPy가 링크한 BLAS에 따라 달라질 수 있으므로 test_hash에는 포함하지 않는다.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def reference_gemm(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """float64 누적 행렬곱. A: (M, K), B: (K, N) → (M, N) float64 (C-contiguous)."""
    A64 = np.ascontiguousarray(A, dtype=np.float64)
    B64 = np.ascontiguousarray(B, dtype=np.float64)
    if A64.ndim != 2 or B64.ndim != 2 or A64.shape[1] != B64.shape[0]:
        raise ValueError(f"reference_gemm: incompatible shapes {A64.shape} @ {B64.shape}")
    return np.ascontiguousarray(A64 @ B64)


def main(argv: list[str] | None = None) -> int:
    """npz 테스트 파일의 A, B로 참조값을 다시 계산해 저장된 C_ref와의 최대 절대 차이를 출력한다."""
    parser = argparse.ArgumentParser(description="gemm float64 reference (sanity check against a test npz)")
    parser.add_argument("--npz", type=Path, required=True, help="gen_tests.py가 만든 npz 파일")
    args = parser.parse_args(argv)
    with np.load(args.npz) as data:
        A, B, C_ref = data["A"], data["B"], data["C_ref"]
        M, N, K = int(data["M"]), int(data["N"]), int(data["K"])
    out = reference_gemm(A, B)
    if out.shape != (M, N):
        print(f"[reference] shape mismatch: got {out.shape}, expected {(M, N)}", file=sys.stderr)
        return 1
    max_abs_diff = float(np.max(np.abs(out - C_ref))) if out.size else 0.0
    print(f"{args.npz.name}: M={M} N={N} K={K} max|recomputed - C_ref| = {max_abs_diff:.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
