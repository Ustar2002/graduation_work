"""gemm 테스트 입력 생성기 (docs/harness_design.md §3).

사용: PYTHON_EXE tasks/gemm/gen_tests.py [--seed 20261103] [--out tasks/gemm/tests]

- 형태(SHAPES 순서: tiny, rect, small, large)마다 독립 시드 ``seed + index`` 로
  ``np.random.default_rng`` 를 만들고 A, B를 float32 uniform[-1, 1)로 채운다.
- 파일 ``<name>.npz`` (``np.savez``, 비압축) 키: ``A``, ``B`` (float32), ``C_ref`` (float64,
  reference.py), ``M``, ``N``, ``K``, ``seed`` (그 형태에 실제로 쓴 시드 = seed + index).
- ``manifest.json``: seed, generated_at, c_ref_method, c_ref_blas, files{name: {M, N, K,
  sha256_inputs}}, test_hash. ``test_hash`` 는 **입력만으로** 정의한다:
  sha256( str(seed) ‖ 형태마다 "M,N,K" ‖ A.tobytes() ‖ B.tobytes() (형태 순서대로) )[:16].
  C_ref는 BLAS에 따라 비트가 달라질 수 있어 해시에서 제외한다.
  ``sha256_inputs`` = sha256(A.tobytes() ‖ B.tobytes()) (형태별, 64자 전체).
- 같은 시드로 다시 실행하면 npz 파일은 바이트 단위로 동일하다(zip 항목 시각은 고정값).
  manifest.json은 ``generated_at`` 만 달라진다.
- 모든 파일은 원자적으로 쓴다(tmp → os.replace). stdout에는 아무것도 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # 어느 cwd에서 실행해도 같은 디렉터리의 reference.py를 찾도록
    sys.path.insert(0, str(HERE))
from reference import reference_gemm  # noqa: E402

DEFAULT_SEED = 20261103
DEFAULT_OUT = HERE / "tests"
C_REF_METHOD = "numpy float64 matmul"


@dataclass(frozen=True)
class Shape:
    """테스트 형태 하나: 파일 이름과 (M, N, K)."""

    name: str
    M: int
    N: int
    K: int


# 순서가 곧 index(시드 오프셋)이자 test_hash 입력 순서다. 바꾸면 test_hash가 바뀐다.
SHAPES: tuple[Shape, ...] = (
    Shape("tiny", 7, 5, 3),
    Shape("rect", 200, 130, 312),
    Shape("small", 256, 256, 256),
    Shape("large", 1024, 1024, 1024),
)


def _log(msg: str) -> None:
    print(f"[gen_tests] {msg}", file=sys.stderr)


def uniform_pm1(rng: np.random.Generator, shape: tuple[int, int]) -> np.ndarray:
    """float32 uniform[-1, 1). rng.random(dtype=float32)는 [0, 1)이고, 2x−1은 float32에서 정확히 [-1, 1)에 든다
    (최댓값 2·(1−2⁻²⁴)−1 = 1−2⁻²³ < 1, 최솟값 −1)."""
    x = rng.random(shape, dtype=np.float32)
    out = x * np.float32(2.0) - np.float32(1.0)
    if out.dtype != np.float32:  # NEP 50 아래에서는 유지되지만, 계약이므로 확인한다
        raise RuntimeError(f"uniform_pm1: dtype drifted to {out.dtype}")
    return np.ascontiguousarray(out)


def make_inputs(shape: Shape, shape_seed: int) -> tuple[np.ndarray, np.ndarray]:
    """형태 하나의 (A, B)를 생성한다. A: (M, K), B: (K, N), float32, C-contiguous."""
    rng = np.random.default_rng(shape_seed)
    A = uniform_pm1(rng, (shape.M, shape.K))
    B = uniform_pm1(rng, (shape.K, shape.N))
    return A, B


def blas_name() -> str:
    """현재 프로세스의 NumPy가 빌드 시 링크한 BLAS 이름(+버전). manifest의 c_ref_blas 용."""
    try:
        blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    except Exception as exc:  # show_config 형식이 바뀌어도 생성은 계속한다
        _log(f"numpy.show_config failed: {exc!r}")
        return "unknown"
    name = str(blas.get("name") or "unknown")
    version = blas.get("version")
    return f"{name} {version}" if version else name


def write_npz_atomic(path: Path, **arrays: np.ndarray | int) -> None:
    """np.savez를 tmp 파일에 쓰고 os.replace로 교체한다(파일 객체를 넘겨 .npz 자동 접미 방지)."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json_atomic(path: Path, obj: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def generate(seed: int, out_dir: Path) -> dict:
    """모든 형태의 npz와 manifest.json을 out_dir에 쓰고 manifest dict를 반환한다."""
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    h_all = hashlib.sha256()
    h_all.update(str(seed).encode("ascii"))
    files: dict[str, dict] = {}
    for index, shape in enumerate(SHAPES):
        shape_seed = seed + index
        A, B = make_inputs(shape, shape_seed)
        C_ref = reference_gemm(A, B)
        if C_ref.dtype != np.float64 or C_ref.shape != (shape.M, shape.N):
            raise RuntimeError(f"reference_gemm returned {C_ref.dtype} {C_ref.shape} for {shape}")

        a_bytes, b_bytes = A.tobytes(), B.tobytes()
        h_in = hashlib.sha256()
        h_in.update(a_bytes)
        h_in.update(b_bytes)
        h_all.update(f"{shape.M},{shape.N},{shape.K}".encode("ascii"))
        h_all.update(a_bytes)
        h_all.update(b_bytes)

        path = out_dir / f"{shape.name}.npz"
        write_npz_atomic(
            path,
            A=A,
            B=B,
            C_ref=C_ref,
            M=np.int64(shape.M),
            N=np.int64(shape.N),
            K=np.int64(shape.K),
            seed=np.int64(shape_seed),
        )
        files[shape.name] = {
            "M": shape.M,
            "N": shape.N,
            "K": shape.K,
            "sha256_inputs": h_in.hexdigest(),
        }
        _log(
            f"wrote {path} M={shape.M} N={shape.N} K={shape.K} seed={shape_seed} "
            f"size={path.stat().st_size} B sha256_inputs={h_in.hexdigest()[:16]}"
        )

    manifest = {
        "seed": seed,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "c_ref_method": C_REF_METHOD,
        "c_ref_blas": blas_name(),
        "files": files,
        "test_hash": h_all.hexdigest()[:16],
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    _log(f"wrote {out_dir / 'manifest.json'} test_hash={manifest['test_hash']} c_ref_blas={manifest['c_ref_blas']}")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="gemm 테스트 입력(npz)과 manifest.json을 시드 고정으로 생성한다 (재생성 시 바이트 동일)."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"기본 시드 (기본 {DEFAULT_SEED}); 형태 i는 seed+i")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help=f"출력 디렉터리 (기본 {DEFAULT_OUT})")
    args = parser.parse_args(argv)
    try:
        generate(args.seed, args.out.resolve())
    except Exception as exc:
        _log(f"error: {type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
