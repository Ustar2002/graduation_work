"""과제 공통 테스트 입력 생성 헬퍼 (M2). ``tasks/<task>/gen_tests.py`` 가 쓴다(gemm 은 M1 의 자체 구현을 유지).

- 형태마다 독립 시드 ``seed + index``. npz 키: 입력 배열(인자 이름), ``<out>_ref``, dim 스칼라, ``seed``.
- 참조 dtype(설계 v3 §2.4): float 출력은 float64, int32 출력은 **int32**(파일 크기 때문에 int64 로 넓히지 않는다).
  호출자가 ``out_dtype`` 으로 출력 dtype 을 알리고 ``generate()`` 가 ``make_case`` 의 참조 dtype 을 검사한다.
- ``sentinel`` 을 주면 참조에 그 값이 없음을 확인한다(정수 출력 과제: 워커가 출력 버퍼를 센티널로 미리 채우므로
  정당한 출력이 센티널과 같으면 "쓰지 않음"과 구별할 수 없다).
- ``manifest.json`` 의 ``test_hash`` 는 입력만으로 정의한다: sha256(str(seed) ‖ 형태마다 "k=v,…" ‖ 입력 바이트(인자 순서))[:16].
- 같은 시드로 다시 만들면 npz 는 바이트 단위로 같다. 파일은 원자적으로 쓴다.
- 형태 이름은 모든 과제가 ``tiny, rect, small, large`` 다(1-D 과제에서 ``rect`` 는 "2의 거듭제곱이 아닌 중간 크기"라는 뜻).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

TEST_NAMES = ("tiny", "rect", "small", "large")  # config.measurement 의 correctness_tests + performance_test 와 같아야 한다
DEFAULT_SEED = 20261105

# 출력 dtype → 참조(``<out>_ref``) dtype
REF_DTYPES: dict[str, np.dtype] = {"float32": np.dtype(np.float64), "int32": np.dtype(np.int32)}
DEFAULT_OUT_DTYPE = "float32"


def log(task: str, msg: str) -> None:
    print(f"[gen_tests:{task}] {msg}", file=sys.stderr)


def write_npz_atomic(path: Path, **arrays: Any) -> None:
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
        with open(tmp, "w", encoding="utf-8", newline="\n") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def check_reference(task: str, name: str, ref: np.ndarray, out_dtype: str, sentinel: float | int | None) -> None:
    """참조 배열이 출력 dtype 에 맞고(``REF_DTYPES``) 센티널 값을 포함하지 않는지 확인한다. 어긋나면 ``RuntimeError``."""
    if out_dtype == "float32":
        if ref.dtype != np.float64:
            raise RuntimeError(f"{task}/{name}: 참조는 float64 여야 한다({ref.dtype})")
    elif ref.dtype != REF_DTYPES[out_dtype]:
        raise RuntimeError(f"{task}/{name}: {out_dtype} 출력의 참조는 {REF_DTYPES[out_dtype]} 여야 한다({ref.dtype}); "
                           "NumPy 의 기본 정수형은 int64 이므로 명시적으로 변환할 것")
    if sentinel is not None:
        n_hit = int(np.count_nonzero(ref == sentinel))
        if n_hit:
            raise RuntimeError(f"{task}/{name}: 참조 출력에 센티널 값 {sentinel!r} 이 {n_hit}개 있다 "
                               "(입력 생성 또는 tolerance.yaml 의 sentinel 을 바꿀 것)")


def generate(
    task: str,
    seed: int,
    out_dir: Path,
    shapes: list[tuple[str, dict[str, int]]],
    make_case: Callable[[dict[str, int], np.random.Generator], tuple[dict[str, int], dict[str, np.ndarray], str, np.ndarray]],
    ref_method: str,
    *,
    out_dtype: str = DEFAULT_OUT_DTYPE,
    sentinel: float | int | None = None,
) -> dict:
    """``make_case(shape, rng) -> (dims, inputs, out_name, ref)``. ``dims`` 는 npz 에 쓰는 dim 스칼라(인자 이름),
    ``inputs`` 는 인자 순서의 입력 배열. 형태 이름은 ``TEST_NAMES`` 와 같아야 한다.

    ``out_dtype``: 과제의 출력 dtype(``float32`` 기본 | ``int32``) — 참조 dtype 을 정한다(float64 | int32).
    ``sentinel``: 주면 참조에 그 값이 없음을 확인한다(``tolerance.yaml`` 의 ``sentinel`` 과 같은 값을 넘긴다)."""
    if [n for n, _ in shapes] != list(TEST_NAMES):
        raise ValueError(f"형태 이름은 {TEST_NAMES} 순서여야 한다: {[n for n, _ in shapes]}")
    if out_dtype not in REF_DTYPES:
        raise ValueError(f"out_dtype 은 {tuple(REF_DTYPES)} 중 하나여야 한다: {out_dtype!r}")
    if out_dtype != "float32" and sentinel is None:
        log(task, f"경고: {out_dtype} 출력인데 sentinel 을 주지 않아 참조의 센티널 충돌을 확인하지 않는다")
    out_dir.mkdir(parents=True, exist_ok=True)
    h_all = hashlib.sha256(str(seed).encode("ascii"))
    files: dict[str, dict] = {}
    for index, (name, shape) in enumerate(shapes):
        shape_seed = seed + index
        dims, inputs, out_name, ref = make_case(shape, np.random.default_rng(shape_seed))
        check_reference(task, name, ref, out_dtype, sentinel)
        h_in = hashlib.sha256()
        h_all.update(",".join(f"{k}={v}" for k, v in dims.items()).encode("ascii"))
        for arr in inputs.values():
            b = np.ascontiguousarray(arr).tobytes()
            h_in.update(b)
            h_all.update(b)
        path = out_dir / f"{name}.npz"
        write_npz_atomic(path, **inputs, **{f"{out_name}_ref": ref}, **{k: np.int64(v) for k, v in dims.items()},
                         seed=np.int64(shape_seed))
        files[name] = {**dims, "sha256_inputs": h_in.hexdigest()}
        log(task, f"wrote {path} dims={dims} seed={shape_seed} size={path.stat().st_size} B")
    manifest = {
        "task": task, "seed": seed,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "ref_method": ref_method, "files": files, "test_hash": h_all.hexdigest()[:16],
    }
    write_json_atomic(out_dir / "manifest.json", manifest)
    log(task, f"wrote manifest test_hash={manifest['test_hash']}")
    return manifest


def main_for(
    task: str,
    shapes,
    make_case,
    ref_method: str,
    default_out: Path,
    argv: list[str] | None = None,
    *,
    out_dtype: str = DEFAULT_OUT_DTYPE,
    sentinel: float | int | None = None,
) -> int:
    """``tasks/<task>/gen_tests.py`` 의 CLI 본체. ``out_dtype``·``sentinel`` 은 ``generate`` 로 그대로 넘어간다."""
    import argparse
    parser = argparse.ArgumentParser(description=f"{task} 테스트 입력(npz)과 manifest.json 을 시드 고정으로 생성한다.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=default_out)
    ns = parser.parse_args(argv)
    try:
        generate(task, ns.seed, ns.out.resolve(), shapes, make_case, ref_method, out_dtype=out_dtype, sentinel=sentinel)
    except Exception as exc:
        log(task, f"error: {type(exc).__name__}: {exc}")
        return 1
    return 0
