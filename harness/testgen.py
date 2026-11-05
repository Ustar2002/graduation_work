"""과제 공통 테스트 입력 생성 헬퍼 (M2). ``tasks/<task>/gen_tests.py`` 가 쓴다(gemm 은 M1 의 자체 구현을 유지).

- 형태마다 독립 시드 ``seed + index``. npz 키: 입력 배열(인자 이름), ``<out>_ref``(float64), dim 스칼라, ``seed``.
- ``manifest.json`` 의 ``test_hash`` 는 입력만으로 정의한다: sha256(str(seed) ‖ 형태마다 "k=v,…" ‖ 입력 바이트(인자 순서))[:16].
- 같은 시드로 다시 만들면 npz 는 바이트 단위로 같다. 파일은 원자적으로 쓴다.
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
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def generate(
    task: str,
    seed: int,
    out_dir: Path,
    shapes: list[tuple[str, dict[str, int]]],
    make_case: Callable[[dict[str, int], np.random.Generator], tuple[dict[str, int], dict[str, np.ndarray], str, np.ndarray]],
    ref_method: str,
) -> dict:
    """``make_case(shape, rng) -> (dims, inputs, out_name, ref64)``. ``dims`` 는 npz 에 쓰는 dim 스칼라(인자 이름),
    ``inputs`` 는 인자 순서의 입력 배열. 형태 이름은 ``TEST_NAMES`` 와 같아야 한다."""
    if [n for n, _ in shapes] != list(TEST_NAMES):
        raise ValueError(f"형태 이름은 {TEST_NAMES} 순서여야 한다: {[n for n, _ in shapes]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    h_all = hashlib.sha256(str(seed).encode("ascii"))
    files: dict[str, dict] = {}
    for index, (name, shape) in enumerate(shapes):
        shape_seed = seed + index
        dims, inputs, out_name, ref = make_case(shape, np.random.default_rng(shape_seed))
        if ref.dtype != np.float64:
            raise RuntimeError(f"{task}/{name}: 참조는 float64 여야 한다({ref.dtype})")
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


def main_for(task: str, shapes, make_case, ref_method: str, default_out: Path, argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=f"{task} 테스트 입력(npz)과 manifest.json 을 시드 고정으로 생성한다.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out", type=Path, default=default_out)
    ns = parser.parse_args(argv)
    try:
        generate(task, ns.seed, ns.out.resolve(), shapes, make_case, ref_method)
    except Exception as exc:
        log(task, f"error: {type(exc).__name__}: {exc}")
        return 1
    return 0
