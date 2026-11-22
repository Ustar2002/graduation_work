"""``tests/test_perf_summary.py`` 가 읽는 ``*.import.json`` 픽스처를 실제 Numba·Halide 로 다시 만든다.

개발 도구다(테스트는 이 스크립트를 부르지 않는다; 측정 도구도 아니다). 워커 import 모드의 ``--inspect`` 가 결과 문서에 넣는
``inspect`` (설계 v3 §5.1)와 같은 키를 이 디렉터리의 손으로 쓴 커널에서 뽑아 저장한다:

- numba: ``signatures``, ``parallel_diagnostics`` (``Dispatcher.parallel_diagnostics(level=4)`` 의 stdout, 8000자까지),
  ``threading_layer`` (``parallel=True`` 함수를 컴파일한 뒤에야 정해진다; 그 전에는 None), ``num_threads``
- halide: ``loop_nest`` (``Func.print_loop_nest()`` 가 stderr 에 쓰는 텍스트), ``schedule_source`` (오토스케줄러 결과),
  ``autoscheduler``, ``target``, ``autoschedule_ms``, ``jit_ms`` — 파이프라인은 v3 §5.3 대로 만든다(배열 인자는 ``ImageParam``,
  dim 인자는 그 형태의 실제 크기, auto 면 입력·출력 estimate 를 실제 크기로 주고 Mullapudi2016 을 ``parallelism=4`` 로 적용).

커널은 이 저장소 사람이 쓴 픽스처라 이 프로세스에서 직접 임포트한다(LLM 이 쓴 코드는 ``harness.worker`` 밖에서 실행하지 않는다).
``target``·``num_threads``·시간 값과 진단 머리말은 머신마다 다르다. 테스트가 읽는 것은 ``loop_nest``·``schedule_source``·
``parallel_diagnostics`` 의 루프 목록뿐이다.

실행(저장소 루트에서): ``PYTHON tests/fixtures/perf_summary/make_inspect_fixtures.py --write``
``--write`` 없이는 아무것도 쓰지 않는다(추적되는 픽스처를 제자리에서 덮어쓰는 도구라, 옵션 없이·``--help`` 로 실수로 돌려도
파일이 바뀌지 않게 한다).
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:  # 직접 실행해도 harness 패키지를 찾게 한다
    sys.path.insert(0, str(ROOT))

from harness.paths import TASKS_DIR, atomic_write_json  # noqa: E402
from harness.taskspec import TaskSpec, load_taskspec  # noqa: E402

_LOG = "[make_inspect_fixtures]"
AUTOSCHEDULER = "Mullapudi2016"
PARALLELISM = 4
DIAGNOSTICS_CHARS = 8000

# (소스, 과제, halide_mode, dim 값, 출력)
HALIDE_CASES: list[tuple[str, str, str, dict[str, int], str]] = [
    ("halide/gemm_auto.py", "gemm", "auto", {"M": 1024, "N": 1024, "K": 1024}, "halide/gemm_auto.large.import.json"),
    ("halide/gemm_auto.py", "gemm", "auto", {"M": 7, "N": 5, "K": 3}, "halide/gemm_auto.tiny.import.json"),
    ("halide/gemm_manual.py", "gemm", "manual", {"M": 1024, "N": 1024, "K": 1024}, "halide/gemm_manual.import.json"),
    ("halide/gemm_manual_tiled.py", "gemm", "manual", {"M": 1024, "N": 1024, "K": 1024}, "halide/gemm_manual_tiled.import.json"),
    ("halide/jacobi2d_auto.py", "jacobi2d", "auto", {"H": 4096, "W": 4096, "T": 4}, "halide/jacobi2d_auto.import.json"),
    ("halide/jacobi2d_manual.py", "jacobi2d", "manual", {"H": 4096, "W": 4096, "T": 4}, "halide/jacobi2d_manual.import.json"),
    ("halide/dot_auto.py", "dot", "auto", {"n": 1 << 25}, "halide/dot_auto.import.json"),
    ("halide/dot_manual.py", "dot", "manual", {"n": 1 << 25}, "halide/dot_manual.import.json"),
]
# (소스, 과제, 출력). 워커는 샘플마다 새 프로세스다: parallel=True 함수를 컴파일하면 스레딩 계층이 정해지므로, 그 전 상태
# (threading_layer = None)가 나와야 하는 직렬 커널을 맨 앞에 둔다.
NUMBA_CASES: list[tuple[str, str, str]] = [
    ("numba/gemm_blocked.py", "gemm", "numba/gemm_blocked.import.json"),
    ("numba/gemm_prange.py", "gemm", "numba/gemm_prange.import.json"),
    ("numba/dot_prange.py", "dot", "numba/dot_prange.import.json"),
    ("numba/jacobi2d_array_expr.py", "jacobi2d", "numba/jacobi2d_array_expr.import.json"),
]


def load_kernel(source: str, function: str) -> Callable[..., Any]:
    """픽스처 모듈을 워커처럼 파일 경로로 임포트해 과제 함수를 돌려준다."""
    spec = importlib.util.spec_from_file_location("kernel", HERE / source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"{source}: 모듈을 만들 수 없다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function)


def capture_stderr(fn: Callable[[], Any]) -> str:
    """``fn`` 이 fd 2 에 쓰는 것을 가로챈다(Halide 는 C++ 쪽에서 stderr 에 직접 쓴다)."""
    sys.stderr.flush()
    saved = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b") as tmp:
        os.dup2(tmp.fileno(), 2)
        try:
            fn()
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)
        tmp.seek(0)
        return tmp.read().decode("utf-8", errors="replace")


def load_autoscheduler(name: str) -> None:
    """오토스케줄러 플러그인을 절대 경로로 로드한다(이름으로는 로드되지 않는다 — v3 §5.3)."""
    import halide as hl

    lib_dir = Path(hl.__file__).parent
    found = sorted(p for p in lib_dir.rglob(f"*autoschedule_{name.lower()}*") if p.suffix in (".so", ".dll", ".dylib"))
    if not found:
        raise RuntimeError(f"오토스케줄러 플러그인 {name} 을 {lib_dir} 아래에서 찾지 못했다")
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(found[0].parent))
    hl.load_plugin(str(found[0]))


def halide_inspect(source: str, spec: TaskSpec, mode: str, dims: dict[str, int]) -> dict[str, Any]:
    import halide as hl

    build = load_kernel(source, spec.function)
    types = {"float32": hl.Float(32), "int32": hl.Int(32)}
    args: list[Any] = []
    for arg in spec.args:
        if arg.kind == "dim":
            args.append(dims[arg.name])
        elif arg.kind == "in":
            shape = spec.shape_of(arg, dims)
            param = hl.ImageParam(types[str(arg.dtype)], len(shape), arg.name)
            for d, extent in enumerate(reversed(shape)):     # 마지막 축이 Halide 의 첫 좌표다
                param.dim(d).set_estimate(0, extent)
            args.append(param)
    func = build(*args)
    target = hl.get_host_target()
    pipeline = hl.Pipeline(func)
    schedule_source = ""
    autoschedule_ms = None
    if mode == "auto":
        func.set_estimates([hl.Range(0, extent) for extent in reversed(spec.shape_of(spec.output, dims))])
        t0 = time.perf_counter()
        result = pipeline.apply_autoscheduler(target, hl.AutoschedulerParams(AUTOSCHEDULER, {"parallelism": str(PARALLELISM)}))
        autoschedule_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        schedule_source = result.schedule_source
    loop_nest = capture_stderr(func.print_loop_nest)
    t0 = time.perf_counter()
    pipeline.compile_jit(target)
    jit_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "loop_nest": loop_nest, "schedule_source": schedule_source, "autoscheduler": AUTOSCHEDULER if mode == "auto" else None,
        "target": str(target), "autoschedule_ms": autoschedule_ms, "jit_ms": jit_ms,
    }


def numba_inspect(source: str, spec: TaskSpec) -> dict[str, Any]:
    import numba

    fn = load_kernel(source, spec.function)
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(RuntimeError):     # parallel=True 가 아닌 함수에는 진단이 없다
        fn.parallel_diagnostics(level=4)
    try:
        layer = numba.threading_layer()
    except ValueError:              # parallel=True 함수를 컴파일하기 전에는 정해지지 않는다
        layer = None
    return {
        "signatures": [str(sig) for sig in fn.signatures], "parallel_diagnostics": out.getvalue()[:DIAGNOSTICS_CHARS],
        "threading_layer": layer, "num_threads": numba.get_num_threads(),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="perf_summary 의 *.import.json 픽스처를 실제 Numba·Halide 로 다시 만든다(제자리 덮어쓰기)")
    parser.add_argument("--write", action="store_true", help="픽스처 파일을 덮어쓴다(없으면 아무것도 하지 않는다)")
    ns = parser.parse_args(argv)
    if not ns.write:
        targets = [out for *_, out in HALIDE_CASES] + [out for *_, out in NUMBA_CASES]
        print(f"{_LOG} --write 가 없어 아무것도 쓰지 않았다. --write 를 주면 이 디렉터리의 픽스처 {len(targets)}개를 덮어쓴다: "
              + ", ".join(targets), file=sys.stderr)
        return 2
    specs = {task: load_taskspec(TASKS_DIR / task / "signatures.yaml") for task in ("gemm", "jacobi2d", "dot")}
    load_autoscheduler(AUTOSCHEDULER)
    for source, task, mode, dims, out in HALIDE_CASES:
        doc = {"mode": "import", "backend": "halide", "dims": dims, "inspect": halide_inspect(source, specs[task], mode, dims)}
        atomic_write_json(HERE / out, doc)
        print(f"{_LOG} {out}", file=sys.stderr)
    for source, task, out in NUMBA_CASES:
        doc = {"mode": "import", "backend": "numba", "inspect": numba_inspect(source, specs[task])}
        atomic_write_json(HERE / out, doc)
        print(f"{_LOG} {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
