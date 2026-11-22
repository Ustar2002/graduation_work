"""Halide "빌드" (설계 v3 §7.3): ``kernel.py`` 저장 + ``compile()`` 구문 검사 + AST 관찰.

Halide 커널은 파이프라인 빌드 함수(``def {function}(입력 ImageParam…, 크기 int…)`` 가 출력 ``hl.Func`` 를 돌려준다; 설계 v3 §8.3)를
담은 파이썬 모듈이다. 파이프라인 구성·스케줄·JIT 는 워커가 형태마다 한다. 여기서는 코드를 임포트·실행하지 않는다
(``compile()`` 과 ``ast`` 만).

최상위에 ``def {function}`` 이 없으면 ``signature_mismatch``/``function_not_found`` 다(추출 단계를 거쳤으면 나오지 않는다).

``BuildResult.extra`` (구문 오류면 구문 트리가 없으므로 빈 사전)::

    {"schedule_calls": [메서드 이름…],     # 소스에 쓴 스케줄 호출(호출마다 하나, 이름순)
     "imports_halide": bool}              # halide 패키지를 임포트하는가

- 스케줄 호출: 메서드 이름이 ``SCHEDULE_METHODS`` 에 있는 속성 호출(``f.parallel(y)``). 리터럴에 대한 호출(``"a,b".split(",")``,
  f-문자열 포함)과 임포트한 모듈의 함수(``np.tile(…)``, ``os.path.split(…)``)는 뺀다(``harness.perf_summary`` 도 소스를 직접 읽을 때
  문자열 메서드와 모듈 함수를 뺀다). AST 로는 수신 객체의 타입을 알 수 없으므로 그 밖의 호출은 모두 센다 — 변수에 담긴 문자열의
  ``split`` 도 스케줄 호출로 보인다.
- ``halide_mode == "auto"`` (알고리즘만 쓰고 하네스가 오토스케줄러를 적용하는 조건)인데 스케줄 호출이 있어도 빌드는 통과한다 —
  오류가 아니라 기록이다(decisions.md 2026-11-12·2026-11-23). 워커는 모델의 빌드 함수가 도는 동안 스케줄 메서드를 기록만 하는
  무동작으로 바꿔 두고(``inspect.ignored_schedule_calls``), run.py 는 ``extra["schedule_calls"]`` 를 보고 행에 계약 위반 태그
  ``schedule_in_auto`` 를 남긴다. 판정(pass/fail)은 바뀌지 않는다. 즉 ``halide_mode`` 는 빌드 판정을 바꾸지 않는다(값만 확인한다).
- 오토스케줄러(기본 Li2018, ``parallelism`` = 스레드 수)와 그 매개변수 검사는 설정 로더(``harness.config``)와 워커의 일이다.
  빌드 단계는 오토스케줄러를 부르지 않는다.
- ``SCHEDULE_METHODS`` 는 ``harness.perf_summary`` 도 임포트해 쓴다. 그 모듈의 ``DEFAULT_SCHEDULE_METHODS`` 와 같은 집합이어야 한다.
  Halide 22.1 의 ``hl.Func``·``hl.Stage`` 에 있는 스케줄 지시 전부다(``aligned_split``·``split_storage`` 등 22.1 에서 확인한 10개 포함).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from harness.build.c import BuildResult
from harness.build.python import build_python_source, dotted_name, imported_names, package_version, top_level_function

COMPILER_NAME = "halide"
HALIDE_MODES: tuple[str, ...] = ("auto", "manual")
FUNCTION_NOT_FOUND = "function_not_found"

# Halide 의 스케줄 지시 메서드(Func·Stage 의 메서드; Python 바인딩 이름 — 예약어와 겹치는 것은 끝에 밑줄: ``async_``, ``in_``)
SCHEDULE_METHODS: frozenset[str] = frozenset({
    "align_bounds", "align_extent", "align_storage", "allow_race_conditions", "always_partition", "always_partition_all",
    "async_", "atomic", "bound", "bound_extent", "bound_storage", "clone_in", "compute_at", "compute_inline", "compute_root",
    "compute_with", "copy_to_device", "copy_to_host", "fold_storage", "fuse", "gpu", "gpu_blocks", "gpu_lanes",
    "gpu_single_thread", "gpu_threads", "gpu_tile", "hexagon", "hoist_storage", "hoist_storage_root", "in_", "memoize",
    "never_partition", "never_partition_all", "parallel", "partition", "prefetch", "rename", "reorder", "reorder_storage",
    "rfactor", "ring_buffer", "serial", "specialize", "split", "store_at", "store_in", "store_root", "tile", "unroll", "vectorize",
    # Halide 22.1 의 바인딩에서 확인한 나머지 스케줄 지시(halide_auto 에서 무시·기록할 목록과 같다)
    "aligned_split", "eager_inline", "gpu_max_registers", "hoist_invariants", "sme_streaming", "specialize_fail", "split_storage",
    "stream_loads", "stream_stores", "unscheduled",
})


def schedule_calls(tree: ast.Module) -> list[str]:
    """소스에 쓴 스케줄 호출의 메서드 이름(호출마다 하나, 이름순). 리터럴과 임포트한 모듈에 대한 호출은 스케줄이 아니다."""
    imports = imported_names(tree)
    names: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in SCHEDULE_METHODS):
            continue
        receiver = node.func.value
        module_path = dotted_name(receiver)
        if isinstance(receiver, (ast.Constant, ast.JoinedStr)) or (module_path is not None and module_path[0] in imports):
            continue
        names.append(node.func.attr)
    return sorted(names)


def imports_halide(tree: ast.Module) -> bool:
    """``import halide [as hl]`` 또는 ``from halide import …`` 가 있는가."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            modules = [node.module or ""]
        else:
            continue
        if any(m.split(".")[0] == "halide" for m in modules):
            return True
    return False


def inspect_halide(tree: ast.Module, function: str) -> tuple[str, str, dict[str, Any]]:
    """Halide 커널의 AST 관찰 → ``(error_kind, error_detail, extra)``. 최상위에 과제 함수가 없으면 ``signature_mismatch``."""
    extra: dict[str, Any] = {"schedule_calls": schedule_calls(tree), "imports_halide": imports_halide(tree)}
    if top_level_function(tree, function) is None:
        return "signature_mismatch", FUNCTION_NOT_FOUND, extra
    return "", "", extra


def build_halide(code: str, out_dir: Path, *, function: str, halide_mode: str) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 구문을 검사한 뒤 스케줄 호출·임포트를 ``extra`` 에 적는다. ``build.json`` 도 저장.

    ``halide_mode`` ∈ auto | manual. 어느 쪽이든 판정은 같다(모듈 docstring).
    """
    if halide_mode not in HALIDE_MODES:
        raise ValueError(f"알 수 없는 halide_mode: {halide_mode!r} (허용: {HALIDE_MODES})")
    return build_python_source(
        code, out_dir, compiler=COMPILER_NAME, compiler_version=package_version("halide"),
        inspect=lambda tree: inspect_halide(tree, function),
    )


def main(argv: list[str] | None = None) -> int:
    """Halide 커널 소스 파일 하나를 검사하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.halide", description="Halide 커널 구문 검사·스케줄 호출 관찰 (설계 v3 §7.3)")
    parser.add_argument("--src", type=Path, required=True, help="Python 소스 파일")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉터리(kernel.py/build.json)")
    parser.add_argument("--function", required=True, help="과제 함수 이름")
    parser.add_argument("--halide-mode", choices=HALIDE_MODES, required=True)
    ns = parser.parse_args(argv)
    res = build_halide(ns.src.read_text(encoding="utf-8"), ns.out_dir, function=ns.function, halide_mode=ns.halide_mode)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    # 컴파일러 출력·경로·한국어 진단을 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
