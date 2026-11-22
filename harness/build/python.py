"""Python "빌드" (설계 §7): ``kernel.py`` 저장 + ``compile()`` 구문 검사만.

임포트 검사는 하지 않는다(오케스트레이터는 LLM 코드를 실행하지 않는다 — 워커 ``--mode import`` 가 한다, §8).
구문 오류 → ``error_kind="compile_error"``, ``error_detail="SyntaxError: ..."``. artifact = ``kernel.py``.

파이썬 소스를 쓰는 다른 백엔드(``harness.build.numba``, ``harness.build.halide``; 설계 v3 §7.2·§7.3)도 ``build_python_source`` 로
같은 저장·구문 검사를 하고, 그 위에 AST 검사(``inspect``)의 판정과 관찰값(``BuildResult.extra``)을 얹는다.
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Callable

from harness.build.c import BUILD_JSON, BuildResult
from harness.paths import atomic_write_json

_LOG = "[build.python]"

KERNEL_PY = "kernel.py"
COMPILER_NAME = "python"

# AST 검사: 구문 트리 → (error_kind, error_detail, extra). 통과면 error_kind == "".
Inspector = Callable[[ast.Module], tuple[str, str, dict[str, Any]]]


def package_version(dist: str) -> str:
    """설치된 배포판의 버전(임포트 없이 메타데이터만 읽는다). 없으면 ``""``."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return ""


def imported_names(tree: ast.Module) -> dict[str, str]:
    """임포트로 묶인 이름 → 전체 이름: ``import numpy as np`` → ``{"np": "numpy"}``,
    ``from numba import njit`` → ``{"njit": "numba.njit"}``. ``from m import *`` 는 이름을 묶지 않는 것으로 본다."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                out[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if alias.name != "*":
                    out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def dotted_name(node: ast.AST) -> list[str] | None:
    """``nb.njit`` → ``["nb", "njit"]``, ``njit`` → ``["njit"]``. 이름과 속성 접근만으로 된 식이 아니면 None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return parts[::-1]


def top_level_function(tree: ast.Module, function: str) -> ast.FunctionDef | None:
    """최상위 ``def {function}``. 같은 이름이 여러 번이면 마지막 정의가 유효하다(파이썬 의미론). ``async def`` 는 치지 않는다."""
    found: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function:
            found = node
    return found


def build_python_source(
    code: str, out_dir: Path, *, compiler: str, compiler_version: str, inspect: Inspector | None = None
) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 ``compile(code, "kernel.py", "exec")`` 로 구문을 검사한 뒤, 통과하면 ``inspect`` (AST 검사)의
    판정과 관찰값을 더한다. ``build.json`` 도 저장한다. 코드는 실행하지 않는다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / KERNEL_PY
    artifact.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")

    t0 = time.perf_counter()
    error_kind = ""
    error_detail = ""
    extra: dict[str, Any] = {}
    try:
        compile(code, KERNEL_PY, "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as e:
        # SyntaxError 는 줄 번호를 함께 적는다. ValueError 는 널 바이트 등 소스 자체의 문제다.
        error_kind = "compile_error"
        if isinstance(e, SyntaxError):
            loc = f" (line {e.lineno})" if e.lineno is not None else ""
            error_detail = f"{type(e).__name__}: {e.msg}{loc}"
        else:
            error_detail = f"{type(e).__name__}: {e}"
    else:
        if inspect is not None:
            error_kind, error_detail, extra = inspect(ast.parse(code))
    build_time_ms = (time.perf_counter() - t0) * 1000.0

    res = BuildResult(
        ok=not error_kind,
        artifact=artifact if not error_kind else None,
        error_kind=error_kind,
        error_detail=error_detail,
        log=error_detail,
        build_time_ms=build_time_ms,
        command=[COMPILER_NAME, "compile", KERNEL_PY],
        compiler=compiler,
        compiler_version=compiler_version,
        n_warnings=0,
        extra=extra,
    )
    atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
    return res


def build_python(code: str, out_dir: Path) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 ``compile(code, "kernel.py", "exec")`` 로 구문만 검사한다. ``build.json`` 도 저장."""
    return build_python_source(code, out_dir, compiler=COMPILER_NAME, compiler_version=platform.python_version())


def main(argv: list[str] | None = None) -> int:
    """Python 소스 파일 하나를 검사하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.python", description="Python 커널 구문 검사 (설계 §7)")
    parser.add_argument("--src", type=Path, required=True, help="Python 소스 파일")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉터리(kernel.py/build.json)")
    ns = parser.parse_args(argv)
    res = build_python(ns.src.read_text(encoding="utf-8"), ns.out_dir)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    # 컴파일러 출력·경로·한국어 진단을 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
