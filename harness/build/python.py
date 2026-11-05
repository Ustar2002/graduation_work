"""Python "빌드" (설계 §7): ``kernel.py`` 저장 + ``compile()`` 구문 검사만.

임포트 검사는 하지 않는다(오케스트레이터는 LLM 코드를 실행하지 않는다 — 워커 ``--mode import`` 가 한다, §8).
구문 오류 → ``error_kind="compile_error"``, ``error_detail="SyntaxError: ..."``. artifact = ``kernel.py``.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from harness.build.c import BUILD_JSON, BuildResult
from harness.paths import atomic_write_json

_LOG = "[build.python]"

KERNEL_PY = "kernel.py"
COMPILER_NAME = "python"


def build_python(code: str, out_dir: Path) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 ``compile(code, "kernel.py", "exec")`` 로 구문만 검사한다. ``build.json`` 도 저장."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / KERNEL_PY
    artifact.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")

    cc_version = platform.python_version()
    t0 = time.perf_counter()
    error_kind = ""
    error_detail = ""
    log = ""
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
        log = error_detail
    build_time_ms = (time.perf_counter() - t0) * 1000.0

    res = BuildResult(
        ok=not error_kind,
        artifact=artifact if not error_kind else None,
        error_kind=error_kind,
        error_detail=error_detail,
        log=log,
        build_time_ms=build_time_ms,
        command=[COMPILER_NAME, "compile", KERNEL_PY],
        compiler=COMPILER_NAME,
        compiler_version=cc_version,
        n_warnings=0,
    )
    atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
    return res


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
    sys.exit(main())
