"""C 빌드 (설계 §7): ``kernel.c`` → ``kernel.so`` 컴파일, ``nm`` 심벌 검사, ``build.json`` 기록.

- 명령: ``launch_prefix + [cc, *base_flags, *cflags, (openmp_flag), *diag_flags, "-o", "kernel.so", "kernel.c", *ldflags]``
  (cwd = out_dir; ``launch_prefix`` 는 taskset 접두 — P-core 에서 컴파일해 ``-march=native`` 해석을 고정한다).
- ``run_isolated(timeout=compile_timeout_s, stderr_path=build.log)``. 실패 시 ``error_detail`` 은 ``error:`` 를 포함한 첫 3줄.
  타임아웃은 ``error_kind="timeout"`` (run.py 가 ``compile_error``/``compile_timeout`` 으로 기록).
- 성공 시 ``nm -D --defined-only`` 에서 세 번째 토큰이 정확히 심벌 이름이고 타입 ``T``/``W`` 인 줄이 없으면
  ``signature_mismatch``/``symbol gemm not exported``. ``nm -D --undefined-only`` 의 ``U`` 심벌이 ``undefined_symbols``
  (버전 접미 ``@GLIBC_…`` 제거); 그중 ``forbidden_symbols`` 일치 → ``forbidden_symbols``, ``thread_symbols_prefix`` 접두 일치 → ``thread_symbols``.
- 이 모듈은 LLM 코드를 실행하지 않는다(gcc/nm 만).
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.config import BuildCfg, ConfigError, load_config
from harness.paths import atomic_write_json
from harness.procutil import run_isolated
from harness.protocol import compiler_version, launch_env_minimal, launch_prefix, resolve_march

_LOG = "[build.c]"

KERNEL_SRC = "kernel.c"
KERNEL_SO = "kernel.so"
BUILD_LOG = "build.log"
BUILD_JSON = "build.json"
DEFAULT_SYMBOL = "gemm"

_ERROR_LINE_RE = re.compile(r"\berror:")
_WARNING_LINE_RE = re.compile(r"\bwarning:")
_NM_TAIL_CHARS = 1_000_000


@dataclass
class BuildResult:
    """빌드 결과(설계 §7). ``error_kind`` ∈ "" | compile_error | signature_mismatch | timeout."""

    ok: bool
    artifact: Path | None
    error_kind: str
    error_detail: str
    log: str
    build_time_ms: float
    command: list[str]
    compiler: str
    compiler_version: str
    n_warnings: int
    undefined_symbols: list[str] = field(default_factory=list)
    forbidden_symbols: list[str] = field(default_factory=list)
    thread_symbols: list[str] = field(default_factory=list)
    march_resolved: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON 가능한 dict(``artifact`` 는 문자열)."""
        d = asdict(self)
        d["artifact"] = str(self.artifact) if self.artifact is not None else None
        return d


# ----------------------------------------------------------------------------- 헬퍼


@functools.lru_cache(maxsize=None)
def cached_compiler_version(cc: str) -> str:
    """``cc --version`` 첫 줄(모듈 캐시). 없으면 ``""``."""
    return compiler_version(cc) or ""


@functools.lru_cache(maxsize=None)
def cached_march_resolved(cc: str) -> str:
    """``cc -march=native -Q --help=target`` 의 ``-march=``/``-mtune=`` 해석값(모듈 캐시). 없으면 ``""``."""
    return resolve_march(cc) or ""


def compile_command(bcfg: BuildCfg, *, openmp: bool) -> list[str]:
    """taskset 접두를 뺀 컴파일 명령(cwd = out_dir 기준의 상대 파일 이름)."""
    cmd = [bcfg.cc, *bcfg.base_flags, *bcfg.cflags]
    if openmp:
        cmd.append(bcfg.openmp_flag)
    cmd += [*bcfg.diag_flags, "-o", KERNEL_SO, KERNEL_SRC, *bcfg.ldflags]
    return cmd


def strip_symbol_version(name: str) -> str:
    """``memset@GLIBC_2.2.5`` → ``memset`` (``@@`` 도 처리)."""
    return name.split("@", 1)[0]


def parse_nm_lines(text: str) -> list[tuple[str, str]]:
    """``nm -D`` 출력을 ``(type, name)`` 리스트로. 주소 유무(2 또는 3 토큰)를 모두 처리한다."""
    out: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            out.append((parts[1], parts[2]))
        elif len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def symbol_exported(nm_defined_text: str, symbol: str = DEFAULT_SYMBOL) -> bool:
    """정의 심벌 목록에서 이름이 정확히 ``symbol`` 이고 타입이 ``T``/``W`` 인 줄이 있는가."""
    for sym_type, name in parse_nm_lines(nm_defined_text):
        if strip_symbol_version(name) == symbol and sym_type in ("T", "W"):
            return True
    return False


def undefined_symbols_of(nm_undefined_text: str) -> list[str]:
    """``U`` 타입 미정의 심벌 이름(버전 접미 제거, 중복 제거, 등장 순)."""
    seen: list[str] = []
    for sym_type, name in parse_nm_lines(nm_undefined_text):
        if sym_type != "U":
            continue
        bare = strip_symbol_version(name)
        if bare not in seen:
            seen.append(bare)
    return seen


def error_lines(log: str, limit: int = 3) -> list[str]:
    """``error:`` 를 포함한 첫 ``limit`` 줄. 없으면 마지막 비어 있지 않은 ``limit`` 줄."""
    errs = [ln.strip() for ln in log.splitlines() if _ERROR_LINE_RE.search(ln)]
    if errs:
        return errs[:limit]
    tail = [ln.strip() for ln in log.splitlines() if ln.strip()]
    return tail[-limit:]


def count_warnings(log: str) -> int:
    """``warning:`` 줄 수."""
    return sum(1 for ln in log.splitlines() if _WARNING_LINE_RE.search(ln))


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _nm(out_dir: Path, mode_flag: str) -> str:
    """``nm -D <mode_flag> kernel.so`` 의 stdout. nm 실패는 빈 문자열 + 경고."""
    r = run_isolated(
        ["nm", "-D", mode_flag, KERNEL_SO],
        timeout_s=30,
        cwd=out_dir,
        env=launch_env_minimal(),
        tail_chars=_NM_TAIL_CHARS,
    )
    if r.rc != 0:
        print(f"{_LOG} 경고: nm {mode_flag} 실패(rc={r.rc}): {r.stderr_tail.strip()[:200]}", file=sys.stderr)
        return ""
    return r.stdout_tail


# ----------------------------------------------------------------------------- 빌드


def build_c(
    code: str,
    out_dir: Path,
    bcfg: BuildCfg,
    *,
    openmp: bool,
    launch_prefix: list[str],
    symbol: str = DEFAULT_SYMBOL,
) -> BuildResult:
    """``out_dir/kernel.c`` 를 쓰고 공유 라이브러리로 컴파일한 뒤 심벌을 검사한다. 결과는 ``out_dir/build.json`` 에도 저장."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / KERNEL_SRC
    so = out_dir / KERNEL_SO
    log_path = out_dir / BUILD_LOG
    if so.exists():
        so.unlink()
    src.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")

    cmd = [*launch_prefix, *compile_command(bcfg, openmp=openmp)]
    cc_version = cached_compiler_version(bcfg.cc)
    march = cached_march_resolved(bcfg.cc)

    t0 = time.perf_counter()
    proc = run_isolated(cmd, timeout_s=bcfg.compile_timeout_s, cwd=out_dir, env=launch_env_minimal(), stderr_path=log_path)
    build_time_ms = (time.perf_counter() - t0) * 1000.0
    log = _read_log(log_path)
    n_warnings = count_warnings(log)

    def _fail(kind: str, detail: str) -> BuildResult:
        res = BuildResult(
            ok=False, artifact=None, error_kind=kind, error_detail=detail, log=log, build_time_ms=build_time_ms,
            command=cmd, compiler=bcfg.cc, compiler_version=cc_version, n_warnings=n_warnings, march_resolved=march,
        )
        atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
        return res

    if proc.timed_out:
        return _fail("timeout", f"compile timeout after {bcfg.compile_timeout_s:g}s")
    if proc.rc != 0 or not so.exists():
        detail = " | ".join(error_lines(log)) or f"rc={proc.rc}"
        if proc.signal_name:
            detail = f"compiler died with {proc.signal_name}: {detail}"
        return _fail("compile_error", detail)

    defined = _nm(out_dir, "--defined-only")
    undefined_text = _nm(out_dir, "--undefined-only")
    undefined = undefined_symbols_of(undefined_text)
    forbidden = [s for s in undefined if s in bcfg.forbidden_symbols]
    threads = [s for s in undefined if any(s.startswith(p) for p in bcfg.thread_symbols_prefix)]

    if not symbol_exported(defined, symbol):
        res = BuildResult(
            ok=False, artifact=None, error_kind="signature_mismatch", error_detail=f"symbol {symbol} not exported",
            log=log, build_time_ms=build_time_ms, command=cmd, compiler=bcfg.cc, compiler_version=cc_version,
            n_warnings=n_warnings, undefined_symbols=undefined, forbidden_symbols=forbidden, thread_symbols=threads,
            march_resolved=march,
        )
        atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
        return res

    res = BuildResult(
        ok=True, artifact=so, error_kind="", error_detail="", log=log, build_time_ms=build_time_ms, command=cmd,
        compiler=bcfg.cc, compiler_version=cc_version, n_warnings=n_warnings, undefined_symbols=undefined,
        forbidden_symbols=forbidden, thread_symbols=threads, march_resolved=march,
    )
    atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
    return res


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """C 소스 파일 하나를 설정대로 빌드하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.c", description="C 커널 빌드 (설계 §7)")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--src", type=Path, required=True, help="C 소스 파일")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉터리(kernel.c/kernel.so/build.log/build.json)")
    parser.add_argument("--openmp", action="store_true", help="-fopenmp 포함")
    parser.add_argument("--no-taskset", action="store_true", help="taskset 접두 없이 컴파일")
    ns = parser.parse_args(argv)
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 2
    prefix = [] if ns.no_taskset else launch_prefix(cfg, cfg.measurement.threads if ns.openmp else 1)
    code = ns.src.read_text(encoding="utf-8")
    res = build_c(code, ns.out_dir, cfg.build, openmp=ns.openmp, launch_prefix=prefix)
    d = res.to_dict()
    d["log"] = d["log"][-2000:]
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
