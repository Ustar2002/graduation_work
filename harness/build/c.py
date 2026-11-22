"""C 빌드 (설계 §7, v3 §6): ``kernel.c`` → ``kernel.so`` 컴파일, ``nm`` 심벌 검사, ``build.json`` 기록.

- 명령: ``launch_prefix + [cc, *base_flags, *cflags, *extra_flags, (openmp_flag), *diag_flags, "-o", artifact_name, "kernel.c",
  *ldflags]`` (cwd = out_dir; ``launch_prefix`` 는 taskset 접두 — P-core 에서 컴파일해 ``-march=native`` 해석을 고정한다).
  기본 인자(``extra_flags`` 없음, ``artifact_name="kernel.so"``)에서는 v3 이전과 같은 명령이다.
- ``args_header`` (keyword 모드의 매개변수 구조체 헤더 전문)가 있으면 ``out_dir/kernel_args.h`` 로 쓴다.
- ``extra_flags`` 에 ``-fsanitize=`` 플래그가 있으면(ASan/UBSan 빌드) 링크 플래그에서 ``-Wl,--no-undefined`` 를 뺀다 — sanitizer
  런타임 심벌은 로드 시점에 선행 로드된 libasan 이 채울 수 있고, 미정의 심벌 검사는 일반 빌드가 이미 했다.
- 컴파일러·``nm`` 실행 파일을 시작할 수 없으면 ``ToolchainMissing`` 을 던진다. 없는 도구가 ``compile_error`` 행이 되면 안 된다
  (``taskset`` 접두가 있으면 없는 컴파일러가 예외가 아니라 ``taskset`` 의 rc=127 로 나타나므로 실행 전에 PATH 를 본다).
  ``nm`` 이 실패했을 때(rc != 0, 시간 초과)도 같다 — 심벌 표를 읽지 못한 것을 "심벌이 없다"로 기록하지 않는다.
- ``run_isolated(timeout=compile_timeout_s, stderr_path=build.log)``. 실패 시 ``error_detail`` 은 ``error:`` 를 포함한 첫 3줄.
  타임아웃은 ``error_kind="timeout"`` (run.py 가 ``compile_error``/``compile_timeout`` 으로 기록).
- 성공 시 ``nm -D --defined-only`` 에서 세 번째 토큰이 정확히 심벌 이름이고 타입 ``T``/``W``/``i`` 인 줄이 없으면
  ``signature_mismatch``/``symbol gemm not exported``. ``nm -D --undefined-only`` 의 ``U`` 심벌이 ``undefined_symbols``
  (버전 접미 ``@GLIBC_…`` 제거); 그중 ``forbidden_symbols`` 일치 → ``forbidden_symbols``, ``thread_symbols_prefix`` 접두 일치 → ``thread_symbols``.
- ``BuildResult`` 는 모든 빌드 백엔드(c·python·numba·halide·rust)의 공통 결과형이다. ``extra`` 는 백엔드별 관찰값.
- 이 모듈은 LLM 코드를 실행하지 않는다(gcc/nm 만).
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from harness.config import BuildCfg, ConfigError, load_config
from harness.paths import atomic_write_json, atomic_write_text
from harness.procutil import run_isolated
from harness.protocol import compiler_version, launch_env_minimal, launch_prefix, resolve_march
from harness.sigs import C_ARGS_HEADER_NAME

_LOG = "[build.c]"

KERNEL_SRC = "kernel.c"
KERNEL_SO = "kernel.so"
BUILD_LOG = "build.log"
BUILD_JSON = "build.json"
DEFAULT_SYMBOL = "gemm"
NM = "nm"
NM_TIMEOUT_S = 30.0
NO_UNDEFINED_FLAG = "-Wl,--no-undefined"
SANITIZE_FLAG_PREFIX = "-fsanitize="

_ERROR_LINE_RE = re.compile(r"\berror:")
_WARNING_LINE_RE = re.compile(r"\bwarning:")
_NM_TAIL_CHARS = 1_000_000


class ToolchainMissing(RuntimeError):
    """빌드 도구(컴파일러, ``nm``, cargo)를 시작할 수 없거나 쓸 수 없는 상태다. 생성된 코드의 잘못이 아니다 —
    호출자는 이것을 모델 오류(``compile_error``)로 기록하지 않고 행을 쓰기 전에 멈춘다(설계 v3 §10.6)."""


@dataclass
class BuildResult:
    """빌드 결과(설계 §7). ``error_kind`` ∈ "" | compile_error | signature_mismatch | timeout.

    ``extra`` 는 백엔드별 관찰값(설계 v3 §6; JSON 가능한 값만)으로 ``build.json`` 에 함께 저장된다.
    """

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
    extra: dict[str, Any] = field(default_factory=dict)

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


def compile_command(
    bcfg: BuildCfg, *, openmp: bool, extra_flags: list[str] | None = None, artifact_name: str = KERNEL_SO
) -> list[str]:
    """taskset 접두를 뺀 컴파일 명령(cwd = out_dir 기준의 상대 파일 이름).

    ``extra_flags`` 는 ``cflags`` 바로 뒤에 붙는다. 그 안에 ``-fsanitize=`` 플래그가 있으면 링크 플래그에서
    ``-Wl,--no-undefined`` 를 뺀다(모듈 docstring).
    """
    extra = list(extra_flags or [])
    cmd = [bcfg.cc, *bcfg.base_flags, *bcfg.cflags, *extra]
    if openmp:
        cmd.append(bcfg.openmp_flag)
    ldflags = list(bcfg.ldflags)
    if any(flag.startswith(SANITIZE_FLAG_PREFIX) for flag in extra):
        ldflags = [flag for flag in ldflags if flag != NO_UNDEFINED_FLAG]
    cmd += [*bcfg.diag_flags, "-o", artifact_name, KERNEL_SRC, *ldflags]
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


# 내보낸 함수로 보는 ``nm`` 심벌 타입: 전역 코드(T), 약한 심벌(W), GNU 간접 함수(i — ``target_clones``·``ifunc`` 가 만드는
# STT_GNU_IFUNC; 동적 링커·ctypes 가 해석기를 거쳐 정상적으로 부른다). 대문자 ``I`` 는 간접 참조이지 정의된 함수가 아니다.
EXPORTED_SYMBOL_TYPES = frozenset({"T", "W", "i"})


def symbol_exported(nm_defined_text: str, symbol: str = DEFAULT_SYMBOL) -> bool:
    """정의 심벌 목록에서 이름이 정확히 ``symbol`` 이고 타입이 ``T``/``W``/``i`` (``EXPORTED_SYMBOL_TYPES``)인 줄이 있는가."""
    for sym_type, name in parse_nm_lines(nm_defined_text):
        if strip_symbol_version(name) == symbol and sym_type in EXPORTED_SYMBOL_TYPES:
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


def nm_dynamic(out_dir: Path, mode_flag: str, artifact_name: str = KERNEL_SO) -> str:
    """``nm -D <mode_flag> <artifact_name>`` 의 stdout.

    nm 을 시작할 수 없거나 nm 이 실패하면(rc != 0, 시간 초과) ``ToolchainMissing`` 이다. 심벌 표를 읽지 못한 것은 하네스가
    판정하지 못한 것이지 커널이 심벌을 내보내지 않은 것이 아니다 — 빈 출력으로 넘기면 ``symbol … not exported`` 행이 된다.
    (내보낸 심벌이 하나도 없는 라이브러리에서 GNU nm 은 ``no symbols`` 를 알리고 rc=0 으로 끝난다: 그것은 실패가 아니다.)
    """
    cmd = [NM, "-D", mode_flag, artifact_name]
    try:
        r = run_isolated(cmd, timeout_s=NM_TIMEOUT_S, cwd=out_dir, env=launch_env_minimal(), tail_chars=_NM_TAIL_CHARS)
    except OSError as e:
        raise ToolchainMissing(f"{NM} 을 실행할 수 없다: {e}") from e
    if r.timed_out or r.rc != 0:
        why = f"{NM_TIMEOUT_S:g}s 안에 끝나지 않았다" if r.timed_out else f"rc={r.rc}"
        raise ToolchainMissing(f"`{' '.join(cmd)}` 실패({why}): {r.stderr_tail.strip()[:200]}")
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
    args_header: str | None = None,
    extra_flags: list[str] | None = None,
    artifact_name: str = KERNEL_SO,
    write_json: bool = True,
) -> BuildResult:
    """``out_dir/kernel.c`` 를 쓰고 공유 라이브러리(``artifact_name``)로 컴파일한 뒤 심벌을 검사한다.

    - ``args_header``: 있으면 ``out_dir/kernel_args.h`` 로 쓴다(keyword 모드의 매개변수 구조체).
    - ``extra_flags``: ``cflags`` 뒤에 붙는다(``compile_command``; ``-fsanitize=`` 가 있으면 ``-Wl,--no-undefined`` 제외).
    - ``write_json``: 결과를 ``out_dir/build.json`` 에도 저장한다.

    컴파일러를 시작할 수 없거나 ``nm`` 을 쓸 수 없으면(없음·실패) ``ToolchainMissing`` 을 던진다(결과도 ``build.json`` 도 없다).
    """
    if shutil.which(bcfg.cc) is None:
        raise ToolchainMissing(f"C 컴파일러 {bcfg.cc!r} 를 찾을 수 없다(PATH)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src = out_dir / KERNEL_SRC
    so = out_dir / artifact_name
    log_path = out_dir / BUILD_LOG
    if so.exists():
        so.unlink()
    src.write_text(code if code.endswith("\n") else code + "\n", encoding="utf-8")
    if args_header is not None:
        atomic_write_text(out_dir / C_ARGS_HEADER_NAME, args_header)

    cmd = [*launch_prefix, *compile_command(bcfg, openmp=openmp, extra_flags=extra_flags, artifact_name=artifact_name)]
    cc_version = cached_compiler_version(bcfg.cc)
    march = cached_march_resolved(bcfg.cc)

    t0 = time.perf_counter()
    try:
        proc = run_isolated(cmd, timeout_s=bcfg.compile_timeout_s, cwd=out_dir, env=launch_env_minimal(), stderr_path=log_path)
    except OSError as e:
        raise ToolchainMissing(f"컴파일 명령을 시작할 수 없다({cmd[0]}): {e}") from e
    build_time_ms = (time.perf_counter() - t0) * 1000.0
    log = _read_log(log_path)

    def _result(ok: bool, kind: str = "", detail: str = "", **symbols: list[str]) -> BuildResult:
        res = BuildResult(
            ok=ok, artifact=so if ok else None, error_kind=kind, error_detail=detail, log=log, build_time_ms=build_time_ms,
            command=cmd, compiler=bcfg.cc, compiler_version=cc_version, n_warnings=count_warnings(log), march_resolved=march,
            **symbols,
        )
        if write_json:
            atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
        return res

    if proc.timed_out:
        return _result(False, "timeout", f"compile timeout after {bcfg.compile_timeout_s:g}s")
    if proc.rc != 0 or not so.exists():
        detail = " | ".join(error_lines(log)) or f"rc={proc.rc}"
        if proc.signal_name:
            detail = f"compiler died with {proc.signal_name}: {detail}"
        return _result(False, "compile_error", detail)

    defined = nm_dynamic(out_dir, "--defined-only", artifact_name)
    undefined = undefined_symbols_of(nm_dynamic(out_dir, "--undefined-only", artifact_name))
    symbols = {
        "undefined_symbols": undefined,
        "forbidden_symbols": [s for s in undefined if s in bcfg.forbidden_symbols],
        "thread_symbols": [s for s in undefined if any(s.startswith(p) for p in bcfg.thread_symbols_prefix)],
    }
    if not symbol_exported(defined, symbol):
        return _result(False, "signature_mismatch", f"symbol {symbol} not exported", **symbols)
    return _result(True, **symbols)


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
    try:
        res = build_c(code, ns.out_dir, cfg.build, openmp=ns.openmp, launch_prefix=prefix)
    except ToolchainMissing as e:
        print(f"{_LOG} 도구 없음: {e}", file=sys.stderr)
        return 2
    d = res.to_dict()
    d["log"] = d["log"][-2000:]
    print(json.dumps(d, ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    # 컴파일러 출력·경로·한국어 진단을 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
