"""Rust 빌드 (설계 v3 §6): cargo 로 ``cdylib`` 를 만들고, 워커는 C 와 같은 ctypes 경로(``--backend c``)로 부른다.

- 파일: ``out_dir/Cargo.toml``·``Cargo.lock`` (``rust_template/`` 의 사본 — 의존성은 버전을 고정한 ``rayon`` 하나),
  ``out_dir/src/lib.rs`` (생성된 커널), keyword 모드면 ``out_dir/src/kernel_args.rs`` (매개변수 구조체; ``args_module``).
- 명령: ``launch_prefix + [cargo, "build", "--release", "--offline", "--locked"]`` (cwd = out_dir). ``RUSTFLAGS`` 는 설정의
  ``rustflags`` 를 공백으로 이은 것, ``CARGO_TARGET_DIR`` 는 ``target_dir`` (기본 ``out_dir/target``). stderr 는 ``build.log``.
  ``offline: false`` 설정이면 ``--offline`` 을 뺀다.
- 산출물: ``<target_dir>/release/libkernel.so`` 를 ``out_dir/kernel.so`` 로 복사한다(Windows 개발 머신: ``kernel.dll``).
- 심벌 검사: 산출물이 ELF 이고 ``nm`` 이 있으면 C 와 같은 ``nm -D --defined-only`` 검사 — 심벌이 없으면
  ``signature_mismatch``/``symbol {symbol} not exported``. 할 수 없으면 건너뛰고 ``extra["symbol_check"]`` 에 이유를 적는다.
- 실패한 빌드의 ``error_detail``: rustc 오류의 첫 3개(``error[E0425]: … --> src/lib.rs:LINE:COL``). 타임아웃은 ``error_kind="timeout"``.
- ``BuildResult``: ``compiler="rustc"``, ``compiler_version`` = ``rustc --version`` 첫 줄,
  ``extra = {"rustflags": [...], "symbol_check": "nm" | "skipped:not_elf" | "skipped:nm_not_found"}``.
- 이 모듈은 LLM 코드를 실행하지 않는다(cargo/rustc/nm 만). 커널에 ``build.rs`` 는 없고 의존성의 빌드 스크립트는 고정된 크레이트의 것이다.

**target 디렉터리를 샘플끼리 공유할 때**(run.py 가 run 당 하나를 넘겨 의존성 빌드를 재사용한다): 샘플 디렉터리가 달라도 패키지
이름이 같아 cargo 의 fingerprint 와 출력 파일(``release/deps/libkernel.so``)이 하나로 겹치고, cargo 는 소스의 mtime 으로만 신선도를
판단한다. 실측(cargo 1.92): 다른 샘플을 빌드한 뒤 mtime 이 오래된 샘플 디렉터리에서 빌드하면 컴파일 오류가 있는 소스인데도
rc=0 으로 끝나고 남아 있는 라이브러리는 **다른 샘플의 것**이다. 그래서 빌드 전에 항상 ``cargo clean --release -p kernel`` 로
커널 패키지의 산출물만 지운다(의존성은 남는다) — 복사하는 라이브러리가 지금 소스에서 나온 것임이 시계와 무관하게 보장된다.

**잠금**: 여러 프로세스가 같은 target 디렉터리를 쓰면(같은 run id 의 harness.run·harness.repair 두 개, 병렬 devcheck) 한쪽의
정리·빌드 사이에 다른 쪽이 커널 패키지를 지우거나 덮어써서 남의 라이브러리를 복사할 수 있다(실측: 'symbol nbody not found',
'산출물 없음'). 그래서 정리 → 빌드 → 산출물 읽기 전체를 ``<target>/.harness-build.lock`` 의 프로세스 간 배타 잠금 아래에서 한다
(Windows ``msvcrt.locking``, POSIX ``fcntl.flock``). 대기 상한은 ``LOCK_WAIT_FACTOR × (정리 + 빌드 시간 초과)`` 이고, 넘기면
``ToolchainMissing`` 이다(샘플 코드와 무관하다 — ``compile_error`` 행이 되지 않는다).

**툴체인 점검** ``probe_toolchain``: run 시작 전에 템플릿의 자리표시 크레이트를 같은 명령·환경·잠금으로 한 번 빌드한다. 알려진 정상
코드이므로 어떤 실패든 환경 문제(``ToolchainMissing``)이고, 성공하면 의존성 빌드가 target 디렉터리에 남아 샘플이 재사용한다.

**모델 오류가 아닌 실패는 ``ToolchainMissing``** (``harness.build.c``): cargo 를 시작할 수 없을 때, 그리고 cargo 가 커널 크레이트를
컴파일하기 전에 실패하거나 시간을 다 썼을 때(의존성 캐시 없음 — ``cargo fetch --locked --manifest-path
harness/build/rust_template/Cargo.toml`` 를 한 번 실행해야 한다 —, 잠금 파일 불일치, 의존성 컴파일 실패, 링커 없음). 로그에
``error: could not compile `kernel``` 이 있을 때만 ``compile_error`` 이고, ``Compiling kernel`` 뒤의 시간 초과만 ``timeout`` 이다.
cargo 가 rc=0 으로 끝났는데 라이브러리가 제자리에 없을 때와 ``nm`` 이 실패했을 때도 ``ToolchainMissing`` 이다.

환경: POSIX 는 ``protocol.launch_env_minimal()`` (화이트리스트)에 ``CARGO_HOME``·``RUSTUP_HOME``·``RUSTUP_TOOLCHAIN`` 만 더해 넘긴다
(툴체인 위치). 그 밖(개발용 Windows)은 현재 환경을 상속한다. 어디서나 ``RUSTFLAGS``·``CARGO_TARGET_DIR`` 와
``CARGO_TERM_COLOR=never``·``CARGO_TERM_QUIET=false`` (로그를 읽어 원인을 가르므로 색 제어 문자를 막고 상태 줄을 켠다)를 덮어쓴다.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import shutil
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from harness.build.c import BUILD_JSON, BUILD_LOG, NM, BuildResult, ToolchainMissing, nm_dynamic, symbol_exported
from harness.config import ConfigError, load_config
from harness.paths import ROOT, atomic_write_bytes, atomic_write_json, atomic_write_text
from harness.procutil import IS_POSIX, ProcResult, run_isolated
from harness.protocol import launch_prefix
from harness.protocol import toolchain_env as _protocol_toolchain_env
from harness.sigs import RUST_ARGS_MODULE_NAME

_LOG = "[build.rust]"

TEMPLATE_DIR = Path(__file__).resolve().parent / "rust_template"
TEMPLATE_FILES: tuple[str, ...] = ("Cargo.toml", "Cargo.lock")
PACKAGE = "kernel"  # rust_template/Cargo.toml 의 패키지 이름
KERNEL_SRC = "src/lib.rs"
# cargo 가 <target_dir>/release/ 에 만드는 cdylib 이름과, out_dir 에 두는 산출물 이름(워커가 ctypes 로 여는 파일)
CARGO_CDYLIB = "kernel.dll" if os.name == "nt" else "libkernel.so"
ARTIFACT_NAME = "kernel.dll" if os.name == "nt" else "kernel.so"

# 설정(cfg.build.rust)에 없는 값의 기본(설계 v3 §8)
DEFAULT_CARGO = "cargo"
DEFAULT_RUSTFLAGS: tuple[str, ...] = ("-C", "target-cpu=native")
DEFAULT_BUILD_TIMEOUT_S = 600.0
DEFAULT_OFFLINE = True

RUSTC = "rustc"
# 아래 정규식들이 build.log 를 읽어 실패의 원인을 가른다: 사용자의 cargo 설정(term.color, term.quiet)과 무관하게 색 제어 문자가
# 섞이지 않고 상태 줄(``Compiling kernel …``)이 찍히게 한다
CARGO_OUTPUT_ENV: dict[str, str] = {"CARGO_TERM_COLOR": "never", "CARGO_TERM_QUIET": "false"}
CLEAN_TIMEOUT_S = 120.0
# target 디렉터리의 프로세스 간 잠금(모듈 docstring의 "잠금")
LOCK_NAME = ".harness-build.lock"
LOCK_POLL_S = 0.2
LOCK_WAIT_FACTOR = 2.0  # 잠금 대기 상한 = 이 배수 × (정리 + 빌드 시간 초과): 앞선 빌드 하나와 그 앞에 줄 선 하나까지
# probe_toolchain 의 자리표시 패키지 디렉터리(target 디렉터리 안)와 로그
PROBE_DIR_NAME = "_toolchain_probe"
PROBE_LOG = "probe.log"
ELF_MAGIC = b"\x7fELF"
FETCH_HINT = "cargo fetch --locked --manifest-path harness/build/rust_template/Cargo.toml"

_RUSTC_ERROR_RE = re.compile(r"^error(?:\[E\d+\])?: ")
_RUSTC_LOCATION_RE = re.compile(r"^\s*--> (\S.*)$")
_CARGO_SUMMARY_RE = re.compile(r"^error: (?:could not compile|aborting due to) ")
_KERNEL_STARTED_RE = re.compile(rf"^\s*Compiling {PACKAGE} v", re.M)
_KERNEL_FAILED_RE = re.compile(rf"^error: could not compile `{PACKAGE}`", re.M)
_LINKER_MISSING_RE = re.compile(r"^error: linker `[^`]*` not found", re.M)
_RUSTC_WARNING_RE = re.compile(r"^warning(?:\[\w+\])?: ")
_WARNING_SUMMARY_RE = re.compile(r"^warning: `[^`]*` \(.*\) generated \d+ warnings?")


# ----------------------------------------------------------------------------- 헬퍼


def toolchain_env() -> dict[str, str]:
    """cargo·rustc 를 띄우는 환경(모듈 docstring 의 "환경"). 구현은 ``protocol.toolchain_env`` 하나다 — 도구 점검
    (``protocol.toolchain_identity``)이 같은 환경으로 cargo·rustc 를 보므로 지문의 rustc 와 빌드한 rustc 가 같다."""
    return _protocol_toolchain_env(posix=IS_POSIX)


@functools.lru_cache(maxsize=None)
def rustc_version() -> str:
    """``rustc --version`` 첫 줄(모듈 캐시). 실행할 수 없으면 ``""``."""
    try:
        r = run_isolated([RUSTC, "--version"], timeout_s=30, cwd=ROOT, env=toolchain_env())
    except OSError:
        return ""
    lines = r.stdout_tail.strip().splitlines()
    return lines[0].strip() if r.rc == 0 and lines else ""


def cargo_error_detail(log: str, limit: int = 3) -> str:
    """cargo 로그에서 rustc 오류의 첫 ``limit`` 개를 `` | `` 로 이은 것. 오류 줄 다음의 위치 줄(`` --> src/lib.rs:5:13``)을
    붙이고 경로 구분자는 ``/`` 로 맞춘다. 요약 줄(``could not compile``, ``aborting due to``)은 뺀다.
    rustc 오류 줄이 없으면(컴파일러가 죽은 경우 등) 로그의 마지막 비어 있지 않은 ``limit`` 줄."""
    lines = log.splitlines()
    found: list[str] = []
    for i, line in enumerate(lines):
        if not _RUSTC_ERROR_RE.match(line) or _CARGO_SUMMARY_RE.match(line):
            continue
        location = _RUSTC_LOCATION_RE.match(lines[i + 1]) if i + 1 < len(lines) else None
        where = " --> " + location.group(1).strip().replace("\\", "/") if location else ""
        found.append(line.strip() + where)
        if len(found) == limit:
            break
    if not found:
        found = [ln.strip() for ln in lines if ln.strip()][-limit:]
    return " | ".join(found)


def count_rust_warnings(log: str) -> int:
    """rustc 경고 수(``warning: …`` 줄; cargo 의 요약 줄 ``warning: `kernel` (lib) generated N warnings`` 는 세지 않는다)."""
    return sum(1 for ln in log.splitlines() if _RUSTC_WARNING_RE.match(ln) and not _WARNING_SUMMARY_RE.match(ln))


def _read_log(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _log_excerpt(log: str, limit: int = 4) -> str:
    """``ToolchainMissing`` 메시지에 넣는 로그 첫머리(비어 있지 않은 ``limit`` 줄)."""
    return " | ".join([ln.strip() for ln in log.splitlines() if ln.strip()][:limit])


def lock_wait_s(build_timeout_s: float) -> float:
    """target 잠금을 기다리는 상한: 빌드 하나가 쓸 수 있는 최대 시간(정리 + 빌드)의 ``LOCK_WAIT_FACTOR`` 배(모듈 docstring)."""
    return LOCK_WAIT_FACTOR * (CLEAN_TIMEOUT_S + build_timeout_s)


def _try_lock(fd: int) -> None:
    """``fd`` 첫 바이트에 배타 잠금을 기다리지 않고 건다. 다른 쪽이 잡고 있으면 ``OSError``."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def target_lock(target: Path, wait_s: float) -> Iterator[None]:
    """``target/.harness-build.lock`` 에 프로세스 간 배타 잠금을 걸고 있는 동안 본문을 실행한다(모듈 docstring).

    ``wait_s`` 안에 잠금을 얻지 못하거나 잠금 파일을 열 수 없으면 ``ToolchainMissing`` — 샘플 코드와 무관한 환경 문제다.
    """
    target.mkdir(parents=True, exist_ok=True)
    path = target / LOCK_NAME
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    except OSError as e:
        raise ToolchainMissing(f"target 잠금 파일을 열 수 없다({path}): {e}") from e
    try:
        deadline = time.monotonic() + wait_s
        while True:
            try:
                _try_lock(fd)
                break
            except OSError as e:
                if time.monotonic() >= deadline:
                    raise ToolchainMissing(f"target 디렉터리 잠금을 {wait_s:g}s 안에 얻지 못했다({path}) — 같은 target 을 쓰는 "
                                           f"다른 빌드가 끝나지 않는다: {e}") from e
                time.sleep(LOCK_POLL_S)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


def _clean_kernel_package(cargo: str, offline_flags: list[str], env: dict[str, str], out_dir: Path) -> None:
    """target 디렉터리에서 커널 패키지의 예전 산출물·fingerprint 만 지운다(모듈 docstring; 의존성 빌드는 남는다)."""
    cmd = [cargo, "clean", "--release", "-p", PACKAGE, *offline_flags, "--locked"]
    try:
        r = run_isolated(cmd, timeout_s=CLEAN_TIMEOUT_S, cwd=out_dir, env=env)
    except OSError as e:
        raise ToolchainMissing(f"cargo 를 실행할 수 없다({cargo}): {e}") from e
    if r.timed_out or r.rc != 0:
        why = f"{CLEAN_TIMEOUT_S:g}s 안에 끝나지 않았다" if r.timed_out else f"rc={r.rc}"
        raise ToolchainMissing(f"`{' '.join(cmd)}` 실패({why}): {_log_excerpt(r.stderr_tail)} — 의존성 캐시가 없으면 `{FETCH_HINT}`")


# ----------------------------------------------------------------------------- 빌드


@dataclass
class _CargoSettings:
    """설정의 ``build.rust`` 에서 읽은 값(없는 값은 설계 v3 §8 의 기본값)."""

    cargo: str
    rustflags: list[str]
    timeout_s: float
    offline_flags: list[str]


def _cargo_settings(rcfg: Any) -> _CargoSettings:
    """``rcfg`` 를 속성으로 읽는다. cargo 가 PATH 에 없으면 ``ToolchainMissing``."""
    settings = _CargoSettings(
        cargo=str(getattr(rcfg, "cargo", DEFAULT_CARGO)),
        rustflags=[str(flag) for flag in getattr(rcfg, "rustflags", DEFAULT_RUSTFLAGS)],
        timeout_s=float(getattr(rcfg, "build_timeout_s", DEFAULT_BUILD_TIMEOUT_S)),
        offline_flags=["--offline"] if getattr(rcfg, "offline", DEFAULT_OFFLINE) else [],
    )
    if shutil.which(settings.cargo) is None:
        raise ToolchainMissing(f"cargo {settings.cargo!r} 를 찾을 수 없다(PATH)")
    return settings


def _write_package(out_dir: Path, code: str, args_module: str | None) -> None:
    """``out_dir`` 에 템플릿의 ``Cargo.toml``·``Cargo.lock`` 과 ``src/lib.rs`` (와 ``src/kernel_args.rs``)를 쓴다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in TEMPLATE_FILES:
        atomic_write_bytes(out_dir / name, (TEMPLATE_DIR / name).read_bytes())
    atomic_write_text(out_dir / KERNEL_SRC, code if code.endswith("\n") else code + "\n")
    if args_module is not None:
        atomic_write_text(out_dir / "src" / RUST_ARGS_MODULE_NAME, args_module)


@dataclass
class _CargoRun:
    """target 잠금 아래에서 한 정리 → 빌드 → 산출물 읽기의 결과."""

    cmd: list[str]
    proc: ProcResult
    build_time_ms: float
    log: str
    built: Path  # <target>/release/<cdylib>
    library: bytes | None  # rc=0 이고 ``built`` 가 있을 때 그 내용(잠금을 쥔 채 읽었다)


def _cargo_build(settings: _CargoSettings, out_dir: Path, target: Path, launch_prefix: list[str], log_path: Path) -> _CargoRun:
    """``out_dir`` 의 패키지를 ``target`` 에서 빌드한다. 정리·빌드·산출물 읽기 전체가 ``target_lock`` 안이다 — 같은 target 을 쓰는
    다른 프로세스가 그 사이에 커널 패키지를 지우거나 덮어써서 남의 라이브러리를 복사하는 일이 없다(모듈 docstring)."""
    env = {**toolchain_env(), "RUSTFLAGS": " ".join(settings.rustflags), "CARGO_TARGET_DIR": str(target), **CARGO_OUTPUT_ENV}
    cmd = [*launch_prefix, settings.cargo, "build", "--release", *settings.offline_flags, "--locked"]
    built = target / "release" / CARGO_CDYLIB
    with target_lock(target, lock_wait_s(settings.timeout_s)):
        _clean_kernel_package(settings.cargo, settings.offline_flags, env, out_dir)
        t0 = time.perf_counter()
        try:
            proc = run_isolated(cmd, timeout_s=settings.timeout_s, cwd=out_dir, env=env, stderr_path=log_path)
        except OSError as e:
            raise ToolchainMissing(f"빌드 명령을 시작할 수 없다({cmd[0]}): {e}") from e
        build_time_ms = (time.perf_counter() - t0) * 1000.0
        library = built.read_bytes() if not proc.timed_out and proc.rc == 0 and built.is_file() else None
    return _CargoRun(cmd=cmd, proc=proc, build_time_ms=build_time_ms, log=_read_log(log_path), built=built, library=library)


def build_rust(
    code: str,
    out_dir: Path,
    rcfg: Any,
    *,
    launch_prefix: list[str],
    symbol: str,
    args_module: str | None = None,
    target_dir: Path | None = None,
) -> BuildResult:
    """``out_dir`` 에 cargo 패키지를 쓰고 ``cdylib`` 로 빌드해 ``out_dir/kernel.so`` (Windows: ``kernel.dll``)로 복사한다.
    결과는 ``out_dir/build.json`` 에도 저장한다.

    - ``rcfg``: 설정의 ``build.rust`` (``cargo``, ``rustflags``, ``build_timeout_s``, ``offline``). 속성으로 읽고 없는 값은
      설계 v3 §8 의 기본값을 쓴다.
    - ``args_module``: keyword 모드의 ``kernel_args.rs`` 전문. 있으면 ``out_dir/src/kernel_args.rs`` 로 쓴다.
    - ``target_dir``: ``CARGO_TARGET_DIR``. 여러 샘플이 같이 쓰면 의존성(rayon)을 한 번만 빌드한다.

    cargo 를 쓸 수 없거나 커널 크레이트를 컴파일하기 전에 실패하면 ``ToolchainMissing`` (모델 오류로 기록하지 않는다).
    """
    settings = _cargo_settings(rcfg)
    rustflags, timeout_s = settings.rustflags, settings.timeout_s
    out_dir = Path(out_dir).resolve()
    target = Path(target_dir).resolve() if target_dir is not None else out_dir / "target"
    artifact = out_dir / ARTIFACT_NAME
    if artifact.exists():
        artifact.unlink()
    _write_package(out_dir, code, args_module)

    run = _cargo_build(settings, out_dir, target, launch_prefix, out_dir / BUILD_LOG)
    cmd, proc, build_time_ms, log, library = run.cmd, run.proc, run.build_time_ms, run.log, run.library
    extra: dict[str, Any] = {"rustflags": rustflags}

    def _result(ok: bool, kind: str = "", detail: str = "") -> BuildResult:
        res = BuildResult(
            ok=ok, artifact=artifact if ok else None, error_kind=kind, error_detail=detail, log=log, build_time_ms=build_time_ms,
            command=cmd, compiler=RUSTC, compiler_version=rustc_version(), n_warnings=count_rust_warnings(log), extra=extra,
        )
        atomic_write_json(out_dir / BUILD_JSON, res.to_dict())
        return res

    if proc.timed_out:
        if _KERNEL_STARTED_RE.search(log) is None:  # 아직 의존성을 빌드하던 중이다: 이 샘플의 코드와 무관하다
            raise ToolchainMissing(f"cargo 가 {timeout_s:g}s 안에 커널 크레이트의 컴파일을 시작하지 못했다: {_log_excerpt(log)}")
        return _result(False, "timeout", f"compile timeout after {timeout_s:g}s")
    if proc.rc != 0:
        if _KERNEL_FAILED_RE.search(log) is None or _LINKER_MISSING_RE.search(log) is not None:
            raise ToolchainMissing(f"cargo 가 커널 크레이트를 컴파일하지 못하고 끝났다(rc={proc.rc}): {_log_excerpt(log)} — "
                                   f"의존성 캐시가 없으면 `{FETCH_HINT}`")
        return _result(False, "compile_error", cargo_error_detail(log))

    if library is None:  # 커널 소스로는 바꿀 수 없는 자리다(cargo 설정의 build.target 등이 출력 디렉터리를 옮긴 경우)
        raise ToolchainMissing(f"cargo 는 성공했는데(rc=0) 산출물이 없다: {run.built} — cargo 설정(build.target 등)을 확인한다")
    atomic_write_bytes(artifact, library)

    if not library.startswith(ELF_MAGIC):
        extra["symbol_check"] = "skipped:not_elf"  # nm -D 는 ELF 의 동적 심벌 표를 읽는다(Windows 개발 머신의 DLL 은 대상이 아니다)
    elif shutil.which(NM) is None:
        extra["symbol_check"] = "skipped:nm_not_found"
    else:
        extra["symbol_check"] = "nm"
        if not symbol_exported(nm_dynamic(out_dir, "--defined-only", ARTIFACT_NAME), symbol):
            return _result(False, "signature_mismatch", f"symbol {symbol} not exported")
    return _result(True)


def probe_toolchain(rcfg: Any, target_dir: Path, *, launch_prefix: list[str]) -> None:
    """run 시작 전 1회: 템플릿의 자리표시 크레이트(``rust_template/src/lib.rs``)를 ``build_rust`` 와 똑같은 명령·환경·플래그·잠금으로
    ``target_dir`` 에서 빌드한다. 자리표시는 알려진 정상 코드이므로 어떤 실패든(cargo 없음, 의존성 캐시 없음, 링커 실패, 시간 초과,
    산출물 없음) 환경 문제다 — 샘플마다 ``compile_error`` 행이 쌓이기 전에 cargo 출력의 결정적인 줄과 함께 ``ToolchainMissing``.

    성공하면 의존성(rayon) 빌드가 ``target_dir`` 에 남아 샘플 빌드가 재사용한다. 패키지와 로그는 ``target_dir/_toolchain_probe/`` 에
    남는다(``probe.log``). 심벌 검사는 하지 않는다(자리표시는 함수를 내보내지 않는다).
    """
    settings = _cargo_settings(rcfg)
    target = Path(target_dir).resolve()
    probe_dir = target / PROBE_DIR_NAME
    log_path = probe_dir / PROBE_LOG
    _write_package(probe_dir, (TEMPLATE_DIR / KERNEL_SRC).read_text(encoding="utf-8"), None)
    run = _cargo_build(settings, probe_dir, target, launch_prefix, log_path)
    if run.library is not None:
        return
    if run.proc.timed_out:
        why = f"{settings.timeout_s:g}s 안에 끝나지 않았다"
    elif run.proc.rc != 0:
        why = f"rc={run.proc.rc}"
    else:
        why = f"rc=0 인데 산출물이 없다({run.built}; cargo 설정의 build.target 등)"
    raise ToolchainMissing(f"Rust 툴체인 점검 실패 — 자리표시 크레이트 빌드가 {why}: {cargo_error_detail(run.log) or 'cargo 출력 없음'} "
                           f"(전체 로그: {log_path}; 의존성 캐시가 없으면 `{FETCH_HINT}`)")


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """Rust 소스 파일 하나를 설정대로 빌드하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.rust", description="Rust 커널 빌드 (설계 v3 §6)")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--src", type=Path, required=True, help="Rust 소스 파일(src/lib.rs 로 쓰인다)")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉터리(Cargo.toml/src/kernel.so/build.log/build.json)")
    parser.add_argument("--symbol", required=True, help="내보내야 하는 함수 이름")
    parser.add_argument("--target-dir", type=Path, default=None, help="CARGO_TARGET_DIR (기본 <out-dir>/target)")
    parser.add_argument("--no-taskset", action="store_true", help="taskset 접두 없이 빌드")
    ns = parser.parse_args(argv)
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 2
    prefix = [] if ns.no_taskset else launch_prefix(cfg, cfg.measurement.threads)
    try:
        res = build_rust(ns.src.read_text(encoding="utf-8"), ns.out_dir, getattr(cfg.build, "rust", None), launch_prefix=prefix,
                         symbol=ns.symbol, target_dir=ns.target_dir)
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
