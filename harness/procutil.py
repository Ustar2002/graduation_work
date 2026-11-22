"""격리 서브프로세스 실행기 (설계 §7). 하네스의 모든 서브프로세스는 이 모듈을 거친다.

핵심 규칙:
- ``Popen(start_new_session=True)`` 로 새 세션(=새 프로세스 그룹)에서 시작한다.
- ``communicate(timeout)`` 이 만료되면 프로세스 **그룹 전체**에 SIGKILL을 보내고(``os.killpg``)
  ``wait()`` 한다. ``subprocess.run(timeout=)`` 은 직접 자식만 죽여 ``cc1``·OpenMP 자식이 남는다(실측).
- stdout/stderr 경로가 주어지면 파일로 리다이렉트하고 실행 후 꼬리만 읽는다. 아니면 PIPE로 받아
  꼬리 ``tail_chars`` 글자만 보관한다.
- rc<0 이면 ``signal_name`` 에 시그널 이름을 적는다(예: ``SIGKILL``, ``SIGSEGV``).
- ``communicate`` 를 타임아웃이 아닌 예외(``KeyboardInterrupt``=Ctrl-C, ``SystemExit`` 등 ``BaseException``)로
  빠져나갈 때도 그룹 전체를 SIGKILL 하고 ``wait()`` 한 뒤 예외를 다시 올린다. 새 세션의 워커는 터미널의 Ctrl-C 를
  받지 않으므로, 오케스트레이터가 죽을 때 여기서 죽이지 않으면 멈춘 커널이 고정 코어를 계속 점유한다(실측).
- 자식은 exec 직전에 ``prctl(PR_SET_PDEATHSIG, SIGKILL)`` 을 건다(``pdeathsig_preexec``): 부모가 SIGKILL·OOM 등
  핸들러가 돌 수 없는 방식으로 죽어도 커널이 직접 자식을 죽인다(exec 를 거쳐 ``taskset`` → 워커까지 유지; 손자는 제외).
- ``install_terminate_handlers()``: SIGTERM·SIGHUP 을 ``KeyboardInterrupt`` 로 바꿔 기본 처리(즉시 종료)로는 돌지
  못하는 위 정리 코드가 돌게 한다. run.py 가 main 에서 호출한다.
- ``setup_stdio()``: 진단용 CLI 진입점(``if __name__ == "__main__":``)이 표준 스트림을 정리하는 공용 헬퍼 — 파이프에는
  UTF-8 로 쓰고, 인코딩할 수 없는 글자는 ``\\uXXXX`` 로 찍는다. ``main(argv)`` 안에서는 부르지 않는다.

비-POSIX(개발용 Windows; 설계 v3 §5.6): 위 규칙은 **POSIX 에서만** 성립한다. ``os.name != "posix"`` 이면 임포트만 안전하게
하고 ``run_isolated`` 는 축소 경로로 돈다 — ``start_new_session``·``preexec_fn`` 없이 띄우고, 타임아웃이나 다른 예외로
빠져나갈 때 직접 자식만 ``proc.kill()``·``wait()`` 한다(프로세스 그룹 정리·PDEATHSIG 없음: 손자 프로세스는 남을 수 있다).
이때 ``rc`` 는 운영체제가 준 값(Windows 의 ``TerminateProcess`` 는 1)이고 ``signal_name`` 은 None 이므로 호출자는
``timed_out`` 을 먼저 본다. 측정은 Linux 에서만 하며 POSIX 경로의 동작은 바꾸지 않는다.
"""

from __future__ import annotations

import argparse
import ctypes
import functools
import json
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

_LOG = "[procutil]"

# 프로세스 그룹·PDEATHSIG 격리는 POSIX 에서만 한다(모듈 docstring 의 "비-POSIX" 참고).
IS_POSIX: bool = os.name == "posix"

# 타임아웃 뒤 남은 파이프 출력을 회수할 때 기다리는 최대 시간(초).
_DRAIN_TIMEOUT_S = 5.0

# prctl(2) 옵션: 부모(정확히는 부모 스레드)가 죽으면 커널이 자식에게 보낼 시그널.
PR_SET_PDEATHSIG = 1

# SIGKILL: POSIX 에서는 ``signal.SIGKILL`` 그대로. Windows 의 signal 모듈에는 없어서 번호(9)로 대신한다(임포트 안전성).
_SIGKILL: Any = getattr(signal, "SIGKILL", 9)

# 기본 처리가 "즉시 종료"라 KeyboardInterrupt 로 바꿔 받는 시그널(SIGINT 는 Python 이 이미 그렇게 한다).
# SIGHUP 은 Windows 에 없으므로 있을 때만 넣는다(POSIX 에서는 예전과 같은 (SIGTERM, SIGHUP)).
TERMINATE_SIGNALS: tuple[signal.Signals, ...] = tuple(
    s for s in (getattr(signal, "SIGTERM", None), getattr(signal, "SIGHUP", None)) if s is not None
)

# ``signal.Signals`` 가 모르는 번호의 이름(Linux x86 번호). POSIX 에서는 이 번호들이 전부 ``signal.Signals`` 에 있어
# 쓰이지 않는다 — Windows 에서 ``rc=-9`` 같은 값을 ``SIG9`` 가 아니라 ``SIGKILL`` 로 읽기 위한 것이다.
_FALLBACK_SIGNAL_NAMES: dict[int, str] = {
    1: "SIGHUP", 2: "SIGINT", 3: "SIGQUIT", 4: "SIGILL", 5: "SIGTRAP", 6: "SIGABRT", 7: "SIGBUS", 8: "SIGFPE",
    9: "SIGKILL", 10: "SIGUSR1", 11: "SIGSEGV", 12: "SIGUSR2", 13: "SIGPIPE", 14: "SIGALRM", 15: "SIGTERM",
}


@dataclass
class ProcResult:
    """서브프로세스 실행 결과."""

    rc: int
    timed_out: bool
    elapsed_s: float
    stdout_tail: str
    stderr_tail: str
    signal_name: str | None


def tail_text(data: bytes | str | None, tail_chars: int) -> str:
    """바이트/문자열의 마지막 ``tail_chars`` 글자(UTF-8, 깨진 바이트는 대체 문자)."""
    if data is None:
        return ""
    text = data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
    if tail_chars <= 0:
        return ""
    return text[-tail_chars:]


def tail_of_file(path: Path, tail_chars: int) -> str:
    """파일 끝부분만 읽어 마지막 ``tail_chars`` 글자를 돌려준다(큰 로그도 전체를 읽지 않음)."""
    if tail_chars <= 0:
        return ""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        return ""
    # UTF-8 최대 4바이트/글자 → 넉넉히 읽은 뒤 글자 수로 자른다.
    read_bytes = min(size, tail_chars * 4)
    with path.open("rb") as fh:
        fh.seek(size - read_bytes)
        data = fh.read(read_bytes)
    return tail_text(data, tail_chars)


def signal_name_of(rc: int) -> str | None:
    """음수 반환 코드를 시그널 이름으로 바꾼다. 알 수 없는 번호면 ``SIG<n>``."""
    if rc >= 0:
        return None
    try:
        return signal.Signals(-rc).name
    except ValueError:
        return _FALLBACK_SIGNAL_NAMES.get(-rc, f"SIG{-rc}")


def kill_process_group(proc: subprocess.Popen[bytes]) -> None:
    """프로세스 그룹 전체에 SIGKILL. 이미 사라졌으면(``ProcessLookupError``) 무시한다.

    ``start_new_session=True`` 로 시작했으므로 그룹 id는 자식 pid와 같지만, 설계대로
    ``os.getpgid`` 로 확인한다. 만에 하나 우리 그룹과 같으면 자식 하나만 죽인다(자살 방지).
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    if pgid == os.getpgid(0):
        print(f"{_LOG} 경고: 자식 pid={proc.pid} 가 호출자와 같은 프로세스 그룹이다; 단일 kill", file=sys.stderr)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def kill_direct_child(proc: subprocess.Popen[bytes]) -> None:
    """비-POSIX 축소 경로: 직접 자식 하나만 죽인다(``proc.kill()``). 이미 끝났으면 아무 일도 하지 않는다.

    프로세스 그룹이 없으므로 자식이 띄운 손자 프로세스는 남을 수 있다(설계 v3 §5.6 — 측정용이 아니다).
    """
    if proc.returncode is not None:
        return
    try:
        proc.kill()
    except OSError as e:  # 종료 직전의 경쟁(ERROR_ACCESS_DENIED 등): 거두기는 호출자의 wait() 가 한다
        print(f"{_LOG} 경고: 자식 pid={proc.pid} kill 실패: {e}", file=sys.stderr)


def _kill_child(proc: subprocess.Popen[bytes]) -> None:
    """플랫폼에 맞는 정리: POSIX 는 프로세스 그룹 전체 SIGKILL, 그 밖은 직접 자식만."""
    if IS_POSIX:
        kill_process_group(proc)
    else:
        kill_direct_child(proc)


def _isolation_kwargs() -> dict[str, Any]:
    """``Popen`` 의 격리 인자. POSIX: 새 세션 + PDEATHSIG(예전과 같은 인자). 비-POSIX: 없음."""
    if IS_POSIX:
        return {"start_new_session": True, "preexec_fn": pdeathsig_preexec()}
    return {}


def _close_pipes(proc: subprocess.Popen[bytes]) -> None:
    """``communicate`` 가 정상 완료되지 못했을 때 남은 PIPE 핸들을 닫는다."""
    for fh in (proc.stdout, proc.stderr):
        if fh is not None:
            fh.close()


@functools.cache
def _libc_handle() -> ctypes.CDLL | None:
    """``prctl`` 을 가진 libc 핸들(한 번만 로드). 없으면 None."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
    except (OSError, TypeError):  # TypeError: Windows 의 ctypes 는 CDLL(None) 을 받지 않는다
        return None
    return libc if hasattr(libc, "prctl") else None


def pdeathsig_preexec(sig: signal.Signals = _SIGKILL) -> Callable[[], None]:
    """``Popen(preexec_fn=)`` 용 함수: 자식이 exec 직전에 ``prctl(PR_SET_PDEATHSIG, sig)`` 를 건다.

    부모가 SIGKILL·OOM 처럼 핸들러가 돌 수 없는 방식으로 죽어도 커널이 자식을 죽인다. 설정은 exec 를 거쳐도
    유지되므로 ``taskset`` → ``python -m harness.worker`` 까지 닿는다(손자 프로세스·setuid 바이너리는 제외).
    fork 와 prctl 사이에 부모가 이미 죽었으면(``getppid`` 가 바뀜) 자식이 스스로 끝난다. prctl 실패는 무시한다.
    """
    parent_pid = os.getpid()
    libc = _libc_handle()  # 부모에서 미리 로드한다(fork 뒤 자식에서 dlopen 하지 않게)

    def _preexec() -> None:
        if libc is not None:
            try:
                libc.prctl(PR_SET_PDEATHSIG, int(sig), 0, 0, 0)
            except Exception:  # preexec_fn 안의 예외는 Popen 자체를 실패시키므로 삼킨다
                pass
        if os.getppid() != parent_pid:
            os._exit(1)

    return _preexec


def install_terminate_handlers(signals: tuple[signal.Signals, ...] = TERMINATE_SIGNALS) -> list[str]:
    """SIGTERM·SIGHUP 을 ``KeyboardInterrupt`` 로 바꾼다(주 스레드에서 호출). 설치한 시그널 이름을 돌려준다.

    기본 처리(즉시 종료)로는 ``run_isolated`` 의 그룹 정리가 돌지 못해 워커가 고정 코어에 남는다.
    이미 무시된 시그널(``SIG_IGN``, 예: ``nohup``)은 그대로 둔다. 주 스레드가 아니면 경고만 하고 아무것도 설치하지 않는다.
    SIGHUP 이 없는 플랫폼(Windows)에서는 기본값 ``TERMINATE_SIGNALS`` 가 SIGTERM 하나다.
    """
    if threading.current_thread() is not threading.main_thread():
        print(f"{_LOG} 경고: 주 스레드가 아니라 시그널 핸들러를 설치하지 않는다", file=sys.stderr)
        return []

    def _raise(signum: int, _frame: object) -> None:
        raise KeyboardInterrupt(signal.Signals(signum).name)

    installed: list[str] = []
    for sig in signals:
        if signal.getsignal(sig) == signal.SIG_IGN:
            continue
        signal.signal(sig, _raise)
        installed.append(sig.name)
    return installed


def setup_stdio() -> None:
    """진단용 CLI 의 표준 출력·오류 인코딩을 정리한다(``harness.procutil``·``harness.sigs``·``harness.paths``·``tools/devcheck.py``).

    - 파이프·파일로 나갈 때는 UTF-8 로 쓴다(Windows 의 로캘 코드페이지 대신 — 로그와 다른 도구가 읽을 수 있게). 콘솔(tty)이면
      인코딩은 그대로 둔다.
    - 어느 쪽이든 인코딩할 수 없는 글자는 ``\\uXXXX`` 로 바꿔 찍는다. 자식 프로세스의 출력·파일 경로·YAML 원문처럼
      내용을 미리 알 수 없는 문자열을 찍다가 ``UnicodeEncodeError`` 로 죽지 않는다(로캘이 cp949 인 Windows 의 파이프에서 실측).

    **프로세스 진입점(``if __name__ == "__main__":``)에서만 부른다.** ``main(argv)`` 안에서 부르면 테스트처럼 같은 프로세스에서
    ``main`` 을 부르는 쪽의 실제 ``sys.stdout`` 이 그 프로세스가 끝날 때까지 바뀐다. UTF-8 로캘의 Linux 에서는 인코딩이 그대로다.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream.isatty():
                stream.reconfigure(errors="backslashreplace")  # type: ignore[union-attr]
            else:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):  # 다른 객체로 바뀐 스트림(StringIO 등)·닫힌 스트림: 그대로 둔다
            pass


def run_isolated(
    cmd: list[str],
    *,
    timeout_s: float,
    cwd: Path,
    env: dict[str, str] | None = None,
    stdout_path: Path | None = None,
    stderr_path: Path | None = None,
    tail_chars: int = 2000,
) -> ProcResult:
    """명령을 새 프로세스 그룹에서 실행하고 타임아웃 시 그룹 전체를 SIGKILL 한다.

    - ``env=None`` 이면 현재 환경을 상속한다(워커는 ``protocol.launch_env`` 의 화이트리스트를 넘긴다).
    - ``stdout_path``/``stderr_path`` 가 주어지면 그 파일에 쓰고(기존 내용 덮어씀) 꼬리만 읽는다.
    - 명령을 찾지 못하는 등 실행 자체의 실패(``FileNotFoundError``)는 삼키지 않고 그대로 올린다.
    - 기다리는 동안 ``KeyboardInterrupt`` 등 다른 예외가 나면 그룹 전체를 SIGKILL·``wait()`` 한 뒤 그대로 올린다.
    - 자식에는 ``PR_SET_PDEATHSIG=SIGKILL`` 을 건다(부모가 시그널로 죽어도 직접 자식은 남지 않음).
    - 비-POSIX(Windows)에서는 새 세션·PDEATHSIG 없이 띄우고 타임아웃·예외 시 직접 자식만 ``kill()``·``wait()`` 한다
      (모듈 docstring 참고; 개발용 축소 경로).
    """
    cmd = [str(c) for c in cmd]
    out_fh = open(stdout_path, "wb") if stdout_path is not None else None
    err_fh = open(stderr_path, "wb") if stderr_path is not None else None
    proc: subprocess.Popen[bytes] | None = None
    t0 = time.perf_counter()
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out_fh if out_fh is not None else subprocess.PIPE,
            stderr=err_fh if err_fh is not None else subprocess.PIPE,
            **_isolation_kwargs(),  # POSIX: start_new_session=True, preexec_fn=pdeathsig_preexec()
        )
        timed_out = False
        out_b: bytes | None = b""
        err_b: bytes | None = b""
        try:
            out_b, err_b = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            _kill_child(proc)
            proc.wait()
            # 그룹이 죽었으므로 파이프가 닫힌다. 남은 출력을 회수한다(처음부터 누적된 전체가 돌아온다).
            # 그룹을 빠져나간 프로세스가 파이프를 잡고 있으면 짧게 기다린 뒤 부분 출력으로 만족한다.
            try:
                out_b, err_b = proc.communicate(timeout=_DRAIN_TIMEOUT_S)
            except subprocess.TimeoutExpired as exc2:
                print(f"{_LOG} 경고: 타임아웃 뒤 파이프 회수 실패(그룹 밖 프로세스가 잡고 있음): {cmd[0]}", file=sys.stderr)
                out_b = exc2.stdout if exc2.stdout is not None else exc.stdout
                err_b = exc2.stderr if exc2.stderr is not None else exc.stderr
                if IS_POSIX:
                    _close_pipes(proc)
                # 비-POSIX: communicate 의 읽기 스레드가 파이프 객체의 잠금을 쥔 채 read() 에 막혀 있어 close() 가
                # 손자 프로세스가 끝날 때까지 돌아오지 않는다 → 닫지 않고 둔다(데몬 스레드; 개발용 축소 경로).
        elapsed = time.perf_counter() - t0
        rc = proc.returncode if proc.returncode is not None else -int(_SIGKILL)
    except BaseException:
        # 타임아웃이 아닌 모든 탈출(Ctrl-C 의 KeyboardInterrupt, 시그널 핸들러의 예외, SystemExit, 그 밖의 예외):
        # 자식 핸들을 얻은 뒤 아직 거두지 않았으면 그룹 전체를 죽이고 거둔 다음 예외를 그대로 올린다.
        # (Popen 자체의 실패 — FileNotFoundError 등 — 는 proc 이 None 이라 죽일 것이 없다.)
        if proc is not None and proc.returncode is None:
            _kill_child(proc)
            proc.wait()
            if IS_POSIX:
                _close_pipes(proc)  # 비-POSIX 에서는 위와 같은 이유로 닫지 않는다
        raise
    finally:
        if out_fh is not None:
            out_fh.close()
        if err_fh is not None:
            err_fh.close()

    stdout_tail = tail_of_file(stdout_path, tail_chars) if stdout_path is not None else tail_text(out_b, tail_chars)
    stderr_tail = tail_of_file(stderr_path, tail_chars) if stderr_path is not None else tail_text(err_b, tail_chars)
    return ProcResult(
        rc=rc,
        timed_out=timed_out,
        elapsed_s=elapsed,
        stdout_tail=stdout_tail,
        stderr_tail=stderr_tail,
        signal_name=signal_name_of(rc),
    )


def main(argv: list[str] | None = None) -> int:
    """명령을 격리 실행하고 ProcResult를 JSON으로 출력한다(진단용).

    예: ``python -m harness.procutil --timeout 2 -- bash -c "sleep 300 & sleep 300"``
    """
    parser = argparse.ArgumentParser(prog="python -m harness.procutil", description="격리 서브프로세스 실행기")
    parser.add_argument("--timeout", type=float, default=60.0, help="타임아웃(초)")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="작업 디렉터리")
    parser.add_argument("--stdout", type=Path, default=None, help="stdout 리다이렉트 파일")
    parser.add_argument("--stderr", type=Path, default=None, help="stderr 리다이렉트 파일")
    parser.add_argument("--tail", type=int, default=2000, help="보관할 꼬리 글자 수")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- 뒤에 실행할 명령")
    ns = parser.parse_args(argv)
    cmd = ns.cmd[1:] if ns.cmd[:1] == ["--"] else ns.cmd
    if not cmd:
        parser.error("실행할 명령이 없다 (예: -- echo hi)")
    result = run_isolated(
        cmd, timeout_s=ns.timeout, cwd=ns.cwd, stdout_path=ns.stdout, stderr_path=ns.stderr, tail_chars=ns.tail
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    return 0 if result.rc == 0 and not result.timed_out else 1


if __name__ == "__main__":
    # 꼬리에는 자식이 찍은 임의의 글자(와 깨진 바이트의 대체 문자 U+FFFD)가 들어 있다. 로캘 코드페이지(cp949 등)로는
    # 인코딩할 수 없는 글자가 있어도 JSON 을 찍다가 죽지 않게 스트림을 정리한다(진입점에서만; ``setup_stdio`` 참고).
    setup_stdio()
    sys.exit(main())
