"""procutil.run_isolated 테스트: 타임아웃 뒤, 그리고 부모가 Ctrl-C·SIGTERM·SIGKILL 로 죽은 뒤에도
프로세스 그룹(손자 포함)이 비어 있어야 한다.

프로세스 그룹·PDEATHSIG 테스트(``RunIsolatedTest``, ``ParentDeathTest``)는 Linux 전용이다. ``PortableRunIsolatedTest`` 는
bash 없이 파이썬 자식만으로 어느 플랫폼에서나 도는 계약(종료 코드·꼬리·타임아웃·예외 시 자식 정리)을 확인한다 —
비-POSIX 에서는 ``run_isolated`` 의 축소 경로(직접 자식만 kill; 설계 v3 §0)를 지난다.

같은 기반 모듈인 ``harness/paths.py`` 의 v3 추가분(턴별 경로 §9.1·§9.2, 비-POSIX 의 ``PYTHON_EXE``)도 여기서 확인한다(``PathsTest``).
진단용 CLI 와 진입점의 스트림 정리(``setup_stdio``)는 ``CliTest``, 플랫폼 표시 헬퍼 ``tests/_support.py`` 의 ``requires_modules`` 는
``SupportMarkersTest`` 가 본다."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from harness import paths, procutil
from harness.procutil import (
    TERMINATE_SIGNALS,
    ProcResult,
    install_terminate_handlers,
    run_isolated,
    signal_name_of,
    tail_text,
)
from tests import _support
from tests._support import requires_linux, requires_modules

ROOT = Path(__file__).resolve().parents[1]

# 오케스트레이터 역할의 부모 스크립트: run_isolated 로 sleep 그룹(선택적으로 손자 포함)을 띄우고 시그널을 기다린다.
# 자식 bash 는 $$ 를 pgid 파일에 쓴 뒤 sleep 으로 exec 하므로 직접 자식의 pid == 그룹 id 다.
_PARENT_SRC = textwrap.dedent(
    """
    import signal, sys
    from pathlib import Path
    from harness.procutil import install_terminate_handlers, run_isolated

    tmp, mode, grandchild = Path(sys.argv[1]), sys.argv[2], sys.argv[3] == "1"
    # 테스트 러너가 물려준 시그널 처리와 무관하게 기본 상태(SIGINT → KeyboardInterrupt, 나머지 기본 처리)에서 시작한다.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    for s in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(s, signal.SIG_DFL)
    if mode == "handlers":
        install_terminate_handlers()
    script = f"echo $$ > {tmp}/pgid; " + ("sleep 300 & " if grandchild else "") + "exec sleep 300"
    try:
        run_isolated(["bash", "-c", script], timeout_s=100, cwd=tmp)
    except KeyboardInterrupt as e:
        print(f"interrupted {e}", flush=True)
        raise SystemExit(130)
    print("finished-unexpectedly", flush=True)
    """
)


def _pgrep_group(pgid: int) -> str:
    """``pgrep -g <pgid>`` 출력(빈 문자열이면 그룹에 프로세스 없음)."""
    p = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True)
    return p.stdout.strip()


def _wait_group_empty(pgid: int, deadline_s: float = 3.0) -> str:
    """init 이 손자 좀비를 거두는 짧은 지연을 허용하며 그룹이 빌 때까지 기다린다."""
    t_end = time.monotonic() + deadline_s
    out = _pgrep_group(pgid)
    while out and time.monotonic() < t_end:
        time.sleep(0.1)
        out = _pgrep_group(pgid)
    return out


def _wait_group_count(pgid: int, n: int, deadline_s: float = 3.0) -> int:
    """그룹의 프로세스 수가 ``n`` 이상이 될 때까지 기다린다(손자가 fork 되는 짧은 지연 허용). 현재 수를 돌려준다."""
    t_end = time.monotonic() + deadline_s
    count = len(_pgrep_group(pgid).split())
    while count < n and time.monotonic() < t_end:
        time.sleep(0.1)
        count = len(_pgrep_group(pgid).split())
    return count


def _killpg_quiet(pgid: int) -> None:
    """테스트 실패 시에도 sleep 그룹을 남기지 않기 위한 정리(이미 없으면 무시)."""
    try:
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


@requires_linux("프로세스 그룹(killpg)·음수 종료 코드는 POSIX 경로에서만 성립한다")
@unittest.skipIf(shutil.which("bash") is None or shutil.which("pgrep") is None, "bash/pgrep 필요")
class RunIsolatedTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="procutil_test_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_timeout_kills_whole_group_including_grandchild(self) -> None:
        pgid_file = self.tmp / "pgid.txt"
        # $$ = bash 의 pid = 새 세션의 프로세스 그룹 id. 백그라운드 sleep 은 손자 프로세스다.
        cmd = ["bash", "-c", f"echo $$ > {pgid_file}; echo started; sleep 300 & sleep 300"]
        t0 = time.monotonic()
        res = run_isolated(cmd, timeout_s=1.0, cwd=self.tmp)
        elapsed = time.monotonic() - t0
        self.assertTrue(res.timed_out)
        self.assertEqual(res.rc, -9)
        self.assertEqual(res.signal_name, "SIGKILL")
        self.assertLess(elapsed, 10.0, "타임아웃 뒤 바로 돌아와야 한다")
        self.assertIn("started", res.stdout_tail, "타임아웃 전 출력은 보존돼야 한다")
        pgid = int(pgid_file.read_text().strip())
        leftover = _wait_group_empty(pgid)
        self.assertEqual(leftover, "", f"프로세스 그룹 {pgid} 에 살아남은 프로세스: {leftover}")

    def test_timeout_with_file_redirect_leaves_no_process(self) -> None:
        pgid_file = self.tmp / "pgid.txt"
        out_log = self.tmp / "out.log"
        err_log = self.tmp / "err.log"
        cmd = ["bash", "-c", f"echo $$ > {pgid_file}; echo to-stdout; echo to-stderr >&2; sleep 300 & sleep 300"]
        res = run_isolated(cmd, timeout_s=1.0, cwd=self.tmp, stdout_path=out_log, stderr_path=err_log)
        self.assertTrue(res.timed_out)
        self.assertEqual(res.stdout_tail.strip(), "to-stdout")
        self.assertEqual(res.stderr_tail.strip(), "to-stderr")
        pgid = int(pgid_file.read_text().strip())
        self.assertEqual(_wait_group_empty(pgid), "")

    def test_normal_run_captures_tails_and_rc(self) -> None:
        res = run_isolated(["bash", "-c", "echo hello; echo oops >&2; exit 3"], timeout_s=10, cwd=self.tmp)
        self.assertIsInstance(res, ProcResult)
        self.assertEqual(res.rc, 3)
        self.assertFalse(res.timed_out)
        self.assertIsNone(res.signal_name)
        self.assertEqual(res.stdout_tail, "hello\n")
        self.assertEqual(res.stderr_tail, "oops\n")
        self.assertGreater(res.elapsed_s, 0.0)

    def test_tail_chars_limits_pipe_output(self) -> None:
        res = run_isolated(["bash", "-c", "printf 'a%.0s' $(seq 1 5000)"], timeout_s=10, cwd=self.tmp, tail_chars=100)
        self.assertEqual(res.rc, 0)
        self.assertEqual(len(res.stdout_tail), 100)

    def test_file_redirect_writes_full_output_and_reads_tail(self) -> None:
        out_log = self.tmp / "o.log"
        res = run_isolated(["bash", "-c", "printf 'b%.0s' $(seq 1 5000); echo END"], timeout_s=10, cwd=self.tmp,
                           stdout_path=out_log, tail_chars=50)
        self.assertEqual(res.rc, 0)
        self.assertEqual(out_log.stat().st_size, 5004)
        self.assertEqual(len(res.stdout_tail), 50)
        self.assertTrue(res.stdout_tail.endswith("END\n"))

    def test_signal_death_is_named(self) -> None:
        res = run_isolated(["bash", "-c", "kill -TERM $$"], timeout_s=10, cwd=self.tmp)
        self.assertEqual(res.rc, -15)
        self.assertEqual(res.signal_name, "SIGTERM")
        self.assertFalse(res.timed_out)

    def test_env_is_passed_through(self) -> None:
        res = run_isolated(["bash", "-c", "echo $PROBE_VAR"], timeout_s=10, cwd=self.tmp,
                           env={"PATH": "/usr/bin:/bin", "PROBE_VAR": "xyz"})
        self.assertEqual(res.stdout_tail.strip(), "xyz")

    def test_missing_executable_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_isolated(["/nonexistent/binary-xyz"], timeout_s=5, cwd=self.tmp)


@requires_linux("부모 사망 시 그룹 정리·PR_SET_PDEATHSIG 는 Linux 경로에서만 성립한다")
@unittest.skipIf(shutil.which("bash") is None or shutil.which("pgrep") is None, "bash/pgrep 필요")
class ParentDeathTest(unittest.TestCase):
    """오케스트레이터(부모)가 Ctrl-C·SIGTERM·SIGKILL 로 죽을 때 run_isolated 의 자식 그룹이 남지 않아야 한다.

    회귀: 예전에는 TimeoutExpired 에서만 그룹을 죽여, 부모가 KeyboardInterrupt/SIGTERM 으로 죽으면 새 세션의
    워커(sleep)가 PPID=1 로 살아남아 고정 코어를 계속 점유했다.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="procutil_parent_")
        self.tmp = Path(self._tmp.name)
        self.stderr_log = self.tmp / "parent.stderr.log"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stderr_text(self) -> str:
        try:
            return self.stderr_log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""

    def _spawn_parent(self, mode: str, *, grandchild: bool) -> tuple[subprocess.Popen[str], int]:
        """부모 스크립트를 띄우고 자식 그룹이 생길 때까지 기다린다. (부모, 자식 그룹 pgid)."""
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        stderr_fh = self.stderr_log.open("wb")
        self.addCleanup(stderr_fh.close)
        parent = subprocess.Popen(
            [sys.executable, "-c", _PARENT_SRC, str(self.tmp), mode, "1" if grandchild else "0"],
            cwd=str(ROOT), env=env, stdout=subprocess.PIPE, stderr=stderr_fh, text=True,
        )
        self.addCleanup(self._reap_parent, parent)
        pgid_file = self.tmp / "pgid"
        t_end = time.monotonic() + 15.0
        while time.monotonic() < t_end:
            if parent.poll() is not None:
                self.fail(f"부모가 자식을 띄우기 전에 끝났다 (rc={parent.returncode}):\n{self._stderr_text()}")
            text = pgid_file.read_text().strip() if pgid_file.exists() else ""
            if text:
                break
            time.sleep(0.05)
        else:
            self.fail(f"자식 그룹이 생기지 않았다:\n{self._stderr_text()}")
        pgid = int(text)
        self.addCleanup(_killpg_quiet, pgid)
        expected = 2 if grandchild else 1
        self.assertGreaterEqual(_wait_group_count(pgid, expected), expected, "자식 그룹이 예상대로 만들어지지 않았다")
        time.sleep(0.2)  # 부모가 communicate() 에 들어갈 여유
        return parent, pgid

    @staticmethod
    def _reap_parent(parent: subprocess.Popen[str]) -> None:
        if parent.poll() is None:
            parent.kill()
        parent.wait()
        if parent.stdout is not None:
            parent.stdout.close()

    def test_keyboard_interrupt_kills_whole_group(self) -> None:
        """Ctrl-C(SIGINT → KeyboardInterrupt)로 부모가 풀릴 때 run_isolated 가 그룹(손자 포함)을 죽이고 예외를 다시 올린다."""
        parent, pgid = self._spawn_parent("plain", grandchild=True)
        parent.send_signal(signal.SIGINT)
        out, _ = parent.communicate(timeout=15)
        self.assertIn("interrupted", out, f"KeyboardInterrupt 가 다시 올라오지 않았다:\n{self._stderr_text()}")
        self.assertEqual(parent.returncode, 130)
        leftover = _wait_group_empty(pgid)
        self.assertEqual(leftover, "", f"Ctrl-C 뒤 자식 그룹 {pgid} 에 살아남은 프로세스: {leftover}")

    def test_sigterm_with_handlers_kills_whole_group(self) -> None:
        """install_terminate_handlers 뒤 SIGTERM 은 KeyboardInterrupt 가 되어 그룹을 정리하고 130 으로 끝난다."""
        parent, pgid = self._spawn_parent("handlers", grandchild=True)
        parent.send_signal(signal.SIGTERM)
        out, _ = parent.communicate(timeout=15)
        self.assertIn("interrupted SIGTERM", out, f"SIGTERM 이 KeyboardInterrupt 로 바뀌지 않았다:\n{self._stderr_text()}")
        self.assertEqual(parent.returncode, 130)
        leftover = _wait_group_empty(pgid)
        self.assertEqual(leftover, "", f"SIGTERM 뒤 자식 그룹 {pgid} 에 살아남은 프로세스: {leftover}")

    def test_parent_sigkill_kills_direct_child_via_pdeathsig(self) -> None:
        """부모가 SIGKILL(핸들러가 돌 수 없음)로 죽어도 PR_SET_PDEATHSIG 로 직접 자식은 커널이 죽인다."""
        parent, pgid = self._spawn_parent("plain", grandchild=False)
        parent.send_signal(signal.SIGKILL)
        parent.communicate(timeout=15)
        self.assertEqual(parent.returncode, -9)
        leftover = _wait_group_empty(pgid)
        self.assertEqual(leftover, "", f"부모 SIGKILL 뒤 직접 자식이 남았다 (PR_SET_PDEATHSIG): {leftover}")


# 자식 스크립트: 시작을 알리고(heartbeat 파일을 만들고 "started" 를 출력) 0.05 초마다 heartbeat 에 한 줄씩 덧붙인다.
# 죽었는지는 "heartbeat 가 더 자라지 않는다"로 본다(pid 조회 없이 어느 플랫폼에서나 같은 방법).
_HEARTBEAT_SRC = textwrap.dedent(
    """
    import sys, time
    path = sys.argv[1]
    with open(path, "a", encoding="utf-8") as fh:
        fh.write("start\\n"); fh.flush()
        print("started", flush=True)
        print("warming-up", file=sys.stderr, flush=True)
        while True:
            time.sleep(0.05)
            fh.write("beat\\n"); fh.flush()
    """
)


def _wait_for_file(path: Path, deadline_s: float = 15.0) -> bool:
    t_end = time.monotonic() + deadline_s
    while time.monotonic() < t_end:
        if path.exists() and path.stat().st_size > 0:
            return True
        time.sleep(0.02)
    return False


def _stopped_growing(path: Path, settle_s: float = 0.6) -> bool:
    """``settle_s`` 동안 파일 크기가 그대로인가(= 자식이 더는 쓰지 않는다)."""
    before = path.stat().st_size
    time.sleep(settle_s)
    return path.stat().st_size == before


class PortableRunIsolatedTest(unittest.TestCase):
    """파이썬 자식만 쓰는 run_isolated 계약 테스트: POSIX·비-POSIX 어디서나 돈다."""

    # 타임아웃 테스트의 제한 시간: 자식 인터프리터가 뜨고 첫 줄을 쓸 여유(부하가 있는 머신에서도)를 둔다.
    TIMEOUT_S = 3.0

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="procutil_portable_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _py(self, code: str, *args: str) -> list[str]:
        return [sys.executable, "-c", code, *args]

    def test_exit_code_and_tails_are_reported(self) -> None:
        code = "import sys; print('hello'); print('oops', file=sys.stderr); sys.exit(3)"
        res = run_isolated(self._py(code), timeout_s=60, cwd=self.tmp)
        self.assertIsInstance(res, ProcResult)
        self.assertEqual(res.rc, 3)
        self.assertFalse(res.timed_out)
        self.assertIsNone(res.signal_name)
        self.assertEqual(res.stdout_tail.splitlines(), ["hello"])
        self.assertEqual(res.stderr_tail.splitlines(), ["oops"])
        self.assertGreater(res.elapsed_s, 0.0)

    def test_success_is_rc_zero(self) -> None:
        res = run_isolated(self._py("print('ok')"), timeout_s=60, cwd=self.tmp)
        self.assertEqual((res.rc, res.timed_out, res.signal_name), (0, False, None))
        self.assertEqual(res.stdout_tail.strip(), "ok")
        self.assertEqual(res.stderr_tail, "")

    # 자식이 작업 디렉터리를 UTF-8 바이트로 쓴다. print() 는 파이프에 로캘 코드페이지(Windows 의 cp949 등)로 쓰고
    # run_isolated 는 꼬리를 UTF-8 로 읽으므로, 경로에 비-ASCII 글자가 있으면 print() 로는 비교가 깨진다.
    _CWD_SRC = "import os, sys; sys.stdout.buffer.write(os.getcwd().encode('utf-8'))"

    def test_cwd_is_applied(self) -> None:
        res = run_isolated(self._py(self._CWD_SRC), timeout_s=60, cwd=self.tmp)
        self.assertEqual(res.rc, 0, res.stderr_tail)
        self.assertEqual(Path(res.stdout_tail.strip()).resolve(), self.tmp.resolve())

    def test_cwd_with_non_ascii_characters(self) -> None:
        """회귀: 임시 디렉터리 경로에 한글이 있으면(예: 한글 사용자 이름의 %TEMP%) 이 모듈의 테스트가 깨졌다."""
        sub = self.tmp / "임시 폴더"
        sub.mkdir()
        res = run_isolated(self._py(self._CWD_SRC), timeout_s=60, cwd=sub)
        self.assertEqual(res.rc, 0, res.stderr_tail)
        self.assertEqual(Path(res.stdout_tail.strip()).resolve(), sub.resolve())

    def test_timeout_kills_the_child_and_keeps_its_output(self) -> None:
        beat = self.tmp / "heartbeat.txt"
        t0 = time.monotonic()
        res = run_isolated(self._py(_HEARTBEAT_SRC, str(beat)), timeout_s=self.TIMEOUT_S, cwd=self.tmp)
        elapsed = time.monotonic() - t0
        self.assertTrue(res.timed_out)
        self.assertLess(elapsed, self.TIMEOUT_S + 12.0, "타임아웃 뒤 바로 돌아와야 한다")
        self.assertGreaterEqual(res.elapsed_s, self.TIMEOUT_S)
        self.assertIn("started", res.stdout_tail, "타임아웃 전 출력은 보존돼야 한다")
        self.assertIn("warming-up", res.stderr_tail)
        self.assertTrue(beat.exists() and beat.stat().st_size > 0, "자식이 실제로 돌았어야 한다")
        self.assertTrue(_stopped_growing(beat), "타임아웃 뒤 자식이 살아 있다(heartbeat 가 계속 자란다)")
        if os.name == "posix":
            self.assertEqual((res.rc, res.signal_name), (-9, "SIGKILL"))
        else:
            # 비-POSIX 축소 경로: 운영체제가 준 종료 코드(TerminateProcess → 1)이고 시그널 이름은 없다
            self.assertNotEqual(res.rc, 0)
            self.assertIsNone(res.signal_name)

    def test_timeout_with_file_redirect_reads_tails_from_files(self) -> None:
        beat = self.tmp / "heartbeat.txt"
        out_log, err_log = self.tmp / "out.log", self.tmp / "err.log"
        res = run_isolated(self._py(_HEARTBEAT_SRC, str(beat)), timeout_s=self.TIMEOUT_S, cwd=self.tmp,
                           stdout_path=out_log, stderr_path=err_log)
        self.assertTrue(res.timed_out)
        self.assertEqual(res.stdout_tail.strip(), "started")
        self.assertEqual(res.stderr_tail.strip(), "warming-up")
        self.assertEqual(out_log.read_text(encoding="utf-8").strip(), "started")
        self.assertTrue(_stopped_growing(beat), "타임아웃 뒤 자식이 살아 있다")

    def test_file_redirect_writes_full_output_and_reads_tail(self) -> None:
        out_log = self.tmp / "o.log"
        code = "import sys; sys.stdout.buffer.write(b'b' * 5000 + b'END\\n')"
        res = run_isolated(self._py(code), timeout_s=60, cwd=self.tmp, stdout_path=out_log, tail_chars=50)
        self.assertEqual(res.rc, 0, res.stderr_tail)
        self.assertEqual(out_log.stat().st_size, 5004)
        self.assertEqual(len(res.stdout_tail), 50)
        self.assertTrue(res.stdout_tail.endswith("END\n"))

    def test_tail_chars_limits_pipe_output(self) -> None:
        res = run_isolated(self._py("import sys; sys.stdout.write('a' * 5000)"), timeout_s=60, cwd=self.tmp, tail_chars=100)
        self.assertEqual(res.rc, 0, res.stderr_tail)
        self.assertEqual(res.stdout_tail, "a" * 100)

    def test_env_is_passed_through_and_not_inherited(self) -> None:
        code = "import os; print(os.environ.get('PROBE_VAR')); print(os.environ.get('PROCUTIL_PARENT_ONLY'))"
        env = {"PROBE_VAR": "xyz"}
        for key in ("SYSTEMROOT", "PATH"):  # Windows 의 파이썬 자식이 DLL 을 찾는 데 필요한 것만 물려준다
            if key in os.environ:
                env[key] = os.environ[key]
        with mock.patch.dict(os.environ, {"PROCUTIL_PARENT_ONLY": "leak"}):
            res = run_isolated(self._py(code), timeout_s=60, cwd=self.tmp, env=env)
        self.assertEqual(res.rc, 0, res.stderr_tail)
        self.assertEqual(res.stdout_tail.split(), ["xyz", "None"])

    def test_missing_executable_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            run_isolated([str(self.tmp / "no-such-binary-xyz")], timeout_s=5, cwd=self.tmp)

    def test_any_exception_while_waiting_kills_the_child_and_reraises(self) -> None:
        """communicate 가 타임아웃이 아닌 예외(Ctrl-C 의 KeyboardInterrupt)로 풀려도 자식을 죽이고 그대로 올린다."""
        beat = self.tmp / "heartbeat.txt"
        started: list[bool] = []

        def interrupted(proc: subprocess.Popen, *a: object, **kw: object) -> None:  # type: ignore[type-arg]
            # 자식이 실제로 뜬 뒤에 Ctrl-C 가 온 상황을 만든다
            started.append(_wait_for_file(beat))
            raise KeyboardInterrupt("SIGINT")

        with mock.patch.object(subprocess.Popen, "communicate", interrupted):
            with self.assertRaises(KeyboardInterrupt) as cm:
                run_isolated(self._py(_HEARTBEAT_SRC, str(beat)), timeout_s=120, cwd=self.tmp,
                             stdout_path=self.tmp / "o.log", stderr_path=self.tmp / "e.log")
        self.assertEqual(str(cm.exception), "SIGINT")
        self.assertEqual(started, [True], "자식이 뜬 뒤에 중단돼야 한다")
        self.assertTrue(_stopped_growing(beat), "예외로 빠져나간 뒤 자식이 살아 있다")

    def test_isolation_arguments_match_platform(self) -> None:
        """POSIX 는 예전과 같은 Popen 격리 인자(새 세션 + PDEATHSIG preexec), 비-POSIX 는 없음."""
        kwargs = procutil._isolation_kwargs()
        if os.name == "posix":
            self.assertTrue(procutil.IS_POSIX)
            self.assertEqual(set(kwargs), {"start_new_session", "preexec_fn"})
            self.assertIs(kwargs["start_new_session"], True)
            self.assertTrue(callable(kwargs["preexec_fn"]))
        else:
            self.assertFalse(procutil.IS_POSIX)
            self.assertEqual(kwargs, {})


class TerminateHandlersTest(unittest.TestCase):
    """install_terminate_handlers: 기본 처리인 시그널만 바꾸고 SIG_IGN(nohup)은 유지한다."""

    def test_terminate_signals_match_platform(self) -> None:
        self.assertEqual(TERMINATE_SIGNALS[0], signal.SIGTERM)
        if hasattr(signal, "SIGHUP"):
            self.assertEqual(TERMINATE_SIGNALS, (signal.SIGTERM, signal.SIGHUP))
        else:  # Windows: SIGHUP 이 없다
            self.assertEqual(TERMINATE_SIGNALS, (signal.SIGTERM,))

    def test_sigterm_handler_and_ignored_signal_on_any_platform(self) -> None:
        saved = {s: signal.getsignal(s) for s in TERMINATE_SIGNALS}
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertEqual(install_terminate_handlers((signal.SIGTERM,)), ["SIGTERM"])
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            with self.assertRaises(KeyboardInterrupt) as cm:
                handler(int(signal.SIGTERM), None)  # type: ignore[operator]
            self.assertEqual(str(cm.exception), "SIGTERM")
            # 이미 무시된 시그널은 건드리지 않는다
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            self.assertEqual(install_terminate_handlers((signal.SIGTERM,)), [])
            self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_IGN)
            # 기본 인자(TERMINATE_SIGNALS)로도 설치된다
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            self.assertIn("SIGTERM", install_terminate_handlers())
        finally:
            for s, h in saved.items():
                signal.signal(s, h if h is not None else signal.SIG_DFL)

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "SIGHUP 이 없는 플랫폼(Windows)")
    def test_installs_default_and_keeps_ignored(self) -> None:
        saved = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGHUP)}
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            signal.signal(signal.SIGHUP, signal.SIG_IGN)
            self.assertEqual(install_terminate_handlers(), ["SIGTERM"])
            self.assertEqual(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
            handler = signal.getsignal(signal.SIGTERM)
            self.assertTrue(callable(handler))
            with self.assertRaises(KeyboardInterrupt) as cm:
                handler(int(signal.SIGTERM), None)  # type: ignore[operator]
            self.assertEqual(str(cm.exception), "SIGTERM")
        finally:
            for s, h in saved.items():
                signal.signal(s, h)


class HelpersTest(unittest.TestCase):
    def test_signal_name_of(self) -> None:
        self.assertIsNone(signal_name_of(0))
        self.assertIsNone(signal_name_of(2))
        self.assertEqual(signal_name_of(-9), "SIGKILL")
        self.assertEqual(signal_name_of(-11), "SIGSEGV")
        self.assertEqual(signal_name_of(-200), "SIG200")

    def test_tail_text(self) -> None:
        self.assertEqual(tail_text(None, 10), "")
        self.assertEqual(tail_text(b"abcdef", 3), "def")
        self.assertEqual(tail_text("abcdef", 100), "abcdef")
        self.assertEqual(tail_text(b"\xff\xfeok", 10), "��ok")


class PathsTest(unittest.TestCase):
    """harness/paths.py: 턴별 raw·산출물 경로(설계 v3 §9.1·§9.2), 원자적 쓰기의 LF 유지, 비-POSIX 의 PYTHON_EXE."""

    TAG = "args-positional_hint-off_fb-compiler"

    def test_turn_zero_paths_are_the_existing_sample_paths(self) -> None:
        raw = paths.raw_turn_path("results/raw", "r1", "gemm", "numpy", self.TAG, 3, 0)
        self.assertEqual(raw, paths.raw_sample_path("results/raw", "r1", "gemm", "numpy", self.TAG, 3))
        self.assertEqual(raw.name, "sample_3.json")
        built = paths.built_turn_dir("results/built", "r1", "gemm", "numpy", self.TAG, 3, 0)
        self.assertEqual(built, paths.built_sample_dir("results/built", "r1", "gemm", "numpy", self.TAG, 3))
        self.assertEqual(built.name, "sample_3")

    def test_repair_turn_paths(self) -> None:
        base = paths.raw_sample_path("results/raw", "r1", "gemm", "numpy", self.TAG, 3)
        for turn in (1, 2, 3):
            raw = paths.raw_turn_path("results/raw", "r1", "gemm", "numpy", self.TAG, 3, turn)
            self.assertEqual(raw, base.parent / f"sample_3.turn{turn}.json")
            built = paths.built_turn_dir("results/built", "r1", "gemm", "numpy", self.TAG, 3, turn)
            self.assertEqual(built, paths.built_sample_dir("results/built", "r1", "gemm", "numpy", self.TAG, 3) / f"turn_{turn}")
        # 상대 디렉터리는 저장소 루트 기준, 절대 디렉터리는 그대로
        self.assertEqual(paths.raw_turn_path("results/raw", "r", "t", "l", "g", 0, 1).parents[4], paths.ROOT / "results" / "raw")
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(paths.built_turn_dir(tmp, "r", "t", "l", "g", 0, 2), Path(tmp) / "r" / "t" / "l" / "g" / "sample_0" / "turn_2")

    def test_invalid_turn_is_rejected(self) -> None:
        for bad in (-1, True, "1", 1.0, None, np.int64(-1), np.float64(1.0), np.bool_(True)):
            with self.assertRaises(ValueError, msg=repr(bad)):
                paths.raw_turn_path("results/raw", "r", "t", "l", "g", 0, bad)  # type: ignore[arg-type]
            with self.assertRaises(ValueError, msg=repr(bad)):
                paths.built_turn_dir("results/built", "r", "t", "l", "g", 0, bad)  # type: ignore[arg-type]

    def test_turn_accepts_numpy_integers(self) -> None:
        """회귀: ``isinstance(turn, int)`` 검사가 pandas/NumPy 로 읽은 행의 turn(``numpy.int64``)을 거부했다."""
        for turn in (np.int64(2), np.int32(2), np.uint8(2)):
            with self.subTest(type=type(turn).__name__):
                raw = paths.raw_turn_path("results/raw", "r", "t", "l", "g", 5, turn)  # type: ignore[arg-type]
                self.assertEqual(raw, paths.raw_turn_path("results/raw", "r", "t", "l", "g", 5, 2))
                self.assertEqual(raw.name, "sample_5.turn2.json")
                built = paths.built_turn_dir("results/built", "r", "t", "l", "g", 5, turn)  # type: ignore[arg-type]
                self.assertEqual(built, paths.built_turn_dir("results/built", "r", "t", "l", "g", 5, 2))
                self.assertEqual((built.parent.name, built.name), ("sample_5", "turn_2"))
        zero = np.int64(0)
        self.assertEqual(paths.raw_turn_path("results/raw", "r", "t", "l", "g", 5, zero),  # type: ignore[arg-type]
                         paths.raw_sample_path("results/raw", "r", "t", "l", "g", 5))
        self.assertEqual(paths.built_turn_dir("results/built", "r", "t", "l", "g", 5, zero),  # type: ignore[arg-type]
                         paths.built_sample_dir("results/built", "r", "t", "l", "g", 5))

    def test_rel_to_root_uses_forward_slashes_on_every_platform(self) -> None:
        """회귀: Windows 에서 ``results\\raw\\…`` 로 기록돼 행의 raw_path·built_path 를 Linux 에서 경로로 쓸 수 없었다."""
        # 저장소 안의(만들지는 않는) 디렉터리를 쓴다: results/ 가 다른 디스크로의 심볼릭 링크인 머신에서도 ROOT 안의 경로다
        raw = paths.raw_turn_path("tests/_rel_to_root/raw", "r1", "gemm", "numpy", self.TAG, 0, 1)
        self.assertEqual(paths.rel_to_root(raw), f"tests/_rel_to_root/raw/r1/gemm/numpy/{self.TAG}/sample_0.turn1.json")
        built = paths.built_turn_dir("tests/_rel_to_root/built", "r1", "gemm", "numpy", self.TAG, 0, 2)
        self.assertEqual(paths.rel_to_root(built), f"tests/_rel_to_root/built/r1/gemm/numpy/{self.TAG}/sample_0/turn_2")
        self.assertEqual(paths.rel_to_root(paths.ROOT / "harness" / "paths.py"), "harness/paths.py")
        self.assertEqual(paths.resolve(paths.rel_to_root(raw)), raw, "상대 문자열은 ROOT 기준으로 되읽힌다")
        # ROOT 밖의 경로는 절대 문자열 그대로다
        with tempfile.TemporaryDirectory() as tmp:
            outside = Path(tmp).resolve() / "x.json"
            self.assertEqual(paths.rel_to_root(outside), str(outside))

    def test_raw_turn_re(self) -> None:
        self.assertEqual(paths.RAW_TURN_RE, r"^sample_(\d+)(?:\.turn(\d+))?\.json$")
        pattern = re.compile(paths.RAW_TURN_RE)
        self.assertEqual(pattern.match("sample_0.json").groups(), ("0", None))  # type: ignore[union-attr]
        self.assertEqual(pattern.match("sample_12.turn3.json").groups(), ("12", "3"))  # type: ignore[union-attr]
        for name in ("sample_0.prev1.json", "sample_0.turn.json", "sample_x.json", "sample_0.turn1.json.bak", "xsample_0.json",
                     "sample_0.turn-1.json", "sample_.json", "sample_0.JSON"):
            self.assertIsNone(pattern.match(name), name)
        self.assertEqual(paths.parse_raw_turn_name("sample_7.json"), (7, 0))
        self.assertEqual(paths.parse_raw_turn_name("sample_7.turn2.json"), (7, 2))
        self.assertIsNone(paths.parse_raw_turn_name("sample_7.prev1.json"))
        # 경로 헬퍼가 만든 이름은 정규식으로 되읽힌다
        for turn in (0, 1, 3):
            name = paths.raw_turn_path("results/raw", "r", "t", "l", "g", 5, turn).name
            self.assertEqual(paths.parse_raw_turn_name(name), (5, turn))

    def test_atomic_writes_keep_lf_and_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            text_path = Path(tmp) / "sub" / "kernel.c"
            paths.atomic_write_text(text_path, "int x;\n/* 한글 */\n")
            self.assertEqual(text_path.read_bytes(), "int x;\n/* 한글 */\n".encode("utf-8"))
            json_path = Path(tmp) / "result.json"
            paths.atomic_write_json(json_path, {"detail": "누적", "n": 1})
            raw = json_path.read_bytes()
            self.assertNotIn(b"\r", raw)
            self.assertTrue(raw.endswith(b"}\n"))
            self.assertEqual(json.loads(raw.decode("utf-8")), {"detail": "누적", "n": 1})
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["result.json", "sub"], "임시 파일이 남으면 안 된다")

    def test_python_exe_exists_and_import_is_quiet_without_posix_venv(self) -> None:
        self.assertTrue(paths.PYTHON_EXE.exists())
        venv_python = paths.ROOT / ".venv" / "bin" / "python"
        if venv_python.exists():
            self.assertEqual(paths.PYTHON_EXE, venv_python)
        else:
            self.assertEqual(paths.PYTHON_EXE, Path(sys.executable).resolve())
        res = run_isolated([sys.executable, "-c", "import harness.paths as p; print(p.PYTHON_EXE)"], timeout_s=60, cwd=ROOT,
                           env={**os.environ, "PYTHONPATH": str(ROOT)})
        self.assertEqual(res.rc, 0, res.stderr_tail)
        if os.name != "posix":
            # 개발용 Windows: .venv/bin/python 이 없는 것이 정상이라 경고하지 않는다
            self.assertEqual(res.stderr_tail, "")
            self.assertEqual(Path(res.stdout_tail.strip()), Path(sys.executable).resolve())


class CliTest(unittest.TestCase):
    """``python -m harness.procutil`` (진단용 CLI)와 CLI 진입점의 스트림 정리(``setup_stdio``)."""

    def test_setup_stdio_makes_pipes_utf8_and_never_strict(self) -> None:
        class FakeStream:
            def __init__(self, tty: bool) -> None:
                self.tty = tty
                self.calls: list[dict[str, Any]] = []

            def isatty(self) -> bool:
                return self.tty

            def reconfigure(self, **kwargs: Any) -> None:
                self.calls.append(kwargs)

        pipe, console = FakeStream(False), FakeStream(True)
        with mock.patch.object(sys, "stdout", pipe), mock.patch.object(sys, "stderr", console):
            procutil.setup_stdio()
        self.assertEqual(pipe.calls, [{"encoding": "utf-8", "errors": "backslashreplace"}])
        self.assertEqual(console.calls, [{"errors": "backslashreplace"}], "콘솔의 인코딩은 바꾸지 않는다")
        # reconfigure 가 없는 스트림(StringIO 로 바뀐 경우)·없는 스트림(None)은 그대로 둔다
        with mock.patch.object(sys, "stdout", io.StringIO()), mock.patch.object(sys, "stderr", None):
            procutil.setup_stdio()

    def test_main_functions_do_not_reconfigure_the_callers_streams(self) -> None:
        """``setup_stdio`` 는 진입점(``__main__``)에서만 부른다: 같은 프로세스에서 ``main()`` 을 부른 쪽의 스트림은 그대로다."""
        from harness import sigs

        with mock.patch.object(procutil, "setup_stdio") as setup, contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(paths.main(["--json"]), 0)
            self.assertEqual(procutil.main(["--timeout", "60", "--", sys.executable, "-c", "print('hi')"]), 0)
            self.assertEqual(sigs.main(["--task", "gemm", "--backend", "c"]), 0)
        setup.assert_not_called()
        self.assertIn('"ROOT"', out.getvalue())
        self.assertIn('"stdout_tail": "hi', out.getvalue())

    def test_main_reports_the_result_as_json_and_maps_the_exit_code(self) -> None:
        with tempfile.TemporaryDirectory(prefix="procutil_cli_") as tmp:
            with contextlib.redirect_stdout(io.StringIO()) as out:
                rc = procutil.main(["--timeout", "60", "--cwd", tmp, "--", sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"])
        doc = json.loads(out.getvalue())
        self.assertEqual(rc, 1)
        self.assertEqual((doc["rc"], doc["timed_out"], doc["signal_name"], doc["stdout_tail"].strip()), (3, False, None, "hi"))
        self.assertEqual(set(doc), {"rc", "timed_out", "elapsed_s", "stdout_tail", "stderr_tail", "signal_name"})
        with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(io.StringIO()):
            procutil.main(["--timeout", "5"])  # 실행할 명령이 없다
        self.assertEqual(cm.exception.code, 2)

    def test_entry_point_prints_child_output_that_the_stream_encoding_cannot_encode(self) -> None:
        """회귀: 꼬리에 로캘 코드페이지로 인코딩할 수 없는 글자(깨진 바이트의 대체 문자 U+FFFD, ``—`` 등)가 있으면
        ``python -m harness.procutil … | cat`` 이 JSON 을 찍다가 ``UnicodeEncodeError`` 로 죽었다(cp949 파이프에서 실측).

        ``PYTHONIOENCODING=ascii`` 로 어느 플랫폼에서나 같은 상황(스트림 인코딩이 꼬리의 글자를 담지 못함)을 만든다."""
        child = ("import sys; sys.stdout.buffer.write('한글 — 출력'.encode('utf-8')); "
                 "sys.stderr.buffer.write(b'\\xff\\xfe')")
        env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "ascii"}
        with tempfile.TemporaryDirectory(prefix="procutil_cli_") as tmp:
            out_path = Path(tmp) / "out.json"
            res = run_isolated([sys.executable, "-m", "harness.procutil", "--timeout", "60", "--", sys.executable, "-c", child],
                               timeout_s=120, cwd=ROOT, env=env, stdout_path=out_path)
            self.assertEqual(res.rc, 0, res.stderr_tail)
            doc = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(doc["stdout_tail"], "한글 — 출력")
        self.assertEqual(doc["stderr_tail"], "��")
        self.assertEqual((doc["rc"], doc["timed_out"]), (0, False))


class SupportMarkersTest(unittest.TestCase):
    """``tests/_support.py`` 의 ``requires_modules``: 없는 패키지는 비-Linux 에서만 건너뛰고 Linux 에서는 실패시킨다.

    회귀: LLM SDK 가 없을 때 모듈 전체를 플랫폼과 무관하게 skip 해, Linux 에서 ``pip install -r requirements.txt`` 가 깨져도
    ``make unittest`` 가 초록으로 끝났다(예전에는 임포트 오류였다). 모듈 단위 skip 은 같은 모듈의 SDK 가 필요 없는 테스트까지
    조용히 빼므로 표시는 클래스·메서드에 붙인다."""

    MISSING = "no_such_package_for_support_test"

    @staticmethod
    def run_marked(marker: Any, *, on_class: bool) -> tuple[unittest.TestResult, list[str]]:
        ran: list[str] = []
        if on_class:
            @marker
            class Inner(unittest.TestCase):
                def test_a(self) -> None:
                    ran.append("a")

                def test_b(self) -> None:
                    ran.append("b")
        else:
            class Inner(unittest.TestCase):  # type: ignore[no-redef]
                @marker
                def test_a(self) -> None:
                    ran.append("a")

                def test_b(self) -> None:
                    ran.append("b")
        result = unittest.TestResult()
        unittest.defaultTestLoader.loadTestsFromTestCase(Inner).run(result)
        return result, ran

    def test_installed_modules_change_nothing(self) -> None:
        for is_linux in (False, True):
            with mock.patch.object(_support, "IS_LINUX", is_linux):
                marker = requires_modules("json", "unittest")
            for on_class in (True, False):
                result, ran = self.run_marked(marker, on_class=on_class)
                self.assertEqual((ran, result.skipped, result.errors, result.failures), (["a", "b"], [], [], []))

    def test_missing_module_is_skipped_off_linux(self) -> None:
        with mock.patch.object(_support, "IS_LINUX", False):
            marker = requires_modules("json", self.MISSING)
        result, ran = self.run_marked(marker, on_class=True)
        self.assertEqual((ran, len(result.skipped), result.errors, result.failures), ([], 2, [], []))
        self.assertIn(self.MISSING, result.skipped[0][1])
        self.assertNotIn("json", result.skipped[0][1], "있는 패키지는 이유에 적지 않는다")
        result, ran = self.run_marked(marker, on_class=False)
        self.assertEqual((ran, len(result.skipped), result.errors), (["b"], 1, []), "표시한 메서드만 건너뛴다")

    def test_missing_module_fails_on_linux_instead_of_skipping(self) -> None:
        with mock.patch.object(_support, "IS_LINUX", True):
            marker = requires_modules(self.MISSING)
        result, ran = self.run_marked(marker, on_class=True)
        self.assertEqual((ran, result.skipped, len(result.errors)), ([], [], 1), "클래스 전체가 setUpClass 에서 실패한다")
        self.assertFalse(result.wasSuccessful())
        self.assertIn("ModuleNotFoundError", result.errors[0][1])
        self.assertIn(self.MISSING, result.errors[0][1])
        result, ran = self.run_marked(marker, on_class=False)
        self.assertEqual((ran, result.skipped, len(result.errors)), (["b"], [], 1), "표시한 메서드만 실패한다")
        self.assertIn("ModuleNotFoundError", result.errors[0][1])

    def test_marker_needs_a_module_name(self) -> None:
        with self.assertRaises(ValueError):
            requires_modules()


if __name__ == "__main__":
    unittest.main()
