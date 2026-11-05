"""procutil.run_isolated 테스트: 타임아웃 뒤, 그리고 부모가 Ctrl-C·SIGTERM·SIGKILL 로 죽은 뒤에도
프로세스 그룹(손자 포함)이 비어 있어야 한다."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from harness.procutil import ProcResult, install_terminate_handlers, run_isolated, signal_name_of, tail_text

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


class TerminateHandlersTest(unittest.TestCase):
    """install_terminate_handlers: 기본 처리인 시그널만 바꾸고 SIG_IGN(nohup)은 유지한다."""

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


if __name__ == "__main__":
    unittest.main()
