"""harness/run.py 의 종료 규칙(verdict)과 기준선 건너뜀 사전 평가(needs_processing) 테스트."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from harness.config import load_config
from harness.procutil import ProcResult
from harness.run import HarnessError, RunContext, WorkerOutcome, needs_processing, sha256_file


def _proc(rc: int, *, timed_out: bool = False, signal_name: str | None = None) -> ProcResult:
    return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.1, stdout_tail="", stderr_tail="stderr tail", signal_name=signal_name)


def _outcome(rc: int, doc: dict | None, **kw) -> WorkerOutcome:
    return WorkerOutcome(stage="check", proc=_proc(rc, **kw), doc=doc, out_path=Path("/nonexistent/check.json"), stdout_bytes=0, timeout_s=180.0)


class VerdictTest(unittest.TestCase):
    """설계 §8-6: harness_error 는 worker_error 가 있거나 결과 문서 없이 rc=3 일 때만."""

    def test_worker_error_in_doc_is_harness_error(self):
        with self.assertRaises(HarnessError):
            _outcome(3, {"worker_error": "Traceback ...", "complete": False}).verdict()

    def test_rc3_without_any_doc_is_harness_error(self):
        with self.assertRaises(HarnessError):
            _outcome(3, None).verdict()
        with self.assertRaises(HarnessError):
            _outcome(3, {}).verdict()

    def test_rc3_with_partial_doc_is_kernel_exit_not_harness_error(self):
        # LLM 커널이 exit(3) 을 부른 경우: 워커는 in_progress 까지 썼지만 worker_error 는 없다 → runtime_crash
        doc = {"complete": False, "in_progress": {"test": "tiny", "call_idx": 0}}
        kind, detail = _outcome(3, doc).verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("rc=3", detail)

    def test_signal_is_runtime_crash_with_location(self):
        doc = {"complete": False, "in_progress": {"test": "rect", "call_idx": 1}}
        kind, detail = _outcome(-11, doc, signal_name="SIGSEGV").verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("SIGSEGV", detail)
        self.assertIn("rect call 1", detail)

    def test_timeout(self):
        kind, _ = _outcome(-9, {"complete": False}, timed_out=True, signal_name="SIGKILL").verdict()
        self.assertEqual(kind, "timeout")

    def test_rc0_incomplete_is_runtime_crash(self):
        kind, detail = _outcome(0, {"complete": False}).verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("complete", detail)

    def test_complete_is_ok(self):
        self.assertIsNone(_outcome(0, {"complete": True, "classification": "pass"}).verdict())


class NeedsProcessingTest(unittest.TestCase):
    """기준선 재측정 여부를 정하는 사전 평가가 process_sample 의 건너뜀 조건과 같은 답을 내야 한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        cfg = load_config()
        self.cfg = replace(cfg, paths=replace(cfg.paths, built_dir=root / "built", raw_dir=root / "raw"))
        self.raw = root / "raw" / "r1" / "gemm" / "numpy" / "args-positional_hint-off_fb-compiler" / "sample_0.json"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text(json.dumps({"language": "numpy", "sample_idx": 0, "task": "gemm",
                                        "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"}}))

    def _ctx(self, force: bool = False) -> RunContext:
        return RunContext(cfg=self.cfg, machine={}, static=None, fingerprint={}, env_hash="x", run_id="r1",
                          skip_bench=False, force=force, version="test")

    def _result_path(self) -> Path:
        return self.cfg.paths.built_dir / "r1" / "gemm" / "numpy" / "args-positional_hint-off_fb-compiler" / "sample_0" / "result.json"

    def test_no_result_json_needs_processing(self):
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertFalse(self._result_path().parent.exists(), "사전 평가는 디렉터리를 만들지 않는다")

    def test_matching_result_json_skips(self):
        rp = self._result_path(); rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps({"raw_sha256": sha256_file(self.raw), "row": {}}))
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertTrue(needs_processing(self._ctx(force=True), "gemm", self.raw), "--force 는 항상 처리")

    def test_changed_raw_needs_processing(self):
        rp = self._result_path(); rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps({"raw_sha256": sha256_file(self.raw), "row": {}}))
        self.raw.write_text(self.raw.read_text() + " ")
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw))

    def test_unknown_language_is_not_pending(self):
        self.raw.write_text(json.dumps({"language": "fortran", "sample_idx": 0, "conditions": {}}))
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))


if __name__ == "__main__":
    unittest.main()
