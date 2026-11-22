"""수정 루프(harness/repair.py) 테스트 (설계 v3 §9.3).

- 진단(``build_diagnostics``): 단계별 가지를 실제 실행에서 나온 것과 같은 꼴의 단계 문서로 본다. ``harness/classify.py`` 의
  어떤 패턴 이름·판정 이름도, 참조에서 나온 값도 수정 요청에 들어가지 않는다.
- 대상 선택, 턴 1·턴 2 의 대화, 린트 절의 세 변형, raw 의 키, 시드 복사, 종료 코드.

``harness.run`` 의 재사용 API(§10.1: ``setup_run``/``process_sample``/``finish_run``)는 가짜(``FakeRun``)로 바꿔 끼운다 — 계약만
본다. LLM 은 dummy 제공자(임시 디렉터리의 응답 파일)와 가짜 클라이언트만 쓴다: 실제 API 호출·네트워크·워커 실행 없음.
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import json
import os
import re
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

from harness import classify, generate, llm_client, repair
from harness.build.c import ToolchainMissing
from harness.config import Config, load_config
from harness.llm_client import LLMResponse
from harness.paths import DEFAULT_EXPERIMENT_YAML, built_turn_dir, cond_tag, raw_turn_path, rel_to_root
from harness.schema import RunRow

TEMPLATE = repair.load_template()
TAG = "args-positional_hint-off_fb-compiler"
TAG_LINT = "args-positional_hint-off_fb-compiler+lint"
RAW_KEYS = [
    "run_id", "task", "language", "backend", "sample_idx", "turn", "conditions", "provider", "model_requested", "model_served",
    "sampling", "max_tokens", "prompt", "prompt_hash", "response_text", "stop_reason", "raw_stop_reason", "stop_details_category",
    "n_thinking_blocks", "n_text_blocks", "tokens_in", "tokens_out", "tokens_thinking", "tokens_text", "latency_ms", "request_id",
    "base_url", "api_key_present", "sdk_version", "error", "created_at", "harness_version", "config_snapshot",
]

# ----------------------------------------------------------------------------- 단계 문서(실제 실행에서 나온 꼴)

GEMM_SHAPES = {"tiny": {"M": 7, "N": 5, "K": 3}, "rect": {"M": 200, "N": 130, "K": 312}, "small": {"M": 256, "N": 256, "K": 256}}
LARGE = {"M": 1024, "N": 1024, "K": 1024}

# gcc 15 (LC_ALL=C) 의 컴파일 오류: 더미 c_seq/1.md 의 세미콜론 누락
GCC_LOG = """\
kernel.c: In function 'gemm':
kernel.c:6:13: error: expected ',' or ';' before 'for'
    6 |             for (int k = 0; k < K; k++)
      |             ^~~
kernel.c:6:29: error: 'k' undeclared (first use in this function)
    6 |             for (int k = 0; k < K; k++)
      |                             ^
kernel.c:6:29: note: each undeclared identifier is reported only once for each function it appears in
kernel.c:6:39: error: expected ';' before ')' token
    6 |             for (int k = 0; k < K; k++)
      |                                       ^
      |                                       ;
"""
# 컴파일은 됐고(벡터화 리마크) 링크에서 실패: 링크하지 않는 라이브러리의 함수 호출
LINK_LOG = """\
kernel.c:9:27: optimized: loop vectorized using 32 byte vectors
kernel.c:7:23: missed: couldn't vectorize loop
kernel.c:7:23: missed: not vectorized: unsupported control flow in loop.
/usr/bin/ld: /tmp/ccJ7x2aB.o: in function `gemm':
/home/ustar/dev/graduate_project/results/built/r1/gemm/c_seq/args-positional_hint-off_fb-compiler/sample_3/kernel.c:12:(.text+0x5e): undefined reference to `cblas_sgemm'
collect2: error: ld returned 1 exit status
"""
CARGO_LOG = """\
   Compiling kernel v0.1.0 (/home/ustar/dev/graduate_project/results/built/r1/gemm/rust/args-positional_hint-off_fb-compiler/sample_1)
error[E0425]: cannot find value `sum` in this scope
 --> src/lib.rs:9:13
  |
9 |             sum += *A.add(i * k + p) * *B.add(p * n + j);
  |             ^^^ not found in this scope

error: could not compile `kernel` (lib) due to 1 previous error
"""
MATMUL_MESSAGE = ("matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) "
                  "(size 7 is different from 5)")
SAMPLE_PATH = "/home/ustar/dev/graduate_project/results/built/r1/gemm/numpy/args-positional_hint-off_fb-compiler/sample_3"
KERNEL_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ustar/dev/graduate_project/harness/worker.py", line 415, in call_kernel\n'
    "    ret = handle.invoke()\n"
    '  File "/home/ustar/dev/graduate_project/harness/kernels/pyfunc.py", line 92, in invoke\n'
    "    return self._call()\n"
    "           ~~~~~~~~~~^^\n"
    f'  File "{SAMPLE_PATH}/kernel.py", line 5, in gemm\n'
    "    np.matmul(B, A, out=C)\n"
    "    ~~~~~~~~~^^^^^^^^^^^^^\n"
    f"ValueError: {MATMUL_MESSAGE}\n")
# 기록된 실패의 트레이스백(워커가 남기는 꼴: Python 3.13, numba·halide 의 실제 출력 형식)
NUMBA_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ustar/dev/graduate_project/harness/worker.py", line 480, in call_kernel\n'
    "    ret = handle.invoke()\n"
    '  File "/home/ustar/dev/graduate_project/harness/kernels/pyfunc.py", line 92, in invoke\n'
    "    return self._call()\n"
    "           ~~~~~~~~~~^^\n"
    "ZeroDivisionError: division by zero\n")
HALIDE_MESSAGE = ('Error: Func "f" is computed at the following invalid location:\n  f.compute_at(g, zz);\n'
                  "Legal locations for this function are:\n  f.compute_root();\n  f.compute_at(g, Var::outermost());\n"
                  '  f.compute_at(g, y);\n  f.compute_at(g, x);\n"f" is used in the following places:\n for g.s0.y:\n  for g.s0.x:\n'
                  "   g uses f")
HALIDE_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ustar/dev/graduate_project/harness/kernels/halide_pipe.py", line 250, in _build\n'
    "    pipeline.compile_jit(rt.target)\n"
    "    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^\n"
    f"halide.halide_.HalideError: {HALIDE_MESSAGE}\n\n")
HALIDE_BUILDER_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ustar/dev/graduate_project/harness/kernels/halide_pipe.py", line 222, in _build\n'
    "    func = self._builder(**values) if self._keyword else self._builder(*values.values())\n"
    "                                                         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^\n"
    f'  File "{SAMPLE_PATH}/kernel.py", line 9, in gemm\n'
    "    C[x, y] = hl.sum(A[r, y] * B[x, r])\n"
    "                       ^\n"
    "NameError: name 'r' is not defined\n")
IMPORT_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "/home/ustar/dev/graduate_project/harness/kernels/pyfunc.py", line 55, in load_function\n'
    "    mspec.loader.exec_module(module)\n"
    "    ~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^\n"
    '  File "<frozen importlib._bootstrap_external>", line 762, in exec_module\n'
    '  File "<frozen importlib._bootstrap>", line 491, in _call_with_frames_removed\n'
    f'  File "{SAMPLE_PATH}/kernel.py", line 2, in <module>\n'
    "    import scipy.linalg\n"
    "ModuleNotFoundError: No module named 'scipy'\n")


def summary(**over: Any) -> dict[str, Any]:
    """``WorkerOutcome.summary()``: result.json 의 stages 에 들어가는 워커 단계 요약."""
    base = {"rc": 0, "timed_out": False, "signal": None, "elapsed_s": 0.5, "timeout_s": 180.0, "stdout_bytes": 0, "stdout_tail": "",
            "stderr_tail": "", "result_file": "check.json", "complete": True, "classification": "pass", "error_detail": ""}
    return {**base, **over}


def run_doc(classification: str = "pass", pattern: str = "", **over: Any) -> dict[str, Any]:
    """check 결과 문서의 호출 기록 하나(통과한 호출의 실제 값이 기본)."""
    base = {"classification": classification, "pattern": pattern, "max_abs_err": 1.2571034346819943e-07,
            "max_tol_ratio": 5.348906566298234e-05, "n_unwritten": 0, "nan_inf": False, "elapsed_ms": 0.0851, "exception": None,
            "return_type": "NoneType", "inputs_mutated": False, "oob_write": False}
    return {**base, **over}


def wrong_run(pattern: str = "mismatch", classification: str = "wrong_output", **over: Any) -> dict[str, Any]:
    return run_doc(classification, pattern, **{"max_abs_err": 3.1415926, "max_tol_ratio": 2996.875510289198, **over})


def shape_doc(name: str, runs: list[dict[str, Any]], *, nondeterministic: bool = False, dims: dict[str, int] | None = None) -> dict[str, Any]:
    """check 결과 문서의 테스트 하나(워커와 같은 규칙: 판정·패턴은 통과하지 못한 첫 호출의 것)."""
    dims = GEMM_SHAPES[name] if dims is None else dims
    failing = [run for run in runs if run["classification"] != "pass"]
    return {"name": name, **dims, "dims": dict(dims), "runs": runs, "classification": failing[0]["classification"] if failing else "pass",
            "pattern": failing[0]["pattern"] if failing else "", "nondeterministic": nondeterministic,
            "max_abs_err": max((run["max_abs_err"] or 0.0 for run in runs), default=None),
            "max_tol_ratio": max((run["max_tol_ratio"] or 0.0 for run in runs), default=None)}


def check_doc(tests: list[dict[str, Any]], classification: str, error_detail: str, **over: Any) -> dict[str, Any]:
    base = {"mode": "check", "backend": "python", "complete": True, "in_progress": None, "load_ok": True, "load_error": None,
            "classification": classification, "error_detail": error_detail, "task": "gemm", "tests": tests, "inputs_mutated": False,
            "oob_write": False, "nondeterministic": any(test["nondeterministic"] for test in tests)}
    return {**base, **over}


def bench_doc(**over: Any) -> dict[str, Any]:
    final = {"classification": "pass", "pattern": "", "max_abs_err": 6.1e-05, "max_tol_ratio": 0.0184, "n_fail": 0, "nan_inf": False,
             "n_unwritten": 0, "n_total": 1048576, "max_abs_ref": 123.456789, "oob_write": False, "return_type": None}
    base = {"mode": "bench", "test": "large", **LARGE, "dims": dict(LARGE), "complete": True, "in_progress": None, "warmup": 3,
            "repeats": 10, "warmup_ms": [80.1, 77.9, 77.5], "timed_ms": [77.2] * 10, "median_ms": 77.2, "nondeterministic_runs": 0,
            "nondeterministic_bitwise_runs": 0, "final_check": final, "inputs_mutated": False, "oob_write": False,
            "classification": "pass", "error_detail": "", "exception": None}
    doc = {**base, **over}
    doc["final_check"] = {**final, **(over.get("final_check") or {})}
    return doc


def build_stage(**over: Any) -> dict[str, Any]:
    """``BuildResult.to_dict()`` 에서 log 를 뺀 것(성공한 python 빌드가 기본)."""
    base = {"ok": True, "artifact": "kernel.py", "error_kind": "", "error_detail": "", "build_time_ms": 0.3,
            "command": ["python", "compile", "kernel.py"], "compiler": "python", "compiler_version": "3.14.4", "n_warnings": 0,
            "undefined_symbols": [], "forbidden_symbols": [], "thread_symbols": [], "march_resolved": "", "extra": {}}
    return {**base, **over}


def row_of(stage: str, error_type: str, detail: str = "", **over: Any) -> dict[str, Any]:
    """``result.json`` 의 ``row`` 에서 수정 루프가 읽는 부분."""
    return {"task": "gemm", "language": "numpy", "sample_idx": 1, "turn": 0, "valid_sample": True, "passed": False,
            "error_stage": stage, "error_type": error_type, "error_detail": detail, **over}


class DiagnosticsCase(unittest.TestCase):
    """샘플 디렉터리(임시)에 단계 파일을 놓고 ``build_diagnostics`` 를 부르는 틀."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def put(self, name: str, content: Any) -> None:
        text = content if isinstance(content, str) else json.dumps(content)
        (self.dir / name).write_text(text, encoding="utf-8")

    def check(self, doc: dict[str, Any], row: dict[str, Any], **stage_over: Any) -> str:
        self.put("check.json", doc)
        stages = {"check": summary(classification=doc["classification"], error_detail=doc["error_detail"], **stage_over)}
        return repair.build_diagnostics(row, stages, self.dir)

    def lines(self, text: str) -> list[str]:
        return text.splitlines()


# ----------------------------------------------------------------------------- 작은 헬퍼


class HelpersTest(unittest.TestCase):
    def test_basename_paths_keeps_file_names_only(self) -> None:
        cases = {
            "/home/ustar/dev/graduate_project/results/built/r1/sample_0/kernel.c:5:3: error: x": "kernel.c:5:3: error: x",
            "In file included from /usr/lib/gcc/x86_64-linux-gnu/15/include/immintrin.h:39,": "In file included from immintrin.h:39,",
            '  File "/home/ustar/x/harness/worker.py", line 415, in call_kernel': '  File "worker.py", line 415, in call_kernel',
            '  File "C:\\Users\\dev\\AppData\\Local\\Temp\\sample_2\\kernel.py", line 5': '  File "kernel.py", line 5',
            "/usr/bin/ld: /tmp/ccJ7x2aB.o: in function `gemm':": "ld: ccJ7x2aB.o: in function `gemm':",
            # 상대 경로와 코드는 그대로다
            "kernel.c:6:13: error: expected ';'": "kernel.c:6:13: error: expected ';'",
            " --> src/lib.rs:9:13": " --> src/lib.rs:9:13",
            "    C[i*N + j] = (a/b/c) + x / y / z;": "    C[i*N + j] = (a/b/c) + x / y / z;",
            "    return (total)/n/2": "    return (total)/n/2",
            "see https://example.org/a/b for details": "see https://example.org/a/b for details",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(repair.basename_paths(text), expected)

    def test_clip_keeps_the_head_within_the_limit(self) -> None:
        self.assertEqual(repair.clip("short"), "short")
        exact = "x" * repair.DIAGNOSTICS_MAX_CHARS
        self.assertEqual(repair.clip(exact), exact)
        clipped = repair.clip("line\n" * 2000)
        self.assertEqual(len(clipped) <= repair.DIAGNOSTICS_MAX_CHARS, True)
        self.assertTrue(clipped.startswith("line\nline\n"))
        self.assertTrue(clipped.endswith("[... truncated]"))

    def test_compiler_excerpt_starts_at_the_first_error(self) -> None:
        self.assertEqual(repair.compiler_excerpt(GCC_LOG).splitlines()[0], "kernel.c:6:13: error: expected ',' or ';' before 'for'")
        self.assertTrue(repair.compiler_excerpt(GCC_LOG).endswith("      |                                       ;"))
        link = repair.compiler_excerpt(LINK_LOG)
        self.assertEqual(link.splitlines(), ["kernel.c:12:(.text+0x5e): undefined reference to `cblas_sgemm'",
                                             "collect2: error: ld returned 1 exit status"])
        cargo = repair.compiler_excerpt(CARGO_LOG)
        self.assertEqual(cargo.splitlines()[:2], ["error[E0425]: cannot find value `sum` in this scope", " --> src/lib.rs:9:13"])
        self.assertNotIn("/home/ustar", cargo)

    def test_compiler_excerpt_without_an_error_line_is_the_tail(self) -> None:
        log = "".join(f"note {i}\n\n" for i in range(100))
        excerpt = repair.compiler_excerpt(log).splitlines()
        self.assertEqual(len(excerpt), repair.LOG_TAIL_LINES)
        self.assertEqual((excerpt[0], excerpt[-1]), ("note 70", "note 99"))
        self.assertEqual(repair.compiler_excerpt("-Werror=implicit is set\nwarning: unused variable\n"),
                         "-Werror=implicit is set\nwarning: unused variable")

    def test_process_outcome(self) -> None:
        self.assertIsNone(repair.process_outcome(summary()))
        self.assertEqual(repair.process_outcome(summary(timed_out=True, rc=-9, signal="SIGKILL", complete=False)), "timed out after 180 s")
        self.assertEqual(repair.process_outcome(summary(timed_out=True, timeout_s=0.5, complete=False)), "timed out after 0.5 s")
        self.assertEqual(repair.process_outcome(summary(rc=-11, signal="SIGSEGV", complete=False)), "crashed: signal SIGSEGV")
        self.assertEqual(repair.process_outcome(summary(rc=3, complete=False)), "crashed: the process exited with code 3")
        self.assertEqual(repair.process_outcome(summary(rc=0, complete=False)), "crashed: the process exited before the run was complete")

    def test_exception_text(self) -> None:
        text = repair.exception_text({"type": "ValueError", "message": MATMUL_MESSAGE, "traceback": KERNEL_TRACEBACK})
        lines = text.splitlines()
        self.assertEqual(lines[0], f"raised ValueError: {MATMUL_MESSAGE}")
        # 하네스의 프레임(worker.py call_kernel, kernels/pyfunc.py invoke)은 빠지고 커널의 프레임과 예외 줄이 남는다
        self.assertEqual(lines[1:], ["    Traceback (most recent call last):", '      File "kernel.py", line 5, in gemm',
                                     "        np.matmul(B, A, out=C)", "        ~~~~~~~~~^^^^^^^^^^^^^", f"    ValueError: {MATMUL_MESSAGE}"])
        self.assertNotIn("/home/ustar", text)
        self.assertNotIn("call_kernel", text)
        deep = "Traceback (most recent call last):\n" + "".join(f'  File "kernel.py", line {i}, in f{i}\n    f{i + 1}()\n' for i in range(40))
        tail = repair.exception_text({"type": "RecursionError", "message": "too deep\nsecond line", "traceback": deep + "RecursionError: too deep\n"})
        self.assertEqual(tail.splitlines()[0], "raised RecursionError: too deep")
        self.assertEqual(len(tail.splitlines()), 1 + repair.TRACEBACK_TAIL_LINES)
        self.assertIn("in f39", tail)
        self.assertNotIn("in f30\n", tail + "\n")

    def test_harness_frames_are_dropped_from_recorded_failures(self) -> None:
        # numba: @njit 커널 안의 ZeroDivisionError — 컴파일된 커널은 파이썬 프레임이 없어 하네스 프레임만 남는다 → 예외 줄만
        self.assertEqual(repair.exception_text({"type": "ZeroDivisionError", "message": "division by zero", "traceback": NUMBA_TRACEBACK}),
                         "raised ZeroDivisionError: division by zero")
        # halide: 스케줄이 틀린 파이프라인의 JIT 실패(halide_pipe.py _build 안) → 예외 줄과 여러 줄 메시지의 나머지
        text = repair.exception_text({"type": "HalideError", "message": HALIDE_MESSAGE, "traceback": HALIDE_TRACEBACK})
        self.assertEqual(text.splitlines(), [f"raised HalideError: {HALIDE_MESSAGE.splitlines()[0]}".rstrip(":"),
                                             *("    " + line for line in HALIDE_MESSAGE.splitlines()[1:])])
        self.assertNotIn("_build", text)
        self.assertNotIn("Traceback", text)
        # halide: LLM 의 빌드 함수 안에서 난 예외 → 하네스 프레임만 빠지고 커널 프레임은 남는다
        text = repair.exception_text({"type": "NameError", "message": "name 'r' is not defined", "traceback": HALIDE_BUILDER_TRACEBACK})
        kept = [line for i, line in enumerate(HALIDE_BUILDER_TRACEBACK.splitlines()) if i not in (1, 2, 3)]
        self.assertEqual(text.splitlines(), ["raised NameError: name 'r' is not defined", *("    " + repair.basename_paths(line) for line in kept)])
        self.assertIn('      File "kernel.py", line 9, in gemm', text.splitlines())

    def test_harness_files_are_recognised_wherever_the_repository_was(self) -> None:
        for path in ("/home/ustar/dev/graduate_project/harness/worker.py", "/srv/x/harness/kernels/pyfunc.py",
                     "C:\\graduation_work\\harness\\kernels\\halide_pipe.py"):
            self.assertTrue(repair.is_harness_file(path), path)
        for path in ("kernel.py", "<frozen importlib._bootstrap_external>", "/usr/lib/python3.12/site-packages/numba/core/dispatcher.py",
                     "/home/u/harness/results/built/r1/gemm/numpy/t/sample_0/kernel.py",  # 저장소가 harness 라는 디렉터리에 있어도
                     "/home/u/harness/worker_of_mine.py"):
            self.assertFalse(repair.is_harness_file(path), path)

    def test_exception_record_without_a_traceback_is_a_plain_failure(self) -> None:
        # 예외 없이 난 준비 실패(반환값 검사): 워커는 판정 이름을 type 자리에 적는다 — 그 이름은 내보내지 않는다
        record = {"type": "signature_mismatch", "message": "gemm returned an hl.Func with 1 dimension(s); the output C has 2", "traceback": ""}
        self.assertEqual(repair.exception_text(record), "failed: gemm returned an hl.Func with 1 dimension(s); the output C has 2")

    def test_output_wrong_reads_the_metrics_not_the_verdict(self) -> None:
        self.assertFalse(repair.output_wrong(run_doc()))
        self.assertTrue(repair.output_wrong(wrong_run()))
        self.assertTrue(repair.output_wrong(run_doc("numeric_instability", "nan_inf", nan_inf=True)))
        self.assertTrue(repair.output_wrong(run_doc("numeric_instability", "nan_inf", nan_inf=True, max_abs_err=None, max_tol_ratio=None)))
        # 출력은 맞는데 입력을 고쳤거나 버퍼 밖에 쓴 호출: 워커의 판정은 wrong_output 이지만 출력이 틀린 것은 아니다
        self.assertFalse(repair.output_wrong(run_doc("wrong_output", "inputs_mutated", inputs_mutated=True)))
        self.assertFalse(repair.output_wrong(run_doc("wrong_output", "out_of_bounds_write", oob_write=True)))
        # 정확 비교(정수 출력): 허용오차 배율이 없다
        self.assertFalse(repair.output_wrong(run_doc(max_abs_err=0.0, max_tol_ratio=None)))
        self.assertTrue(repair.output_wrong(run_doc("wrong_output", "mismatch", max_abs_err=3.0, max_tol_ratio=None)))


# ----------------------------------------------------------------------------- 진단: 단계별 가지


class ExtractDiagnosticsTest(DiagnosticsCase):
    def diagnose(self, detail: str) -> str:
        stages = {"extract": {"ok": False, "signature_ok": False, "signature_exact": False, "detail": detail, "n_blocks": 1}}
        return repair.build_diagnostics(row_of("extract", "signature_mismatch", detail), stages, self.dir)

    def test_signature_mismatch_states_what_was_received_and_expected(self) -> None:
        self.assertEqual(self.diagnose("param_mismatch:got (A, B, M, N, K) expected (A, B, C, M, N, K)"),
                         "The signature of the function does not match the required signature: got (A, B, M, N, K) expected "
                         "(A, B, C, M, N, K).")
        self.assertEqual(self.diagnose("param_mismatch:param 2 (B) type double* expected float*;preprocess_failed"),
                         "The signature of the function does not match the required signature: param 2 (B) type double* expected float*.")
        self.assertIn("5 params expected 6", self.diagnose("param_mismatch:5 params expected 6"))

    def test_each_failure_reason_is_spelled_out(self) -> None:
        expected = {
            "no_code_block": "No fenced code block",
            "function_not_found": "does not contain a definition of the required function",
            "function_not_found;multiple_blocks_chose:1": "does not contain a definition of the required function",
            "static_linkage": "defined `static`",
            "no_mangle_missing": "not marked `#[no_mangle]`",
            "param_mismatch:async_def": "defined with `async def`",
        }
        for detail, phrase in expected.items():
            with self.subTest(detail=detail):
                text = self.diagnose(detail)
                self.assertIn(phrase, text)
                self.assertEqual(len(text.splitlines()), 1)
                for token in ("no_code_block", "function_not_found", "static_linkage", "no_mangle_missing", "param_mismatch",
                              "async_def", "multiple_blocks_chose"):
                    self.assertNotIn(token, text)

    def test_unknown_reason_is_passed_on_rather_than_hidden(self) -> None:
        self.assertEqual(self.diagnose("some_new_reason"), "The code could not be taken from the reply: some_new_reason.")


class BuildDiagnosticsTest(DiagnosticsCase):
    def diagnose(self, stages: dict[str, Any], error_type: str = "compile_error") -> str:
        return repair.build_diagnostics(row_of("build", error_type, "whatever the row says"), stages, self.dir)

    def test_c_compile_error_is_the_compiler_output_from_the_first_error(self) -> None:
        self.put("build.log", GCC_LOG)
        build = build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="gcc",
                            error_detail="kernel.c:6:13: error: expected ',' or ';' before 'for' | kernel.c:6:29: error: 'k' undeclared")
        text = self.diagnose({"build": build})
        self.assertEqual(text.splitlines()[:2], ["Compiler output:", "kernel.c:6:13: error: expected ',' or ';' before 'for'"])
        self.assertIn("kernel.c:6:39: error: expected ';' before ')' token", text)
        self.assertNotIn("In function", text)  # 첫 오류 줄 앞은 뺀다

    def test_link_error_skips_the_vectorization_remarks(self) -> None:
        self.put("build.log", LINK_LOG)
        build = build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="gcc",
                            error_detail="collect2: error: ld returned 1 exit status")
        text = self.diagnose({"build": build})
        self.assertEqual(text, "Compiler output:\nkernel.c:12:(.text+0x5e): undefined reference to `cblas_sgemm'\n"
                               "collect2: error: ld returned 1 exit status")

    def test_rust_compile_error(self) -> None:
        self.put("build.log", CARGO_LOG)
        build = build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="rustc",
                            error_detail="error[E0425]: cannot find value `sum` in this scope --> src/lib.rs:9:13")
        text = self.diagnose({"build": build})
        self.assertTrue(text.startswith("Compiler output:\nerror[E0425]: cannot find value `sum` in this scope\n --> src/lib.rs:9:13\n"))
        self.assertTrue(text.endswith("error: could not compile `kernel` (lib) due to 1 previous error"))
        self.assertNotIn("Compiling kernel", text)

    def test_python_syntax_error_has_no_compiler_log(self) -> None:
        build = build_stage(ok=False, artifact=None, error_kind="compile_error", error_detail="SyntaxError: expected ':' (line 4)")
        self.assertEqual(self.diagnose({"build": build}), "The code does not compile: SyntaxError: expected ':' (line 4)")

    def test_compile_timeout(self) -> None:
        build = build_stage(ok=False, artifact=None, error_kind="timeout", compiler="gcc", error_detail="compile timeout after 60s")
        self.assertEqual(self.diagnose({"build": build}), "The build did not finish: compile timeout after 60s.")

    def test_missing_symbol_and_missing_numba_signature(self) -> None:
        build = build_stage(ok=False, artifact=None, error_kind="signature_mismatch", compiler="gcc", error_detail="symbol gemm not exported")
        self.assertEqual(self.diagnose({"build": build}, "signature_mismatch"),
                         "The build succeeded, but the required function is missing from the result: symbol gemm not exported.")
        build = build_stage(ok=False, artifact=None, error_kind="signature_mismatch", compiler="numba", error_detail="numba_signature_missing")
        text = self.diagnose({"build": build}, "signature_mismatch")
        self.assertIn("`njit` decorator called with the signature string as its first argument", text)
        self.assertNotIn("numba_signature_missing", text)
        build = build_stage(ok=False, artifact=None, error_kind="signature_mismatch", compiler="halide", error_detail="function_not_found")
        self.assertEqual(self.diagnose({"build": build}, "signature_mismatch"), "The module does not define the required function at top level.")

    def test_import_exception_is_the_exception_with_its_traceback_tail(self) -> None:
        detail = "ModuleNotFoundError: No module named 'scipy'"
        self.put("import.json", {"mode": "import", "complete": True, "load_ok": False, "load_error": detail, "classification": "compile_error",
                                 "error_detail": detail,
                                 "exception": {"type": "ModuleNotFoundError", "message": "No module named 'scipy'", "traceback": IMPORT_TRACEBACK}})
        stages = {"build": build_stage(), "import": summary(classification="compile_error", error_detail=detail)}
        lines = self.diagnose(stages).splitlines()
        self.assertEqual(lines[0], "Loading the built code failed: raised ModuleNotFoundError: No module named 'scipy'")
        self.assertIn('      File "kernel.py", line 2, in <module>', lines)
        self.assertEqual(lines[-1], "    ModuleNotFoundError: No module named 'scipy'")
        self.assertNotIn("/home/ustar", "\n".join(lines))

    def test_load_failure_without_an_exception_record_is_one_line(self) -> None:
        detail = "OSError: /home/ustar/dev/graduate_project/results/built/r1/sample_2/kernel.so: undefined symbol: cblas_sgemm"
        self.put("import.json", {"mode": "import", "complete": True, "load_ok": False, "load_error": detail, "classification": "compile_error",
                                 "error_detail": detail})
        stages = {"build": build_stage(compiler="gcc"), "import": summary(classification="compile_error", error_detail=detail)}
        self.assertEqual(self.diagnose(stages), "Loading the built code failed: OSError: kernel.so: undefined symbol: cblas_sgemm")

    def test_numba_function_compiled_lazily(self) -> None:
        self.put("import.json", {"mode": "import", "complete": True, "load_ok": False, "load_error": "numba_signature_missing",
                                 "classification": "signature_mismatch", "error_detail": "numba_signature_missing"})
        stages = {"build": build_stage(compiler="numba"), "import": summary(classification="signature_mismatch", error_detail="numba_signature_missing")}
        self.assertIn("is not compiled when the module is imported", self.diagnose(stages, "signature_mismatch"))

    def test_import_that_does_not_come_back(self) -> None:
        stages = {"build": build_stage(), "import": summary(timed_out=True, rc=-9, signal="SIGKILL", complete=False, classification=None)}
        self.assertEqual(self.diagnose(stages, "timeout"), "Loading the built code did not finish: timed out after 180 s.")
        stages = {"build": build_stage(compiler="gcc"), "import": summary(rc=-11, signal="SIGSEGV", complete=False, classification=None)}
        self.assertEqual(self.diagnose(stages, "runtime_crash"), "Loading the built code did not finish: crashed: signal SIGSEGV.")

    def test_long_compiler_output_is_cut_at_the_limit(self) -> None:
        self.put("build.log", "".join(f"kernel.c:{i}:1: error: unknown type name 'flaot{i}'\n" for i in range(400)))
        text = self.diagnose({"build": build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="gcc", error_detail="x")})
        self.assertEqual(len(text), repair.DIAGNOSTICS_MAX_CHARS)
        self.assertTrue(text.startswith("Compiler output:\nkernel.c:0:1: error: unknown type name 'flaot0'"))
        self.assertTrue(text.endswith("\n[... truncated]"))

    def test_missing_stage_record_is_a_harness_error(self) -> None:
        with self.assertRaises(repair.RepairError):
            self.diagnose({})
        with self.assertRaises(repair.RepairError):
            self.diagnose({"build": build_stage()})  # 빌드는 됐는데 import 기록이 없다
        with self.assertRaises(repair.RepairError):
            repair.build_diagnostics(row_of("generate", ""), {}, self.dir)


class CheckDiagnosticsTest(DiagnosticsCase):
    HEADER = "Correctness check, one line per shape that was run:"

    def test_one_line_per_shape_with_the_observed_outcome(self) -> None:
        tests = [shape_doc("tiny", [run_doc(), run_doc()]),
                 shape_doc("rect", [wrong_run(max_abs_err=3.1), wrong_run(max_abs_err=2.9)]),
                 shape_doc("small", [wrong_run(max_abs_err=0.2353), wrong_run(max_abs_err=0.2353)])]
        text = self.check(check_doc(tests, "wrong_output", "mismatch"), row_of("check", "wrong_output", "mismatch"))
        self.assertEqual(self.lines(text), [self.HEADER,
                                            "- shape (M=7, N=5, K=3): passed",
                                            "- shape (M=200, N=130, K=312): incorrect output (max abs error 3.1e+00)",
                                            "- shape (M=256, N=256, K=256): incorrect output (max abs error 2.4e-01)"])

    def test_counts_are_given_when_the_worker_recorded_them(self) -> None:
        tests = [shape_doc("tiny", [wrong_run(max_abs_err=3.1, n_fail=35, n_total=35), wrong_run(max_abs_err=3.0, n_fail=34, n_total=35)])]
        text = self.check(check_doc(tests, "wrong_output", "mismatch"), row_of("check", "wrong_output", "mismatch"))
        self.assertEqual(self.lines(text)[1], "- shape (M=7, N=5, K=3): incorrect output (max abs error 3.1e+00, 35 of 35 elements outside tolerance)")

    def test_non_finite_output(self) -> None:
        nan = run_doc("numeric_instability", "nan_inf", nan_inf=True)
        text = self.check(check_doc([shape_doc("tiny", [nan, nan])], "numeric_instability", "nan_inf"), row_of("check", "numeric_instability", "nan_inf"))
        self.assertEqual(self.lines(text)[1], "- shape (M=7, N=5, K=3): non-finite values in the output")

    def test_exception_ends_the_run_at_that_shape(self) -> None:
        crashed = run_doc("runtime_crash", f"ValueError: {MATMUL_MESSAGE}", max_abs_err=None, max_tol_ratio=None, n_unwritten=None, nan_inf=None,
                          inputs_mutated=None, oob_write=None,
                          exception={"type": "ValueError", "message": MATMUL_MESSAGE, "traceback": KERNEL_TRACEBACK})
        doc = check_doc([shape_doc("tiny", [crashed])], "runtime_crash", f"tiny: ValueError: {MATMUL_MESSAGE}")
        lines = self.lines(self.check(doc, row_of("check", "runtime_crash", doc["error_detail"])))
        self.assertEqual(lines[1], f"- shape (M=7, N=5, K=3): raised ValueError: {MATMUL_MESSAGE}")
        self.assertEqual(lines[2], "    Traceback (most recent call last):")
        self.assertEqual(lines[-1], f"    ValueError: {MATMUL_MESSAGE}")
        self.assertEqual(len(lines), 2 + 5)  # 하네스 프레임을 뺀 트레이스백: 머리 줄, 커널 프레임 세 줄, 예외 줄
        self.assertNotIn("invoke", "\n".join(lines))

    def test_signal_is_reported_at_the_shape_that_was_running(self) -> None:
        # 더미 c_openmp/1: 첫 호출에서 SIGSEGV. 워커는 호출 직전에 문서를 써 두었다(complete=false, in_progress)
        partial = check_doc([shape_doc("tiny", [run_doc(), run_doc()]),
                             {"name": "rect", **GEMM_SHAPES["rect"], "dims": dict(GEMM_SHAPES["rect"]), "runs": [], "classification": None,
                              "pattern": "", "nondeterministic": False}],
                            None, "", complete=False, in_progress={"test": "rect", "call_idx": 0})  # type: ignore[arg-type]
        text = self.check(partial, row_of("check", "runtime_crash", "signal SIGSEGV during rect call 0"),
                          rc=-11, signal="SIGSEGV", complete=False)
        self.assertEqual(self.lines(text), [self.HEADER, "- shape (M=7, N=5, K=3): passed",
                                            "- shape (M=200, N=130, K=312): crashed: signal SIGSEGV"])

    def test_timeout_is_reported_at_the_shape_that_was_running(self) -> None:
        partial = check_doc([{"name": "tiny", **GEMM_SHAPES["tiny"], "dims": dict(GEMM_SHAPES["tiny"]), "runs": [], "classification": None,
                              "pattern": "", "nondeterministic": False}],
                            None, "", complete=False, in_progress={"test": "tiny", "call_idx": 0})  # type: ignore[arg-type]
        text = self.check(partial, row_of("check", "timeout", "worker timeout after 180s during tiny call 0"),
                          timed_out=True, rc=-9, signal="SIGKILL", complete=False)
        self.assertEqual(self.lines(text), [self.HEADER, "- shape (M=7, N=5, K=3): timed out after 180 s"])

    def test_process_that_exits_outside_a_call(self) -> None:
        doc = check_doc([], None, "", complete=False, in_progress=None)  # type: ignore[arg-type]
        text = self.check(doc, row_of("check", "runtime_crash", "rc=1: ..."), rc=1, complete=False)
        self.assertEqual(self.lines(text), [self.HEADER, "- crashed: the process exited with code 1 (outside the calls of the function)"])

    def test_results_that_differ_between_two_calls(self) -> None:
        # 더미 c_openmp/2: 두 호출이 모두 틀리고 서로 다르다
        tests = [shape_doc("rect", [wrong_run(max_abs_err=3.0), wrong_run(max_abs_err=4.0)], nondeterministic=True),
                 shape_doc("small", [run_doc(), run_doc()], nondeterministic=True)]
        text = self.check(check_doc(tests, "race_condition", "nondeterministic:rect,small"),
                          row_of("check", "race_condition", "nondeterministic:rect,small"))
        self.assertEqual(self.lines(text)[1:], [
            "- shape (M=200, N=130, K=312): incorrect output (max abs error 4.0e+00); results differ between two identical calls",
            "- shape (M=256, N=256, K=256): results differ between two identical calls"])

    def test_modified_input_and_write_outside_the_buffer(self) -> None:
        mutated = run_doc("wrong_output", "inputs_mutated", inputs_mutated=True)
        overrun = run_doc("wrong_output", "out_of_bounds_write", oob_write=True)
        both = wrong_run("mismatch", inputs_mutated=True, oob_write=True, max_abs_err=0.43)
        tests = [shape_doc("tiny", [mutated, mutated]), shape_doc("rect", [overrun, overrun]), shape_doc("small", [both, both])]
        text = self.check(check_doc(tests, "wrong_output", "inputs_mutated", inputs_mutated=True, oob_write=True),
                          row_of("check", "wrong_output", "inputs_mutated"))
        self.assertEqual(self.lines(text)[1:], [
            "- shape (M=7, N=5, K=3): modified a read-only input",
            "- shape (M=200, N=130, K=312): wrote outside the output buffer",
            "- shape (M=256, N=256, K=256): incorrect output (max abs error 4.3e-01); modified a read-only input; wrote outside the output buffer"])

    def test_exact_comparison_task(self) -> None:
        wrong = run_doc("wrong_output", "mismatch", max_abs_err=3.0, max_tol_ratio=None)
        ok = run_doc(max_abs_err=0.0, max_tol_ratio=None)
        tests = [shape_doc("tiny", [ok, ok], dims={"n": 10, "n_bins": 4}), shape_doc("rect", [wrong, wrong], dims={"n": 100003, "n_bins": 1000})]
        text = self.check(check_doc(tests, "wrong_output", "mismatch"), row_of("check", "wrong_output", "mismatch", task="histogram"))
        self.assertEqual(self.lines(text)[1:], ["- shape (n=10, n_bins=4): passed",
                                                "- shape (n=100003, n_bins=1000): incorrect output (max abs error 3.0e+00)"])

    def test_halide_pipeline_that_cannot_be_built_for_a_shape(self) -> None:
        record = {"type": "signature_mismatch", "message": "gemm returned an hl.Func with 1 dimension(s); the output C has 2", "traceback": ""}
        crashed = run_doc("runtime_crash", record["message"], max_abs_err=None, max_tol_ratio=None, exception=record)
        doc = check_doc([shape_doc("tiny", [crashed])], "runtime_crash", "tiny: " + record["message"])
        text = self.check(doc, row_of("check", "runtime_crash", doc["error_detail"], language="halide_manual"))
        self.assertEqual(self.lines(text)[1], "- shape (M=7, N=5, K=3): failed: gemm returned an hl.Func with 1 dimension(s); the output C has 2")


class BenchDiagnosticsTest(DiagnosticsCase):
    HEADER = "The correctness check passed on every correctness shape. Timed run on the performance shape:"

    def bench(self, doc: dict[str, Any], row: dict[str, Any], **stage_over: Any) -> list[str]:
        self.put("bench.json", doc)
        stages = {"check": summary(), "bench": summary(classification=doc["classification"], error_detail=doc["error_detail"],
                                                        timeout_s=300.0, **stage_over)}
        return repair.build_diagnostics(row, stages, self.dir).splitlines()

    def test_wrong_output_on_the_performance_shape(self) -> None:
        doc = bench_doc(classification="wrong_output", error_detail="mismatch",
                        final_check={"classification": "wrong_output", "pattern": "mismatch", "max_abs_err": 12.5, "max_tol_ratio": 9040.2,
                                     "n_fail": 4096, "n_total": 1048576})
        self.assertEqual(self.bench(doc, row_of("bench", "wrong_output", "mismatch")), [
            self.HEADER, "- shape (M=1024, N=1024, K=1024): incorrect output (max abs error 1.2e+01, 4096 of 1048576 elements outside tolerance)"])

    def test_results_that_differ_between_timed_calls(self) -> None:
        doc = bench_doc(classification="race_condition", error_detail="nondeterministic_runs=3/10", nondeterministic_runs=3)
        self.assertEqual(self.bench(doc, row_of("bench", "race_condition", "nondeterministic_runs=3/10"))[1],
                         "- shape (M=1024, N=1024, K=1024): results differ between two identical calls")

    def test_exception_and_signal(self) -> None:
        doc = bench_doc(classification="runtime_crash", error_detail="MemoryError: ", final_check=None, complete=True,
                        exception={"type": "MemoryError", "message": "", "traceback": "Traceback (most recent call last):\n"
                                   f'  File "{SAMPLE_PATH}/kernel.py", line 9, in gemm\n    T = np.empty((M, N, K))\nMemoryError\n'})
        lines = self.bench(doc, row_of("bench", "runtime_crash", "MemoryError"))
        self.assertEqual(lines[1], "- shape (M=1024, N=1024, K=1024): raised MemoryError")
        self.assertEqual(lines[-1], "    MemoryError")
        partial = bench_doc(classification=None, error_detail="", complete=False, final_check=None, in_progress={"test": "large", "call_idx": 4})
        lines = self.bench(partial, row_of("bench", "runtime_crash", "signal SIGBUS during large call 4"), rc=-7, signal="SIGBUS", complete=False)
        self.assertEqual(lines, [self.HEADER, "- shape (M=1024, N=1024, K=1024): crashed: signal SIGBUS"])

    def test_modified_input(self) -> None:
        doc = bench_doc(classification="wrong_output", error_detail="inputs_mutated", inputs_mutated=True,
                        final_check={"classification": "wrong_output", "pattern": "inputs_mutated"})
        self.assertEqual(self.bench(doc, row_of("bench", "wrong_output", "inputs_mutated"))[1],
                         "- shape (M=1024, N=1024, K=1024): modified a read-only input")


# ----------------------------------------------------------------------------- 하네스의 진단 이름·참조값은 나가지 않는다


def classify_names() -> tuple[list[str], list[str]]:
    """``harness/classify.py`` 가 돌려주는 (패턴 이름, 판정 이름). 패턴은 소스에서 읽는다: ``return <판정>, <패턴>, …`` 의 둘째 값,
    문자열 하나를 돌려주는 ``return``, ``pattern = …`` 대입에 나오는 문자열 상수(f-string·덧셈의 머리 ``partial_write:`` 포함)."""
    statuses = sorted({*classify.OUTPUT_STATUSES, *classify.SAMPLE_STATUSES})
    tree = ast.parse(Path(classify.__file__).read_text(encoding="utf-8"))

    def strings(node: ast.AST) -> list[str]:
        return [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Constant) and isinstance(sub.value, str)]

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                found.add(value.value)
            elif isinstance(value, ast.Tuple) and len(value.elts) >= 2 and isinstance(value.elts[0], ast.Constant) \
                    and value.elts[0].value in statuses:
                found.update(strings(value.elts[1]))
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "pattern" for target in node.targets):
            found.update(strings(node.value))
    patterns = sorted(name for name in found if re.fullmatch(r"[A-Za-z_]\w*:?", name) and name not in statuses)
    return patterns, statuses


PATTERN_INSTANCES = {"partial_write:": "partial_write:12/35", "nondeterministic:": "nondeterministic:tiny,rect",
                     "mixed:": "mixed:arg_order/transposed_output,numeric_instability/imprecise"}


def word(token: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![A-Za-z0-9_]){re.escape(token)}(?![A-Za-z0-9_])")


class NoHarnessVocabularyTest(DiagnosticsCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.patterns, cls.statuses = classify_names()

    def test_pattern_names_are_read_from_classify(self) -> None:
        known = {"transposed_output", "swapped_operands", "column_major_all", "swapped_MN_dims", "nan_inf", "imprecise", "mismatch",
                 "returned_array_not_written", "output_not_written", "accumulated_into_c", "partial_write:", "out_of_bounds_write",
                 "inputs_mutated", "nondeterministic:", "mixed:"}
        self.assertLessEqual(known, set(self.patterns), self.patterns)
        self.assertEqual(set(self.statuses), {"pass", "wrong_output", "numeric_instability", "arg_order", "runtime_crash", "race_condition"})

    def requests(self, pattern: str, status: str) -> list[str]:
        """그 패턴·판정이 하네스가 기록하는 모든 자리에 들어 있는 check·bench 실패의 수정 요청(두 피드백 조건)."""
        instance = PATTERN_INSTANCES.get(pattern, pattern)
        # 참조에서 나온 값·센티널에서 나온 수: 어느 것도 나가면 안 된다
        secret = {"max_tol_ratio": 9876.54321, "n_unwritten": 31337}
        failing = run_doc(status, instance, max_abs_err=3.1415926, nan_inf=status == "numeric_instability" and pattern == "nan_inf", **secret)
        tests = [shape_doc("tiny", [run_doc(), run_doc()]), shape_doc("rect", [failing, failing], nondeterministic=status == "race_condition")]
        row = row_of("check", status if status != "pass" else "wrong_output", instance, check_tests=tests, check_max_tol_ratio=9876.54321)
        out: list[str] = []
        self.put("check.json", check_doc(tests, status, instance, max_tol_ratio=9876.54321))
        stages: dict[str, Any] = {"check": summary(classification=status, error_detail=instance), "lint": {
            "tool": "baseline:ruff+pyflakes", "ok": True, "error": "", "summary": {}, "findings": []}}
        for feedback in ("compiler", "compiler+lint"):
            out.append(repair.render_request(TEMPLATE, row, stages, self.dir, feedback).text)
        final = {"classification": status, "pattern": instance, "max_abs_err": 3.1415926, "n_fail": 7, "n_total": 1048576,
                 "max_abs_ref": 123.456789, **secret}
        self.put("bench.json", bench_doc(classification=status, error_detail=instance, final_check=final,
                                         nondeterministic_runs=3 if status == "race_condition" else 0))
        bench_row = {**row, "error_stage": "bench"}
        stages["bench"] = summary(classification=status, error_detail=instance)
        out.append(repair.render_request(TEMPLATE, bench_row, stages, self.dir, "compiler").text)
        return out

    def test_no_pattern_or_verdict_name_reaches_the_request(self) -> None:
        failing_statuses = [status for status in self.statuses if status not in ("pass", "runtime_crash")]
        for pattern in self.patterns:
            for status in failing_statuses:
                with self.subTest(pattern=pattern, status=status):
                    for text in self.requests(pattern, status):
                        for token in [name.rstrip(":") for name in self.patterns] + [s for s in self.statuses if s != "pass"]:
                            self.assertIsNone(word(token).search(text), f"{token!r} 가 수정 요청에 들어갔다:\n{text}")
                        self.assertIsNone(word("pass").search(text.split("## Diagnostics")[1]))

    def test_no_reference_derived_value_reaches_the_request(self) -> None:
        for text in self.requests("mismatch", "wrong_output"):
            for secret in ("9876", "31337", "123.45", "1.2e+02", "-2048", "2048"):
                self.assertNotIn(secret, text)
            self.assertIn("max abs error 3.1e+00", text)

    def test_error_type_text_hides_the_classifier_and_the_sanitizer(self) -> None:
        text = {name: repair.feedback_error_type(row_of("check", name)) for name in repair.ERROR_TYPE_TEXT}
        self.assertEqual(text, {"signature_mismatch": "signature mismatch", "compile_error": "compile error",
                                "runtime_crash": "runtime crash", "timeout": "timeout", "wrong_output": "incorrect output",
                                "arg_order": "incorrect output", "numeric_instability": "incorrect output",
                                "race_condition": "non-deterministic output"})
        # sanitizer 가 재분류한 행(설계 v3 §10.2): 수정 요청은 safety.enabled 와 무관하게 재분류 전의 판정을 쓴다
        self.assertEqual(repair.feedback_error_type(row_of("check", "memory_error", "asan:heap-buffer-overflow; was runtime_crash/signal SIGSEGV during tiny call 0")),
                         "runtime crash")
        self.assertEqual(repair.feedback_error_type(row_of("check", "memory_error", "asan:stack-buffer-overflow; was wrong_output/mismatch")),
                         "incorrect output")
        for bad in (row_of("check", "memory_error", "asan:heap-buffer-overflow"), row_of("check", ""), row_of("check", "brand_new_type")):
            with self.subTest(bad=bad), self.assertRaises(repair.RepairError):
                repair.feedback_error_type(bad)

    def test_every_taxonomy_error_type_has_a_request_text(self) -> None:
        from harness.schema import ERROR_TYPES

        self.assertEqual({name for name in ERROR_TYPES if name}, set(repair.ERROR_TYPE_TEXT) | {"memory_error"})


# ----------------------------------------------------------------------------- 린트 절·수정 요청


LINT_FINDINGS = [
    {"rule": "F841", "severity": "warning", "line": 12, "message": "Local variable `tmp` is assigned to\n  but never used", "category": "api_misuse"},
    {"rule": "PERF401", "severity": "warning", "line": 30, "message": "Use a list comprehension", "category": "performance"},
]


def lint_stage(findings: list[dict[str, Any]] | None = None, **over: Any) -> dict[str, Any]:
    findings = [] if findings is None else findings
    counts = {category: sum(f["category"] == category for f in findings)
              for category in ("memory", "numeric", "concurrency", "shape_dtype", "api_misuse", "performance")}
    return {"tool": "baseline:ruff+pyflakes", "findings": findings, "summary": counts, "ok": True, "error": "", **over}


class LintSectionTest(unittest.TestCase):
    def test_compiler_feedback_has_no_section(self) -> None:
        self.assertEqual(repair.render_lint_section({"lint": lint_stage(LINT_FINDINGS)}, "compiler"), ("", None))

    def test_findings_are_listed_as_line_and_message(self) -> None:
        text, n = repair.render_lint_section({"lint": lint_stage(LINT_FINDINGS)}, "compiler+lint")
        self.assertEqual(text, "\n## Static analysis findings\n\n- line 12: Local variable `tmp` is assigned to but never used\n"
                               "- line 30: Use a list comprehension\n")
        self.assertEqual(n, 2)

    def test_at_most_twenty_findings(self) -> None:
        many = [{"rule": "X", "severity": "warning", "line": i, "message": f"finding {i}", "category": "performance"} for i in range(1, 31)]
        text, n = repair.render_lint_section({"lint": lint_stage(many)}, "compiler+lint")
        self.assertEqual(n, 20)
        self.assertEqual(text.count("\n- line "), 20)
        self.assertIn("- line 20: finding 20\n", text)
        self.assertNotIn("finding 21", text)

    def test_lint_that_ran_and_found_nothing(self) -> None:
        self.assertEqual(repair.render_lint_section({"lint": lint_stage([])}, "compiler+lint"), ("\n## Static analysis findings\n\nNone.\n", 0))

    def test_lint_that_did_not_run(self) -> None:
        unavailable = ("\n## Static analysis findings\n\nNot available.\n", None)
        for stages in ({}, {"lint": lint_stage(LINT_FINDINGS, ok=False, error="timeout after 60s")},
                       {"lint": lint_stage([], tool="none")}, {"lint": None}):
            with self.subTest(stages=stages):
                self.assertEqual(repair.render_lint_section(stages, "compiler+lint"), unavailable)


class RequestTest(DiagnosticsCase):
    def request(self, feedback: str, lint: dict[str, Any] | None = None) -> repair.RepairRequest:
        self.put("build.log", GCC_LOG)
        build = build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="gcc", error_detail="kernel.c:6:13: error: ...")
        stages: dict[str, Any] = {"build": build}
        if lint is not None:
            stages["lint"] = lint
        return repair.render_request(TEMPLATE, row_of("build", "compile_error", build["error_detail"]), stages, self.dir, feedback)

    def test_template_has_exactly_the_four_variables(self) -> None:
        self.assertEqual(set(re.findall(r"\$\{(\w+)\}", TEMPLATE)), {"stage", "error_type", "diagnostics", "lint_section"})
        self.assertIn("${diagnostics}\n${lint_section}\n## ", TEMPLATE)
        self.assertTrue(TEMPLATE.isascii())

    def test_compiler_request(self) -> None:
        req = self.request("compiler", lint_stage(LINT_FINDINGS))
        self.assertEqual((req.lint_included, req.n_lint_findings), (False, None))
        self.assertTrue(req.diagnostics.startswith("Compiler output:\nkernel.c:6:13: error: expected"))
        self.assertIn("- Stage that failed: building the code (compiling it, or importing the module)\n", req.text)
        self.assertIn("- Kind of failure: compile error\n", req.text)
        self.assertIn(f"## Diagnostics\n\n{req.diagnostics}\n\n## Request\n", req.text)
        self.assertNotIn("Static analysis", req.text)
        self.assertNotIn("$", req.text)
        self.assertTrue(req.text.endswith("exactly one fenced code block containing the complete source file.\n"))

    def test_the_two_feedback_conditions_differ_only_by_the_lint_section(self) -> None:
        plain = self.request("compiler", lint_stage(LINT_FINDINGS))
        linted = self.request("compiler+lint", lint_stage(LINT_FINDINGS))
        section = "\n## Static analysis findings\n\n- line 12: Local variable `tmp` is assigned to but never used\n- line 30: Use a list comprehension\n"
        self.assertEqual((linted.lint_included, linted.n_lint_findings, linted.diagnostics), (True, 2, plain.diagnostics))
        self.assertEqual(linted.text, plain.text.replace("\n\n## Request\n", f"\n{section}\n## Request\n"))
        empty = self.request("compiler+lint", lint_stage([]))
        self.assertEqual((empty.lint_included, empty.n_lint_findings), (True, 0))
        self.assertIn("\n## Static analysis findings\n\nNone.\n\n## Request\n", empty.text)
        missing = self.request("compiler+lint")
        self.assertEqual((missing.lint_included, missing.n_lint_findings), (True, None))
        self.assertIn("\n## Static analysis findings\n\nNot available.\n\n## Request\n", missing.text)

    def test_stage_texts(self) -> None:
        self.assertEqual(set(repair.STAGE_TEXT), {"extract", "build", "check", "bench"})
        with self.assertRaises(repair.RepairError):
            repair.render_request(TEMPLATE, row_of("generate", "", "refusal:"), {}, self.dir, "compiler")

    def test_template_with_other_variables_is_rejected(self) -> None:
        for content in ("${stage} ${error_type} ${diagnostics}\n", "${stage} ${error_type} ${diagnostics} ${lint_section} ${hint}\n",
                        "${stage} ${error_type} ${diagnostics} ${lint_section} $ 5\n"):
            self.put("repair.md", content)
            with self.subTest(content=content), self.assertRaises(repair.ConfigError):
                repair.load_template(self.dir / "repair.md")
        with self.assertRaises(repair.ConfigError):
            repair.load_template(self.dir / "missing.md")

    def test_prompt_hash_definition(self) -> None:
        messages = [{"role": "user", "content": "프롬프트 α"}, {"role": "assistant", "content": "응답"}, {"role": "user", "content": "수정 요청"}]
        expected = hashlib.sha256(("SYSTEM" + "\n\x00\n" + json.dumps(messages, ensure_ascii=False)).encode("utf-8")).hexdigest()[:16]
        self.assertEqual(repair.repair_prompt_hash("SYSTEM", messages), expected)


# ----------------------------------------------------------------------------- 가짜 harness.run 과 실행 틀


class FakeRun:
    """``harness.run`` 의 재사용 API(설계 v3 §10.1) 자리에 끼우는 가짜. raw 의 응답 글에 따라 행·단계 기록을 만든다:
    ``PASS`` 통과, ``COMPILE_ERROR`` 빌드 실패(build.log), ``BENCH_TIMEOUT`` bench 시간 초과(pass=1), 그 밖은 check 실패."""

    EXIT_TOOLCHAIN = 7

    class RunSetupError(RuntimeError):
        def __init__(self, message: str, exit_code: int) -> None:
            super().__init__(message)
            self.exit_code = exit_code

    class HarnessError(RuntimeError):
        pass

    def __init__(self) -> None:
        self.setup_calls: list[dict[str, Any]] = []
        self.processed: list[Path] = []
        self.finished: list[list[tuple[RunRow, Path]]] = []
        self.setup_error: Exception | None = None
        self.process_error: BaseException | None = None
        self.lint: dict[str, Any] | None = None

    def setup_run(self, cfg: Config, run_id: str, *, skip_bench: bool = False, force: bool = False, strict_protocol: bool = False,
                  no_baseline: bool = False, tasks: list[str] | None = None, languages: set[str] | None = None) -> Any:
        self.setup_calls.append({"run_id": run_id, "skip_bench": skip_bench, "force": force, "strict_protocol": strict_protocol,
                                 "no_baseline": no_baseline, "tasks": tasks, "languages": languages})
        if self.setup_error is not None:
            raise self.setup_error
        return types.SimpleNamespace(cfg=cfg, run_id=run_id, tasks={"gemm": "TASK-CONTEXT-gemm"})

    def process_sample(self, ctx: Any, tctx: Any, raw_path: Path) -> tuple[RunRow | None, str]:
        if self.process_error is not None:
            raise self.process_error
        assert tctx == "TASK-CONTEXT-gemm", tctx
        self.processed.append(Path(raw_path))
        return self.write_result(ctx.cfg, Path(raw_path)), "processed"

    def finish_run(self, ctx: Any, processed: list[tuple[RunRow, Path]]) -> None:
        self.finished.append(list(processed))

    def write_result(self, cfg: Config, raw_path: Path) -> RunRow:
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        cond = raw["conditions"]
        tag = cond_tag(cond["args"], cond["grammar_hint"], cond["feedback"])
        sample_dir = built_turn_dir(cfg.paths.built_dir, raw["run_id"], raw["task"], raw["language"], tag, raw["sample_idx"], raw["turn"])
        sample_dir.mkdir(parents=True, exist_ok=True)
        row = RunRow(run_id=raw["run_id"], task=raw["task"], language=raw["language"], backend=raw["backend"], args=cond["args"],
                     grammar_hint=cond["grammar_hint"], feedback=cond["feedback"], sample_idx=raw["sample_idx"], turn=raw["turn"],
                     provider=raw["provider"], valid_sample=True, passed=False, error_stage="", error_type="", error_detail="",
                     raw_path=rel_to_root(raw_path), built_path=rel_to_root(sample_dir))
        stages: dict[str, Any] = {}
        text = raw["response_text"]
        if raw["error"] or raw["stop_reason"] == "max_tokens":  # 생성 게이트: API 오류·절단은 유효 샘플이 아니다
            row.valid_sample, row.error_stage, row.error_detail = False, "generate", "api_error" if raw["error"] else "truncated"
        elif raw["stop_reason"] == "refusal":
            row.error_stage, row.error_detail = "generate", "refusal:"
        elif "COMPILE_ERROR" in text:
            row.error_stage, row.error_type, row.error_detail = "build", "compile_error", "kernel.c:6:13: error: expected ',' or ';' before 'for'"
            (sample_dir / "build.log").write_text(GCC_LOG, encoding="utf-8")
            stages["build"] = build_stage(ok=False, artifact=None, error_kind="compile_error", compiler="gcc", error_detail=row.error_detail)
        else:
            stages["build"] = build_stage()
            stages["import"] = summary()
            if "PASS" in text or "BENCH_TIMEOUT" in text:
                row.passed = True
                stages["check"] = summary()
                if "BENCH_TIMEOUT" in text:
                    row.error_stage, row.error_type, row.error_detail = "bench", "timeout", "first_call_186929ms_projected_748s"
            else:
                row.error_stage, row.error_type, row.error_detail = "check", "arg_order", "transposed_output"
                wrong = wrong_run("transposed_output", "arg_order")
                doc = check_doc([shape_doc(name, [wrong, wrong]) for name in GEMM_SHAPES], "arg_order", "transposed_output")
                (sample_dir / "check.json").write_text(json.dumps(doc), encoding="utf-8")
                stages["check"] = summary(classification="arg_order", error_detail="transposed_output")
        if self.lint is not None and "build" in stages:
            stages["lint"] = self.lint
        result = {"row": row.to_dict(), "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(), "stages": stages,
                  "written_at": "2026-11-22T12:00:00+09:00"}
        (sample_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
        return row


class FakeClient:
    """``LLMClient`` 자리에 끼우는 가짜(실제 제공자를 흉내 낸다; SDK·네트워크 없음)."""

    instances: list["FakeClient"] = []
    replies: list[tuple[str, str | None]] = []  # (응답 글, 오류) — 비면 ("PASS", None)

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.calls: list[dict[str, Any]] = []
        self.preflighted = False
        FakeClient.instances.append(self)

    def preflight(self) -> None:
        self.preflighted = True

    def complete(self, system: str, user: str, *, messages: Any = None, dummy_key: Any = None) -> LLMResponse:
        self.calls.append({"system": system, "user": user, "messages": messages, "dummy_key": dummy_key})
        text, error = FakeClient.replies.pop(0) if FakeClient.replies else ("PASS", None)
        return make_response(self.cfg, text, error=error, model_served="served-model")


def make_response(cfg: Config, text: str, *, error: str | None = None, stop_reason: str = "end_turn", model_served: str = "dummy") -> LLMResponse:
    ok = error is None
    return LLMResponse(text=text if ok else "", model_requested=cfg.model.name, model_served=model_served if ok else "",
                       provider=cfg.model.provider, tokens_in=100 if ok else 0, tokens_out=20 if ok else 0, tokens_thinking=None,
                       tokens_text=None, latency_ms=1.0, stop_reason=stop_reason if ok else "", raw_stop_reason=stop_reason if ok else "",
                       stop_details_category=None, n_thinking_blocks=0, n_text_blocks=1 if ok else 0, request_id=None, raw_usage={},
                       base_url=None, sdk_version=None, api_key_present=False, error=error)


class LoopCase(unittest.TestCase):
    """임시 디렉터리의 run(raw + 가짜 처리 결과)과 dummy 응답으로 ``repair.main`` 을 돌리는 틀."""

    DUMMY = {"0.md": "PASS: a correct reply", "1.md": "COMPILE_ERROR: a reply that does not compile", "1.turn1.md": "PASS: repaired",
             "2.md": "WRONG: a reply with a transposed output"}

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.out = Path(self._tmp.name)
        for language in ("numpy", "c_seq"):
            directory = self.out / "dummy" / "gemm" / language
            directory.mkdir(parents=True)
            for name, text in self.DUMMY.items():
                (directory / name).write_text(text + "\n", encoding="utf-8")
        self.fake = FakeRun()
        FakeClient.instances, FakeClient.replies = [], []
        for patcher in (mock.patch.object(llm_client, "DUMMY_ROOT", self.out / "dummy"),
                        mock.patch.object(repair, "_run_api", return_value=self.fake),
                        mock.patch.object(repair, "harness_version", return_value="test-version"),
                        mock.patch.object(repair, "install_terminate_handlers"),  # 테스트 프로세스의 시그널 처리는 건드리지 않는다
                        # dummy 제공자는 SDK 경로에 들어가면 안 된다
                        mock.patch.object(llm_client.LLMClient, "_init_anthropic", side_effect=AssertionError("anthropic SDK 를 쓰면 안 된다")),
                        mock.patch.object(llm_client.LLMClient, "_init_openai_compat", side_effect=AssertionError("openai SDK 를 쓰면 안 된다")),
                        mock.patch.dict(os.environ)):
            patcher.start()
            self.addCleanup(patcher.stop)
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)
        self.config_path = self.write_config()
        self.cfg = load_config(self.config_path)

    def write_config(self, *, provider: str = "dummy", max_repair_turns: int = 0) -> Path:
        doc = yaml.safe_load(DEFAULT_EXPERIMENT_YAML.read_text(encoding="utf-8"))
        doc["run"].update({"tasks": ["gemm"], "max_repair_turns": max_repair_turns})
        doc["model"]["provider"] = provider
        doc["paths"] = {"results_csv": str(self.out / "runs.csv"), "results_csv_dummy": str(self.out / "runs_dummy.csv"),
                        "raw_dir": str(self.out / "raw"), "built_dir": str(self.out / "built")}
        path = self.out / "experiment.yaml"
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        return path

    # ------------------------------------------------------------------ run 만들기

    def raw_path(self, idx: int, turn: int = 0, *, language: str = "numpy", tag: str = TAG, run_id: str = "r1") -> Path:
        return raw_turn_path(self.cfg.paths.raw_dir, run_id, "gemm", language, tag, idx, turn)

    def raw(self, idx: int, turn: int = 0, **where: Any) -> dict[str, Any]:
        return json.loads(self.raw_path(idx, turn, **where).read_text(encoding="utf-8"))

    def add_turn_zero(self, idx: int, text: str | None = None, *, language: str = "numpy", feedback: str = "compiler", run_id: str = "r1",
                      error: str | None = None, stop_reason: str = "end_turn", process: bool = True) -> Path:
        """generate + run 이 한 일을 흉내 낸다: 턴 0 raw 를 쓰고(응답은 dummy 파일 ``idx mod 3``) 가짜 run 으로 처리한다."""
        cfg = self.cfg
        if text is None:
            text = (self.DUMMY[f"{idx % 3}.md"]) + "\n"
        conditions = {"args": "positional", "grammar_hint": "off", "feedback": feedback}
        system, user = "SYSTEM PROMPT\n", f"TURN 0 PROMPT for gemm/{language}\n"
        raw = generate.build_raw(cfg, run_id=run_id, task="gemm", lang=cfg.languages[language], sample_idx=idx, turn=0,
                                 conditions=conditions, prompt={"system": system, "user": user},
                                 prompt_hash=generate.prompt_hash(system, user), resp=make_response(cfg, text, error=error, stop_reason=stop_reason),
                                 snapshot={"note": "test"}, version="test-version")
        path = raw_turn_path(cfg.paths.raw_dir, run_id, "gemm", language, cond_tag("positional", "off", feedback), idx, 0)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        if process:
            self.fake.write_result(cfg, path)
        return path

    def standard_run(self) -> None:
        """샘플 0 통과, 샘플 1 컴파일 오류(턴 1 응답 파일이 고친다), 샘플 2 틀린 출력(턴 파일 없음: 같은 실패 반복)."""
        for idx in range(3):
            self.add_turn_zero(idx)

    def main(self, *args: str, run_id: str = "r1") -> tuple[int, str, str]:
        argv = ["--config", str(self.config_path), "--run-id", run_id, *args]
        with contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
            rc = repair.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def turn_files(self, run_id: str = "r1") -> list[str]:
        base = self.cfg.paths.raw_dir / run_id
        return sorted(p.relative_to(base).as_posix() for p in base.rglob("*.turn*.json"))


class SelectionTest(LoopCase):
    def test_is_repair_target(self) -> None:
        target = {"valid_sample": True, "passed": False, "error_stage": "check", "error_type": "wrong_output"}
        self.assertTrue(repair.is_repair_target(target))
        for stage in ("extract", "build", "bench"):
            self.assertTrue(repair.is_repair_target({**target, "error_stage": stage}))
        self.assertFalse(repair.is_repair_target({**target, "passed": True}))
        self.assertFalse(repair.is_repair_target({**target, "passed": True, "error_stage": "bench", "error_type": "timeout"}))  # bench timeout
        self.assertFalse(repair.is_repair_target({**target, "error_stage": "generate", "error_type": ""}))  # 거절
        self.assertFalse(repair.is_repair_target({**target, "valid_sample": False, "error_stage": "generate"}))  # API 오류·절단
        self.assertFalse(repair.is_repair_target({**target, "valid_sample": None}))

    def test_only_failed_valid_samples_are_called(self) -> None:
        self.add_turn_zero(0)                                             # 통과
        self.add_turn_zero(1)                                             # 빌드 실패 → 대상
        self.add_turn_zero(2)                                             # check 실패 → 대상
        self.add_turn_zero(3, "BENCH_TIMEOUT: slow but correct\n")        # bench timeout(pass=1)
        self.add_turn_zero(4, "", stop_reason="refusal")                  # 거절
        self.add_turn_zero(5, "", error="transient:OverloadedError:529:Overloaded")  # API 오류
        self.add_turn_zero(6, "WRONG: cut off in the middle", stop_reason="max_tokens")  # 절단
        rc, out, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_1.turn1.json", f"gemm/numpy/{TAG}/sample_2.turn1.json"])
        self.assertIn("calls=2", out)

    def test_scan_keeps_to_the_feedback_condition_and_the_filters(self) -> None:
        self.standard_run()
        self.add_turn_zero(1, feedback="compiler+lint")
        self.add_turn_zero(1, language="c_seq")

        def scan(**kwargs: Any) -> list[tuple[str, str, int]]:
            return [(s.language, s.tag, s.sample_idx) for s in repair.scan_samples(self.cfg, "r1", **kwargs)]

        self.assertEqual(scan(feedback="compiler"), [("c_seq", TAG, 1), ("numpy", TAG, 0), ("numpy", TAG, 1), ("numpy", TAG, 2)])
        self.assertEqual(scan(feedback="compiler+lint"), [("numpy", TAG_LINT, 1)])
        self.assertEqual(scan(feedback="compiler", languages={"numpy"}, samples={1, 2}), [("numpy", TAG, 1), ("numpy", TAG, 2)])
        self.assertEqual(scan(feedback="compiler", tasks=["jacobi2d"]), [])
        self.assertEqual(scan(feedback="compiler", tasks=["gemm"], languages={"c_seq"}), [("c_seq", TAG, 1)])

    def test_scan_follows_the_turns_that_have_a_response(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "2")[0], 0)
        chains = {s.sample_idx: s for s in repair.scan_samples(self.cfg, "r1", feedback="compiler")}
        self.assertEqual({idx: s.last_turn for idx, s in chains.items()}, {0: 0, 1: 1, 2: 2})
        self.assertEqual([p.name for p in chains[2].paths], ["sample_2.json", "sample_2.turn1.json", "sample_2.turn2.json"])
        # 생성에 실패한 턴은 사슬에 잇지 않는다(다시 호출할 턴이다)
        broken = self.raw(2, 2)
        self.raw_path(2, 2).write_text(json.dumps({**broken, "error": "transient:APIConnectionError::reset", "stop_reason": ""}), encoding="utf-8")
        self.assertEqual({s.sample_idx: s.last_turn for s in repair.scan_samples(self.cfg, "r1", feedback="compiler")}, {0: 0, 1: 1, 2: 1})

    def test_unreadable_raw_and_unknown_language_are_skipped(self) -> None:
        self.standard_run()
        self.raw_path(0).write_text("{broken", encoding="utf-8")
        stray = self.cfg.paths.raw_dir / "r1" / "gemm" / "cobol" / TAG / "sample_0.json"
        stray.parent.mkdir(parents=True)
        stray.write_text(self.raw_path(1).read_text(encoding="utf-8"), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            found = [(s.language, s.sample_idx) for s in repair.scan_samples(self.cfg, "r1", feedback="compiler")]
        self.assertEqual(found, [("numpy", 1), ("numpy", 2)])
        self.assertEqual(err.getvalue().count("[repair] 경고"), 2)


class LoopTest(LoopCase):
    def test_dummy_loop_repairs_until_pass_or_the_turn_limit(self) -> None:
        self.standard_run()
        rc, out, err = self.main("--max-turns", "2")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_1.turn1.json", f"gemm/numpy/{TAG}/sample_2.turn1.json",
                                             f"gemm/numpy/{TAG}/sample_2.turn2.json"])
        # 처리: 준비 한 번, 새 raw 마다 process_sample (턴 순서), 끝에 run 의 모든 행으로 finish_run 한 번
        self.assertEqual(self.fake.setup_calls, [{"run_id": "r1", "skip_bench": False, "force": False, "strict_protocol": False,
                                                  "no_baseline": False, "tasks": None, "languages": None}])
        self.assertEqual(self.fake.processed, [self.raw_path(1, 1), self.raw_path(2, 1), self.raw_path(2, 2)])
        self.assertEqual(len(self.fake.finished), 1)
        entries = self.fake.finished[0]
        self.assertEqual(sorted((row.sample_idx, row.turn, bool(row.passed)) for row, _ in entries),
                         [(0, 0, True), (1, 0, False), (1, 1, True), (2, 0, False), (2, 1, False), (2, 2, False)])
        for row, result_path in entries:
            self.assertTrue(result_path.is_file(), result_path)
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8"))["row"]["turn"], row.turn)
        self.assertIn("run_id=r1 feedback=compiler calls=3 failed=0 passed_after_repair=1", out)
        self.assertIn("2 dummy calls", err)
        self.assertIn("1 dummy calls", err)

    def test_turn_one_raw_has_the_turn_zero_keys_plus_the_conversation(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "1")[0], 0)
        first, raw = self.raw(1), self.raw(1, 1)
        self.assertEqual(list(raw), [*RAW_KEYS, "repair"])
        self.assertEqual(list(first), RAW_KEYS)
        for key in ("run_id", "task", "language", "backend", "sample_idx", "conditions", "provider", "max_tokens", "harness_version"):
            self.assertEqual(raw[key], first[key], key)
        self.assertEqual((raw["turn"], first["turn"]), (1, 0))
        self.assertEqual(raw["response_text"], "PASS: repaired\n")  # dummy 의 1.turn1.md
        self.assertEqual(list(raw["prompt"]), ["system", "user", "messages"])
        self.assertEqual((raw["prompt"]["system"], raw["prompt"]["user"]), (first["prompt"]["system"], first["prompt"]["user"]))
        messages = raw["prompt"]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual((messages[0]["content"], messages[1]["content"]), (first["prompt"]["user"], first["response_text"]))
        request = messages[2]["content"]
        self.assertIn("- Kind of failure: compile error\n", request)
        self.assertIn("Compiler output:\nkernel.c:6:13: error: expected ',' or ';' before 'for'", request)
        digest = hashlib.sha256((raw["prompt"]["system"] + "\n\x00\n" + json.dumps(messages, ensure_ascii=False)).encode("utf-8")).hexdigest()
        self.assertEqual(raw["prompt_hash"], digest[:16])
        self.assertNotEqual(raw["prompt_hash"], first["prompt_hash"])
        self.assertEqual(raw["repair"], {"parent_raw": rel_to_root(self.raw_path(1)), "feedback": "compiler",
                                         "diagnostics": request.split("## Diagnostics\n\n")[1].split("\n\n## Request")[0],
                                         "lint_included": False, "n_lint_findings": None})
        self.assertEqual(raw["config_snapshot"]["run"]["conditions"]["feedback"], "compiler")

    def test_turn_two_continues_the_same_conversation(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "2")[0], 0)
        zero, one, two = self.raw(2), self.raw(2, 1), self.raw(2, 2)
        sent_one, sent_two = one["prompt"]["messages"], two["prompt"]["messages"]
        self.assertEqual([m["role"] for m in sent_two], ["user", "assistant", "user", "assistant", "user"])
        self.assertEqual(sent_two[:3], sent_one)  # 앞 턴의 수정 요청은 그때 보낸 원문 그대로다
        self.assertEqual(sent_two[3], {"role": "assistant", "content": one["response_text"]})
        self.assertEqual(one["response_text"], zero["response_text"])  # 턴 파일이 없는 더미: 같은 응답이 반복된다
        self.assertIn("- Kind of failure: incorrect output\n", sent_two[4]["content"])
        self.assertIn("- shape (M=200, N=130, K=312): incorrect output (max abs error 3.1e+00)", sent_two[4]["content"])
        self.assertEqual((two["turn"], two["prompt"]["user"], two["prompt"]["system"]), (2, zero["prompt"]["user"], zero["prompt"]["system"]))
        self.assertEqual(two["repair"]["parent_raw"], rel_to_root(self.raw_path(2, 1)))
        for text in (sent_two[2]["content"], sent_two[4]["content"]):
            self.assertNotIn("transposed_output", text)
            self.assertNotIn("arg_order", text)

    def test_rerun_continues_where_it_stopped(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "1")[0], 0)
        self.fake.processed.clear()
        rc, out, _ = self.main("--max-turns", "3")
        self.assertEqual(rc, 0)
        self.assertEqual(self.fake.processed, [self.raw_path(2, 2), self.raw_path(2, 3)])  # 턴 1 은 다시 묻지 않는다
        self.assertIn("calls=2", out)
        before = self.raw_path(2, 1).read_bytes()
        self.fake.processed.clear()
        setups = len(self.fake.setup_calls)
        rc, _, err = self.main("--max-turns", "3")
        self.assertEqual((rc, self.fake.processed, len(self.fake.setup_calls)), (0, [], setups))  # 할 일이 없으면 준비도 하지 않는다
        self.assertIn("수정할 샘플이 없다", err)
        self.assertEqual(self.raw_path(2, 1).read_bytes(), before)

    def test_failed_turn_is_called_again_and_the_old_raw_is_kept(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "1")[0], 0)
        path = self.raw_path(2, 1)
        failed = {**self.raw(2, 1), "error": "transient:OverloadedError:529:Overloaded", "stop_reason": "", "response_text": ""}
        path.write_text(json.dumps(failed), encoding="utf-8")
        self.fake.write_result(self.cfg, path)
        self.fake.processed.clear()
        rc, _, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fake.processed, [path])
        self.assertIsNone(self.raw(2, 1)["error"])
        archived = path.with_name("sample_2.turn1.prev1.json")
        self.assertEqual(json.loads(archived.read_text(encoding="utf-8"))["error"], "transient:OverloadedError:529:Overloaded")
        self.assertEqual(self.raw(2, 1)["prompt"]["messages"], failed["prompt"]["messages"])  # 같은 턴 0 결과에서 만든 같은 요청

    def test_turn_of_an_older_conversation_is_called_again(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "2")[0], 0)
        old_turn_one = self.raw(2, 1)
        # 턴 0 을 다시 생성했다(generate --force + run): 남아 있는 턴 1·2 는 옛 응답에 이어진 대화다
        self.add_turn_zero(2, "WRONG: a different first reply\n")
        self.fake.processed.clear()
        rc, _, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fake.processed, [self.raw_path(2, 1)])
        self.assertIn("이어진 대화가 아니다", err)
        new_turn_one = self.raw(2, 1)
        self.assertEqual(new_turn_one["prompt"]["messages"][1], {"role": "assistant", "content": "WRONG: a different first reply\n"})
        self.assertEqual(json.loads(self.raw_path(2, 1).with_name("sample_2.turn1.prev1.json").read_text(encoding="utf-8")), old_turn_one)
        with contextlib.redirect_stderr(io.StringIO()):
            chains = {s.sample_idx: s.last_turn for s in repair.scan_samples(self.cfg, "r1", feedback="compiler")}
        self.assertEqual(chains[2], 1)  # 옛 턴 2 는 새 턴 1 에서 이어지지 않는다

    def set_row(self, idx: int, turn: int, **fields: Any) -> None:
        """그 턴의 result.json 행을 바꾼다(raw 는 그대로라 raw_sha256 이 같다) — ``harness.run --force`` 의 재분류를 흉내 낸다."""
        path = built_turn_dir(self.cfg.paths.built_dir, "r1", "gemm", "numpy", TAG, idx, turn) / "result.json"
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc["row"].update(fields)
        path.write_text(json.dumps(doc), encoding="utf-8")

    def test_chain_is_not_extended_past_an_earlier_turn_that_now_passes(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "2")[0], 0)  # 샘플 2: 턴 0·1·2 모두 실패
        self.set_row(2, 1, passed=True, error_stage="", error_type="", error_detail="")
        self.add_turn_zero(5)  # 다른 실패 샘플은 그대로 수정된다
        self.fake.processed.clear()
        rc, out, err = self.main("--max-turns", "3")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.fake.processed, [self.raw_path(5, 1), self.raw_path(5, 2), self.raw_path(5, 3)])
        self.assertFalse(self.raw_path(2, 3).exists())
        self.assertIn("calls=3", out)
        warnings = [line for line in err.splitlines() if "낡은 결과" in line]
        self.assertEqual(len(warnings), 1)
        self.assertIn("gemm/numpy/sample_2/turn_1", warnings[0])
        self.assertIn("턴 2..2", warnings[0])
        self.assertTrue(self.raw_path(2, 2).exists())  # 낡은 턴의 raw·행은 지우지 않는다

    def test_chain_whose_only_sample_settled_earlier_has_nothing_to_do(self) -> None:
        self.add_turn_zero(2)
        self.assertEqual(self.main("--max-turns", "2")[0], 0)
        self.set_row(2, 0, valid_sample=False, error_stage="generate", error_type="", error_detail="api_error")
        self.fake.processed.clear()
        rc, _, err = self.main("--max-turns", "3")
        self.assertEqual((rc, self.fake.processed), (0, []), err)
        self.assertIn("수정할 샘플이 없다", err)
        self.assertIn("gemm/numpy/sample_2/turn_0", err)

    def test_last_turn_that_now_passes_just_ends_the_chain(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "2")[0], 0)
        self.set_row(2, 2, passed=True, error_stage="", error_type="", error_detail="")
        self.fake.processed.clear()
        rc, _, err = self.main("--max-turns", "3")
        self.assertEqual((rc, self.fake.processed), (0, []), err)
        self.assertNotIn("낡은 결과", err)
        self.assertIn("수정할 샘플이 없다", err)

    def test_empty_reply_is_not_sent_back_and_does_not_stop_the_others(self) -> None:
        # 사고 블록만 있던 end_turn 등: 응답 글이 비어도 유효 샘플(추출 실패)이지만, 빈 assistant 턴은 API 가 거부한다
        self.add_turn_zero(0, "")
        self.add_turn_zero(1, "  \n")
        self.add_turn_zero(2)
        rc, out, err = self.main("--max-turns", "2")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_2.turn1.json", f"gemm/numpy/{TAG}/sample_2.turn2.json"])
        self.assertEqual(err.count("응답이 비어 있어"), 2)  # 샘플마다 한 번
        self.assertIn("calls=2", out)
        self.assertIn("skipped_empty_reply=2", out)

    def test_max_turns(self) -> None:
        self.standard_run()
        rc, _, err = self.main()  # 설정의 run.max_repair_turns = 0
        self.assertEqual((rc, self.turn_files(), self.fake.setup_calls), (0, [], []))
        self.assertIn("할 일이 없다", err)
        for bad in ("4", "-1"):
            self.assertEqual(self.main("--max-turns", bad)[0], 1)
        self.assertEqual(self.turn_files(), [])
        self.config_path = self.write_config(max_repair_turns=2)
        self.assertEqual(self.main()[0], 0)
        self.assertEqual(len(self.turn_files()), 3)

    def test_filters_limit_the_scope_and_reach_setup_run(self) -> None:
        self.standard_run()
        self.add_turn_zero(1, language="c_seq")
        rc, _, err = self.main("--max-turns", "1", "--languages", "c_seq", "--tasks", "gemm", "--samples", "1,2", "--skip-bench")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.turn_files(), [f"gemm/c_seq/{TAG}/sample_1.turn1.json"])
        self.assertEqual(self.fake.setup_calls[0], {"run_id": "r1", "skip_bench": True, "force": False, "strict_protocol": False,
                                                    "no_baseline": False, "tasks": ["gemm"], "languages": {"c_seq"}})

    def test_samples_that_run_has_not_processed_are_left_alone(self) -> None:
        self.standard_run()
        self.add_turn_zero(4, process=False)                       # 턴 0 raw 만 있다
        stale = self.raw_path(2)
        stale.write_text(stale.read_text(encoding="utf-8") + "\n", encoding="utf-8")   # 처리 뒤에 raw 가 바뀌었다
        rc, _, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 0, err)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_1.turn1.json"])
        warnings = [line for line in err.splitlines() if "harness.run" in line]
        self.assertEqual(len(warnings), 2)
        self.assertTrue(any("sample_4" in line for line in warnings) and any("sample_2" in line for line in warnings))


class FeedbackConditionTest(LoopCase):
    def test_lint_condition_adds_only_the_lint_section(self) -> None:
        self.fake.lint = lint_stage(LINT_FINDINGS)
        self.add_turn_zero(2)
        self.add_turn_zero(2, feedback="compiler+lint")
        self.assertEqual(self.main("--max-turns", "1")[0], 0)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_2.turn1.json"])  # 기본 조건은 compiler
        self.assertEqual(self.main("--max-turns", "1", "--feedback", "compiler+lint")[0], 0)
        plain, linted = self.raw(2, 1), self.raw(2, 1, tag=TAG_LINT)
        self.assertEqual((plain["conditions"]["feedback"], linted["conditions"]["feedback"]), ("compiler", "compiler+lint"))
        self.assertEqual(plain["repair"]["diagnostics"], linted["repair"]["diagnostics"])
        self.assertEqual((plain["repair"]["lint_included"], plain["repair"]["n_lint_findings"]), (False, None))
        self.assertEqual((linted["repair"]["lint_included"], linted["repair"]["n_lint_findings"], linted["repair"]["feedback"]),
                         (True, 2, "compiler+lint"))
        section = "\n## Static analysis findings\n\n- line 12: Local variable `tmp` is assigned to but never used\n- line 30: Use a list comprehension\n"
        request_plain, request_linted = plain["prompt"]["messages"][-1]["content"], linted["prompt"]["messages"][-1]["content"]
        self.assertEqual(request_linted, request_plain.replace("\n\n## Request\n", f"\n{section}\n## Request\n"))
        self.assertEqual(linted["config_snapshot"]["run"]["conditions"]["feedback"], "compiler+lint")

    def test_seeding_copies_turn_zero_into_the_other_condition_without_calls(self) -> None:
        self.standard_run()
        self.add_turn_zero(1, language="c_seq")
        rc, _, err = self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")
        self.assertEqual(rc, 0, err)
        self.assertEqual((self.fake.setup_calls, self.fake.processed, self.turn_files()), ([], [], []))
        self.assertIn("4개", err)
        for idx, language in ((0, "numpy"), (1, "numpy"), (2, "numpy"), (1, "c_seq")):
            source, copy = self.raw(idx, language=language), self.raw(idx, language=language, tag=TAG_LINT)
            self.assertEqual(list(copy), [*RAW_KEYS, "seeded_from", "seeded_sha256"])
            self.assertEqual(copy["conditions"], {"args": "positional", "grammar_hint": "off", "feedback": "compiler+lint"})
            self.assertEqual(copy["seeded_from"], rel_to_root(self.raw_path(idx, language=language)))
            self.assertEqual(copy["seeded_sha256"], hashlib.sha256(self.raw_path(idx, language=language).read_bytes()).hexdigest())
            self.assertEqual({k: v for k, v in copy.items() if k not in ("conditions", "seeded_from", "seeded_sha256")},
                             {k: v for k, v in source.items() if k != "conditions"})
        self.assertEqual(self.raw(1)["conditions"]["feedback"], "compiler")  # 원본은 그대로다

    def test_seeded_samples_diverge_only_after_turn_zero(self) -> None:
        self.fake.lint = lint_stage([])
        self.standard_run()
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")[0], 0)
        # 복사한 턴 0 은 harness.run 이 처리한다(여기서는 가짜가): 그 전에는 수정 대상이 아니다
        rc, _, err = self.main("--max-turns", "1", "--feedback", "compiler+lint")
        self.assertEqual(rc, 1)
        self.assertIn("harness.run", err)
        for idx in range(3):
            self.fake.write_result(self.cfg, self.raw_path(idx, tag=TAG_LINT))
        self.assertEqual(self.main("--max-turns", "1", "--feedback", "compiler+lint")[0], 0)
        self.assertEqual(self.main("--max-turns", "1")[0], 0)
        for idx in (1, 2):
            plain, linted = self.raw(idx, 1), self.raw(idx, 1, tag=TAG_LINT)
            self.assertEqual(linted["prompt"]["messages"][:2], plain["prompt"]["messages"][:2])  # 같은 턴 0
            self.assertIn("## Static analysis findings\n\nNone.\n", linted["prompt"]["messages"][2]["content"])
            self.assertNotIn("Static analysis", plain["prompt"]["messages"][2]["content"])

    def test_seeding_keeps_existing_files_and_respects_filters(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint", "--samples", "1")[0], 0)
        copied = sorted(p.name for p in (self.cfg.paths.raw_dir / "r1" / "gemm" / "numpy" / TAG_LINT).glob("*.json"))
        self.assertEqual(copied, ["sample_1.json"])
        marker = self.raw_path(1, tag=TAG_LINT)
        marker.write_text(json.dumps({**self.raw(1, tag=TAG_LINT), "response_text": "edited"}), encoding="utf-8")
        rc, _, err = self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")
        self.assertEqual(rc, 0)
        self.assertEqual(self.raw(1, tag=TAG_LINT)["response_text"], "edited")  # 덮어쓰지 않는다
        self.assertIn("2개", err)
        self.assertIn("이미 있던 것 1개", err)

    def test_seeding_skips_sources_that_failed_to_generate(self) -> None:
        self.add_turn_zero(0)
        self.add_turn_zero(1, "", error="transient:InternalServerError:529:Overloaded")   # API 오류
        self.add_turn_zero(2, "WRONG: cut off in the middle", stop_reason="max_tokens")   # 절단
        self.add_turn_zero(3, "", stop_reason="refusal")                                  # 거절
        rc, _, err = self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")
        self.assertEqual(rc, 0, err)
        copied = sorted(p.name for p in (self.cfg.paths.raw_dir / "r1" / "gemm" / "numpy" / TAG_LINT).glob("*.json"))
        self.assertEqual(copied, ["sample_0.json"])
        self.assertEqual(err.count("생성에 실패한 raw"), 3)
        self.assertIn("생성 실패라 건너뛴 원본 3개", err)
        # generate 가 원본을 채운 뒤 다시 시드하면 그때 복사된다
        generate.archive_previous(self.raw_path(1))
        self.add_turn_zero(1)
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")[0], 0)
        self.assertIsNone(self.raw(1, tag=TAG_LINT)["error"])
        self.assertEqual(self.raw(1, tag=TAG_LINT)["response_text"], self.raw(1)["response_text"])

    def test_reseeding_replaces_a_copy_whose_source_was_generated_again(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")[0], 0)
        old_copy = self.raw(1, tag=TAG_LINT)
        # generate 가 원본을 다시 만들었다(--force 등): 옛 원본은 .prev1 로 가고 다른 응답이 생긴다
        generate.archive_previous(self.raw_path(1))
        self.add_turn_zero(1, "WRONG: a different first reply\n")
        rc, _, err = self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")
        self.assertEqual(rc, 0, err)
        self.assertIn("다시 생성됐다", err)
        self.assertIn("턴 0 raw 1개를", err)
        self.assertIn("이미 있던 것 2개", err)  # 원본이 그대로인 샘플 0·2 는 그대로 둔다
        copy = self.raw(1, tag=TAG_LINT)
        self.assertEqual(copy["response_text"], "WRONG: a different first reply\n")
        self.assertEqual(copy["seeded_sha256"], hashlib.sha256(self.raw_path(1).read_bytes()).hexdigest())
        archived = self.raw_path(1, tag=TAG_LINT).with_name("sample_1.prev1.json")
        self.assertEqual(json.loads(archived.read_text(encoding="utf-8")), old_copy)

    def test_reseeding_is_refused_when_the_copy_already_has_repair_turns(self) -> None:
        self.fake.lint = lint_stage([])
        self.standard_run()
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")[0], 0)
        for idx in range(3):
            self.fake.write_result(self.cfg, self.raw_path(idx, tag=TAG_LINT))
        self.assertEqual(self.main("--max-turns", "1", "--feedback", "compiler+lint")[0], 0)
        self.assertTrue(self.raw_path(2, 1, tag=TAG_LINT).exists())
        before = self.raw_path(2, tag=TAG_LINT).read_bytes()
        generate.archive_previous(self.raw_path(2))
        self.add_turn_zero(2, "WRONG: a different first reply\n")
        rc, _, err = self.main("--seed-from-feedback", "compiler", "--feedback", "compiler+lint")
        self.assertEqual(rc, 1, err)
        self.assertIn("수정 턴이 1개 있다", err)
        self.assertEqual(self.raw_path(2, tag=TAG_LINT).read_bytes(), before)  # 두 이력을 섞지 않는다
        self.assertFalse(self.raw_path(2, tag=TAG_LINT).with_name("sample_2.prev1.json").exists())

    def test_seeding_input_errors(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--seed-from-feedback", "compiler", "--feedback", "compiler")[0], 1)
        self.assertEqual(self.main("--seed-from-feedback", "compiler")[0], 1)  # 설정의 피드백도 compiler 다
        self.assertEqual(self.main("--seed-from-feedback", "compiler+lint", "--feedback", "compiler")[0], 1)  # 그 조건의 raw 가 없다
        self.assertEqual(self.fake.setup_calls, [])


class ExitCodeTest(LoopCase):
    def test_input_errors_exit_one(self) -> None:
        self.standard_run()
        self.assertEqual(self.main("--max-turns", "1", run_id="no-such-run")[0], 1)
        self.assertEqual(self.main("--max-turns", "1", "--samples", "one")[0], 1)
        self.assertEqual(self.main("--max-turns", "1", "--languages", "c_seq")[0], 1)  # 그 범위에 raw 가 없다
        self.assertEqual(self.main("--max-turns", "1", "--feedback", "compiler+lint")[0], 1)
        broken = self.out / "broken.yaml"
        broken.write_text("run: [", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(repair.main(["--config", str(broken), "--run-id", "r1"]), 1)
        self.assertEqual((self.fake.setup_calls, self.turn_files()), ([], []))

    def test_run_made_with_another_provider_or_model_is_refused(self) -> None:
        self.standard_run()  # provider=dummy 로 만든 run
        for env in ({"LLM_PROVIDER": "anthropic"}, {"LLM_MODEL": "another-model"}):
            with self.subTest(env=env), mock.patch.dict(os.environ, env), mock.patch.object(repair, "LLMClient", FakeClient):
                rc, _, err = self.main("--max-turns", "1", "--yes")
                self.assertEqual(rc, 1)
                self.assertIn("LLM_PROVIDER", err)
        self.assertEqual((FakeClient.instances, self.fake.setup_calls, self.turn_files()), ([], [], []))

    def test_nothing_processed_yet_exits_one(self) -> None:
        for idx in range(3):
            self.add_turn_zero(idx, process=False)
        rc, _, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 1)
        self.assertIn("harness.run", err)
        self.assertEqual(self.fake.setup_calls, [])

    def test_setup_failure_returns_the_code_of_harness_run(self) -> None:
        self.standard_run()
        for code in (7, 4, 6):
            self.fake.setup_error = FakeRun.RunSetupError("중단: 필요한 도구가 없다", code)
            rc, _, err = self.main("--max-turns", "1")
            self.assertEqual(rc, code)
            self.assertIn("필요한 도구가 없다", err)
        self.assertEqual(self.turn_files(), [])  # 준비가 실패하면 호출하지 않는다

    def test_harness_trouble_while_processing_writes_no_more_calls(self) -> None:
        cases: list[tuple[BaseException, int]] = [
            (FakeRun.HarnessError("check: worker_error"), 5), (ToolchainMissing("gcc 를 찾을 수 없다"), 7), (KeyError("unexpected"), 5),
            (KeyboardInterrupt(), 130)]
        for error, code in cases:
            with self.subTest(error=type(error).__name__):
                self.standard_run()
                for path in list((self.cfg.paths.raw_dir / "r1").rglob("*.turn*.json")):
                    path.unlink()
                self.fake.process_error = error
                rc, _, err = self.main("--max-turns", "2")
                self.assertEqual(rc, code, err)
                self.assertEqual(len(self.turn_files()), 1)  # 첫 샘플의 raw 는 남고(다음 실행이 처리한다) 거기서 멈춘다
                self.assertEqual(self.fake.finished, [])
                self.fake.process_error = None

    def test_unknown_stage_in_a_row_is_a_harness_error(self) -> None:
        self.standard_run()
        result = built_turn_dir(self.cfg.paths.built_dir, "r1", "gemm", "numpy", TAG, 1, 0) / "result.json"
        doc = json.loads(result.read_text(encoding="utf-8"))
        doc["row"]["error_type"] = "brand_new_type"
        result.write_text(json.dumps(doc), encoding="utf-8")
        rc, _, err = self.main("--max-turns", "1")
        self.assertEqual(rc, 5)
        self.assertIn("brand_new_type", err)
        self.assertEqual(self.turn_files(), [])


class RealProviderGateTest(LoopCase):
    """실제 제공자일 때의 확인 게이트와 호출 인자(가짜 클라이언트)."""

    def setUp(self) -> None:
        super().setUp()
        self.config_path = self.write_config(provider="anthropic")
        self.cfg = load_config(self.config_path)
        patcher = mock.patch.object(repair, "LLMClient", FakeClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.standard_run()

    def test_without_yes_a_non_interactive_run_is_declined(self) -> None:
        with mock.patch("sys.stdin", io.StringIO("")):
            rc, _, err = self.main("--max-turns", "2")
        self.assertEqual(rc, 2)
        self.assertIn("[repair] 2 calls to", err)
        self.assertIn("--yes", err)
        client = FakeClient.instances[0]
        self.assertEqual((client.preflighted, client.calls, self.turn_files()), (True, [], []))

    def test_each_turn_asks_again(self) -> None:
        answers = iter(["y", "n"])
        with mock.patch("sys.stdin", types.SimpleNamespace(isatty=lambda: True)), mock.patch("builtins.input", lambda prompt: next(answers)):
            FakeClient.replies = [("WRONG again", None), ("WRONG again", None)]
            rc, _, _ = self.main("--max-turns", "3")
        self.assertEqual(rc, 2)
        self.assertEqual(len(FakeClient.instances[0].calls), 2)  # 턴 1 은 했고 턴 2 에서 거절했다
        self.assertEqual(len(self.turn_files()), 2)
        self.assertEqual(len(self.fake.finished), 1)  # 처리한 턴까지는 마무리한다

    def test_yes_sends_the_conversation_with_the_stored_system_prompt(self) -> None:
        rc, out, err = self.main("--max-turns", "1", "--yes")
        self.assertEqual(rc, 0, err)
        client = FakeClient.instances[0]
        self.assertEqual([call["dummy_key"] for call in client.calls], [("gemm", "numpy", 1, 1), ("gemm", "numpy", 2, 1)])
        for call, idx in zip(client.calls, (1, 2)):
            first = self.raw(idx)
            self.assertEqual((call["system"], call["user"]), (first["prompt"]["system"], first["prompt"]["user"]))
            self.assertEqual(call["messages"], self.raw(idx, 1)["prompt"]["messages"])
            self.assertEqual(call["messages"][-1]["role"], "user")
        raw = self.raw(1, 1)
        self.assertEqual((raw["provider"], raw["model_served"], raw["sampling"]["effort"]), ("anthropic", "served-model", "medium"))
        self.assertIn("passed_after_repair=2", out)

    def test_fatal_api_error_stops_with_three_after_recording_the_raw(self) -> None:
        FakeClient.replies = [("", "fatal:AuthenticationError:401:invalid x-api-key")]
        rc, out, _ = self.main("--max-turns", "2", "--yes")
        self.assertEqual(rc, 3)
        self.assertEqual(len(FakeClient.instances[0].calls), 1)
        self.assertEqual(self.turn_files(), [f"gemm/numpy/{TAG}/sample_1.turn1.json"])
        self.assertEqual(self.raw(1, 1)["error"], "fatal:AuthenticationError:401:invalid x-api-key")
        self.assertEqual(self.fake.processed, [self.raw_path(1, 1)])  # 행은 생성 실패(valid_sample=0)로 남는다
        self.assertEqual(len(self.fake.finished), 1)
        self.assertIn("calls=1 failed=1", out)

    def test_transient_error_does_not_stop_the_turn_and_ends_that_chain_for_now(self) -> None:
        FakeClient.replies = [("", "transient:OverloadedError:529:Overloaded"), ("WRONG again", None), ("PASS", None)]
        rc, out, err = self.main("--max-turns", "2", "--yes")
        self.assertEqual(rc, 0, err)
        calls = FakeClient.instances[0].calls
        self.assertEqual([call["dummy_key"] for call in calls], [("gemm", "numpy", 1, 1), ("gemm", "numpy", 2, 1), ("gemm", "numpy", 2, 2)])
        self.assertIn("calls=3 failed=1 passed_after_repair=1", out)

    def test_preflight_failure_exits_three(self) -> None:
        with mock.patch.object(FakeClient, "preflight", side_effect=RuntimeError("401 invalid x-api-key")):
            rc, _, err = self.main("--max-turns", "1", "--yes")
        self.assertEqual(rc, 3)
        self.assertIn("preflight", err)
        self.assertEqual((self.fake.setup_calls, self.turn_files()), ([], []))


if __name__ == "__main__":
    unittest.main()
