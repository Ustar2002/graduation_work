"""harness/safety.py 테스트 (설계 v3 §12).

- 린트: 계약 검사(``parse_lint_output``), 가짜 플러그인 스크립트를 실제 ``run_isolated`` 로 띄운 ``run_lint`` 의 실패 경로,
  ``lint_columns``/``lint_messages``. 진짜 플러그인과의 연결은 ``tests/test_lint_plugin.py`` 가 본다.
- sanitizer: 환경변수 조립, ``tests/fixtures/safety/`` 의 로그(gcc libasan/libubsan 이 내는 형식으로 쓴 것)에 대한 파싱,
  빌드·워커를 가짜로 바꾼 ``run_sanitizers`` 의 순서와 판정. 실제 gcc·ASan 실행은 Linux 에서만 도는 마지막 클래스가 한다.

fixtures: ``<사례>/asan.<pid>``·``ubsan.<pid>`` 는 log_path 가 가리키는 디렉터리의 모습 그대로다.
``single_runtime`` 은 ASan 런타임에 든 UBSan 이 한 파일에 같이 쓴 경우, ``separate_runtimes`` 는 두 런타임이 따로 쓴 경우,
``stderr_*.txt`` 는 워커 stderr 의 꼬리(로그 파일이 아닌 곳에 남은 것)다.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

from harness import safety
from harness.config import load_config
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import ProcResult, run_isolated
from harness.protocol import launch_env_minimal
from tests._support import requires_linux, requires_tool

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "safety"
ZERO_SUMMARY = dict.fromkeys(safety.LINT_CATEGORIES, 0)


def proc_result(rc: int = 0, *, timed_out: bool = False, signal_name: str | None = None, stdout: str = "",
                stderr: str = "") -> ProcResult:
    return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.1, stdout_tail=stdout, stderr_tail=stderr, signal_name=signal_name)


def finding(line: int = 3, category: str = "memory", **extra: Any) -> dict[str, Any]:
    return {"rule": "ruff/F821", "severity": "error", "line": line, "message": "Undefined name `x`", "category": category, **extra}


def lint_doc(findings: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    summary = dict(ZERO_SUMMARY)
    for f in findings:
        summary[f["category"]] += 1
    return {"tool": "fake", "findings": findings, "summary": summary, **extra}


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="safety_test_")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)


# ----------------------------------------------------------------------------- 설정 읽기


class SettingTest(unittest.TestCase):
    def test_old_config_without_sections_falls_back_to_defaults(self) -> None:
        old = SimpleNamespace(build=SimpleNamespace(cc="gcc"))  # lint/safety 섹션도 build.sanitize_flags 도 없다
        self.assertIs(safety._setting(old, "lint", "enabled", True), True)
        self.assertIs(safety._setting(old, "safety", "enabled", False), False)
        self.assertEqual(safety._setting(old, "build", "sanitize_flags", safety.DEFAULT_SANITIZE_FLAGS), safety.DEFAULT_SANITIZE_FLAGS)
        self.assertEqual(safety._setting(old, "build", "cc", "cc"), "gcc")

    def test_new_config_values_win(self) -> None:
        new = SimpleNamespace(lint=SimpleNamespace(enabled=False, timeout_s=5), safety=SimpleNamespace(enabled=True, timeout_s=42))
        self.assertIs(safety._setting(new, "lint", "enabled", True), False)
        self.assertEqual(safety._setting(new, "lint", "timeout_s", 60.0), 5)
        self.assertIs(safety._setting(new, "safety", "enabled", False), True)
        self.assertEqual(safety._setting(new, "lint", "plugin", "plugins/lint/run_lint.py"), "plugins/lint/run_lint.py")

    def test_defaults_are_the_design_v3_values(self) -> None:
        self.assertEqual(safety.DEFAULT_LINT_PLUGIN, "plugins/lint/run_lint.py")
        self.assertEqual(safety.DEFAULT_LINT_TIMEOUT_S, 60.0)
        self.assertEqual(safety.DEFAULT_SAFETY_TIMEOUT_S, 300.0)
        self.assertEqual(list(safety.DEFAULT_SANITIZE_FLAGS), ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"])


# ----------------------------------------------------------------------------- 린트: 계약 검사


class ParseLintOutputTest(unittest.TestCase):
    def test_valid_output(self) -> None:
        res = safety.parse_lint_output(lint_doc([finding(3, "memory"), finding(9, "numeric"), finding(9, "numeric")]))
        self.assertTrue(res.ok)
        self.assertEqual(res.error, "")
        self.assertEqual(res.tool, "fake")
        self.assertEqual(res.summary, {**ZERO_SUMMARY, "memory": 1, "numeric": 2})
        self.assertEqual([f["line"] for f in res.findings], [3, 9, 9])

    def test_empty_findings_is_valid(self) -> None:
        res = safety.parse_lint_output({"tool": "none", "findings": [], "summary": dict(ZERO_SUMMARY)})
        self.assertTrue(res.ok)
        self.assertEqual(res.findings, [])

    def test_extra_keys_are_ignored_and_dropped(self) -> None:
        doc = lint_doc([finding(confidence=0.9)], unmapped=4, versions={"ruff": "0.17.6"})
        res = safety.parse_lint_output(doc)
        self.assertEqual(list(res.findings[0]), list(safety.FINDING_KEYS))

    def test_line_zero_is_allowed_for_file_level_findings(self) -> None:
        self.assertEqual(safety.parse_lint_output(lint_doc([finding(0)])).findings[0]["line"], 0)

    def test_null_category_means_not_checked(self) -> None:
        doc = lint_doc([finding(3, "memory")])
        doc["summary"].update(concurrency=None, performance=None)
        res = safety.parse_lint_output(doc)
        self.assertEqual(res.summary, {**ZERO_SUMMARY, "memory": 1, "concurrency": None, "performance": None})
        doc["summary"]["memory"] = None  # 검사하지 않았다면서 그 범주의 finding 이 있다
        with self.assertRaisesRegex(ValueError, "summary.memory"):
            safety.parse_lint_output(doc)

    def test_violations(self) -> None:
        good = lint_doc([finding()])
        cases: dict[str, Any] = {
            "최상위가 리스트": [good],
            "tool 없음": {k: v for k, v in good.items() if k != "tool"},
            "tool 빈 문자열": {**good, "tool": ""},
            "findings 가 리스트가 아님": {**good, "findings": {"0": finding()}},
            "finding 이 객체가 아님": {**good, "findings": ["line 3: x"]},
            "finding 에 message 없음": lint_doc([{k: v for k, v in finding().items() if k != "message"}]),
            "rule 빈 문자열": lint_doc([finding(rule="")]),
            "severity 가 문자열이 아님": lint_doc([finding(severity=2)]),
            "message 가 문자열이 아님": lint_doc([finding(message=None)]),
            "line 이 문자열": lint_doc([finding(line="3")]),
            "line 이 불리언": lint_doc([finding(line=True)]),
            "line 이 음수": lint_doc([finding(line=-1)]),
            "summary 없음": {k: v for k, v in good.items() if k != "summary"},
            "summary 에 범주 하나 없음": {**good, "summary": {k: v for k, v in good["summary"].items() if k != "performance"}},
            "summary 에 모르는 키": {**good, "summary": {**good["summary"], "style": 0}},
            "summary 수가 findings 와 다름": {**good, "summary": {**good["summary"], "memory": 2}},
            "summary 값이 실수": {**good, "summary": {**good["summary"], "memory": 1.0}},
        }
        for name, doc in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                safety.parse_lint_output(doc)

    def test_unknown_category_is_a_violation(self) -> None:
        doc = {"tool": "fake", "findings": [finding(category="style")], "summary": dict(ZERO_SUMMARY)}
        with self.assertRaisesRegex(ValueError, "category"):
            safety.parse_lint_output(doc)


# ----------------------------------------------------------------------------- 린트: 플러그인 실행


FAKE_PLUGIN_HEAD = """\
import json, sys, time
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
"""


class RunLintTest(TempDirTest):
    """가짜 플러그인 스크립트를 진짜 ``run_isolated`` 로 띄운다(계약의 CLI 로 불리는지, 실패를 삼키는지)."""

    def setUp(self) -> None:
        super().setUp()
        self.source = self.tmp / "kernel.py"
        self.source.write_text("def gemm(A, B, C, M, N, K):\n    pass\n", encoding="utf-8")
        self.out = self.tmp / "sample" / "lint.json"
        self.out.parent.mkdir()

    def plugin(self, body: str) -> Path:
        path = self.tmp / "fake_plugin.py"
        path.write_text(FAKE_PLUGIN_HEAD + body, encoding="utf-8")
        return path

    def run_lint(self, plugin: Path, *, timeout_s: float = 30.0, enabled: bool = True,
                 cc: str | None = None) -> tuple[safety.LintResult, str]:
        cfg = SimpleNamespace(lint=SimpleNamespace(enabled=enabled, plugin=plugin, timeout_s=timeout_s))
        if cc is not None:
            cfg.build = SimpleNamespace(cc=cc)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res = safety.run_lint(cfg, language="numpy", backend="python", file=self.source, task="gemm",
                                  signature={"function": "gemm", "params": ["A", "B"], "note": 'quote " and 한글'},
                                  out=self.out, cwd=self.tmp)
        return res, stderr.getvalue()

    def assert_failed(self, res: safety.LintResult, stderr: str, error_prefix: str) -> None:
        self.assertFalse(res.ok)
        self.assertTrue(res.error.startswith(error_prefix), res.error)
        self.assertEqual((res.tool, res.findings, res.summary), (safety.NO_TOOL, [], ZERO_SUMMARY))
        lines = stderr.splitlines()
        self.assertEqual(len(lines), 1, f"경고는 정확히 한 줄이어야 한다: {stderr!r}")
        self.assertTrue(lines[0].startswith("[safety] "), lines[0])
        self.assertIn("numpy/python", lines[0])
        self.assertIn("kernel.py", lines[0])
        self.assertIn(error_prefix, lines[0])

    def test_plugin_is_called_with_the_contract_cli_and_its_output_is_kept(self) -> None:
        doc = lint_doc([finding(2, "memory"), finding(2, "performance")], unmapped=1)
        plugin = self.plugin(
            f"json.dump(sys.argv[1:], open(args['--out'] + '.argv', 'w'))\njson.dump({doc!r}, open(args['--out'], 'w'))\n"
        )
        res, stderr = self.run_lint(plugin)
        self.assertEqual(stderr, "")
        self.assertTrue(res.ok, res.error)
        self.assertEqual(res.tool, "fake")
        self.assertEqual(res.summary, {**ZERO_SUMMARY, "memory": 1, "performance": 1})
        self.assertEqual(json.loads(self.out.read_text(encoding="utf-8")), doc, "플러그인 출력 원문이 out 에 남는다")
        argv = json.loads(Path(str(self.out) + ".argv").read_text(encoding="utf-8"))
        self.assertEqual(argv[0::2], ["--language", "--file", "--task", "--signature", "--out"])
        self.assertEqual(argv[1], "numpy")
        self.assertEqual(Path(argv[3]), self.source)
        self.assertEqual(argv[5], "gemm")
        self.assertEqual(json.loads(argv[7]), {"function": "gemm", "params": ["A", "B"], "note": 'quote " and 한글'})
        self.assertEqual(Path(argv[9]), self.out)

    def test_configured_c_compiler_is_passed_as_cc(self) -> None:
        doc = lint_doc([])
        body = f"import os\njson.dump(os.environ.get('CC'), open(args['--out'] + '.cc', 'w'))\njson.dump({doc!r}, open(args['--out'], 'w'))\n"
        cc_file = Path(str(self.out) + ".cc")
        for cc, expected in (("gcc-15", "gcc-15"), (None, None)):
            with self.subTest(cc=cc):
                res, _ = self.run_lint(self.plugin(body), cc=cc)
                self.assertTrue(res.ok, res.error)
                self.assertEqual(json.loads(cc_file.read_text(encoding="utf-8")), expected)

    def test_missing_plugin_file(self) -> None:
        res, stderr = self.run_lint(self.tmp / "no_such_plugin.py")
        self.assert_failed(res, stderr, "plugin_missing")

    def test_nonzero_exit(self) -> None:
        res, stderr = self.run_lint(self.plugin("print('rules.yaml is broken', file=sys.stderr)\nsys.exit(2)\n"))
        self.assert_failed(res, stderr, "plugin_failed: rc=2")
        self.assertIn("rules.yaml is broken", res.error)

    def test_plugin_stderr_is_kept_next_to_the_output(self) -> None:
        doc = lint_doc([])
        plugin = self.plugin(f"print('[lint] clang-tidy: omp.h not found', file=sys.stderr)\njson.dump({doc!r}, open(args['--out'], 'w'))\n")
        res, stderr = self.run_lint(plugin)
        self.assertEqual((res.ok, stderr), (True, ""), "플러그인의 stderr 는 하네스의 stderr 로 새지 않는다")
        self.assertEqual(sorted(p.name for p in self.out.parent.iterdir()), ["lint.json", "lint.stderr.log"])
        self.assertIn("clang-tidy: omp.h not found", (self.out.parent / "lint.stderr.log").read_text(encoding="utf-8"))

    def test_output_directory_is_created(self) -> None:
        self.out = self.tmp / "not_yet" / "turn_1" / "lint.json"
        doc = lint_doc([finding()])
        res, _ = self.run_lint(self.plugin(f"json.dump({doc!r}, open(args['--out'], 'w'))\n"))
        self.assertTrue(res.ok, res.error)

    def test_nonzero_exit_ignores_whatever_was_written(self) -> None:
        doc = lint_doc([finding()])
        res, stderr = self.run_lint(self.plugin(f"json.dump({doc!r}, open(args['--out'], 'w'))\nsys.exit(1)\n"))
        self.assert_failed(res, stderr, "plugin_failed: rc=1")

    def test_timeout(self) -> None:
        res, stderr = self.run_lint(self.plugin("time.sleep(60)\n"), timeout_s=1.0)
        self.assert_failed(res, stderr, "timeout after 1s")

    def test_invalid_json(self) -> None:
        res, stderr = self.run_lint(self.plugin("open(args['--out'], 'w').write('{\"tool\": ')\n"))
        self.assert_failed(res, stderr, "invalid_json")

    def test_no_output_file(self) -> None:
        res, stderr = self.run_lint(self.plugin("pass\n"))
        self.assert_failed(res, stderr, "no_output")

    def test_stale_output_of_an_earlier_run_is_not_read(self) -> None:
        self.out.write_text(json.dumps(lint_doc([finding()])), encoding="utf-8")
        res, stderr = self.run_lint(self.plugin("pass\n"))
        self.assert_failed(res, stderr, "no_output")
        self.assertFalse(self.out.exists())

    def test_contract_violation(self) -> None:
        doc = {**lint_doc([finding()]), "summary": {**ZERO_SUMMARY, "memory": 5}}
        res, stderr = self.run_lint(self.plugin(f"json.dump({doc!r}, open(args['--out'], 'w'))\n"))
        self.assert_failed(res, stderr, "contract_violation")

    def test_disabled_does_not_run_the_plugin_and_does_not_warn(self) -> None:
        marker = self.tmp / "ran"
        res, stderr = self.run_lint(self.plugin(f"open({str(marker)!r}, 'w').close()\n"), enabled=False)
        self.assertEqual((res.ok, res.error, stderr), (False, "disabled", ""))
        self.assertFalse(marker.exists())


class LintColumnsAndMessagesTest(unittest.TestCase):
    def result(self, findings: list[dict[str, Any]], tool: str = "baseline:ruff+pyflakes") -> safety.LintResult:
        return safety.parse_lint_output({**lint_doc(findings), "tool": tool})

    def test_columns_of_a_real_result(self) -> None:
        res = self.result([finding(1, "memory"), finding(2, "numeric"), finding(3, "numeric"), finding(4, "performance")])
        self.assertEqual(safety.lint_columns(res), {
            "lint_findings_total": 4, "lint_memory": 1, "lint_numeric": 2, "lint_concurrency": 0, "lint_shape_dtype": 0,
            "lint_api_misuse": 0, "lint_performance": 1,
        })

    def test_columns_of_a_clean_result_are_zero(self) -> None:
        self.assertEqual(set(safety.lint_columns(self.result([])).values()), {0})

    def test_columns_are_na_when_the_plugin_failed(self) -> None:
        columns = safety.lint_columns(safety._empty_lint("timeout after 60s"))
        self.assertEqual(len(columns), 7)
        self.assertEqual(set(columns.values()), {None})

    def test_columns_are_na_when_no_tool_ran(self) -> None:
        self.assertEqual(set(safety.lint_columns(self.result([], tool=safety.NO_TOOL)).values()), {None})

    def test_categories_no_tool_could_check_are_na_not_zero(self) -> None:
        # c_openmp 에서 clang-tidy 가 빠지고 cppcheck 만 돈 경우: cppcheck 가 보지 않는 범주는 null
        doc = {**lint_doc([finding(1, "memory"), finding(2, "memory")]), "tool": "baseline:cppcheck"}
        doc["summary"].update(concurrency=None, api_misuse=None, performance=None)
        self.assertEqual(safety.lint_columns(safety.parse_lint_output(doc)), {
            "lint_findings_total": 2, "lint_memory": 2, "lint_numeric": 0, "lint_concurrency": None, "lint_shape_dtype": 0,
            "lint_api_misuse": None, "lint_performance": None,
        })
        doc = {"tool": "other-plugin", "findings": [], "summary": dict.fromkeys(safety.LINT_CATEGORIES)}
        self.assertEqual(set(safety.lint_columns(safety.parse_lint_output(doc)).values()), {None},
                         "검사한 범주가 하나도 없으면 합계도 NA 다")

    def test_column_names_are_row_fields(self) -> None:
        from harness.schema import RunRow

        fields = set(RunRow.__dataclass_fields__)
        self.assertLessEqual(set(safety.lint_columns(self.result([]))), fields)

    def test_messages_keep_plugin_order_and_are_one_line_each(self) -> None:
        res = self.result([finding(12, message="Undefined name `acc`"), finding(3, message="two\n  lines\there")])
        self.assertEqual(safety.lint_messages(res), ["line 12: Undefined name `acc`", "line 3: two lines here"])

    def test_messages_limit(self) -> None:
        res = self.result([finding(i) for i in range(1, 31)])
        self.assertEqual(len(safety.lint_messages(res)), 20)
        self.assertEqual(safety.lint_messages(res, limit=2), ["line 1: Undefined name `x`", "line 2: Undefined name `x`"])
        self.assertEqual(safety.lint_messages(safety._empty_lint("disabled")), [])


# ----------------------------------------------------------------------------- sanitizer: 환경


class SanitizerOptionsTest(unittest.TestCase):
    def test_plain_values(self) -> None:
        self.assertEqual(safety.sanitizer_options({"detect_leaks": "0", "log_path": "/r/built/x/san/asan"}),
                         "detect_leaks=0:log_path=/r/built/x/san/asan")

    def test_values_with_separators_are_quoted(self) -> None:
        self.assertEqual(safety.sanitizer_options({"log_path": "/home/u/my runs/san/asan"}), "log_path='/home/u/my runs/san/asan'")
        self.assertEqual(safety.sanitizer_options({"log_path": "/a,b/asan"}), "log_path='/a,b/asan'")
        self.assertEqual(safety.sanitizer_options({"log_path": "/a:b/asan"}), "log_path='/a:b/asan'")
        self.assertEqual(safety.sanitizer_options({"log_path": "/it's here/asan"}), 'log_path="/it\'s here/asan"')

    def test_unquotable_value_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            safety.sanitizer_options({"log_path": "/a 'b' \"c\"/asan"})


def fake_print_file_name(libasan_output: str, libubsan_output: str = "", rc: int = 0) -> Any:
    """``run_isolated`` 자리: ``<cc> -print-file-name=libasan.so`` / ``libubsan.so`` 에 정해진 출력을 준다."""

    def run(cmd: list[str], **kwargs: Any) -> ProcResult:
        return proc_result(rc, stdout=libubsan_output if cmd[-1].endswith("libubsan.so") else libasan_output)

    return run


class SanitizerEnvTest(TempDirTest):
    def env(self, cfg: Any, compiler_output: str, rc: int = 0, ubsan_output: str = "") -> tuple[dict[str, str], mock.Mock, str]:
        stderr = io.StringIO()
        with mock.patch.object(safety, "run_isolated", side_effect=fake_print_file_name(compiler_output, ubsan_output, rc)) as run, \
                contextlib.redirect_stderr(stderr):
            env = safety.sanitizer_env(cfg, "gcc-15", self.tmp)
        return env, run, stderr.getvalue()

    def test_env_for_a_resolved_libasan(self) -> None:
        libasan = self.tmp / "libasan.so"
        libasan.write_bytes(b"\x7fELF")
        libubsan = self.tmp / "libubsan.so"
        libubsan.write_bytes(b"\x7fELF")
        cfg = SimpleNamespace(build=SimpleNamespace(compile_timeout_s=45))
        env, run, stderr = self.env(cfg, f"{libasan}\n", ubsan_output=f"{libubsan}\n")
        self.assertEqual(stderr, "")
        self.assertEqual(set(env), {"LD_PRELOAD", "ASAN_OPTIONS", "UBSAN_OPTIONS"})
        # gcc 의 libubsan 도 선행 로드한다(워커가 PR_SET_DUMPABLE 을 끄기 전에 UBSAN_OPTIONS 를 읽게). libasan 이 먼저다.
        self.assertEqual(env["LD_PRELOAD"], f"{libasan}:{libubsan}")
        self.assertEqual([c.args[0] for c in run.call_args_list],
                         [["gcc-15", "-print-file-name=libasan.so"], ["gcc-15", "-print-file-name=libubsan.so"]])
        self.assertEqual({c.kwargs["timeout_s"] for c in run.call_args_list}, {45.0})
        asan = env["ASAN_OPTIONS"]
        for option in ("detect_leaks=0", "abort_on_error=1", "allocator_may_return_null=1", "handle_sigfpe=0"):
            self.assertIn(option, asan.split(":"))
        self.assertTrue(asan.endswith(safety.sanitizer_options({"log_path": str(self.tmp / "asan")})), asan)
        ubsan = env["UBSAN_OPTIONS"]
        for option in ("print_stacktrace=1", "halt_on_error=0"):
            self.assertIn(option, ubsan.split(":"))
        self.assertTrue(ubsan.endswith(safety.sanitizer_options({"log_path": str(self.tmp / "ubsan")})), ubsan)

    def test_ubsan_inside_the_asan_runtime_adds_nothing(self) -> None:
        # clang 처럼 UBSan 이 ASan 런타임 안에 있으면 libubsan.so 가 없다(넘겨받은 이름이 그대로 나온다): 경고 없이 libasan 만
        libasan = self.tmp / "libasan.so"
        libasan.write_bytes(b"\x7fELF")
        env, _, stderr = self.env(SimpleNamespace(), f"{libasan}\n", ubsan_output="libubsan.so\n")
        self.assertEqual((env["LD_PRELOAD"], stderr), (str(libasan), ""))

    def test_unresolved_libasan_is_returned_anyway_with_a_note(self) -> None:
        # gcc 는 파일을 못 찾으면 넘겨받은 이름을 그대로 출력한다
        env, run, stderr = self.env(SimpleNamespace(), "libasan.so\n", ubsan_output=f"{self.tmp}\n")
        self.assertEqual(env["LD_PRELOAD"], "libasan.so", "libasan 이 없으면 libubsan 만 앞에 두지 않는다")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.kwargs["timeout_s"], safety.DEFAULT_COMPILE_TIMEOUT_S)
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertTrue(stderr.startswith("[safety] "))
        self.assertIn("libasan.so", stderr)

    def test_compiler_that_prints_nothing(self) -> None:
        env, _, stderr = self.env(SimpleNamespace(), "", rc=1)
        self.assertEqual(env["LD_PRELOAD"], "")
        self.assertEqual(len(stderr.splitlines()), 1)

    @requires_linux
    @requires_tool("gcc")
    def test_real_gcc_resolves_an_existing_libasan(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            env = safety.sanitizer_env(SimpleNamespace(), "gcc", self.tmp)
        preload = env["LD_PRELOAD"].split(":")
        self.assertEqual(len(preload), 2, f"gcc 는 libasan 과 libubsan 을 둘 다 선행 로드한다: {env['LD_PRELOAD']!r}")
        for path in preload:
            self.assertTrue(Path(path).is_file(), f"{env['LD_PRELOAD']!r} / {stderr.getvalue()}")
        self.assertIn("libasan", Path(preload[0]).name)


# ----------------------------------------------------------------------------- sanitizer: 로그 파싱


class SanitizerLogParsingTest(TempDirTest):
    def test_fixture_directories(self) -> None:
        expected = {
            "heap_overflow_write": (1, 0, ["heap-buffer-overflow"]),
            "heap_overflow_read": (1, 0, ["heap-buffer-overflow"]),
            "stack_overflow": (1, 0, ["stack-buffer-overflow"]),
            "use_after_free": (1, 0, ["heap-use-after-free"]),
            "segv": (1, 0, ["SEGV"]),
            "ubsan_three_kinds": (0, 3, ["signed-integer-overflow", "out-of-bounds-index", "misaligned-pointer-use"]),
            "ubsan_repeated": (0, 2, ["signed-integer-overflow", "out-of-bounds-index"]),
            "single_runtime": (1, 2, ["stack-buffer-underflow", "signed-integer-overflow", "out-of-bounds-index"]),
            "separate_runtimes": (1, 2, ["double-free", "invalid-shift-exponent", "divide-by-zero"]),
        }
        self.assertEqual(set(expected), {p.name for p in FIXTURES.iterdir() if p.is_dir()}, "fixture 디렉터리와 기대표가 어긋난다")
        for name, result in expected.items():
            with self.subTest(name):
                self.assertEqual(safety.parse_sanitizer_logs(FIXTURES / name), result)

    def test_the_fixtures_carry_the_lines_the_parser_keys_on(self) -> None:
        write = (FIXTURES / "heap_overflow_write" / "asan.12345").read_text(encoding="utf-8")
        self.assertIn("==12345==ERROR: AddressSanitizer: heap-buffer-overflow on address ", write)
        self.assertIn("\nWRITE of size 4 ", write)
        self.assertIn("\nSUMMARY: AddressSanitizer: heap-buffer-overflow kernel.c:12 in gemm\n", write)
        self.assertIn("\nREAD of size 4 ", (FIXTURES / "heap_overflow_read" / "asan.12350").read_text(encoding="utf-8"))
        ubsan = (FIXTURES / "ubsan_three_kinds" / "ubsan.12349").read_text(encoding="utf-8")
        self.assertIn("kernel.c:10:15: runtime error: signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'", ubsan)
        self.assertIn("runtime error: index 8 out of bounds for type 'float [8]'", ubsan)
        self.assertIn("runtime error: load of misaligned address ", ubsan)

    def test_repeated_location_counts_once(self) -> None:
        text = (FIXTURES / "ubsan_repeated" / "ubsan.12351").read_text(encoding="utf-8")
        self.assertEqual(text.count("kernel.c:10:15: runtime error:"), 40)
        self.assertEqual(text.count("kernel.c:12:9: runtime error:"), 3)
        self.assertEqual(safety.ubsan_errors(text), {"kernel.c:10:15": "signed-integer-overflow", "kernel.c:12:9": "out-of-bounds-index"})

    def test_location_is_the_same_with_or_without_directories(self) -> None:
        text = ("/abs/san/kernel.c:10:15: runtime error: signed integer overflow: 1 + 2147483647 cannot be represented in type 'int'\n"
                "kernel.c:10:15: runtime error: signed integer overflow: 2 + 2147483647 cannot be represented in type 'int'\n")
        self.assertEqual(list(safety.ubsan_errors(text)), ["kernel.c:10:15"])

    def test_ubsan_kinds(self) -> None:
        messages = {
            "signed integer overflow: 2147483647 + 1 cannot be represented in type 'int'": "signed-integer-overflow",
            "negation of -2147483648 cannot be represented in type 'int'; cast to an unsigned type to negate this value to itself": "signed-integer-overflow",
            "division by zero": "divide-by-zero",
            "index 8 out of bounds for type 'float [8]'": "out-of-bounds-index",
            "load of misaligned address 0x7c5e2a1e0041 for type 'const float', which requires 4 byte alignment": "misaligned-pointer-use",
            "store to null pointer of type 'float'": "null-pointer-use",
            "shift exponent 40 is too large for 32-bit type 'int'": "invalid-shift-exponent",
            "left shift of 1073741824 by 2 places cannot be represented in type 'int'": "invalid-shift-base",
            "1e+10 is outside the range of representable values of type 'int'": "float-cast-overflow",
            "pointer index expression with base 0x7ffe00000010 overflowed to 0x000000000008": "pointer-overflow",
            "variable length array bound evaluates to non-positive value -3": "non-positive-vla-index",
            "load of address 0x7ffc5d4e3a10 with insufficient space for an object of type 'float'": "insufficient-object-size",
            "execution reached an unreachable program point": "undefined-behavior",
        }
        for i, (message, kind) in enumerate(messages.items(), start=1):
            with self.subTest(message):
                self.assertEqual(safety.ubsan_errors(f"kernel.c:{i}:1: runtime error: {message}\n"), {f"kernel.c:{i}:1": kind})

    def test_empty_and_missing_log_directory(self) -> None:
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp), (0, 0, []))
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp / "never_created"), (0, 0, []))
        self.assertEqual(safety.sanitizer_log_files(self.tmp / "never_created"), [])

    def test_only_log_files_are_read(self) -> None:
        report = (FIXTURES / "segv" / "asan.12348").read_text(encoding="utf-8")
        for name in ("build.log", "kernel.c", "asan", "notasan.1"):
            (self.tmp / name).write_text(report, encoding="utf-8")
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp), (0, 0, []))
        (self.tmp / "ubsan.77").write_text(report, encoding="utf-8")
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp), (1, 0, ["SEGV"]))
        self.assertEqual([p.name for p in safety.sanitizer_log_files(self.tmp)], ["ubsan.77"])

    def test_stderr_text_alone(self) -> None:
        tail = (FIXTURES / "stderr_asan_tail.txt").read_text(encoding="utf-8")
        self.assertNotIn("ERROR: AddressSanitizer", tail, "꼬리에는 ERROR 줄이 잘려 나가고 SUMMARY 만 남아 있다")
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp, tail), (1, 0, ["heap-buffer-overflow"]))

    def test_a_report_in_both_the_log_and_stderr_counts_once(self) -> None:
        tail = (FIXTURES / "stderr_asan_tail.txt").read_text(encoding="utf-8")
        self.assertEqual(safety.parse_sanitizer_logs(FIXTURES / "heap_overflow_write", tail), (1, 0, ["heap-buffer-overflow"]))
        ubsan = (FIXTURES / "ubsan_three_kinds" / "ubsan.12349").read_text(encoding="utf-8")
        self.assertEqual(safety.parse_sanitizer_logs(FIXTURES / "ubsan_three_kinds", ubsan)[1], 3)

    def test_runtime_startup_failure_is_not_a_finding(self) -> None:
        text = (FIXTURES / "stderr_preload_failed.txt").read_text(encoding="utf-8")
        self.assertIn("ASan runtime does not come first", text)
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp, text), (0, 0, []))

    def test_report_cut_off_before_its_summary_still_counts(self) -> None:
        head = "".join((FIXTURES / "heap_overflow_write" / "asan.12345").read_text(encoding="utf-8").splitlines(keepends=True)[:6])
        self.assertNotIn("SUMMARY", head)
        self.assertEqual(list(safety.asan_reports(head).values()), ["heap-buffer-overflow"])
        double_free = "==7==ERROR: AddressSanitizer: attempting double-free on 0x7b8e5c9e0040 in thread T0:\n    #0 0x1 in free\n"
        self.assertEqual(list(safety.asan_reports(double_free).values()), ["double-free"])
        bad_free = "==7==ERROR: AddressSanitizer: attempting free on address which was not malloc()-ed: 0x7ffd3a in thread T0\n"
        self.assertEqual(list(safety.asan_reports(bad_free).values()), ["bad-free"])

    def test_leak_summary_and_warnings_are_not_error_reports(self) -> None:
        text = ("==9==WARNING: AddressSanitizer failed to allocate 0x10000000000 bytes\n"
                "==9==ERROR: LeakSanitizer: detected memory leaks\n"
                "SUMMARY: AddressSanitizer: 4096 byte(s) leaked in 1 allocation(s).\n")
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp, text), (0, 0, []))

    def test_several_log_files_and_crlf_line_endings(self) -> None:
        # autocrlf 로 체크아웃된 fixture 나 Windows 에서 옮긴 로그도 같은 결과여야 한다
        crlf: dict[str, str] = {}
        for name in ("single_runtime", "separate_runtimes"):
            for src in (FIXTURES / name).iterdir():
                crlf[src.name] = src.read_text(encoding="utf-8").replace("\n", "\r\n")
                (self.tmp / src.name).write_bytes(crlf[src.name].encode("utf-8"))
        # 파일은 이름순(asan.12353, ubsan.12352, ubsan.12353)으로 읽고, 종류는 ASan 것이 먼저다
        expected = (2, 4, ["double-free", "stack-buffer-underflow", "signed-integer-overflow", "out-of-bounds-index",
                           "invalid-shift-exponent", "divide-by-zero"])
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp), expected)
        joined = "".join(crlf[name] for name in sorted(crlf))
        self.assertEqual(safety.parse_sanitizer_logs(self.tmp / "never_created", joined), expected)

    def test_cli_parse_logs(self) -> None:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = safety.main(["--parse-logs", str(FIXTURES / "separate_runtimes")])
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(stdout.getvalue()), {
            "asan_findings": 1, "ubsan_findings": 2, "kinds": ["double-free", "invalid-shift-exponent", "divide-by-zero"],
            "log_files": ["asan.12353", "ubsan.12353"],
        })


# ----------------------------------------------------------------------------- sanitizer: 빌드·실행 (가짜 빌드·워커)


class ToolchainMissingStandIn(RuntimeError):
    """``harness.build.c.ToolchainMissing`` 자리: 컴파일러가 없을 때 build_c 가 던지는 예외."""


class RunSanitizersTest(TempDirTest):
    CODE = "void gemm(const float* A, const float* B, float* C, int M, int N, int K) {}\n"

    def setUp(self) -> None:
        super().setUp()
        self.sample_dir = self.tmp / "sample_0"
        self.san_dir = self.sample_dir / "san"
        self.libasan = self.tmp / "libasan.so"
        self.libasan.write_bytes(b"\x7fELF")
        self.libubsan = self.tmp / "libubsan.so"
        self.libubsan.write_bytes(b"\x7fELF")
        self.tests = [self.tmp / "tiny.npz", self.tmp / "rect.npz"]
        self.build_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.worker_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.cfg = SimpleNamespace(safety=SimpleNamespace(enabled=True), build=SimpleNamespace(cc="gcc"))

    def fake_build(self, *args: Any, ok: bool = True, **kwargs: Any) -> SimpleNamespace:
        self.build_calls.append((args, kwargs))
        out_dir = args[1]
        self.assertTrue(out_dir.is_dir(), "build_c 를 부르기 전에 san/ 이 있어야 한다")
        command = ["gcc", "-O3", *kwargs["extra_flags"], "-o", kwargs["artifact_name"], "kernel.c"]
        if not ok:
            return SimpleNamespace(ok=False, artifact=None, error_kind="compile_error", error_detail="kernel.c:3:1: error: boom",
                                   command=command)
        return SimpleNamespace(ok=True, artifact=out_dir / kwargs["artifact_name"], error_kind="", error_detail="", command=command)

    def worker(self, *, proc: ProcResult, doc: dict[str, Any] | None, logs: tuple[Path, ...] = (),
               stderr_file: str | None = None) -> Any:
        """sanitizer 워커 흉내: 불리면 ``logs`` (fixture 파일)를 san/ 에 복사하고, ``stderr_file`` 이 있으면 run.run_worker 처럼
        샘플 디렉터리의 ``safety.stderr.log`` 에 쓰고, 정해진 결과를 돌려준다."""

        def call(*args: Any, **kwargs: Any) -> SimpleNamespace:
            self.worker_calls.append((args, kwargs))
            for src in logs:
                shutil.copy(src, self.san_dir / src.name)
            if stderr_file is not None:
                (self.sample_dir / "safety.stderr.log").write_text(stderr_file, encoding="utf-8")
            return SimpleNamespace(proc=proc, doc=doc, summary=lambda: {"rc": proc.rc, "signal": proc.signal_name})

        return call

    def sanitize(self, worker: Any, *, cfg: Any = None, is_linux: bool = True, build_ok: bool = True) -> tuple[safety.SanitizeResult, str]:
        """``run_sanitizers`` 를 가짜 빌드·가짜 libasan 경로로 부른다(Linux 인 것으로 친다). (결과, stderr)."""
        stderr = io.StringIO()
        with mock.patch.object(safety, "IS_LINUX", is_linux), \
                mock.patch.object(safety, "build_c", side_effect=lambda *a, **k: self.fake_build(*a, ok=build_ok, **k)), \
                mock.patch.object(safety, "run_isolated", side_effect=fake_print_file_name(f"{self.libasan}\n", f"{self.libubsan}\n")), \
                contextlib.redirect_stderr(stderr):
            res = safety.run_sanitizers(cfg or self.cfg, code=self.CODE, sample_dir=self.sample_dir, openmp=True,
                                        launch_prefix=["taskset", "-c", "0,2,4,6"], symbol="gemm",
                                        args_header="typedef struct {} gemm_args;\n", tests=self.tests, worker=worker)
        return res, stderr.getvalue()

    def test_disabled_by_default(self) -> None:
        never = self.worker(proc=proc_result(), doc={"complete": True})
        for cfg in (SimpleNamespace(build=SimpleNamespace(cc="gcc")), SimpleNamespace(safety=SimpleNamespace(enabled=False))):
            res, _ = self.sanitize(never, cfg=cfg)
            self.assertEqual(res, safety.SanitizeResult(ran=False, build_ok=False, asan_findings=None, ubsan_findings=None,
                                                         kinds=[], detail="disabled", summary={}))
        self.assertEqual((self.build_calls, self.worker_calls), ([], []))
        self.assertFalse(self.sample_dir.exists())

    def test_not_linux(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc={"complete": True}), is_linux=False)
        self.assertEqual((res.ran, res.build_ok, res.asan_findings, res.ubsan_findings), (False, False, None, None))
        self.assertEqual(res.detail, f"unsupported_platform:{sys.platform}")
        self.assertEqual((self.build_calls, self.worker_calls), ([], []))

    def test_build_failure(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc={"complete": True}), build_ok=False)
        self.assertEqual((res.ran, res.build_ok, res.asan_findings, res.ubsan_findings, res.kinds), (False, False, None, None, []))
        self.assertEqual(res.detail, "build_failed:compile_error: kernel.c:3:1: error: boom")
        self.assertEqual(res.summary["build"]["ok"], False)
        self.assertEqual(self.worker_calls, [])

    def test_toolchain_missing_propagates(self) -> None:
        with mock.patch.object(safety, "IS_LINUX", True), \
                mock.patch.object(safety, "build_c", side_effect=ToolchainMissingStandIn("gcc: command not found")), \
                self.assertRaises(ToolchainMissingStandIn):
            safety.run_sanitizers(self.cfg, code=self.CODE, sample_dir=self.sample_dir, openmp=False, launch_prefix=[],
                                  symbol="gemm", args_header=None, tests=self.tests,
                                  worker=self.worker(proc=proc_result(), doc=None))

    def test_asan_abort_is_a_finding_and_the_calls_follow_the_contract(self) -> None:
        aborted = proc_result(-6, signal_name="SIGABRT")
        partial = {"complete": False, "load_ok": True, "in_progress": {"test": "tiny", "call_idx": 0}}
        res, stderr = self.sanitize(self.worker(proc=aborted, doc=partial, logs=[FIXTURES / "heap_overflow_write" / "asan.12345"]))
        self.assertEqual(stderr, "")
        self.assertEqual((res.ran, res.build_ok), (True, True))
        self.assertEqual(res.asan_findings, 1)
        self.assertIsNone(res.ubsan_findings, "첫 ASan 오류에서 죽었으므로 UBSan 0 건은 판정 불가다")
        self.assertEqual(res.kinds, ["heap-buffer-overflow"])
        self.assertEqual(res.detail, "incomplete:signal SIGABRT")

        (build_args, build_kwargs), = self.build_calls
        self.assertEqual(build_args, (self.CODE, self.san_dir, self.cfg.build))
        self.assertEqual(build_kwargs, {
            "openmp": True, "launch_prefix": ["taskset", "-c", "0,2,4,6"], "symbol": "gemm",
            "args_header": "typedef struct {} gemm_args;\n",
            "extra_flags": ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"], "artifact_name": "kernel_san.so",
        })
        (worker_args, worker_kwargs), = self.worker_calls
        # --sanitize: 워커가 가드 영역 없이 정확한 크기로 할당해 하네스 배열을 넘는 접근이 ASan 에 보이게 한다
        self.assertEqual(worker_args, ("safety", "check", self.tests, 300.0, ["--check-repeats", "1", "--sanitize"]))
        self.assertEqual(set(worker_kwargs), {"artifact", "env_extra", "max_mem_gb"})
        self.assertEqual(worker_kwargs["artifact"], self.san_dir / "kernel_san.so")
        self.assertEqual(worker_kwargs["max_mem_gb"], 0)
        env = worker_kwargs["env_extra"]
        self.assertEqual(env["LD_PRELOAD"], f"{self.libasan}:{self.libubsan}")
        self.assertIn(str(self.san_dir / "asan"), env["ASAN_OPTIONS"])
        self.assertIn(str(self.san_dir / "ubsan"), env["UBSAN_OPTIONS"])

        self.assertEqual(set(res.summary), {"build", "env", "log_files", "worker"})
        self.assertEqual(res.summary["build"]["command"][-3:], ["-o", "kernel_san.so", "kernel.c"])
        self.assertEqual(res.summary["env"], env)
        self.assertEqual(res.summary["log_files"], ["asan.12345"])
        self.assertEqual(res.summary["worker"], {"rc": -6, "signal": "SIGABRT"})
        json.dumps(res.summary)  # result.json 에 그대로 들어간다

    def test_clean_run_is_zero_findings(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc={"complete": True, "load_ok": True, "classification": "pass"}))
        self.assertEqual((res.ran, res.build_ok, res.asan_findings, res.ubsan_findings, res.kinds, res.detail),
                         (True, True, 0, 0, [], ""))

    def test_wrong_output_without_sanitizer_reports_is_still_zero(self) -> None:
        doc = {"complete": True, "load_ok": True, "classification": "wrong_output"}
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc=doc))
        self.assertEqual((res.asan_findings, res.ubsan_findings), (0, 0))

    def test_ubsan_findings_of_a_complete_run(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc={"complete": True, "load_ok": True},
                                      logs=[FIXTURES / "ubsan_three_kinds" / "ubsan.12349"]))
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.detail), (0, 3, ""))
        self.assertEqual(res.kinds, ["signed-integer-overflow", "out-of-bounds-index", "misaligned-pointer-use"])

    def test_both_sanitizers_report_into_one_file(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(-6, signal_name="SIGABRT"), doc={"complete": False, "load_ok": True},
                                      logs=[FIXTURES / "single_runtime" / "ubsan.12352"]))
        self.assertEqual((res.asan_findings, res.ubsan_findings), (1, 2))
        self.assertEqual(res.kinds[0], "stack-buffer-underflow", "kinds 의 첫 항목은 ASan 의 종류다")

    def test_timeout_keeps_what_was_found_but_no_zero(self) -> None:
        cfg = SimpleNamespace(safety=SimpleNamespace(enabled=True, timeout_s=7), build=SimpleNamespace(cc="gcc"))
        timed_out = proc_result(-9, timed_out=True, signal_name="SIGKILL")
        res, _ = self.sanitize(self.worker(proc=timed_out, doc={"complete": False, "load_ok": True},
                                      logs=[FIXTURES / "ubsan_repeated" / "ubsan.12351"]), cfg=cfg)
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.detail), (None, 2, "incomplete:timeout after 7s"))
        self.assertEqual(self.worker_calls[0][0][3], 7.0)

    def test_runtime_that_never_started_is_inconclusive(self) -> None:
        stderr_tail = (FIXTURES / "stderr_preload_failed.txt").read_text(encoding="utf-8")
        res, _ = self.sanitize(self.worker(proc=proc_result(1, stderr=stderr_tail), doc={"complete": False, "load_ok": None}))
        self.assertEqual((res.ran, res.asan_findings, res.ubsan_findings, res.kinds), (True, None, None, []))
        self.assertTrue(res.detail.startswith("incomplete:rc=1: "), res.detail)
        self.assertIn("ASan runtime does not come first", res.detail)

    def test_kernel_that_did_not_load_is_not_a_clean_run(self) -> None:
        # clang 의 -shared 계측 라이브러리처럼 런타임 심볼이 선행 로드에서 오지 않으면 dlopen 이 실패한다: 워커는 load_ok=false 로
        # 문서를 끝까지 쓰고 rc 0 으로 끝나지만 계측된 커널은 한 번도 돌지 않았다 → 0 이 아니라 판정 불가
        doc = {"complete": True, "load_ok": False, "classification": "compile_error",
               "load_error": "kernel_san.so: undefined symbol: __asan_report_load4"}
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc=doc))
        self.assertEqual((res.ran, res.build_ok, res.asan_findings, res.ubsan_findings), (True, True, None, None))
        self.assertEqual(res.detail, "load_failed:kernel_san.so: undefined symbol: __asan_report_load4")
        doc = {"complete": True, "classification": "pass"}  # load_ok 를 모르는 문서도 끝까지 간 실행으로 치지 않는다
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc=doc))
        self.assertEqual((res.asan_findings, res.ubsan_findings), (None, None))
        self.assertTrue(res.detail.startswith("load_failed:"), res.detail)

    def test_full_stderr_file_is_parsed_not_only_the_tail(self) -> None:
        # UBSan 보고가 log_path 파일이 아니라 stderr 로 간 경우: 꼬리 2000 글자에는 뒤쪽 보고 몇 개만 남는다
        lines = [f"/work/sample_0/kernel.c:{n}:9: runtime error: signed integer overflow: 2147483647 + {n} cannot be "
                 "represented in type 'int'" for n in range(1, 31)]
        text = "\n".join(lines) + "\n"
        tail = text[-2000:]
        self.assertLess(len(safety.ubsan_errors(tail)), 30)
        res, _ = self.sanitize(self.worker(proc=proc_result(stderr=tail), doc={"complete": True, "load_ok": True},
                                           stderr_file=text))
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.detail), (0, 30, ""))
        # 워커가 stderr 파일을 쓰지 않았으면 이전 실행의 파일은 지워지고 꼬리만 본다
        res, _ = self.sanitize(self.worker(proc=proc_result(stderr=tail), doc={"complete": True, "load_ok": True}))
        self.assertEqual(res.ubsan_findings, len(safety.ubsan_errors(tail)))

    def test_crash_without_a_result_document(self) -> None:
        res, _ = self.sanitize(self.worker(proc=proc_result(-11, signal_name="SIGSEGV"), doc=None))
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.detail), (None, None, "incomplete:signal SIGSEGV"))

    def test_stale_logs_of_an_earlier_run_are_removed_first(self) -> None:
        self.san_dir.mkdir(parents=True)
        shutil.copy(FIXTURES / "segv" / "asan.12348", self.san_dir / "asan.999")
        shutil.copy(FIXTURES / "ubsan_three_kinds" / "ubsan.12349", self.san_dir / "ubsan.999")
        (self.san_dir / "build.log").write_text("kept", encoding="utf-8")
        res, _ = self.sanitize(self.worker(proc=proc_result(), doc={"complete": True, "load_ok": True}))
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.summary["log_files"]), (0, 0, []))
        self.assertTrue((self.san_dir / "build.log").exists())

    def test_flags_come_from_the_config_when_present(self) -> None:
        cfg = SimpleNamespace(safety=SimpleNamespace(enabled=True, timeout_s=120),
                              build=SimpleNamespace(cc="gcc", sanitize_flags=["-fsanitize=address", "-g3"]))
        self.sanitize(self.worker(proc=proc_result(), doc={"complete": True, "load_ok": True}), cfg=cfg)
        self.assertEqual(self.build_calls[0][1]["extra_flags"], ["-fsanitize=address", "-g3"])
        self.assertEqual(self.worker_calls[0][0][3], 120.0)


# ----------------------------------------------------------------------------- sanitizer: 실제 gcc·ASan (Linux 전용)


SAN_TOLERANCE_YAML = """task: gemm
dtype: float32
rtol: 1.0e-3
atol: 1.0e-3
reference_dtype: float64
numeric_pattern_tol_ratio: 20
sentinel: -2048.0
"""

GEMM_BODY = """
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
"""

C_CLEAN = "void gemm(const float* A, const float* B, float* C, int M, int N, int K) {" + GEMM_BODY + "}\n"

# 커널이 스스로 할당한 버퍼를 한 칸 넘겨 쓴다 → ASan heap-buffer-overflow (워커의 가드 영역과 무관하다).
# 인덱스를 volatile 로 읽고 그 뒤에 row 를 읽으므로 -O3 가 이 쓰기를 죽은 저장으로 지우지 못한다.
C_HEAP_OVERFLOW = """#include <stdlib.h>
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    volatile int past_end = N;
    float* row = malloc((size_t)N * sizeof(float));
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) row[j] = 0.0f;
        row[past_end] = 0.0f;
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++) row[j] += A[i * K + k] * B[k * N + j];
        for (int j = 0; j < N; j++) C[i * N + j] = row[j];
    }
    free(row);
}
"""

# 하네스가 준 출력 배열을 한 칸 넘겨 쓴다(C[M*N]) → 워커의 --sanitize 모드에서 ASan heap-buffer-overflow
C_OUTPUT_OVERFLOW = "void gemm(const float* A, const float* B, float* C, int M, int N, int K) {" + GEMM_BODY + \
    "    C[M * N] = 0.0f;\n}\n"

# 결과는 맞지만 호출마다 부호 있는 정수 오버플로가 난다 → UBSan 만, 워커는 끝까지 돈다
C_SIGNED_OVERFLOW = """
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    volatile int big = 2147483647;
    volatile int wrapped = big + M;
    (void)wrapped;
""" + GEMM_BODY + "}\n"


@requires_linux("gcc -fsanitize=address,undefined 와 libasan 선행 로드")
@requires_tool("gcc", "nm")
class RealSanitizerRunTest(TempDirTest):
    """``run_sanitizers`` 를 진짜 ``build_c`` (설정의 플래그 + sanitize_flags)와 진짜 ``harness.worker`` 로 끝까지 돌린다."""

    def setUp(self) -> None:
        super().setUp()
        rng = np.random.default_rng(1)
        M, N, K = 7, 5, 3
        A = rng.random((M, K), dtype=np.float32)
        B = rng.random((K, N), dtype=np.float32)
        self.test_npz = self.tmp / "tiny.npz"
        with open(self.test_npz, "wb") as fh:
            np.savez(fh, A=A, B=B, C_ref=A.astype(np.float64) @ B.astype(np.float64), M=np.int64(M), N=np.int64(N),
                     K=np.int64(K), seed=np.int64(1))
        self.tolerance = self.tmp / "tolerance.yaml"
        self.tolerance.write_text(SAN_TOLERANCE_YAML, encoding="utf-8")
        self.sample_dir = self.tmp / "sample_0"
        self.sample_dir.mkdir()
        self.cfg = SimpleNamespace(build=load_config().build, safety=SimpleNamespace(enabled=True, timeout_s=120))

    def worker(self, stage: str, mode: str, tests: list[Path], timeout_s: float, extra: list[str] | None = None, *,
               artifact: Path | None = None, env_extra: dict[str, str] | None = None, max_mem_gb: float | None = None) -> Any:
        """run.py 의 워커 클로저(설계 v3 §10.3)와 같은 시그니처로 ``harness.worker`` 를 띄운다."""
        out = self.sample_dir / f"{stage}.json"
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", "c", "--artifact", str(artifact), "--mode", mode,
               "--tests", ",".join(str(t) for t in tests), "--tolerance", str(self.tolerance),
               "--signatures", str(ROOT / "tasks" / "gemm" / "signatures.yaml"), "--out", str(out),
               "--max-mem-gb", str(max_mem_gb), *(extra or [])]
        proc = run_isolated(cmd, timeout_s=timeout_s, cwd=self.sample_dir, env={**launch_env_minimal(), **(env_extra or {})},
                            stderr_path=self.sample_dir / f"{stage}.stderr.log")  # run.run_worker 와 같은 자리
        doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return SimpleNamespace(proc=proc, doc=doc, summary=lambda: {"rc": proc.rc, "signal": proc.signal_name})

    def sanitize(self, code: str) -> safety.SanitizeResult:
        return safety.run_sanitizers(self.cfg, code=code, sample_dir=self.sample_dir, openmp=False, launch_prefix=[],
                                     symbol="gemm", args_header=None, tests=[self.test_npz], worker=self.worker)

    def test_clean_kernel_has_no_findings(self) -> None:
        res = self.sanitize(C_CLEAN)
        self.assertEqual((res.ran, res.build_ok, res.asan_findings, res.ubsan_findings, res.detail), (True, True, 0, 0, ""),
                         res.summary)

    def test_heap_overflow_in_a_kernel_buffer_is_an_asan_finding(self) -> None:
        res = self.sanitize(C_HEAP_OVERFLOW)
        self.assertEqual((res.asan_findings, res.kinds[:1]), (1, ["heap-buffer-overflow"]), res.summary)
        self.assertEqual(res.detail, "incomplete:signal SIGABRT")
        self.assertTrue(res.summary["log_files"], "보고는 san/ 의 로그 파일에 있어야 한다")

    def test_write_one_past_the_harness_output_array_is_an_asan_finding(self) -> None:
        # 워커의 --sanitize 모드: 출력 배열이 가드 영역 없이 정확한 크기라 C[M*N] 쓰기가 malloc 블록 밖이다
        res = self.sanitize(C_OUTPUT_OVERFLOW)
        self.assertGreaterEqual(res.asan_findings or 0, 1, res.summary)
        self.assertEqual(res.kinds[:1], ["heap-buffer-overflow"], res.summary)

    def test_signed_overflow_is_a_ubsan_finding_and_the_run_completes(self) -> None:
        res = self.sanitize(C_SIGNED_OVERFLOW)
        self.assertEqual((res.asan_findings, res.ubsan_findings, res.kinds, res.detail), (0, 1, ["signed-integer-overflow"], ""),
                         res.summary)


if __name__ == "__main__":
    unittest.main()
