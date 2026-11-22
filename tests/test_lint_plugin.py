"""린트 플러그인 베이스라인(plugins/lint/run_lint.py, rules.yaml) 테스트.

플러그인은 하네스와 독립된 CLI 프로그램이므로 이 모듈도 하네스 없이 돈다: 스크립트를 파일 경로로 로드해 순수 함수를 보고,
끝에서 끝까지는 CLI 를 서브프로세스로 띄운다. 하네스 쪽 연결(``harness.safety.run_lint`` → ``run_isolated`` → 플러그인)은
마지막 클래스 하나가 본다.

- cppcheck·clang-tidy 는 개발 머신에 없을 수 있다 → 출력 파서는 그 도구들의 출력 형식으로 쓴 고정 텍스트로,
  도구 조합(병합·실패 처리)은 ``run_tool`` 을 가짜로 바꿔 본다.
- ruff·pyflakes 는 설치돼 있으면 진짜로 돌린다(없으면 그 클래스만 건너뛴다).
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib.util
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
PLUGIN = ROOT / "plugins" / "lint" / "run_lint.py"
CATEGORIES = ("memory", "numeric", "concurrency", "shape_dtype", "api_misuse", "performance")
FINDING_KEYS = {"rule", "severity", "line", "message", "category"}


def load_plugin() -> Any:
    """플러그인 스크립트를 모듈로 로드한다(패키지가 아니므로 경로로)."""
    spec = importlib.util.spec_from_file_location("lint_plugin_under_test", PLUGIN)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass 가 자기 모듈을 찾을 수 있게
    spec.loader.exec_module(module)
    return module


plugin = load_plugin()
RULES = plugin.load_rules()
HAVE_PYTHON_LINTERS = all(importlib.util.find_spec(name) is not None for name in ("ruff", "pyflakes"))


def completed(stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


# ----------------------------------------------------------------------------- 독립성


class IndependenceTest(unittest.TestCase):
    def test_plugin_imports_only_the_standard_library_and_pyyaml(self) -> None:
        """다른 구현으로 바꿔 끼울 자리이므로 하네스(와 그 밖의 패키지)를 임포트하지 않는다."""
        import ast

        imported: set[str] = set()
        for node in ast.walk(ast.parse(PLUGIN.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "상대 임포트 금지")
                imported.add((node.module or "").split(".")[0])
        self.assertEqual(imported - set(sys.stdlib_module_names), {"yaml"})


# ----------------------------------------------------------------------------- 규칙 표


class RulesTableTest(unittest.TestCase):
    def tables(self) -> dict[str, dict[str, dict[str, str]]]:
        return {tool: RULES[tool]["rules"] for tool in ("ruff", "cppcheck", "clang-tidy")}

    def test_every_entry_has_a_category_a_severity_and_a_reason(self) -> None:
        for tool, table in self.tables().items():
            for key, entry in table.items():
                with self.subTest(tool=tool, rule=key):
                    self.assertIn(entry["category"], CATEGORIES)
                    self.assertIn(entry["severity"], plugin.SEVERITIES)
                    self.assertTrue(entry["reason"].strip())

    def test_each_language_family_covers_all_six_categories(self) -> None:
        tables = self.tables()
        python = {entry["category"] for entry in tables["ruff"].values()}
        c = {entry["category"] for tool in ("cppcheck", "clang-tidy") for entry in tables[tool].values()}
        self.assertEqual(python, set(CATEGORIES))
        self.assertEqual(c, set(CATEGORIES))

    def test_exact_entry_beats_prefix_and_longest_prefix_wins(self) -> None:
        table = {
            "a-*": {"category": "memory", "severity": "error", "reason": "r"},
            "a-b-*": {"category": "numeric", "severity": "warning", "reason": "r"},
            "a-b-c": {"category": "performance", "severity": "info", "reason": "r"},
        }
        self.assertEqual(plugin.find_rule(table, "a-b-c")["category"], "performance")
        self.assertEqual(plugin.find_rule(table, "a-b-d")["category"], "numeric")
        self.assertEqual(plugin.find_rule(table, "a-x")["category"], "memory")
        self.assertIsNone(plugin.find_rule(table, "b-a-b-c"))
        self.assertIsNone(plugin.find_rule(table, "a"))

    def test_lookups_in_the_real_tables(self) -> None:
        expected = {
            ("ruff", "F821"): ("memory", "error"),
            ("ruff", "F841"): ("performance", "info"),
            ("ruff", "NPY001"): ("shape_dtype", "error"),
            ("ruff", "NPY002"): ("api_misuse", "warning"),
            ("ruff", "PERF401"): ("performance", "info"),
            ("ruff", "PLW0177"): ("numeric", "warning"),
            ("ruff", "PLW2101"): ("concurrency", "warning"),
            ("cppcheck", "arrayIndexOutOfBounds"): ("memory", "error"),
            ("cppcheck", "arrayIndexOutOfBoundsCond"): ("memory", "warning"),
            ("cppcheck", "bufferAccessOutOfBounds"): ("memory", "error"),
            ("cppcheck", "uninitvar"): ("memory", "error"),
            ("cppcheck", "uninitdata"): ("memory", "error"),
            ("cppcheck", "memleak"): ("memory", "error"),
            ("cppcheck", "nullPointer"): ("memory", "error"),
            ("cppcheck", "nullPointerOutOfMemory"): ("memory", "warning"),
            ("cppcheck", "invalidPointerCast"): ("memory", "warning"),
            ("cppcheck", "integerOverflow"): ("numeric", "error"),
            ("cppcheck", "zerodiv"): ("numeric", "error"),
            ("cppcheck", "zerodivcond"): ("numeric", "warning"),
            ("cppcheck", "floatConversionOverflow"): ("numeric", "error"),
            ("cppcheck", "shiftTooManyBits"): ("numeric", "error"),
            ("cppcheck", "shiftTooManyBitsSigned"): ("numeric", "error"),
            ("cppcheck", "signConversion"): ("shape_dtype", "warning"),
            ("clang-tidy", "clang-analyzer-core.NullDereference"): ("memory", "error"),
            ("clang-tidy", "clang-analyzer-core.uninitialized.Assign"): ("memory", "error"),
            ("clang-tidy", "clang-analyzer-core.DivideZero"): ("numeric", "error"),
            ("clang-tidy", "clang-analyzer-unix.Malloc"): ("memory", "error"),
            ("clang-tidy", "clang-analyzer-unix.MallocSizeof"): ("memory", "error"),
            ("clang-tidy", "clang-analyzer-deadcode.DeadStores"): ("performance", "info"),
            ("clang-tidy", "bugprone-integer-division"): ("numeric", "warning"),
            ("clang-tidy", "bugprone-narrowing-conversions"): ("shape_dtype", "info"),
            ("clang-tidy", "bugprone-sizeof-expression"): ("memory", "warning"),
            ("clang-tidy", "cert-flp30-c"): ("numeric", "warning"),
            ("clang-tidy", "openmp-use-default-none"): ("concurrency", "info"),
            ("clang-tidy", "performance-no-int-to-ptr"): ("performance", "info"),
            ("clang-tidy", "performance-type-promotion-in-math-fn"): ("shape_dtype", "info"),
        }
        for (tool, rule), (category, severity) in expected.items():
            with self.subTest(tool=tool, rule=rule):
                entry = plugin.find_rule(RULES[tool]["rules"], rule)
                self.assertIsNotNone(entry)
                self.assertEqual((entry["category"], entry["severity"]), (category, severity))

    def test_style_and_hygiene_rules_are_not_mapped(self) -> None:
        unmapped = {
            "ruff": ["F401", "F811", "B007", "B015", "E501", "E741", "W291", "N806", "D100", "I001", "UP006", "NPY003"],
            "cppcheck": ["variableScope", "constParameterPointer", "unusedVariable", "unreadVariable", "shadowVariable",
                         "missingIncludeSystem", "checkersReport", "syntaxError"],
            "clang-tidy": ["bugprone-easily-swappable-parameters", "bugprone-reserved-identifier", "bugprone-branch-clone",
                           "clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling",
                           "clang-analyzer-security.FloatLoopCounter", "readability-magic-numbers", "clang-diagnostic-error"],
        }
        for tool, rules in unmapped.items():
            for rule in rules:
                with self.subTest(tool=tool, rule=rule):
                    self.assertIsNone(plugin.find_rule(RULES[tool]["rules"], rule))

    def test_load_rules_rejects_malformed_tables(self) -> None:
        import copy

        import yaml

        def broken(edit: Any) -> dict[str, Any]:
            doc = copy.deepcopy(RULES)
            edit(doc)
            return doc

        cases = {
            "모르는 범주": broken(lambda d: d["ruff"]["rules"]["F821"].update(category="style")),
            "모르는 심각도": broken(lambda d: d["cppcheck"]["rules"]["zerodiv"].update(severity="fatal")),
            "근거 없음": broken(lambda d: d["clang-tidy"]["rules"]["cert-flp30-c"].pop("reason")),
            "빈 근거": broken(lambda d: d["ruff"]["rules"]["F821"].update(reason="  ")),
            "도구 섹션 없음": broken(lambda d: d.pop("cppcheck")),
            "모르는 도구 섹션": broken(lambda d: d.update(mypy={"rules": {}})),
            "select 가 문자열": broken(lambda d: d["ruff"].update(select="F,E9")),
            "빈 규칙 표": broken(lambda d: d["cppcheck"].update(rules={})),
            "pyflakes 코드가 ruff 표에 없음": broken(lambda d: d["pyflakes"]["messages"].update(F401="imported but unused")),
            "pyflakes 정규식 오류": broken(lambda d: d["pyflakes"]["messages"].update(F821="undefined name (")),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rules.yaml"
            for name, doc in cases.items():
                path.write_text(yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8")
                with self.subTest(name), self.assertRaises(ValueError):
                    plugin.load_rules(path)
            path.write_text(yaml.safe_dump(RULES, allow_unicode=True), encoding="utf-8")
            self.assertEqual(plugin.load_rules(path), RULES)


# ----------------------------------------------------------------------------- 명령 조립


class CommandTest(unittest.TestCase):
    FILE = Path("/work/sample_0/kernel.py")

    def test_ruff_command(self) -> None:
        cmd = plugin.ruff_command(self.FILE, RULES["ruff"])
        self.assertEqual(cmd[:5], [sys.executable, "-P", "-m", "ruff", "check"])
        self.assertEqual(cmd[-1], str(self.FILE))
        for flag in ("--isolated", "--no-cache", "--ignore-noqa"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
        self.assertEqual(cmd[cmd.index("--target-version") + 1], f"py{sys.version_info.major}{sys.version_info.minor}")
        select = cmd[cmd.index("--select") + 1].split(",")
        self.assertEqual(select, sorted(set(select)))
        for code in ("F", "E9", "B", "PERF", "NPY001", "NPY002", "NPY201", "PLW0177", "PLW0603", "PLW2101", "PLE0643", "RUF017"):
            self.assertIn(code, select)
        self.assertNotIn("NPY", select, "NPY 를 통째로 켜면 NPY003 과 NPY201 이 같은 줄을 두 번 센다")
        self.assertFalse([code for code in select if "*" in code])

    def test_cppcheck_command(self) -> None:
        cmd = plugin.cppcheck_command(Path("/work/sample_0/kernel.c"), RULES["cppcheck"])
        self.assertEqual(cmd[0], "cppcheck")
        self.assertEqual(cmd[-1], str(Path("/work/sample_0/kernel.c")))
        self.assertIn("--enable=warning,portability", cmd)
        self.assertIn("--xml", cmd)
        self.assertIn("--language=c", cmd)

    def test_clang_tidy_command(self) -> None:
        file = Path("/work/sample_0/kernel.c")
        cmd = plugin.clang_tidy_command(file, RULES["clang-tidy"], openmp=False)
        self.assertEqual(cmd[:2], ["clang-tidy", str(file)])
        self.assertIn("--config={}", cmd)
        dashes = cmd.index("--")
        self.assertEqual(cmd[dashes + 1:], ["-std=gnu11", "-march=native"])
        checks = next(arg for arg in cmd[:dashes] if arg.startswith("--checks=")).removeprefix("--checks=").split(",")
        self.assertEqual(checks[0], "-*", "먼저 전부 끄고 고른 것만 켠다")
        for glob in ("clang-analyzer-*", "bugprone-*", "openmp-*", "performance-*", "cert-flp30-c", "bugprone-integer-division"):
            self.assertIn(glob, checks)
        disabled = [check for check in checks[1:] if check.startswith("-")]
        self.assertEqual(disabled, ["-bugprone-easily-swappable-parameters", "-clang-analyzer-security.insecureAPI.*"])
        self.assertEqual(checks[-len(disabled):], disabled, "끄는 항목이 마지막이어야 앞의 glob 을 덮는다")

    def test_clang_tidy_gets_fopenmp_after_the_double_dash_for_openmp(self) -> None:
        cmd = plugin.clang_tidy_command(Path("kernel.c"), RULES["clang-tidy"], openmp=True)
        self.assertEqual(cmd[cmd.index("--") + 1:], ["-std=gnu11", "-march=native", "-fopenmp"])


# ----------------------------------------------------------------------------- 출력 파서 (고정 텍스트)


RUFF_JSON = json.dumps([
    {"cell": None, "code": "F821", "filename": "/work/kernel.py", "location": {"column": 12, "row": 18},
     "end_location": {"column": 27, "row": 18}, "message": "Undefined name `acc`", "fix": None, "noqa_row": 18,
     "severity": "error", "url": "https://docs.astral.sh/ruff/rules/undefined-name"},
    {"cell": None, "code": "PERF401", "filename": "/work/kernel.py", "location": {"column": 9, "row": 15},
     "end_location": {"column": 26, "row": 15}, "message": "Use a list comprehension to create a transformed list", "fix": None},
])

CPPCHECK_XML = """<?xml version="1.0" encoding="UTF-8"?>
<results version="2">
    <cppcheck version="2.17.1"/>
    <errors>
        <error id="nullPointerOutOfMemory" severity="warning" msg="If memory allocation fails, then there is a possible null pointer dereference: tmp" verbose="If memory allocation fails, then there is a possible null pointer dereference: tmp" cwe="476" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel.c" line="9" column="9" info="Null pointer dereference"/>
            <location file="/work/sample_0/kernel.c" line="7" column="24" info="Assuming allocation function fails"/>
            <location file="/work/sample_0/kernel.c" line="7" column="18" info="Assignment &apos;tmp=malloc((size_t)M*N*sizeof(float))&apos;, assigned value is 0"/>
            <symbol>tmp</symbol>
        </error>
        <error id="arrayIndexOutOfBounds" severity="error" msg="Array &apos;acc[8]&apos; accessed at index 8, which is out of bounds." verbose="Array &apos;acc[8]&apos; accessed at index 8, which is out of bounds." cwe="788" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel.c" line="14" column="12" info="Array index out of bounds"/>
            <symbol>acc</symbol>
        </error>
        <error id="uninitvar" severity="error" msg="Uninitialized variable: sum" verbose="Uninitialized variable: sum" cwe="457" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel.c" line="18" column="13"/>
            <symbol>sum</symbol>
        </error>
        <error id="invalidPointerCast" severity="portability" msg="Casting between float * and double * which have an incompatible binary data representation." verbose="Casting between float * and double * which have an incompatible binary data representation." cwe="704" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel.c" line="20" column="24"/>
        </error>
        <error id="memleak" severity="error" msg="Memory leak: tmp" verbose="Memory leak: tmp" cwe="401" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel.c" line="23" column="1"/>
            <symbol>tmp</symbol>
        </error>
        <error id="uninitStructMember" severity="error" msg="Uninitialized struct member: args.K" verbose="Uninitialized struct member: args.K" cwe="908" file0="/work/sample_0/kernel.c">
            <location file="/work/sample_0/kernel_args.h" line="8" column="9"/>
        </error>
        <error id="toomanyconfigs" severity="information" msg="Too many #ifdef configurations - cppcheck only checks 12 configurations. Use --force to check all configurations." verbose="Too many #ifdef configurations" cwe="398"/>
    </errors>
</results>
"""

CLANG_TIDY_OUT = """/work/sample_0/kernel.c:9:9: warning: Array access (from variable 'tmp') results in a null pointer dereference [clang-analyzer-core.NullDereference]
    9 |         tmp[i] = 0.0f;
      |         ^~~
/work/sample_0/kernel.c:7:18: note: Assuming pointer value is null
    7 |     float* tmp = malloc((size_t)M * N * sizeof(float));
      |                  ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
/work/sample_0/kernel.c:9:9: note: Array access (from variable 'tmp') results in a null pointer dereference
    9 |         tmp[i] = 0.0f;
      |         ^~~
/work/sample_0/kernel.c:3:6: warning: declaration uses identifier '_gemm_block', which is reserved in the global namespace [bugprone-reserved-identifier]
    3 | void _gemm_block(const float* A, const float* B, float* C, int n);
      |      ^~~~~~~~~~~
/work/sample_0/kernel.c:12:23: warning: result of integer division used in a floating point context; possible loss of precision [bugprone-integer-division]
   12 |     float scale = 1 / N;   /* warning: this text is part of the source line, not a diagnostic */
      |                   ~~^~~
/work/sample_0/kernel.c:15:5: warning: OpenMP directive 'parallel for' does not specify 'default' clause, consider specifying 'default(none)' clause [openmp-use-default-none]
   15 |     #pragma omp parallel for
      |     ^
/work/sample_0/kernel.c:16:10: warning: loop induction expression should not have floating-point type [bugprone-float-loop-counter,cert-flp30-c]
   16 |     for (float t = 0.0f; t < 1.0f; t += 0.1f) {
      |          ^
/work/sample_0/kernel.c:19:26: warning: call to 'sqrt' promotes float to double [performance-type-promotion-in-math-fn]
   19 |         C[i] = (float)sqrt(sum);   /* fprintf(stderr, "kernel.c:1:1: error: not a diagnostic either"); */
      |                       ^~~~
      |                       sqrtf
/work/sample_0/kernel.c:21:16: warning: narrowing conversion from 'double' to 'float' [bugprone-narrowing-conversions]
   21 |     float half = total * 0.5;
      |                  ^
/work/sample_0/kernel_args.h:4:5: warning: suspicious usage of 'sizeof(K)'; did you mean 'K'? [bugprone-sizeof-expression]
    4 |     int M;
      |     ^
"""

CLANG_TIDY_COMPILE_ERROR = """/work/sample_0/kernel.c:2:10: error: 'omp.h' file not found [clang-diagnostic-error]
    2 | #include <omp.h>
      |          ^~~~~~~
"""


class ParserTest(unittest.TestCase):
    C_FILE = Path("/work/sample_0/kernel.c")

    def test_parse_ruff(self) -> None:
        self.assertEqual(plugin.parse_ruff(RUFF_JSON), [
            plugin.Report("F821", 18, "Undefined name `acc`"),
            plugin.Report("PERF401", 15, "Use a list comprehension to create a transformed list"),
        ])
        self.assertEqual(plugin.parse_ruff("[]"), [])

    def test_ruff_syntax_error_is_an_analysis_failure(self) -> None:
        item = {"code": "invalid-syntax", "location": {"row": 5, "column": 22}, "message": "Expected `:`, found newline"}
        for doc in ([item], [{**item, "code": None}], [json.loads(RUFF_JSON)[0], item]):
            with self.subTest(code=doc[-1]["code"]), self.assertRaisesRegex(plugin.ToolFailure, "line 5"):
                plugin.parse_ruff(json.dumps(doc))

    def test_ruff_output_that_is_not_the_json_list(self) -> None:
        for text in ("", "error: invalid value 'XYZ' for '--select'", '{"code": "F821"}'):
            with self.subTest(text), self.assertRaises(plugin.ToolFailure):
                plugin.parse_ruff(text)

    def test_parse_pyflakes(self) -> None:
        messages = RULES["pyflakes"]["messages"]
        for file in (Path("/work/sample_0/kernel.py"), Path(r"C:\work\sample 0\kernel.py")):
            stdout = (f"{file}:1:1: 'os' imported but unused\n"
                      f"{file}:17:5: local variable 'x' is assigned to but never used\n"
                      f"{file}:18:12: undefined name 'undefined_thing'\n"
                      f"{file}:20:8: use ==/!= to compare constant literals (str, bytes, int, float, tuple)\n"
                      f"{file}:24:9: local variable 'total' defined in enclosing scope on line 3 referenced before assignment\n"
                      "some/other/file.py:3:1: undefined name 'y'\n")
            with self.subTest(file=str(file)):
                self.assertEqual(plugin.parse_pyflakes(stdout, "", file, messages), [
                    plugin.Report(plugin.PYFLAKES_UNCLASSIFIED, 1, "'os' imported but unused"),
                    plugin.Report("F841", 17, "local variable 'x' is assigned to but never used"),
                    plugin.Report("F821", 18, "undefined name 'undefined_thing'"),
                    plugin.Report("F632", 20, "use ==/!= to compare constant literals (str, bytes, int, float, tuple)"),
                    plugin.Report("F823", 24, "local variable 'total' defined in enclosing scope on line 3 referenced before assignment"),
                ])

    def test_pyflakes_syntax_error_is_an_analysis_failure(self) -> None:
        file = Path("/work/sample_0/kernel.py")
        stderr = f"{file}:5:22: expected ':'\n    for i in range(M)\n                     ^\n"
        with self.assertRaisesRegex(plugin.ToolFailure, "5:22"):
            plugin.parse_pyflakes("", stderr, file, RULES["pyflakes"]["messages"])
        # 파일과 무관한 stderr(인터프리터 경고 등)는 실패가 아니다
        self.assertEqual(plugin.parse_pyflakes("", "DeprecationWarning: something\n", file, RULES["pyflakes"]["messages"]), [])

    def test_parse_cppcheck(self) -> None:
        reports, version = plugin.parse_cppcheck(CPPCHECK_XML, self.C_FILE)
        self.assertEqual(version, "2.17.1")
        self.assertEqual([(r.rule, r.line) for r in reports], [
            ("nullPointerOutOfMemory", 9),  # 첫 location 이 오류 지점이다
            ("arrayIndexOutOfBounds", 14),
            ("uninitvar", 18),
            ("invalidPointerCast", 20),
            ("memleak", 23),
            ("toomanyconfigs", 0),  # location 없는 보고; 헤더(kernel_args.h) 안의 보고는 빠졌다
        ])
        self.assertEqual(reports[1].message, "Array 'acc[8]' accessed at index 8, which is out of bounds.")

    def test_cppcheck_xml_after_other_stderr_text(self) -> None:
        reports, _ = plugin.parse_cppcheck("cppcheck: some notice\n" + CPPCHECK_XML, self.C_FILE)
        self.assertEqual(len(reports), 6)

    def test_cppcheck_analysis_failures(self) -> None:
        syntax = CPPCHECK_XML.replace('id="uninitvar"', 'id="syntaxError"')
        macro = CPPCHECK_XML.replace('id="memleak"', 'id="unknownMacro"')
        for name, text in (("syntaxError", syntax), ("unknownMacro", macro), ("XML 아님", "cppcheck: error: could not find or open any of the paths given."),
                           ("잘린 XML", CPPCHECK_XML[:400])):
            with self.subTest(name), self.assertRaises(plugin.ToolFailure):
                plugin.parse_cppcheck(text, self.C_FILE)

    def test_parse_clang_tidy(self) -> None:
        reports = plugin.parse_clang_tidy(CLANG_TIDY_OUT, self.C_FILE, RULES["clang-tidy"]["rules"])
        self.assertEqual([(r.rule, r.line) for r in reports], [
            ("clang-analyzer-core.NullDereference", 9),
            ("bugprone-reserved-identifier", 3),
            ("bugprone-integer-division", 12),
            ("openmp-use-default-none", 15),
            ("bugprone-float-loop-counter", 16),  # 별칭 둘 중 표에 있는 첫 이름
            ("performance-type-promotion-in-math-fn", 19),
            ("bugprone-narrowing-conversions", 21),
        ])
        self.assertEqual(reports[2].message, "result of integer division used in a floating point context; possible loss of precision")

    def test_clang_tidy_alias_falls_back_to_the_mapped_name(self) -> None:
        line = "/w/kernel.c:16:10: warning: loop induction expression should not have floating-point type [cppcoreguidelines-x,cert-flp30-c]\n"
        self.assertEqual(plugin.parse_clang_tidy(line, Path("/w/kernel.c"), RULES["clang-tidy"]["rules"])[0].rule, "cert-flp30-c")
        line = "/w/kernel.c:1:1: warning: something [misc-a,misc-b]\n"
        self.assertEqual(plugin.parse_clang_tidy(line, Path("/w/kernel.c"), RULES["clang-tidy"]["rules"])[0].rule, "misc-a")

    def test_clang_tidy_compile_error_is_an_analysis_failure(self) -> None:
        for text in (CLANG_TIDY_COMPILE_ERROR, CLANG_TIDY_OUT + CLANG_TIDY_COMPILE_ERROR,
                     "error: unknown argument: '-march=native' [clang-diagnostic-error]\n",
                     "/work/sample_0/kernel.c:4:1: fatal error: too many errors emitted, stopping now [clang-diagnostic-error]\n"):
            with self.subTest(text[:40]), self.assertRaisesRegex(plugin.ToolFailure, "error"):
                plugin.parse_clang_tidy(text, self.C_FILE, RULES["clang-tidy"]["rules"])

    def test_clang_tidy_windows_paths(self) -> None:
        line = r"C:\work\sample 0\kernel.c:12:23: warning: result of integer division used in a floating point context; possible loss of precision [bugprone-integer-division]" + "\n"
        reports = plugin.parse_clang_tidy(line, Path(r"C:\work\sample 0\kernel.c"), RULES["clang-tidy"]["rules"])
        self.assertEqual([(r.rule, r.line) for r in reports], [("bugprone-integer-division", 12)])


# ----------------------------------------------------------------------------- 병합


def f(line: int, category: str | None, rule: str = "t/r") -> dict[str, Any]:
    return {"rule": rule, "severity": "error" if category else None, "line": line, "message": "m", "category": category}


class MergeTest(unittest.TestCase):
    def test_same_line_and_category_from_two_tools_is_one_finding(self) -> None:
        merged = plugin.merge_across_tools([[f(5, "memory", "ruff/F821")], [f(5, "memory", "pyflakes/F821")]])
        self.assertEqual(merged, [f(5, "memory", "ruff/F821")], "먼저 돈 도구의 보고가 남는다")

    def test_different_line_or_category_is_kept(self) -> None:
        merged = plugin.merge_across_tools([[f(5, "memory", "a/1")], [f(6, "memory", "b/1"), f(5, "numeric", "b/2")]])
        self.assertEqual([m["rule"] for m in merged], ["a/1", "b/1", "b/2"])

    def test_one_tool_reporting_twice_on_a_line_is_not_merged_with_itself(self) -> None:
        merged = plugin.merge_across_tools([[f(3, "memory", "a/1"), f(3, "memory", "a/2")]])
        self.assertEqual(len(merged), 2)

    def test_counts_per_line_are_the_multiset_maximum(self) -> None:
        two, three = [f(3, "memory", "a/1")] * 2, [f(3, "memory", "b/1"), f(3, "memory", "b/2"), f(3, "memory", "b/3")]
        self.assertEqual([m["rule"] for m in plugin.merge_across_tools([two, two])], ["a/1", "a/1"])
        self.assertEqual([m["rule"] for m in plugin.merge_across_tools([two, three])], ["a/1", "a/1", "b/3"])
        self.assertEqual([m["rule"] for m in plugin.merge_across_tools([three, two])], ["b/1", "b/2", "b/3"])

    def test_unmapped_reports_are_merged_the_same_way(self) -> None:
        merged = plugin.merge_across_tools([[f(1, None, "ruff/F401")], [f(1, None, "pyflakes/unclassified"), f(1, "memory", "pyflakes/F821")]])
        self.assertEqual([m["rule"] for m in merged], ["ruff/F401", "pyflakes/F821"])

    def test_no_tools(self) -> None:
        self.assertEqual(plugin.merge_across_tools([]), [])


# ----------------------------------------------------------------------------- 도구 조합 (가짜 run_tool)


class LintWithFakeToolsTest(unittest.TestCase):
    """cppcheck·clang-tidy 가 설치된 것처럼 꾸며 ``lint`` 의 조합 규칙을 본다."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="lint_fake_")
        self.addCleanup(tmp.cleanup)
        self.file = Path(tmp.name) / "kernel.c"
        self.file.write_text("void gemm(void) {}\n", encoding="utf-8")
        self.commands: list[list[str]] = []
        self.outputs: dict[str, Any] = {
            "cppcheck": completed(stderr=CPPCHECK_XML),
            "clang-tidy": completed(stdout=CLANG_TIDY_OUT),
        }
        self.installed = {"cppcheck", "clang-tidy"}

    def fake_run_tool(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(cmd)
        if cmd[:2] == ["clang-tidy", "--version"]:
            return completed(stdout="Ubuntu LLVM version 18.1.3\n  Optimized build.\n")
        output = self.outputs[cmd[0]]
        if isinstance(output, Exception):
            raise output
        return output

    def lint(self, language: str) -> tuple[dict[str, Any], str]:
        stderr = io.StringIO()
        with mock.patch.object(plugin, "run_tool", side_effect=self.fake_run_tool), \
                mock.patch.object(plugin.shutil, "which", side_effect=lambda name: f"/usr/bin/{name}" if name in self.installed else None), \
                contextlib.redirect_stderr(stderr):
            doc = plugin.lint(language, self.file, RULES)
        return doc, stderr.getvalue()

    def test_both_c_tools(self) -> None:
        doc, stderr = self.lint("c_openmp")
        self.assertEqual(doc["tool"], "baseline:cppcheck+clang-tidy")
        self.assertEqual(doc["versions"], {"cppcheck": "2.17.1", "clang-tidy": "18.1.3"})
        self.assertEqual([(x["rule"], x["severity"], x["line"], x["category"]) for x in doc["findings"]], [
            ("cppcheck/arrayIndexOutOfBounds", "error", 14, "memory"),
            ("cppcheck/uninitvar", "error", 18, "memory"),
            ("cppcheck/memleak", "error", 23, "memory"),
            ("cppcheck/nullPointerOutOfMemory", "warning", 9, "memory"),  # clang-tidy 의 9행 memory 보고와 합쳐졌다
            ("clang-tidy/bugprone-integer-division", "warning", 12, "numeric"),
            ("clang-tidy/bugprone-float-loop-counter", "warning", 16, "numeric"),
            ("cppcheck/invalidPointerCast", "warning", 20, "memory"),
            ("clang-tidy/openmp-use-default-none", "info", 15, "concurrency"),
            ("clang-tidy/performance-type-promotion-in-math-fn", "info", 19, "shape_dtype"),
            ("clang-tidy/bugprone-narrowing-conversions", "info", 21, "shape_dtype"),
        ])
        self.assertEqual(doc["summary"], {"memory": 5, "numeric": 2, "concurrency": 1, "shape_dtype": 2, "api_misuse": 0, "performance": 0})
        self.assertEqual(doc["unmapped"], 2)
        self.assertEqual(len(stderr.splitlines()), 1)
        self.assertIn("clang-tidy/bugprone-reserved-identifier x1", stderr)
        self.assertIn("cppcheck/toomanyconfigs x1", stderr)
        clang = next(cmd for cmd in self.commands if cmd[0] == "clang-tidy" and "--version" not in cmd)
        self.assertEqual(clang[-1], "-fopenmp")
        self.assertEqual(clang[1], str(self.file))

    def test_c_seq_does_not_get_fopenmp(self) -> None:
        self.lint("c_seq")
        clang = next(cmd for cmd in self.commands if cmd[0] == "clang-tidy" and "--version" not in cmd)
        self.assertNotIn("-fopenmp", clang)

    def test_tools_that_are_not_installed_are_skipped_silently(self) -> None:
        self.installed = {"cppcheck"}
        doc, stderr = self.lint("c_seq")
        self.assertEqual((doc["tool"], doc["versions"]), ("baseline:cppcheck", {"cppcheck": "2.17.1"}))
        self.assertEqual(doc["summary"]["memory"], 5)
        self.assertEqual([doc["summary"][c] for c in ("concurrency", "api_misuse", "performance")], [None, None, None],
                         "clang-tidy 만 보는 범주는 clang-tidy 가 없으면 검사하지 않은 것이다")
        self.assertNotIn("clang-tidy", stderr)
        self.assertEqual([cmd[0] for cmd in self.commands], ["cppcheck"])

    def test_no_tool_installed(self) -> None:
        self.installed = set()
        doc, stderr = self.lint("c_openmp")
        self.assertEqual(doc, {"tool": "none", "findings": [], "summary": dict.fromkeys(CATEGORIES), "unmapped": 0, "versions": {}})
        self.assertEqual((stderr, self.commands), ("", []))

    def test_compile_error_in_clang_tidy_drops_only_that_tool(self) -> None:
        self.outputs["clang-tidy"] = completed(stdout=CLANG_TIDY_COMPILE_ERROR, returncode=1)
        doc, stderr = self.lint("c_openmp")
        self.assertEqual(doc["tool"], "baseline:cppcheck")
        self.assertEqual({x["rule"].split("/")[0] for x in doc["findings"]}, {"cppcheck"})
        self.assertIn("[lint] clang-tidy:", stderr)
        self.assertIn("omp.h", stderr)
        # cppcheck 가 보는 범주(memory·numeric·shape_dtype)만 수이고, clang-tidy 만 보는 범주는 0 이 아니라 null 이다
        self.assertEqual(doc["summary"], {"memory": 5, "numeric": 0, "concurrency": None, "shape_dtype": 0,
                                          "api_misuse": None, "performance": None})

    def test_tool_categories_follow_the_rules_table(self) -> None:
        self.assertEqual(plugin.tool_categories("ruff", RULES), set(CATEGORIES))
        self.assertEqual(plugin.tool_categories("clang-tidy", RULES), set(CATEGORIES))
        self.assertEqual(plugin.tool_categories("cppcheck", RULES), {"memory", "numeric", "shape_dtype"})
        self.assertEqual(plugin.tool_categories("pyflakes", RULES), {"memory", "numeric", "performance"})

    def openmp_lint(self, language: str, *, header: bool = True, printed: str | None = None) -> list[str]:
        """gcc 가 설치돼 있고 ``-print-file-name=include`` 가 ``include`` 디렉터리를 가리키는 것처럼 꾸며 돌린다.
        clang-tidy 명령을 돌려주고, 그 명령이 불린 순간의 ``-isystem`` 디렉터리 내용은 ``self.isystem_listing`` 에 남는다."""
        include = self.file.parent / "gcc_include"
        include.mkdir(exist_ok=True)
        (include / "immintrin.h").write_text("/* clang 이 해석하지 못하는 헤더 */\n", encoding="utf-8")
        (include / "omp.h").unlink(missing_ok=True)
        if header:
            (include / "omp.h").write_text("int omp_get_max_threads(void);\n", encoding="utf-8")
        self.installed.add("gcc-15")
        self.outputs["gcc-15"] = completed(stdout=f"{include if printed is None else printed}\n")
        self.isystem_listing: dict[str, str] | None = None
        original = self.fake_run_tool

        def fake(cmd: list[str]) -> subprocess.CompletedProcess[str]:
            if cmd[0] == "clang-tidy" and "-isystem" in cmd:
                directory = Path(cmd[cmd.index("-isystem") + 1])
                self.isystem_listing = {p.name: p.read_text(encoding="utf-8") for p in directory.iterdir()}
            return original(cmd)

        self.fake_run_tool = fake  # type: ignore[method-assign]
        with mock.patch.dict(plugin.os.environ, {"CC": "gcc-15"}):
            self.lint(language)
        return next(cmd for cmd in self.commands if cmd[0] == "clang-tidy" and "--version" not in cmd)

    def test_openmp_gets_a_private_directory_with_only_the_compilers_omp_h(self) -> None:
        clang = self.openmp_lint("c_openmp")
        self.assertIn(["gcc-15", "-print-file-name=include"], self.commands)
        i = clang.index("-isystem")
        self.assertGreater(i, clang.index("--"), "-isystem 은 `--` 뒤의 컴파일 인자다")
        self.assertEqual(clang[-1], "-fopenmp")
        self.assertEqual(self.isystem_listing, {"omp.h": "int omp_get_max_threads(void);\n"},
                         "인트린식 헤더가 든 컴파일러 include 디렉터리를 통째로 넣지 않는다")
        self.assertFalse(Path(clang[i + 1]).exists(), "임시 디렉터리는 clang-tidy 가 끝나면 지운다")

    def test_no_isystem_without_openmp_or_without_an_omp_h(self) -> None:
        cases = {"c_seq": {"language": "c_seq"}, "omp.h 없음": {"language": "c_openmp", "header": False},
                 "상대 경로 출력": {"language": "c_openmp", "printed": "include"}}
        for name, kwargs in cases.items():
            with self.subTest(name):
                self.commands = []
                self.assertNotIn("-isystem", self.openmp_lint(**kwargs))
        self.installed.discard("gcc-15")
        self.commands = []
        self.lint("c_openmp")
        self.assertFalse([cmd for cmd in self.commands if cmd[0].startswith("gcc")], "컴파일러가 없으면 묻지 않는다")

    def test_failing_tools(self) -> None:
        failures = {
            "시간 초과": plugin.ToolFailure("20초 안에 끝나지 않았다"),
            "비정상 종료": completed(stdout="cppcheck: error: unrecognized command line option", returncode=1),
            "XML 아님": completed(stderr="Segmentation fault"),
        }
        for name, output in failures.items():
            with self.subTest(name):
                self.outputs["cppcheck"] = output
                doc, stderr = self.lint("c_seq")
                self.assertEqual(doc["tool"], "baseline:clang-tidy")
                self.assertIn("[lint] cppcheck:", stderr)
                self.assertEqual(doc["summary"]["memory"], 1, "cppcheck 가 빠지면 clang-tidy 의 9행 보고가 남는다")

    def test_every_tool_failing_is_tool_none(self) -> None:
        self.outputs = {"cppcheck": plugin.ToolFailure("x"), "clang-tidy": completed(stdout=CLANG_TIDY_COMPILE_ERROR, returncode=1)}
        doc, _ = self.lint("c_seq")
        self.assertEqual((doc["tool"], doc["findings"], doc["unmapped"]), ("none", [], 0))

    def test_rust_and_unknown_languages_run_nothing(self) -> None:
        doc, stderr = self.lint("rust")
        self.assertEqual((doc["tool"], stderr, self.commands), ("none", "", []))
        doc, stderr = self.lint("fortran")
        self.assertEqual((doc["tool"], self.commands), ("none", []))
        self.assertIn("fortran", stderr)

    def test_run_tool_turns_a_missing_executable_and_a_timeout_into_tool_failure(self) -> None:
        with self.assertRaises(plugin.ToolFailure):
            plugin.run_tool(["this-linter-does-not-exist-xyz", "--version"])
        with mock.patch.object(plugin, "TOOL_TIMEOUT_S", 0.5), self.assertRaises(plugin.ToolFailure):
            plugin.run_tool([sys.executable, "-c", "import time; time.sleep(30)"])


# ----------------------------------------------------------------------------- 끝에서 끝까지 (진짜 ruff·pyflakes)


SOURCES = {
    "unused_import": "import os\nimport numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    np.matmul(A, B, out=C)\n",
    "undefined_name": "import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    np.matmul(A, B, out=C)\n    return acc\n",
    "numpy_specific": ("import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n"
                       "    tmp = np.zeros((M, N), dtype=np.float)\n    np.random.seed(0)\n    np.matmul(A, B, out=tmp)\n    C[:] = tmp\n"),
    "perf": ("def dot(x, y, out, n):\n    products = []\n    for i in range(n):\n        products.append(x[i] * y[i])\n"
             "    out[0] = sum(products)\n"),
    "clean": "import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    np.matmul(A, B, out=C)\n",
    "syntax_error": "import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    for i in range(M)\n        C[i] = A[i] @ B\n",
    "noqa_and_order": ("import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    unused = A.copy()\n    np.matmul(A, B, out=C)\n"
                       "    return missing  # noqa\n"),
    "pyflakes_codes": ("def gemm(A, B, C, M, N, K):\n    total = 0.0\n\n    def add(v):\n        total += v\n\n"
                       "    if M is 0:\n        return acc\n    unused = A.copy()\n    add(1.0)\n"),
}


def run_plugin(language: str, file: Path, out: Path, signature: str = '{"function": "gemm"}') -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(PLUGIN), "--language", language, "--file", str(file), "--task", "gemm",
                           "--signature", signature, "--out", str(out)], check=False,
                          stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120)


@unittest.skipUnless(HAVE_PYTHON_LINTERS, "ruff·pyflakes 가 이 인터프리터에 설치돼 있어야 한다 (requirements.txt)")
class EndToEndPythonTest(unittest.TestCase):
    """CLI 를 진짜 ruff·pyflakes 로 돌린다. 시나리오마다 한 번씩만 띄우고 결과를 나눠 쓴다."""

    tmp: Path
    docs: dict[str, dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="lint_e2e_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.docs = {}
        for name, source in SOURCES.items():
            file = cls.tmp / name / "kernel.py"
            file.parent.mkdir()
            file.write_text(source, encoding="utf-8")
            proc = run_plugin("numpy", file, file.with_name("lint.json"))
            assert proc.returncode == 0, f"{name}: rc={proc.returncode}\n{proc.stderr}"
            cls.docs[name] = json.loads(file.with_name("lint.json").read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def brief(self, name: str) -> list[tuple[str, str, int, str]]:
        return [(x["rule"], x["severity"], x["line"], x["category"]) for x in self.docs[name]["findings"]]

    def test_every_output_follows_the_contract(self) -> None:
        for name, doc in self.docs.items():
            with self.subTest(name):
                self.assertEqual(set(doc), {"tool", "findings", "summary", "unmapped", "versions"})
                self.assertEqual(list(doc["summary"]), list(CATEGORIES))
                for x in doc["findings"]:
                    self.assertEqual(set(x), FINDING_KEYS)
                    self.assertIn(x["category"], CATEGORIES)
                    self.assertIsInstance(x["line"], int)
                    self.assertNotIn("\n", x["message"])
                for category in CATEGORIES:
                    count = sum(1 for x in doc["findings"] if x["category"] == category)
                    # 도구가 하나도 돌지 않았으면 전부 null(검사하지 않음), 돌았으면(ruff 는 여섯 범주를 다 본다) 수
                    self.assertEqual(doc["summary"][category], None if doc["tool"] == "none" else count)
                self.assertIsInstance(doc["unmapped"], int)

    def test_unused_import_is_hygiene_dropped_and_counted_once(self) -> None:
        doc = self.docs["unused_import"]
        self.assertEqual((doc["tool"], doc["findings"]), ("baseline:ruff+pyflakes", []))
        self.assertEqual(doc["unmapped"], 1, "ruff 와 pyflakes 가 같은 줄을 보고해도 한 번만 센다")
        self.assertEqual(set(doc["versions"]), {"ruff", "pyflakes"})

    def test_undefined_name_is_one_merged_memory_error(self) -> None:
        self.assertEqual(self.brief("undefined_name"), [("ruff/F821", "error", 6, "memory")])
        self.assertEqual(self.docs["undefined_name"]["summary"]["memory"], 1)
        self.assertEqual(self.docs["undefined_name"]["unmapped"], 0)
        self.assertIn("acc", self.docs["undefined_name"]["findings"][0]["message"])

    def test_numpy_specific_findings(self) -> None:
        self.assertEqual(self.brief("numpy_specific"), [
            ("ruff/NPY001", "error", 5, "shape_dtype"),
            ("ruff/NPY002", "warning", 6, "api_misuse"),
        ])

    def test_perf_finding(self) -> None:
        self.assertEqual(self.brief("perf"), [("ruff/PERF401", "info", 4, "performance")])

    def test_clean_file(self) -> None:
        doc = self.docs["clean"]
        self.assertEqual((doc["tool"], doc["findings"], doc["unmapped"]), ("baseline:ruff+pyflakes", [], 0))
        self.assertEqual(doc["summary"], dict.fromkeys(CATEGORIES, 0))

    def test_syntax_error_is_not_zero_findings_but_no_tool(self) -> None:
        doc = self.docs["syntax_error"]
        self.assertEqual(doc, {"tool": "none", "findings": [], "summary": dict.fromkeys(CATEGORIES), "unmapped": 0, "versions": {}})

    def test_noqa_is_ignored_and_errors_come_first(self) -> None:
        self.assertEqual(self.brief("noqa_and_order"), [
            ("ruff/F821", "error", 7, "memory"),
            ("ruff/F841", "info", 5, "performance"),
        ])

    def test_pyflakes_alone_reports_under_the_same_codes(self) -> None:
        ruff_missing = dataclasses.replace(plugin.TOOLS["ruff"], available=lambda: False)
        file = self.tmp / "pyflakes_codes" / "kernel.py"
        with mock.patch.dict(plugin.TOOLS, {"ruff": ruff_missing}), contextlib.redirect_stderr(io.StringIO()):
            alone = plugin.lint("numpy", file, RULES)
        self.assertEqual(alone["tool"], "baseline:pyflakes")
        # 5행 `total += v`: 대입 전 읽기(F823)이면서 쓰이지 않는 지역 변수(F841)다
        expected = [("F823", "error", 5, "memory"), ("F821", "error", 8, "memory"), ("F632", "warning", 7, "numeric"),
                    ("F841", "info", 5, "performance"), ("F841", "info", 9, "performance")]
        self.assertEqual([(x["rule"], x["severity"], x["line"], x["category"]) for x in alone["findings"]],
                         [(f"pyflakes/{code}", *rest) for code, *rest in expected])
        self.assertEqual(self.brief("pyflakes_codes"), [(f"ruff/{code}", *rest) for code, *rest in expected],
                         "ruff 와 같이 돌면 같은 보고가 ruff 것 하나로 합쳐진다")
        self.assertEqual(alone["summary"], {"memory": 2, "numeric": 1, "concurrency": None, "shape_dtype": None,
                                            "api_misuse": None, "performance": 2},
                         "pyflakes 혼자면 그것이 낼 수 없는 범주는 0 이 아니라 null 이다")

    def test_other_python_family_languages_use_the_same_tools(self) -> None:
        file = self.tmp / "undefined_name" / "kernel.py"
        with contextlib.redirect_stderr(io.StringIO()):
            for language in ("numba", "halide_auto", "halide_manual"):
                with self.subTest(language):
                    doc = plugin.lint(language, file, RULES)
                    self.assertEqual((doc["tool"], doc["summary"]["memory"]), ("baseline:ruff+pyflakes", 1))

    def test_unknown_language_and_rust(self) -> None:
        file = self.tmp / "undefined_name" / "kernel.py"
        for language in ("fortran", "rust"):
            with self.subTest(language):
                out = self.tmp / f"{language}.json"
                proc = run_plugin(language, file, out)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                doc = json.loads(out.read_text(encoding="utf-8"))
                self.assertEqual((doc["tool"], doc["findings"], doc["summary"]), ("none", [], dict.fromkeys(CATEGORIES)))
                self.assertEqual(proc.stderr.startswith("[lint] "), language == "fortran")

    def test_usage_errors_exit_2_and_write_nothing(self) -> None:
        file = self.tmp / "clean" / "kernel.py"
        out = self.tmp / "never.json"
        cases = {
            "없는 파일": run_plugin("numpy", self.tmp / "no_such_kernel.py", out),
            "시그니처가 JSON 이 아님": run_plugin("numpy", file, out, signature="def gemm(A, B)"),
            "시그니처가 객체가 아님": run_plugin("numpy", file, out, signature="[1, 2]"),
        }
        for name, proc in cases.items():
            with self.subTest(name):
                self.assertEqual(proc.returncode, 2, proc.stderr)
                self.assertFalse(out.exists())

    def test_output_parent_directory_is_created_and_no_temp_file_is_left(self) -> None:
        file = self.tmp / "clean" / "kernel.py"
        out = self.tmp / "new_dir" / "deeper" / "lint.json"
        self.assertEqual(run_plugin("numpy", file, out).returncode, 0)
        self.assertEqual([p.name for p in out.parent.iterdir()], ["lint.json"])
        self.assertEqual(sorted(p.name for p in file.parent.iterdir()), ["kernel.py", "lint.json"], "소스 옆에 캐시를 남기지 않는다")


# ----------------------------------------------------------------------------- 하네스 연결


@unittest.skipUnless(HAVE_PYTHON_LINTERS, "ruff·pyflakes 가 이 인터프리터에 설치돼 있어야 한다 (requirements.txt)")
class HarnessRunLintTest(unittest.TestCase):
    """``harness.safety.run_lint`` 가 진짜 플러그인을 ``run_isolated`` 로 띄워 계약 검사를 통과시키는가."""

    def test_run_lint_with_the_real_plugin(self) -> None:
        from harness import safety

        with tempfile.TemporaryDirectory(prefix="lint_harness_") as tmp:
            sample_dir = Path(tmp)
            source = sample_dir / "kernel.py"
            source.write_text(SOURCES["numpy_specific"].replace("C[:] = tmp\n", "C[:] = tmp\n    return acc\n"), encoding="utf-8")
            out = sample_dir / "lint.json"
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                # lint 섹션이 없는 설정(확장 전 Config) → 기본 플러그인 경로·시간 상한
                res = safety.run_lint(SimpleNamespace(), language="numpy", backend="python", file=source, task="gemm",
                                      signature={"function": "gemm", "params": ["A", "B", "C", "M", "N", "K"]}, out=out,
                                      cwd=sample_dir)
            self.assertEqual(stderr.getvalue(), "")
            self.assertTrue(res.ok, res.error)
            self.assertEqual(res.tool, "baseline:ruff+pyflakes")
            self.assertEqual([(x["rule"], x["line"], x["category"]) for x in res.findings], [
                ("ruff/NPY001", 5, "shape_dtype"), ("ruff/F821", 9, "memory"), ("ruff/NPY002", 6, "api_misuse"),
            ])
            self.assertEqual(safety.lint_columns(res), {
                "lint_findings_total": 3, "lint_memory": 1, "lint_numeric": 0, "lint_concurrency": 0, "lint_shape_dtype": 1,
                "lint_api_misuse": 1, "lint_performance": 0,
            })
            self.assertEqual(safety.lint_messages(res)[1], "line 9: Undefined name `acc`")
            raw = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(raw["findings"], res.findings)
            self.assertEqual(sorted(p.name for p in sample_dir.iterdir()), ["kernel.py", "lint.json", "lint.stderr.log"])
            self.assertEqual((sample_dir / "lint.stderr.log").read_text(encoding="utf-8"), "", "버린 보고도 실패한 도구도 없다")


if __name__ == "__main__":
    unittest.main()
