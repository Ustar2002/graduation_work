"""린트 플러그인 베이스라인 (CLAUDE.md §6 의 계약, 설계 v3 §12.2).

    run_lint.py --language <lang> --file <path> --task <task> --signature <json> --out <json>

기성 린터를 계약의 출력 형식으로 감싼 것이다. 하네스(``harness.safety.run_lint``)는 이 CLI 와 출력 JSON 만 알고,
같은 CLI 를 지키는 다른 구현(인지프 ML lint)으로 바꿔 끼울 수 있다. 그래서 이 파일은 하네스를 임포트하지 않는다
(표준 라이브러리 + PyYAML).

- 언어 → 도구: numpy·numba·halide_auto·halide_manual → ruff, pyflakes / c_seq·c_openmp → cppcheck, clang-tidy / rust → 없음.
  설치되지 않은 도구는 조용히 건너뛴다. 하나도 돌지 않으면 ``tool`` 은 ``"none"`` 이고 findings 는 비어 있다(summary 는 전부 null).
- 도구는 이 프로세스의 서브프로세스로 돌린다(도구마다 ``TOOL_TIMEOUT_S``). 도구가 파일을 분석하지 못하면(시간 초과, 비정상 종료,
  읽을 수 없는 출력, **구문·컴파일 오류**) 그 도구는 돌지 않은 것으로 치고 stderr 에 한 줄 남긴다 — 해석하지 못한 파일을
  "보고 0건" 으로 적지 않기 위해서다.
- 규칙 ID → 범주·심각도는 ``rules.yaml``. 표에 없는 규칙의 보고는 버리고 출력의 ``unmapped`` 에 수만 센다.
- 두 도구가 같은 줄에서 같은 범주를 보고하면 같은 결함으로 보고 하나만 남긴다(``merge_across_tools``).
- ``summary`` 의 범주는 그 범주를 볼 수 있는 도구(``rules.yaml`` 의 그 도구 표에 그 범주가 있다, ``tool_categories``)가 하나라도
  끝까지 돌았을 때만 수이고, 아니면 null(검사하지 않음)이다 — 예: c_openmp 에서 clang-tidy 가 빠지면 cppcheck 가 보지 않는
  concurrency·api_misuse·performance 는 null.
- OpenMP 언어는 설정의 C 컴파일러(환경변수 ``CC``, 없으면 gcc)의 ``omp.h`` 한 파일만 담은 임시 디렉터리를 clang-tidy 에
  ``-isystem`` 으로 준다(``openmp_header_dir``; clang 에 libomp 의 omp.h 가 없으면 ``#include <omp.h>`` 가 컴파일 오류가 된다).

출력(``--out``; 계약의 ``tool``·``findings``·``summary`` 에 ``unmapped``·``versions`` 를 더한다)::

    {"tool": "baseline:ruff+pyflakes",
     "findings": [{"rule": "ruff/F821", "severity": "error", "line": 5, "message": "…", "category": "memory"}],
     "summary": {"memory": 1, "numeric": 0, "concurrency": 0, "shape_dtype": 0, "api_misuse": 0, "performance": 0},
     "unmapped": 0, "versions": {"ruff": "0.17.6", "pyflakes": "4.1.0"}}

``findings`` 는 심각도(error → warning → info), 줄 번호 순이다. ``unmapped`` 는 표에 없어 버린 보고의 수이고 어떤 규칙이었는지는
stderr 에 한 줄로 남는다. ``versions`` 는 분석을 마친 도구의 버전이다(같은 코드라도 도구 버전에 따라 보고가 달라진다).

종료 코드: 0 정상(도구가 없거나 실패해도 0 — 출력의 ``tool`` 이 말해 준다), 2 인자·입력 파일·rules.yaml 오류(출력을 쓰지 않는다).
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

_LOG = "[lint]"

RULES_YAML = Path(__file__).resolve().with_name("rules.yaml")

CATEGORIES = ("memory", "numeric", "concurrency", "shape_dtype", "api_misuse", "performance")
SEVERITIES = ("error", "warning", "info")  # findings 의 정렬 순서이기도 하다
NO_TOOL = "none"

# 도구 하나의 실행 상한(초). 언어당 도구가 둘이므로 합이 하네스의 lint.timeout_s(기본 60초)보다 작아야 한다.
TOOL_TIMEOUT_S = 20.0

PYTHON_TOOLS = ("ruff", "pyflakes")
C_TOOLS = ("cppcheck", "clang-tidy")
# 언어 → 도는 도구(먼저 적힌 도구의 보고가 병합에서 남는다). rust 는 이 베이스라인에 린터가 없다.
TOOLS_BY_LANGUAGE: dict[str, tuple[str, ...]] = {
    "numpy": PYTHON_TOOLS, "numba": PYTHON_TOOLS, "halide_auto": PYTHON_TOOLS, "halide_manual": PYTHON_TOOLS,
    "c_seq": C_TOOLS, "c_openmp": C_TOOLS,
    "rust": (),
}
OPENMP_LANGUAGES = ("c_openmp",)  # clang-tidy 의 컴파일 인자에 -fopenmp 를 더하는 언어
DEFAULT_CC = "gcc"  # 환경변수 CC 가 없을 때 omp.h 를 찾는 C 컴파일러(하네스의 기본 build.cc)

RUFF_SYNTAX_ERROR = "invalid-syntax"  # ruff 가 구문 오류에 붙이는 코드(규칙 선택과 무관하게 항상 보고된다)
PYFLAKES_UNCLASSIFIED = "unclassified"  # rules.yaml 의 pyflakes.messages 어느 것에도 맞지 않는 메시지
# cppcheck 가 파일(또는 그 설정)의 분석을 포기했다는 뜻의 ID
CPPCHECK_ANALYSIS_FAILURES = frozenset({"syntaxError", "unknownMacro", "internalAstError", "internalError", "cppcheckError"})

EXIT_OK = 0
EXIT_USAGE = 2

_PYFLAKES_LINE_RE = re.compile(r"(?P<line>\d+):\d+:? (?P<message>.*)")
_CLANG_DIAG_RE = re.compile(
    r"^(?P<path>\S.*?):(?P<line>\d+):\d+: warning: (?P<message>.*) \[(?P<checks>[^\[\]\s]+)\]$"
)
_CLANG_ERROR_RE = re.compile(r"^(?:\S.*?:\d+:\d+: )?(?:fatal )?error: ")


class ToolFailure(Exception):
    """도구가 파일을 분석하지 못했다(시간 초과, 비정상 종료, 읽을 수 없는 출력, 구문·컴파일 오류)."""


@dataclass(frozen=True)
class Report:
    """도구가 낸 보고 하나(규칙 표를 거치기 전)."""

    rule: str
    line: int
    message: str


def log(message: str) -> None:
    print(f"{_LOG} {message}", file=sys.stderr)


def one_line(text: str) -> str:
    return " ".join(text.split())


# ----------------------------------------------------------------------------- 규칙 표


def _check_str_list(where: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{where}: 비어 있지 않은 문자열의 리스트여야 한다")


def _check_table(tool: str, table: Any) -> None:
    if not isinstance(table, dict) or not table:
        raise ValueError(f"{tool}.rules: 비어 있지 않은 매핑이어야 한다")
    for key, entry in table.items():
        where = f"{tool}.rules.{key}"
        if not isinstance(entry, dict) or set(entry) != {"category", "severity", "reason"}:
            raise ValueError(f"{where}: category·severity·reason 세 키가 있어야 한다")
        if entry["category"] not in CATEGORIES:
            raise ValueError(f"{where}.category: {entry['category']!r} 는 {list(CATEGORIES)} 에 없다")
        if entry["severity"] not in SEVERITIES:
            raise ValueError(f"{where}.severity: {entry['severity']!r} 는 {list(SEVERITIES)} 에 없다")
        if not isinstance(entry["reason"], str) or not entry["reason"].strip():
            raise ValueError(f"{where}.reason: 근거 한 줄이 필요하다")


def load_rules(path: Path = RULES_YAML) -> dict[str, Any]:
    """``rules.yaml`` 을 읽고 형식을 검사한다. 틀리면 ``ValueError`` (어느 키인지 적는다)."""
    with path.open("r", encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    sections = ("ruff", "pyflakes", "cppcheck", "clang-tidy")
    if not isinstance(doc, dict) or set(doc) != set(sections) or not all(isinstance(doc[name], dict) for name in sections):
        raise ValueError(f"최상위는 {list(sections)} 네 매핑이어야 한다")
    _check_str_list("ruff.select", doc["ruff"].get("select"))
    _check_str_list("cppcheck.enable", doc["cppcheck"].get("enable"))
    _check_str_list("clang-tidy.checks", doc["clang-tidy"].get("checks"))
    _check_str_list("clang-tidy.compile_args", doc["clang-tidy"].get("compile_args"))
    for tool in ("ruff", "cppcheck", "clang-tidy"):
        _check_table(tool, doc[tool].get("rules"))
    messages = doc["pyflakes"].get("messages")
    if not isinstance(messages, dict) or not messages:
        raise ValueError("pyflakes.messages: 비어 있지 않은 매핑이어야 한다")
    for code, pattern in messages.items():
        if find_rule(doc["ruff"]["rules"], code) is None:
            raise ValueError(f"pyflakes.messages.{code}: ruff 표에 없는 코드다")
        try:
            re.compile(pattern)
        except (re.error, TypeError) as e:
            raise ValueError(f"pyflakes.messages.{code}: 정규식이 아니다 ({e})") from None
    return doc


def find_rule(table: dict[str, dict[str, str]], rule_id: str) -> dict[str, str] | None:
    """규칙 표에서 ``rule_id`` 의 항목: 정확 일치 → 가장 긴 접두(``*`` 로 끝나는 키) 일치 → None."""
    if rule_id in table:
        return table[rule_id]
    prefixes = [key for key in table if key.endswith("*") and rule_id.startswith(key[:-1])]
    return table[max(prefixes, key=len)] if prefixes else None


# ----------------------------------------------------------------------------- 도구 실행


def run_tool(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    """린터 하나를 서브프로세스로 돌린다. 시간 초과·실행 실패는 ``ToolFailure``."""
    try:
        return subprocess.run(cmd, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=TOOL_TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        raise ToolFailure(f"{TOOL_TIMEOUT_S:g}초 안에 끝나지 않았다") from None
    except OSError as e:
        raise ToolFailure(f"실행 실패: {e}") from None


def python_module_command(module: str) -> list[str]:
    """``sys.executable -m <module>``. ``-P``: 작업 디렉터리(LLM 이 쓴 kernel.py 가 있는 곳)를 sys.path 에 넣지 않는다."""
    return [sys.executable, "-P", "-m", module]


def ruff_command(file: Path, cfg: dict[str, Any]) -> list[str]:
    """ruff 명령. ``--isolated``·``--no-cache``: 주변의 설정 파일과 캐시를 쓰지 않는다. ``--ignore-noqa``: 생성된 코드의
    ``# noqa`` 가 보고를 끄지 못하게 한다. ``--target-version``: 이 인터프리터(= 커널을 돌리는 인터프리터)의 문법 기준."""
    select = sorted({*cfg["select"], *(key.rstrip("*") for key in cfg["rules"])})
    target = f"py{sys.version_info.major}{sys.version_info.minor}"
    return [*python_module_command("ruff"), "check", "--output-format", "json", "--isolated", "--no-cache", "--ignore-noqa",
            "--target-version", target, "--select", ",".join(select), str(file)]


def parse_ruff(stdout: str) -> list[Report]:
    """``ruff check --output-format json`` 의 출력 → 보고 목록. 구문 오류가 있으면 ``ToolFailure``."""
    try:
        items = json.loads(stdout)
    except ValueError as e:
        raise ToolFailure(f"JSON 출력을 읽을 수 없다: {e}") from None
    if not isinstance(items, list):
        raise ToolFailure("JSON 출력이 리스트가 아니다")
    reports: list[Report] = []
    for item in items:
        code = item.get("code") or RUFF_SYNTAX_ERROR  # 예전 ruff 는 구문 오류의 code 가 null 이다
        line = int((item.get("location") or {}).get("row") or 0)
        message = one_line(str(item.get("message") or ""))
        if code == RUFF_SYNTAX_ERROR:
            raise ToolFailure(f"구문 오류로 분석하지 못했다 (line {line}: {message})")
        reports.append(Report(code, line, message))
    return reports


def run_ruff(file: Path, language: str, rules: dict[str, Any]) -> tuple[list[Report], str]:
    proc = run_tool(ruff_command(file, rules["ruff"]))
    if proc.returncode not in (0, 1):  # 0 = 보고 없음, 1 = 보고 있음, 그 밖 = 비정상 종료(모르는 규칙 코드 등)
        raise ToolFailure(f"rc={proc.returncode}: {one_line(proc.stderr)[-300:]}")
    return parse_ruff(proc.stdout), importlib.metadata.version("ruff")


def parse_pyflakes(stdout: str, stderr: str, file: Path, messages: dict[str, str]) -> list[Report]:
    """pyflakes 의 ``<file>:<line>:<col>: <message>`` 줄 → 보고 목록(규칙 = ``messages`` 가 정하는 ruff F 코드).

    구문 오류·디코딩 오류는 stderr 로 나온다 → ``ToolFailure``.
    """
    prefix = f"{file}:"
    for line in stderr.splitlines():
        if line.startswith(prefix):
            raise ToolFailure(f"구문 오류로 분석하지 못했다 ({one_line(line[len(prefix):])})")
    reports: list[Report] = []
    for line in stdout.splitlines():
        m = _PYFLAKES_LINE_RE.match(line[len(prefix):]) if line.startswith(prefix) else None
        if m is None:
            continue
        message = one_line(m["message"])
        rule = next((code for code, pattern in messages.items() if re.search(pattern, message)), PYFLAKES_UNCLASSIFIED)
        reports.append(Report(rule, int(m["line"]), message))
    return reports


def run_pyflakes(file: Path, language: str, rules: dict[str, Any]) -> tuple[list[Report], str]:
    proc = run_tool([*python_module_command("pyflakes"), str(file)])
    if proc.returncode not in (0, 1):  # 0 = 보고 없음, 1 = 보고 또는 구문 오류
        raise ToolFailure(f"rc={proc.returncode}: {one_line(proc.stderr)[-300:]}")
    reports = parse_pyflakes(proc.stdout, proc.stderr, file, rules["pyflakes"]["messages"])
    return reports, importlib.metadata.version("pyflakes")


def cppcheck_command(file: Path, cfg: dict[str, Any]) -> list[str]:
    """cppcheck 명령. 결과는 XML(v2)로 stderr 에 나온다. ``--library=posix``: posix_memalign 등의 할당·해제 짝을 알려 준다."""
    return ["cppcheck", f"--enable={','.join(cfg['enable'])}", "--language=c", "--std=c11", "--library=posix", "--xml",
            "--quiet", str(file)]


def parse_cppcheck(xml_text: str, file: Path) -> tuple[list[Report], str]:
    """cppcheck 의 XML 출력 → (보고 목록, cppcheck 버전). 분석을 포기했다는 ID 가 있으면 ``ToolFailure``."""
    start = xml_text.find("<?xml")
    if start < 0:
        raise ToolFailure(f"XML 출력이 없다: {one_line(xml_text)[-300:]}")
    try:
        root = ET.fromstring(xml_text[start:])
    except ET.ParseError as e:
        raise ToolFailure(f"XML 출력을 읽을 수 없다: {e}") from None
    reports: list[Report] = []
    for error in root.iter("error"):
        rule = error.get("id") or ""
        message = one_line(error.get("msg") or "")
        if rule in CPPCHECK_ANALYSIS_FAILURES:
            raise ToolFailure(f"분석하지 못했다 ({rule}: {message})")
        location = error.find("location")  # 첫 location 이 오류 지점이다(뒤의 것들은 거기에 이르는 경로)
        if location is None:
            reports.append(Report(rule, 0, message))
        elif Path(location.get("file") or "").name == file.name:  # 하네스가 만든 헤더(kernel_args.h) 안의 보고는 뺀다
            reports.append(Report(rule, int(location.get("line") or 0), message))
    version = root.find("cppcheck")
    return reports, (version.get("version") if version is not None else None) or "unknown"


def run_cppcheck(file: Path, language: str, rules: dict[str, Any]) -> tuple[list[Report], str]:
    proc = run_tool(cppcheck_command(file, rules["cppcheck"]))
    if proc.returncode != 0:
        raise ToolFailure(f"rc={proc.returncode}: {one_line(proc.stdout + ' ' + proc.stderr)[-300:]}")
    return parse_cppcheck(proc.stderr, file)


def clang_tidy_command(file: Path, cfg: dict[str, Any], *, openmp: bool, system_includes: tuple[str, ...] = ()) -> list[str]:
    """clang-tidy 명령. ``--config={}``: 주변의 .clang-tidy 를 찾지 않는다. ``--`` 뒤는 컴파일 인자(컴파일 DB 를 찾지 않는다).
    ``system_includes``: ``-isystem`` 으로 더할 디렉터리(OpenMP 언어의 omp.h 디렉터리)."""
    enabled = [check for check in cfg["checks"] if not check.startswith("-")]
    disabled = [check for check in cfg["checks"] if check.startswith("-")]
    checks = ",".join(["-*", *enabled, *cfg["rules"], *disabled])  # 뒤의 항목이 앞을 덮으므로 끄는 항목이 마지막이다
    return ["clang-tidy", str(file), f"--checks={checks}", "--config={}", "--quiet", "--", *cfg["compile_args"],
            *(arg for directory in system_includes for arg in ("-isystem", directory)),
            *(["-fopenmp"] if openmp else [])]


def openmp_header_dir(tmp: Path) -> Path | None:
    """설정의 C 컴파일러(환경변수 ``CC``, 없으면 gcc)의 ``omp.h`` 한 파일만 ``tmp`` 에 복사하고 ``tmp`` 를 돌려준다.

    clang-tidy(clang)에는 libomp 의 omp.h 가 없을 수 있다(실험 머신의 clang 21). 그러면 ``#include <omp.h>`` 가 컴파일 오류가
    되어 clang-tidy 가 통째로 빠진다. 컴파일러의 include 디렉터리(``<cc> -print-file-name=include``)를 통째로 넣으면 그 안의
    인트린식 헤더를 clang 이 해석하지 못하므로 omp.h 만 옮긴다. 컴파일러가 없거나 그 디렉터리에 omp.h 가 없으면 None.
    """
    cc = os.environ.get("CC") or DEFAULT_CC
    if shutil.which(cc) is None:
        return None
    try:
        proc = run_tool([cc, "-print-file-name=include"])
    except ToolFailure:
        return None
    printed = proc.stdout.strip().splitlines()
    include = Path(printed[-1].strip()) if proc.returncode == 0 and printed else None
    # 찾지 못하면 gcc 는 넘겨받은 이름("include")을 그대로 출력한다: 상대 경로는 작업 디렉터리의 것이므로 쓰지 않는다
    if include is None or not include.is_absolute() or not (include / "omp.h").is_file():
        return None
    shutil.copyfile(include / "omp.h", tmp / "omp.h")
    log(f"clang-tidy: {include / 'omp.h'} 를 -isystem 으로 준다")
    return tmp


def parse_clang_tidy(stdout: str, file: Path, table: dict[str, dict[str, str]]) -> list[Report]:
    """clang-tidy 의 ``<file>:<line>:<col>: warning: <message> [<check>,…]`` 줄 → 보고 목록.

    ``error:`` 진단(컴파일 오류 — 헤더를 못 찾는 경우 포함)이 하나라도 있으면 분석이 끝까지 가지 않았으므로 ``ToolFailure``.
    한 진단에 검사 이름이 여럿이면(별칭) 표에 있는 첫 이름을 규칙으로 삼는다.
    """
    reports: list[Report] = []
    for line in stdout.splitlines():
        if _CLANG_ERROR_RE.match(line):
            raise ToolFailure(f"컴파일 오류로 분석하지 못했다 ({one_line(line)[-300:]})")
        m = _CLANG_DIAG_RE.match(line)
        if m is None or Path(m["path"]).name != file.name:
            continue
        names = m["checks"].split(",")
        rule = next((name for name in names if find_rule(table, name) is not None), names[0])
        reports.append(Report(rule, int(m["line"]), one_line(m["message"])))
    return reports


def run_clang_tidy(file: Path, language: str, rules: dict[str, Any]) -> tuple[list[Report], str]:
    cfg = rules["clang-tidy"]
    openmp = language in OPENMP_LANGUAGES
    with tempfile.TemporaryDirectory(prefix="lint_omp_") as tmp:
        omp_dir = openmp_header_dir(Path(tmp)) if openmp else None
        proc = run_tool(clang_tidy_command(file, cfg, openmp=openmp, system_includes=(str(omp_dir),) if omp_dir else ()))
    reports = parse_clang_tidy(proc.stdout, file, cfg["rules"])
    if proc.returncode != 0:
        raise ToolFailure(f"rc={proc.returncode}: {one_line(proc.stderr)[-300:]}")
    version = re.search(r"version (\S+)", run_tool(["clang-tidy", "--version"]).stdout)
    return reports, version.group(1) if version else "unknown"


@dataclass(frozen=True)
class Tool:
    """린터 하나: 규칙 표의 이름(pyflakes 는 ruff 표를 쓴다), 설치 여부, 실행 함수(→ 보고 목록, 버전)."""

    table: str
    available: Callable[[], bool]
    run: Callable[[Path, str, dict[str, Any]], tuple[list[Report], str]]


TOOLS: dict[str, Tool] = {
    "ruff": Tool("ruff", lambda: importlib.util.find_spec("ruff") is not None, run_ruff),
    "pyflakes": Tool("ruff", lambda: importlib.util.find_spec("pyflakes") is not None, run_pyflakes),
    "cppcheck": Tool("cppcheck", lambda: shutil.which("cppcheck") is not None, run_cppcheck),
    "clang-tidy": Tool("clang-tidy", lambda: shutil.which("clang-tidy") is not None, run_clang_tidy),
}


# ----------------------------------------------------------------------------- 대응·병합·출력


def merge_across_tools(per_tool: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """도구 간 중복을 합친다: 두 도구가 같은 줄에서 같은 범주를 보고하면 같은 결함으로 보고 먼저 돈 도구의 보고만 남긴다.

    한 도구가 같은 (줄, 범주)에 여러 건을 내면 그 수는 그대로 두고, 다른 도구의 보고는 그 수를 넘는 만큼만 더한다
    (다중집합의 최댓값). 표에 없는 규칙의 보고(범주 None)도 같은 방식으로 합쳐 ``unmapped`` 가 같은 것을 두 번 세지 않게 한다.
    """
    merged: list[dict[str, Any]] = []
    kept: Counter[tuple[int, str | None]] = Counter()
    for findings in per_tool:
        seen: Counter[tuple[int, str | None]] = Counter()
        for finding in findings:
            key = (finding["line"], finding["category"])
            seen[key] += 1
            if seen[key] > kept[key]:
                merged.append(finding)
        for key, count in seen.items():
            kept[key] = max(kept[key], count)
    return merged


def to_finding(tool: str, report: Report, entry: dict[str, str] | None) -> dict[str, Any]:
    """보고 하나를 출력의 finding 형태로. 표에 없는 규칙(``entry`` 가 None)은 범주·심각도가 None 이다(병합 뒤 버려진다)."""
    return {
        "rule": f"{tool}/{report.rule}",
        "severity": entry["severity"] if entry else None,
        "line": report.line,
        "message": report.message,
        "category": entry["category"] if entry else None,
    }


def tool_categories(name: str, rules: dict[str, Any]) -> set[str]:
    """도구가 보고할 수 있는 범주 = ``rules.yaml`` 의 그 도구 표에 있는 범주.
    pyflakes 는 ruff 표를 쓰지만 ``pyflakes.messages`` 가 가리키는 코드만 낼 수 있으므로 그 코드들의 범주다."""
    if name == "pyflakes":
        table = rules["ruff"]["rules"]
        return {find_rule(table, code)["category"] for code in rules["pyflakes"]["messages"]}
    return {entry["category"] for entry in rules[TOOLS[name].table]["rules"].values()}


def lint(language: str, file: Path, rules: dict[str, Any]) -> dict[str, Any]:
    """``file`` 에 그 언어의 도구를 돌려 계약 형식의 출력 문서를 만든다.

    ``summary`` 의 범주는 그 범주를 볼 수 있는 도구가 하나라도 끝까지 돌았을 때만 수이고, 아니면 None(null; 검사하지 않음)이다.
    """
    tools = TOOLS_BY_LANGUAGE.get(language)
    if tools is None:
        log(f"언어 {language!r} 는 이 베이스라인이 다루지 않는다: 도구 없이 빈 결과를 낸다")
        tools = ()
    versions: dict[str, str] = {}  # 분석을 마친 도구 → 버전 (돈 순서)
    covered: set[str] = set()  # 분석을 마친 도구가 볼 수 있는 범주
    per_tool: list[list[dict[str, Any]]] = []
    for name in tools:
        tool = TOOLS[name]
        if not tool.available():
            continue
        try:
            reports, versions[name] = tool.run(file, language, rules)
        except ToolFailure as e:
            log(f"{name}: {e}: 이 도구의 결과 없이 계속한다")
            continue
        covered |= tool_categories(name, rules)
        table = rules[tool.table]["rules"]
        per_tool.append([to_finding(name, report, find_rule(table, report.rule)) for report in reports])
    merged = merge_across_tools(per_tool)
    findings = [f for f in merged if f["category"] is not None]
    findings.sort(key=lambda f: (SEVERITIES.index(f["severity"]), f["line"], f["rule"]))
    unmapped = Counter(f["rule"] for f in merged if f["category"] is None)
    if unmapped:
        log("표에 없는 규칙의 보고를 버렸다: " + ", ".join(f"{rule} x{n}" for rule, n in sorted(unmapped.items())))
    counts = Counter(f["category"] for f in findings)
    return {
        "tool": f"baseline:{'+'.join(versions)}" if versions else NO_TOOL,
        "findings": findings,
        "summary": {category: counts[category] if category in covered else None for category in CATEGORIES},
        "unmapped": sum(unmapped.values()),
        "versions": versions,
    }


def write_json_atomic(path: Path, doc: dict[str, Any]) -> None:
    """같은 디렉터리의 임시 파일에 쓴 뒤 ``os.replace`` 로 바꿔치기한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# ----------------------------------------------------------------------------- CLI


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"JSON 이 아니다: {e}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("JSON 객체여야 한다")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_lint.py", description="린트 플러그인 베이스라인 (CLAUDE.md §6)")
    parser.add_argument("--language", required=True, help="실험 언어 이름 (numpy, c_openmp, …)")
    parser.add_argument("--file", type=Path, required=True, help="검사할 소스 파일")
    parser.add_argument("--task", required=True, help="과제 이름 (계약의 인자; 이 베이스라인은 쓰지 않는다)")
    parser.add_argument("--signature", type=_json_object, required=True,
                        help="요구 시그니처 JSON 객체 (계약의 인자; 이 베이스라인은 형식만 확인한다)")
    parser.add_argument("--out", type=Path, required=True, help="출력 JSON 경로")
    ns = parser.parse_args(argv)
    if not ns.file.is_file():
        log(f"검사할 파일이 없다: {ns.file}")
        return EXIT_USAGE
    try:
        rules = load_rules()
    except (OSError, yaml.YAMLError, ValueError) as e:
        log(f"{RULES_YAML} 를 쓸 수 없다: {e}")
        return EXIT_USAGE
    write_json_atomic(ns.out, lint(ns.language, ns.file.resolve(), rules))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
