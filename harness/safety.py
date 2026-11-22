"""안전성 요약 (설계 v3 §12, CLAUDE.md §6): 린트 플러그인 호출과 ASan/UBSan 재빌드·실행.

린트 — 하네스는 플러그인의 내부를 모르고 CLI·출력 계약만 안다:

    PYTHON_EXE <plugin> --language <lang> --file <path> --task <task> --signature <json> --out <json>

- ``run_lint`` 는 플러그인을 ``run_isolated`` 로 띄우고 출력을 계약대로 검사한다. 플러그인 문제(파일 없음, 0 이 아닌 종료,
  시간 초과, JSON 아님, 계약 위반)로는 예외를 던지지 않는다: 빈 ``LintResult(ok=False, error=…)`` 와 stderr 경고 한 줄로
  끝내 실험이 막히지 않게 한다. 플러그인의 출력 원문(``lint.json``)과 stderr(``lint.stderr.log``)는 샘플 디렉터리에 남는다.
- ``lint_columns`` 는 린트가 실제로 돌았을 때만 수를 준다. 플러그인이 실패했거나 도구가 하나도 돌지 않았으면(``tool == "none"``)
  열은 NA 다 — 0 은 "검사했고 없음" 이어야 한다. 범주 단위도 같다: summary 가 null 인 범주(그 범주를 볼 수 있는 도구가
  끝까지 돌지 않았다)는 NA 다.

sanitizer — Linux 의 C 커널만(Python·Numba·Halide·Rust 는 돌리지 않는다; TSan 은 libgomp 가 계측되지 않아 쓰지 않는다):

- ``run_sanitizers`` 는 샘플 디렉터리의 ``san/`` 에 ``kernel_san.so`` 를 빌드하고(``build.sanitize_flags``), 워커의 check 를
  ASan(gcc 면 libubsan 도) 런타임을 선행 로드한 환경에서 한 번 돌린 뒤(``sanitizer_env``), ``san/asan.<pid>``·
  ``san/ubsan.<pid>`` 로그와 워커의 stderr 파일 전체를 파싱한다(``parse_sanitizer_logs``).
- ASan 은 ``abort_on_error=1`` 이라 첫 오류에서 워커가 SIGABRT 로 죽는다 — 탐지가 드러나는 정상 경로다.
- 0 은 끝까지 돈 실행에서만 0 이다: 워커가 끝까지 가지 못했거나 커널을 로드하지 못했는데 보고도 없으면 그 수는 None(판정 불가)이다.
- sanitizer 실행에서 워커는 ``--sanitize`` 로 입출력 버퍼를 가드 영역 없이 정확한 크기로 할당한다: 하네스 배열을 조금 넘는
  읽기·쓰기도 malloc 블록 밖이라 ASan 에 보인다(정상 check 의 가드 영역은 그대로이고 ``oob_write`` 는 그쪽이 잡는다).

이 모듈은 LLM 코드를 실행하지 않는다(실행은 넘겨받은 ``worker`` 가 ``harness.worker`` 서브프로세스로 한다).

CLI(진단용):
    python -m harness.safety --lint FILE --language LANG [--task T]   # 플러그인을 돌려 LintResult 를 JSON 으로
    python -m harness.safety --parse-logs DIR                         # sanitizer 로그 디렉터리(san/) → 보고 수·종류
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

from harness.build.c import build_c
from harness.config import ConfigError, load_config
from harness.paths import PYTHON_EXE, resolve
from harness.procutil import ProcResult, run_isolated
from harness.protocol import launch_env_minimal

_LOG = "[safety]"

# 린트 범주(CLAUDE.md §6; 결과 행의 lint_<범주> 열과 같은 순서)와 finding 의 키
LINT_CATEGORIES: tuple[str, ...] = ("memory", "numeric", "concurrency", "shape_dtype", "api_misuse", "performance")
FINDING_KEYS: tuple[str, ...] = ("rule", "severity", "line", "message", "category")
NO_TOOL = "none"  # 플러그인이 "도구가 하나도 돌지 않았다"를 알리는 tool 값 (설계 v3 §12.2)

# 설정에 해당 키가 없을 때(lint·safety 섹션과 build.sanitize_flags 가 생기기 전의 Config)의 기본값 = 설계 v3 §3.
# lint.enabled 의 기본값은 True, safety.enabled 는 False 다.
DEFAULT_LINT_PLUGIN = "plugins/lint/run_lint.py"
DEFAULT_LINT_TIMEOUT_S = 60.0
DEFAULT_SAFETY_TIMEOUT_S = 300.0
DEFAULT_SANITIZE_FLAGS: tuple[str, ...] = ("-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g")
DEFAULT_COMPILE_TIMEOUT_S = 60.0  # build.compile_timeout_s (설계 §2)

SAN_DIR = "san"  # 샘플 디렉터리 아래 sanitizer 빌드·로그 디렉터리
SAN_ARTIFACT = "kernel_san.so"
ASAN_LOG = "asan"  # log_path 의 파일 이름 접두: 런타임이 ".<pid>" 를 붙인다
UBSAN_LOG = "ubsan"
# sanitizer 워커의 stderr 파일(run.run_worker 가 <cwd>/<stage>.stderr.log 에 쓰고 cwd 는 샘플 디렉터리다)
SAFETY_STDERR_LOG = "safety.stderr.log"
# sanitizer 워커의 추가 인자: check 한 번, --sanitize = 가드 영역 없이 정확한 크기로 할당(하네스 배열을 넘는 접근이 ASan 에 보인다)
SAFETY_WORKER_ARGS: tuple[str, ...] = ("--check-repeats", "1", "--sanitize")
IS_LINUX: bool = sys.platform.startswith("linux")

_ASAN_ERROR_RE = re.compile(r"^==\d+==ERROR: AddressSanitizer: (?P<what>.+)$")
# 종류가 글자로 시작하는 요약만 오류 보고다("… 4096 byte(s) leaked …" 같은 누수 요약은 아니다)
_ASAN_SUMMARY_RE = re.compile(r"^SUMMARY: AddressSanitizer: (?P<kind>[A-Za-z][\w-]*)")
_UBSAN_ERROR_RE = re.compile(r"^(?P<where>\S.*?): runtime error: (?P<message>.+)$")

# UBSan 메시지에 이 문구가 있으면 그 종류다. 이름은 UBSan 런타임의 오류 종류 이름을 따른다(0 나눗셈만 정수·부동소수를
# 메시지로 가를 수 없어 divide-by-zero 하나로 둔다). 어느 것도 아니면 런타임의 일반 이름 "undefined-behavior".
_UBSAN_KINDS: tuple[tuple[str, str], ...] = (
    ("signed integer overflow", "signed-integer-overflow"),
    ("negation of", "signed-integer-overflow"),
    ("division by zero", "divide-by-zero"),
    ("out of bounds for type", "out-of-bounds-index"),
    ("misaligned address", "misaligned-pointer-use"),
    ("null pointer", "null-pointer-use"),
    ("shift exponent", "invalid-shift-exponent"),
    ("left shift of", "invalid-shift-base"),
    ("outside the range of representable values", "float-cast-overflow"),
    ("overflowed to", "pointer-overflow"),
    ("variable length array bound", "non-positive-vla-index"),
    ("insufficient space for an object", "insufficient-object-size"),
)


def _setting(cfg: Any, section: str, name: str, default: Any) -> Any:
    """``cfg.<section>.<name>``. 섹션이나 필드가 없으면(확장 전 Config) ``default``."""
    return getattr(getattr(cfg, section, None), name, default)


def _last_line(text: str) -> str:
    """마지막 비어 있지 않은 줄(공백 정리, 300 글자까지). 없으면 빈 문자열."""
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    return lines[-1][:300] if lines else ""


# ----------------------------------------------------------------------------- 린트


@dataclass
class LintResult:
    """린트 플러그인의 결과. ``ok=False`` 면 플러그인이 돌지 않았거나 출력을 쓸 수 없었다(``error`` 에 사유; 내용은 비어 있다)."""

    tool: str
    findings: list[dict]
    summary: dict[str, int | None]  # None = 그 범주를 볼 수 있는 도구가 돌지 않았다(검사하지 않음)
    ok: bool
    error: str = ""


def _empty_lint(error: str) -> LintResult:
    return LintResult(tool=NO_TOOL, findings=[], summary=dict.fromkeys(LINT_CATEGORIES, 0), ok=False, error=error)


def parse_lint_output(doc: Any) -> LintResult:
    """플러그인 출력(JSON 을 읽은 값)을 계약(CLAUDE.md §6)대로 검사해 ``LintResult`` 로 만든다. 위반이면 ``ValueError`` (사유).

    계약: ``tool`` (비어 있지 않은 문자열), ``findings`` (각 항목에 rule·severity·line·message·category; category 는 여섯 범주),
    ``summary`` (정확히 여섯 범주의 수 = findings 를 범주별로 센 값, 또는 null = 그 범주를 볼 수 있는 도구가 돌지 않아
    검사하지 않음 — 그 범주의 finding 이 있으면 위반). 그 밖의 최상위 키와 finding 의 추가 키는 무시한다.
    """
    if not isinstance(doc, dict):
        raise ValueError("최상위가 객체가 아니다")
    tool, findings, summary = doc.get("tool"), doc.get("findings"), doc.get("summary")
    if not isinstance(tool, str) or not tool:
        raise ValueError("tool 이 비어 있지 않은 문자열이 아니다")
    if not isinstance(findings, list):
        raise ValueError("findings 가 리스트가 아니다")
    clean: list[dict] = []
    for i, finding in enumerate(findings):
        where = f"findings[{i}]"
        if not isinstance(finding, dict):
            raise ValueError(f"{where} 가 객체가 아니다")
        missing = [key for key in FINDING_KEYS if key not in finding]
        if missing:
            raise ValueError(f"{where} 에 키가 없다: {missing}")
        for key in ("rule", "severity"):
            if not isinstance(finding[key], str) or not finding[key]:
                raise ValueError(f"{where}.{key} 가 비어 있지 않은 문자열이 아니다")
        if not isinstance(finding["message"], str):
            raise ValueError(f"{where}.message 가 문자열이 아니다")
        line = finding["line"]
        if isinstance(line, bool) or not isinstance(line, int) or line < 0:
            raise ValueError(f"{where}.line 이 0 이상의 정수가 아니다: {line!r}")
        if finding["category"] not in LINT_CATEGORIES:
            raise ValueError(f"{where}.category 가 여섯 범주에 없다: {finding['category']!r}")
        clean.append({key: finding[key] for key in FINDING_KEYS})
    if not isinstance(summary, dict) or set(summary) != set(LINT_CATEGORIES):
        raise ValueError(f"summary 가 여섯 범주를 키로 하는 객체가 아니다: {summary!r}")
    for category in LINT_CATEGORIES:
        count = summary[category]
        actual = sum(1 for finding in clean if finding["category"] == category)
        if count is None:  # 검사하지 않은 범주
            if actual:
                raise ValueError(f"summary.{category} 가 null(검사하지 않음)인데 findings 에는 {actual} 건이다")
            continue
        if isinstance(count, bool) or not isinstance(count, int) or count != actual:
            raise ValueError(f"summary.{category} = {count!r} 인데 findings 에는 {actual} 건이다")
    return LintResult(tool=tool, findings=clean, summary={category: summary[category] for category in LINT_CATEGORIES}, ok=True)


def run_lint(cfg: Any, *, language: str, backend: str, file: Path, task: str, signature: dict, out: Path, cwd: Path) -> LintResult:
    """린트 플러그인을 ``file`` 에 돌린다. 플러그인의 출력 원문은 ``out`` (플러그인이 쓴다)에, stderr 는 그 옆의
    ``<out 이름>.stderr.log`` (``lint.json`` → ``lint.stderr.log``)에 남는다.

    플러그인 문제로는 예외를 던지지 않는다: 빈 결과(``ok=False``, ``error`` = 사유)를 돌려주고 stderr 에 경고 한 줄을 쓴다.
    ``lint.enabled`` 가 꺼져 있으면 플러그인을 띄우지 않고 ``error="disabled"`` 인 빈 결과를 준다(경고 없음).
    """
    if not _setting(cfg, "lint", "enabled", True):
        return _empty_lint("disabled")
    plugin = resolve(_setting(cfg, "lint", "plugin", DEFAULT_LINT_PLUGIN))
    timeout_s = float(_setting(cfg, "lint", "timeout_s", DEFAULT_LINT_TIMEOUT_S))
    out = Path(out)

    def fail(error: str) -> LintResult:
        error = " ".join(error.split())  # 경고도 result.json 의 사유도 한 줄로
        print(f"{_LOG} 경고: 린트 결과 없이 계속한다 ({language}/{backend}, {Path(file).name}): {error}", file=sys.stderr)
        return _empty_lint(error)

    if not plugin.is_file():
        return fail(f"plugin_missing: {plugin}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)  # 이전 실행의 출력을 새 결과로 읽지 않게
    stderr_log = out.with_suffix(".stderr.log")  # 어느 도구가 왜 빠졌는지, 표에 없어 버린 규칙이 무엇인지가 여기 남는다
    cmd = [str(PYTHON_EXE), str(plugin), "--language", language, "--file", str(file), "--task", task,
           "--signature", json.dumps(signature), "--out", str(out)]
    env = launch_env_minimal()
    cc = _setting(cfg, "build", "cc", None)
    if cc:  # 설정의 C 컴파일러: 베이스라인 플러그인은 그 컴파일러의 omp.h 를 clang-tidy 에 준다(CLI 계약은 그대로)
        env["CC"] = str(cc)
    try:
        proc = run_isolated(cmd, timeout_s=timeout_s, cwd=cwd, env=env, stderr_path=stderr_log)
    except OSError as e:
        return fail(f"launch_failed: {e}")
    if proc.timed_out:
        return fail(f"timeout after {timeout_s:g}s")
    if proc.rc != 0:
        return fail(f"plugin_failed: rc={proc.rc}: {_last_line(proc.stderr_tail)}")
    try:
        doc = json.loads(out.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return fail(f"no_output: {out.name}")
    except (OSError, ValueError) as e:
        return fail(f"invalid_json: {e}")
    try:
        return parse_lint_output(doc)
    except ValueError as e:
        return fail(f"contract_violation: {e}")


def lint_columns(res: LintResult) -> dict[str, int | None]:
    """결과 행의 ``lint_findings_total``, ``lint_<범주>`` 열.

    린트가 실제로 돌지 않았으면(플러그인 실패, 또는 ``tool == "none"``: 그 언어의 도구가 없거나 파일을 분석하지 못했다)
    값은 전부 None(NA) 이다. 돌았어도 summary 가 null 인 범주(그 범주를 볼 수 있는 도구가 끝까지 돌지 않았다)는 None 이다.
    ``lint_findings_total`` 은 검사한 범주의 합(= findings 수)이고, 검사한 범주가 하나도 없으면 None 이다.
    """
    measured = res.ok and res.tool != NO_TOOL
    checked = measured and any(count is not None for count in res.summary.values())
    columns: dict[str, int | None] = {"lint_findings_total": len(res.findings) if checked else None}
    for category in LINT_CATEGORIES:
        columns[f"lint_{category}"] = res.summary[category] if measured else None
    return columns


def lint_messages(res: LintResult, limit: int = 20) -> list[str]:
    """수정 프롬프트에 넣을 ``"line N: message"`` 문자열. 플러그인이 준 순서(우선순위)대로 앞에서 ``limit`` 개."""
    return [f"line {finding['line']}: {' '.join(finding['message'].split())}" for finding in res.findings[:limit]]


# ----------------------------------------------------------------------------- sanitizer: 환경


def sanitizer_options(options: dict[str, str]) -> str:
    """sanitizer 런타임의 옵션 문자열 ``key=value:key=value``.

    런타임은 공백·쉼표·콜론을 구분자로 읽으므로 그런 글자가 든 값(경로)은 따옴표로 감싼다.
    """
    parts: list[str] = []
    for key, value in options.items():
        quote = ""
        if re.search(r"[\s,:]", value):
            quote = next((q for q in "'\"" if q not in value), "")
            if not quote:
                raise ValueError(f"sanitizer 옵션 {key} 의 값에 구분자와 두 가지 따옴표가 모두 있어 넘길 수 없다: {value!r}")
        parts.append(f"{key}={quote}{value}{quote}")
    return ":".join(parts)


def sanitizer_env(cfg: Any, cc: str, log_dir: Path) -> dict[str, str]:
    """sanitizer 로 빌드한 커널을 워커(계측되지 않은 Python 프로세스)가 로드할 때 덧씌울 환경변수.

    - ``LD_PRELOAD``: ``<cc> -print-file-name=libasan.so`` 가 출력한 경로. ASan 런타임이 프로세스의 첫 라이브러리여야 하므로
      선행 로드한다. 출력이 실제 파일이 아니어도 그대로 넣고 경고한다(그때 워커는 로드에서 죽거나 커널 로드에 실패하고,
      실행은 판정 불가로 기록된다 — ``run_sanitizers``). ``<cc> -print-file-name=libubsan.so`` 가 실제 파일이면 그 뒤에 함께
      선행 로드한다: gcc 의 libubsan 은 따로 된 런타임이고 옵션을 ``/proc/self/environ`` 에서 읽는데, 워커는 커널을 로드하기
      전에 코어 덤프를 끄며(``PR_SET_DUMPABLE=0``) 그 뒤로는 이 파일을 읽을 수 없다. 선행 로드하면 libubsan 이 그 전에 초기화되어
      ``UBSAN_OPTIONS`` 가 실제로 적용된다(clang 처럼 UBSan 이 ASan 런타임 안에 있으면 파일이 없고 더하지 않는다).
    - ``ASAN_OPTIONS``: ``detect_leaks=0`` (CPython 은 종료 때 메모리를 남긴다), ``abort_on_error=1`` (첫 오류에서 SIGABRT),
      ``allocator_may_return_null=1`` (너무 큰 할당은 보고 대신 NULL), ``handle_sigfpe=0`` (정수 0 나눗셈의 SIGFPE 는 메모리
      오류가 아니므로 ASan 보고로 만들지 않는다), ``log_path=<log_dir>/asan``.
    - ``UBSAN_OPTIONS``: ``print_stacktrace=1``, ``halt_on_error=0`` (끝까지 돌며 위치마다 보고), ``log_path=<log_dir>/ubsan``.
    """
    log_dir = Path(log_dir)
    timeout_s = float(_setting(cfg, "build", "compile_timeout_s", DEFAULT_COMPILE_TIMEOUT_S))

    def print_file_name(name: str) -> tuple[str, int]:
        proc = run_isolated([cc, f"-print-file-name={name}"], cwd=log_dir, env=launch_env_minimal(), timeout_s=timeout_s)
        printed = proc.stdout_tail.strip().splitlines()
        return (printed[-1].strip() if printed else ""), proc.rc

    libasan, rc = print_file_name("libasan.so")
    preload = [libasan]
    if libasan and Path(libasan).is_file():
        libubsan, _ = print_file_name("libubsan.so")
        if libubsan and Path(libubsan).is_file():
            preload.append(libubsan)  # libasan 이 먼저다
    else:
        print(f"{_LOG} 경고: `{cc} -print-file-name=libasan.so` 의 출력 {libasan!r} (rc={rc}) 이 파일이 아니다: "
              "ASan 런타임이 선행 로드되지 않으면 sanitizer 실행은 판정 불가로 기록된다", file=sys.stderr)
    return {
        "LD_PRELOAD": ":".join(preload),
        "ASAN_OPTIONS": sanitizer_options({
            "detect_leaks": "0", "abort_on_error": "1", "allocator_may_return_null": "1", "handle_sigfpe": "0",
            "log_path": str(log_dir / ASAN_LOG),
        }),
        "UBSAN_OPTIONS": sanitizer_options({
            "print_stacktrace": "1", "halt_on_error": "0", "log_path": str(log_dir / UBSAN_LOG),
        }),
    }


# ----------------------------------------------------------------------------- sanitizer: 로그 파싱


def asan_reports(text: str) -> dict[str, str]:
    """ASan 오류 보고 → ``{보고를 식별하는 줄: 종류}`` (나온 순서).

    보고 하나는 ``==<pid>==ERROR: AddressSanitizer: <종류> …`` 로 시작해 ``SUMMARY: AddressSanitizer: <종류> <위치>`` 로 끝난다.
    종류는 SUMMARY 의 것을 쓴다(``attempting double-free`` → ``double-free`` 처럼 정리된 이름). 잘린 텍스트도 센다:
    SUMMARY 없이 끝난 보고는 ERROR 줄로, ERROR 줄이 잘려 나간 보고는 SUMMARY 줄로 식별한다.
    """
    reports: dict[str, str] = {}
    pending: tuple[str, str] | None = None  # SUMMARY 를 아직 만나지 못한 (ERROR 줄, 그 줄에서 읽은 종류)
    for line in text.splitlines():
        error = _ASAN_ERROR_RE.match(line)
        if error:
            if pending is not None:
                reports.setdefault(*pending)
            what = error["what"]
            if what.startswith("attempting double-free"):
                kind = "double-free"
            elif what.startswith("attempting free"):
                kind = "bad-free"
            else:
                kind = what.split()[0].rstrip(":")
            pending = (line.strip(), kind)
            continue
        summary = _ASAN_SUMMARY_RE.match(line)
        if summary:
            reports.setdefault(line.strip(), summary["kind"])
            pending = None
    if pending is not None:
        reports.setdefault(*pending)
    return reports


def ubsan_errors(text: str) -> dict[str, str]:
    """UBSan 의 ``<위치>: runtime error: <메시지>`` 줄 → ``{위치: 종류}`` (나온 순서; 같은 위치는 처음 것만).

    위치는 ``<파일 이름>:<줄>:<열>`` 로 정리한다(디렉터리는 뗀다 — 같은 위치가 경로 표기만 달리 나와도 한 번으로 센다).
    """
    errors: dict[str, str] = {}
    for line in text.splitlines():
        m = _UBSAN_ERROR_RE.match(line)
        if m is None:
            continue
        where = m["where"].replace("\\", "/").rsplit("/", 1)[-1]
        kind = next((name for phrase, name in _UBSAN_KINDS if phrase in m["message"]), "undefined-behavior")
        errors.setdefault(where, kind)
    return errors


def sanitizer_log_files(log_dir: Path) -> list[Path]:
    """``log_dir`` 의 sanitizer 로그 파일(``asan.*``, ``ubsan.*``; 이름순). 디렉터리가 없으면 빈 리스트."""
    log_dir = Path(log_dir)
    return sorted(p for prefix in (ASAN_LOG, UBSAN_LOG) for p in log_dir.glob(f"{prefix}.*") if p.is_file())


def parse_sanitizer_logs(log_dir: Path, stderr_text: str = "") -> tuple[int, int, list[str]]:
    """sanitizer 로그와 워커의 stderr 텍스트 → ``(ASan 보고 수, UBSan 'runtime error' 의 서로 다른 위치 수, 종류)``.

    - 어느 런타임이 어느 파일에 쓰는지는 런타임 구성에 달려 있다(ASan 런타임에 UBSan 이 들어 있으면 둘 다 한 파일에 쓴다).
      그래서 모든 로그 파일과 stderr 에서 두 가지 보고를 다 찾고, 같은 보고·같은 위치는 한 번만 센다.
    - 종류는 ASan 의 것이 먼저, 그 뒤에 UBSan 의 것(각각 나온 순서, 중복 없음). 예: ``["heap-buffer-overflow", "signed-integer-overflow"]``.
    """
    texts = [p.read_text(encoding="utf-8", errors="replace") for p in sanitizer_log_files(log_dir)]
    texts.append(stderr_text)
    asan: dict[str, str] = {}
    ubsan: dict[str, str] = {}
    for text in texts:
        for key, kind in asan_reports(text).items():
            asan.setdefault(key, kind)
        for where, kind in ubsan_errors(text).items():
            ubsan.setdefault(where, kind)
    return len(asan), len(ubsan), list(dict.fromkeys([*asan.values(), *ubsan.values()]))


# ----------------------------------------------------------------------------- sanitizer: 빌드·실행


@dataclass
class SanitizeResult:
    """sanitizer 단계의 결과.

    - ``ran``: sanitizer 워커를 띄웠다. ``build_ok``: sanitizer 빌드가 됐다.
    - ``asan_findings`` / ``ubsan_findings``: 보고 수. None = 판정 불가(돌리지 않았다, 빌드 실패, 또는 워커가 끝까지 가지 못했거나
      커널을 로드하지 못했고 그 종류의 보고도 없다). 0 은 커널을 로드해 끝까지 돈 실행에서만 나온다.
    - ``kinds``: 보고의 종류(ASan 이 먼저). ``detail``: 돌리지 않았거나 끝까지 가지 못한 사유(``incomplete:…``,
      ``load_failed:…``; 끝까지 돌았으면 빈 문자열).
    - ``summary``: 재현·진단용 기록(빌드 명령, sanitizer 환경변수, 로그 파일 이름, 워커 실행 요약).
    """

    ran: bool
    build_ok: bool
    asan_findings: int | None
    ubsan_findings: int | None
    kinds: list[str]
    detail: str
    summary: dict


def _not_run(detail: str, summary: dict | None = None) -> SanitizeResult:
    return SanitizeResult(ran=False, build_ok=False, asan_findings=None, ubsan_findings=None, kinds=[], detail=detail,
                          summary=summary or {})


def _ending(proc: ProcResult, timeout_s: float) -> str:
    """끝까지 가지 못한 워커가 어떻게 끝났는가."""
    if proc.timed_out:
        return f"timeout after {timeout_s:g}s"
    if proc.rc < 0:
        return f"signal {proc.signal_name}"
    hint = _last_line(proc.stderr_tail)
    return f"rc={proc.rc}: {hint}" if hint else f"rc={proc.rc}"


def run_sanitizers(cfg: Any, *, code: str, sample_dir: Path, openmp: bool, launch_prefix: list[str], symbol: str,
                   args_header: str | None, tests: list[Path], worker: Callable[..., Any]) -> SanitizeResult:
    """C 커널을 ASan/UBSan 으로 다시 빌드해 check 를 한 번 돌리고 보고를 센다.

    순서: ``san/`` 에 ``build_c(extra_flags=build.sanitize_flags, artifact_name="kernel_san.so")`` →
    ``worker("safety", "check", tests, safety.timeout_s, ["--check-repeats", "1", "--sanitize"], artifact=…,
    env_extra=sanitizer_env(…), max_mem_gb=0)`` (ASan 은 수 TiB 의 가상 주소 공간을 예약하므로 RLIMIT_AS 없이;
    ``--sanitize`` 는 워커가 입출력 버퍼를 가드 영역 없이 정확한 크기로 할당하게 한다) → 로그 파싱(``san/`` 의 로그 파일과
    워커의 stderr 파일 ``safety.stderr.log`` 전체).

    끝까지 간 실행 = 워커 문서가 ``complete`` 이고 커널이 로드됐다(``load_ok`` 가 참). 커널 로드에 실패한 실행(예: 선행 로드가
    맞지 않아 ``undefined symbol: __asan_…``)은 계측된 커널이 돌지 않았으므로 보고가 없는 수는 None 이고 detail 은
    ``load_failed:<사유>`` 다.

    ``safety.enabled`` 가 꺼져 있거나 Linux 가 아니면 아무것도 하지 않는다(``ran=False``). 컴파일러가 없을 때 ``build_c`` 가
    던지는 ``ToolchainMissing`` 은 그대로 올린다.
    """
    if not _setting(cfg, "safety", "enabled", False):
        return _not_run("disabled")
    if not IS_LINUX:
        return _not_run(f"unsupported_platform:{sys.platform}")
    san_dir = Path(sample_dir) / SAN_DIR
    san_dir.mkdir(parents=True, exist_ok=True)
    for stale in sanitizer_log_files(san_dir):  # 이전 실행의 로그를 새 보고로 세지 않게
        stale.unlink()

    flags = list(_setting(cfg, "build", "sanitize_flags", DEFAULT_SANITIZE_FLAGS))
    br = build_c(code, san_dir, cfg.build, openmp=openmp, launch_prefix=launch_prefix, symbol=symbol,
                 args_header=args_header, extra_flags=flags, artifact_name=SAN_ARTIFACT)
    summary: dict[str, Any] = {
        "build": {"ok": br.ok, "error_kind": br.error_kind, "error_detail": br.error_detail, "command": list(br.command)},
    }
    if not br.ok or br.artifact is None:
        return _not_run(f"build_failed:{br.error_kind}: {br.error_detail}", summary)

    env = sanitizer_env(cfg, cfg.build.cc, san_dir)
    timeout_s = float(_setting(cfg, "safety", "timeout_s", DEFAULT_SAFETY_TIMEOUT_S))
    stderr_log = Path(sample_dir) / SAFETY_STDERR_LOG
    stderr_log.unlink(missing_ok=True)  # 이전 실행의 stderr 를 새 보고로 세지 않게
    outcome = worker("safety", "check", tests, timeout_s, list(SAFETY_WORKER_ARGS), artifact=br.artifact, env_extra=env,
                     max_mem_gb=0)
    # 워커의 stderr 는 파일 전체를 본다: 꼬리(stderr_tail)만 보면 보고가 많거나 뒤에 다른 출력이 있을 때 덜 센다.
    # 파일이 없을 때만(워커가 그 파일에 쓰지 않는 호출자) 꼬리를 쓴다 — 꼬리는 그 파일의 끝부분이다.
    try:
        stderr_text = stderr_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        stderr_text = outcome.proc.stderr_tail
    asan, ubsan, kinds = parse_sanitizer_logs(san_dir, stderr_text)
    doc = outcome.doc or {}
    # 커널이 로드되지 않은 실행(load_ok 가 참이 아니다)은 complete 여도 계측된 커널을 돌리지 않았으므로 끝까지 간 실행이 아니다
    loaded = doc.get("load_ok") is True
    complete = bool(doc.get("complete")) and loaded
    if complete:
        detail = ""
    elif doc.get("complete"):
        detail = f"load_failed:{doc.get('load_error') or doc.get('error_detail') or _ending(outcome.proc, timeout_s)}"
    else:
        detail = f"incomplete:{_ending(outcome.proc, timeout_s)}"
    summary.update(env=env, log_files=[p.name for p in sanitizer_log_files(san_dir)], worker=outcome.summary())
    return SanitizeResult(
        ran=True, build_ok=True,
        asan_findings=asan if (asan or complete) else None,
        ubsan_findings=ubsan if (ubsan or complete) else None,
        kinds=kinds, detail=detail, summary=summary,
    )


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """진단용: 린트 플러그인 실행, sanitizer 로그 파싱, sanitizer 환경변수 출력."""
    parser = argparse.ArgumentParser(prog="python -m harness.safety", description="린트·sanitizer 진단 (설계 v3 §12)")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--lint", type=Path, metavar="FILE", help="이 소스 파일에 린트 플러그인을 돌린다 (--language 필요)")
    mode.add_argument("--parse-logs", type=Path, metavar="DIR", help="sanitizer 로그 디렉터리(san/)를 파싱한다")
    parser.add_argument("--language", default=None, help="--lint: 실험 언어 이름")
    parser.add_argument("--task", default="gemm", help="--lint: 과제 이름")
    ns = parser.parse_args(argv)

    if ns.parse_logs is not None:
        asan, ubsan, kinds = parse_sanitizer_logs(ns.parse_logs)
        print(json.dumps({"asan_findings": asan, "ubsan_findings": ubsan, "kinds": kinds,
                          "log_files": [p.name for p in sanitizer_log_files(ns.parse_logs)]}, ensure_ascii=False, indent=2))
        return 0
    if not ns.language:
        parser.error("--lint 에는 --language 가 필요하다")
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 2
    language_cfg = cfg.languages.get(ns.language)
    with tempfile.TemporaryDirectory(prefix="lint_") as tmp:
        res = run_lint(cfg, language=ns.language, backend=language_cfg.backend if language_cfg else "", file=ns.lint.resolve(),
                       task=ns.task, signature={}, out=Path(tmp) / "lint.json", cwd=Path(tmp))
    print(json.dumps(asdict(res), ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
