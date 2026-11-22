"""수정 루프 (설계 v3 §10): 실패한 샘플에 직전 턴의 진단을 돌려주고 다시 묻는다. 턴마다 raw 하나·행 하나가 생긴다.

    PYTHON_EXE -m harness.repair --run-id ID [--config] [--max-turns N] [--tasks a,b] [--languages x,y] [--samples 0,1]
                                 [--feedback compiler|compiler+lint] [--seed-from-feedback compiler] [--yes] [--skip-bench]

전제: 같은 run 의 턴 0 이 ``harness.generate`` + ``harness.run`` 으로 처리돼 있다(직전 턴의 ``result.json`` 이 그 raw 의 것이 아니면
그 샘플은 경고하고 건너뛴다 — ``harness.run`` 을 다시 돌리면 이어진다).

- 범위: ``conditions.feedback`` 이 이 실행의 피드백 조건(``--feedback``, 없으면 설정값)인 샘플. ``args``·``grammar_hint`` 는 raw 의
  것을 그대로 따른다.
- 대상: 직전 턴 행이 ``valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate`` 인 샘플. bench timeout(``pass=1``), 거절, API 오류,
  절단은 대상이 아니다.
- 턴 t = 1..N (N = ``--max-turns``, 없으면 ``run.max_repair_turns``; 설계 상한 3): 대상 수를 보여 주고 확인(generate 와 같은
  게이트) → 샘플마다 대화 조립 → 호출 → raw 저장 → ``harness.run.process_sample`` (bench 는 라운드 1 까지) → 다음 턴. 대상이
  없으면 끝. 턴 루프가 끝나면 이 호출이 만든 행의 bench 라운드 2..R 을 ``harness.run`` 과 같은 함수
  (``run.run_pending_rounds``)로 마치고(decisions.md 2026-11-23), 이 run 의 모든 행으로 ``run.finish_run`` 을 한다 — 수정 턴의
  행은 같은 샘플을 ``harness.run`` 이 처리한 행과 같은 열·같은 result.json 키를 갖는다(``turn``·``turns_to_pass`` 만 다르다).
  그래서 턴 t 의 대상 선택은 라운드 1 까지의 판정을 본다: 라운드 2.. 에서야 실패한 턴(호출 간 불일치 등)은 다음 호출에서
  이어진다. bench 를 하는 호출이면 수정할 과제에 디스크의 run 시작·1스레드 기준선이 없을 때 먼저 잰다(``ensure_baselines``).
- 대화: ``[user: 턴 0 프롬프트, assistant: 턴 0 응답, user: 수정 요청 1, assistant: 턴 1 응답, …, user: 수정 요청 t]``.
  system 은 턴 0 에 보낸 것(``prompts/system.md``) 그대로다. 앞 턴의 수정 요청은 그때 보낸 원문을 raw 에서 읽는다.
- 수정 요청 = ``prompts/repair.md`` 렌더링(``${stage}``, ``${error_type}``, ``${diagnostics}``, ``${lint_section}``).
  주는 정보는 컴파일러·테스트 러너가 줄 법한 것까지다(``build_diagnostics``): 하네스의 패턴 이름(``transposed_output`` 등)과
  참조값은 주지 않는다. ``feedback=compiler+lint`` 는 여기에 린트 절만 더한다(두 조건의 프롬프트는 그 절만 다르다).
- raw: ``…/sample_<i>.turn<t>.json``. 키는 턴 0 과 같고 ``turn``, ``prompt = {system, user(턴 0 원문), messages}``,
  ``prompt_hash = sha256(system + "\\n\\x00\\n" + json.dumps(messages, ensure_ascii=False))[:16]``,
  ``repair = {parent_raw, feedback, diagnostics, lint_included, n_lint_findings}`` 가 다르다.
- 다시 돌리면: 이미 응답이 있는 턴은 다시 묻지 않고 이어서 한다. 그 턴의 raw 가 생성 실패(API 오류·절단·거절)거나 앞 턴이 다시
  생성돼 대화가 이어지지 않으면 그 턴부터 다시 호출한다(generate 와 같은 규칙; 이전 파일은 ``.prev<N>.json`` 으로 보관).
- 제공자·모델은 턴 0 을 만든 것과 같아야 한다(설정·``LLM_PROVIDER``·``LLM_MODEL``). 다르면 호출하지 않고 끝낸다(exit 1) —
  한 샘플의 대화에 두 모델이 섞이거나 dummy run 에 실제 호출이 붙지 않게 한다.
- ``--seed-from-feedback compiler --feedback compiler+lint``: ``fb-compiler`` 조건의 턴 0 raw 를 ``fb-compiler+lint`` 조건
  디렉터리로 복사하고 끝낸다(``conditions.feedback`` 을 바꾸고 ``seeded_from``·``seeded_sha256`` 기록; API 호출 없음). 같은 실패
  샘플에서 두 피드백 조건이 갈라지는 대응 비교용이다. 복사한 턴 0 은 ``harness.run`` 으로 처리한 뒤 이 모듈을
  ``--feedback compiler+lint`` 로 돌린다. 생성에 실패한 원본은 복사하지 않고, 원본이 다시 생성되면 다시 시드할 때 복사본을 바꾼다
  (복사본에 수정 턴이 이미 있으면 거부하고 exit 1; ``seed_feedback``).
- 앞 턴의 지금 행이 수정 대상이 아니면(``harness.run --force`` 의 재분류로 통과 등) 그 샘플은 더 잇지 않는다. 직전 턴의 응답이
  비어 있는 샘플도 수정하지 않는다(빈 assistant 턴은 API 가 거부한다).
- 종료 코드: 0 정상, 1 설정/입력, 2 확인 거부, 3 치명 API 오류, 5 하네스 오류, 130 중단. ``harness.run`` 의 준비·처리·라운드
  단계가 멈추면 그 코드(4 규약, 6 기준선, 7 도구 없음, 5 harness_error)를 그대로 돌려준다.

LLM 코드는 이 프로세스에서 실행하지 않는다(처리는 ``harness.run`` → ``harness.worker`` 서브프로세스).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path, PurePath
from string import Template
from typing import Any

from harness import generate, safety
from harness.build.c import ToolchainMissing
from harness.config import FEEDBACKS, MAX_REPAIR_TURNS, Config, ConfigError, config_snapshot, load_config
from harness.llm_client import LLMClient, LLMClientError
from harness.paths import (
    PROMPTS_DIR, atomic_write_json, built_turn_dir, cond_tag, parse_raw_turn_name, raw_turn_path, rel_to_root,
)
from harness.procutil import install_terminate_handlers
from harness.schema import harness_version, read_result_rows

_LOG = "[repair]"

EXIT_OK = 0
EXIT_INPUT = 1  # 설정·입력 오류(설정 파일, 옵션, raw 없음, 턴 0 미처리)
EXIT_DECLINED = 2  # 확인 거부·비대화형
EXIT_API = 3  # 치명 API 오류(preflight 실패 포함)
EXIT_HARNESS = 5  # 하네스 오류(워커의 harness_error, 예상하지 못한 예외)
EXIT_INTERRUPTED = 130

REPAIR_TEMPLATE: Path = PROMPTS_DIR / "repair.md"
REQUEST_VARIABLES = frozenset({"stage", "error_type", "diagnostics", "lint_section"})
# 샘플(턴) 디렉터리의 파일 이름(설계 v3 §9.7)
RESULT_JSON = "result.json"
BUILD_LOG = "build.log"

DIAGNOSTICS_MAX_CHARS = 4000
TRUNCATION_MARK = "\n[... truncated]"
TRACEBACK_TAIL_LINES = 10
LOG_TAIL_LINES = 30  # 컴파일러 출력에 오류 줄이 없을 때 주는 꼬리
MESSAGE_LINE_CHARS = 500
LINT_MAX_FINDINGS = 20
LINT_HEADING = "## Static analysis findings"

# 수정 요청의 ``${stage}``: 행의 error_stage → 그 단계가 하는 일
STAGE_TEXT: dict[str, str] = {
    "extract": "reading the code block of the reply and checking the function signature",
    "build": "building the code (compiling it, or importing the module)",
    "check": "correctness check on the correctness shapes",
    "bench": "timed run on the performance shape",
}
# 수정 요청의 ``${error_type}``: 행의 error_type → 테스트 러너가 관찰한 수준의 말. 참조와 비교해 하네스가 내린 세부 진단
# (arg_order, numeric_instability)과 추론(race_condition)은 관찰한 사실로 낮춘다 — 진단을 대신해 주면 모델의 수정 능력이 아니라
# 하네스의 분류기를 재게 된다. memory_error 는 ``feedback_error_type`` 이 sanitizer 재분류 전의 판정으로 되돌린다.
ERROR_TYPE_TEXT: dict[str, str] = {
    "signature_mismatch": "signature mismatch",
    "compile_error": "compile error",
    "runtime_crash": "runtime crash",
    "timeout": "timeout",
    "wrong_output": "incorrect output",
    "arg_order": "incorrect output",
    "numeric_instability": "incorrect output",
    "race_condition": "non-deterministic output",
}
# sanitizer 재분류의 error_detail(설계 v3 §9.2): "asan:<종류>; was <이전 type>/<이전 detail>"
_MEMORY_ERROR_WAS_RE = re.compile(r"; was (\w+)/")

# 추출 실패 사유(harness.extract 의 detail 첫 항목) → 풀어 쓴 문장
EXTRACT_REASON_TEXT: dict[str, str] = {
    "no_code_block": "No fenced code block with an implementation was found in the reply.",
    "function_not_found": "The reply does not contain a definition of the required function (a declaration without a body, or a "
                          "function with another name, does not count).",
    "static_linkage": "The required function is defined `static`, so it has no external linkage and cannot be called from outside "
                      "the file.",
    "no_mangle_missing": "The required function is not marked `#[no_mangle]`, so it is not exported under its unmangled name.",
    "param_mismatch:async_def": "The required function is defined with `async def`; it must be an ordinary function.",
}
PARAM_MISMATCH_PREFIX = "param_mismatch:"
# 빌드·로드 단계가 남기는 한 낱말짜리 사유 → 풀어 쓴 문장. 첫 턴 프롬프트가 규칙으로 밝힌 것만 말한다(numba 프롬프트: 모든 컴파일
# 함수의 명시 서명, cache=True 금지, 하네스가 넘기는 배열·크기의 타입).
BUILD_REASON_TEXT: dict[str, str] = {
    "numba_signature_missing": "The required function is not compiled when the module is imported: it must carry the `njit` "
                               "decorator called with the signature string as its first argument.",
    "numba_cache_true": "A Numba decorator in the module sets the `cache` option; `cache=True` is not allowed.",
    "function_not_found": "The module does not define the required function at top level.",
}
# 상세가 "<이름>:<값>" 꼴인 빌드·로드 사유(harness/build/numba.py, harness/kernels/pyfunc.py; decisions.md 2026-11-23)
NUMBA_UNSIGNED_JIT_PREFIX = "numba_unsigned_jit:"
NUMBA_REJECTS_ARGS_PREFIX = "numba_signature_rejects_args:"
_REJECTS_NONE_RE = re.compile(r"^\s*no compiled signature of (?P<fn>\S+) accepts \((?P<types>.*)\)\s*$", re.S)
_REJECTS_AMBIGUOUS_RE = re.compile(r"^\s*ambiguous compiled signatures for \((?P<types>.*)\)\s*$", re.S)
# runtime_crash 상세의 Rust panic(harness/run.py ``with_panic_detail``: "rust_panic:<메시지>")
RUST_PANIC_PREFIX = "rust_panic:"

# 절대 경로(POSIX ``/a/b/kernel.c``, Windows ``C:\\a\\b\\kernel.py``) → 파일 이름. 상대 경로(``kernel.c``, ``src/lib.rs``)는 그대로다.
_ABS_PATH_RE = re.compile(r"""(?<![\w./\\)\]-])(?:[A-Za-z]:[\\/]|/)(?:[^\s:"'`<>()|\\/]+[\\/])+([^\s:"'`<>()|\\/]+)""")
# 파이썬 트레이스백의 프레임 머리 줄(두 칸 들여쓰기). 그 아래 네 칸 들여쓴 줄은 그 프레임의 소스·캐럿 줄이다.
_FRAME_RE = re.compile(r'^  File "(?P<path>[^"]+)", line \d+')
_PATH_SEP_RE = re.compile(r"[\\/]")
HARNESS_DIR = Path(__file__).resolve().parent
# 컴파일러·링커 출력에서 오류가 시작되는 줄
_FIRST_ERROR_RE = re.compile(r"\berror\b|undefined reference")


class RepairError(RuntimeError):
    """하네스가 만든 자료(행·result.json·raw)가 계약과 달라 수정 요청을 만들 수 없다. 모델의 오류가 아니다(exit 5)."""


# ----------------------------------------------------------------------------- 작은 헬퍼


def warn(msg: str) -> None:
    print(f"{_LOG} 경고: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{_LOG} {msg}", file=sys.stderr)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    """JSON 객체 파일. 없거나 읽을 수 없거나 객체가 아니면 None."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def basename_paths(text: str) -> str:
    """절대 경로를 파일 이름만 남긴다(샘플 디렉터리·인터프리터의 위치는 진단이 아니다)."""
    return _ABS_PATH_RE.sub(r"\1", text)


def clip(text: str, limit: int = DIAGNOSTICS_MAX_CHARS) -> str:
    """``limit`` 글자를 넘으면 앞부분만 남기고 잘렸다는 표시를 붙인다(표시까지 ``limit`` 이내)."""
    if len(text) <= limit:
        return text
    return text[: limit - len(TRUNCATION_MARK)].rstrip() + TRUNCATION_MARK


# ----------------------------------------------------------------------------- 진단 (설계 v3 §10)


def compiler_excerpt(log: str) -> str:
    """컴파일러·cargo·링커 출력에서 오류가 처음 나온 줄부터 끝까지. 오류 줄이 없으면 마지막 ``LOG_TAIL_LINES`` 줄."""
    lines = [line.rstrip() for line in basename_paths(log).splitlines()]
    start = next((i for i, line in enumerate(lines) if _FIRST_ERROR_RE.search(line)), None)
    picked = lines[start:] if start is not None else [line for line in lines if line.strip()][-LOG_TAIL_LINES:]
    return "\n".join(picked).strip("\n")


def process_outcome(summary: dict[str, Any]) -> str | None:
    """워커 프로세스가 스스로 끝내지 못했을 때의 사실 한 구절(단계 요약 ``WorkerOutcome.summary()`` 에서). 정상 종료면 None."""
    if summary.get("timed_out"):
        return f"timed out after {float(summary.get('timeout_s') or 0):g} s"
    if summary.get("signal"):
        return f"crashed: signal {summary['signal']}"
    rc = summary.get("rc")
    if isinstance(rc, int) and rc != 0:
        return f"crashed: the process exited with code {rc}"
    if not summary.get("complete"):
        return "crashed: the process exited before the run was complete"
    return None


def is_harness_file(path: str) -> bool:
    """트레이스백 프레임의 파일이 하네스 패키지의 파일인가: 경로의 ``harness/<나머지>`` 가 지금의 하네스 패키지에 있는 파일이다.
    기록한 기계의 저장소 위치와 상관없고, 이름이 harness 인 디렉터리 밑의 커널 파일은 패키지에 없으므로 하네스로 보지 않는다."""
    parts = [part for part in _PATH_SEP_RE.split(path) if part]
    return any(part == "harness" and HARNESS_DIR.joinpath(*parts[i + 1:]).is_file() for i, part in enumerate(parts[:-1]))


def drop_harness_frames(lines: list[str]) -> tuple[list[str], bool, bool]:
    """트레이스백 줄에서 하네스 파일의 프레임(머리 줄과 그 소스·캐럿 줄)을 뺀다 → (남은 줄, 뺀 프레임이 있는가, 다른 프레임이
    남았는가). 테스트 러너가 자기 프레임을 숨기는 것과 같다: 워커·어댑터(``call_kernel``, ``invoke``, ``_build`` …)는 모델이 쓴
    코드가 아니다. 라이브러리·커널의 프레임과 예외 줄은 그대로 둔다."""
    kept: list[str] = []
    skipping = dropped = others = False
    for line in lines:
        frame = _FRAME_RE.match(line)
        if frame:
            skipping = is_harness_file(frame["path"])
            dropped, others = dropped or skipping, others or not skipping
            if skipping:
                continue
        elif skipping and line.startswith("    "):
            continue
        else:
            skipping = False
        kept.append(line)
    return kept, dropped, others


def exception_text(exc: dict[str, Any]) -> str:
    """예외 기록 ``{type, message, traceback}`` → ``raised ValueError: …`` 과 트레이스백 끝 ``TRACEBACK_TAIL_LINES`` 줄.
    트레이스백에서 하네스 파일의 프레임은 먼저 뺀다(``drop_harness_frames``). 하네스 프레임밖에 없던 트레이스백이면 예외 줄만
    준다: ``raised …`` 한 줄(여러 줄 메시지면 나머지 줄을 잇는다). 트레이스백이 없는 기록(예외 없이 난 실패: 반환값 검사 등)은
    ``failed: …`` 한 줄이다."""
    message = str(exc.get("message") or "").strip()
    first = message.splitlines()[0].strip() if message else ""
    lines = [line.rstrip() for line in str(exc.get("traceback") or "").splitlines() if line.strip()]
    if not lines:
        return f"failed: {first}"[:MESSAGE_LINE_CHARS].rstrip(": ")
    head = f"raised {exc.get('type') or 'an exception'}: {first}"[:MESSAGE_LINE_CHARS].rstrip(": ")
    kept, dropped, others = drop_harness_frames(lines)
    if dropped and not others:
        tail = [line.rstrip() for line in basename_paths(message).splitlines()[1:] if line.strip()]
    else:
        tail = [basename_paths(line) for line in kept]
    return "\n".join([head, *("    " + line for line in tail[-TRACEBACK_TAIL_LINES:])])


def reason_text(detail: str) -> str | None:
    """빌드·로드 단계의 사유(행·단계 기록의 ``error_detail``) → 풀어 쓴 문장. 아는 사유가 아니면 None.

    - ``BUILD_REASON_TEXT`` 의 사유는 그 문장.
    - ``numba_unsigned_jit:<이름>``: 그 함수(``<lambda>`` 면 람다)가 명시 서명 없이 컴파일된다.
    - ``numba_signature_rejects_args: …``: 컴파일된 서명이 하네스가 넘기는 인자 타입을 받지 않는다(맞는 것이 없거나 여럿이
      똑같이 맞는다). 그 타입 목록은 프롬프트가 밝힌 계약(배열의 dtype·차원·C-연속, 크기는 Python int)에서 나온 것이다.
    하네스의 사유 이름 자체는 내보내지 않는다."""
    detail = detail.strip()
    if detail in BUILD_REASON_TEXT:
        return BUILD_REASON_TEXT[detail]
    if detail.startswith(NUMBA_UNSIGNED_JIT_PREFIX):
        name = detail[len(NUMBA_UNSIGNED_JIT_PREFIX):].strip()
        subject = "A lambda in the module" if name in ("", "<lambda>") else f"The function `{name}`"
        return (f"{subject} is compiled with `njit` or `jit` without an explicit signature string as the first positional "
                "argument of its decorator. Every compiled function in the module needs one, so that nothing is compiled "
                "during a call.")
    if detail.startswith(NUMBA_REJECTS_ARGS_PREFIX):
        rest = detail[len(NUMBA_REJECTS_ARGS_PREFIX):]
        none = _REJECTS_NONE_RE.match(rest)
        if none:
            return (f"No compiled signature of `{none['fn']}` accepts the argument types the harness passes: "
                    f"({none['types']}).")
        ambiguous = _REJECTS_AMBIGUOUS_RE.match(rest)
        if ambiguous:
            return ("More than one compiled signature of the function matches the argument types the harness passes "
                    f"({ambiguous['types']}), and none of them is a better match than the others.")
        return f"The compiled signatures of the function do not accept the argument types the harness passes: {rest.strip()}"
    return None


def panic_text(summary: dict[str, Any], row: dict[str, Any]) -> str | None:
    """Rust 커널의 panic 한 구절 ``panicked at src/lib.rs:6:20: <메시지>`` (없으면 None). panic=abort 라 프로세스의 끝(abort)은
    그 결과다 — 테스트 러너가 보여 주는 것은 panic 메시지와 위치다. 근거는 그 단계 요약의 ``rust_panic`` (``{message,
    location}``; harness/run.py ``with_panic_detail``)이고, 없으면 행의 상세 ``rust_panic:<메시지>`` 에서 메시지만 읽는다.
    위치의 경로 구분자는 ``/`` 로 맞추고 절대 경로는 파일 이름만 남긴다(기계마다 같은 글이 되게)."""
    found = summary.get("rust_panic")
    if isinstance(found, dict) and found.get("message") is not None:
        message = basename_paths(str(found["message"])).strip()
        location = basename_paths(str(found.get("location") or "")).strip().replace("\\", "/")
        return (f"panicked at {location}: {message}" if location else f"panicked: {message}")[:MESSAGE_LINE_CHARS]
    detail = str(row.get("error_detail") or "")
    if row.get("error_type") == "runtime_crash" and detail.startswith(RUST_PANIC_PREFIX):
        return f"panicked: {basename_paths(detail[len(RUST_PANIC_PREFIX):]).strip()}"[:MESSAGE_LINE_CHARS]
    return None


def bench_record_name(stages: dict[str, Any]) -> str:
    """bench 에서 실패한 행의 단계 기록 이름(``stages`` 의 열쇠이자 ``<이름>.json``). 라운드는 오류가 난 라운드에서 멈추므로
    마지막 라운드의 기록이다: 라운드 1 은 ``bench``, 라운드 r ≥ 2 는 ``bench_r<r>`` (harness/run.py ``round_file``).
    라운드 기록이 없는 결과(라운드 이전 판)는 ``bench``."""
    rounds = stages.get("rounds")
    if isinstance(rounds, list) and rounds and isinstance(rounds[-1], dict) and rounds[-1].get("result_file"):
        return PurePath(str(rounds[-1]["result_file"]).replace("\\", "/")).stem
    return "bench"


def shape_label(dims: Any) -> str:
    """``shape (M=7, N=5, K=3)``. 워커가 크기를 적기 전에 끝난 문서면 ``shape (unknown)``."""
    sizes = ", ".join(f"{name}={value}" for name, value in dict(dims or {}).items())
    return f"shape ({sizes or 'unknown'})"


def output_wrong(run: dict[str, Any]) -> bool:
    """호출 하나(또는 bench 의 final_check)의 출력이 틀렸는가. 판정 이름이 아니라 지표로 본다: 워커는 출력이 맞아도 입력 변조나
    버퍼 밖 쓰기가 있으면 그 호출을 wrong_output 으로 적는다."""
    if run.get("nan_inf"):
        return True
    ratio = run.get("max_tol_ratio")
    if ratio is not None:
        return ratio > 1.0
    error = run.get("max_abs_err")  # 정확 비교(정수 출력) 과제에는 허용오차 배율이 없다
    return error is not None and error > 0


def incorrect_text(runs: list[dict[str, Any]]) -> str:
    """``incorrect output (max abs error 3.1e+00, 35 of 35 elements outside tolerance)``. 가장 많이 틀린 호출의 값으로 쓴다.
    워커가 틀린 원소 수를 남기지 않았으면 그 부분은 뺀다."""
    worst = max(runs, key=lambda run: (run.get("max_abs_err") or 0.0))
    parts: list[str] = []
    if worst.get("max_abs_err") is not None:
        parts.append(f"max abs error {worst['max_abs_err']:.1e}")
    if worst.get("n_fail") is not None and worst.get("n_total") is not None:
        parts.append(f"{worst['n_fail']} of {worst['n_total']} elements outside tolerance")
    return "incorrect output" + (f" ({', '.join(parts)})" if parts else "")


def output_facts(runs: list[dict[str, Any]], *, nondeterministic: bool, inputs_mutated: bool, oob_write: bool) -> list[str]:
    """테스트 러너가 본 사실들(출력, 실행 간 일치, 입력 보존, 버퍼 경계)."""
    facts: list[str] = []
    wrong = [run for run in runs if output_wrong(run)]
    if any(run.get("nan_inf") for run in runs):
        facts.append("non-finite values in the output")
    elif wrong:
        facts.append(incorrect_text(wrong))
    if nondeterministic:
        facts.append("results differ between two identical calls")
    if inputs_mutated:
        facts.append("modified a read-only input")
    if oob_write:
        facts.append("wrote outside the output buffer")
    return facts


def check_outcome(test: dict[str, Any]) -> str:
    """check 결과 문서의 테스트(형태) 하나 → 결과 한 구절."""
    runs = [run for run in (test.get("runs") or []) if isinstance(run, dict)]
    raised = next((run["exception"] for run in runs if run.get("exception")), None)
    if raised:
        return exception_text(raised)
    facts = output_facts(runs, nondeterministic=bool(test.get("nondeterministic")),
                         inputs_mutated=any(run.get("inputs_mutated") for run in runs),
                         oob_write=any(run.get("oob_write") for run in runs))
    if facts:
        return "; ".join(facts)
    return "passed" if test.get("classification") == "pass" else "incorrect output"


def bench_outcome(doc: dict[str, Any]) -> str:
    """bench 결과 문서 → 결과 한 구절."""
    if doc.get("exception"):
        return exception_text(doc["exception"])
    final = doc.get("final_check") if isinstance(doc.get("final_check"), dict) else {}
    facts = output_facts([final], nondeterministic=(doc.get("nondeterministic_runs") or 0) > 0,
                         inputs_mutated=bool(doc.get("inputs_mutated")), oob_write=bool(doc.get("oob_write")))
    return "; ".join(facts) if facts else "incorrect output"


def _stage(stages: dict[str, Any], name: str) -> dict[str, Any]:
    """``result.json`` 의 단계 기록. 그 단계에서 실패한 행인데 기록이 없으면 하네스가 만든 자료가 어긋난 것이다."""
    stage = stages.get(name)
    if not isinstance(stage, dict):
        raise RepairError(f"result.json 의 stages 에 {name!r} 기록이 없다")
    return stage


def _extract_diagnostics(stage: dict[str, Any]) -> str:
    reason = str(stage.get("detail") or "").split(";")[0]
    if reason in EXTRACT_REASON_TEXT:
        return EXTRACT_REASON_TEXT[reason]
    if reason.startswith(PARAM_MISMATCH_PREFIX):
        return f"The signature of the function does not match the required signature: {reason[len(PARAM_MISMATCH_PREFIX):]}."
    return f"The code could not be taken from the reply: {reason or 'no usable code'}."


def _build_diagnostics(row: dict[str, Any], stages: dict[str, Any], sample_dir: Path) -> str:
    build = _stage(stages, "build")
    detail = str(build.get("error_detail") or "")
    if not build.get("ok"):
        if build.get("error_kind") == "timeout":
            return f"The build did not finish: {detail}."
        if build.get("error_kind") == "signature_mismatch":
            return reason_text(detail) or f"The build succeeded, but the required function is missing from the result: {detail}."
        try:
            log = (sample_dir / BUILD_LOG).read_text(encoding="utf-8", errors="replace")
        except OSError:
            log = ""
        if log.strip():
            return "Compiler output:\n" + compiler_excerpt(log)
        return f"The code does not compile: {basename_paths(detail)}"
    # 빌드는 됐고 로드(import 단계)에서 멈췄다
    summary = _stage(stages, "import")
    stopped = panic_text(summary, row) or process_outcome(summary)
    if stopped is not None:
        return f"Loading the built code did not finish: {stopped}."
    doc = read_json(sample_dir / "import.json") or {}
    reason = str(summary.get("error_detail") or doc.get("load_error") or "")
    known = reason_text(reason)  # 하네스가 이름 붙인 로드 사유(Numba 서명 규칙)는 예외 기록이 있어도 그 문장으로 준다
    if known is not None:
        return known
    if doc.get("exception"):
        return "Loading the built code failed: " + exception_text(doc["exception"])
    return f"Loading the built code failed: {basename_paths(reason) or 'no reason was recorded'}"


def _run_diagnostics(stage: str, record: str, row: dict[str, Any], stages: dict[str, Any], sample_dir: Path) -> str:
    """check·bench 의 진단. ``record`` 는 단계 기록의 이름(check 는 ``check``, bench 는 실패한 라운드의 ``bench_record_name``)."""
    summary = _stage(stages, record)
    doc = read_json(sample_dir / f"{record}.json") or {}
    stopped = panic_text(summary, row) or process_outcome(summary)
    running = (doc.get("in_progress") or {}).get("test") if stopped is not None else None
    if stage == "bench":
        outcome = stopped or bench_outcome(doc)
        return ("The correctness check passed on every correctness shape. Timed run on the performance shape:\n"
                f"- {shape_label(doc.get('dims'))}: {outcome}")
    lines = ["Correctness check, one line per shape that was run:"]
    reported = False
    for test in doc.get("tests") or []:
        here = stopped is not None and test.get("name") == running
        reported = reported or here
        lines.append(f"- {shape_label(test.get('dims'))}: {stopped if here else check_outcome(test)}")
    if stopped is not None and not reported:
        lines.append(f"- {stopped} (outside the calls of the function)")
    return "\n".join(lines)


def build_diagnostics(row: dict[str, Any], stages: dict[str, Any], sample_dir: Path) -> str:
    """수정 요청의 ``${diagnostics}``: 실패한 단계에서 컴파일러·테스트 러너가 줄 법한 사실(영어, ``DIAGNOSTICS_MAX_CHARS`` 이내).

    ``row``·``stages`` 는 그 턴 ``result.json`` 의 것, ``sample_dir`` 은 그 턴의 산출물 디렉터리(``build.log``, ``<stage>.json``)다.
    순수 함수다(읽기만 한다).

    - extract: 추출 실패 사유를 풀어 쓴 한 문단.
    - build: 컴파일러·cargo 출력에서 오류가 처음 나온 줄부터, 또는 로드(임포트) 실패의 예외. 경로는 파일 이름만 남긴다.
      Numba 의 서명 규칙 위반(명시 서명 없는 컴파일 함수, ``cache=True``, 하네스의 인자 타입을 받지 않는 서명)은
      ``reason_text`` 의 문장이다.
    - check/bench: 형태별 ``shape (M=…, N=…): <결과>``. bench 는 실패한 라운드(``bench_record_name``)의 기록을 읽는다.
      Rust 커널이 panic 으로 끝났으면 프로세스의 끝 대신 panic 메시지와 위치(``panic_text``)를 준다.

    하네스의 패턴 이름·분류 이름(``classification``, ``pattern``, ``error_detail``)과 참조에서 나온 값은 넣지 않는다.
    """
    stage = str(row.get("error_stage") or "")
    if stage == "extract":
        text = _extract_diagnostics(_stage(stages, "extract"))
    elif stage == "build":
        text = _build_diagnostics(row, stages, Path(sample_dir))
    elif stage == "check":
        text = _run_diagnostics("check", "check", row, stages, Path(sample_dir))
    elif stage == "bench":
        text = _run_diagnostics("bench", bench_record_name(stages), row, stages, Path(sample_dir))
    else:
        raise RepairError(f"진단을 만들 수 없는 error_stage: {stage!r}")
    return clip(text)


def feedback_error_type(row: dict[str, Any]) -> str:
    """수정 요청의 ``${error_type}`` (``ERROR_TYPE_TEXT``). sanitizer 가 재분류한 ``memory_error`` 는 재분류 전의 판정으로 읽는다 —
    수정 요청이 ``safety.enabled`` 에 따라 달라지면 안 된다(sanitizer 는 측정이지 피드백이 아니다)."""
    error_type = str(row.get("error_type") or "")
    if error_type == "memory_error":
        was = _MEMORY_ERROR_WAS_RE.search(str(row.get("error_detail") or ""))
        if was is None:
            raise RepairError(f"memory_error 행의 error_detail 에 이전 판정이 없다: {row.get('error_detail')!r}")
        error_type = was.group(1)
    if error_type not in ERROR_TYPE_TEXT:
        raise RepairError(f"수정 요청으로 옮길 수 없는 error_type: {error_type!r}")
    return ERROR_TYPE_TEXT[error_type]


def render_lint_section(stages: dict[str, Any], feedback: str) -> tuple[str, int | None]:
    """수정 요청의 ``${lint_section}`` 과 거기 실은 린트 메시지 수.

    ``feedback=compiler`` 면 ``("", None)``. ``compiler+lint`` 면 ``"\\n## Static analysis findings\\n\\n<본문>\\n"``: 본문은
    직전 턴 ``stages["lint"]`` 의 메시지 ``- line <n>: <message>`` (최대 ``LINT_MAX_FINDINGS`` 개; ``safety.lint_messages`` 형식),
    린트가 돌았고 없으면 ``None.`` (수 0), 린트가 돌지 않았으면 — 단계가 없거나 ``ok`` 가 거짓이거나 ``tool == "none"`` —
    ``Not available.`` (수 None).
    """
    if feedback != "compiler+lint":
        return "", None
    lint = stages.get("lint")
    if not isinstance(lint, dict) or not lint.get("ok") or lint.get("tool") == safety.NO_TOOL:
        return f"\n{LINT_HEADING}\n\nNot available.\n", None
    result = safety.LintResult(tool=str(lint.get("tool")), findings=list(lint.get("findings") or []),
                               summary=dict(lint.get("summary") or {}), ok=True)
    messages = safety.lint_messages(result, LINT_MAX_FINDINGS)
    body = "\n".join(f"- {message}" for message in messages) if messages else "None."
    return f"\n{LINT_HEADING}\n\n{body}\n", len(messages)


@dataclass(frozen=True)
class RepairRequest:
    """수정 요청 하나: 보낼 글과, raw 의 ``repair`` 에 남길 값."""

    text: str
    diagnostics: str
    lint_included: bool
    n_lint_findings: int | None


def render_request(template: str, row: dict[str, Any], stages: dict[str, Any], sample_dir: Path, feedback: str) -> RepairRequest:
    """수정 요청 템플릿(``load_template`` 이 읽은 ``template``)을 직전 턴의 행·단계 기록으로 채운다."""
    stage = str(row.get("error_stage") or "")
    if stage not in STAGE_TEXT:
        raise RepairError(f"수정 요청으로 옮길 수 없는 error_stage: {stage!r}")
    diagnostics = build_diagnostics(row, stages, sample_dir)
    lint_section, n_lint = render_lint_section(stages, feedback)
    text = Template(template).substitute(stage=STAGE_TEXT[stage], error_type=feedback_error_type(row), diagnostics=diagnostics,
                                         lint_section=lint_section)
    return RepairRequest(text=text.rstrip("\n") + "\n", diagnostics=diagnostics, lint_included=bool(lint_section),
                         n_lint_findings=n_lint)


def load_template(path: Path | None = None) -> str:
    """수정 요청 템플릿(기본 ``prompts/repair.md``). 읽을 수 없거나 변수가 ``REQUEST_VARIABLES`` 와 다르면 ConfigError."""
    path = REPAIR_TEMPLATE if path is None else path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ConfigError(f"수정 요청 템플릿을 읽을 수 없다: {path} ({e})") from None
    template = Template(text)
    if not template.is_valid() or set(template.get_identifiers()) != REQUEST_VARIABLES:
        raise ConfigError(f"{path}: 템플릿 변수는 정확히 {sorted(REQUEST_VARIABLES)} 여야 한다")
    return text


def repair_prompt_hash(system: str, messages: list[dict[str, str]]) -> str:
    """``sha256(system + "\\n\\x00\\n" + json.dumps(messages, ensure_ascii=False))[:16]``."""
    return generate.prompt_hash(system, json.dumps(messages, ensure_ascii=False))


# ----------------------------------------------------------------------------- 샘플(턴의 사슬)


@dataclass
class Sample:
    """샘플 하나의 턴 사슬. ``paths[t]``/``raws[t]`` 는 턴 t 의 raw(턴 0 부터 빈틈없이, 응답이 있는 턴까지)."""

    task: str
    language: str
    sample_idx: int
    paths: list[Path] = field(default_factory=list)
    raws: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_turn(self) -> int:
        return len(self.paths) - 1

    @property
    def conditions(self) -> dict[str, str]:
        return {key: str(value) for key, value in (self.raws[0].get("conditions") or {}).items()}

    @property
    def tag(self) -> str:
        cond = self.conditions
        return cond_tag(cond["args"], cond["grammar_hint"], cond["feedback"])

    def label(self, turn: int | None = None) -> str:
        base = f"{self.task}/{self.language}/sample_{self.sample_idx}"
        return base if not turn else f"{base}/turn_{turn}"


def _turn_zero_files(cfg: Config, run_id: str, tasks: list[str] | None, languages: set[str] | None,
                     samples: set[int] | None) -> list[tuple[str, str, int, Path]]:
    """run 의 턴 0 raw: (과제, 언어, 샘플 번호, 경로) — 필터를 적용하고 과제·언어·조건 디렉터리·번호 순으로."""
    found: list[tuple[str, str, str, int, Path]] = []
    for path in (cfg.paths.raw_dir / run_id).glob("*/*/*/sample_*.json"):
        parsed = parse_raw_turn_name(path.name)
        if parsed is None or parsed[1] != 0:
            continue
        task, language, tag = path.parts[-4], path.parts[-3], path.parts[-2]
        if (tasks is not None and task not in tasks) or (languages is not None and language not in languages) \
                or (samples is not None and parsed[0] not in samples):
            continue
        found.append((task, language, tag, parsed[0], path))
    return [(task, language, idx, path) for task, language, _, idx, path in sorted(found)]


def scan_samples(cfg: Config, run_id: str, *, feedback: str, tasks: list[str] | None = None, languages: set[str] | None = None,
                 samples: set[int] | None = None) -> list[Sample]:
    """피드백 조건이 ``feedback`` 인 샘플들의 턴 사슬. 턴 t ≥ 1 은 응답이 있고(오류·절단·거절이 아니고) 지금의 앞 턴에서 이어진
    raw 까지만 잇는다 — 거기서 끊긴 턴이 다시 호출할 턴이다."""
    out: list[Sample] = []
    for task, language, idx, path in _turn_zero_files(cfg, run_id, tasks, languages, samples):
        raw = read_json(path)
        cond = (raw or {}).get("conditions")
        if raw is None or not isinstance(cond, dict) or not all(key in cond for key in ("args", "grammar_hint", "feedback")):
            warn(f"raw 를 읽을 수 없거나 conditions 가 없어 건너뛴다: {path}")
            continue
        if cond["feedback"] != feedback:
            continue
        if task not in cfg.run.tasks or language not in cfg.languages:
            warn(f"설정에 없는 과제·언어의 raw 라 건너뛴다: {path}")
            continue
        sample = Sample(task=task, language=language, sample_idx=idx, paths=[path], raws=[raw])
        while True:
            nxt = raw_turn_path(cfg.paths.raw_dir, run_id, task, language, sample.tag, idx, sample.last_turn + 1)
            if not nxt.exists() or generate.decide_action(nxt, force=False, no_retry_failed=False)[0] != "skip_ok":
                break
            nxt_raw = read_json(nxt) or {}
            sent = (nxt_raw.get("prompt") or {}).get("messages")
            if not isinstance(sent, list) or sent[:-1] != conversation_so_far(sample):
                # 앞 턴이 다시 생성됐다(generate --force 등): 옛 대화에 붙은 응답을 지금의 사슬에 잇지 않는다
                warn(f"{nxt}: 지금의 앞 턴에서 이어진 대화가 아니다 — 이 턴부터 다시 묻는다")
                break
            sample.paths.append(nxt)
            sample.raws.append(nxt_raw)
        out.append(sample)
    return out


def read_result(cfg: Config, run_id: str, sample: Sample, turn: int) -> tuple[dict[str, Any], Path] | None:
    """그 턴의 (``result.json`` 문서, 산출물 디렉터리). 없거나 그 raw 를 처리한 것이 아니면(``raw_sha256`` 불일치) None."""
    sample_dir = built_turn_dir(cfg.paths.built_dir, run_id, sample.task, sample.language, sample.tag, sample.sample_idx, turn)
    doc = read_json(sample_dir / RESULT_JSON)
    if doc is None or not isinstance(doc.get("row"), dict) or doc.get("raw_sha256") != sha256_file(sample.paths[turn]):
        return None
    return doc, sample_dir


def is_repair_target(row: dict[str, Any]) -> bool:
    """수정 대상인가: ``valid_sample=1 ∧ pass=0 ∧ error_stage ≠ generate``."""
    return bool(row.get("valid_sample")) and not row.get("passed") and row.get("error_stage") != "generate"


def has_reply(raw: dict[str, Any]) -> bool:
    """raw 의 응답 글이 비어 있지 않은가(공백만이면 빈 것이다). 빈 응답은 대화의 assistant 턴으로 보낼 수 없다 — Messages API 는
    마지막이 아닌 메시지의 빈 내용을 거부한다(400, 치명 오류)."""
    return bool(str(raw.get("response_text") or "").strip())


def settled_turn(cfg: Config, run_id: str, sample: Sample) -> tuple[int, str] | None:
    """사슬의 마지막 턴 앞에서 사슬이 끝났어야 할 턴: (턴, 사유). 앞 턴이 모두 지금 처리된 수정 대상이면 None.

    ``harness.run --force`` 의 재분류(하네스 수정, 경쟁 커널의 우연한 통과)로 앞 턴이 통과로 바뀌면 그 뒤의 턴은 낡은 결과다 —
    그 샘플은 더 잇지 않는다(낡은 raw·행은 지우지 않는다)."""
    for turn in range(sample.last_turn):
        result = read_result(cfg, run_id, sample, turn)
        if result is None:
            return turn, "그 raw 를 처리한 result.json 이 없다(`python -m harness.run` 을 먼저 돌려라)"
        if not is_repair_target(result[0]["row"]):
            return turn, "지금의 행이 수정 대상이 아니다(통과, 유효하지 않은 샘플, 생성 단계 실패)"
    return None


def conversation_so_far(sample: Sample) -> list[dict[str, str]]:
    """마지막 턴까지의 대화: 그 턴에 보낸 메시지(턴 0 은 프롬프트 하나)와 그 응답."""
    last = sample.raws[-1]
    prompt = last.get("prompt") or {}
    sent = prompt.get("messages") if sample.last_turn >= 1 else [{"role": "user", "content": prompt.get("user")}]
    if not isinstance(sent, list) or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in sent):
        raise RepairError(f"{sample.paths[-1]}: raw 의 prompt 에서 보낸 대화를 읽을 수 없다")
    return [*({"role": str(m["role"]), "content": m["content"]} for m in sent),
            {"role": "assistant", "content": str(last.get("response_text") or "")}]


# ----------------------------------------------------------------------------- 시드 복사


@dataclass
class SeedCounts:
    """``seed_feedback`` 의 결과."""

    copied: int = 0  # 복사한 수(다시 복사한 것 포함)
    kept: int = 0  # 이미 있어 그대로 둔 수
    refreshed: int = 0  # 원본이 다시 생성돼 옛 복사본을 보관하고 다시 복사한 수(copied 에 포함)
    failed_source: int = 0  # 원본이 생성 실패(API 오류·절단·거절)라 복사하지 않은 수
    refused: int = 0  # 원본이 다시 생성됐는데 복사본에 수정 턴이 있어 그대로 둔 수


_SEED_KEYS = ("conditions", "seeded_from", "seeded_sha256")


def _seeded_from_current(copy: dict[str, Any], source_raw: dict[str, Any], source_sha256: str) -> bool:
    """복사본이 지금의 원본에서 만든 것인가. ``seeded_sha256`` (복사할 때 원본 파일의 sha256)으로 본다 — 복사본을 고쳤어도 원본이
    그대로면 참이다. 그 키가 없는 옛 복사본은 내용(조건·시드 키 제외)을 원본과 비교한다."""
    if "seeded_sha256" in copy:
        return copy["seeded_sha256"] == source_sha256
    return {k: v for k, v in copy.items() if k not in _SEED_KEYS} == {k: v for k, v in source_raw.items() if k not in _SEED_KEYS}


def seed_feedback(cfg: Config, run_id: str, *, source: str, target: str, tasks: list[str] | None, languages: set[str] | None,
                  samples: set[int] | None) -> SeedCounts:
    """``source`` 피드백 조건의 턴 0 raw 를 ``target`` 조건 디렉터리로 복사한다(``SeedCounts``).

    복사본은 ``conditions.feedback`` 만 ``target`` 으로 바뀌고 ``seeded_from`` (원본 경로)과 ``seeded_sha256`` (그때 원본 파일의
    sha256)이 붙는다. 두 조건이 같은 턴 0 응답에서 갈라져야 대응 비교가 성립한다:

    - 원본이 생성 실패(generate 가 다시 호출할 raw: API 오류·절단·거절)면 복사하지 않는다 — generate 로 채운 뒤 다시 시드한다.
    - 이미 있는 복사본은 원본이 그대로면 덮어쓰지 않는다(복사본을 고쳤어도 그대로 둔다). 이 원본의 복사본이 아닌 파일도 그대로 둔다.
    - 복사한 뒤 원본이 다시 생성됐으면, 복사본에 수정 턴이 없을 때만 옛 복사본을 ``.prev<N>.json`` 으로 보관하고 다시 복사한다.
      수정 턴이 있으면 두 이력을 섞지 않게 그대로 두고 거부로 센다.
    """
    counts = SeedCounts()
    for task, language, idx, path in _turn_zero_files(cfg, run_id, tasks, languages, samples):
        raw = read_json(path)
        cond = (raw or {}).get("conditions")
        if raw is None or not isinstance(cond, dict) or cond.get("feedback") != source:
            continue
        source_rel = rel_to_root(path)
        if generate.decide_action(path, force=False, no_retry_failed=False)[0] != "skip_ok":
            warn(f"{source_rel}: 생성에 실패한 raw 라(API 오류·절단·거절) 복사하지 않는다 — `python -m harness.generate` 로 채운 뒤 다시 시드하라")
            counts.failed_source += 1
            continue
        dest = raw_turn_path(cfg.paths.raw_dir, run_id, task, language,
                             cond_tag(str(cond.get("args")), str(cond.get("grammar_hint")), target), idx, 0)
        source_sha256 = sha256_file(path)
        if dest.exists():
            copy = read_json(dest) or {}
            if copy.get("seeded_from") != source_rel:
                warn(f"{rel_to_root(dest)} 가 이미 있다(이 원본의 복사본이 아니다) — 덮어쓰지 않는다")
                counts.kept += 1
                continue
            if _seeded_from_current(copy, raw, source_sha256):
                counts.kept += 1
                continue
            later = [p for p in dest.parent.glob(f"sample_{idx}.turn*.json") if (parse_raw_turn_name(p.name) or (idx, 0))[1] >= 1]
            if later:
                warn(f"{rel_to_root(dest)}: 원본 {source_rel} 이 복사한 뒤 다시 생성됐는데 이 복사본에 수정 턴이 {len(later)}개 있다 — "
                     "두 이력을 섞지 않게 그대로 둔다(그 조건의 수정 턴을 치운 뒤 다시 시드하라)")
                counts.refused += 1
                continue
            archived = generate.archive_previous(dest)
            warn(f"{rel_to_root(dest)}: 원본이 복사한 뒤 다시 생성됐다 — 옛 복사본을 {archived.name if archived else '?'} 로 보관하고 다시 복사한다")
            counts.refreshed += 1
        atomic_write_json(dest, {**raw, "conditions": {**cond, "feedback": target}, "seeded_from": source_rel,
                                 "seeded_sha256": source_sha256})
        counts.copied += 1
    return counts


# ----------------------------------------------------------------------------- 실행


@dataclass
class Summary:
    calls: int = 0
    failed: int = 0
    passed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    empty_replies: int = 0  # 직전 턴의 응답이 비어 수정하지 않은 샘플

    def line(self, run_id: str, feedback: str) -> str:
        return (f"{_LOG} run_id={run_id} feedback={feedback} calls={self.calls} failed={self.failed} "
                f"passed_after_repair={self.passed} tokens_in={self.tokens_in} tokens_out={self.tokens_out} "
                f"skipped_empty_reply={self.empty_replies}")


@dataclass
class Session:
    """수정 루프 한 번의 실행에 공통인 것들."""

    cfg: Config
    run_id: str
    feedback: str
    template: str  # 수정 요청 템플릿(``load_template``)
    client: LLMClient
    run: Any  # harness.run 모듈(``_run_api``)
    ctx: Any  # ``run.setup_run`` 이 돌려준 RunContext
    snapshot: dict[str, Any]
    version: str
    summary: Summary = field(default_factory=Summary)
    # 샘플 번호(``samples`` 의 인덱스) → 그 샘플의 마지막 턴의 (result.json 문서, 산출물 디렉터리). 처리된 턴만 들어 있다.
    results: dict[int, tuple[dict[str, Any], Path]] = field(default_factory=dict)
    # 이 호출이 처리한 턴의 (행, result.json 경로) — 처리 순서대로. 턴 루프 뒤에 ``run.run_pending_rounds`` 에 넘긴다.
    produced: list[tuple[Any, Path]] = field(default_factory=list)


def ensure_baselines(run: Any, ctx: Any, tasks: Any) -> None:
    """수정할 샘플이 있는 과제(``tasks``)에 run 시작·1스레드 기준선이 없으면 잰다(bench 를 하는 호출에서만).

    ``run.setup_run`` 은 준비할 때 처리할 raw 가 없는 과제의 기준선을 다시 재지 않고 디스크의 것을 읽는다
    (``adopt_existing_baselines``) — 수정 턴의 raw 는 준비 뒤에 생기므로 지금의 턴들이 모두 최신이면 그 경우다. 디스크에 기준선이
    없으면(그 과제의 앞선 처리가 모두 ``--skip-bench`` 였거나 통과한 행이 없었다) 수정 턴의 행만 ``baseline_run_time_ms``·
    ``baseline_time_1t_ms`` 가 비게 된다. ``harness.run`` 은 같은 raw 를 처리할 때 기준선을 재므로, 같은 함수
    (``run.prepare_baselines``)로 재서 두 경로의 행이 같은 열을 채우게 한다. 이미 있는 기준선은 다시 재지 않는다(이미 쓴 행의
    증빙과 어긋나지 않게). 예외(``BaselineError``·``HarnessError``·``ToolchainMissing``)는 그대로 올린다."""
    if ctx.skip_bench:
        return
    for task in sorted(tasks):
        tctx = ctx.tasks.get(task)
        if tctx is None or (tctx.baseline_run is not None and tctx.baseline_1t is not None):
            continue
        info(f"{task}: 디스크에 run 시작·1스레드 기준선이 없어 잰다 (수정 턴의 행이 쓴다)")
        run.prepare_baselines(ctx, tctx, reuse=False)


def _run_api() -> Any:
    """``harness.run`` (설계 v3 §9.1 의 재사용 API). 쓰는 자리에서 임포트한다: 시드 복사와 ``--help`` 는 실행 단계 없이 돈다."""
    from harness import run

    return run


def _csv_list(text: str | None) -> list[str] | None:
    return [item.strip() for item in text.split(",") if item.strip()] if text else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.repair", description="수정 루프: 실패한 샘플에 진단을 돌려주고 다시 묻는다 (설계 v3 §10)")
    parser.add_argument("--run-id", required=True, help="results/raw/<run_id> 의 run (턴 0 이 generate + run 으로 처리돼 있어야 한다)")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--max-turns", type=int, default=None, help=f"수정 턴 수 (기본 run.max_repair_turns; 0–{MAX_REPAIR_TURNS})")
    parser.add_argument("--tasks", default=None, help="쉼표 구분 과제 필터")
    parser.add_argument("--languages", default=None, help="쉼표 구분 언어 필터")
    parser.add_argument("--samples", default=None, help="쉼표 구분 sample_idx 필터 (예: 0,1)")
    parser.add_argument("--feedback", choices=FEEDBACKS, default=None, help="수정할 샘플의 피드백 조건 (기본 run.conditions.feedback)")
    parser.add_argument("--seed-from-feedback", choices=FEEDBACKS, default=None,
                        help="이 조건의 턴 0 raw 를 --feedback 조건으로 복사하고 끝낸다 (API 호출 없음)")
    parser.add_argument("--yes", action="store_true", help="실제 API 호출 확인 프롬프트 생략")
    parser.add_argument("--skip-bench", action="store_true", help="새 턴을 check 까지만 처리 (bench·대응 기준선 생략)")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    # 새 턴의 처리는 워커를 띄운다: SIGTERM/SIGHUP 도 KeyboardInterrupt 로 받아 워커 그룹을 정리하고 끝낸다(harness.run 과 같다).
    install_terminate_handlers()
    try:
        return _main(ns)
    except RepairError as e:
        print(f"{_LOG} 하네스 오류 — 중단한다: {e}", file=sys.stderr)
        return EXIT_HARNESS
    except KeyboardInterrupt as e:
        print(f"{_LOG} 중단됨 ({e or 'SIGINT'}) (exit {EXIT_INTERRUPTED})", file=sys.stderr)
        return EXIT_INTERRUPTED


def _main(ns: argparse.Namespace) -> int:
    try:
        cfg = load_config(ns.config, overrides={"feedback": ns.feedback})
        samples_filter = {int(s) for s in _csv_list(ns.samples) or []} if ns.samples else None
    except (ConfigError, ValueError) as e:
        print(f"{_LOG} 설정·옵션 오류: {e}", file=sys.stderr)
        return EXIT_INPUT
    feedback = cfg.run.conditions.feedback
    tasks = _csv_list(ns.tasks)
    languages = set(_csv_list(ns.languages) or []) if ns.languages else None
    run_id = ns.run_id
    if not (cfg.paths.raw_dir / run_id).is_dir():
        print(f"{_LOG} raw 디렉터리가 없다: {cfg.paths.raw_dir / run_id} — 먼저 `python -m harness.generate --run-id {run_id}`", file=sys.stderr)
        return EXIT_INPUT

    if ns.seed_from_feedback is not None:
        if ns.seed_from_feedback == feedback:
            print(f"{_LOG} --seed-from-feedback 과 --feedback 이 같다({feedback}): 복사할 조건이 없다", file=sys.stderr)
            return EXIT_INPUT
        seeded = seed_feedback(cfg, run_id, source=ns.seed_from_feedback, target=feedback, tasks=tasks, languages=languages,
                               samples=samples_filter)
        if seeded.copied + seeded.kept + seeded.failed_source + seeded.refused == 0:
            print(f"{_LOG} feedback={ns.seed_from_feedback} 조건의 턴 0 raw 가 없다 (run_id={run_id})", file=sys.stderr)
            return EXIT_INPUT
        info(f"턴 0 raw {seeded.copied}개를 feedback={feedback} 조건으로 복사했다(이미 있던 것 {seeded.kept}개, 원본이 다시 생성돼 "
             f"다시 복사한 것 {seeded.refreshed}개, 생성 실패라 건너뛴 원본 {seeded.failed_source}개). 다음: "
             f"`python -m harness.run --run-id {run_id}` 로 처리한 뒤 `python -m harness.repair --run-id {run_id} --feedback {feedback}`")
        if seeded.refused:
            print(f"{_LOG} 원본이 다시 생성됐는데 수정 턴이 이미 있는 복사본 {seeded.refused}개는 그대로 뒀다 (위 경고)", file=sys.stderr)
            return EXIT_INPUT
        return EXIT_OK

    max_turns = cfg.run.max_repair_turns if ns.max_turns is None else ns.max_turns
    if not 0 <= max_turns <= MAX_REPAIR_TURNS:
        print(f"{_LOG} --max-turns 는 0 이상 {MAX_REPAIR_TURNS} 이하여야 한다 (현재 {max_turns})", file=sys.stderr)
        return EXIT_INPUT
    try:
        template = load_template()
    except ConfigError as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return EXIT_INPUT

    samples = scan_samples(cfg, run_id, feedback=feedback, tasks=tasks, languages=languages, samples=samples_filter)
    if not samples:
        print(f"{_LOG} feedback={feedback} 조건의 턴 0 raw 가 없다 (run_id={run_id}; 필터와 --feedback 을 확인하라)", file=sys.stderr)
        return EXIT_INPUT
    # 한 샘플의 대화는 한 제공자·한 모델이어야 한다(dummy 로 만든 run 을 실제 API 로 잇거나 그 반대가 되면 행이 섞인다)
    made_with = {(str(sample.raws[0].get("provider")), str(sample.raws[0].get("model_requested"))) for sample in samples}
    if made_with != {(cfg.model.provider, cfg.model.name)}:
        print(f"{_LOG} 이 run 의 턴 0 은 (provider, model) = {sorted(made_with)} 로 만들어졌는데 지금 설정은 "
              f"({cfg.model.provider!r}, {cfg.model.name!r}) 이다 — 환경변수 LLM_PROVIDER·LLM_MODEL 로 맞춰라", file=sys.stderr)
        return EXIT_INPUT
    if max_turns == 0:
        info("수정 턴 수가 0 이다(run.max_repair_turns 또는 --max-turns): 할 일이 없다")
        return EXIT_OK

    # 직전 턴의 결과: 샘플 → (result.json 문서, 산출물 디렉터리). 처리되지 않은 샘플과 앞 턴에서 끝났어야 할 사슬은 한 번 알리고 뺀다.
    results: dict[int, tuple[dict[str, Any], Path]] = {}
    n_settled = 0
    for i, sample in enumerate(samples):
        result = read_result(cfg, run_id, sample, sample.last_turn)
        if result is None:
            warn(f"{sample.label(sample.last_turn)}: 그 raw 를 처리한 result.json 이 없다 — `python -m harness.run --run-id {run_id}` 를 먼저 돌려라")
            continue
        settled = settled_turn(cfg, run_id, sample)
        if settled is not None:
            n_settled += 1
            warn(f"{sample.label()}/turn_{settled[0]}: {settled[1]} — 턴 {settled[0] + 1}..{sample.last_turn} 은 낡은 결과라 "
                 "이 샘플은 더 잇지 않는다")
            continue
        results[i] = result
    if not results and not n_settled:
        print(f"{_LOG} 처리된 샘플이 하나도 없다: 턴 0 을 harness.run 으로 먼저 처리해야 한다", file=sys.stderr)
        return EXIT_INPUT

    empty_replies: set[int] = set()  # 직전 턴의 응답이 비어 다시 물을 수 없는 샘플(한 번만 알린다)

    def targets_of(turn: int) -> list[int]:
        out: list[int] = []
        for i, sample in enumerate(samples):
            if sample.last_turn != turn - 1 or i not in results or not is_repair_target(results[i][0]["row"]):
                continue
            if not has_reply(sample.raws[-1]):
                if i not in empty_replies:
                    empty_replies.add(i)
                    warn(f"{sample.label(sample.last_turn)}: 응답이 비어 있어 대화를 이을 수 없다 — 이 샘플은 수정하지 않는다")
                continue
            out.append(i)
        return out

    target_tasks = {samples[i].task for turn in range(1, max_turns + 1) for i in targets_of(turn)}
    if not target_tasks:
        info(f"수정할 샘플이 없다 (샘플 {len(samples)}개, 턴 상한 {max_turns})")
        return EXIT_OK

    provider = cfg.model.provider
    try:
        client = LLMClient(cfg)
    except LLMClientError as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return EXIT_INPUT
    version = harness_version()
    try:
        client.preflight()
    except Exception as e:  # noqa: BLE001 — 인증/모델명 오류를 호출 전에 드러낸다
        print(f"{_LOG} preflight 실패: {type(e).__name__}: {e}", file=sys.stderr)
        return EXIT_API if provider != "dummy" else EXIT_INPUT
    if provider != "dummy" and version.startswith("nogit-"):
        warn(f"git 저장소가 아니라 harness_version={version} 이다 (원칙 4: 재현성을 위해 git 으로 관리하라)")

    run = _run_api()
    try:
        ctx = run.setup_run(cfg, run_id, skip_bench=ns.skip_bench, tasks=tasks, languages=languages, samples=samples_filter)
    except run.RunSetupError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return e.exit_code
    try:
        ensure_baselines(run, ctx, target_tasks)
    except ToolchainMissing as e:
        print(f"{_LOG} 중단: 도구 없음 (기준선 빌드): {e}", file=sys.stderr)
        return run.EXIT_TOOLCHAIN
    except run.BaselineError as e:
        print(f"{_LOG} 중단: {e}", file=sys.stderr)
        return run.EXIT_BASELINE
    except run.HarnessError as e:
        print(f"{_LOG} harness_error (기준선): {e}", file=sys.stderr)
        return EXIT_HARNESS

    session = Session(cfg=cfg, run_id=run_id, feedback=feedback, template=template, client=client, run=run, ctx=ctx,
                      snapshot=config_snapshot(cfg), version=version, results=results)
    rc = EXIT_OK
    for turn in range(1, max_turns + 1):
        targets = targets_of(turn)
        if not targets:
            if not any(sample.last_turn >= turn for sample in samples):
                break
            continue
        if provider == "dummy":
            info(f"턴 {turn}: {len(targets)} dummy calls (run_id={run_id})")
        else:
            info(f"턴 {turn}: 수정 대상 {len(targets)}개")
            if not generate.confirm_calls(len(targets), cfg, yes=ns.yes, log=_LOG):
                rc = EXIT_DECLINED
                break
        for i in targets:
            label = samples[i].label(turn)
            try:
                rc = repair_sample(session, samples[i], i, turn)
            except ToolchainMissing as e:
                print(f"{_LOG} 도구 없음 — 행을 쓰지 않고 중단한다 ({label}): {e}", file=sys.stderr)
                return run.EXIT_TOOLCHAIN
            except (RepairError, run.HarnessError) as e:
                print(f"{_LOG} 하네스 오류 — 중단한다 ({label}):\n{e}", file=sys.stderr)
                return EXIT_HARNESS
            except Exception:  # noqa: BLE001 — 하네스의 버그를 모델 오류로 남기지 않고 멈춘다
                print(f"{_LOG} 처리 중 예외 ({label}):\n{traceback.format_exc()}", file=sys.stderr)
                return EXIT_HARNESS
            if rc != EXIT_OK:
                break
        if rc != EXIT_OK:
            break

    # 이 호출이 만든 턴의 bench 라운드 2..R 을 harness.run 과 같은 함수로 마친다(2026-11-12 규약, decisions.md 2026-11-23).
    # 확인 거부·치명 API 오류로 멈췄어도 처리한 턴까지는 마친다. 도구 없음·하네스 오류는 harness.run 과 같은 종료 코드다.
    try:
        run.run_pending_rounds(ctx, session.produced)
    except ToolchainMissing as e:
        print(f"{_LOG} 도구 없음 — bench 라운드를 멈춘다: {e}", file=sys.stderr)
        return run.EXIT_TOOLCHAIN
    except run.HarnessError as e:
        print(f"{_LOG} harness_error — {e}", file=sys.stderr)
        return EXIT_HARNESS
    except Exception:  # noqa: BLE001 — 하네스의 버그를 행으로 남기지 않고 멈춘다
        print(f"{_LOG} bench 라운드 중 예외:\n{traceback.format_exc()}", file=sys.stderr)
        return EXIT_HARNESS
    # 라운드 2.. 에서 실패한 턴(호출 간 불일치 등)은 통과가 아니다: 라운드까지 마친 행으로 센다
    session.summary.passed = sum(bool(row.passed) for row, _ in session.produced)

    # 이 run 의 모든 행(턴 0 포함)으로 drift·model_served 를 보정한다 — harness.run 을 필터 없이 돌렸을 때와 같은 기준이다.
    rows = read_result_rows(cfg.paths.built_dir / run_id)
    run.finish_run(ctx, [(row, run.result_json_path(row)) for row in rows])
    session.summary.empty_replies = len(empty_replies)
    print(session.summary.line(run_id, feedback))
    return rc


def repair_sample(session: Session, sample: Sample, index: int, turn: int) -> int:
    """샘플 하나의 턴 ``turn``: 수정 요청 → 호출 → raw 저장 → 처리. 치명 API 오류면 ``EXIT_API``, 아니면 ``EXIT_OK``.

    ``index`` 는 ``session.results`` 의 열쇠다. 그 샘플의 직전 턴 결과를 꺼내 쓰고, 새 턴에 응답이 있으면 그 턴의 결과로 바꿔 둔다
    (응답이 없으면 — API 오류·절단·거절 — 비워 둔다: 이번 실행에서는 더 잇지 않는다).
    """
    cfg, run, run_id, summary = session.cfg, session.run, session.run_id, session.summary
    doc, sample_dir = session.results.pop(index)
    request = render_request(session.template, doc["row"], doc.get("stages") or {}, sample_dir, session.feedback)
    first = sample.raws[0].get("prompt") or {}
    system, user = first.get("system"), first.get("user")
    if not isinstance(system, str) or not isinstance(user, str):
        raise RepairError(f"{sample.paths[0]}: raw 에 prompt.system·prompt.user 가 없다")
    messages = [*conversation_so_far(sample), {"role": "user", "content": request.text}]
    label = sample.label(turn)

    resp = session.client.complete(system, user, messages=messages, dummy_key=(sample.task, sample.language, sample.sample_idx, turn))
    raw = generate.build_raw(cfg, run_id=run_id, task=sample.task, lang=cfg.languages[sample.language], sample_idx=sample.sample_idx,
                             turn=turn, conditions=sample.conditions, prompt={"system": system, "user": user, "messages": messages},
                             prompt_hash=repair_prompt_hash(system, messages), resp=resp, snapshot=session.snapshot,
                             version=session.version)
    raw["repair"] = {"parent_raw": rel_to_root(sample.paths[-1]), "feedback": session.feedback, "diagnostics": request.diagnostics,
                     "lint_included": request.lint_included, "n_lint_findings": request.n_lint_findings}
    path = raw_turn_path(cfg.paths.raw_dir, run_id, sample.task, sample.language, sample.tag, sample.sample_idx, turn)
    archived = generate.archive_previous(path)
    atomic_write_json(path, raw)
    summary.calls += 1
    summary.tokens_in += resp.tokens_in
    summary.tokens_out += resp.tokens_out
    note = f" (이전 파일 → {archived.name})" if archived else ""
    if resp.error is not None:
        summary.failed += 1
        info(f"{label}: error={resp.error}{note}")
    else:
        info(f"{label}: stop={resp.stop_reason} model={resp.model_served} tokens_in={resp.tokens_in} tokens_out={resp.tokens_out} "
             f"latency={resp.latency_ms:.0f}ms{note}")

    row, status = run.process_sample(session.ctx, session.ctx.tasks[sample.task], path)
    if row is None:
        raise RepairError(f"{label}: harness.run 이 방금 쓴 raw 를 처리하지 않았다 (status={status}): {path}")
    session.produced.append((row, run.result_json_path(row)))
    if resp.error is None and resp.stop_reason in generate.OK_STOP_REASONS:  # 응답이 있는 턴만 사슬에 잇는다(scan_samples 와 같은 규칙)
        sample.paths.append(path)
        sample.raws.append(raw)
        result = read_result(cfg, run_id, sample, turn)
        if result is None:
            raise RepairError(f"{label}: 처리한 턴의 result.json 을 읽을 수 없다")
        session.results[index] = result
    if resp.error is not None and resp.error.startswith("fatal:"):
        print(f"{_LOG} 치명 오류 — 중단한다 (exit {EXIT_API}). raw 저장: {rel_to_root(path)}", file=sys.stderr)
        return EXIT_API
    return EXIT_OK


if __name__ == "__main__":
    # 진단은 한국어다: 파이프·파일로 나갈 때 로캘 코드페이지가 아니라 UTF-8 로 남게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
