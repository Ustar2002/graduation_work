"""결과 행 스키마와 CSV 입출력 (설계 §11).

- ``RunRow``: 결과 행 dataclass. 필드명은 ``passed`` 이고 CSV 열 이름은 ``pass`` 다
  (``FIELD_TO_COLUMN`` / ``COLUMN_TO_FIELD``). 그 외 필드명 == 열 이름.
- ``COLUMNS``: 열 순서(고정). ``list(row.to_csv_dict().keys()) == COLUMNS`` 가 항상 성립한다.
- 직렬화: 불리언 ``1``/``0``, NA(None) 빈 문자열, 리스트/딕트 JSON 문자열.
- ``ROW_KEY``: 행의 유일 키. ``upsert_row`` 는 같은 키의 행을 교체한다(전체를 tmp에 다시 쓰고 ``os.replace``).
- ``result.json`` 이 진실의 원천이다: ``rebuild_csv`` 는 built 디렉터리의 모든 ``result.json`` 의
  최상위 ``"row"`` 딕트에서 CSV를 다시 만든다.
- ``harness_version()``: ``git describe --always --dirty`` (dirty면 ``+<소스해시12>`` 병기);
  git 저장소가 아니면 ``nogit-<소스해시12>``.
"""

from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import io
import json
import shutil
import sys
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from harness.paths import ROOT, atomic_write_text
from harness.procutil import run_isolated

_LOG = "[schema]"


class SchemaError(ValueError):
    """CSV 헤더 불일치 등 스키마 오류."""


# 열 순서(설계 §11; 고정)
COLUMNS: list[str] = [
    "run_id", "created_at", "task", "language", "backend", "args", "grammar_hint", "feedback", "sample_idx", "turn",
    "provider", "model", "model_served", "sampling", "effort", "max_tokens", "temperature", "seed", "prompt_hash",
    "stop_reason", "stop_details_category",
    "tokens_in", "tokens_out", "tokens_thinking", "tokens_text", "n_thinking_blocks", "latency_ms", "api_error",
    "valid_sample",
    "extract_ok", "signature_ok", "signature_exact", "build_ok", "build_time_ms", "n_warnings", "undefined_symbols",
    "contract_violation",
    "pass", "turns_to_pass", "error_type", "error_stage", "error_detail", "inputs_mutated", "oob_write",
    "nondeterministic_runs", "nondeterministic_bitwise_runs",
    "check_max_abs_err", "check_max_tol_ratio", "check_tests",
    "bench_ran", "time_median_ms", "time_mean_ms", "time_min_ms", "time_max_ms", "time_cv", "time_samples_ms",
    "warmup", "repeats", "repeats_reduced", "gflops",
    "baseline_time_ms", "baseline_time_cv", "baseline_samples_ms", "baseline_run_time_ms", "baseline_time_1t_ms",
    "baseline_drift_pct", "speedup_vs_baseline", "speedup_vs_baseline_1t",
    "cpu_mhz_median", "cpu_mhz_min", "throttle_events", "temp_c_start", "temp_c_end",
    "compiler", "compiler_version", "cflags", "march_resolved", "threads", "cores", "omp_env",
    "governor", "governor_ok", "epp", "no_turbo", "load_1m", "load_ok", "pinned_busy_pct", "sibling_busy_pct",
    "taskset_used", "protocol_ok", "protocol_flags", "protocol_checked_at",
    "hostname", "cpu_model", "python_version", "numpy_version", "numpy_blas", "sdk_version", "test_seed", "test_hash",
    "env_hash",
    "asan_findings", "ubsan_findings", "lint_findings_total", "lint_memory", "lint_numeric", "lint_concurrency",
    "lint_shape_dtype", "lint_api_misuse", "lint_performance", "perf_summary", "roofline_pct",
    "raw_path", "built_path", "harness_version",
]

FIELD_TO_COLUMN: dict[str, str] = {"passed": "pass"}
COLUMN_TO_FIELD: dict[str, str] = {v: k for k, v in FIELD_TO_COLUMN.items()}

ROW_KEY: tuple[str, ...] = ("run_id", "task", "language", "args", "grammar_hint", "feedback", "sample_idx", "turn")

ERROR_STAGES = ("", "generate", "extract", "build", "check", "bench")
ERROR_TYPES = (
    "", "compile_error", "runtime_crash", "timeout", "wrong_output", "numeric_instability", "arg_order",
    "memory_error", "race_condition", "signature_mismatch",
)


def column_of(field_name: str) -> str:
    """dataclass 필드명 → CSV 열 이름."""
    return FIELD_TO_COLUMN.get(field_name, field_name)


def field_of(column: str) -> str:
    """CSV 열 이름 → dataclass 필드명."""
    return COLUMN_TO_FIELD.get(column, column)


# 역직렬화용 필드 종류. 여기 없는 필드는 문자열이다.
_BOOL_FIELDS = frozenset({
    "valid_sample", "extract_ok", "signature_ok", "signature_exact", "build_ok", "passed", "inputs_mutated",
    "oob_write", "bench_ran", "repeats_reduced", "governor_ok", "load_ok", "taskset_used", "protocol_ok",
})
_INT_FIELDS = frozenset({
    "sample_idx", "turn", "max_tokens", "seed", "tokens_in", "tokens_out", "tokens_thinking", "tokens_text",
    "n_thinking_blocks", "n_warnings", "turns_to_pass", "nondeterministic_runs", "nondeterministic_bitwise_runs",
    "warmup", "repeats", "throttle_events", "threads", "no_turbo", "test_seed", "asan_findings", "ubsan_findings",
    "lint_findings_total", "lint_memory", "lint_numeric", "lint_concurrency", "lint_shape_dtype", "lint_api_misuse",
    "lint_performance",
})
_FLOAT_FIELDS = frozenset({
    "temperature", "latency_ms", "build_time_ms", "check_max_abs_err", "check_max_tol_ratio", "time_median_ms",
    "time_mean_ms", "time_min_ms", "time_max_ms", "time_cv", "gflops", "baseline_time_ms", "baseline_time_cv",
    "baseline_run_time_ms", "baseline_time_1t_ms", "baseline_drift_pct", "speedup_vs_baseline",
    "speedup_vs_baseline_1t", "cpu_mhz_median", "cpu_mhz_min", "temp_c_start", "temp_c_end", "load_1m",
    "pinned_busy_pct", "sibling_busy_pct", "roofline_pct",
})
_JSON_FIELDS = frozenset({
    "sampling", "undefined_symbols", "contract_violation", "check_tests", "time_samples_ms", "baseline_samples_ms",
    "omp_env", "protocol_flags", "perf_summary",
})


def field_kind(field_name: str) -> str:
    """필드의 역직렬화 종류: ``bool`` | ``int`` | ``float`` | ``json`` | ``str``."""
    if field_name in _BOOL_FIELDS:
        return "bool"
    if field_name in _INT_FIELDS:
        return "int"
    if field_name in _FLOAT_FIELDS:
        return "float"
    if field_name in _JSON_FIELDS:
        return "json"
    return "str"


@dataclass
class RunRow:
    """결과 행. 모든 필드는 NA(None)를 허용한다. 필드 순서 == ``COLUMNS`` 순서."""

    run_id: str | None = None
    created_at: str | None = None
    task: str | None = None
    language: str | None = None
    backend: str | None = None
    args: str | None = None
    grammar_hint: str | None = None
    feedback: str | None = None
    sample_idx: int | None = None
    turn: int | None = None
    provider: str | None = None
    model: str | None = None
    model_served: str | None = None
    sampling: dict[str, Any] | None = None
    effort: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    seed: int | None = None
    prompt_hash: str | None = None
    stop_reason: str | None = None
    stop_details_category: str | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tokens_thinking: int | None = None
    tokens_text: int | None = None
    n_thinking_blocks: int | None = None
    latency_ms: float | None = None
    api_error: str | None = None
    valid_sample: bool | None = None
    extract_ok: bool | None = None
    signature_ok: bool | None = None
    signature_exact: bool | None = None
    build_ok: bool | None = None
    build_time_ms: float | None = None
    n_warnings: int | None = None
    undefined_symbols: list[str] | None = None
    contract_violation: list[str] | None = None
    passed: bool | None = None
    turns_to_pass: int | None = None
    error_type: str | None = None
    error_stage: str | None = None
    error_detail: str | None = None
    inputs_mutated: bool | None = None
    oob_write: bool | None = None
    nondeterministic_runs: int | None = None
    nondeterministic_bitwise_runs: int | None = None
    check_max_abs_err: float | None = None
    check_max_tol_ratio: float | None = None
    check_tests: list[dict[str, Any]] | None = None
    bench_ran: bool | None = None
    time_median_ms: float | None = None
    time_mean_ms: float | None = None
    time_min_ms: float | None = None
    time_max_ms: float | None = None
    time_cv: float | None = None
    time_samples_ms: dict[str, list[float]] | None = None
    warmup: int | None = None
    repeats: int | None = None
    repeats_reduced: bool | None = None
    gflops: float | None = None
    baseline_time_ms: float | None = None
    baseline_time_cv: float | None = None
    baseline_samples_ms: dict[str, list[float]] | None = None
    baseline_run_time_ms: float | None = None
    baseline_time_1t_ms: float | None = None
    baseline_drift_pct: float | None = None
    speedup_vs_baseline: float | None = None
    speedup_vs_baseline_1t: float | None = None
    cpu_mhz_median: float | None = None
    cpu_mhz_min: float | None = None
    throttle_events: int | None = None
    temp_c_start: float | None = None
    temp_c_end: float | None = None
    compiler: str | None = None
    compiler_version: str | None = None
    cflags: str | None = None
    march_resolved: str | None = None
    threads: int | None = None
    cores: str | None = None
    omp_env: dict[str, str] | None = None
    governor: str | None = None
    governor_ok: bool | None = None
    epp: str | None = None
    no_turbo: int | None = None
    load_1m: float | None = None
    load_ok: bool | None = None
    pinned_busy_pct: float | None = None
    sibling_busy_pct: float | None = None
    taskset_used: bool | None = None
    protocol_ok: bool | None = None
    protocol_flags: list[str] | None = None
    protocol_checked_at: str | None = None
    hostname: str | None = None
    cpu_model: str | None = None
    python_version: str | None = None
    numpy_version: str | None = None
    numpy_blas: str | None = None
    sdk_version: str | None = None
    test_seed: int | None = None
    test_hash: str | None = None
    env_hash: str | None = None
    asan_findings: int | None = None
    ubsan_findings: int | None = None
    lint_findings_total: int | None = None
    lint_memory: int | None = None
    lint_numeric: int | None = None
    lint_concurrency: int | None = None
    lint_shape_dtype: int | None = None
    lint_api_misuse: int | None = None
    lint_performance: int | None = None
    perf_summary: dict[str, Any] | None = None
    roofline_pct: float | None = None
    raw_path: str | None = None
    built_path: str | None = None
    harness_version: str | None = None

    # ------------------------------------------------------------------ 직렬화

    def to_csv_dict(self) -> dict[str, str]:
        """CSV 한 행(열 이름 → 문자열). 키 순서는 ``COLUMNS`` 와 같다."""
        return {column_of(f.name): serialize_value(getattr(self, f.name)) for f in fields(self)}

    def to_dict(self) -> dict[str, Any]:
        """타입을 유지한 dict(필드명 기준; ``result.json`` 의 ``"row"`` 에 넣는 형태)."""
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_csv_dict(cls, d: dict[str, Any]) -> "RunRow":
        """CSV 행(문자열 값)에서 복원한다. 키는 열 이름이든 필드명이든 허용한다."""
        return cls.from_dict(d)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRow":
        """열 이름/필드명 혼용 dict(문자열 값이든 타입 값이든)에서 복원한다. 모르는 키는 경고 후 무시."""
        known = {f.name for f in fields(cls)}
        kwargs: dict[str, Any] = {}
        unknown: list[str] = []
        for key, value in d.items():
            name = field_of(str(key))
            if name not in known:
                unknown.append(str(key))
                continue
            kwargs[name] = parse_value(name, value)
        if unknown:
            print(f"{_LOG} 경고: 알 수 없는 열 무시 {unknown}", file=sys.stderr)
        return cls(**kwargs)

    def key(self) -> tuple[str, ...]:
        """행 키(``ROW_KEY`` 필드의 직렬화 문자열 튜플)."""
        return tuple(serialize_value(getattr(self, k)) for k in ROW_KEY)


def _unwrap_scalar(value: Any) -> Any:
    """numpy 스칼라 등 ``.item()`` 을 가진 값을 파이썬 값으로 푼다."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes, dict, list, tuple)):
        try:
            return value.item()
        except (TypeError, ValueError):
            return value
    return value


def serialize_value(value: Any) -> str:
    """직렬화 규칙: None → "", bool → 1/0, list/dict → JSON, 그 외 str()."""
    value = _unwrap_scalar(value)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


def parse_value(field_name: str, value: Any) -> Any:
    """필드 종류에 맞춰 값을 복원한다(문자열이면 CSV 규칙, 아니면 타입 값으로 간주)."""
    value = _unwrap_scalar(value)
    if value is None:
        return None
    kind = field_kind(field_name)
    if isinstance(value, str):
        if kind == "str":
            # CSV 에서 NA 와 빈 문자열은 같은 칸이다. 문자열 열은 "" 를 그대로 둔다
            # (error_type/error_stage 는 "" 가 정식 값이다 — 설계 §11).
            return value
        if value == "":
            return None
        if kind == "bool":
            return value.strip().lower() in ("1", "true", "t", "yes")
        if kind == "int":
            try:
                return int(value)
            except ValueError:
                return int(float(value))
        if kind == "float":
            return float(value)
        if kind == "json":
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
    if kind == "bool":
        return bool(value)
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "str":
        # 문자열 열에 리스트/딕트가 오면 JSON으로 고정한다
        if isinstance(value, (list, dict)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return str(value)
    return value


def row_key_of(d: dict[str, Any]) -> tuple[str, ...]:
    """CSV 딕트(열 이름 키)에서 행 키를 뽑는다."""
    return tuple(serialize_value(d.get(k)) for k in ROW_KEY)


# ---------------------------------------------------------------------- CSV I/O


def _write_csv_atomic(csv_path: Path, rows: Iterable[dict[str, str]]) -> None:
    """헤더 + 행들을 메모리에서 만든 뒤 원자적으로(tmp → ``os.replace``) 쓴다."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n", extrasaction="raise")
    writer.writeheader()
    for r in rows:
        writer.writerow({c: r.get(c, "") for c in COLUMNS})
    atomic_write_text(Path(csv_path), buf.getvalue())


def _read_csv_raw(csv_path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    """(헤더, 행 딕트 리스트). 파일이 없거나 비어 있으면 (None, [])."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None, []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return header, rows


def check_header(header: list[str] | None, csv_path: Path) -> None:
    """헤더가 ``COLUMNS`` 와 정확히 같은지 검사한다. 아니면 ``SchemaError``."""
    if header is None:
        return
    if header != COLUMNS:
        missing = [c for c in COLUMNS if c not in header]
        extra = [c for c in header if c not in COLUMNS]

        def _short(items: list[str]) -> str:
            return str(items[:5]) + (f" 외 {len(items) - 5}개" if len(items) > 5 else "")

        reason = "열 순서가 다르다" if not missing and not extra else f"누락 {_short(missing)}, 추가 {_short(extra)}"
        raise SchemaError(
            f"{csv_path}: CSV 헤더가 스키마와 다르다 ({reason}; 열 수 {len(header)} vs {len(COLUMNS)}). "
            "`.venv/bin/python -m harness.run --rebuild-csv` 로 result.json에서 다시 만들어라."
        )


def upsert_row(csv_path: Path, row: RunRow) -> bool:
    """행을 CSV에 넣는다. 헤더가 없으면 쓰고, 있으면 열 일치를 검사한다.

    같은 ``ROW_KEY`` 의 행은 교체한다(전체를 tmp에 다시 쓰고 ``os.replace``).
    교체했으면 True, 새로 추가했으면 False.
    """
    csv_path = Path(csv_path)
    header, existing = _read_csv_raw(csv_path)
    check_header(header, csv_path)
    new = row.to_csv_dict()
    key = row.key()
    replaced = False
    out: list[dict[str, str]] = []
    for r in existing:
        if row_key_of(r) == key:
            if not replaced:
                out.append(new)
                replaced = True
            # 같은 키가 둘 이상이던 기존 중복은 이 기회에 하나로 합친다
            continue
        out.append(r)
    if not replaced:
        out.append(new)
    _write_csv_atomic(csv_path, out)
    return replaced


def _created_at_sort_key(value: str | None) -> tuple[int, Any]:
    """created_at(ISO 8601)을 비교 가능한 키로. 파싱 실패 시 문자열 비교로 폴백."""
    if not value:
        return (0, "")
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.astimezone()
        return (1, dt.timestamp())
    except ValueError:
        return (0, value)


def dedupe_rows(rows: Iterable[dict[str, str]], *, where: str = "") -> list[dict[str, str]]:
    """같은 키의 행은 ``created_at`` 이 최신인 것만 남긴다(첫 등장 순서 유지). 중복이 있으면 경고."""
    kept: dict[tuple[str, ...], dict[str, str]] = {}
    n_dup = 0
    for r in rows:
        k = row_key_of(r)
        prev = kept.get(k)
        if prev is None:
            kept[k] = r
            continue
        n_dup += 1
        if _created_at_sort_key(r.get("created_at")) >= _created_at_sort_key(prev.get("created_at")):
            kept[k] = r
    if n_dup:
        print(f"{_LOG} 경고: 키 중복 {n_dup}건 — created_at 최신 행만 사용 {where}".rstrip(), file=sys.stderr)
    return list(kept.values())


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    """CSV를 읽어 행 딕트(열 이름 → 문자열) 리스트로 돌려준다. 키 중복은 ``created_at`` 최신만 남기고 경고."""
    csv_path = Path(csv_path)
    header, rows = _read_csv_raw(csv_path)
    check_header(header, csv_path)
    return dedupe_rows(rows, where=f"({csv_path})")


def read_result_rows(built_dir: Path) -> list[RunRow]:
    """built 디렉터리 아래 모든 ``result.json`` 의 최상위 ``"row"`` 를 ``RunRow`` 로 읽는다."""
    built_dir = Path(built_dir)
    out: list[RunRow] = []
    if not built_dir.exists():
        print(f"{_LOG} 경고: built 디렉터리가 없다: {built_dir}", file=sys.stderr)
        return out
    for path in sorted(p for p in built_dir.rglob("result.json") if "__pycache__" not in p.parts):
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:
            print(f"{_LOG} 경고: {path} 읽기 실패, 건너뜀: {e}", file=sys.stderr)
            continue
        row = data.get("row") if isinstance(data, dict) else None
        if not isinstance(row, dict):
            print(f"{_LOG} 경고: {path} 에 최상위 'row' 딕트가 없다, 건너뜀", file=sys.stderr)
            continue
        out.append(RunRow.from_dict(row))
    return out


def rebuild_csv(built_dir: Path, csv_path: Path, *, providers: Iterable[str] | None = None) -> int:
    """built 디렉터리의 모든 ``result.json`` 에서 CSV를 다시 쓴다. 쓴 행 수를 돌려준다.

    ``providers`` 가 주어지면 그 provider의 행만 넣는다(dummy/실제 CSV 분리용).
    """
    allowed = set(providers) if providers is not None else None
    rows = read_result_rows(Path(built_dir))
    csv_rows = [r.to_csv_dict() for r in rows if allowed is None or (r.provider or "") in allowed]
    deduped = dedupe_rows(csv_rows, where=f"(rebuild {built_dir})")
    deduped.sort(key=lambda r: (_created_at_sort_key(r.get("created_at")), row_key_of(r)))
    _write_csv_atomic(Path(csv_path), deduped)
    print(f"{_LOG} {csv_path}: {len(deduped)}행 재생성 (result.json {len(rows)}개)", file=sys.stderr)
    return len(deduped)


def rebuild_csvs(built_dir: Path, results_csv: Path, results_csv_dummy: Path) -> dict[str, int]:
    """provider==dummy 행은 ``results_csv_dummy`` 로, 나머지는 ``results_csv`` 로 나눠 재생성한다."""
    rows = read_result_rows(Path(built_dir))
    providers = {r.provider or "" for r in rows}
    real = providers - {"dummy"}
    return {
        str(results_csv): rebuild_csv(built_dir, results_csv, providers=real),
        str(results_csv_dummy): rebuild_csv(built_dir, results_csv_dummy, providers={"dummy"}),
    }


# ---------------------------------------------------------------------- 버전


def _source_files() -> list[Path]:
    """harness_version 해시 대상 파일(정렬, 결정적)."""
    files: list[Path] = []
    files += sorted(p for p in ROOT.glob("harness/**/*.py") if "__pycache__" not in p.parts)
    files += sorted(ROOT.glob("analysis/*.py"))
    files += sorted(p for p in ROOT.glob("prompts/**/*") if p.is_file())
    files += [p for name in ("spec.md", "signatures.yaml", "tolerance.yaml") for p in sorted((ROOT / "tasks").glob(f"*/{name}"))]
    files += [ROOT / "config" / "experiment.yaml"]
    return [p for p in files if p.is_file()]


def source_hash() -> str:
    """하네스 소스(설계 §11 목록)의 sha256[:12]. 경로와 내용을 함께 섞는다."""
    h = hashlib.sha256()
    for p in _source_files():
        h.update(str(p.relative_to(ROOT)).encode("utf-8"))
        h.update(b"\0")
        h.update(p.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


@functools.lru_cache(maxsize=1)
def harness_version() -> str:
    """``git describe --always --dirty`` (dirty면 ``+<hash12>`` 병기); git 저장소가 아니면 ``nogit-<hash12>``."""
    digest = source_hash()
    if shutil.which("git") is not None and (ROOT / ".git").exists():
        try:
            r = run_isolated(["git", "describe", "--always", "--dirty"], timeout_s=30, cwd=ROOT)
        except OSError as e:
            print(f"{_LOG} 경고: git 실행 실패({e}); nogit 버전 사용", file=sys.stderr)
            return f"nogit-{digest}"
        desc = r.stdout_tail.strip().splitlines()[-1].strip() if r.stdout_tail.strip() else ""
        if r.rc == 0 and desc:
            return f"{desc}+{digest}" if desc.endswith("-dirty") else desc
        print(f"{_LOG} 경고: git describe 실패(rc={r.rc}): {r.stderr_tail.strip()[:200]}; nogit 버전 사용", file=sys.stderr)
    return f"nogit-{digest}"


# ---------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """스키마 진단: 열 목록 출력, CSV 검사, 재생성."""
    parser = argparse.ArgumentParser(prog="python -m harness.schema", description="결과 행 스키마 도구")
    parser.add_argument("--columns", action="store_true", help="열 순서 출력")
    parser.add_argument("--version", action="store_true", help="harness_version 출력")
    parser.add_argument("--check", type=Path, default=None, metavar="CSV", help="CSV 헤더·중복 검사")
    parser.add_argument("--rebuild", nargs=2, type=Path, default=None, metavar=("BUILT_DIR", "CSV"), help="result.json에서 CSV 재생성")
    parser.add_argument("--providers", default=None, help="--rebuild 시 포함할 provider (쉼표 구분)")
    ns = parser.parse_args(argv)
    did = False
    if ns.columns:
        print("\n".join(COLUMNS))
        did = True
    if ns.version:
        print(harness_version())
        did = True
    if ns.check is not None:
        try:
            rows = read_rows(ns.check)
        except SchemaError as e:
            print(f"{_LOG} {e}", file=sys.stderr)
            return 2
        print(f"{ns.check}: {len(rows)}행, 헤더 일치")
        did = True
    if ns.rebuild is not None:
        providers = [p.strip() for p in ns.providers.split(",")] if ns.providers else None
        n = rebuild_csv(ns.rebuild[0], ns.rebuild[1], providers=providers)
        print(f"{ns.rebuild[1]}: {n}행")
        did = True
    if not did:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
