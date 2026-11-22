"""결과 CSV 집계 (설계 §12, v3 §13).

``results/runs.csv`` (또는 ``results/runs_dummy.csv``)를 pandas 로 읽고(``dtype=str, keep_default_na=False``
뒤 수동 변환), 행 키(``schema.ROW_KEY``) 중복은 ``created_at`` 최신 행만 남긴 뒤,
그룹 ``(run_id, provider, model, task, language, args, grammar_hint, feedback, turn)`` 별로
pass@k(비편향 추정량)·시간·속도비·토큰·오류 유형을 표로 낸다. run 단위 요약(기준선,
``model_served``/``effort`` 단일값 검사)을 함께 출력한다.

CLI:
    python -m analysis.aggregate [--csv results/runs.csv] [--run-id ID | --latest | --all-runs] [--json]
                                 [--ci] [--coverage] [--by-turn] [--conditions]

    --latest (기본)  created_at 이 가장 최신인 행의 run_id 하나만 집계
    --run-id ID      그 run 만
    --all-runs       CSV 의 모든 run
    --json           표 대신 JSON 객체 {"csv", "selection", "runs", "groups"} 출력. 아래 옵션은 키를 더한다:
                     "coverage" · "by_turn" · "conditions"(행 레코드 리스트), 그룹 표에 붙는 안내가 있으면 "notes"
    --ci             그룹 표에 pass_at_1 · speedup_median 의 95 % 부트스트랩 구간 열(``*_lo``, ``*_hi``)을 더한다
    --coverage       과제 × 언어 표: 칸마다 첫 시도 유효 샘플 수 n, unsupported 조합은 n/a, 행이 없는 조합은 -
    --by-turn        그룹별로 수정 턴을 t 번까지 허용했을 때의 누적 통과율·누적 토큰과 turns_to_pass 분포
    --conditions     같은 provider·model·과제·언어 안에서 조건(args, grammar_hint, feedback) 수준 쌍의 통과율 차와 구간

수정 루프 행(``turn ≥ 1``)이 있으면 그룹 표는 첫 시도(``turn == 0``)만 세고 그 사실을 한 줄로 알린다
(첫 시도 결과끼리 비교할 수 있게). 턴을 합친 결과는 ``--by-turn`` 이 낸다. run 요약 줄은 모든 턴의 행을 센다.
옵션이 더하는 표에서 언어는 성능 결정 부담 순(``LANGUAGE_ORDER``)이고, 그룹 표의 행 순서는 예전 그대로(이름순)다.

종료 코드: 0 정상(행이 없어도 0), 1 CSV 없음 / run_id 없음, 2 CSV 열이 스키마와 맞지 않음.

그룹 열의 의미(README.md 에도 적음):
    n_rows              그룹의 행 수(valid_sample=0 포함)
    n                   valid_sample=1 인 행 수 (pass@k 분모)
    n_pass              그중 pass=1 인 행 수
    pass_at_1/5/n       1 − C(n−c, k)/C(n, k) (k > n 이면 NA); pass_at_1 == n_pass/n
    n_generate_failed   생성 단계 API 오류 행(valid_sample=0; api_error 기록)
    n_refusal           생성 단계 거절 행(valid_sample=1, pass=0, error_type 빈칸)
    n_truncated         생성 단계 max_tokens 절단 행(valid_sample=0)
    benchmarked         bench_ran=1 인 행 수
    n_repeats_reduced   repeats_reduced=1 인 행 수(느린 커널의 반복 축소)
    n_protocol_flagged  protocol_ok=0 이거나 protocol_flags 에 정보용 외 플래그가 있는 행 수
    time_median_ms      pass 행들의 time_median_ms 중앙값(bench timeout 행은 시간 NA 라 제외)
    speedup_*           pass 행들의 speedup_vs_baseline 중앙값/최소/최대, speedup_1t_median 은 1스레드 기준선 대비
    tokens_out_mean     tokens_out(thinking 포함) 평균 — 토큰 값이 있는 모든 행(절단 행 포함)
    tokens_text_mean    tokens_text 평균
    tokens_per_pass     Σ(tokens_in + tokens_out) / n_pass (토큰 값이 있는 모든 행의 합; n_pass=0 이면 NA)
    tokens_out_per_pass Σ tokens_out / n_pass
    error_types         pass=0 행의 error_type 집계 "유형:수, …" (수 내림차순, 같으면 이름순; 빈 error_type 은 제외)
    pass_at_1_lo/hi, speedup_median_lo/hi   (--ci) 95 % 백분위 부트스트랩 구간. n = 5 에서 뜻하는 것은 ``analysis.stats``
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml

from analysis.stats import bootstrap_ci, bootstrap_pass_rate, paired_bootstrap_diff, unpaired_bootstrap_diff
from harness.paths import ROOT, TASKS_DIR, resolve
from harness.protocol import is_informational
from harness.schema import COLUMNS, ERROR_TYPES, ROW_KEY

_LOG = "[aggregate]"

DEFAULT_CSV: Path = ROOT / "results" / "runs.csv"

# 그룹 키 (설계 §12)
GROUP_KEY: tuple[str, ...] = (
    "run_id", "provider", "model", "task", "language", "args", "grammar_hint", "feedback", "turn",
)

# 그룹 표의 지표 열(순서 고정)
METRIC_COLUMNS: tuple[str, ...] = (
    "n_rows", "n", "n_pass", "pass_at_1", "pass_at_5", "pass_at_n",
    "n_generate_failed", "n_refusal", "n_truncated", "benchmarked", "n_repeats_reduced", "n_protocol_flagged",
    "time_median_ms", "speedup_median", "speedup_min", "speedup_max", "speedup_1t_median",
    "tokens_out_mean", "tokens_text_mean", "tokens_per_pass", "tokens_out_per_pass", "error_types",
)

# --ci 가 그룹 표에 더하는 구간 열: 점추정 열 → (하한, 상한). 표에서는 점추정 열 바로 뒤에 온다.
CI_COLUMNS: dict[str, tuple[str, str]] = {
    "pass_at_1": ("pass_at_1_lo", "pass_at_1_hi"),
    "speedup_median": ("speedup_median_lo", "speedup_median_hi"),
}

# 언어 축: 성능 결정 부담이 적은 것부터(CLAUDE.md §1). 목록에 없는 언어는 그 뒤에 이름순으로 온다.
LANGUAGE_ORDER: tuple[str, ...] = (
    "numpy", "numba", "halide_auto", "halide_manual", "c_openmp", "c_seq", "rust",
    "opencl", "c_neon", "llvm_ir", "aarch64_asm",
)

# 마이크로 조건(CLAUDE.md §1)과 수준 순서. 앞의 수준이 비교의 기준(base)이다. 목록에 없는 수준은 그 뒤에 이름순.
CONDITION_LEVELS: dict[str, tuple[str, ...]] = {
    "args": ("positional", "keyword"),
    "grammar_hint": ("off", "on"),
    "feedback": ("compiler", "compiler+lint"),
}
CONDITION_COLUMNS: tuple[str, ...] = tuple(CONDITION_LEVELS)

# --by-turn 의 그룹(턴을 합친다)과 그 안의 샘플 하나를 가리키는 키
BY_TURN_KEY: tuple[str, ...] = tuple(k for k in GROUP_KEY if k != "turn")
SAMPLE_KEY: tuple[str, ...] = (*BY_TURN_KEY, "sample_idx")

# --conditions 가 수준을 비교하는 범위와 표의 열
EFFECT_KEY: tuple[str, ...] = ("provider", "model", "task", "language")
EFFECT_COLUMNS: tuple[str, ...] = (
    *EFFECT_KEY, "factor", "base", "alt", "at", "quantity", "method",
    "n_base", "n_alt", "rate_base", "rate_alt", "diff", "ci_low", "ci_high",
)

COVERAGE_COLUMNS: tuple[str, ...] = ("task", "language", "n", "status", "reason")

# 행의 결과 범주(``outcome_labels``): 통과, 오류 분류표(변경 금지), 생성 단계 실패, 그 밖
OUTCOME_ORDER: tuple[str, ...] = (
    "pass", "pass_bench_timeout", *(t for t in ERROR_TYPES if t), "refusal", "truncated", "generate_failed", "other",
)

# 수동 변환 대상: 불리언("1"/"0"), 숫자, 그대로 두는 문자열
_BOOL_COLUMNS: tuple[str, ...] = ("valid_sample", "pass", "bench_ran", "repeats_reduced", "protocol_ok")
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "tokens_in", "tokens_out", "tokens_text", "time_median_ms", "speedup_vs_baseline", "speedup_vs_baseline_1t",
    "baseline_time_ms", "baseline_run_time_ms", "baseline_time_1t_ms", "baseline_drift_pct",
)
_TEXT_COLUMNS: tuple[str, ...] = (
    "created_at", "sample_idx", "error_stage", "error_detail", "error_type", "stop_reason", "api_error",
    "protocol_flags", "model_served", "effort",
)
# 집계에 꼭 있어야 하는 열(없으면 오류 + --rebuild-csv 안내)
REQUIRED_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys([*GROUP_KEY, *ROW_KEY, *_TEXT_COLUMNS, *_BOOL_COLUMNS, *_NUMERIC_COLUMNS])
)

_TRUE_WORDS = frozenset({"1", "true", "t", "yes"})
_FALSE_WORDS = frozenset({"0", "false", "f", "no"})

EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_SCHEMA = 2


class AggregateError(Exception):
    """CSV 없음·run_id 없음·열 불일치 등 집계 불가 사유. 메시지는 사용자에게 그대로 보여주고 ``exit_code`` 로 끝낸다."""

    def __init__(self, message: str, exit_code: int = EXIT_NOT_FOUND) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ----------------------------------------------------------------------------- pass@k


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """비편향 pass@k 추정량 ``1 − C(n−c, k)/C(n, k)``. ``k > n`` 이거나 ``n == 0`` 이면 None(NA)."""
    if n <= 0 or k <= 0 or k > n:
        return None
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


# ----------------------------------------------------------------------------- 정렬 순서


def _rank(order: Sequence[str], value: str) -> tuple[int, str]:
    """고정 순서 안의 위치. 목록에 없으면 맨 뒤에서 이름순."""
    return (order.index(value), "") if value in order else (len(order), value)


def language_sort_key(language: str) -> tuple[int, str]:
    """언어를 ``LANGUAGE_ORDER``(성능 결정 부담 순)로 정렬하는 키."""
    return _rank(LANGUAGE_ORDER, language)


def level_sort_key(factor: str, level: str) -> tuple[int, str]:
    """조건 ``factor`` 의 수준을 ``CONDITION_LEVELS`` 순으로 정렬하는 키."""
    return _rank(CONDITION_LEVELS[factor], level)


def _key_sort_key(column: str, value: str) -> tuple[int, str]:
    """표의 키 열 값 정렬 키: 언어는 부담 순, 조건은 수준 순, 그 밖은 문자열 순."""
    if column == "language":
        return language_sort_key(value)
    if column in CONDITION_LEVELS:
        return level_sort_key(column, value)
    return (0, value)


def _key_strings(columns: Iterable[str], key: Iterable[Any]) -> dict[str, str]:
    """groupby 키 튜플 → {열: 문자열} (NA 는 빈 문자열)."""
    return {k: ("" if pd.isna(v) else str(v)) for k, v in zip(columns, key)}


# ----------------------------------------------------------------------------- 읽기·변환


def _to_bool(s: pd.Series) -> pd.Series:
    """CSV 불리언 열("1"/"0"; "true"/"false" 도 허용) → nullable ``boolean`` (빈칸은 NA)."""
    words = s.astype(str).str.strip().str.lower()
    out = pd.Series(pd.NA, index=s.index, dtype="boolean")
    return out.mask(words.isin(_TRUE_WORDS), True).mask(words.isin(_FALSE_WORDS), False)


def _is_true(s: pd.Series) -> pd.Series:
    """nullable boolean → NA 를 False 로 본 bool 마스크."""
    return s.fillna(False).astype(bool)


def _is_false(s: pd.Series) -> pd.Series:
    """nullable boolean → 명시적으로 False 인 행만 True 인 bool 마스크(NA 는 False)."""
    return s.eq(False).fillna(False).astype(bool)


def valid_mask(df: pd.DataFrame) -> pd.Series:
    """pass@k 분모에 드는 행(``valid_sample=1``)."""
    return _is_true(df["valid_sample"])


def pass_mask(df: pd.DataFrame) -> pd.Series:
    """유효하고 통과한 행(``valid_sample=1`` ∧ ``pass=1``; bench timeout 행 포함)."""
    return valid_mask(df) & _is_true(df["pass"])


def turn_numbers(df: pd.DataFrame) -> pd.Series:
    """``turn`` 열(문자열)의 정수값. 읽을 수 없는 값은 첫 시도(0)로 본다."""
    return pd.to_numeric(df["turn"], errors="coerce").fillna(0).astype(int)


def first_attempt_rows(df: pd.DataFrame) -> pd.DataFrame:
    """첫 시도(``turn == 0``) 행만. 수정 루프 행이 없으면 입력과 같은 행들이다."""
    return df[turn_numbers(df).eq(0)]


def _empty_frame() -> pd.DataFrame:
    """스키마 열을 가진 빈 문자열 프레임."""
    return pd.DataFrame({c: pd.Series([], dtype=str) for c in COLUMNS})


def check_columns(columns: list[str], csv_path: Path) -> None:
    """집계에 필요한 열이 전부 있는지 검사한다. 스키마와 정확히 같지 않으면 경고만 한다."""
    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        shown = missing[:5] + (["…"] if len(missing) > 5 else [])
        raise AggregateError(
            f"{csv_path}: 집계에 필요한 열이 없다 {shown} (열 수 {len(columns)} vs 스키마 {len(COLUMNS)}). "
            "`.venv/bin/python -m harness.run --rebuild-csv` 로 result.json에서 다시 만들어라.",
            EXIT_SCHEMA,
        )
    if list(columns) != COLUMNS:
        extra = [c for c in columns if c not in COLUMNS]
        absent = [c for c in COLUMNS if c not in columns]
        print(
            f"{_LOG} 경고: {csv_path} 의 열이 스키마와 정확히 같지 않다 (추가 {extra[:5]}, 누락 {absent[:5]}); "
            "필요한 열은 있으므로 계속한다",
            file=sys.stderr,
        )


def read_csv_str(csv_path: Path) -> pd.DataFrame:
    """CSV 를 모두 문자열로 읽는다(``dtype=str, keep_default_na=False``). 없으면 ``AggregateError``, 0바이트면 빈 프레임."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise AggregateError(
            f"CSV가 없다: {csv_path} — 먼저 `make dummy`(results/runs_dummy.csv) 또는 "
            "`.venv/bin/python -m harness.run --run-id ID` 로 행을 만들어라."
        )
    if csv_path.stat().st_size == 0:
        print(f"{_LOG} {csv_path}: 0바이트 파일 — 집계할 행이 없다", file=sys.stderr)
        return _empty_frame()
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    check_columns(list(df.columns), csv_path)
    return df


def _created_ts(s: pd.Series) -> pd.Series:
    """``created_at``(ISO 8601, 시간대 포함) → UTC Timestamp (파싱 실패는 NaT)."""
    return pd.to_datetime(s, utc=True, format="ISO8601", errors="coerce")


def dedupe_latest(df: pd.DataFrame) -> pd.DataFrame:
    """같은 ``ROW_KEY`` 의 행은 ``created_at`` 이 최신인 것만 남긴다(파싱 불가는 가장 오래된 것으로 취급). 원래 순서 유지."""
    if df.empty:
        return df
    ordered = df.assign(_ts=_created_ts(df["created_at"])).sort_values(
        ["_ts", "created_at"], na_position="first", kind="stable"
    )
    kept = ordered.drop_duplicates(subset=list(ROW_KEY), keep="last")
    n_dup = len(df) - len(kept)
    if n_dup:
        print(f"{_LOG} 경고: 키 중복 {n_dup}건 — created_at 최신 행만 사용", file=sys.stderr)
    return kept.drop(columns="_ts").sort_index()


def convert_types(df: pd.DataFrame) -> pd.DataFrame:
    """문자열 프레임의 불리언·숫자 열을 수동 변환한다(불리언 → nullable boolean, 숫자 → float64, 빈칸 → NA/NaN)."""
    out = df.copy()
    for col in _BOOL_COLUMNS:
        out[col] = _to_bool(out[col])
    for col in _NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64")
    return out


def load_runs(csv_path: Path) -> pd.DataFrame:
    """CSV 읽기 → 열 검사 → 키 중복 제거 → 타입 변환. 행이 없으면 빈 프레임(열은 있음)."""
    df = read_csv_str(Path(csv_path))
    if df.empty:
        return convert_types(df)
    return convert_types(dedupe_latest(df))


# ----------------------------------------------------------------------------- run 선택


def latest_run_id(df: pd.DataFrame) -> str:
    """``created_at`` 이 가장 최신인 행의 run_id. created_at 을 하나도 못 읽으면 run_id 문자열 최대값."""
    ts = _created_ts(df["created_at"])
    if ts.notna().any():
        return str(df.at[ts.idxmax(), "run_id"])
    return str(df["run_id"].max())


def select_runs(df: pd.DataFrame, *, run_id: str | None = None, all_runs: bool = False) -> tuple[pd.DataFrame, list[str]]:
    """집계할 행과 run_id 목록. ``all_runs`` 면 전부, ``run_id`` 가 있으면 그 run, 아니면 최신 run."""
    all_ids = sorted(set(df["run_id"].astype(str)))
    if all_runs:
        return df, all_ids
    if run_id is None:
        run_id = latest_run_id(df)
    sub = df[df["run_id"] == run_id]
    if sub.empty:
        raise AggregateError(f"run_id {run_id!r} 의 행이 없다. CSV 에 있는 run_id: {all_ids}")
    return sub, [run_id]


# ----------------------------------------------------------------------------- 그룹 집계


def has_protocol_flag(text: str) -> bool:
    """``protocol_flags`` JSON 리스트에 정보용(``turbo=``, ``env_overrides_dropped``) 외 플래그가 있는가.
    비어 있으면 False, JSON 이 아닌 비어 있지 않은 문자열이면(알 수 없는 기록) 보수적으로 True."""
    if not text or not str(text).strip():
        return False
    try:
        flags = json.loads(text)
    except (TypeError, ValueError):
        return True
    if not isinstance(flags, list):
        return False
    return any(isinstance(f, str) and not is_informational(f) for f in flags)


def _nan_if_none(v: float | None) -> float:
    return math.nan if v is None else float(v)


def _error_types_string(error_types: pd.Series) -> str:
    """"유형:수, …" — 수 내림차순, 같으면 이름순. 빈 error_type 은 제외."""
    counts = error_types[error_types.ne("")].value_counts()
    items = sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    return ", ".join(f"{name}:{int(cnt)}" for name, cnt in items)


def _generate_stage_split(g: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """생성 단계 실패 행의 마스크 ``(generate_failed, refusal, truncated)``.

    생성 게이트(설계 §8-5) 순서대로 분할: API 오류 → 절단 → 거절. 세 집합은 서로 겹치지 않는다."""
    gen = g["error_stage"].eq("generate")
    api_error = gen & g["api_error"].ne("")
    truncated = gen & ~api_error & (g["error_detail"].eq("truncated") | g["stop_reason"].eq("max_tokens"))
    refusal = gen & ~api_error & ~truncated & (g["error_detail"].str.startswith("refusal") | g["stop_reason"].eq("refusal"))
    generate_failed = gen & ~truncated & ~refusal
    return generate_failed, refusal, truncated


def outcome_labels(df: pd.DataFrame) -> pd.Series:
    """행마다 결과 범주 하나(``OUTCOME_ORDER`` 의 값; 오류 분류 막대 그래프용). 모든 행이 정확히 한 범주에 든다.

    ``pass`` / ``pass_bench_timeout``(정확성은 통과, bench 가 시간 초과) / 실패 행의 ``error_type`` /
    생성 단계의 ``refusal`` · ``truncated`` · ``generate_failed``(API 오류) / 분류표 밖의 값은 ``other``.
    """
    generate_failed, refusal, truncated = _generate_stage_split(df)
    passed = pass_mask(df)
    labels = df["error_type"].where(df["error_type"].isin([t for t in ERROR_TYPES if t]), "other")
    labels = labels.mask(passed, "pass").mask(passed & df["error_type"].eq("timeout"), "pass_bench_timeout")
    return labels.mask(generate_failed, "generate_failed").mask(refusal, "refusal").mask(truncated, "truncated")


def group_metrics(g: pd.DataFrame) -> dict[str, Any]:
    """그룹 하나의 지표(``METRIC_COLUMNS`` 순서의 키). 숫자 NA 는 NaN."""
    valid = valid_mask(g)
    passed = pass_mask(g)
    n = int(valid.sum())
    c = int(passed.sum())

    generate_failed, refusal, truncated = _generate_stage_split(g)

    flagged = _is_false(g["protocol_ok"]) | g["protocol_flags"].astype(str).map(has_protocol_flag).astype(bool)

    times = g.loc[passed, "time_median_ms"]
    speedup = g.loc[passed, "speedup_vs_baseline"]
    speedup_1t = g.loc[passed, "speedup_vs_baseline_1t"]

    tokens_in = g["tokens_in"]
    tokens_out = g["tokens_out"]
    tokens_text = g["tokens_text"]
    have_tokens = bool(tokens_in.notna().any() or tokens_out.notna().any())
    total_tokens = float(tokens_in.sum(skipna=True) + tokens_out.sum(skipna=True))
    total_out = float(tokens_out.sum(skipna=True))

    return {
        "n_rows": int(len(g)),
        "n": n,
        "n_pass": c,
        "pass_at_1": _nan_if_none(pass_at_k(n, c, 1)),
        "pass_at_5": _nan_if_none(pass_at_k(n, c, 5)),
        "pass_at_n": _nan_if_none(pass_at_k(n, c, n)),
        "n_generate_failed": int(generate_failed.sum()),
        "n_refusal": int(refusal.sum()),
        "n_truncated": int(truncated.sum()),
        "benchmarked": int(_is_true(g["bench_ran"]).sum()),
        "n_repeats_reduced": int(_is_true(g["repeats_reduced"]).sum()),
        "n_protocol_flagged": int(flagged.sum()),
        "time_median_ms": float(times.median()),
        "speedup_median": float(speedup.median()),
        "speedup_min": float(speedup.min()),
        "speedup_max": float(speedup.max()),
        "speedup_1t_median": float(speedup_1t.median()),
        "tokens_out_mean": float(tokens_out.mean()),
        "tokens_text_mean": float(tokens_text.mean()),
        "tokens_per_pass": total_tokens / c if (c > 0 and have_tokens) else math.nan,
        "tokens_out_per_pass": total_out / c if (c > 0 and tokens_out.notna().any()) else math.nan,
        "error_types": _error_types_string(g.loc[_is_false(g["pass"]), "error_type"].astype(str)),
    }


def group_intervals(g: pd.DataFrame) -> dict[str, float]:
    """``--ci``: 그룹 하나의 ``pass_at_1`` 과 ``speedup_median`` 의 95 % 백분위 부트스트랩 구간(``CI_COLUMNS`` 의 키).

    통과율은 유효 샘플의 통과 여부를, 속도비는 통과 샘플 중 시간이 있는 것을 재추출한다. 샘플이 없으면 NaN.
    """
    valid = valid_mask(g)
    passed = pass_mask(g)
    _, pass_lo, pass_hi = bootstrap_pass_rate(passed[valid].to_numpy())
    _, speedup_lo, speedup_hi = bootstrap_ci(g.loc[passed, "speedup_vs_baseline"].to_numpy(), stat=np.median)
    return {
        "pass_at_1_lo": pass_lo, "pass_at_1_hi": pass_hi,
        "speedup_median_lo": speedup_lo, "speedup_median_hi": speedup_hi,
    }


def metric_columns(ci: bool = False) -> list[str]:
    """그룹 표의 지표 열. ``ci`` 면 구간 열이 해당 점추정 열 바로 뒤에 온다."""
    return [c for metric in METRIC_COLUMNS for c in (metric, *(CI_COLUMNS.get(metric, ()) if ci else ()))]


def aggregate_groups(df: pd.DataFrame, *, ci: bool = False) -> pd.DataFrame:
    """그룹(``GROUP_KEY``)별 지표 표. 열 = ``GROUP_KEY + metric_columns(ci)``."""
    columns = [*GROUP_KEY, *metric_columns(ci)]
    if df.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for key, g in df.groupby(list(GROUP_KEY), sort=True, dropna=False):
        rec: dict[str, Any] = _key_strings(GROUP_KEY, key)
        rec.update(group_metrics(g))
        if ci:
            rec.update(group_intervals(g))
        records.append(rec)
    return pd.DataFrame.from_records(records, columns=columns)


# ----------------------------------------------------------------------------- 수정 턴 (--by-turn)


def sample_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """턴 0 이 유효한 샘플마다 한 행: ``SAMPLE_KEY`` + ``pass_turn`` + ``known_until``.

    - ``pass_turn``: 처음 통과한 턴(통과 행의 ``turn`` = 그 행의 ``turns_to_pass``). 통과한 적이 없으면 NaN.
    - ``known_until``: 수정 턴을 t 번까지 허용했을 때의 결과를 아는 가장 큰 t. 통과했거나, 마지막 유효 행이 생성 단계
      실패(거절 — 수정 루프의 대상이 아니다, 설계 v3 §9.3)면 무한대다. 그 밖에는 마지막 유효 행의 턴이다: 그 뒤 턴은
      시도하지 않았거나(수정 루프 미실행·최대 턴 도달) 호출이 실패했으므로 통과 여부를 모른다.
    """
    turn = turn_numbers(df)
    valid = valid_mask(df)
    work = df[list(SAMPLE_KEY)].assign(
        valid0=valid & turn.eq(0),
        pass_turn=turn.where(pass_mask(df)),
        last_turn=turn.where(valid),
        refusal_turn=turn.where(valid & df["error_stage"].eq("generate")),
    )
    samples = work.groupby(list(SAMPLE_KEY), sort=False, dropna=False).agg(
        valid0=("valid0", "any"), pass_turn=("pass_turn", "min"),
        last_turn=("last_turn", "max"), refusal_turn=("refusal_turn", "max"),
    ).reset_index()
    samples = samples[samples["valid0"]]
    resolved = samples["pass_turn"].notna() | samples["refusal_turn"].eq(samples["last_turn"])
    samples = samples.assign(known_until=samples["last_turn"].where(~resolved, math.inf))
    return samples[[*SAMPLE_KEY, "pass_turn", "known_until"]].reset_index(drop=True)


def drop_rows_after_pass(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """샘플(``SAMPLE_KEY``)이 처음 통과한 턴보다 뒤 턴의 행을 뺀 행과 뺀 행 수.

    수정 루프는 통과한 샘플을 더 고치지 않는다(대상은 ``pass=0``). 그런데도 뒤 턴 행이 있으면 앞 턴을 ``--force`` 로 다시
    처리해 통과가 된 뒤 남은 낡은 행이다. 수정 턴 집계(누적 통과율·누적 토큰·``turns_to_pass``·그림 4)는 이 행을 세지 않는다.
    """
    turn = turn_numbers(df)
    first_pass = df[list(SAMPLE_KEY)].assign(_pass_turn=turn.where(pass_mask(df))).groupby(
        list(SAMPLE_KEY), sort=False, dropna=False,
    )["_pass_turn"].transform("min")
    stale = turn.gt(first_pass)   # 통과한 적이 없으면 비교가 NaN → False
    return df[~stale], int(stale.sum())


def _turns_to_pass_string(pass_turn: pd.Series) -> str:
    """"턴:샘플 수, …, NA:통과 못 한 수" — 턴 오름차순."""
    counts = pass_turn.dropna().astype(int).value_counts().sort_index()
    items = [f"{t}:{int(c)}" for t, c in counts.items()]
    n_never = int(pass_turn.isna().sum())
    if n_never:
        items.append(f"NA:{n_never}")
    return ", ".join(items)


def by_turn_table(df: pd.DataFrame) -> pd.DataFrame:
    """``--by-turn``: 그룹(``BY_TURN_KEY``)마다 수정 턴을 t 번까지 허용했을 때의 누적 통과율·누적 토큰과 통과 턴 분포.

    열: ``BY_TURN_KEY``, ``n``(턴 0 이 유효한 샘플 수 = 분모), ``pass_within_<t>``(t = 0..선택된 행의 최대 턴;
    ``pass_within_0`` == ``pass_at_1``), ``turns_to_pass``("턴:샘플 수, …, NA:통과 못 한 수"),
    ``tokens_cum_<t>``(턴 t 이하 모든 행의 tokens_in + tokens_out 합 — 수정 턴의 입력에는 앞선 대화가 다시 들어 있다).

    그룹에 턴 t 까지의 결과를 모르는 샘플(``sample_outcomes`` 의 ``known_until`` < t)이 하나라도 있으면
    ``pass_within_<t>`` 와 ``tokens_cum_<t>`` 는 NA 다: 고쳐 보지 않은 턴을 실패로 세지 않는다. 모르는 샘플은 모두
    그때까지 실패한 샘플이라, 빼고 계산하면 통과율이 부풀고 실패로 세면 API 오류가 모델의 실패가 된다.

    샘플이 처음 통과한 턴보다 뒤 턴의 행(``drop_rows_after_pass``)은 모든 열에서 빼고 stderr 에 한 줄로 알린다.
    """
    df, n_stale = drop_rows_after_pass(df)
    if n_stale:
        print(
            f"{_LOG} 경고: 처음 통과한 턴보다 뒤 턴의 행 {n_stale}개를 수정 턴 집계에서 뺀다 "
            "(앞 턴을 --force 로 다시 처리해 통과가 된 뒤 남은 행)",
            file=sys.stderr,
        )
    turn = turn_numbers(df)
    turns = range((int(turn.max()) if len(df) else 0) + 1)
    columns = [
        *BY_TURN_KEY, "n", *(f"pass_within_{t}" for t in turns), "turns_to_pass", *(f"tokens_cum_{t}" for t in turns),
    ]
    work = df.assign(_turn=turn, _tokens=df[["tokens_in", "tokens_out"]].sum(axis=1, min_count=1))
    samples_of = dict(iter(sample_outcomes(df).groupby(list(BY_TURN_KEY), sort=False, dropna=False)))
    records: list[dict[str, Any]] = []
    for key, g in work.groupby(list(BY_TURN_KEY), sort=False, dropna=False):
        samples = samples_of.get(key)
        n = 0 if samples is None else len(samples)
        rec: dict[str, Any] = {**_key_strings(BY_TURN_KEY, key), "n": n}
        for t in turns:
            known = n > 0 and bool((samples["known_until"] >= t).all())
            rec[f"pass_within_{t}"] = float((samples["pass_turn"] <= t).sum()) / n if known else math.nan
            rec[f"tokens_cum_{t}"] = float(g.loc[g["_turn"] <= t, "_tokens"].sum(min_count=1)) if known else math.nan
        rec["turns_to_pass"] = _turns_to_pass_string(samples["pass_turn"]) if n else ""
        records.append(rec)
    records.sort(key=lambda r: tuple(_key_sort_key(k, r[k]) for k in BY_TURN_KEY))
    return pd.DataFrame.from_records(records, columns=columns)


# ----------------------------------------------------------------------------- 조건 효과 (--conditions)


def _feedback_pairs(base: pd.DataFrame, alt: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """같은 run 의 같은 ``sample_idx`` 끼리 짝지은 표(``pass_turn_base/alt``, ``known_until_base/alt``)와 짝이 없어 빠진 샘플 수."""
    on = ["run_id", "sample_idx"]
    columns = [*on, "pass_turn", "known_until"]
    pairs = base[columns].merge(alt[columns], on=on, suffixes=("_base", "_alt"))
    return pairs, len(base) + len(alt) - 2 * len(pairs)


def _common_horizon(pairs: pd.DataFrame) -> int | None:
    """모든 짝의 두 수준 결과를 다 아는 가장 큰 수정 턴 수 H(짝이 없으면 None).

    짝마다 ``min(known_until_base, known_until_alt)`` 의 최솟값이다 — 수정 루프를 돌리지 않은 수준, API 오류로 멈춘 턴,
    두 수준의 ``--max-turns`` 차이가 모두 여기서 H 를 낮춘다. 모든 짝이 끝났으면(무한대) 짝들이 통과한 가장 늦은 턴
    (통과가 없으면 0)이다: 그보다 큰 t 에서도 결과가 같다.
    """
    if pairs.empty:
        return None
    horizon = float(np.minimum(pairs["known_until_base"], pairs["known_until_alt"]).min())
    if math.isinf(horizon):
        latest = pd.concat([pairs["pass_turn_base"], pairs["pass_turn_alt"]]).max()
        horizon = 0.0 if pd.isna(latest) else float(latest)
    return int(horizon)


def condition_effects(df: pd.DataFrame) -> pd.DataFrame:
    """``--conditions``: 같은 provider·model·과제·언어 안에서 조건 수준 쌍의 통과율 차 ``alt − base`` 와 95 % 부트스트랩 구간.

    조건 하나(``factor``)의 수준만 다르고 나머지 두 조건(``at`` 열)이 같은 샘플끼리 비교한다. 수준이 둘 이상인 조건만
    행이 생긴다. 열은 ``EFFECT_COLUMNS``.

    - ``args``, ``grammar_hint``: 프롬프트가 달라 샘플이 서로 독립이다 → 첫 시도 통과율(``pass_at_1``)의 차,
      독립 표본 부트스트랩(``n_base``, ``n_alt`` = 수준별 유효 샘플 수; 선택한 run 이 여럿이면 합쳐 센다).
      턴 0 은 feedback 과 무관하므로 feedback 으로 나누지 않는다(``at`` 은 나머지 한 조건만): feedback 수준만 다른
      같은 샘플(``--seed-from-feedback`` 이 복사한 턴 0)은 수준 순서가 앞선 것 하나만 센다 — 한 측정을 두 번 세지 않는다.
    - ``feedback``: 두 수준이 같은 턴 0 응답에서 갈라진다(``harness.repair --seed-from-feedback``) → 같은 run 의 같은
      ``sample_idx`` 끼리 짝지어 "수정 턴을 H 번까지 허용했을 때 통과했는가"(``quantity`` = ``pass_within_<H>``)의 차,
      대응 표본 부트스트랩(``n_base`` == ``n_alt`` = 짝 수). 짝이 없는 샘플은 빼고 경고한다. H 는 모든 짝의 두 수준
      결과를 다 아는 가장 큰 턴 수(``_common_horizon``; ``by_turn_table`` 의 NA 규칙과 같다 — 고쳐 보지 않은 턴이나
      API 오류 턴을 실패로 세지 않는다). H < 1 이면(두 수준 모두에서 결과를 아는 수정 턴이 없다) 비교하지 않는다:
      통과율·차·구간이 NA 이고 경고한다. 짝이 없으면 ``quantity`` 는 ``pass_within_NA`` 다.

    n = 5 에서 이 구간이 뜻하는 것(넓다)은 ``analysis.stats`` 에 적혀 있다.
    """
    samples = sample_outcomes(df)
    samples = samples.assign(first=samples["pass_turn"].eq(0).astype(float))
    # args·grammar_hint 비교용: feedback 수준만 다른 같은 턴 0 샘플은 수준 순서가 앞선 하나만 남긴다
    feedback_order = sorted(samples["feedback"].unique(), key=lambda level: level_sort_key("feedback", level))
    first_attempts = (
        samples.assign(_rank=samples["feedback"].map({level: i for i, level in enumerate(feedback_order)}))
        .sort_values("_rank", kind="stable")
        .drop_duplicates([k for k in SAMPLE_KEY if k != "feedback"])
        .drop(columns="_rank")
    )
    records: list[dict[str, Any]] = []
    for factor in CONDITION_COLUMNS:
        if factor == "feedback":
            pool, others = samples, [c for c in CONDITION_COLUMNS if c != factor]
        else:
            pool, others = first_attempts, [c for c in CONDITION_COLUMNS if c not in (factor, "feedback")]
        for key, stratum in pool.groupby([*EFFECT_KEY, *others], sort=False, dropna=False):
            fixed = _key_strings([*EFFECT_KEY, *others], key)
            at = ", ".join(f"{c}={fixed[c]}" for c in others)
            levels = sorted(stratum[factor].unique(), key=lambda level: level_sort_key(factor, level))
            for base, alt in itertools.combinations(levels, 2):
                rows_base, rows_alt = stratum[stratum[factor].eq(base)], stratum[stratum[factor].eq(alt)]
                if factor == "feedback":
                    pairs, n_unpaired = _feedback_pairs(rows_base, rows_alt)
                    where = "/".join(fixed[k] for k in EFFECT_KEY)
                    if n_unpaired:
                        print(f"{_LOG} 경고: {where} ({at}) feedback {base} 대 {alt}: 짝이 없는 샘플 {n_unpaired}개 제외", file=sys.stderr)
                    horizon = _common_horizon(pairs)
                    quantity, method = f"pass_within_{'NA' if horizon is None else horizon}", "paired"
                    if horizon is not None and horizon >= 1:
                        x_base = pairs["pass_turn_base"].le(horizon).to_numpy(dtype=float)
                        x_alt = pairs["pass_turn_alt"].le(horizon).to_numpy(dtype=float)
                    else:   # 짝이 없거나 수정 턴을 비교할 수 없다 → 통과율·차·구간 모두 NA
                        if horizon is not None:
                            print(
                                f"{_LOG} 경고: {where} ({at}) feedback {base} 대 {alt}: 두 수준 모두에서 결과를 아는 수정 턴이 없어 "
                                "비교하지 않는다 (수정 루프 미실행·API 오류 턴)",
                                file=sys.stderr,
                            )
                        x_base = x_alt = np.full(len(pairs), math.nan)
                    diff, low, high = paired_bootstrap_diff(x_alt, x_base)
                else:
                    x_base, x_alt = rows_base["first"].to_numpy(), rows_alt["first"].to_numpy()
                    quantity, method = "pass_at_1", "unpaired"
                    diff, low, high = unpaired_bootstrap_diff(x_alt, x_base)
                records.append({
                    **{k: fixed[k] for k in EFFECT_KEY},
                    "factor": factor, "base": base, "alt": alt, "at": at, "quantity": quantity, "method": method,
                    "n_base": len(x_base), "n_alt": len(x_alt),
                    "rate_base": float(x_base.mean()) if len(x_base) else math.nan,
                    "rate_alt": float(x_alt.mean()) if len(x_alt) else math.nan,
                    "diff": diff, "ci_low": low, "ci_high": high,
                })
    records.sort(key=lambda r: (
        *(_key_sort_key(k, r[k]) for k in EFFECT_KEY), CONDITION_COLUMNS.index(r["factor"]),
        level_sort_key(r["factor"], r["base"]), level_sort_key(r["factor"], r["alt"]), r["at"],
    ))
    return pd.DataFrame.from_records(records, columns=list(EFFECT_COLUMNS))


# ----------------------------------------------------------------------------- 과제 × 언어 (--coverage)


def load_unsupported(tasks: Iterable[str], tasks_dir: Path | None = None) -> dict[str, dict[str, str]]:
    """과제 → {돌리지 않는 언어: 이유}. ``tasks/<task>/signatures.yaml`` 의 ``unsupported`` 키를 그대로 읽는다(설계 v3 §2.1).

    명세 파일이 없거나 키가 맵이 아니면 경고하고 빈 맵으로 둔다(그 과제의 unsupported 조합은 "행 없음"으로 보인다).
    """
    root = TASKS_DIR if tasks_dir is None else Path(tasks_dir)
    out: dict[str, dict[str, str]] = {}
    for task in tasks:
        path = root / task / "signatures.yaml"
        out[task] = {}
        if not path.is_file():
            print(f"{_LOG} 경고: {path} 가 없어 {task} 의 unsupported 언어를 모른다", file=sys.stderr)
            continue
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        unsupported = doc.get("unsupported") if isinstance(doc, dict) else None
        if isinstance(unsupported, dict):
            out[task] = {str(language): str(reason) for language, reason in unsupported.items()}
        elif unsupported is not None:
            print(f"{_LOG} 경고: {path} 의 unsupported 가 맵이 아니다 — 무시", file=sys.stderr)
    return out


def coverage_table(df: pd.DataFrame, tasks_dir: Path | None = None) -> pd.DataFrame:
    """``--coverage``: 선택된 행에 나오는 과제 × 언어 조합마다 한 행(열 ``COVERAGE_COLUMNS``; 언어는 ``LANGUAGE_ORDER`` 순).

    ``status``: ``rows``(행이 있다; ``n`` = 첫 시도 유효 샘플 수, 조건·run 을 합친 값) / ``unsupported``(과제 명세가
    그 언어를 돌리지 않는다고 적었다; ``reason`` = 명세의 이유) / ``missing``(지원하는 조합인데 행이 없다).
    unsupported 인데 행이 있으면 경고하고 ``rows`` 로 센다.
    """
    tasks = sorted(set(df["task"]))
    languages = sorted(set(df["language"]), key=language_sort_key)
    unsupported = load_unsupported(tasks, tasks_dir)
    n_rows = df.groupby(["task", "language"]).size()
    first = first_attempt_rows(df)
    n_valid = first[valid_mask(first)].groupby(["task", "language"]).size()
    records: list[dict[str, Any]] = []
    for task, language in itertools.product(tasks, languages):
        reason = unsupported[task].get(language, "")
        n: int | None = None
        if n_rows.get((task, language), 0):
            status, n = "rows", int(n_valid.get((task, language), 0))
            if reason:
                print(f"{_LOG} 경고: {task}/{language} 는 unsupported 인데 행이 있다 ({reason})", file=sys.stderr)
        else:
            status = "unsupported" if reason else "missing"
        records.append({"task": task, "language": language, "n": n, "status": status, "reason": reason})
    table = pd.DataFrame.from_records(records, columns=list(COVERAGE_COLUMNS))
    return table.astype({"n": "Int64"})


# ----------------------------------------------------------------------------- run 요약


def _distinct_text(s: pd.Series) -> list[str]:
    """비어 있지 않은 문자열 값의 정렬된 집합."""
    return sorted({str(v) for v in s if str(v) != ""})


def _single_number(s: pd.Series, run_id: str, name: str, warnings: list[str]) -> float:
    """run 안에서 하나여야 하는 숫자 열. 없으면 NaN, 여러 값이면 경고 + 중앙값."""
    distinct = sorted(set(s.dropna().tolist()))
    if not distinct:
        return math.nan
    if len(distinct) > 1:
        w = f"{name} 가 run 안에서 단일값이 아니다 ({len(distinct)}개: {distinct[:4]}) — 중앙값 사용"
        warnings.append(w)
        print(f"{_LOG} 경고: run {run_id}: {w}", file=sys.stderr)
        return float(s.median())
    return float(distinct[0])


def summarize_runs(df: pd.DataFrame) -> list[dict[str, Any]]:
    """run 단위 요약: 기준선(run 시작/1스레드/대응 평균), model_served·effort 단일값 검사(아니면 경고)."""
    out: list[dict[str, Any]] = []
    if df.empty:
        return out
    for run_id, r in df.groupby("run_id", sort=True):
        run_id = str(run_id)
        warnings: list[str] = []
        served = _distinct_text(r["model_served"])
        efforts = _distinct_text(r["effort"])
        if len(served) > 1:
            w = f"model_served 가 단일값이 아니다: {served}"
            warnings.append(w)
            print(f"{_LOG} 경고: run {run_id}: {w}", file=sys.stderr)
        if len(efforts) > 1:
            w = f"effort 가 단일값이 아니다: {efforts}"
            warnings.append(w)
            print(f"{_LOG} 경고: run {run_id}: {w}", file=sys.stderr)
        valid = valid_mask(r)
        passed = pass_mask(r)
        paired = r["baseline_time_ms"].dropna()
        # 기준선은 과제마다 다르다(M2): 과제별로 요약하고, run 수준 값은 과제가 하나일 때만 채운다.
        by_task: list[dict[str, Any]] = []
        for task, rt in r.groupby("task", sort=True):
            tp = rt["baseline_time_ms"].dropna()
            label = f"{run_id}/{task}"
            by_task.append({
                "task": str(task),
                "baseline_run_time_ms": _single_number(rt["baseline_run_time_ms"], label, "baseline_run_time_ms", warnings),
                "baseline_time_1t_ms": _single_number(rt["baseline_time_1t_ms"], label, "baseline_time_1t_ms", warnings),
                "baseline_paired_mean_ms": float(tp.mean()) if len(tp) else math.nan,
                "baseline_paired_n": int(len(tp)),
                "baseline_drift_abs_max_pct": float(rt["baseline_drift_pct"].abs().max()),
            })
        single = by_task[0] if len(by_task) == 1 else None
        out.append({
            "run_id": run_id,
            "provider": "|".join(_distinct_text(r["provider"])),
            "model": "|".join(_distinct_text(r["model"])),
            "model_served": "|".join(served),
            "effort": "|".join(efforts),
            "languages": _distinct_text(r["language"]),
            "n_rows": int(len(r)),
            "n_valid": int(valid.sum()),
            "n_pass": int(passed.sum()),
            "tasks": [b["task"] for b in by_task],
            "baselines_by_task": by_task,
            "baseline_run_time_ms": single["baseline_run_time_ms"] if single else math.nan,
            "baseline_time_1t_ms": single["baseline_time_1t_ms"] if single else math.nan,
            "baseline_paired_mean_ms": float(paired.mean()) if len(paired) else math.nan,
            "baseline_paired_n": int(len(paired)),
            "baseline_drift_abs_max_pct": float(r["baseline_drift_pct"].abs().max()),
            "warnings": warnings,
        })
    return out


# ----------------------------------------------------------------------------- 출력


def _fmt(ndigits: int) -> Callable[[Any], str]:
    """표용 숫자 포맷터: NaN → "NA", 아니면 소수 ndigits 자리."""

    def f(x: Any) -> str:
        try:
            if pd.isna(x):
                return "NA"
        except (TypeError, ValueError):
            return str(x)
        return f"{float(x):.{ndigits}f}"

    return f


_FORMATTERS: dict[str, Callable[[Any], str]] = {
    **{c: _fmt(3) for c in ("pass_at_1", "pass_at_5", "pass_at_n", *CI_COLUMNS["pass_at_1"])},
    **{c: _fmt(2) for c in ("time_median_ms",)},
    **{c: _fmt(3) for c in ("speedup_median", "speedup_min", "speedup_max", "speedup_1t_median", *CI_COLUMNS["speedup_median"])},
    **{c: _fmt(0) for c in ("tokens_out_mean", "tokens_text_mean", "tokens_per_pass", "tokens_out_per_pass")},
    **{c: _fmt(3) for c in ("rate_base", "rate_alt", "diff", "ci_low", "ci_high")},
}
# --by-turn 의 턴별 열(이름이 턴 수에 따라 늘어난다)은 접두사로 포맷터를 고른다
_PREFIX_FORMATTERS: dict[str, Callable[[Any], str]] = {"pass_within_": _fmt(3), "tokens_cum_": _fmt(0)}


def _formatters(columns: Iterable[str]) -> dict[str, Callable[[Any], str]]:
    """표의 열 이름 → 숫자 포맷터(포맷터가 없는 열은 pandas 기본)."""
    out: dict[str, Callable[[Any], str]] = {}
    for column in columns:
        if column in _FORMATTERS:
            out[column] = _FORMATTERS[column]
        for prefix, formatter in _PREFIX_FORMATTERS.items():
            if column.startswith(prefix):
                out[column] = formatter
    return out


def _format_keyed_table(table: pd.DataFrame, key_columns: Sequence[str], empty: str) -> str:
    """표를 ``to_string()`` 으로. 전체에서 값이 하나뿐인 키 열은 '공통:' 줄로 빼고 표에서 뺀다. 행이 없으면 ``empty``."""
    if table.empty:
        return empty
    constant = {k: str(table[k].iloc[0]) for k in key_columns if table[k].nunique(dropna=False) == 1}
    lines: list[str] = []
    if constant:
        lines.append("공통: " + "  ".join(f"{k}={v}" for k, v in constant.items()))
    shown = table.drop(columns=list(constant))
    lines.append(shown.to_string(index=False, formatters=_formatters(shown.columns), na_rep="NA"))
    return "\n".join(lines)


def format_table(groups: pd.DataFrame) -> str:
    """그룹 표를 ``to_string()`` 으로. 선택 전체에서 값이 하나뿐인 그룹 키 열은 '공통:' 줄로 빼고 표에서 뺀다."""
    return _format_keyed_table(groups, GROUP_KEY, "(그룹 없음)")


def format_by_turn(table: pd.DataFrame) -> str:
    """``by_turn_table`` 결과를 표로."""
    return _format_keyed_table(table, BY_TURN_KEY, "(그룹 없음)")


def format_conditions(table: pd.DataFrame) -> str:
    """``condition_effects`` 결과를 표로."""
    return _format_keyed_table(table, EFFECT_KEY, "(비교할 조건 수준 쌍 없음: 첫 시도가 유효한 샘플에 수준이 둘 이상인 조건이 없다)")


def format_coverage(table: pd.DataFrame) -> str:
    """``coverage_table`` 결과를 과제 × 언어 행렬로(칸: n, ``n/a``, ``-``). unsupported 조합의 이유를 행렬 아래에 한 줄씩 붙인다."""
    if table.empty:
        return "(과제 × 언어 조합 없음)"
    cell = table["n"].astype(str).where(table["status"].eq("rows"), table["status"].map({"unsupported": "n/a", "missing": "-"}))
    matrix = table.assign(cell=cell).pivot(index="task", columns="language", values="cell")
    matrix = matrix[list(dict.fromkeys(table["language"]))].rename_axis(columns=None).reset_index()
    lines = [matrix.to_string(index=False)]
    for row in table[table["status"].eq("unsupported")].itertuples(index=False):
        lines.append(f"n/a {row.task} × {row.language}: {row.reason}")
    return "\n".join(lines)


@dataclass(frozen=True)
class Section:
    """옵션으로 켜는 추가 표 하나: CLI 옵션 ``--<name 의 _ 를 - 로>``, JSON 키 ``name``, 텍스트 출력의 제목 줄."""

    name: str
    title: str
    build: Callable[[pd.DataFrame], pd.DataFrame]
    render: Callable[[pd.DataFrame], str]


SECTIONS: tuple[Section, ...] = (
    Section(
        "coverage", "# coverage: 과제 × 언어 — 칸은 첫 시도(turn 0) 유효 샘플 수 n, n/a = unsupported, - = 행 없음",
        coverage_table, format_coverage,
    ),
    Section(
        "by_turn",
        "# by-turn: 수정 턴을 t 번까지 허용했을 때의 누적 통과율 pass_within_t · 누적 토큰 tokens_cum_t "
        "(NA = 턴 t 의 결과를 모르는 실패 샘플이 있다: 고쳐 보지 않았거나 그 호출이 무효)",
        by_turn_table, format_by_turn,
    ),
    Section(
        "conditions",
        "# conditions: 조건 수준 쌍의 통과율 차 diff = alt − base 와 95 % 부트스트랩 구간 "
        "(args·grammar_hint 는 첫 시도의 독립 표본, feedback 은 sample_idx 대응으로 두 수준 모두 결과를 아는 수정 턴 수 H "
        "까지(quantity = pass_within_<H>); n = 5 에서는 구간이 넓다)",
        condition_effects, format_conditions,
    ),
)

_CI_NOTE = (
    "*_lo/*_hi 는 95 % 백분위 부트스트랩 구간(재추출 2000회, seed 0)이다. n = 5 에서는 넓고, "
    "전부 통과·전부 실패 그룹의 폭 0 은 변동을 관측하지 못했다는 뜻이다"
)


def render_text(
    csv_path: Path, mode: str, runs: list[dict[str, Any]], groups: pd.DataFrame, notes: Sequence[str] = ()
) -> str:
    """사람용 텍스트 출력: run 요약 줄들 + (있으면) 그룹 표를 읽는 법을 적은 ``notes`` + 그룹 표."""
    ms = _fmt(2)
    pct = _fmt(1)
    lines = [f"# aggregate: {csv_path} (선택: {mode}, run {len(runs)}개, 그룹 {len(groups)}개)"]
    for r in runs:
        lines.append(
            f"== run {r['run_id']}: provider={r['provider'] or 'NA'} model={r['model'] or 'NA'} "
            f"model_served={r['model_served'] or 'NA'} effort={r['effort'] or 'NA'} "
            f"rows={r['n_rows']} valid={r['n_valid']} pass={r['n_pass']}"
        )
        for b in r.get("baselines_by_task") or []:
            lines.append(
                f"   baseline[{b['task']}]: run_start={ms(b['baseline_run_time_ms'])} ms, 1t={ms(b['baseline_time_1t_ms'])} ms, "
                f"paired_mean={ms(b['baseline_paired_mean_ms'])} ms (n={b['baseline_paired_n']}, "
                f"max|drift|={pct(b['baseline_drift_abs_max_pct'])}%)"
            )
        lines.append(f"   languages: {', '.join(r['languages']) or 'NA'}")
        for w in r["warnings"]:
            lines.append(f"   경고: {w}")
    lines.extend(f"주: {note}" for note in notes)
    lines.append(format_table(groups))
    return "\n".join(lines)


def _jsonable(v: Any) -> Any:
    """pandas/numpy 값을 JSON 가능 값으로. NaN/NA/NaT → None."""
    if v is None or v is pd.NA or v is pd.NaT:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, str)):
        return v
    if isinstance(v, float):
        return None if math.isnan(v) else v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if hasattr(v, "item"):
        try:
            return _jsonable(v.item())
        except (TypeError, ValueError):
            pass
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return str(v)


def build_report(
    csv_path: Path, mode: str, run_ids: list[str], runs: list[dict[str, Any]], groups: pd.DataFrame,
    notes: Sequence[str] = (), sections: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """``--json`` 출력 객체. ``notes`` 가 있으면 ``"notes"`` 키, ``sections`` 의 표는 그 이름의 키(행 레코드 리스트)로 더한다."""
    report: dict[str, Any] = {
        "csv": str(csv_path),
        "selection": {"mode": mode, "run_ids": list(run_ids)},
        "runs": _jsonable(runs),
        "groups": _jsonable(groups.to_dict(orient="records")),
    }
    if notes:
        report["notes"] = list(notes)
    for name, table in (sections or {}).items():
        report[name] = _jsonable(table.to_dict(orient="records"))
    return report


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """CSV 를 집계해 표(또는 ``--json``)를 stdout 에 출력한다."""
    parser = argparse.ArgumentParser(prog="python -m analysis.aggregate", description="결과 CSV 집계 (pass@k·속도비·토큰·오류 유형)")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"결과 CSV (기본 {DEFAULT_CSV.relative_to(ROOT)})")
    sel = parser.add_mutually_exclusive_group()
    sel.add_argument("--run-id", default=None, metavar="ID", help="이 run 만 집계")
    sel.add_argument("--latest", action="store_true", help="created_at 최신 run 만 집계 (기본)")
    sel.add_argument("--all-runs", action="store_true", help="모든 run 집계")
    parser.add_argument("--json", action="store_true", help="표 대신 JSON 출력")
    parser.add_argument("--ci", action="store_true", help="그룹 표에 pass_at_1·speedup_median 의 95%% 부트스트랩 구간 열 추가")
    parser.add_argument("--coverage", action="store_true", help="과제 × 언어 표 (n, unsupported 는 n/a, 행 없음은 -)")
    parser.add_argument("--by-turn", action="store_true", help="수정 턴별 누적 통과율·누적 토큰, turns_to_pass 분포")
    parser.add_argument("--conditions", action="store_true", help="조건(args, grammar_hint, feedback) 수준 쌍의 통과율 차와 구간")
    ns = parser.parse_args(argv)

    csv_path = resolve(ns.csv)
    mode = "run_id" if ns.run_id else ("all" if ns.all_runs else "latest")
    wanted = [s for s in SECTIONS if getattr(ns, s.name)]
    try:
        df = load_runs(csv_path)
    except AggregateError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return e.exit_code

    if df.empty:
        print(f"{_LOG} {csv_path}: 집계할 행이 없다 (헤더만 있거나 빈 파일)", file=sys.stderr)
        if ns.json:
            sections = {s.name: s.build(df) for s in wanted}
            report = build_report(csv_path, mode, [], [], aggregate_groups(df, ci=ns.ci), sections=sections)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"# aggregate: {csv_path} — 행 없음")
        return EXIT_OK

    try:
        selected, run_ids = select_runs(df, run_id=ns.run_id, all_runs=ns.all_runs)
    except AggregateError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return e.exit_code

    runs = summarize_runs(selected)
    first = first_attempt_rows(selected)
    notes: list[str] = []
    if len(first) < len(selected):
        notes.append(
            f"수정 루프 행(turn ≥ 1) {len(selected) - len(first)}개가 있어 아래 표는 첫 시도(turn 0)만 센다 "
            "— 턴을 합친 결과는 --by-turn"
        )
    if ns.ci:
        notes.append(_CI_NOTE)
    groups = aggregate_groups(first, ci=ns.ci)
    sections = {s.name: s.build(selected) for s in wanted}
    if ns.json:
        print(json.dumps(build_report(csv_path, mode, run_ids, runs, groups, notes, sections), ensure_ascii=False, indent=2))
    else:
        print(render_text(csv_path, mode, runs, groups, notes))
        for s in wanted:
            print(s.title)
            print(s.render(sections[s.name]))
    return EXIT_OK


if __name__ == "__main__":
    from harness.procutil import setup_stdio

    setup_stdio()   # 진입점에서만(설계 v3 §16): 파이프로 나가는 표가 cp949 콘솔에서 UnicodeEncodeError 로 죽지 않게
    sys.exit(main())
