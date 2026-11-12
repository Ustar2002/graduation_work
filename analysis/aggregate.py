"""결과 CSV 집계 (설계 §12).

``results/runs.csv`` (또는 ``results/runs_dummy.csv``)를 pandas 로 읽고(``dtype=str, keep_default_na=False``
뒤 수동 변환), 행 키(``schema.ROW_KEY``) 중복은 ``created_at`` 최신 행만 남긴 뒤,
그룹 ``(run_id, provider, model, task, language, args, grammar_hint, feedback, turn)`` 별로
pass@k(비편향 추정량)·시간·속도비·토큰·오류 유형을 표로 낸다. run 단위 요약(기준선,
``model_served``/``effort`` 단일값 검사)을 함께 출력한다. 부트스트랩 CI는 M6.

CLI:
    python -m analysis.aggregate [--csv results/runs.csv] [--run-id ID | --latest | --all-runs] [--json]

    --latest (기본)  created_at 이 가장 최신인 행의 run_id 하나만 집계
    --run-id ID      그 run 만
    --all-runs       CSV 의 모든 run
    --json           표 대신 JSON 객체 {"csv", "selection", "runs", "groups"} 출력

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
    speedup_*           pass 행들의 speedup_vs_baseline(기준선 중앙값 / 커널 중앙값) 중앙값/최소/최대, speedup_1t_median 은 1스레드 기준선 대비
    speedup_mean_median pass 행들의 speedup_mean(기준선 평균 / 커널 평균; 2026-11-12 규약의 주 지표, 라운드를 모은 값) 중앙값
    tokens_out_mean     tokens_out(thinking 포함) 평균 — 토큰 값이 있는 모든 행(절단 행 포함)
    tokens_text_mean    tokens_text 평균
    tokens_per_pass     Σ(tokens_in + tokens_out) / n_pass (토큰 값이 있는 모든 행의 합; n_pass=0 이면 NA)
    tokens_out_per_pass Σ tokens_out / n_pass
    error_types         pass=0 행의 error_type 집계 "유형:수, …" (수 내림차순, 같으면 이름순; 빈 error_type 은 제외)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from harness.paths import ROOT, resolve
from harness.protocol import is_informational
from harness.schema import COLUMNS, ROW_KEY

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
    "time_median_ms", "speedup_median", "speedup_min", "speedup_max", "speedup_mean_median", "speedup_1t_median",
    "tokens_out_mean", "tokens_text_mean", "tokens_per_pass", "tokens_out_per_pass", "error_types",
)

# 수동 변환 대상: 불리언("1"/"0"), 숫자, 그대로 두는 문자열
_BOOL_COLUMNS: tuple[str, ...] = ("valid_sample", "pass", "bench_ran", "repeats_reduced", "protocol_ok")
_NUMERIC_COLUMNS: tuple[str, ...] = (
    "tokens_in", "tokens_out", "tokens_text", "time_median_ms", "speedup_vs_baseline", "speedup_vs_baseline_1t",
    "baseline_time_ms", "baseline_run_time_ms", "baseline_time_1t_ms", "baseline_drift_pct",
)
# 2026-11-12 규약(라운드 평균)의 열. 이전 CSV 에는 없을 수 있어 필수로 두지 않는다(없으면 NaN).
_OPTIONAL_NUMERIC_COLUMNS: tuple[str, ...] = ("speedup_mean", "speedup_ci_lo", "speedup_ci_hi", "bench_rounds")
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
    """CSV 없음·run_id 없음·열 불일치 등 집계 불가 사유. 메시지는 사용자에게 그대로 보여준다."""


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
            "`.venv/bin/python -m harness.run --rebuild-csv` 로 result.json에서 다시 만들어라."
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
    for col in _OPTIONAL_NUMERIC_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float64") if col in out.columns else math.nan
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


def group_metrics(g: pd.DataFrame) -> dict[str, Any]:
    """그룹 하나의 지표(``METRIC_COLUMNS`` 순서의 키). 숫자 NA 는 NaN."""
    valid = _is_true(g["valid_sample"])
    passed = valid & _is_true(g["pass"])
    n = int(valid.sum())
    c = int(passed.sum())

    # 생성 게이트(설계 §8-5) 순서대로 분할: API 오류 → 절단 → 거절. 세 집합은 서로 겹치지 않는다.
    gen = g["error_stage"].eq("generate")
    api_error = gen & g["api_error"].ne("")
    truncated = gen & ~api_error & (g["error_detail"].eq("truncated") | g["stop_reason"].eq("max_tokens"))
    refusal = gen & ~api_error & ~truncated & (g["error_detail"].str.startswith("refusal") | g["stop_reason"].eq("refusal"))
    generate_failed = gen & ~truncated & ~refusal

    flagged = _is_false(g["protocol_ok"]) | g["protocol_flags"].astype(str).map(has_protocol_flag).astype(bool)

    times = g.loc[passed, "time_median_ms"]
    speedup = g.loc[passed, "speedup_vs_baseline"]
    speedup_1t = g.loc[passed, "speedup_vs_baseline_1t"]
    speedup_mean = g.loc[passed, "speedup_mean"]   # 샘플마다 (기준선 평균 / 커널 평균), 라운드를 모은 값(2026-11-12)

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
        "speedup_mean_median": float(speedup_mean.median()),
        "speedup_1t_median": float(speedup_1t.median()),
        "tokens_out_mean": float(tokens_out.mean()),
        "tokens_text_mean": float(tokens_text.mean()),
        "tokens_per_pass": total_tokens / c if (c > 0 and have_tokens) else math.nan,
        "tokens_out_per_pass": total_out / c if (c > 0 and tokens_out.notna().any()) else math.nan,
        "error_types": _error_types_string(g.loc[_is_false(g["pass"]), "error_type"].astype(str)),
    }


def aggregate_groups(df: pd.DataFrame) -> pd.DataFrame:
    """그룹(``GROUP_KEY``)별 지표 표. 열 = ``GROUP_KEY + METRIC_COLUMNS``."""
    columns = [*GROUP_KEY, *METRIC_COLUMNS]
    if df.empty:
        return pd.DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for key, g in df.groupby(list(GROUP_KEY), sort=True, dropna=False):
        rec: dict[str, Any] = {k: ("" if pd.isna(v) else str(v)) for k, v in zip(GROUP_KEY, key)}
        rec.update(group_metrics(g))
        records.append(rec)
    return pd.DataFrame.from_records(records, columns=columns)


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
        valid = _is_true(r["valid_sample"])
        passed = valid & _is_true(r["pass"])
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
    **{c: _fmt(3) for c in ("pass_at_1", "pass_at_5", "pass_at_n")},
    **{c: _fmt(2) for c in ("time_median_ms",)},
    **{c: _fmt(3) for c in ("speedup_median", "speedup_min", "speedup_max", "speedup_mean_median", "speedup_1t_median")},
    **{c: _fmt(0) for c in ("tokens_out_mean", "tokens_text_mean", "tokens_per_pass", "tokens_out_per_pass")},
}


def format_table(groups: pd.DataFrame) -> str:
    """그룹 표를 ``to_string()`` 으로. 선택 전체에서 값이 하나뿐인 그룹 키 열은 '공통:' 줄로 빼고 표에서 뺀다."""
    if groups.empty:
        return "(그룹 없음)"
    constant = {k: str(groups[k].iloc[0]) for k in GROUP_KEY if groups[k].nunique(dropna=False) == 1}
    lines: list[str] = []
    if constant:
        lines.append("공통: " + "  ".join(f"{k}={v}" for k, v in constant.items()))
    table = groups.drop(columns=list(constant))
    lines.append(table.to_string(index=False, formatters=_FORMATTERS, na_rep="NA"))
    return "\n".join(lines)


def render_text(csv_path: Path, mode: str, runs: list[dict[str, Any]], groups: pd.DataFrame) -> str:
    """사람용 텍스트 출력: run 요약 줄들 + 그룹 표."""
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


def build_report(csv_path: Path, mode: str, run_ids: list[str], runs: list[dict[str, Any]], groups: pd.DataFrame) -> dict[str, Any]:
    """``--json`` 출력 객체."""
    return {
        "csv": str(csv_path),
        "selection": {"mode": mode, "run_ids": list(run_ids)},
        "runs": _jsonable(runs),
        "groups": _jsonable(groups.to_dict(orient="records")),
    }


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
    ns = parser.parse_args(argv)

    csv_path = resolve(ns.csv)
    mode = "run_id" if ns.run_id else ("all" if ns.all_runs else "latest")
    try:
        df = load_runs(csv_path)
    except AggregateError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return EXIT_SCHEMA if "필요한 열" in str(e) else EXIT_NOT_FOUND

    if df.empty:
        print(f"{_LOG} {csv_path}: 집계할 행이 없다 (헤더만 있거나 빈 파일)", file=sys.stderr)
        if ns.json:
            print(json.dumps(build_report(csv_path, mode, [], [], aggregate_groups(df)), ensure_ascii=False, indent=2))
        else:
            print(f"# aggregate: {csv_path} — 행 없음")
        return EXIT_OK

    try:
        selected, run_ids = select_runs(df, run_id=ns.run_id, all_runs=ns.all_runs)
    except AggregateError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return EXIT_NOT_FOUND

    runs = summarize_runs(selected)
    groups = aggregate_groups(selected)
    if ns.json:
        print(json.dumps(build_report(csv_path, mode, run_ids, runs, groups), ensure_ascii=False, indent=2))
    else:
        print(render_text(csv_path, mode, runs, groups))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
