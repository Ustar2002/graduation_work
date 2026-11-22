"""aggregate 테스트: pass@k 추정량, 합성 CSV(harness.schema.COLUMNS) 집계 수치, 키 중복 제거, run 선택, CLI(JSON·빈 CSV·없는 CSV).

뒤쪽 절은 M3–M6 의 새 행(수정 턴, 조건 수준 여럿, 새 백엔드)을 담은 합성 run 으로 ``--ci``·``--by-turn``·
``--conditions``·``--coverage`` 를 손계산 기대값과 맞춰 본다. 그 합성 자료(``synthetic_rows``)는 ``tests.test_plots`` 도 쓴다.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import runpy
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any, Sequence
from unittest import mock

import analysis.aggregate as aggregate
from analysis.aggregate import (
    BY_TURN_KEY,
    COVERAGE_COLUMNS,
    EFFECT_COLUMNS,
    EXIT_NOT_FOUND,
    EXIT_OK,
    GROUP_KEY,
    LANGUAGE_ORDER,
    METRIC_COLUMNS,
    OUTCOME_ORDER,
    aggregate_groups,
    by_turn_table,
    condition_effects,
    coverage_table,
    first_attempt_rows,
    format_coverage,
    has_protocol_flag,
    language_sort_key,
    latest_run_id,
    level_sort_key,
    load_runs,
    load_unsupported,
    main,
    metric_columns,
    outcome_labels,
    pass_at_k,
    sample_outcomes,
    select_runs,
    summarize_runs,
)
from harness import procutil
from harness.schema import COLUMNS, RunRow

RUN_A = "20261104-100000-dummy"   # 최신 run
RUN_B = "20261103-090000-dummy"   # 오래된 run


def _row(**over: object) -> dict[str, str]:
    """기본값 위에 덮어쓴 RunRow 의 CSV 딕트(열 이름 → 문자열)."""
    base: dict[str, object] = dict(
        run_id=RUN_A, created_at="2026-11-04T10:00:00+09:00", task="gemm", language="numpy", backend="python",
        args="positional", grammar_hint="off", feedback="compiler", sample_idx=0, turn=0,
        provider="dummy", model="dummy", model_served="dummy", effort="medium",
        valid_sample=True, passed=False, error_type="", error_stage="", error_detail="", stop_reason="end_turn",
        api_error="", bench_ran=False, repeats_reduced=False, protocol_ok=True, protocol_flags=["turbo=on"],
        tokens_in=1000, baseline_run_time_ms=14.7, baseline_time_1t_ms=51.0,
    )
    base.update(over)
    return RunRow(**base).to_csv_dict()  # type: ignore[arg-type]


def _rows() -> list[dict[str, str]]:
    """합성 결과 행. 기대값은 각 테스트의 주석 참조."""
    numpy_rows = [
        # pass 3개: 시간 14/16, 세 번째는 bench timeout(pass=1, 시간 NA)
        _row(sample_idx=0, passed=True, bench_ran=True, time_median_ms=14.0, speedup_vs_baseline=1.05,
             speedup_vs_baseline_1t=3.6, tokens_out=500, tokens_text=400, baseline_time_ms=14.5, baseline_drift_pct=-1.36),
        _row(sample_idx=1, passed=True, bench_ran=True, time_median_ms=16.0, speedup_vs_baseline=0.92,
             speedup_vs_baseline_1t=3.2, tokens_out=700, tokens_text=600, baseline_time_ms=15.1, baseline_drift_pct=2.72,
             repeats_reduced=True, protocol_ok=False, protocol_flags=["governor=powersave"]),
        _row(sample_idx=2, passed=True, bench_ran=True, error_stage="bench", error_type="timeout",
             error_detail="first_call_200000ms_projected_800s", tokens_out=900, tokens_text=800, baseline_time_ms=14.9,
             baseline_drift_pct=1.36, protocol_ok=True, protocol_flags=["turbo=on", "baseline_drift=12.3%"]),
        # 실패 2개 (check)
        _row(sample_idx=3, error_stage="check", error_type="wrong_output", error_detail="mismatch", tokens_out=300, tokens_text=300),
        _row(sample_idx=4, error_stage="check", error_type="numeric_instability", error_detail="imprecise", tokens_out=400, tokens_text=400),
        # max_tokens 절단: valid_sample=0 (분모 제외), 토큰은 과금됨
        _row(sample_idx=5, valid_sample=False, error_stage="generate", error_detail="truncated", stop_reason="max_tokens",
             tokens_out=32000, tokens_text=None),
    ]
    c_seq_rows = [
        # 거절: valid_sample=1, pass=0, error_type 빈칸
        _row(language="c_seq", backend="c", sample_idx=0, error_stage="generate", error_detail="refusal:harmful",
             stop_reason="refusal", stop_details_category="harmful", tokens_out=50, tokens_text=50),
        # API 오류: valid_sample=0, 토큰 NA
        _row(language="c_seq", backend="c", sample_idx=1, valid_sample=False, error_stage="generate",
             api_error="transient:APIConnectionError:None:timeout", stop_reason="", tokens_in=None),
        # 빌드 실패 (model_served 가 다른 값 → run 요약 경고)
        _row(language="c_seq", backend="c", sample_idx=2, error_stage="build", error_type="compile_error",
             error_detail="error: expected ';'", tokens_out=600, tokens_text=500, model_served="dummy-2",
             created_at="2026-11-04T10:30:00+09:00"),
        # 위 행의 오래된 중복(키 같음, created_at 이전, pass=1): 중복 제거로 버려져야 한다
        _row(language="c_seq", backend="c", sample_idx=2, passed=True, created_at="2026-11-04T09:00:00+09:00",
             tokens_out=5, tokens_text=5),
    ]
    run_b_rows = [
        _row(run_id=RUN_B, created_at="2026-11-03T09:00:00+09:00", passed=True, bench_ran=True, time_median_ms=20.0,
             speedup_vs_baseline=0.7, tokens_out=100, tokens_text=100, baseline_run_time_ms=14.0, baseline_time_1t_ms=50.0),
    ]
    return numpy_rows + c_seq_rows + run_b_rows


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _group(groups, language: str, run_id: str = RUN_A) -> dict[str, object]:
    sub = groups[(groups["language"] == language) & (groups["run_id"] == run_id)]
    assert len(sub) == 1, f"그룹이 하나여야 한다: {language} {run_id} → {len(sub)}"
    return sub.iloc[0].to_dict()


class PassAtKTest(unittest.TestCase):
    def test_estimator(self) -> None:
        self.assertAlmostEqual(pass_at_k(5, 2, 1), 0.4)
        self.assertAlmostEqual(pass_at_k(5, 2, 2), 1 - 3 / 10)   # 1 − C(3,2)/C(5,2)
        self.assertAlmostEqual(pass_at_k(5, 2, 4), 1.0)          # n−c < k
        self.assertAlmostEqual(pass_at_k(5, 2, 5), 1.0)
        self.assertEqual(pass_at_k(5, 0, 3), 0.0)
        self.assertAlmostEqual(pass_at_k(10, 3, 5), 1 - math.comb(7, 5) / math.comb(10, 5))

    def test_na_cases(self) -> None:
        self.assertIsNone(pass_at_k(3, 1, 5))   # k > n
        self.assertIsNone(pass_at_k(0, 0, 1))   # n == 0
        self.assertIsNone(pass_at_k(3, 1, 0))


class ProtocolFlagTest(unittest.TestCase):
    def test_informational_flags_do_not_count(self) -> None:
        self.assertFalse(has_protocol_flag(""))
        self.assertFalse(has_protocol_flag("[]"))
        self.assertFalse(has_protocol_flag('["turbo=on","env_overrides_dropped"]'))
        self.assertTrue(has_protocol_flag('["turbo=on","governor=powersave"]'))
        self.assertTrue(has_protocol_flag('["baseline_drift=12.3%"]'))
        self.assertTrue(has_protocol_flag("not json"))


class AggregateTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aggregate_test_")
        self.tmp = Path(self._tmp.name)
        self.csv = self.tmp / "runs_dummy.csv"
        _write_csv(self.csv, _rows())
        with contextlib.redirect_stderr(io.StringIO()):
            self.df = load_runs(self.csv)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_dedupe_keeps_latest_created_at(self) -> None:
        self.assertEqual(len(self.df), 10)  # 11행 중 c_seq/2 의 오래된 중복 1행 제거
        c2 = self.df[(self.df["language"] == "c_seq") & (self.df["sample_idx"] == "2")]
        self.assertEqual(len(c2), 1)
        self.assertEqual(c2.iloc[0]["error_type"], "compile_error")
        self.assertIs(bool(c2.iloc[0]["pass"]), False)

    def test_types_converted(self) -> None:
        self.assertEqual(str(self.df["pass"].dtype), "boolean")
        self.assertEqual(str(self.df["time_median_ms"].dtype), "float64")
        self.assertTrue(math.isnan(self.df.loc[self.df["sample_idx"] == "2", "time_median_ms"].iloc[0]))

    def test_group_columns_and_keys(self) -> None:
        groups = aggregate_groups(self.df)
        self.assertEqual(list(groups.columns), [*GROUP_KEY, *METRIC_COLUMNS])
        self.assertEqual(len(groups), 3)  # (A, numpy), (A, c_seq), (B, numpy)
        self.assertEqual(set(groups["turn"]), {"0"})

    def test_numpy_group_numbers(self) -> None:
        g = _group(aggregate_groups(self.df), "numpy")
        self.assertEqual(g["n_rows"], 6)
        self.assertEqual(g["n"], 5)             # 절단 행은 valid_sample=0
        self.assertEqual(g["n_pass"], 3)
        self.assertAlmostEqual(g["pass_at_1"], 0.6)
        self.assertAlmostEqual(g["pass_at_5"], 1.0)   # n−c=2 < 5
        self.assertAlmostEqual(g["pass_at_n"], 1.0)
        self.assertEqual(g["n_generate_failed"], 0)
        self.assertEqual(g["n_refusal"], 0)
        self.assertEqual(g["n_truncated"], 1)
        self.assertEqual(g["benchmarked"], 3)
        self.assertEqual(g["n_repeats_reduced"], 1)
        self.assertEqual(g["n_protocol_flagged"], 2)  # protocol_ok=0 한 행 + baseline_drift 플래그 한 행 (turbo=on 은 정보용)
        self.assertAlmostEqual(g["time_median_ms"], 15.0)   # (14 + 16)/2; timeout 행의 NA 제외
        self.assertAlmostEqual(g["speedup_median"], 0.985)
        self.assertAlmostEqual(g["speedup_min"], 0.92)
        self.assertAlmostEqual(g["speedup_max"], 1.05)
        self.assertAlmostEqual(g["speedup_1t_median"], 3.4)
        self.assertAlmostEqual(g["tokens_out_mean"], (500 + 700 + 900 + 300 + 400 + 32000) / 6)
        self.assertAlmostEqual(g["tokens_text_mean"], (400 + 600 + 800 + 300 + 400) / 5)
        self.assertAlmostEqual(g["tokens_per_pass"], (6 * 1000 + 34800) / 3)
        self.assertAlmostEqual(g["tokens_out_per_pass"], 34800 / 3)
        self.assertEqual(g["error_types"], "numeric_instability:1, wrong_output:1")  # 수 같으면 이름순

    def test_c_seq_group_numbers(self) -> None:
        g = _group(aggregate_groups(self.df), "c_seq")
        self.assertEqual(g["n_rows"], 3)
        self.assertEqual(g["n"], 2)             # 거절은 valid, API 오류는 invalid
        self.assertEqual(g["n_pass"], 0)
        self.assertAlmostEqual(g["pass_at_1"], 0.0)
        self.assertTrue(math.isnan(g["pass_at_5"]))   # k=5 > n=2 → NA
        self.assertAlmostEqual(g["pass_at_n"], 0.0)
        self.assertEqual(g["n_generate_failed"], 1)
        self.assertEqual(g["n_refusal"], 1)
        self.assertEqual(g["n_truncated"], 0)
        self.assertEqual(g["benchmarked"], 0)
        self.assertEqual(g["n_protocol_flagged"], 0)
        self.assertTrue(math.isnan(g["time_median_ms"]))
        self.assertTrue(math.isnan(g["speedup_median"]))
        self.assertAlmostEqual(g["tokens_out_mean"], (50 + 600) / 2)
        self.assertAlmostEqual(g["tokens_text_mean"], (50 + 500) / 2)
        self.assertTrue(math.isnan(g["tokens_per_pass"]))   # n_pass=0
        self.assertEqual(g["error_types"], "compile_error:1")  # 거절·API 오류 행은 error_type 빈칸이라 제외

    def test_run_summary_and_warnings(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            runs = summarize_runs(self.df)
        self.assertEqual([r["run_id"] for r in runs], [RUN_B, RUN_A])
        a = next(r for r in runs if r["run_id"] == RUN_A)
        self.assertEqual(a["n_rows"], 9)
        self.assertEqual(a["n_valid"], 7)
        self.assertEqual(a["n_pass"], 3)
        self.assertEqual(a["languages"], ["c_seq", "numpy"])
        self.assertEqual(a["model_served"], "dummy|dummy-2")
        self.assertEqual(a["effort"], "medium")
        self.assertAlmostEqual(a["baseline_run_time_ms"], 14.7)
        self.assertAlmostEqual(a["baseline_time_1t_ms"], 51.0)
        self.assertAlmostEqual(a["baseline_paired_mean_ms"], (14.5 + 15.1 + 14.9) / 3)
        self.assertEqual(a["baseline_paired_n"], 3)
        self.assertAlmostEqual(a["baseline_drift_abs_max_pct"], 2.72)
        self.assertTrue(any("model_served" in w for w in a["warnings"]))
        self.assertFalse(any("effort" in w for w in a["warnings"]))
        self.assertIn("model_served", err.getvalue())
        b = next(r for r in runs if r["run_id"] == RUN_B)
        self.assertEqual(b["warnings"], [])
        self.assertEqual(b["baseline_paired_n"], 0)
        self.assertTrue(math.isnan(b["baseline_paired_mean_ms"]))

    def test_selection(self) -> None:
        self.assertEqual(latest_run_id(self.df), RUN_A)
        sel, ids = select_runs(self.df)
        self.assertEqual(ids, [RUN_A])
        self.assertEqual(set(sel["run_id"]), {RUN_A})
        sel, ids = select_runs(self.df, run_id=RUN_B)
        self.assertEqual(ids, [RUN_B])
        self.assertEqual(len(sel), 1)
        sel, ids = select_runs(self.df, all_runs=True)
        self.assertEqual(ids, [RUN_B, RUN_A])
        self.assertEqual(len(sel), 10)
        with self.assertRaises(Exception) as ctx:
            select_runs(self.df, run_id="없는-run")
        self.assertIn(RUN_A, str(ctx.exception))


class CliTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aggregate_cli_")
        self.tmp = Path(self._tmp.name)
        self.csv = self.tmp / "runs.csv"
        _write_csv(self.csv, _rows())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _run(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = main(list(args))
        return rc, out.getvalue(), err.getvalue()

    def test_json_latest_default(self) -> None:
        rc, out, _ = self._run("--csv", str(self.csv), "--json")
        self.assertEqual(rc, EXIT_OK)
        report = json.loads(out)
        self.assertEqual(report["selection"], {"mode": "latest", "run_ids": [RUN_A]})
        self.assertEqual(len(report["runs"]), 1)
        self.assertEqual(len(report["groups"]), 2)
        numpy_group = next(g for g in report["groups"] if g["language"] == "numpy")
        self.assertEqual(numpy_group["n_pass"], 3)
        self.assertAlmostEqual(numpy_group["pass_at_1"], 0.6)
        c_seq_group = next(g for g in report["groups"] if g["language"] == "c_seq")
        self.assertIsNone(c_seq_group["pass_at_5"])        # NaN → null
        self.assertIsNone(c_seq_group["time_median_ms"])

    def test_text_run_id_and_all_runs(self) -> None:
        rc, out, _ = self._run("--csv", str(self.csv), "--run-id", RUN_B)
        self.assertEqual(rc, EXIT_OK)
        self.assertIn(f"== run {RUN_B}", out)
        self.assertNotIn(RUN_A, out)
        self.assertIn("pass_at_1", out)
        rc, out, _ = self._run("--csv", str(self.csv), "--all-runs")
        self.assertEqual(rc, EXIT_OK)
        self.assertIn(f"== run {RUN_A}", out)
        self.assertIn(f"== run {RUN_B}", out)
        self.assertIn("numeric_instability:1, wrong_output:1", out)

    def test_unknown_run_id(self) -> None:
        rc, _, err = self._run("--csv", str(self.csv), "--run-id", "nope")
        self.assertEqual(rc, EXIT_NOT_FOUND)
        self.assertIn("nope", err)

    def test_missing_csv(self) -> None:
        rc, _, err = self._run("--csv", str(self.tmp / "absent.csv"))
        self.assertEqual(rc, EXIT_NOT_FOUND)
        self.assertIn("CSV가 없다", err)

    def test_header_only_and_zero_byte_csv(self) -> None:
        empty = self.tmp / "empty.csv"
        _write_csv(empty, [])
        rc, out, err = self._run("--csv", str(empty))
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("행이 없다", err)
        self.assertIn("행 없음", out)
        rc, out, _ = self._run("--csv", str(empty), "--json")
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual(json.loads(out)["groups"], [])
        zero = self.tmp / "zero.csv"
        zero.write_bytes(b"")
        rc, _, err = self._run("--csv", str(zero))
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("행이 없다", err)

    def test_missing_required_column_is_schema_error(self) -> None:
        bad = self.tmp / "bad.csv"
        bad.write_text("run_id,foo\n1,2\n", encoding="utf-8")
        rc, _, err = self._run("--csv", str(bad))
        self.assertEqual(rc, 2)
        self.assertIn("rebuild-csv", err)


# ============================================================================= 수정 턴·조건·새 백엔드 (합성 run)

ROOT = Path(__file__).resolve().parent.parent
REAL_CSV = ROOT / "results" / "runs.csv"
DUMMY_CSV = ROOT / "results" / "runs_dummy.csv"
REAL_RUN = "20261105-real"        # 첫 실제 실행(GEMM × 3언어 × 3샘플, 9/9 통과; docs/decisions.md "M1 완료")

RUN_S = "20261120-090000-synth"
UNSUPPORTED_REASON = "sorting is not expressible as a Halide pipeline"


def csv_header() -> list[str]:
    """결과 CSV 의 열 순서. 저장소의 실제 결과 파일 첫 줄에서 읽는다(합성 행이 실제 파일과 같은 헤더를 갖도록)."""
    with REAL_CSV.open(encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def _cell(value: Any) -> str:
    """CSV 직렬화 규칙(설계 §11): NA 는 빈칸, 불리언은 1/0."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def synth_row(**over: Any) -> dict[str, str]:
    """기본값 위에 덮어쓴 합성 결과 행(열 이름 → 문자열). ``passed`` 는 ``pass`` 열이다.

    입력 토큰은 턴마다 1000씩 는다(수정 턴의 프롬프트에는 앞선 대화가 다시 들어간다).
    """
    values: dict[str, Any] = dict(
        run_id=RUN_S, created_at="2026-11-20T09:00:00+09:00", task="gemm", language="numpy", backend="python",
        args="positional", grammar_hint="off", feedback="compiler", sample_idx=0, turn=0,
        provider="dummy", model="synth-model", model_served="synth-model", effort="medium",
        valid_sample=True, passed=False, bench_ran=False, repeats_reduced=False, protocol_ok=True, protocol_flags="[]",
        stop_reason="end_turn", tokens_in=1000 * (int(over.get("turn", 0)) + 1), tokens_out=100,
        baseline_run_time_ms=10.0, baseline_time_1t_ms=30.0,
    )
    values.update(over)
    values["pass"] = values.pop("passed")
    header = csv_header()
    unknown = sorted(set(values) - set(header))
    assert not unknown, f"결과 CSV 에 없는 열: {unknown}"
    return {column: _cell(values.get(column)) for column in header}


def _ok(sample_idx: int, speedup: float, turn: int = 0, **key: Any) -> dict[str, str]:
    """턴 ``turn`` 에서 통과하고 bench 까지 돈 행(대응 기준선 10 ms 대비 속도비 ``speedup``)."""
    return synth_row(
        sample_idx=sample_idx, turn=turn, passed=True, turns_to_pass=turn, bench_ran=True,
        time_median_ms=10.0 / speedup, baseline_time_ms=10.0, speedup_vs_baseline=speedup, **key,
    )


def _fail(sample_idx: int, error_type: str, stage: str = "check", turn: int = 0, **key: Any) -> dict[str, str]:
    """턴 ``turn`` 에서 ``error_type`` 으로 실패한 행(수정 루프의 대상)."""
    return synth_row(sample_idx=sample_idx, turn=turn, error_type=error_type, error_stage=stage, **key)


def _refusal(sample_idx: int, **key: Any) -> dict[str, str]:
    """거절: 유효한 샘플이지만 통과가 아니고 수정 루프의 대상도 아니다."""
    return synth_row(sample_idx=sample_idx, error_stage="generate", error_detail="refusal:policy", stop_reason="refusal", **key)


def _api_error(sample_idx: int, turn: int = 0, **key: Any) -> dict[str, str]:
    """API 오류: 무효 샘플(valid_sample=0), 토큰 기록 없음."""
    return synth_row(
        sample_idx=sample_idx, turn=turn, valid_sample=False, error_stage="generate", stop_reason="",
        api_error="transient:APIConnectionError:None:timeout", tokens_in=None, tokens_out=None, **key,
    )


def synthetic_rows() -> list[dict[str, str]]:
    """수정 턴·조건 수준·새 백엔드가 섞인 run 하나의 합성 결과 54행(턴 0 이 45행, 수정 턴이 9행).

    그룹마다 샘플 0..4 (숫자는 기준선 대비 속도비):

    =====  ===========  ================  ==========================================================================
    과제   언어         조건              내용
    =====  ===========  ================  ==========================================================================
    gemm   numpy        (기준)            턴 0: 통과 1.00 0.90 1.10, wrong_output, compile_error → s3 은 턴 1(0.80), s4 는 턴 2(0.50)에서 통과
    gemm   numpy        args=keyword      턴 0: 다섯 개 모두 통과(1.00)
    gemm   c_seq        (기준)            턴 0: 통과 0.30 0.35, compile_error, runtime_crash, 거절 → s2 는 턴 2 까지 실패, s3 은 턴 1(0.25)에서 통과
    gemm   c_seq        fb=compiler+lint  턴 0 은 위와 같은 응답(0.31 0.34) → s2(0.28)·s3(0.26) 가 턴 1 에서 통과
    gemm   c_seq        grammar_hint=on   턴 0: 통과 세 개(0.30), wrong_output 두 개; 수정 루프 없음
    gemm   rust         (기준)            턴 0: 통과 1.20 1.50, 통과했지만 bench timeout, race_condition, API 오류(무효); 수정 루프 없음
    gemm   halide_auto  (기준)            턴 0: 다섯 개 모두 통과(2.0 2.5 3.0 3.5 4.0)
    sort   numpy        (기준)            턴 0: 통과 0.5 0.6 0.7 0.8, wrong_output → s4 의 턴 1 호출이 API 오류(무효)
    sort   c_seq        (기준)            턴 0: 통과 0.20, wrong_output 두 개, timeout(check), signature_mismatch; 수정 루프 없음
    =====  ===========  ================  ==========================================================================

    기준 조건은 args=positional, grammar_hint=off, feedback=compiler. sort × rust 는 행이 없고, sort × halide_auto 는
    ``write_task_specs`` 의 과제 명세가 unsupported 로 적는다.
    """
    numpy = dict(task="gemm", language="numpy", backend="python")
    numpy_kw = dict(numpy, args="keyword")
    c_seq = dict(task="gemm", language="c_seq", backend="c")
    c_seq_lint = dict(c_seq, feedback="compiler+lint")
    c_seq_hint = dict(c_seq, grammar_hint="on")
    rust = dict(task="gemm", language="rust", backend="rust")
    halide = dict(task="gemm", language="halide_auto", backend="halide")
    sort_numpy = dict(task="sort", language="numpy", backend="python")
    sort_c = dict(task="sort", language="c_seq", backend="c")
    return [
        _ok(0, 1.00, **numpy), _ok(1, 0.90, **numpy), _ok(2, 1.10, **numpy),
        _fail(3, "wrong_output", **numpy), _ok(3, 0.80, turn=1, **numpy),
        _fail(4, "compile_error", "build", **numpy), _fail(4, "wrong_output", turn=1, **numpy), _ok(4, 0.50, turn=2, **numpy),
        *(_ok(i, 1.00, **numpy_kw) for i in range(5)),
        _ok(0, 0.30, **c_seq), _ok(1, 0.35, **c_seq),
        _fail(2, "compile_error", "build", **c_seq), _fail(2, "compile_error", "build", turn=1, **c_seq),
        _fail(2, "compile_error", "build", turn=2, **c_seq),
        _fail(3, "runtime_crash", **c_seq), _ok(3, 0.25, turn=1, **c_seq),
        _refusal(4, **c_seq),
        _ok(0, 0.31, **c_seq_lint), _ok(1, 0.34, **c_seq_lint),
        _fail(2, "compile_error", "build", **c_seq_lint), _ok(2, 0.28, turn=1, **c_seq_lint),
        _fail(3, "runtime_crash", **c_seq_lint), _ok(3, 0.26, turn=1, **c_seq_lint),
        _refusal(4, **c_seq_lint),
        *(_ok(i, 0.30, **c_seq_hint) for i in range(3)), _fail(3, "wrong_output", **c_seq_hint), _fail(4, "wrong_output", **c_seq_hint),
        _ok(0, 1.20, **rust), _ok(1, 1.50, **rust),
        synth_row(sample_idx=2, passed=True, turns_to_pass=0, bench_ran=True, error_type="timeout", error_stage="bench",
                  error_detail="first_call_200000ms_projected_800s", **rust),
        _fail(3, "race_condition", **rust), _api_error(4, **rust),
        *(_ok(i, speedup, **halide) for i, speedup in enumerate((2.0, 2.5, 3.0, 3.5, 4.0))),
        *(_ok(i, speedup, **sort_numpy) for i, speedup in enumerate((0.5, 0.6, 0.7, 0.8))),
        _fail(4, "wrong_output", **sort_numpy), _api_error(4, turn=1, **sort_numpy),
        _ok(0, 0.20, **sort_c), _fail(1, "wrong_output", **sort_c), _fail(2, "wrong_output", **sort_c),
        _fail(3, "timeout", **sort_c), _fail(4, "signature_mismatch", "extract", **sort_c),
    ]


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """합성 행을 실제 결과 파일과 같은 헤더의 CSV 로 쓴다."""
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_header(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_task_specs(tasks_dir: Path) -> Path:
    """합성 run 의 과제 명세 디렉터리: gemm 은 제한 없음, sort 는 Halide 두 조건이 unsupported."""
    for task, unsupported in (("gemm", ""), ("sort", f'unsupported:\n  halide_auto: "{UNSUPPORTED_REASON}"\n  halide_manual: "{UNSUPPORTED_REASON}"\n')):
        (tasks_dir / task).mkdir(parents=True)
        (tasks_dir / task / "signatures.yaml").write_text(f"task: {task}\nfunction: {task}\n{unsupported}", encoding="utf-8")
    return tasks_dir


def _pick(records: Any, **where: Any) -> dict[str, Any]:
    """표(DataFrame 또는 레코드 리스트)에서 조건에 맞는 단 하나의 행."""
    rows = records.to_dict(orient="records") if hasattr(records, "to_dict") else records
    found = [r for r in rows if all(r[k] == v for k, v in where.items())]
    assert len(found) == 1, f"행이 하나여야 한다: {where} → {len(found)}"
    return found[0]


def _run_main(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = main(list(args))
    return rc, out.getvalue(), err.getvalue()


# 그룹별 기대값: (n, 턴 0..2 이하 누적 통과율, turns_to_pass 분포, 턴 0..2 이하 누적 토큰). None = 턴 t 까지 시도하지 않은
# 실패 샘플이 있어 모른다. 키는 (task, language, args, grammar_hint, feedback), 순서는 표의 행 순서(언어는 부담 순).
# 토큰: 턴 0 행은 1000 + 100, 턴 1 행은 2000 + 100, 턴 2 행은 3000 + 100.
SYNTH_BY_TURN: dict[tuple[str, ...], tuple[int, list[float | None], str, list[float | None]]] = {
    ("gemm", "numpy", "positional", "off", "compiler"): (5, [0.6, 0.8, 1.0], "0:3, 1:1, 2:1", [5500, 9700, 12800]),
    ("gemm", "numpy", "keyword", "off", "compiler"): (5, [1.0, 1.0, 1.0], "0:5", [5500, 5500, 5500]),
    ("gemm", "halide_auto", "positional", "off", "compiler"): (5, [1.0, 1.0, 1.0], "0:5", [5500, 5500, 5500]),
    # s2 는 턴 2 까지 시도했고 s4(거절)는 수정 대상이 아니다 → 턴 2 까지 다 안다
    ("gemm", "c_seq", "positional", "off", "compiler"): (5, [0.4, 0.6, 0.6], "0:2, 1:1, NA:2", [5500, 9700, 12800]),
    ("gemm", "c_seq", "positional", "off", "compiler+lint"): (5, [0.4, 0.8, 0.8], "0:2, 1:2, NA:1", [5500, 9700, 9700]),
    # 실패한 s3·s4 를 고쳐 보지 않았다 → 턴 1 부터 모른다
    ("gemm", "c_seq", "positional", "on", "compiler"): (5, [0.6, None, None], "0:3, NA:2", [5500, None, None]),
    # API 오류 s4 는 분모에서 빠지고(토큰 기록도 없다) race_condition s3 은 고쳐 보지 않았다
    ("gemm", "rust", "positional", "off", "compiler"): (4, [0.75, None, None], "0:3, NA:1", [4400, None, None]),
    # s4 의 턴 1 호출이 API 오류 → 턴 1 의 결과를 모른다
    ("sort", "numpy", "positional", "off", "compiler"): (5, [0.8, None, None], "0:4, NA:1", [5500, None, None]),
    ("sort", "c_seq", "positional", "off", "compiler"): (5, [0.2, None, None], "0:1, NA:4", [5500, None, None]),
}
_CONDITION_KEY = ("task", "language", "args", "grammar_hint", "feedback")


class SyntheticRunCase(unittest.TestCase):
    """합성 run 을 임시 CSV 로 쓰고 읽어 두는 공통 준비."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="aggregate_synth_")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.csv = self.tmp / "runs.csv"
        write_rows(self.csv, synthetic_rows())
        self.tasks_dir = write_task_specs(self.tmp / "tasks")
        self.df = load_runs(self.csv)

    def assert_values(self, got: Sequence[Any], want: Sequence[float | None]) -> None:
        """숫자 리스트 비교. ``None`` 은 NaN 을 기대한다."""
        self.assertEqual(len(got), len(want))
        for g, w in zip(got, want):
            if w is None:
                self.assertTrue(math.isnan(g), f"NaN 이어야 한다: {got}")
            else:
                self.assertAlmostEqual(g, w, places=12)


class LanguageOrderTest(unittest.TestCase):
    def test_performance_decision_burden_order(self) -> None:
        self.assertEqual(LANGUAGE_ORDER, (
            "numpy", "numba", "halide_auto", "halide_manual", "c_openmp", "c_seq", "rust",
            "opencl", "c_neon", "llvm_ir", "aarch64_asm",
        ))

    def test_unknown_languages_come_last_in_name_order(self) -> None:
        mixed = ["zig", "rust", "aarch64_asm", "ada", "numpy", "c_seq", "llvm_ir", "c_openmp", "halide_manual"]
        self.assertEqual(
            sorted(mixed, key=language_sort_key),
            ["numpy", "halide_manual", "c_openmp", "c_seq", "rust", "llvm_ir", "aarch64_asm", "ada", "zig"],
        )

    def test_condition_levels_put_the_baseline_first(self) -> None:
        self.assertEqual(sorted(["keyword", "positional"], key=lambda v: level_sort_key("args", v)), ["positional", "keyword"])
        self.assertEqual(sorted(["on", "off"], key=lambda v: level_sort_key("grammar_hint", v)), ["off", "on"])
        self.assertEqual(
            sorted(["compiler+lint", "another", "compiler"], key=lambda v: level_sort_key("feedback", v)),
            ["compiler", "compiler+lint", "another"],
        )


class FirstAttemptTableTest(SyntheticRunCase):
    """수정 루프 행이 있으면 기본 그룹 표는 첫 시도(turn 0)만 센다."""

    def test_dataset_shape(self) -> None:
        self.assertEqual(len(self.df), 54)
        self.assertEqual(len(first_attempt_rows(self.df)), 45)
        self.assertEqual(set(self.df["backend"]), {"python", "c", "rust", "halide"})
        self.assertEqual(set(self.df["turn"]), {"0", "1", "2"})

    def test_group_table_counts_turn_zero_only(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv), "--json")
        self.assertEqual(rc, EXIT_OK)
        report = json.loads(out)
        self.assertEqual(len(report["groups"]), 9)
        self.assertEqual({g["turn"] for g in report["groups"]}, {"0"})
        g = _pick(report["groups"], task="gemm", language="numpy", args="positional")
        self.assertEqual((g["n_rows"], g["n"], g["n_pass"]), (5, 5, 3))   # 턴 1·2 에서 통과한 s3·s4 는 세지 않는다
        self.assertAlmostEqual(g["pass_at_1"], 0.6)
        self.assertAlmostEqual(g["speedup_median"], 1.0)                  # 턴 0 통과 샘플 1.00 0.90 1.10
        self.assertAlmostEqual(g["tokens_per_pass"], 5 * 1100 / 3)
        self.assertEqual(g["error_types"], "compile_error:1, wrong_output:1")
        g = _pick(report["groups"], task="gemm", language="c_seq", feedback="compiler+lint")
        self.assertEqual((g["n"], g["n_pass"], g["n_refusal"]), (5, 2, 1))
        self.assertEqual(len(report["notes"]), 1)
        self.assertIn("turn ≥ 1", report["notes"][0])
        self.assertIn("9개", report["notes"][0])
        # run 요약은 모든 턴의 행을 센다: 무효 2행(API 오류), 통과 행 33개
        run = report["runs"][0]
        self.assertEqual((run["n_rows"], run["n_valid"], run["n_pass"]), (54, 52, 33))

    def test_text_report_says_so_in_one_line(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv))
        self.assertEqual(rc, EXIT_OK)
        lines = out.splitlines()
        notes = [line for line in lines if line.startswith("주: ")]
        self.assertEqual(len(notes), 1)
        self.assertIn("첫 시도(turn 0)만", notes[0])
        self.assertIn("--by-turn", notes[0])
        self.assertIn("그룹 9개", lines[0])
        self.assertTrue(any(line.startswith("공통: ") and "turn=0" in line for line in lines))

    def test_nothing_changes_without_repair_rows(self) -> None:
        write_rows(self.csv, [r for r in synthetic_rows() if r["turn"] == "0"])
        rc, out, _ = _run_main("--csv", str(self.csv))
        self.assertEqual(rc, EXIT_OK)
        self.assertNotIn("주: ", out)
        rc, out, _ = _run_main("--csv", str(self.csv), "--json")
        report = json.loads(out)
        self.assertEqual(list(report), ["csv", "selection", "runs", "groups"])
        self.assertEqual(list(report["groups"][0]), [*GROUP_KEY, *METRIC_COLUMNS])
        self.assertEqual(report["runs"][0]["n_rows"], 45)


class ByTurnTest(SyntheticRunCase):
    def test_sample_outcomes(self) -> None:
        samples = sample_outcomes(self.df)
        self.assertEqual(len(samples), 44)   # 턴 0 행 45개 중 rust s4(API 오류)만 빠진다
        base = dict(args="positional", grammar_hint="off", feedback="compiler")

        def outcome(task: str, language: str, sample_idx: int, **cond: str) -> list[float]:
            row = _pick(samples, task=task, language=language, sample_idx=str(sample_idx), **{**base, **cond})
            return [row["pass_turn"], row["known_until"]]

        # [통과한 턴, 결과를 아는 마지막 턴 수]
        self.assert_values(outcome("gemm", "numpy", 0), [0, math.inf])
        self.assert_values(outcome("gemm", "numpy", 4), [2, math.inf])
        self.assert_values(outcome("gemm", "c_seq", 2), [None, 2])              # 턴 2 까지 실패
        self.assert_values(outcome("gemm", "c_seq", 4), [None, math.inf])       # 거절: 더 볼 턴이 없다
        self.assert_values(outcome("gemm", "rust", 3), [None, 0])               # 고쳐 보지 않은 실패
        self.assert_values(outcome("gemm", "rust", 2), [0, math.inf])           # bench timeout 은 통과
        self.assert_values(outcome("sort", "numpy", 4), [None, 0])              # 턴 1 호출이 무효
        self.assert_values(outcome("gemm", "c_seq", 2, feedback="compiler+lint"), [1, math.inf])

    def test_table(self) -> None:
        table = by_turn_table(self.df)
        self.assertEqual(list(table.columns), [
            *BY_TURN_KEY, "n", "pass_within_0", "pass_within_1", "pass_within_2", "turns_to_pass",
            "tokens_cum_0", "tokens_cum_1", "tokens_cum_2",
        ])
        records = table.to_dict(orient="records")
        self.assertEqual([tuple(r[k] for k in _CONDITION_KEY) for r in records], list(SYNTH_BY_TURN))
        for rec, (key, (n, rates, distribution, tokens)) in zip(records, SYNTH_BY_TURN.items()):
            with self.subTest(group=key):
                self.assertEqual(rec["run_id"], RUN_S)
                self.assertEqual(rec["n"], n)
                self.assert_values([rec[f"pass_within_{t}"] for t in range(3)], rates)
                self.assertEqual(rec["turns_to_pass"], distribution)
                self.assert_values([rec[f"tokens_cum_{t}"] for t in range(3)], tokens)

    def test_first_column_equals_pass_at_1_and_turns_stop_at_the_data(self) -> None:
        only_first = [r for r in synthetic_rows() if r["turn"] == "0"]
        write_rows(self.csv, only_first)
        df = load_runs(self.csv)
        table = by_turn_table(df)
        self.assertEqual(
            list(table.columns), [*BY_TURN_KEY, "n", "pass_within_0", "turns_to_pass", "tokens_cum_0"]
        )
        groups = aggregate_groups(df)
        for rec in table.to_dict(orient="records"):
            g = _pick(groups, **{k: rec[k] for k in BY_TURN_KEY})
            self.assertAlmostEqual(rec["pass_within_0"], g["pass_at_1"])
            self.assertEqual(rec["n"], g["n"])

    def test_rows_after_the_first_passing_turn_are_ignored_with_one_warning(self) -> None:
        # s0 은 턴 0, s3 은 턴 1 에서 통과했다 → 뒤 턴 행은 앞 턴을 --force 로 다시 처리한 뒤 남은 낡은 행
        numpy = dict(task="gemm", language="numpy", backend="python")
        write_rows(self.csv, synthetic_rows() + [_fail(3, "wrong_output", turn=2, **numpy), _fail(0, "wrong_output", turn=3, **numpy)])
        df = load_runs(self.csv)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            table = by_turn_table(df)
        self.assertEqual(err.getvalue().count("\n"), 1)
        self.assertIn("처음 통과한 턴보다 뒤 턴의 행 2개", err.getvalue())
        self.assertNotIn("pass_within_3", table.columns)    # 낡은 턴 3 행이 열을 늘리지 않는다
        rec = _pick(table, task="gemm", language="numpy", args="positional", feedback="compiler")
        _n, rates, distribution, tokens = SYNTH_BY_TURN[("gemm", "numpy", "positional", "off", "compiler")]
        self.assert_values([rec[f"pass_within_{t}"] for t in range(3)], rates)
        self.assertEqual(rec["turns_to_pass"], distribution)
        self.assert_values([rec[f"tokens_cum_{t}"] for t in range(3)], tokens)   # 낡은 행의 토큰을 더하지 않는다
        rc, _, err_text = _run_main("--csv", str(self.csv), "--by-turn", "--conditions")
        self.assertEqual(rc, EXIT_OK)
        self.assertEqual(err_text.count("처음 통과한 턴보다 뒤 턴의 행"), 1)

    def test_group_without_a_valid_first_attempt(self) -> None:
        write_rows(self.csv, [_api_error(0), _api_error(1)])
        rec = by_turn_table(load_runs(self.csv)).iloc[0].to_dict()
        self.assertEqual(rec["n"], 0)
        self.assertTrue(math.isnan(rec["pass_within_0"]))
        self.assertEqual(rec["turns_to_pass"], "")

    def test_cli_text(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv), "--by-turn")
        self.assertEqual(rc, EXIT_OK)
        lines = out.splitlines()
        title = next(i for i, line in enumerate(lines) if line.startswith("# by-turn:"))
        section = lines[title + 1:]
        self.assertTrue(section[0].startswith("공통: "))
        self.assertIn("pass_within_2", section[1])
        self.assertIn("tokens_cum_2", section[1])
        self.assertEqual(len(section), 2 + len(SYNTH_BY_TURN))
        halide = next(line for line in section if "halide_auto" in line)
        self.assertRegex(halide, r"\s5\s+1\.000\s+1\.000\s+1\.000\s+0:5\s+5500\s+5500\s+5500$")
        rust = next(line for line in section if " rust " in line)
        self.assertRegex(rust, r"\s4\s+0\.750\s+NA\s+NA\s+0:3, NA:1\s+4400\s+NA\s+NA$")
        self.assertEqual(sum(line.startswith("주: ") for line in lines), 1)   # 기본 표의 안내는 그대로 한 줄

    def test_cli_json(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv), "--by-turn", "--json")
        self.assertEqual(rc, EXIT_OK)
        report = json.loads(out)
        self.assertEqual(len(report["by_turn"]), len(SYNTH_BY_TURN))
        rec = _pick(report["by_turn"], task="sort", language="numpy")
        self.assertEqual((rec["pass_within_0"], rec["pass_within_1"], rec["tokens_cum_2"]), (0.8, None, None))   # NaN → null
        self.assertNotIn("conditions", report)
        self.assertNotIn("coverage", report)


class ConfidenceIntervalTest(SyntheticRunCase):
    """``--ci``: 구간의 기대값은 tests/test_stats.py 와 같은 손계산이다."""

    def test_columns_follow_their_point_estimates(self) -> None:
        self.assertEqual(metric_columns(), list(METRIC_COLUMNS))
        with_ci = metric_columns(ci=True)
        i = with_ci.index("pass_at_1")
        self.assertEqual(with_ci[i:i + 3], ["pass_at_1", "pass_at_1_lo", "pass_at_1_hi"])
        i = with_ci.index("speedup_median")
        self.assertEqual(with_ci[i:i + 3], ["speedup_median", "speedup_median_lo", "speedup_median_hi"])
        self.assertEqual(len(with_ci), len(METRIC_COLUMNS) + 4)
        groups = aggregate_groups(first_attempt_rows(self.df), ci=True)
        self.assertEqual(list(groups.columns), [*GROUP_KEY, *with_ci])
        self.assertEqual(list(aggregate_groups(self.df.iloc[:0], ci=True).columns), [*GROUP_KEY, *with_ci])

    def test_intervals(self) -> None:
        groups = aggregate_groups(first_attempt_rows(self.df), ci=True)
        base = dict(args="positional", grammar_hint="off", feedback="compiler")
        # (pass_at_1 구간, speedup_median 과 그 구간)
        expected = {
            ("gemm", "numpy"): ((0.2, 1.0), (1.0, 0.9, 1.1)),          # 3/5; 통과 3개의 중앙값 구간 = [min, max]
            ("gemm", "halide_auto"): ((1.0, 1.0), (3.0, 2.0, 4.0)),    # 5/5 → 폭 0; 서로 다른 5개 → [min, max]
            ("gemm", "c_seq"): ((0.0, 0.8), (0.325, 0.30, 0.35)),      # 2/5
            ("gemm", "rust"): ((0.25, 1.0), (1.35, 1.2, 1.5)),         # 3/4; 시간이 있는 통과 샘플은 2개
            ("sort", "numpy"): ((0.4, 1.0), (0.65, 0.5, 0.8)),         # 4/5
            ("sort", "c_seq"): ((0.0, 0.6), (0.2, 0.2, 0.2)),          # 1/5; 통과 1개 → 퇴화 구간
        }
        for (task, language), (pass_ci, speedup) in expected.items():
            with self.subTest(task=task, language=language):
                g = _pick(groups, task=task, language=language, **base)
                self.assert_values([g["pass_at_1_lo"], g["pass_at_1_hi"]], list(pass_ci))
                self.assert_values([g["speedup_median"], g["speedup_median_lo"], g["speedup_median_hi"]], list(speedup))
                self.assertLessEqual(g["pass_at_1_lo"], g["pass_at_1"])
                self.assertLessEqual(g["pass_at_1"], g["pass_at_1_hi"])

    def test_group_without_samples_has_no_interval(self) -> None:
        write_rows(self.csv, [_api_error(0), _fail(1, "wrong_output", language="c_seq", backend="c")])
        groups = aggregate_groups(load_runs(self.csv), ci=True)
        g = _pick(groups, language="numpy")      # 유효 샘플 0개
        self.assert_values([g["pass_at_1_lo"], g["pass_at_1_hi"], g["speedup_median_lo"], g["speedup_median_hi"]], [None] * 4)
        g = _pick(groups, language="c_seq")      # 통과 0/1 → 통과율 구간은 [0, 0], 속도비는 없음
        self.assert_values([g["pass_at_1_lo"], g["pass_at_1_hi"], g["speedup_median_lo"], g["speedup_median_hi"]], [0.0, 0.0, None, None])

    def test_cli(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv), "--ci")
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("pass_at_1 pass_at_1_lo pass_at_1_hi", out)
        self.assertIn("speedup_median speedup_median_lo speedup_median_hi", out)
        notes = [line for line in out.splitlines() if line.startswith("주: ")]
        self.assertEqual(len(notes), 2)   # 첫 시도 안내 + 구간 읽는 법
        self.assertIn("95 % 백분위 부트스트랩 구간(재추출 2000회, seed 0)", notes[1])   # analysis.stats 의 기본값
        rc, out, _ = _run_main("--csv", str(self.csv), "--ci", "--json")
        g = _pick(json.loads(out)["groups"], task="gemm", language="numpy", args="positional")
        self.assertEqual((g["pass_at_1_lo"], g["pass_at_1_hi"]), (0.2, 1.0))
        rc, out, _ = _run_main("--csv", str(self.csv))
        self.assertNotIn("pass_at_1_lo", out)


class ConditionEffectsTest(SyntheticRunCase):
    def effect(self, table: Any, factor: str) -> dict[str, Any]:
        return _pick(table, factor=factor)

    def test_table(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            table = condition_effects(self.df)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(list(table.columns), list(EFFECT_COLUMNS))
        self.assertEqual(
            [(r["task"], r["language"], r["factor"]) for r in table.to_dict(orient="records")],
            [("gemm", "numpy", "args"), ("gemm", "c_seq", "grammar_hint"), ("gemm", "c_seq", "feedback")],
        )
        self.assertEqual(set(table["provider"]), {"dummy"})
        self.assertEqual(set(table["model"]), {"synth-model"})

        # args: 독립 표본, 첫 시도 통과율 3/5 대 5/5 → 차 0.4, 구간 [0, 0.8] (0 을 포함한다)
        args = self.effect(table, "args")
        # 턴 0 은 feedback 과 무관하다 → args·grammar_hint 비교는 feedback 으로 나누지 않는다
        self.assertEqual((args["base"], args["alt"], args["at"]), ("positional", "keyword", "grammar_hint=off"))
        self.assertEqual((args["quantity"], args["method"], args["n_base"], args["n_alt"]), ("pass_at_1", "unpaired", 5, 5))
        self.assert_values([args["rate_base"], args["rate_alt"], args["diff"], args["ci_low"], args["ci_high"]], [0.6, 1.0, 0.4, 0.0, 0.8])

        # grammar_hint: 독립 표본, 2/5 대 3/5 → 차 0.2, 구간 [−0.4, 0.8]
        hint = self.effect(table, "grammar_hint")
        self.assertEqual((hint["base"], hint["alt"], hint["at"]), ("off", "on", "args=positional"))
        self.assertEqual((hint["quantity"], hint["method"], hint["n_base"], hint["n_alt"]), ("pass_at_1", "unpaired", 5, 5))
        self.assert_values([hint["rate_base"], hint["rate_alt"], hint["diff"], hint["ci_low"], hint["ci_high"]], [0.4, 0.6, 0.2, -0.4, 0.8])

        # feedback: sample_idx 대응. 모든 짝의 결과를 아는 수정 턴 수 H = 2 (compiler 의 s2 가 턴 2 까지 실패, 나머지는 끝났다)
        # → 턴 2 이하 통과 3/5 대 4/5 (s2 만 달라진다) → 차 0.2, 구간 [0, 0.6]
        feedback = self.effect(table, "feedback")
        self.assertEqual((feedback["base"], feedback["alt"], feedback["at"]), ("compiler", "compiler+lint", "args=positional, grammar_hint=off"))
        self.assertEqual((feedback["quantity"], feedback["method"], feedback["n_base"], feedback["n_alt"]), ("pass_within_2", "paired", 5, 5))
        self.assert_values(
            [feedback["rate_base"], feedback["rate_alt"], feedback["diff"], feedback["ci_low"], feedback["ci_high"]],
            [0.6, 0.8, 0.2, 0.0, 0.6],
        )

    def test_feedback_samples_without_a_partner_are_dropped_with_a_warning(self) -> None:
        rows = [r for r in synthetic_rows() if not (r["feedback"] == "compiler+lint" and r["sample_idx"] == "4")]
        write_rows(self.csv, rows)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            feedback = self.effect(condition_effects(load_runs(self.csv)), "feedback")
        self.assertIn("짝이 없는 샘플 1개 제외", err.getvalue())
        self.assertEqual((feedback["n_base"], feedback["n_alt"]), (4, 4))      # 거절 샘플 s4 의 짝이 없다
        self.assert_values([feedback["rate_base"], feedback["rate_alt"], feedback["diff"]], [0.75, 1.0, 0.25])

    def _feedback_of(self, rows: list[dict[str, str]]) -> tuple[dict[str, Any], str]:
        write_rows(self.csv, rows)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            table = condition_effects(load_runs(self.csv))
        return self.effect(table, "feedback"), err.getvalue()

    def test_feedback_without_a_known_repair_turn_in_both_levels_is_not_compared(self) -> None:
        c_seq_lint = dict(task="gemm", language="c_seq", backend="c", feedback="compiler+lint")
        is_c_seq = lambda r: (r["task"], r["language"], r["grammar_hint"]) == ("gemm", "c_seq", "off")
        repaired = lambda r: is_c_seq(r) and r["turn"] != "0"
        rows = synthetic_rows()
        cases = {
            # compiler+lint 의 s2 턴 1 호출이 API 오류이고 s3 은 고쳐 보지 않았다 (전에는 차 −0.2 로 셌다)
            "api_error_and_unrepaired": [r for r in rows if not (repaired(r) and r["feedback"] == "compiler+lint")]
            + [_api_error(2, turn=1, **c_seq_lint)],
            # compiler+lint 수정 루프를 돌리지 않았다
            "lint_not_repaired": [r for r in rows if not (repaired(r) and r["feedback"] == "compiler+lint")],
            # 어느 수준도 수정하지 않았다 (전에는 차 0, 구간 [0, 0] — 확실한 무효과처럼 읽혔다)
            "neither_repaired": [r for r in rows if not repaired(r)],
        }
        for name, case in cases.items():
            with self.subTest(case=name):
                feedback, err = self._feedback_of(case)
                self.assertEqual((feedback["quantity"], feedback["n_base"], feedback["n_alt"]), ("pass_within_0", 5, 5))
                self.assert_values(
                    [feedback["rate_base"], feedback["rate_alt"], feedback["diff"], feedback["ci_low"], feedback["ci_high"]], [None] * 5,
                )
                self.assertEqual(err.count("결과를 아는 수정 턴이 없어 비교하지 않는다"), 1)

    def test_feedback_levels_with_different_turn_budgets_are_compared_at_the_shared_horizon(self) -> None:
        # compiler 는 --max-turns 2 로 s2 를 턴 2 에서 고쳤고, compiler+lint 는 --max-turns 1 로 s2 가 턴 1 에서도 실패했다
        c_seq = dict(task="gemm", language="c_seq", backend="c")
        rows = []
        for r in synthetic_rows():
            if (r["language"], r["grammar_hint"], r["sample_idx"]) == ("c_seq", "off", "2") and r["turn"] != "0":
                continue
            rows.append(r)
        rows += [
            _fail(2, "compile_error", "build", turn=1, **c_seq), _ok(2, 0.27, turn=2, **c_seq),
            _fail(2, "compile_error", "build", turn=1, **dict(c_seq, feedback="compiler+lint")),
        ]
        feedback, err = self._feedback_of(rows)
        self.assertEqual(err, "")
        # H = 1: 턴 1 이하 통과 3/5 대 3/5 (compiler 의 턴 2 통과는 compiler+lint 가 시도하지 않은 턴이다; 전에는 0.8 대 0.6)
        self.assertEqual(feedback["quantity"], "pass_within_1")
        self.assert_values([feedback["rate_base"], feedback["rate_alt"], feedback["diff"]], [0.6, 0.6, 0.0])
        by_turn = by_turn_table(load_runs(self.csv))
        for level, rate in (("compiler", feedback["rate_base"]), ("compiler+lint", feedback["rate_alt"])):
            self.assertEqual(_pick(by_turn, task="gemm", language="c_seq", grammar_hint="off", feedback=level)["pass_within_1"], rate)

    def test_seeded_feedback_copies_do_not_repeat_the_turn_zero_comparisons(self) -> None:
        # gemm numpy 의 턴 0 (positional·keyword)을 --seed-from-feedback 처럼 compiler+lint 로 복사하고 수정하지 않았다
        numpy_first = [r for r in synthetic_rows() if (r["task"], r["language"], r["turn"]) == ("gemm", "numpy", "0")]
        write_rows(self.csv, synthetic_rows() + [dict(r, feedback="compiler+lint") for r in numpy_first])
        with contextlib.redirect_stderr(io.StringIO()):
            table = condition_effects(load_runs(self.csv))
        args = self.effect(table, "args")     # 같은 측정을 두 번 세지 않는다: 행이 하나
        self.assertEqual((args["at"], args["n_base"], args["n_alt"]), ("grammar_hint=off", 5, 5))
        self.assert_values([args["rate_base"], args["rate_alt"], args["diff"], args["ci_low"], args["ci_high"]], [0.6, 1.0, 0.4, 0.0, 0.8])
        numpy_feedback = table[table["language"].eq("numpy") & table["factor"].eq("feedback")]
        self.assertEqual(len(numpy_feedback), 2)            # positional·keyword — 복사본은 수정하지 않았다 → 비교 없음
        self.assertTrue(numpy_feedback["diff"].isna().all())

    def test_independent_samples_pool_over_runs_but_feedback_pairs_stay_inside_a_run(self) -> None:
        other_run = "20261120-100000-synth"
        rows = synthetic_rows()
        for r in rows:
            if r["args"] == "keyword" or r["feedback"] == "compiler+lint":
                r["run_id"] = other_run
        write_rows(self.csv, rows)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            table = condition_effects(load_runs(self.csv))
        args = self.effect(table, "args")
        self.assertEqual((args["n_base"], args["n_alt"]), (5, 5))
        self.assert_values([args["diff"], args["ci_low"], args["ci_high"]], [0.4, 0.0, 0.8])
        feedback = self.effect(table, "feedback")       # 다른 run 의 sample_idx 는 같은 응답이 아니다 → 짝 없음
        self.assertEqual((feedback["n_base"], feedback["n_alt"]), (0, 0))
        self.assert_values([feedback["rate_base"], feedback["diff"], feedback["ci_low"], feedback["ci_high"]], [None] * 4)
        self.assertIn("짝이 없는 샘플 10개 제외", err.getvalue())

    def test_single_level_conditions_give_no_rows(self) -> None:
        write_rows(self.csv, [r for r in synthetic_rows() if (r["args"], r["grammar_hint"], r["feedback"]) == ("positional", "off", "compiler")])
        table = condition_effects(load_runs(self.csv))
        self.assertTrue(table.empty)
        self.assertEqual(list(table.columns), list(EFFECT_COLUMNS))
        rc, out, _ = _run_main("--csv", str(self.csv), "--conditions")
        self.assertEqual(rc, EXIT_OK)
        self.assertIn("비교할 조건 수준 쌍 없음", out)

    def test_cli(self) -> None:
        rc, out, _ = _run_main("--csv", str(self.csv), "--conditions")
        self.assertEqual(rc, EXIT_OK)
        lines = out.splitlines()
        title = next(i for i, line in enumerate(lines) if line.startswith("# conditions:"))
        section = lines[title + 1:]
        self.assertEqual(section[0], "공통: provider=dummy  model=synth-model  task=gemm")
        self.assertEqual(len(section), 2 + 3)
        self.assertRegex(section[2], r"numpy\s+args\s+positional\s+keyword .* unpaired\s+5\s+5\s+0\.600\s+1\.000\s+0\.400\s+0\.000\s+0\.800$")
        self.assertRegex(section[4], r"c_seq\s+feedback\s+compiler\s+compiler\+lint .* paired\s+5\s+5\s+0\.600\s+0\.800\s+0\.200\s+0\.000\s+0\.600$")
        rc, out, _ = _run_main("--csv", str(self.csv), "--conditions", "--json")
        report = json.loads(out)
        self.assertEqual([r["factor"] for r in report["conditions"]], ["args", "grammar_hint", "feedback"])
        self.assertEqual(report["conditions"][1]["ci_low"], -0.4)


class CoverageTest(SyntheticRunCase):
    def test_load_unsupported(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            got = load_unsupported(["gemm", "sort"], self.tasks_dir)
        self.assertEqual(got, {"gemm": {}, "sort": {"halide_auto": UNSUPPORTED_REASON, "halide_manual": UNSUPPORTED_REASON}})
        self.assertEqual(err.getvalue(), "")

    def test_load_unsupported_warns_about_missing_or_malformed_specs(self) -> None:
        (self.tasks_dir / "bfs").mkdir()
        (self.tasks_dir / "bfs" / "signatures.yaml").write_text("task: bfs\nunsupported: [halide_auto]\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            got = load_unsupported(["bfs", "fft"], self.tasks_dir)
        self.assertEqual(got, {"bfs": {}, "fft": {}})
        self.assertIn("unsupported 가 맵이 아니다", err.getvalue())
        self.assertIn("fft 의 unsupported 언어를 모른다", err.getvalue())

    def test_the_repository_task_specs_are_readable(self) -> None:
        tasks = sorted(p.parent.name for p in (ROOT / "tasks").glob("*/signatures.yaml"))
        self.assertIn("gemm", tasks)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            got = load_unsupported(tasks)
        self.assertEqual(sorted(got), tasks)
        self.assertEqual(err.getvalue(), "")
        for task, languages in got.items():
            for language, reason in languages.items():
                self.assertTrue(language and reason.strip(), f"{task}: unsupported 항목에 언어와 이유가 있어야 한다")

    def test_table(self) -> None:
        table = coverage_table(self.df, self.tasks_dir)
        self.assertEqual(list(table.columns), list(COVERAGE_COLUMNS))
        self.assertEqual(list(dict.fromkeys(table["language"])), ["numpy", "halide_auto", "c_seq", "rust"])   # 부담 순
        cells = {(r["task"], r["language"]): r for r in table.to_dict(orient="records")}
        self.assertEqual(len(cells), 8)
        # n = 첫 시도 유효 샘플 수(조건을 합친다): numpy 5 + keyword 5, c_seq 5 × 조건 3개, rust 는 API 오류 하나가 빠진다
        counts = {key: (None if cell["status"] != "rows" else int(cell["n"])) for key, cell in cells.items()}
        self.assertEqual(counts, {
            ("gemm", "numpy"): 10, ("gemm", "halide_auto"): 5, ("gemm", "c_seq"): 15, ("gemm", "rust"): 4,
            ("sort", "numpy"): 5, ("sort", "halide_auto"): None, ("sort", "c_seq"): 5, ("sort", "rust"): None,
        })
        self.assertEqual(cells[("sort", "halide_auto")]["status"], "unsupported")
        self.assertEqual(cells[("sort", "halide_auto")]["reason"], UNSUPPORTED_REASON)
        self.assertEqual(cells[("sort", "rust")]["status"], "missing")
        self.assertEqual(cells[("sort", "rust")]["reason"], "")

    def test_matrix_text(self) -> None:
        self.assertEqual(format_coverage(coverage_table(self.df, self.tasks_dir)).splitlines(), [
            "task numpy halide_auto c_seq rust",
            "gemm    10           5    15    4",
            "sort     5         n/a     5    -",
            f"n/a sort × halide_auto: {UNSUPPORTED_REASON}",
        ])

    def test_rows_for_an_unsupported_combination_are_counted_with_a_warning(self) -> None:
        rows = synthetic_rows() + [_ok(0, 1.0, task="sort", language="halide_auto", backend="halide")]
        write_rows(self.csv, rows)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            table = coverage_table(load_runs(self.csv), self.tasks_dir)
        cell = _pick(table, task="sort", language="halide_auto")
        self.assertEqual((cell["status"], int(cell["n"])), ("rows", 1))
        self.assertIn("sort/halide_auto 는 unsupported 인데 행이 있다", err.getvalue())

    def test_cli(self) -> None:
        with mock.patch.object(aggregate, "TASKS_DIR", self.tasks_dir):
            rc, out, err = _run_main("--csv", str(self.csv), "--coverage")
            self.assertEqual(rc, EXIT_OK)
            lines = out.splitlines()
            title = next(i for i, line in enumerate(lines) if line.startswith("# coverage:"))
            self.assertEqual(lines[title + 1:], [
                "task numpy halide_auto c_seq rust",
                "gemm    10           5    15    4",
                "sort     5         n/a     5    -",
                f"n/a sort × halide_auto: {UNSUPPORTED_REASON}",
            ])
            self.assertEqual(err, "")
            rc, out, _ = _run_main("--csv", str(self.csv), "--coverage", "--json")
        cell = _pick(json.loads(out)["coverage"], task="sort", language="halide_auto")
        self.assertEqual(cell, {"task": "sort", "language": "halide_auto", "n": None, "status": "unsupported", "reason": UNSUPPORTED_REASON})
        cell = _pick(json.loads(out)["coverage"], task="gemm", language="c_seq")
        self.assertEqual((cell["n"], cell["status"]), (15, "rows"))


class OutcomeLabelsTest(SyntheticRunCase):
    def test_every_first_attempt_has_exactly_one_label(self) -> None:
        first = first_attempt_rows(self.df)
        counts = outcome_labels(first).value_counts().to_dict()
        self.assertEqual(counts, {
            "pass": 27, "pass_bench_timeout": 1, "wrong_output": 6, "compile_error": 3, "runtime_crash": 2,
            "timeout": 1, "race_condition": 1, "signature_mismatch": 1, "refusal": 2, "generate_failed": 1,
        })
        self.assertEqual(sum(counts.values()), 45)
        self.assertLessEqual(set(counts), set(OUTCOME_ORDER))

    def test_truncated_and_unknown_error_types(self) -> None:
        write_rows(self.csv, [
            synth_row(sample_idx=0, valid_sample=False, error_stage="generate", error_detail="truncated", stop_reason="max_tokens"),
            synth_row(sample_idx=1, error_stage="check", error_type="not_in_the_taxonomy"),
            synth_row(sample_idx=2, error_stage="check"),
        ])
        self.assertEqual(list(outcome_labels(load_runs(self.csv))), ["truncated", "other", "other"])


class EmptyCsvOptionsTest(unittest.TestCase):
    def test_header_only_csv_with_every_option(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aggregate_empty_") as tmp:
            empty = Path(tmp) / "empty.csv"
            write_rows(empty, [])
            rc, out, _ = _run_main("--csv", str(empty), "--ci", "--coverage", "--by-turn", "--conditions")
            self.assertEqual(rc, EXIT_OK)
            self.assertIn("행 없음", out)
            rc, out, _ = _run_main("--csv", str(empty), "--ci", "--coverage", "--by-turn", "--conditions", "--json")
        self.assertEqual(rc, EXIT_OK)
        report = json.loads(out)
        self.assertEqual((report["groups"], report["coverage"], report["by_turn"], report["conditions"]), ([], [], [], []))


class EntryPointTest(unittest.TestCase):
    def test_streams_are_set_up_in_the_entry_point_only(self) -> None:
        """``python -m analysis.aggregate`` 는 ``setup_stdio`` 를 부르고(cp949 파이프에서 죽지 않게), 테스트가 부르는 ``main()`` 은 부르지 않는다."""
        absent = str(Path(tempfile.gettempdir()) / "aggregate_absent_dir" / "absent.csv")
        with mock.patch.object(procutil, "setup_stdio") as setup, \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["--csv", absent]), EXIT_NOT_FOUND)
            setup.assert_not_called()
            with mock.patch.object(sys, "argv", ["aggregate", "--csv", absent]), warnings.catch_warnings(), \
                    self.assertRaises(SystemExit) as stop:
                warnings.simplefilter("ignore", RuntimeWarning)   # 이미 import 한 모듈을 __main__ 으로 다시 실행한다는 경고
                runpy.run_module("analysis.aggregate", run_name="__main__")
        setup.assert_called_once_with()
        self.assertEqual(stop.exception.code, EXIT_NOT_FOUND)


class RealCsvSmokeTest(unittest.TestCase):
    """저장소의 실제 결과 파일이 새 옵션과 함께 그대로 집계된다."""

    OPTIONS = ("--ci", "--coverage", "--by-turn", "--conditions")

    def test_first_real_run(self) -> None:
        rc, out, _ = _run_main("--csv", str(REAL_CSV), "--run-id", REAL_RUN, *self.OPTIONS, "--json")
        self.assertEqual(rc, EXIT_OK)
        report = json.loads(out)
        self.assertEqual([g["language"] for g in report["groups"]], ["c_openmp", "c_seq", "numpy"])   # 기본 표의 순서는 그대로
        for g in report["groups"]:
            self.assertEqual((g["n"], g["n_pass"], g["pass_at_1"]), (3, 3, 1.0))
            self.assertEqual((g["pass_at_1_lo"], g["pass_at_1_hi"]), (1.0, 1.0))      # 3/3: 변동을 보지 못했다
            self.assertEqual((g["speedup_median_lo"], g["speedup_median_hi"]), (g["speedup_min"], g["speedup_max"]))
        self.assertEqual([(c["task"], c["language"], c["n"], c["status"]) for c in report["coverage"]], [
            ("gemm", "numpy", 3, "rows"), ("gemm", "c_openmp", 3, "rows"), ("gemm", "c_seq", 3, "rows"),
        ])
        self.assertEqual([(r["language"], r["n"], r["pass_within_0"], r["turns_to_pass"]) for r in report["by_turn"]], [
            ("numpy", 3, 1.0, "0:3"), ("c_openmp", 3, 1.0, "0:3"), ("c_seq", 3, 1.0, "0:3"),
        ])
        self.assertEqual(report["conditions"], [])
        self.assertEqual(len(report["notes"]), 1)   # 수정 루프 행이 없으므로 구간 읽는 법 한 줄뿐

    def test_text_reports(self) -> None:
        for csv_path in (REAL_CSV, DUMMY_CSV):
            with self.subTest(csv=csv_path.name):
                rc, out, _ = _run_main("--csv", str(csv_path), "--all-runs", *self.OPTIONS)
                self.assertEqual(rc, EXIT_OK)
                for title in ("# aggregate:", "# coverage:", "# by-turn:", "# conditions:"):
                    self.assertEqual(sum(line.startswith(title) for line in out.splitlines()), 1, title)


if __name__ == "__main__":
    unittest.main()
