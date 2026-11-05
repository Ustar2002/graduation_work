"""aggregate 테스트: pass@k 추정량, 합성 CSV(harness.schema.COLUMNS) 집계 수치, 키 중복 제거, run 선택, CLI(JSON·빈 CSV·없는 CSV)."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from analysis.aggregate import (
    EXIT_NOT_FOUND,
    EXIT_OK,
    GROUP_KEY,
    METRIC_COLUMNS,
    aggregate_groups,
    has_protocol_flag,
    latest_run_id,
    load_runs,
    main,
    pass_at_k,
    select_runs,
    summarize_runs,
)
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


if __name__ == "__main__":
    unittest.main()
