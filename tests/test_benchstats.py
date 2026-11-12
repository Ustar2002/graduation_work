"""benchstats(라운드 합치기)와 run.apply_rounds / run.next_round 테스트 (2026-11-12 측정 규약)."""

from __future__ import annotations

import json
import statistics as st
import tempfile
import unittest
from pathlib import Path

from harness.benchstats import bootstrap_ratio_ci, combine_rounds, describe, paired_rounds, ratio_of_means, seed_of


def _doc(kernel: list[float], baseline: list[float] | None, *, warm: int = 1, gflops_at_median: float | None = None,
         mhz: list[float] | None = None, throttle: int | None = 0, t0: float = 50.0, t1: float = 60.0, cls: str = "pass") -> dict:
    med = st.median(kernel) if kernel else None
    d = {"timed_ms": kernel, "warmup_ms": [kernel[0]] * warm if kernel else [], "median_ms": med,
         "gflops": gflops_at_median, "cpu_mhz_samples": mhz or [3000.0], "throttle_events": throttle,
         "temp_c_start": t0, "temp_c_end": t1, "nondeterministic_runs": 0, "nondeterministic_bitwise_runs": 0,
         "repeats_reduced": False, "classification": cls}
    if baseline is not None:
        d["baseline_interleaved"] = {"classification": "pass", "timed_ms": baseline, "warmup_ms": [baseline[0]] * warm,
                                     "median_ms": st.median(baseline)}
    return d


class DescribeTest(unittest.TestCase):
    def test_empty_and_single(self) -> None:
        self.assertEqual(describe([]), {"median_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None, "cv": None})
        d = describe([4.0])
        self.assertEqual((d["median_ms"], d["mean_ms"], d["cv"]), (4.0, 4.0, None))

    def test_cv_is_sample_std_over_mean(self) -> None:
        xs = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(describe(xs)["cv"], st.stdev(xs) / st.fmean(xs))


class RatioTest(unittest.TestCase):
    def test_ratio_of_means_pools_rounds(self) -> None:
        pairs = paired_rounds([[1.0, 1.0], [2.0]], [[2.0, 2.0], [2.0]])
        # 기준선 합 6 / 커널 합 4
        self.assertAlmostEqual(ratio_of_means(pairs), 1.5)

    def test_paired_rounds_truncates_to_shorter(self) -> None:
        pairs = paired_rounds([[1.0, 2.0, 3.0]], [[4.0, 5.0]])
        self.assertEqual(pairs, [[(1.0, 4.0), (2.0, 5.0)]])
        self.assertEqual(paired_rounds([[1.0]], [[]]), [])

    def test_bootstrap_is_deterministic_and_brackets_the_estimate(self) -> None:
        k = [[10.0, 11.0, 9.5, 10.2, 10.1], [10.4, 9.9, 10.0, 10.3, 9.8]]
        b = [[20.0, 21.5, 19.0, 20.4, 20.1], [20.9, 19.8, 20.2, 20.5, 19.7]]
        pairs = paired_rounds(k, b)
        lo1, hi1 = bootstrap_ratio_ci(pairs, seed=7)
        lo2, hi2 = bootstrap_ratio_ci(pairs, seed=7)
        self.assertEqual((lo1, hi1), (lo2, hi2), "같은 씨앗이면 같은 구간")
        est = ratio_of_means(pairs)
        assert lo1 is not None and hi1 is not None and est is not None
        self.assertLessEqual(lo1, est)
        self.assertGreaterEqual(hi1, est)
        self.assertLess(hi1 - lo1, 0.2, "잡음이 작으면 구간도 좁다")

    def test_bootstrap_constant_data_is_degenerate(self) -> None:
        pairs = paired_rounds([[5.0] * 4, [5.0] * 4], [[10.0] * 4, [10.0] * 4])
        self.assertEqual(bootstrap_ratio_ci(pairs, seed=1), (2.0, 2.0))

    def test_seed_of_is_stable(self) -> None:
        self.assertEqual(seed_of(("a", 1)), seed_of(("a", 1)))
        self.assertNotEqual(seed_of(("a", 1)), seed_of(("a", 2)))


class CombineRoundsTest(unittest.TestCase):
    def test_three_rounds(self) -> None:
        docs = [_doc([10.0, 12.0], [20.0, 24.0], gflops_at_median=2.0, mhz=[3000, 3100], throttle=1, t0=50, t1=55),
                _doc([11.0, 11.0], [22.0, 22.0], gflops_at_median=99.0, mhz=[2900], throttle=2, t0=56, t1=58),
                _doc([9.0, 13.0], [18.0, 26.0], mhz=[3050], throttle=None, t0=59, t1=61)]
        c = combine_rounds(docs, seed=3)
        k_all = [10.0, 12.0, 11.0, 11.0, 9.0, 13.0]
        self.assertEqual(c["time_samples_ms"]["timed"], k_all)
        self.assertEqual(c["time_samples_ms"]["rounds"], [2, 2, 2])
        self.assertAlmostEqual(c["kernel"]["median_ms"], st.median(k_all))
        self.assertAlmostEqual(c["kernel"]["mean_ms"], st.fmean(k_all))
        self.assertAlmostEqual(c["speedup_mean"], 2.0)
        self.assertAlmostEqual(c["speedup_median"], 2.0)
        self.assertEqual(c["speedup_rounds"], [2.0, 2.0, 2.0])
        lo, hi = c["speedup_ci"]
        self.assertEqual((lo, hi), (2.0, 2.0))
        self.assertEqual((c["rounds"], c["repeats"], c["warmup"]), (3, 6, 3))
        self.assertEqual(c["throttle_events"], 3, "None 라운드는 빼고 합한다")
        self.assertEqual((c["temp_c_start"], c["temp_c_end"]), (50, 61))
        self.assertEqual(c["cpu_mhz_min"], 2900.0)
        # gflops: 첫 라운드의 작업량(2.0 GFLOPS × 11 ms 중앙값)을 모은 중앙값(11 ms)으로 다시 나눈다
        self.assertAlmostEqual(c["gflops"], 2.0)

    def test_round_without_baseline_is_left_out_of_ratios(self) -> None:
        docs = [_doc([10.0, 10.0], [20.0, 20.0]), _doc([10.0, 10.0], None), _doc([10.0, 10.0], [30.0, 30.0])]
        c = combine_rounds(docs, seed=1)
        self.assertEqual(c["speedup_rounds"], [2.0, 3.0])
        self.assertAlmostEqual(c["speedup_mean"], 2.5)
        self.assertEqual(c["baseline_samples_ms"]["rounds"], [2, 0, 2])

    def test_no_baseline_at_all(self) -> None:
        c = combine_rounds([_doc([10.0, 11.0], None)], seed=1)
        self.assertIsNone(c["baseline"])
        self.assertIsNone(c["speedup_mean"])
        self.assertEqual(c["speedup_ci"], (None, None))

    def test_single_round_matches_previous_protocol(self) -> None:
        k, b = [10.0, 12.0, 11.0], [21.0, 23.0, 22.0]
        c = combine_rounds([_doc(k, b)], seed=1)
        self.assertAlmostEqual(c["speedup_median"], st.median(b) / st.median(k))
        self.assertEqual(c["time_samples_ms"]["timed"], k)


class ApplyRoundsTest(unittest.TestCase):
    """run.apply_rounds 와 run.next_round: 결과 파일에서 행을 채우고, 다음 라운드를 고른다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="rounds_test_")
        self.dir = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _stages(self, docs: list[dict], statuses: list[str] | None = None) -> dict:
        rounds = []
        for i, d in enumerate(docs, start=1):
            p = self.dir / ("bench.json" if i == 1 else f"bench_r{i}.json")
            p.write_text(json.dumps(d), encoding="utf-8")
            rounds.append({"round": i, "status": (statuses or ["ok"] * len(docs))[i - 1], "result_file": str(p), "finished_at": 0.0})
        return {"rounds": rounds, "rounds_planned": 3, "bench_artifact": str(self.dir / "kernel.so")}

    def _row(self, **kw):
        from harness.schema import RunRow
        base = dict(run_id="r", task="gemm", language="c_seq", args="positional", grammar_hint="off", feedback="compiler",
                    sample_idx=0, turn=0, passed=True, bench_ran=True, error_type="", error_stage="", env_hash="h",
                    baseline_time_1t_ms=40.0)
        base.update(kw)
        return RunRow(**base)

    def test_clean_rows_get_speedups(self) -> None:
        from harness.run import apply_rounds
        row = self._row()
        apply_rounds(row, self._stages([_doc([10.0, 10.0], [20.0, 20.0]), _doc([10.0, 10.0], [30.0, 30.0])]))
        self.assertEqual(row.bench_rounds, 2)
        self.assertAlmostEqual(row.speedup_mean, 2.5)
        self.assertAlmostEqual(row.speedup_vs_baseline, 25.0 / 10.0)
        self.assertEqual(row.speedup_rounds, [2.0, 3.0])
        self.assertAlmostEqual(row.speedup_vs_baseline_1t, 4.0)
        self.assertAlmostEqual(row.baseline_time_mean_ms, 25.0)
        self.assertIsNotNone(row.speedup_ci_lo)

    def test_error_in_a_later_round_removes_speedups(self) -> None:
        from harness.run import apply_rounds
        row = self._row(passed=False, error_type="runtime_crash", error_stage="bench")
        apply_rounds(row, self._stages([_doc([10.0], [20.0]), _doc([10.0], [20.0])], ["ok", "error"]))
        self.assertIsNone(row.speedup_mean)
        self.assertIsNone(row.speedup_vs_baseline)
        self.assertEqual(row.bench_rounds, 1)
        self.assertAlmostEqual(row.time_median_ms, 10.0, msg="부분 시간은 남긴다")

    def test_timeout_rows_keep_no_time_statistics(self) -> None:
        from harness.run import apply_rounds
        row = self._row(error_type="timeout", error_stage="bench")
        apply_rounds(row, self._stages([_doc([500.0], [20.0], cls="timeout")], ["error"]))
        self.assertIsNone(row.time_median_ms)
        self.assertIsNone(row.speedup_mean)

    def test_next_round_rules(self) -> None:
        from harness.run import RunContext, next_round
        from harness.config import load_config
        import dataclasses
        cfg = load_config()
        cfg = dataclasses.replace(cfg, measurement=dataclasses.replace(cfg.measurement, bench_rounds=3))
        ctx = RunContext(cfg=cfg, machine={}, static=None, fingerprint={}, env_hash="h", run_id="r", skip_bench=False,  # type: ignore[arg-type]
                         force=False, version="v")
        st1 = self._stages([_doc([10.0], [20.0])])
        self.assertEqual(next_round(ctx, self._row(), st1), 2)
        st3 = self._stages([_doc([10.0], [20.0])] * 3)
        self.assertIsNone(next_round(ctx, self._row(), st3), "계획한 라운드를 다 돌았다")
        self.assertIsNone(next_round(ctx, self._row(env_hash="other"), st1), "환경 지문이 다르면 섞지 않는다")
        self.assertIsNone(next_round(ctx, self._row(passed=False, error_type="wrong_output"), st1))
        st_old = dict(st1, rounds_planned=1)
        self.assertIsNone(next_round(ctx, self._row(), st_old), "이전 규약(1라운드)으로 잰 행은 이어 재지 않는다")
        st_err = self._stages([_doc([10.0], [20.0])], ["error"])
        self.assertIsNone(next_round(ctx, self._row(), st_err))
        ctx_skip = dataclasses.replace(ctx, skip_bench=True)
        self.assertIsNone(next_round(ctx_skip, self._row(), st1))


class SchemaMigrationTest(unittest.TestCase):
    """이전 스키마(새 열이 없는) CSV 에 행을 넣으면 새 열을 빈 값으로 더해 옮겨 쓴다(2026-11-12)."""

    def test_additive_header_is_migrated(self) -> None:
        import csv
        from harness.schema import COLUMNS, RunRow, upsert_row
        new_cols = {"bench_rounds", "baseline_time_mean_ms", "speedup_mean", "speedup_ci_lo", "speedup_ci_hi", "speedup_rounds"}
        old_cols = [c for c in COLUMNS if c not in new_cols]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "runs.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=old_cols, lineterminator="\n")
                w.writeheader()
                w.writerow({c: "" for c in old_cols} | {"run_id": "old", "task": "gemm", "language": "numpy", "sample_idx": "0",
                                                        "turn": "0", "args": "positional", "grammar_hint": "off", "feedback": "compiler"})
            row = RunRow(run_id="new", task="gemm", language="numpy", args="positional", grammar_hint="off", feedback="compiler",
                         sample_idx=0, turn=0, speedup_mean=1.5, bench_rounds=3)
            self.assertFalse(upsert_row(path, row))
            with path.open(newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                self.assertEqual(reader.fieldnames, COLUMNS)
                rows = list(reader)
            self.assertEqual([r["run_id"] for r in rows], ["old", "new"])
            self.assertEqual(rows[0]["speedup_mean"], "")
            self.assertEqual(rows[1]["speedup_mean"], "1.5")

    def test_unknown_columns_still_fail(self) -> None:
        import csv
        from harness.schema import COLUMNS, RunRow, SchemaError, upsert_row
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "runs.csv"
            with path.open("w", newline="", encoding="utf-8") as fh:
                csv.writer(fh).writerow([*COLUMNS[:5], "mystery_column"])
            with self.assertRaises(SchemaError):
                upsert_row(path, RunRow(run_id="x", task="t", language="numpy", sample_idx=0, turn=0))


if __name__ == "__main__":
    unittest.main()
