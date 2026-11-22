"""schema 테스트: 열 일치, 직렬화 규칙, upsert 교체, rebuild(수정 턴의 중첩 result.json 포함), read_rows 중복 제거,
harness_version 형식과 그 해시의 대상 파일."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest import mock

import harness.schema as schema
from harness.schema import (
    COLUMN_TO_FIELD,
    COLUMNS,
    FIELD_TO_COLUMN,
    ROW_KEY,
    RunRow,
    SchemaError,
    harness_version,
    migrate_header_if_additive,
    read_result_rows,
    read_rows,
    rebuild_csv,
    rebuild_csvs,
    source_hash,
    upsert_row,
)


def _row(**over: object) -> RunRow:
    base = dict(
        run_id="20261104-100000-dummy", created_at="2026-11-04T10:00:00+09:00", task="gemm", language="numpy",
        backend="python", args="positional", grammar_hint="off", feedback="compiler", sample_idx=0, turn=0,
        provider="dummy", model="dummy", passed=True, valid_sample=True, time_median_ms=14.7, tokens_in=1234,
        protocol_flags=["governor=powersave"], sampling={"temperature_effective": "provider_default"},
        error_type="", error_stage="",
    )
    base.update(over)
    return RunRow(**base)  # type: ignore[arg-type]


class ColumnsTest(unittest.TestCase):
    def test_to_csv_dict_keys_match_columns(self) -> None:
        self.assertEqual(list(_row().to_csv_dict().keys()), COLUMNS)
        self.assertEqual(list(RunRow().to_csv_dict().keys()), COLUMNS)

    def test_columns_unique_and_field_mapping(self) -> None:
        self.assertEqual(len(COLUMNS), len(set(COLUMNS)))
        self.assertEqual(FIELD_TO_COLUMN, {"passed": "pass"})
        self.assertEqual(COLUMN_TO_FIELD, {"pass": "passed"})
        self.assertIn("pass", COLUMNS)
        self.assertNotIn("passed", COLUMNS)
        field_names = [f.name for f in fields(RunRow)]
        self.assertEqual([FIELD_TO_COLUMN.get(n, n) for n in field_names], COLUMNS)
        for k in ROW_KEY:
            self.assertIn(k, field_names)

    def test_column_order_fixed_head_and_tail(self) -> None:
        self.assertEqual(COLUMNS[:3], ["run_id", "created_at", "task"])
        self.assertEqual(COLUMNS[-3:], ["raw_path", "built_path", "harness_version"])
        self.assertEqual(COLUMNS.index("pass") + 1, COLUMNS.index("turns_to_pass"))

    def test_column_count_and_round_columns(self) -> None:
        """열은 121개다(2026-11-12 에 bench 라운드 열 6개가 speedup_vs_baseline_1t 바로 뒤에 더해졌다)."""
        self.assertEqual(len(COLUMNS), 121)
        i = COLUMNS.index("speedup_vs_baseline_1t")
        self.assertEqual(COLUMNS[i + 1:i + 7], list(ROUND_COLUMNS))


# 2026-11-12 에 더해진 열(그 전 스키마의 CSV 는 이 열이 없는 115열이다)
ROUND_COLUMNS = ("bench_rounds", "baseline_time_mean_ms", "speedup_mean", "speedup_ci_lo", "speedup_ci_hi", "speedup_rounds")
OLD_COLUMNS = [c for c in COLUMNS if c not in ROUND_COLUMNS]


class SerializationTest(unittest.TestCase):
    def test_rules(self) -> None:
        d = _row(passed=False, extract_ok=None, undefined_symbols=[], check_tests=[{"name": "tiny", "classification": "pass"}]).to_csv_dict()
        self.assertEqual(d["pass"], "0")
        self.assertEqual(d["valid_sample"], "1")
        self.assertEqual(d["extract_ok"], "")  # NA
        self.assertEqual(d["time_mean_ms"], "")  # NA
        self.assertEqual(d["undefined_symbols"], "[]")
        self.assertEqual(d["protocol_flags"], '["governor=powersave"]')
        self.assertEqual(json.loads(d["check_tests"]), [{"name": "tiny", "classification": "pass"}])
        self.assertEqual(d["time_median_ms"], "14.7")
        self.assertEqual(d["tokens_in"], "1234")
        self.assertEqual(d["error_type"], "")

    def test_roundtrip_from_csv_dict(self) -> None:
        r = _row(signature_ok=False, build_time_ms=12.5, repeats=10, omp_env={"OMP_PLACES": "cores"}, seed=None)
        back = RunRow.from_csv_dict(r.to_csv_dict())
        # CSV 공간에서는 NA 와 "" 가 같은 칸이므로 동등성은 CSV 딕트로 비교한다
        self.assertEqual(back.to_csv_dict(), r.to_csv_dict())
        self.assertIs(back.signature_ok, False)
        self.assertIs(back.passed, True)
        self.assertIsNone(back.seed)
        self.assertIsNone(back.time_mean_ms)  # 숫자 열의 "" 는 NA
        self.assertEqual(back.error_type, "")  # 문자열 열의 "" 는 "" (정식 값)
        self.assertEqual(back.error_detail, "")  # 문자열 열의 NA 도 "" 로 읽힌다
        self.assertEqual(back.omp_env, {"OMP_PLACES": "cores"})
        self.assertEqual(back.repeats, 10)
        self.assertEqual(back.build_time_ms, 12.5)

    def test_from_dict_accepts_typed_and_field_names(self) -> None:
        r = _row()
        typed = r.to_dict()
        self.assertIn("passed", typed)
        self.assertEqual(RunRow.from_dict(typed), r)
        mixed = dict(typed)
        mixed["pass"] = mixed.pop("passed")  # 열 이름으로도 받는다
        self.assertEqual(RunRow.from_dict(mixed), r)

    def test_key(self) -> None:
        self.assertEqual(_row().key(), ("20261104-100000-dummy", "gemm", "numpy", "positional", "off", "compiler", "0", "0"))


class CsvTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="schema_test_")
        self.tmp = Path(self._tmp.name)
        self.csv = self.tmp / "runs.csv"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _data_rows(self) -> list[dict[str, str]]:
        with self.csv.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            self.assertEqual(reader.fieldnames, COLUMNS)
            return list(reader)

    def test_upsert_replaces_same_key_instead_of_duplicating(self) -> None:
        self.assertFalse(upsert_row(self.csv, _row(time_median_ms=14.7)))
        self.assertTrue(upsert_row(self.csv, _row(time_median_ms=15.1, created_at="2026-11-04T10:05:00+09:00")))
        rows = self._data_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["time_median_ms"], "15.1")
        # 다른 키는 추가
        self.assertFalse(upsert_row(self.csv, _row(sample_idx=1)))
        self.assertFalse(upsert_row(self.csv, _row(language="c_seq", backend="c")))
        rows = self._data_rows()
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["sample_idx"] for r in rows], ["0", "1", "0"])
        self.assertFalse(list(self.tmp.glob("*.tmp")), "임시 파일이 남으면 안 된다")

    def test_upsert_header_mismatch_raises(self) -> None:
        self.csv.write_text("run_id,foo\n1,2\n", encoding="utf-8")
        with self.assertRaises(SchemaError) as ctx:
            upsert_row(self.csv, _row())
        self.assertIn("rebuild-csv", str(ctx.exception))
        with self.assertRaises(SchemaError):
            read_rows(self.csv)

    def test_read_rows_dedupes_latest_created_at(self) -> None:
        older = _row(created_at="2026-11-04T10:00:00+09:00", time_median_ms=1.0).to_csv_dict()
        newer = _row(created_at="2026-11-04T11:00:00+09:00", time_median_ms=2.0).to_csv_dict()
        other = _row(sample_idx=2, time_median_ms=3.0).to_csv_dict()
        with self.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLUMNS)
            w.writeheader()
            w.writerow(newer)
            w.writerow(other)
            w.writerow(older)  # 파일 순서와 무관하게 created_at 최신이 남아야 한다
        rows = read_rows(self.csv)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["time_median_ms"], "2.0")
        self.assertEqual(rows[1]["sample_idx"], "2")

    def test_rebuild_csv_from_result_json(self) -> None:
        built = self.tmp / "built"
        r0 = _row(sample_idx=0)
        r1 = _row(sample_idx=1, provider="anthropic")
        r0_old = _row(sample_idx=0, created_at="2026-11-01T00:00:00+09:00", time_median_ms=99.0)
        for sub, row in (("run/gemm/numpy/c/sample_0", r0), ("run/gemm/numpy/c/sample_1", r1), ("old/gemm/numpy/c/sample_0", r0_old)):
            d = built / sub
            d.mkdir(parents=True)
            payload = {"row": row.to_dict() if sub.startswith("run") else row.to_csv_dict(), "raw_sha256": "deadbeef"}
            (d / "result.json").write_text(json.dumps(payload), encoding="utf-8")
        (built / "run/gemm/baseline").mkdir(parents=True)
        (built / "run/gemm/baseline/bench_run.json").write_text("{}", encoding="utf-8")  # row 없는 JSON은 무시 대상이 아님(이름이 다름)
        n = rebuild_csv(built, self.csv)
        self.assertEqual(n, 2)  # r0_old 는 같은 키의 오래된 행 → 제거
        rows = read_rows(self.csv)
        self.assertEqual({r["sample_idx"] for r in rows}, {"0", "1"})
        self.assertEqual([r["time_median_ms"] for r in rows if r["sample_idx"] == "0"], ["14.7"])
        # provider 분리
        counts = rebuild_csvs(built, self.tmp / "real.csv", self.tmp / "dummy.csv")
        self.assertEqual(counts[str(self.tmp / "real.csv")], 1)
        self.assertEqual(counts[str(self.tmp / "dummy.csv")], 1)
        self.assertEqual(read_rows(self.tmp / "real.csv")[0]["provider"], "anthropic")

    def test_rebuild_keeps_rows_without_result_json(self) -> None:
        """다시 만들 수 없는 행은 버리지 않는다: built 디렉터리가 비었거나 없으면 두 CSV 가 바이트 단위로 그대로이고,
        실제 provider 의 result.json 이 하나도 없어도 runs.csv 의 실제 행은 남는다(dummy 행은 runs.csv 로 옮겨가지 않는다)."""
        real_csv, dummy_csv = self.tmp / "real.csv", self.tmp / "dummy.csv"
        upsert_row(real_csv, _row(run_id="20261105-real", provider="anthropic"))
        upsert_row(dummy_csv, _row(run_id="d1", sample_idx=3))
        before = (real_csv.read_bytes(), dummy_csv.read_bytes())
        for built in (self.tmp / "missing", self.tmp / "empty"):
            if built.name == "empty":
                built.mkdir()
            rebuild_csvs(built, real_csv, dummy_csv)
            self.assertEqual((real_csv.read_bytes(), dummy_csv.read_bytes()), before, built.name)
        # built 에는 dummy result.json 만 있다: runs.csv 의 anthropic 행은 남고, dummy CSV 에는 기존 행과 새 행이 같이 있다
        d = self.tmp / "built" / "d2" / "gemm" / "numpy" / "c" / "sample_0"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"row": _row(run_id="d2").to_dict()}), encoding="utf-8")
        counts = rebuild_csvs(self.tmp / "built", real_csv, dummy_csv)
        self.assertEqual(counts, {str(real_csv): 1, str(dummy_csv): 2})
        self.assertEqual(real_csv.read_bytes(), before[0])
        self.assertEqual([(r["run_id"], r["provider"]) for r in read_rows(real_csv)], [("20261105-real", "anthropic")])
        self.assertEqual(sorted(r["run_id"] for r in read_rows(dummy_csv)), ["d1", "d2"])

    def test_rebuild_replaces_existing_row_with_result_json(self) -> None:
        """같은 키의 행이 CSV 와 result.json 에 다 있으면 result.json 이 원본이다(기존 CSV 의 값은 버린다)."""
        upsert_row(self.csv, _row(time_median_ms=99.0, created_at="2026-11-30T00:00:00+09:00"))
        d = self.tmp / "built" / "run" / "gemm" / "numpy" / "c" / "sample_0"
        d.mkdir(parents=True)
        (d / "result.json").write_text(json.dumps({"row": _row(time_median_ms=14.7).to_dict()}), encoding="utf-8")
        self.assertEqual(rebuild_csv(self.tmp / "built", self.csv), 1)
        self.assertEqual([r["time_median_ms"] for r in read_rows(self.csv)], ["14.7"])

    def test_turn_rows_are_separate_rows(self) -> None:
        """``turn`` 은 행 키의 일부다: 수정 턴의 행은 턴 0 행을 덮어쓰지 않고, 같은 턴을 다시 쓰면 그 행만 바뀐다."""
        self.assertFalse(upsert_row(self.csv, _row(turn=0, passed=False)))
        self.assertFalse(upsert_row(self.csv, _row(turn=1, passed=False)))
        self.assertTrue(upsert_row(self.csv, _row(turn=1, passed=True, turns_to_pass=1)))
        self.assertEqual([(r["turn"], r["pass"], r["turns_to_pass"]) for r in self._data_rows()], [("0", "0", ""), ("1", "1", "1")])

    def test_turn_results_are_read_from_nested_directories(self) -> None:
        """수정 턴의 ``sample_<i>/turn_<t>/result.json`` (설계 v3 §9.2)도 턴 0 의 것과 같이 읽혀 CSV 로 되살아난다."""
        sample = self.tmp / "built" / "run" / "gemm" / "c_seq" / "args-positional_hint-off_fb-compiler" / "sample_1"
        rows = {
            sample: _row(language="c_seq", backend="c", sample_idx=1, turn=0, passed=False, error_type="wrong_output"),
            sample / "turn_1": _row(language="c_seq", backend="c", sample_idx=1, turn=1, passed=False, error_type="wrong_output"),
            sample / "turn_2": _row(language="c_seq", backend="c", sample_idx=1, turn=2, passed=True, turns_to_pass=2),
        }
        for directory, row in rows.items():
            directory.mkdir(parents=True)
            (directory / "result.json").write_text(json.dumps({"row": row.to_dict(), "raw_sha256": "x", "stages": {}}), encoding="utf-8")
        # 턴 디렉터리의 다른 문서(워커 결과, sanitizer 빌드 기록)는 행이 아니다
        (sample / "turn_1" / "check.json").write_text(json.dumps({"row": "not a result"}), encoding="utf-8")
        (sample / "turn_2" / "san").mkdir()
        (sample / "turn_2" / "san" / "build.json").write_text("{}", encoding="utf-8")

        read = read_result_rows(self.tmp / "built")
        self.assertEqual(sorted((r.sample_idx, r.turn, r.passed) for r in read), [(1, 0, False), (1, 1, False), (1, 2, True)])
        self.assertEqual(rebuild_csv(self.tmp / "built", self.csv), 3)
        self.assertEqual(sorted((r["turn"], r["pass"], r["turns_to_pass"]) for r in read_rows(self.csv)),
                         [("0", "0", ""), ("1", "0", ""), ("2", "1", "2")])


class HeaderMigrationTest(unittest.TestCase):
    """이전 스키마(열이 추가되기만 한 115열) CSV 의 옮겨 쓰기(2026-11-12)와, rebuild 가 result.json 이 없는 행을 버리지 않는 규칙이
    함께 성립하는지 본다: 옛 헤더의 행은 새 열을 빈 값으로 더해 남고, result.json 이 있는 행만 그것으로 바뀐다."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="schema_migrate_test_")
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.csv = self.tmp / "runs.csv"

    def _write(self, header: list[str], rows: list[dict[str, str]], path: Path | None = None) -> Path:
        path = path or self.csv
        with path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n", extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return path

    def _read(self, path: Path | None = None) -> tuple[list[str], list[dict[str, str]]]:
        with (path or self.csv).open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            return list(reader.fieldnames or []), list(reader)

    def _old_row(self, **over: object) -> dict[str, str]:
        d = _row(**over).to_csv_dict()
        return {c: d[c] for c in OLD_COLUMNS}

    def test_migrate_rewrites_subset_header_and_keeps_values(self) -> None:
        old = [self._old_row(run_id="a", time_median_ms=1.5), self._old_row(run_id="b", sample_idx=1)]
        self._write(OLD_COLUMNS, old)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertTrue(migrate_header_if_additive(OLD_COLUMNS, self.csv))
        self.assertIn("speedup_mean", err.getvalue())
        header, rows = self._read()
        self.assertEqual(header, COLUMNS)
        self.assertEqual([r["run_id"] for r in rows], ["a", "b"])
        for before, after in zip(old, rows):
            self.assertEqual({c: after[c] for c in OLD_COLUMNS}, before)
            self.assertEqual({after[c] for c in ROUND_COLUMNS}, {""})
        self.assertEqual(read_rows(self.csv)[0]["time_median_ms"], "1.5")

    def test_migrate_leaves_everything_else_alone(self) -> None:
        """헤더 없음·현재 헤더·순서만 다른 헤더·모르는 열이 있는 헤더는 옮겨 쓰지 않는다(뒤의 둘은 check_header 가 거부)."""
        self.assertFalse(migrate_header_if_additive(None, self.csv))
        for header in (COLUMNS, list(reversed(COLUMNS)), [*OLD_COLUMNS, "mystery_column"]):
            path = self._write(header, [self._old_row()])
            before = path.read_bytes()
            self.assertFalse(migrate_header_if_additive(header, path))
            self.assertEqual(path.read_bytes(), before)
            if header != COLUMNS:
                with self.assertRaises(SchemaError):
                    read_rows(path)

    def test_upsert_into_old_schema_keeps_old_rows(self) -> None:
        self._write(OLD_COLUMNS, [self._old_row(run_id="old", provider="anthropic")])
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertFalse(upsert_row(self.csv, _row(run_id="new", bench_rounds=3, speedup_mean=1.5, speedup_rounds=[1.4, 1.6])))
        header, rows = self._read()
        self.assertEqual(header, COLUMNS)
        self.assertEqual([(r["run_id"], r["speedup_mean"], r["bench_rounds"]) for r in rows], [("old", "", ""), ("new", "1.5", "3")])

    def test_rebuild_without_result_json_migrates_and_keeps_old_rows(self) -> None:
        """result.json 이 하나도 없는 머신(built 없음/빈 built)에서도 옛 헤더의 실제 행은 모두 남고 헤더만 새 스키마가 된다."""
        real_csv, dummy_csv = self.tmp / "real.csv", self.tmp / "dummy.csv"
        real_rows = [self._old_row(run_id="r", sample_idx=i, provider="anthropic", time_median_ms=10.0 + i) for i in range(3)]
        self._write(OLD_COLUMNS, real_rows, real_csv)
        self._write(OLD_COLUMNS, [self._old_row(run_id="d")], dummy_csv)
        logs: list[str] = []
        for built in (self.tmp / "missing", self.tmp / "empty"):
            if built.name == "empty":
                built.mkdir()
            with contextlib.redirect_stderr(io.StringIO()) as err:
                counts = rebuild_csvs(built, real_csv, dummy_csv)
            logs.append(err.getvalue())
            self.assertEqual(counts, {str(real_csv): 3, str(dummy_csv): 1}, built.name)
            header, rows = self._read(real_csv)
            self.assertEqual(header, COLUMNS)
            self.assertEqual([(r["sample_idx"], r["time_median_ms"], r["speedup_mean"]) for r in rows],
                             [("0", "10.0", ""), ("1", "11.0", ""), ("2", "12.0", "")])
            self.assertEqual([r["run_id"] for r in read_rows(dummy_csv)], ["d"])
        # 첫 rebuild 가 옛 헤더를 알리고 옮겨 썼으므로 두 번째 rebuild 때는 이미 새 헤더다
        self.assertIn("헤더가 스키마와 다르다", logs[0])
        self.assertNotIn("헤더가 스키마와 다르다", logs[1])

    def test_rebuild_old_schema_with_some_result_json(self) -> None:
        """result.json 이 있는 키는 그것으로(새 열 포함) 바뀌고, 없는 키는 옛 행이 새 열 빈 값으로 남는다."""
        self._write(OLD_COLUMNS, [self._old_row(run_id="r", sample_idx=0, provider="anthropic", time_median_ms=99.0),
                                  self._old_row(run_id="r", sample_idx=1, provider="anthropic", time_median_ms=11.0)])
        d = self.tmp / "built" / "r" / "gemm" / "numpy" / "c" / "sample_0"
        d.mkdir(parents=True)
        fresh = _row(run_id="r", sample_idx=0, provider="anthropic", time_median_ms=10.0, bench_rounds=3, speedup_mean=2.0)
        (d / "result.json").write_text(json.dumps({"row": fresh.to_dict()}), encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(rebuild_csv(self.tmp / "built", self.csv, exclude_providers={"dummy"}), 2)
        self.assertIn("헤더가 스키마와 다르다", err.getvalue())
        header, rows = self._read()
        self.assertEqual(header, COLUMNS)
        by_idx = {r["sample_idx"]: r for r in rows}
        self.assertEqual((by_idx["0"]["time_median_ms"], by_idx["0"]["bench_rounds"], by_idx["0"]["speedup_mean"]), ("10.0", "3", "2.0"))
        self.assertEqual((by_idx["1"]["time_median_ms"], by_idx["1"]["bench_rounds"], by_idx["1"]["speedup_mean"]), ("11.0", "", ""))

    def test_rebuild_unknown_columns_keeps_rows_and_drops_only_those_columns(self) -> None:
        """모르는 열이 있는 CSV 도 rebuild 는 행을 버리지 않는다(그 열의 값만 경고와 함께 버린다). upsert 는 계속 거부한다."""
        self._write([*COLUMNS, "mystery_column"], [{**_row(provider="anthropic").to_csv_dict(), "mystery_column": "x"}])
        with self.assertRaises(SchemaError):
            upsert_row(self.csv, _row(sample_idx=9))
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(rebuild_csv(self.tmp / "missing", self.csv), 1)
        self.assertIn("mystery_column", err.getvalue())
        header, rows = self._read()
        self.assertEqual(header, COLUMNS)
        self.assertEqual(rows[0]["time_median_ms"], "14.7")


class SourceHashTest(unittest.TestCase):
    """``harness_version`` 의 dirty 해시가 덮는 파일(설계 §11, v3 §12.2): 결과 행의 값을 바꿀 수 있는 것들.
    임시 ROOT 에 저장소의 축소판을 만들어 대상 목록과, 대상 하나가 바뀔 때마다 해시가 바뀌는지를 본다."""

    HASHED = (
        "harness/run.py", "harness/build/rust.py", "harness/kernels/halide_pipe.py", "analysis/aggregate.py",
        "prompts/system.md", "prompts/lang/numba.md", "prompts/hints/rust.md", "config/experiment.yaml",
        "plugins/lint/run_lint.py", "plugins/lint/rules.yaml",
        "harness/build/rust_template/Cargo.toml", "harness/build/rust_template/Cargo.lock",
        "tasks/gemm/spec.md", "tasks/gemm/signatures.yaml", "tasks/gemm/tolerance.yaml",
        "tasks/bfs/spec.md", "tasks/bfs/signatures.yaml", "tasks/bfs/tolerance.yaml",
    )
    NOT_HASHED = (
        "harness/__pycache__/run.py", "config/machine.yaml", "plugins/lint/README.md", "plugins/other/run_lint.py",
        "tasks/gemm/gen_tests.py", "tasks/gemm/tests/manifest.json", "tasks/gemm/baseline/tolerance.yaml",
        "docs/decisions.md", "results/runs.csv",
    )

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="schema_hash_test_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for rel in (*self.HASHED, *self.NOT_HASHED):
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"{rel}\n", encoding="utf-8")
        patcher = mock.patch.object(schema, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def listed(self) -> list[str]:
        return [p.relative_to(self.root).as_posix() for p in schema._source_files()]

    def test_inputs(self) -> None:
        self.assertEqual(sorted(self.listed()), sorted(self.HASHED))
        self.assertEqual(self.listed(), self.listed(), "순서가 결정적이어야 같은 소스가 같은 해시를 낸다")

    def test_every_input_changes_the_hash_and_nothing_else_does(self) -> None:
        base = source_hash()
        self.assertRegex(base, r"^[0-9a-f]{12}$")
        for rel in self.HASHED:
            path = self.root / rel
            path.write_text("changed\n", encoding="utf-8")
            self.assertNotEqual(source_hash(), base, rel)
            path.write_text(f"{rel}\n", encoding="utf-8")
            self.assertEqual(source_hash(), base, rel)
        for rel in self.NOT_HASHED:
            (self.root / rel).write_text("changed\n", encoding="utf-8")
            self.assertEqual(source_hash(), base, rel)

    def test_new_task_and_removed_rule_table_change_the_hash(self) -> None:
        base = source_hash()
        new_task = self.root / "tasks" / "fft"
        new_task.mkdir()
        (new_task / "signatures.yaml").write_text("task: fft\n", encoding="utf-8")  # spec.md 가 아직 없는 과제도 있는 파일은 들어간다
        self.assertIn("tasks/fft/signatures.yaml", self.listed())
        with_task = source_hash()
        self.assertNotEqual(with_task, base)
        (self.root / "plugins" / "lint" / "rules.yaml").unlink()
        self.assertNotIn("plugins/lint/rules.yaml", self.listed())
        self.assertNotEqual(source_hash(), with_task)

    def test_renaming_an_input_changes_the_hash(self) -> None:
        """경로도 해시에 들어간다: 내용과 정렬 순서가 그대로여도 이름이 바뀌면(다른 언어의 힌트가 됐다) 다른 해시다."""
        base = source_hash()
        before = self.listed()
        (self.root / "prompts" / "hints" / "rust.md").rename(self.root / "prompts" / "hints" / "go.md")
        after = self.listed()
        self.assertEqual(before.index("prompts/hints/rust.md"), after.index("prompts/hints/go.md"))
        self.assertNotEqual(source_hash(), base)


class RepoSourceFilesTest(unittest.TestCase):
    """실제 저장소의 해시 대상: 린트 플러그인과 그 규칙 표, 과제 12개 전부의 시그니처·허용오차(와 있는 명세)가 들어 있다."""

    TASKS = ("gemm", "jacobi2d", "spmv_csr", "dot", "histogram", "scan", "nbody", "sort", "cholesky", "floyd_warshall",
             "bfs", "fft")

    def test_lint_plugin_and_every_task_are_covered(self) -> None:
        root = schema.ROOT
        files = {p.relative_to(root).as_posix() for p in schema._source_files()}
        self.assertLessEqual({"plugins/lint/run_lint.py", "plugins/lint/rules.yaml", "config/experiment.yaml", "harness/run.py",
                              "harness/safety.py", "harness/perf_summary.py", "analysis/aggregate.py", "prompts/system.md",
                              "harness/build/rust_template/Cargo.toml", "harness/build/rust_template/Cargo.lock"}, files)
        for task in self.TASKS:
            for name in ("signatures.yaml", "tolerance.yaml"):
                self.assertIn(f"tasks/{task}/{name}", files)
        for name in ("spec.md", "signatures.yaml", "tolerance.yaml"):
            for path in (root / "tasks").glob(f"*/{name}"):
                self.assertIn(path.relative_to(root).as_posix(), files)
        self.assertFalse([f for f in files if "__pycache__" in f or f.startswith(("results/", "docs/"))])


class VersionTest(unittest.TestCase):
    """harness_version 형식(설계 §11): git 없음 → ``nogit-<hash12>``; git → ``git describe --always --dirty`` 출력;
    dirty 면 ``-dirty+<hash12>`` 병기. ``harness.schema.ROOT`` 를 임시 디렉터리로 바꿔 세 경우를 직접 만든다
    (실제 ROOT 값만 보면 현재 환경의 한 경우밖에 검증되지 않는다)."""

    _GIT_ENV = {"GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}  # 사용자 git 설정(gpgsign·템플릿 훅) 차단
    _GIT_IDENT = ["-c", "user.name=test", "-c", "user.email=test@example.com"]

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="schema_version_test_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        (self.root / "harness").mkdir()
        self.src = self.root / "harness" / "mod.py"  # §11 해시 대상(harness/**/*.py)
        self.src.write_text("X = 1\n", encoding="utf-8")
        patcher = mock.patch.object(schema, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        # lru_cache 라 ROOT 를 바꿀 때마다 비워야 한다. 테스트 뒤에도 비워 다른 테스트가 실제 ROOT 값을 다시 계산하게 한다.
        self.addCleanup(harness_version.cache_clear)

    def _version(self) -> str:
        harness_version.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()):
            return harness_version()

    def _git(self, *args: str) -> str:
        if shutil.which("git") is None:
            self.skipTest("git 이 없다")
        r = subprocess.run(
            ["git", *self._GIT_IDENT, *args], cwd=self.root, env={**os.environ, **self._GIT_ENV},
            capture_output=True, text=True, check=True,
        )
        return r.stdout.strip()

    def _init_and_commit(self) -> str:
        """임시 ROOT 를 git 저장소로 만들고 한 번 커밋한다. HEAD 의 전체 해시를 돌려준다."""
        self._git("init", "-q")
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "init")
        return self._git("rev-parse", "HEAD")

    def test_nogit_root_uses_source_hash(self) -> None:
        v = self._version()
        self.assertRegex(v, r"^nogit-[0-9a-f]{12}$")
        self.assertEqual(v, f"nogit-{source_hash()}")
        self.src.write_text("X = 2\n", encoding="utf-8")
        v2 = self._version()
        self.assertRegex(v2, r"^nogit-[0-9a-f]{12}$")
        self.assertNotEqual(v2, v, "해시 대상 소스가 바뀌면 nogit 해시도 바뀌어야 한다")

    def test_git_without_commit_falls_back_to_nogit_with_warning(self) -> None:
        self._git("init", "-q")  # .git 은 있지만 HEAD 가 없어 describe 가 실패한다(rc=128)
        harness_version.cache_clear()
        with contextlib.redirect_stderr(io.StringIO()) as err:
            v = harness_version()
        self.assertEqual(v, f"nogit-{source_hash()}")
        self.assertIn("git describe 실패", err.getvalue())

    def test_git_describe_hash_clean_then_dirty(self) -> None:
        head = self._init_and_commit()
        clean = self._version()
        self.assertRegex(clean, r"^[0-9a-f]{7,40}$", "태그가 없으면 --always 의 축약 커밋 해시")
        self.assertTrue(head.startswith(clean), (head, clean))
        # 추적 파일 수정 → -dirty+<hash12> 가 반드시 붙고, hash12 는 바뀐 소스의 해시다
        self.src.write_text("X = 2\n", encoding="utf-8")
        dirty = self._version()
        self.assertRegex(dirty, r"^[0-9a-f]{7,40}-dirty\+[0-9a-f]{12}$")
        self.assertEqual(dirty, f"{clean}-dirty+{source_hash()}")
        # 되돌리면 다시 clean
        self.src.write_text("X = 1\n", encoding="utf-8")
        self.assertEqual(self._version(), clean)

    def test_git_describe_tag_passthrough(self) -> None:
        self._init_and_commit()
        self._git("tag", "-a", "v0.1", "-m", "v0.1")  # describe 는 주석 태그만 본다
        self.assertEqual(self._version(), "v0.1")
        self.src.write_text("X = 2\n", encoding="utf-8")
        self.assertEqual(self._version(), f"v0.1-dirty+{source_hash()}")

    def test_cached_within_process(self) -> None:
        v = self._version()
        self.src.write_text("X = 2\n", encoding="utf-8")
        self.assertEqual(harness_version(), v, "캐시: cache_clear 전에는 한 프로세스 안에서 일정")


class RepoVersionTest(unittest.TestCase):
    """실제 ROOT 의 harness_version: 어느 경우든 §11 형식이어야 하고, dirty 면 +<hash12> 가 빠지면 안 된다."""

    def test_repo_root_version_format(self) -> None:
        harness_version.cache_clear()
        self.addCleanup(harness_version.cache_clear)
        with contextlib.redirect_stderr(io.StringIO()):
            v = harness_version()
        self.assertRegex(v, r"^\S+$")
        self.assertNotRegex(v, r"-dirty$", "dirty 인데 +<hash12> 병기가 없다(§11)")
        if v.startswith("nogit-"):
            self.assertEqual(v, f"nogit-{source_hash()}")
        elif "-dirty" in v:
            self.assertTrue(v.endswith(f"-dirty+{source_hash()}"), v)
        self.assertEqual(harness_version(), v)  # 캐시: 한 프로세스 안에서 일정


if __name__ == "__main__":
    unittest.main()
