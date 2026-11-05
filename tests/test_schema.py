"""schema 테스트: 열 일치, 직렬화 규칙, upsert 교체, rebuild, read_rows 중복 제거, harness_version 형식."""

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
