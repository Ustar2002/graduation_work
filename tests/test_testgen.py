"""testgen 테스트 (설계 §16): npz 키·dtype·manifest·test_hash 정의(LF 매니페스트), 재생성 시 바이트 동일, 참조는 언제나
float64(정수 출력 과제도 정수 값을 담은 float64 참조 — docs/decisions.md 2026-11-08), CLI(main_for)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.testgen import DEFAULT_SEED, TEST_NAMES, generate, main_for

FLOAT_SHAPES = [("tiny", {"n": 3}), ("rect", {"n": 17}), ("small", {"n": 64}), ("large", {"n": 257})]
INT_SHAPES = [("tiny", {"n": 10, "n_bins": 4}), ("rect", {"n": 101, "n_bins": 7}),
              ("small", {"n": 256, "n_bins": 16}), ("large", {"n": 1000, "n_bins": 32})]


def float_case(shape, rng):
    """입력 두 개(x float32, idx int32), 출력 y(float32) → 참조 float64."""
    n = shape["n"]
    x = rng.random(n, dtype=np.float32)
    idx = rng.integers(0, n, size=n, dtype=np.int32)
    ref = x.astype(np.float64)[idx] * 2.0
    return {"n": n}, {"x": x, "idx": idx}, "y", ref


def int_case(shape, rng):
    """정수 출력(hist int32) 과제: 저장소 과제(histogram·scan·sort·bfs)처럼 참조는 정수 값을 담은 float64 다."""
    n, n_bins = shape["n"], shape["n_bins"]
    keys = rng.integers(0, n_bins, size=n, dtype=np.int32)
    return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", np.bincount(keys, minlength=n_bins).astype(np.float64)


def int32_ref_case(shape, rng):
    """잘못 만든 생성기: 참조를 int32 로 돌려준다(testgen 이 거절한다)."""
    dims, inputs, out_name, ref = int_case(shape, rng)
    return dims, inputs, out_name, ref.astype(np.int32)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestgenBase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="testgen_test_")
        self.tmp = Path(self._tmp.name)
        self.stderr = io.StringIO()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def gen(self, name: str, shapes, make_case, **kw):
        with contextlib.redirect_stderr(self.stderr):
            return generate("demo", kw.pop("seed", 7), self.tmp / name, shapes, make_case, "unit test", **kw)


class DefaultFloatBehaviourTest(TestgenBase):
    def test_files_keys_and_dtypes(self) -> None:
        manifest = self.gen("a", FLOAT_SHAPES, float_case)
        out = self.tmp / "a"
        self.assertEqual(sorted(p.name for p in out.iterdir()), ["large.npz", "manifest.json", "rect.npz", "small.npz", "tiny.npz"])
        for index, (name, shape) in enumerate(FLOAT_SHAPES):
            with np.load(out / f"{name}.npz") as npz:
                self.assertEqual(sorted(npz.files), ["idx", "n", "seed", "x", "y_ref"], name)
                self.assertEqual(npz["x"].dtype, np.float32)
                self.assertEqual(npz["idx"].dtype, np.int32)
                self.assertEqual(npz["y_ref"].dtype, np.float64)
                self.assertEqual(npz["n"].dtype, np.int64)
                self.assertEqual(int(npz["n"]), shape["n"])
                self.assertEqual(int(npz["seed"]), 7 + index, "형태마다 독립 시드 seed + index")
                self.assertEqual(npz["y_ref"].shape, (shape["n"],))
        self.assertEqual(list(manifest["files"]), list(TEST_NAMES))
        self.assertFalse(list(out.glob("*.tmp")), "임시 파일이 남으면 안 된다")

    def test_manifest_and_test_hash_definition(self) -> None:
        manifest = self.gen("a", FLOAT_SHAPES, float_case)
        on_disk = json.loads((self.tmp / "a" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(on_disk, manifest)
        self.assertEqual(sorted(manifest), ["files", "generated_at", "ref_method", "seed", "task", "test_hash"])
        self.assertEqual((manifest["task"], manifest["seed"], manifest["ref_method"]), ("demo", 7, "unit test"))
        self.assertRegex(manifest["generated_at"], r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
        # test_hash = sha256(str(seed) ‖ 형태마다 "k=v,…" ‖ 입력 바이트(인자 순서))[:16] — 참조는 들어가지 않는다
        h_all = hashlib.sha256(b"7")
        for index, (name, shape) in enumerate(FLOAT_SHAPES):
            dims, inputs, _, _ = float_case(shape, np.random.default_rng(7 + index))
            h_all.update(",".join(f"{k}={v}" for k, v in dims.items()).encode("ascii"))
            h_in = hashlib.sha256()
            for arr in inputs.values():
                h_in.update(arr.tobytes())
                h_all.update(arr.tobytes())
            self.assertEqual(manifest["files"][name], {**dims, "sha256_inputs": h_in.hexdigest()})
        self.assertEqual(manifest["test_hash"], h_all.hexdigest()[:16])

    def test_manifest_is_utf8_with_lf_newlines(self) -> None:
        self.gen("a", FLOAT_SHAPES, float_case)
        raw = (self.tmp / "a" / "manifest.json").read_bytes()
        self.assertNotIn(b"\r", raw)
        self.assertTrue(raw.endswith(b"}\n"))
        raw.decode("utf-8")

    def test_regeneration_is_byte_identical_and_seed_changes_everything(self) -> None:
        m1 = self.gen("a", FLOAT_SHAPES, float_case)
        m2 = self.gen("b", FLOAT_SHAPES, float_case)
        m3 = self.gen("c", FLOAT_SHAPES, float_case, seed=8)
        for name in TEST_NAMES:
            self.assertEqual(sha256_file(self.tmp / "a" / f"{name}.npz"), sha256_file(self.tmp / "b" / f"{name}.npz"), name)
            self.assertNotEqual(sha256_file(self.tmp / "a" / f"{name}.npz"), sha256_file(self.tmp / "c" / f"{name}.npz"), name)
        self.assertEqual(m1["test_hash"], m2["test_hash"])
        self.assertEqual(m1["files"], m2["files"])
        self.assertNotEqual(m1["test_hash"], m3["test_hash"])
        # 같은 디렉터리에 다시 써도(덮어쓰기) 같다
        before = {n: sha256_file(self.tmp / "a" / f"{n}.npz") for n in TEST_NAMES}
        self.gen("a", FLOAT_SHAPES, float_case)
        self.assertEqual(before, {n: sha256_file(self.tmp / "a" / f"{n}.npz") for n in TEST_NAMES})

    def test_shape_names_must_be_the_four_standard_names_in_order(self) -> None:
        for bad in (FLOAT_SHAPES[:3], [FLOAT_SHAPES[1], FLOAT_SHAPES[0], *FLOAT_SHAPES[2:]], [("a", {"n": 1})]):
            with self.assertRaises(ValueError):
                self.gen("bad", bad, float_case)
        self.assertFalse((self.tmp / "bad").exists(), "검증 실패 시 디렉터리를 만들지 않는다")

    def test_float_output_requires_float64_reference(self) -> None:
        def float32_ref(shape, rng):
            dims, inputs, out_name, ref = float_case(shape, rng)
            return dims, inputs, out_name, ref.astype(np.float32)

        with self.assertRaises(RuntimeError) as cm:
            self.gen("a", FLOAT_SHAPES, float32_ref)
        self.assertEqual(str(cm.exception), "demo/tiny: 참조는 float64 여야 한다(float32)")


class IntOutputTest(TestgenBase):
    def test_int_output_reference_is_float64_holding_integers(self) -> None:
        manifest = self.gen("a", INT_SHAPES, int_case)
        with np.load(self.tmp / "a" / "small.npz") as npz:
            self.assertEqual(sorted(npz.files), ["hist_ref", "keys", "n", "n_bins", "seed"])
            self.assertEqual(npz["hist_ref"].dtype, np.float64)
            self.assertTrue(np.array_equal(npz["hist_ref"], np.round(npz["hist_ref"])))
            self.assertEqual(int(npz["hist_ref"].sum()), 256)
            self.assertEqual(npz["keys"].dtype, np.int32)
        self.assertEqual(manifest["files"]["tiny"]["n_bins"], 4)

    def test_int32_reference_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            self.gen("a", INT_SHAPES, int32_ref_case)
        self.assertIn("참조는 float64 여야 한다(int32)", str(cm.exception))


class MainForTest(TestgenBase):
    def run_main(self, *args, **kw) -> int:
        with contextlib.redirect_stderr(self.stderr):
            return main_for(*args, **kw)

    def test_default_seed_and_out_argument(self) -> None:
        out = self.tmp / "cli"
        rc = self.run_main("demo", FLOAT_SHAPES, float_case, "unit test", self.tmp / "unused", ["--out", str(out)])
        self.assertEqual(rc, 0)
        manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["seed"], DEFAULT_SEED)
        self.assertEqual(DEFAULT_SEED, 20261105)
        self.assertFalse((self.tmp / "unused").exists())

    def test_seed_argument_and_default_out(self) -> None:
        out = self.tmp / "default_out"
        self.assertEqual(self.run_main("demo", FLOAT_SHAPES, float_case, "unit test", out, ["--seed", "11"]), 0)
        self.assertEqual(json.loads((out / "manifest.json").read_text(encoding="utf-8"))["seed"], 11)

    def test_errors_return_1_and_are_logged(self) -> None:
        out = self.tmp / "err"
        # float64 참조가 필요한데 int32 참조 → 1, stderr 에 사유
        rc = self.run_main("demo", INT_SHAPES, int32_ref_case, "unit test", out, [])
        self.assertEqual(rc, 1)
        self.assertIn("[gen_tests:demo] error: RuntimeError:", self.stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
