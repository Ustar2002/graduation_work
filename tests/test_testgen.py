"""testgen 테스트 (설계 §16 + v3 §2.4): npz 키·dtype·manifest·test_hash 정의, 재생성 시 바이트 동일,
출력 dtype 에 따른 참조 dtype 검사(float32 → float64, int32 → int32), 센티널 충돌 검사, CLI(main_for)."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness import testgen
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
    n, n_bins = shape["n"], shape["n_bins"]
    keys = rng.integers(0, n_bins, size=n, dtype=np.int32)
    return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", np.bincount(keys, minlength=n_bins).astype(np.int32)


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

    def test_explicit_defaults_equal_implicit_defaults(self) -> None:
        self.gen("a", FLOAT_SHAPES, float_case)
        self.gen("b", FLOAT_SHAPES, float_case, out_dtype="float32", sentinel=None)
        for name in TEST_NAMES:
            self.assertEqual(sha256_file(self.tmp / "a" / f"{name}.npz"), sha256_file(self.tmp / "b" / f"{name}.npz"), name)
        self.assertNotIn("경고", self.stderr.getvalue())

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

    def test_sentinel_check_also_applies_to_float_outputs_when_given(self) -> None:
        def with_sentinel(shape, rng):
            dims, inputs, out_name, ref = float_case(shape, rng)
            if dims["n"] == 17:
                ref[5] = -2048.0
            return dims, inputs, out_name, ref

        self.gen("ok", FLOAT_SHAPES, with_sentinel)  # sentinel 을 주지 않으면 검사하지 않는다(기존 동작)
        with self.assertRaises(RuntimeError) as cm:
            self.gen("bad", FLOAT_SHAPES, with_sentinel, sentinel=-2048.0)
        self.assertIn("demo/rect", str(cm.exception))


class IntOutputTest(TestgenBase):
    def test_int32_reference_is_written_as_int32(self) -> None:
        manifest = self.gen("a", INT_SHAPES, int_case, out_dtype="int32", sentinel=-2048)
        with np.load(self.tmp / "a" / "small.npz") as npz:
            self.assertEqual(sorted(npz.files), ["hist_ref", "keys", "n", "n_bins", "seed"])
            self.assertEqual(npz["hist_ref"].dtype, np.int32, "int64 로 넓히지 않는다")
            self.assertEqual(int(npz["hist_ref"].sum()), 256)
        self.assertEqual(manifest["files"]["tiny"]["n_bins"], 4)
        self.assertNotIn("경고", self.stderr.getvalue())

    def test_int_output_rejects_wider_or_float_references(self) -> None:
        for dtype in (np.int64, np.float64, np.uint32):
            def wrong(shape, rng, _dtype=dtype):
                dims, inputs, out_name, ref = int_case(shape, rng)
                return dims, inputs, out_name, ref.astype(_dtype)

            with self.assertRaises(RuntimeError, msg=str(dtype)) as cm:
                self.gen(f"bad_{np.dtype(dtype).name}", INT_SHAPES, wrong, out_dtype="int32", sentinel=-2048)
            self.assertIn("int32 출력의 참조는 int32 여야 한다", str(cm.exception))

    def test_int_reference_given_for_float_output_is_rejected(self) -> None:
        with self.assertRaises(RuntimeError) as cm:
            self.gen("a", INT_SHAPES, int_case)  # out_dtype 기본값은 float32
        self.assertIn("참조는 float64 여야 한다(int32)", str(cm.exception))

    def test_sentinel_in_reference_is_rejected(self) -> None:
        def with_sentinel(shape, rng):
            dims, inputs, out_name, ref = int_case(shape, rng)
            if dims["n"] == 256:
                ref[3] = -2048
                ref[4] = -2048
            return dims, inputs, out_name, ref

        with self.assertRaises(RuntimeError) as cm:
            self.gen("a", INT_SHAPES, with_sentinel, out_dtype="int32", sentinel=-2048)
        message = str(cm.exception)
        self.assertIn("demo/small", message)
        self.assertIn("센티널 값 -2048 이 2개", message)
        # 앞선 형태(tiny, rect)는 이미 쓰였지만 manifest 는 없다 → run.ensure_tests 가 불완전한 디렉터리로 본다
        self.assertFalse((self.tmp / "a" / "manifest.json").exists())
        self.assertFalse((self.tmp / "a" / "small.npz").exists())

    def test_missing_sentinel_for_int_output_warns_but_generates(self) -> None:
        self.gen("a", INT_SHAPES, int_case, out_dtype="int32")
        self.assertTrue((self.tmp / "a" / "manifest.json").exists())
        self.assertIn("sentinel 을 주지 않아", self.stderr.getvalue())

    def test_unknown_out_dtype(self) -> None:
        with self.assertRaises(ValueError):
            self.gen("a", INT_SHAPES, int_case, out_dtype="int64")
        with self.assertRaises(ValueError):
            self.gen("a", FLOAT_SHAPES, float_case, out_dtype="float64")


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

    def test_out_dtype_and_sentinel_are_forwarded(self) -> None:
        out = self.tmp / "int"
        rc = self.run_main("demo", INT_SHAPES, int_case, "unit test", out, [], out_dtype="int32", sentinel=-2048)
        self.assertEqual(rc, 0)
        with np.load(out / "tiny.npz") as npz:
            self.assertEqual(npz["hist_ref"].dtype, np.int32)

    def test_errors_return_1_and_are_logged(self) -> None:
        out = self.tmp / "err"
        # float 참조가 필요한데 int32 참조 → 1, stderr 에 사유
        rc = self.run_main("demo", INT_SHAPES, int_case, "unit test", out, [])
        self.assertEqual(rc, 1)
        self.assertIn("[gen_tests:demo] error: RuntimeError:", self.stderr.getvalue())
        # 센티널 충돌도 1
        def all_sentinel(shape, rng):
            dims, inputs, out_name, ref = int_case(shape, rng)
            return dims, inputs, out_name, np.full_like(ref, 5)

        self.assertEqual(self.run_main("demo", INT_SHAPES, all_sentinel, "unit test", out, [], out_dtype="int32", sentinel=5), 1)


class ReferenceCheckTest(unittest.TestCase):
    def test_check_reference_table(self) -> None:
        ok = [(np.zeros(3, np.float64), "float32", None), (np.zeros(3, np.float64), "float32", -2048.0),
              (np.zeros(3, np.int32), "int32", -2048), (np.zeros(3, np.int32), "int32", None)]
        for ref, out_dtype, sentinel in ok:
            testgen.check_reference("t", "tiny", ref, out_dtype, sentinel)
        bad = [(np.zeros(3, np.float32), "float32", None), (np.zeros(3, np.int32), "float32", None),
               (np.zeros(3, np.int64), "int32", None), (np.zeros(3, np.float64), "int32", None),
               (np.zeros(3, np.int32), "int32", 0), (np.full(3, -2048.0), "float32", -2048.0)]
        for ref, out_dtype, sentinel in bad:
            with self.assertRaises(RuntimeError, msg=f"{ref.dtype} {out_dtype} {sentinel}"):
                testgen.check_reference("t", "tiny", ref, out_dtype, sentinel)


if __name__ == "__main__":
    unittest.main()
