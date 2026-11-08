"""Tier 2 histogram 과제 테스트(설계 §17): 생성기 결정성, tiny 참조 손계산, 정답 python 커널의 worker check,
0 초기화를 빼먹은 커널의 분류(``accumulated_into_c``), 더미 응답 추출, 과제 명세 파일 파싱.

워커는 ``harness.procutil.run_isolated`` 로 격리 실행한다(tests/test_taskspec.py 와 같은 패턴). 테스트 npz 는 임시
디렉터리에 생성기와 같은 코드·시드로 만들므로 ``make tests`` 가 필요 없다.
"""
from __future__ import annotations

import filecmp
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from harness.classify import load_tolerance
from harness.extract import extract
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from harness.testgen import DEFAULT_SEED, TEST_NAMES, generate, write_npz_atomic

TASK_DIR = ROOT / "tasks" / "histogram"
DUMMY_DIR = ROOT / "harness" / "dummy_responses" / "histogram"

PY_OK = (
    "import numpy as np\n"
    "def histogram(data, hist, n, nbins):\n"
    "    hist[...] = np.bincount(data, minlength=nbins)\n"
)
# 심은 버그: 0 초기화를 빼먹고 센티널 위에 더한다 → wrong_output / accumulated_into_c (설계 §10 규칙 4b)
PY_NO_ZERO = (
    "import numpy as np\n"
    "def histogram(data, hist, n, nbins):\n"
    "    hist += np.bincount(data, minlength=nbins).astype(np.int32)\n"
)


def load_gen_tests() -> ModuleType:
    """``tasks/histogram/gen_tests.py`` 를 모듈로 읽는다(실행하지 않음)."""
    spec = importlib.util.spec_from_file_location("histogram_gen_tests", TASK_DIR / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_small_tests(out_dir: Path, names: tuple[str, ...] = ("tiny", "rect")) -> list[Path]:
    """생성기와 같은 코드·시드(``DEFAULT_SEED + index``)로 tiny·rect npz 를 만든다(large 128 MB 는 피한다)."""
    mod = load_gen_tests()
    paths: list[Path] = []
    for index, (name, shape) in enumerate(mod.SHAPES):
        if name not in names:
            continue
        dims, inputs, out_name, ref = mod.make_case(shape, np.random.default_rng(DEFAULT_SEED + index))
        path = out_dir / f"{name}.npz"
        write_npz_atomic(path, **inputs, **{f"{out_name}_ref": ref}, **{k: np.int64(v) for k, v in dims.items()},
                         seed=np.int64(DEFAULT_SEED + index))
        paths.append(path)
    return paths


def run_worker_check(kernel_code: str, tests: list[Path], workdir: Path) -> dict:
    kernel = workdir / "kernel.py"
    kernel.write_text(kernel_code, encoding="utf-8")
    out = workdir / "check.json"
    cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", "python", "--artifact", str(kernel), "--mode", "check",
           "--tests", ",".join(str(t) for t in tests), "--tolerance", str(TASK_DIR / "tolerance.yaml"),
           "--signatures", str(TASK_DIR / "signatures.yaml"), "--out", str(out)]
    proc = run_isolated(cmd, timeout_s=120, cwd=workdir, env=launch_env_minimal())
    if proc.rc != 0:
        raise AssertionError(f"worker rc={proc.rc}: {proc.stderr_tail}")
    return json.loads(out.read_text(encoding="utf-8"))


class GenTestsTest(unittest.TestCase):
    def test_generate_twice_is_byte_identical(self) -> None:
        """같은 시드로 두 번 만들면 네 npz 가 바이트 단위로 같고 test_hash 도 같다. 값 범위·개수 합도 확인한다."""
        mod = load_gen_tests()
        self.assertEqual([n for n, _ in mod.SHAPES], list(TEST_NAMES))
        self.assertEqual([s for _, s in mod.SHAPES], [{"n": 13, "nbins": 4}, {"n": 100_003, "nbins": 37},
                                                      {"n": 1 << 20, "nbins": 256}, {"n": 1 << 25, "nbins": 1024}])
        with tempfile.TemporaryDirectory(prefix="hist_gen_a_") as a, tempfile.TemporaryDirectory(prefix="hist_gen_b_") as b:
            m1 = generate("histogram", DEFAULT_SEED, Path(a), mod.SHAPES, mod.make_case, "test")
            m2 = generate("histogram", DEFAULT_SEED, Path(b), mod.SHAPES, mod.make_case, "test")
            self.assertEqual(m1["test_hash"], m2["test_hash"])
            self.assertEqual(m1["files"], m2["files"])
            for name, shape in mod.SHAPES:
                with self.subTest(shape=name):
                    pa, pb = Path(a) / f"{name}.npz", Path(b) / f"{name}.npz"
                    self.assertTrue(filecmp.cmp(pa, pb, shallow=False), f"{name}.npz 가 바이트 단위로 다르다")
                    with np.load(pa) as npz:
                        data, ref = npz["data"], npz["hist_ref"]
                        self.assertEqual(int(npz["n"]), shape["n"])
                        self.assertEqual(int(npz["nbins"]), shape["nbins"])
                    self.assertEqual(data.dtype, np.int32)
                    self.assertEqual(data.shape, (shape["n"],))
                    self.assertEqual(ref.dtype, np.float64)
                    self.assertEqual(ref.shape, (shape["nbins"],))
                    self.assertTrue(bool(((data >= 0) & (data < shape["nbins"])).all()), "값이 [0, nbins) 를 벗어난다")
                    self.assertEqual(int(ref.sum()), shape["n"])
                    self.assertTrue(bool((ref == np.round(ref)).all()), "참조는 정수 값이어야 한다")

    def test_tiny_reference_equals_hand_count(self) -> None:
        mod = load_gen_tests()
        name, shape = mod.SHAPES[0]
        self.assertEqual(name, "tiny")
        dims, inputs, out_name, ref = mod.make_case(shape, np.random.default_rng(DEFAULT_SEED + 0))
        self.assertEqual((dims, out_name), ({"n": 13, "nbins": 4}, "hist"))
        self.assertEqual(list(inputs), ["data"])
        data = inputs["data"]
        self.assertEqual((data.dtype, data.shape), (np.dtype(np.int32), (13,)))
        counts = [0] * 4
        for v in data.tolist():
            self.assertTrue(0 <= v < 4, v)
            counts[v] += 1
        self.assertEqual(sum(counts), 13)
        self.assertEqual(ref.tolist(), [float(c) for c in counts])
        # 고정 시드(DEFAULT_SEED=20261105)의 tiny 는 빈 1 이 비어 있다 — "개수 0 인 빈도 써야 한다"는 규칙을 검사하는 형태.
        # 생성기 코드·시드를 바꾸면 이 값과 README(harness/dummy_responses/histogram)를 함께 갱신한다.
        self.assertEqual(ref.tolist(), [4.0, 0.0, 3.0, 6.0])


class WorkerCheckTest(unittest.TestCase):
    def test_correct_python_kernel_passes_tiny_and_rect(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hist_ok_") as tmp:
            wd = Path(tmp)
            doc = run_worker_check(PY_OK, write_small_tests(wd), wd)
        self.assertEqual(doc["task"], "histogram")
        self.assertEqual((doc["classification"], doc["error_detail"]), ("pass", ""), doc.get("tests"))
        self.assertFalse(doc["nondeterministic"])
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect"])
        for t in doc["tests"]:
            self.assertEqual(t["classification"], "pass", t)
            self.assertEqual(len(t["runs"]), 2)
            for run in t["runs"]:
                self.assertEqual(run["n_unwritten"], 0)
                self.assertEqual(run["max_abs_err"], 0.0)

    def test_forgetting_to_zero_is_accumulated_into_c(self) -> None:
        """센티널 위에 더한 출력은 (out − sentinel) == ref 가 정확히 성립 → wrong_output / accumulated_into_c."""
        with tempfile.TemporaryDirectory(prefix="hist_nozero_") as tmp:
            wd = Path(tmp)
            doc = run_worker_check(PY_NO_ZERO, write_small_tests(wd), wd)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "accumulated_into_c"))
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("wrong_output", "accumulated_into_c"), t["name"])
            self.assertFalse(t["nondeterministic"])
            self.assertEqual(t["runs"][0]["max_abs_err"], 2048.0)
        # tiny 의 빈 1 은 개수 0 이라 sentinel + 0 == sentinel 로 '미기록'으로 세어진다(n_unwritten=1). 규칙 4b(누적)가
        # 4c(partial_write)보다 먼저이므로 그래도 accumulated_into_c 다; rect 는 모든 빈이 비어 있지 않아 n_unwritten=0.
        by_name = {t["name"]: t for t in doc["tests"]}
        self.assertEqual(by_name["tiny"]["runs"][0]["n_unwritten"], 1)
        self.assertEqual(by_name["rect"]["runs"][0]["n_unwritten"], 0)


class SpecFilesTest(unittest.TestCase):
    def test_taskspec_and_tolerance(self) -> None:
        spec = load_taskspec(TASK_DIR / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("histogram", "histogram", ["data", "hist", "n", "nbins"]))
        self.assertEqual((spec.output.name, spec.output.dtype), ("hist", "int32"))
        self.assertEqual(spec.inputs[0].dtype, "int32")
        self.assertEqual(spec.shape_of(spec.output, {"n": 13, "nbins": 4}), (4,))
        self.assertEqual(spec.shape_of(spec.inputs[0], {"n": 13, "nbins": 4}), (13,))
        self.assertIsNone(spec.work_of({"n": 13, "nbins": 4}))
        self.assertEqual((spec.baseline["backend"], spec.baseline["file"], spec.baseline["openmp"]),
                         ("c", "baseline/baseline.c", True))
        self.assertTrue((TASK_DIR / "baseline" / "baseline.c").is_file())
        tol = load_tolerance(TASK_DIR / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio, tol.sentinel), (0.0, 0.0, 0.0, -2048.0))
        raw = yaml.safe_load((TASK_DIR / "signatures.yaml").read_text(encoding="utf-8"))
        for lang in ("numpy", "c_seq", "c_openmp"):
            self.assertIn("histogram(", raw["languages"][lang]["positional"])
            self.assertTrue(raw["languages"][lang]["param_docs"].strip())
        for fname in ("spec.md", "spec.ko.md"):
            text = (TASK_DIR / fname).read_text(encoding="utf-8")
            self.assertIn("(33,554,432, 1024)", text, fname)
            self.assertIn("(13, 4)", text, fname)

    def test_dummy_responses_extract(self) -> None:
        """더미 응답 4개(numpy/0, c_seq/0, c_openmp/0, c_openmp/1)는 모두 블록 1개·시그니처 OK 로 추출돼야 한다."""
        signatures = yaml.safe_load((TASK_DIR / "signatures.yaml").read_text(encoding="utf-8"))
        expected = {("numpy", 0), ("c_seq", 0), ("c_openmp", 0), ("c_openmp", 1)}
        seen: set[tuple[str, int]] = set()
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            for md in sorted((DUMMY_DIR / lang).glob("*.md")):
                k = int(md.stem)
                seen.add((lang, k))
                res = extract(md.read_text(encoding="utf-8"), backend=backend, signature_spec=signatures)
                with self.subTest(lang=lang, k=k):
                    self.assertEqual(res.n_blocks, 1)
                    self.assertTrue(res.ok, res.detail)
                    self.assertTrue(res.signature_ok)
                    self.assertTrue(res.signature_exact, res.detail)
                    self.assertEqual(res.detail, "")
        self.assertEqual(seen, expected)


if __name__ == "__main__":
    unittest.main()
