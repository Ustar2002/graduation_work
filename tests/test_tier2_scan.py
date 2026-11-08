"""Tier 2 ``scan`` 과제 테스트(설계 §17).

- gen_tests 의 결정성(같은 시드 → 바이트 동일), 참조의 손계산 대조(tiny), 체크인된 manifest 와 생성기의 일치.
- 더미 응답 4개의 추출(펜스 1개·시그니처 OK).
- python 정답 더미(numpy/0)의 worker check(tiny·rect) pass.
- 심은 버그: exclusive 스캔(한 칸 밀림)은 ``wrong_output``/``mismatch`` — 정수 출력은 허용오차 0 이라 ``numeric_instability`` 가 될 수 없다.
- C 기준선·C 더미를 실제 빌드 플래그로 컴파일해 worker check: 정답은 pass, 순차 의존을 무시한 c_openmp/1 은 pass 가 아니다
  (``race_condition`` 또는 ``wrong_output`` — 타이밍 의존; harness/dummy_responses/scan/README.md).

LLM 코드를 흉내 낸 커널은 모두 ``harness.worker`` 서브프로세스에서 실행한다(테스트 프로세스 안에서 실행하지 않는다).
테스트 npz(``tasks/scan/gen_tests.py``)가 없으면 npz 를 쓰는 테스트는 건너뛴다. gcc 가 없으면 C 테스트를 건너뛴다.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

import numpy as np
import yaml

from harness.build.c import build_c
from harness.classify import Tolerance, classify_task, load_tolerance
from harness.config import load_config
from harness.extract import extract
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from harness.testgen import DEFAULT_SEED, generate

TASK_DIR = ROOT / "tasks" / "scan"
TESTS_DIR = TASK_DIR / "tests"
DUMMY_DIR = ROOT / "harness" / "dummy_responses" / "scan"
SIGNATURES: dict = yaml.safe_load((TASK_DIR / "signatures.yaml").read_text(encoding="utf-8"))
EXACT = Tolerance(rtol=0.0, atol=0.0, numeric_pattern_tol_ratio=0.0, sentinel=-2048.0)
REF_METHOD = "numpy int64 cumsum"

# 심은 버그: exclusive 스캔(모든 원소가 한 칸 밀림). 정확 일치에서 wrong_output/mismatch 여야 한다.
PY_EXCLUSIVE = """import numpy as np

def scan(x, out, n):
    out[0] = 0
    out[1:] = np.cumsum(x[:-1], dtype=np.int32)
"""


def _gen_tests_module() -> ModuleType:
    """``tasks/scan/gen_tests.py`` 를 모듈로 로드한다(실행하지 않고 make_case/reference 만 쓴다)."""
    spec = importlib.util.spec_from_file_location("scan_gen_tests", TASK_DIR / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dummy_code(lang: str, k: int, backend: str) -> str:
    res = extract((DUMMY_DIR / lang / f"{k}.md").read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
    assert res.ok and res.code is not None, f"{lang}/{k}: {res.detail}"
    return res.code


def _npz(names: tuple[str, ...]) -> list[Path]:
    paths = [TESTS_DIR / f"{n}.npz" for n in names]
    return paths if all(p.exists() for p in paths) else []


def run_worker(backend: str, artifact: Path, mode: str, tests: list[Path], out: Path, *, threads: int,
               timeout_s: float = 300.0) -> dict:
    """``harness.worker`` 를 서브프로세스로 실행하고 결과 JSON 을 돌려준다(rc != 0 이면 AssertionError)."""
    cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", mode,
           "--tests", ",".join(str(t) for t in tests), "--tolerance", str(TASK_DIR / "tolerance.yaml"),
           "--signatures", str(TASK_DIR / "signatures.yaml"), "--out", str(out)]
    env = dict(launch_env_minimal())
    env.update({"OMP_NUM_THREADS": str(threads), "OPENBLAS_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)})
    proc = run_isolated(cmd, timeout_s=timeout_s, cwd=ROOT, env=env)
    if proc.rc != 0:
        raise AssertionError(f"worker rc={proc.rc}: {proc.stderr_tail}")
    return json.loads(out.read_text(encoding="utf-8"))


class SpecFilesTest(unittest.TestCase):
    """과제 파일(signatures/tolerance/spec)이 §17 계약과 맞는가."""

    def test_taskspec_and_tolerance(self) -> None:
        spec = load_taskspec(TASK_DIR / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("scan", "scan", ["x", "out", "n"]))
        self.assertEqual(spec.inputs[0].dtype, "int32")
        self.assertEqual(spec.output.dtype, "int32")
        self.assertEqual(spec.shape_of(spec.output, {"n": 7}), (7,))
        self.assertIsNone(spec.work)
        self.assertIsNone(spec.work_of({"n": 7}))
        self.assertEqual(spec.baseline["backend"], "c")
        self.assertTrue(spec.baseline["openmp"])
        self.assertTrue((TASK_DIR / spec.baseline["file"]).is_file())
        tol = load_tolerance(TASK_DIR / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio, tol.sentinel), (0.0, 0.0, 0.0, -2048.0))
        self.assertEqual((tol.task, tol.dtype), ("scan", "int32"))
        for lang in ("numpy", "c_seq", "c_openmp"):
            entry = SIGNATURES["languages"][lang]
            self.assertIn("scan(", entry["positional"])
            self.assertTrue(entry["param_docs"].strip())
        self.assertEqual(SIGNATURES["languages"]["numpy"]["keyword"], "def scan(*, x, out, n):")

    def test_spec_structure_is_language_neutral(self) -> None:
        spec_md = (TASK_DIR / "spec.md").read_text(encoding="utf-8")
        for heading in ("## Definition", "## Data layout", "## Contract", "## Accuracy", "## Evaluation", "## Goal"):
            self.assertIn(heading, spec_md)
        for shape in ("7", "100,003", "1,048,576", "33,554,432"):
            self.assertIn(shape, spec_md)
        for word in ("numpy", "NumPy", "OpenMP", "cumsum", "BLAS", "#pragma"):
            self.assertNotIn(word, spec_md)
        self.assertTrue((TASK_DIR / "spec.ko.md").read_text(encoding="utf-8").startswith("# 과제:"))


class GenTestsTest(unittest.TestCase):
    """생성기: 손계산 참조, 형태, 결정성, 체크인된 manifest 와의 일치."""

    SMALL_SHAPES = [("tiny", {"n": 7}), ("rect", {"n": 100_003}), ("small", {"n": 4096}), ("large", {"n": 65_536})]

    def setUp(self) -> None:
        self.gen = _gen_tests_module()

    def test_reference_hand_check(self) -> None:
        x = np.array([3, 0, 31, 5, 5, 0, 7], dtype=np.int32)
        ref = self.gen.reference(x)
        self.assertEqual(ref.dtype, np.float64)
        self.assertEqual(ref.tolist(), [3.0, 3.0, 34.0, 39.0, 44.0, 44.0, 51.0])
        dims, inputs, out_name, ref_tiny = self.gen.make_case({"n": 7}, np.random.default_rng(DEFAULT_SEED))
        self.assertEqual((dims, out_name), ({"n": 7}, "out"))
        xt = inputs["x"]
        self.assertEqual((xt.dtype, xt.shape), (np.int32, (7,)))
        self.assertTrue(np.all((xt >= 0) & (xt < 32)))
        acc, expected = 0, []
        for v in xt.tolist():  # 순수 Python 누적
            acc += v
            expected.append(float(acc))
        self.assertEqual(ref_tiny.tolist(), expected)
        with self.assertRaises(RuntimeError):  # int32 를 넘는 접두합은 계약 위반
            self.gen.reference(np.array([2**30, 2**30], dtype=np.int64))

    def test_shapes(self) -> None:
        self.assertEqual([n for n, _ in self.gen.SHAPES], ["tiny", "rect", "small", "large"])
        self.assertEqual([s["n"] for _, s in self.gen.SHAPES], [7, 100_003, 1 << 20, 1 << 25])
        self.assertLess(31 * (1 << 25), 2**31)  # 최악의 총합도 int32 안

    def test_generate_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            a, b, c = Path(tmp) / "a", Path(tmp) / "b", Path(tmp) / "c"
            ma = generate("scan", DEFAULT_SEED, a, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            mb = generate("scan", DEFAULT_SEED, b, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertEqual(ma["test_hash"], mb["test_hash"])
            self.assertEqual(ma["files"], mb["files"])
            for name, shape in self.SMALL_SHAPES:
                self.assertEqual((a / f"{name}.npz").read_bytes(), (b / f"{name}.npz").read_bytes(), name)
                with np.load(a / f"{name}.npz") as npz:
                    self.assertEqual(set(npz.files), {"x", "out_ref", "n", "seed"})
                    self.assertEqual(npz["x"].dtype, np.int32)
                    self.assertEqual(npz["out_ref"].dtype, np.float64)
                    self.assertEqual((int(npz["n"]), npz["x"].shape), (shape["n"], (shape["n"],)))
                    self.assertTrue(np.array_equal(npz["out_ref"], np.cumsum(npz["x"], dtype=np.int64)))
            mc = generate("scan", DEFAULT_SEED + 1, c, self.SMALL_SHAPES, self.gen.make_case, REF_METHOD)
            self.assertNotEqual(ma["test_hash"], mc["test_hash"])

    def test_checked_in_tests_match_generator(self) -> None:
        manifest_path = TESTS_DIR / "manifest.json"
        if not manifest_path.exists():
            self.skipTest("tasks/scan/tests 없음 (tasks/scan/gen_tests.py)")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shapes = dict(self.gen.SHAPES)
        for index, name in enumerate(("tiny", "rect")):
            dims, inputs, _, ref = self.gen.make_case(shapes[name], np.random.default_rng(int(manifest["seed"]) + index))
            self.assertEqual(manifest["files"][name]["n"], dims["n"])
            self.assertEqual(manifest["files"][name]["sha256_inputs"], hashlib.sha256(inputs["x"].tobytes()).hexdigest())
            with np.load(TESTS_DIR / f"{name}.npz") as npz:
                self.assertTrue(np.array_equal(npz["x"], inputs["x"]), name)
                self.assertTrue(np.array_equal(npz["out_ref"], ref), name)


class ClassifyTest(unittest.TestCase):
    """정수 출력의 정확 일치 판정(설계 §10 규칙을 scan 출력에 적용)."""

    def test_exact_equality_rules(self) -> None:
        x = np.array([3, 0, 31, 5, 5, 0, 7], dtype=np.int32)
        ref = np.cumsum(x, dtype=np.int64).astype(np.float64)
        ok = ref.astype(np.int32)
        inputs, dims = {"x": x}, {"n": 7}
        self.assertEqual(classify_task("scan", ok, ref, inputs, dims, EXACT)[0], "pass")
        exclusive = np.concatenate([[0], ref[:-1]]).astype(np.int32)
        st, pat, m = classify_task("scan", exclusive, ref, inputs, dims, EXACT)
        self.assertEqual((st, pat), ("wrong_output", "mismatch"))
        self.assertTrue(np.isinf(m["max_tol_ratio"]))
        self.assertEqual(m["n_fail"], int(np.count_nonzero(exclusive != ok)))
        off_by_one = ok.copy()
        off_by_one[3] += 1
        self.assertEqual(classify_task("scan", off_by_one, ref, inputs, dims, EXACT)[:2], ("wrong_output", "mismatch"))
        unwritten = np.full(7, -2048, dtype=np.int32)
        self.assertEqual(classify_task("scan", unwritten, ref, inputs, dims, EXACT)[:2], ("wrong_output", "output_not_written"))
        partial = ok.copy()
        partial[5:] = -2048
        self.assertEqual(classify_task("scan", partial, ref, inputs, dims, EXACT)[:2], ("wrong_output", "partial_write:2/7"))
        accumulated = (ok.astype(np.int64) - 2048).astype(np.int32)  # 센티널 위에 더한 커널
        self.assertEqual(classify_task("scan", accumulated, ref, inputs, dims, EXACT)[:2], ("wrong_output", "accumulated_into_c"))


class DummyResponsesTest(unittest.TestCase):
    """더미 응답 4개(numpy/0, c_seq/0, c_openmp/0, c_openmp/1)는 모두 펜스 1개·시그니처 OK 로 추출돼야 한다."""

    def test_all_dummies_extract(self) -> None:
        seen = 0
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            for md in sorted((DUMMY_DIR / lang).glob("*.md")):
                res = extract(md.read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
                with self.subTest(lang=lang, k=md.stem):
                    self.assertEqual(res.n_blocks, 1)
                    self.assertTrue(res.ok, res.detail)
                    self.assertTrue(res.signature_ok)
                    self.assertEqual(res.detail, "")
                seen += 1
        self.assertEqual(seen, 4)


class WorkerTestBase(unittest.TestCase):
    """worker check 를 실제 테스트 npz 로 돌리는 공통 부분(npz 가 없으면 건너뜀)."""

    def check(self, backend: str, artifact: Path, names: tuple[str, ...], threads: int, tmp: Path) -> dict:
        tests = _npz(names)
        if not tests:
            self.skipTest("tasks/scan/tests/*.npz 없음 (tasks/scan/gen_tests.py)")
        return run_worker(backend, artifact, "check", tests, tmp / f"check_{'_'.join(names)}.json", threads=threads)


class PythonKernelsTest(WorkerTestBase):
    def test_numpy_dummy_passes_check(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text(_dummy_code("numpy", 0, "python"), encoding="utf-8")
            doc = self.check("python", kernel, ("tiny", "rect"), 4, Path(tmp))
        self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))
        self.assertEqual(doc["task"], "scan")
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect"])
        self.assertTrue(all(t["classification"] == "pass" for t in doc["tests"]))
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["nondeterministic"])
        self.assertEqual(doc["max_abs_err"], 0.0)

    def test_exclusive_scan_is_wrong_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            kernel = Path(tmp) / "kernel.py"
            kernel.write_text(PY_EXCLUSIVE, encoding="utf-8")
            doc = self.check("python", kernel, ("tiny", "rect"), 4, Path(tmp))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("wrong_output", "mismatch"), t["name"])
            self.assertFalse(t["nondeterministic"])
        self.assertIsNone(doc["max_tol_ratio"])  # 허용오차 0: 배율 inf 는 JSON 에 null 로 기록된다; numeric_instability 불가
        self.assertFalse(doc["oob_write"])


@unittest.skipIf(shutil.which("gcc") is None, "gcc 없음")
class CKernelsTest(WorkerTestBase):
    """C 기준선과 C 더미를 config 의 실제 플래그로 빌드해 check 한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.bcfg = load_config().build

    def build(self, code: str, out_dir: Path, *, openmp: bool) -> Path:
        res = build_c(code, out_dir, self.bcfg, openmp=openmp, launch_prefix=[], symbol="scan")
        self.assertTrue(res.ok, f"{res.error_kind}: {res.error_detail}\n{res.log[-1500:]}")
        self.assertEqual(res.forbidden_symbols, [])
        assert res.artifact is not None
        return res.artifact

    def test_baseline_and_correct_dummies_pass(self) -> None:
        cases = [
            ("baseline", (TASK_DIR / "baseline" / "baseline.c").read_text(encoding="utf-8"), True, 4),
            ("c_seq/0", _dummy_code("c_seq", 0, "c"), False, 1),
            ("c_openmp/0", _dummy_code("c_openmp", 0, "c"), True, 4),
        ]
        for name, code, openmp, threads in cases:
            with self.subTest(kernel=name), tempfile.TemporaryDirectory() as tmp:
                so = self.build(code, Path(tmp), openmp=openmp)
                doc = self.check("c", so, ("tiny", "rect", "small"), threads, Path(tmp))
                self.assertEqual(doc["classification"], "pass", f"{name}: {doc.get('error_detail')}")
                self.assertFalse(doc["oob_write"])
                self.assertFalse(doc["inputs_mutated"])
                self.assertEqual(doc["max_abs_err"], 0.0)

    def test_c_seq_dummy_has_no_thread_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            res = build_c(_dummy_code("c_seq", 0, "c"), Path(tmp), self.bcfg, openmp=False, launch_prefix=[], symbol="scan")
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.thread_symbols, [])

    def test_naive_parallel_for_is_not_pass(self) -> None:
        """c_openmp/1: 순차 의존 out[i-1] 을 무시한 parallel for. 블록 경계에서 미기록 값을 읽어 틀린다 —
        호출마다 다르면 race_condition, 같게 틀리면 wrong_output (둘 다 허용; 어느 쪽인지는 타이밍 의존)."""
        with tempfile.TemporaryDirectory() as tmp:
            so = self.build(_dummy_code("c_openmp", 1, "c"), Path(tmp), openmp=True)
            doc = self.check("c", so, ("tiny", "rect", "small"), 4, Path(tmp))
        self.assertIn(doc["classification"], ("race_condition", "wrong_output"), doc.get("error_detail"))
        self.assertTrue(any(t["classification"] != "pass" for t in doc["tests"]))


if __name__ == "__main__":
    unittest.main()
