"""Tier 2 ``sort`` 과제 테스트 (docs/harness_design.md §17).

- gen_tests 결정성(같은 시드 → npz 바이트 동일·test_hash 동일), tiny 참조의 손계산 대조, 과제 명세·허용오차 파싱, spec.md 의 계약 문구.
- 더미 정답 커널(``harness/dummy_responses/sort/<lang>/0.md``)이 추출 단계를 통과하고 worker check 에서 pass 한다.
  테스트 입력은 여기서 만든다: gen_tests 와 같은 시드의 tiny(7)·rect(100,003) + 중복이 많은 입력(값 [0, 10), n = 1000) + n = 1 + 극값만 있는 입력.
  c_openmp 더미는 스레드 4·3·1 로 각각 검사한다(고르지 않은 분할 포함).
- 심은 버그(python): 내림차순 → ``wrong_output``/``mismatch``, 마지막 원소 미기록 → ``wrong_output``/``partial_write:1/7``,
  입력 제자리 정렬 → ``wrong_output``/``inputs_mutated``.
C 커널 테스트는 gcc/nm 이 없으면 건너뛴다. 저장소 안에는 아무것도 쓰지 않는다(임시 디렉터리만).
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

from harness.build.c import build_c
from harness.classify import load_tolerance
from harness.config import BuildCfg
from harness.extract import extract, parse_fenced_blocks
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.run import load_signatures, signature_spec_for
from harness.taskspec import load_taskspec
from harness.testgen import DEFAULT_SEED, TEST_NAMES, generate

TASK = "sort"
TASK_DIR = ROOT / "tasks" / TASK
DUMMY_DIR = ROOT / "harness" / "dummy_responses" / TASK
SIGNATURES = TASK_DIR / "signatures.yaml"
TOLERANCE = TASK_DIR / "tolerance.yaml"
LANG_BACKENDS = (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c"))
HAVE_GCC = shutil.which("gcc") is not None and shutil.which("nm") is not None
INT32_MAX = 2**31 - 1

# 설계 §7 의 고정 빌드 명령(config/experiment.yaml 의 build 와 같다; 테스트는 설정 로더에 의존하지 않는다)
BCFG = BuildCfg(
    cc="gcc",
    cflags=["-O3", "-march=native"],
    base_flags=["-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition"],
    openmp_flag="-fopenmp",
    ldflags=["-lm", "-Wl,--no-undefined"],
    diag_flags=["-fopt-info-vec-optimized", "-fopt-info-vec-missed"],
    compile_timeout_s=60,
    forbidden_symbols=["pthread_create", "clone", "fork", "vfork", "system", "popen", "execve", "execv", "execl"],
    thread_symbols_prefix=["pthread_create", "GOMP_", "omp_", "__kmpc_"],
)

# 심은 버그(python 백엔드)
DESCENDING = "import numpy as np\n\ndef sort(x, out, n):\n    out[...] = np.sort(x)[::-1]\n"
LAST_UNWRITTEN = "import numpy as np\n\ndef sort(x, out, n):\n    out[:-1] = np.sort(x)[:-1]\n"
INPLACE_INPUT = "import numpy as np\n\ndef sort(x, out, n):\n    x.sort()\n    out[...] = x\n"


def load_gen_tests() -> ModuleType:
    """``tasks/sort/gen_tests.py`` 를 모듈로 읽는다(tasks 는 패키지가 아니다)."""
    spec = importlib.util.spec_from_file_location("sort_gen_tests", TASK_DIR / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dummy_code(language: str) -> str:
    """더미 응답의 유일한 펜스 블록."""
    blocks = parse_fenced_blocks((DUMMY_DIR / language / "0.md").read_text(encoding="utf-8"))
    assert len(blocks) == 1, f"{language}/0.md 의 코드 블록 수가 {len(blocks)} 이다"
    return blocks[0].code


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_case(path: Path, x: np.ndarray) -> None:
    """worker 가 읽는 형식의 테스트 npz(``x``, ``out_ref``, ``n``)."""
    x = np.ascontiguousarray(x, dtype=np.int32)
    with open(path, "wb") as fh:
        np.savez(fh, x=x, out_ref=np.sort(x).astype(np.float64), n=np.int64(x.size), seed=np.int64(0))


class GenTestsTest(unittest.TestCase):
    def test_shapes_follow_the_contract(self) -> None:
        mod = load_gen_tests()
        self.assertEqual([n for n, _ in mod.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in mod.SHAPES], [7, 100_003, 1 << 20, 10_000_000])

    def test_same_seed_gives_identical_files(self) -> None:
        mod = load_gen_tests()
        with tempfile.TemporaryDirectory(prefix="sort_gen_") as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            ma = generate(TASK, DEFAULT_SEED, a, mod.SHAPES, mod.make_case, "test")
            mb = generate(TASK, DEFAULT_SEED, b, mod.SHAPES, mod.make_case, "test")
            self.assertEqual(ma["test_hash"], mb["test_hash"])
            self.assertEqual(ma["files"], mb["files"])
            for name in TEST_NAMES:
                self.assertEqual(sha256_of(a / f"{name}.npz"), sha256_of(b / f"{name}.npz"), name)
            self.assertEqual(ma["files"]["large"]["n"], 10_000_000)

    def test_tiny_reference_is_hand_sorted_copy(self) -> None:
        mod = load_gen_tests()
        dims, inputs, out_name, ref = mod.make_case({"n": 7}, np.random.default_rng(DEFAULT_SEED))  # tiny 는 index 0 → seed + 0
        x = inputs["x"]
        self.assertEqual((dims, out_name, list(inputs)), ({"n": 7}, "out", ["x"]))
        self.assertEqual((x.dtype, ref.dtype, x.shape, ref.shape), (np.dtype(np.int32), np.dtype(np.float64), (7,), (7,)))
        self.assertEqual(ref.tolist(), sorted(int(v) for v in x))
        self.assertTrue(all(0 <= int(v) <= INT32_MAX for v in x))
        repo_tiny = TASK_DIR / "tests" / "tiny.npz"
        if repo_tiny.exists():  # 저장소에 생성된 파일이 있으면 같은 내용이어야 한다
            with np.load(repo_tiny) as npz:
                np.testing.assert_array_equal(npz["x"], x)
                np.testing.assert_array_equal(npz["out_ref"], ref)
                self.assertEqual(int(npz["n"]), 7)

    def test_rect_values_span_the_range_and_reference_is_sorted(self) -> None:
        mod = load_gen_tests()
        _, inputs, _, ref = mod.make_case({"n": 100_003}, np.random.default_rng(DEFAULT_SEED + 1))
        x = inputs["x"]
        self.assertGreaterEqual(int(x.min()), 0)
        self.assertGreater(int(x.max()), INT32_MAX - (1 << 24))  # 상한 근처까지 쓴다(확률적으로 확실)
        self.assertTrue(bool(np.all(np.diff(ref) >= 0)))
        self.assertEqual(np.sort(x.astype(np.float64)).tolist(), ref.tolist())


class SpecTest(unittest.TestCase):
    def test_taskspec_and_tolerance(self) -> None:
        spec = load_taskspec(SIGNATURES)
        self.assertEqual((spec.task, spec.function, spec.params), (TASK, TASK, ["x", "out", "n"]))
        self.assertEqual((spec.output.name, spec.output.dtype, spec.output.shape), ("out", "int32", ("n",)))
        self.assertEqual([(a.name, a.dtype) for a in spec.inputs], [("x", "int32")])
        self.assertIsNone(spec.work)  # gflops NA (§17)
        self.assertEqual(spec.baseline["backend"], "python")
        self.assertEqual(spec.baseline["file"], "baseline/baseline.py")
        self.assertFalse(spec.baseline["openmp"])
        self.assertFalse(spec.baseline["gate_blas"])
        self.assertFalse(spec.baseline["gate_min_gflops"])
        self.assertEqual(spec.shape_of(spec.output, {"n": 12}), (12,))
        tol = load_tolerance(TOLERANCE)
        self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio, tol.sentinel), (0.0, 0.0, 0.0, -2048.0))
        self.assertEqual((tol.task, tol.dtype), (TASK, "int32"))

    def test_spec_md_states_the_contract_and_names_no_library(self) -> None:
        text = (TASK_DIR / "spec.md").read_text(encoding="utf-8")
        for needle in ("`sort`", "x, out, n", "0 <= x[i] <= 2^31 - 1", "7, 100,003 and 1,048,576", "10,000,000",
                       "out[i] == ref[i]", "There is no tolerance", "must not sort x in place", "3 warm-up calls followed by 10 timed calls"):
            self.assertIn(needle, text)
        for forbidden in ("NumPy", "numpy", "qsort", "OpenMP", "radix", "C "):  # 언어 중립·라이브러리·알고리즘 무언급
            self.assertNotIn(forbidden, text)
        ko = (TASK_DIR / "spec.ko.md").read_text(encoding="utf-8")
        self.assertIn("spec.md", ko.splitlines()[2])  # 번역 안내문

    def test_dummies_pass_extraction_with_exact_signature(self) -> None:
        sigs = load_signatures(TASK)
        for lang, backend in LANG_BACKENDS:
            text = (DUMMY_DIR / lang / "0.md").read_text(encoding="utf-8")
            res = extract(text, backend=backend, signature_spec=signature_spec_for(sigs, lang))
            with self.subTest(lang=lang):
                self.assertEqual(res.n_blocks, 1)
                self.assertTrue(res.ok, res.detail)
                self.assertTrue(res.signature_ok)
                self.assertTrue(res.signature_exact, res.detail)
                self.assertEqual(res.detail, "")

    def test_baseline_matches_numpy_dummy_contract(self) -> None:
        code = (TASK_DIR / "baseline" / "baseline.py").read_text(encoding="utf-8")
        self.assertIn("def sort(x, out, n):", code)
        self.assertIn("out[...] = np.sort(x)", code)


class WorkerBase(unittest.TestCase):
    """worker 를 격리 실행하는 공통 준비(임시 디렉터리, 테스트 npz). 저장소 안에는 쓰지 않는다."""

    tmp: Path
    tiny: Path
    rect: Path
    dups: Path
    one: Path
    extremes: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="sort_worker_")
        cls.tmp = Path(cls._tmpdir.name)
        mod = load_gen_tests()
        for index, (name, shape) in enumerate(mod.SHAPES[:2]):  # tiny, rect: gen_tests 와 같은 시드(seed + index)
            _, inputs, _, _ = mod.make_case(shape, np.random.default_rng(DEFAULT_SEED + index))
            write_case(cls.tmp / f"{name}.npz", inputs["x"])
        cls.tiny, cls.rect = cls.tmp / "tiny.npz", cls.tmp / "rect.npz"
        rng = np.random.default_rng(7)
        cls.dups = cls.tmp / "dups.npz"
        write_case(cls.dups, rng.integers(0, 10, size=1000, dtype=np.int32))  # 값 10가지 × 1000개: 중복 처리·안정성
        cls.one = cls.tmp / "one.npz"
        write_case(cls.one, np.array([INT32_MAX], dtype=np.int32))
        cls.extremes = cls.tmp / "extremes.npz"
        ext = np.where(np.arange(1001) % 2 == 0, INT32_MAX, 0).astype(np.int32)  # 상한·0 만, 홀수 길이
        write_case(cls.extremes, ext)
        cls._n = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def all_tests(self) -> list[Path]:
        return [self.tiny, self.rect, self.dups, self.one, self.extremes]

    def workdir(self) -> Path:
        type(self)._n += 1
        d = self.tmp / f"case_{type(self)._n}"
        d.mkdir()
        return d

    def write_py(self, code: str) -> Path:
        p = self.workdir() / "kernel.py"
        p.write_text(code, encoding="utf-8")
        return p

    def compile_c(self, code: str, *, openmp: bool) -> Path:
        res = build_c(code, self.workdir(), BCFG, openmp=openmp, launch_prefix=[], symbol=TASK)
        self.assertTrue(res.ok, f"빌드 실패: {res.error_kind} {res.error_detail}\n{res.log}")
        assert res.artifact is not None
        return res.artifact

    def check(self, backend: str, artifact: Path, tests: list[Path], *, env_extra: dict[str, str] | None = None,
              timeout_s: float = 180.0) -> dict:
        """worker --mode check 를 격리 실행하고 결과 문서를 돌려준다(rc 0 이어야 한다)."""
        wd = self.workdir()
        out = wd / "check.json"
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", "check",
               "--tests", ",".join(str(t) for t in tests), "--tolerance", str(TOLERANCE), "--signatures", str(SIGNATURES),
               "--out", str(out)]
        env = launch_env_minimal()
        env.update(env_extra or {})
        proc = run_isolated(cmd, timeout_s=timeout_s, cwd=wd, env=env)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["task"], TASK)
        return doc

    def assert_pass(self, doc: dict, tests: list[Path]) -> None:
        self.assertEqual(doc["classification"], "pass", json.dumps(doc.get("tests"), ensure_ascii=False)[:2000])
        self.assertEqual([t["name"] for t in doc["tests"]], [t.stem for t in tests])
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["nondeterministic"])
        self.assertEqual(doc["max_abs_err"], 0.0)


class DummyKernelsTest(WorkerBase):
    """세 언어의 정답 더미(k = 0)가 모든 테스트 입력에서 pass 한다."""

    def test_numpy_dummy_passes(self) -> None:
        tests = self.all_tests()
        doc = self.check("python", self.write_py(dummy_code("numpy")), tests)
        self.assert_pass(doc, tests)
        self.assertEqual(doc["tests"][0]["runs"][0]["return_type"], "NoneType")

    @unittest.skipUnless(HAVE_GCC, "gcc/nm 필요")
    def test_c_seq_dummy_passes(self) -> None:
        tests = self.all_tests()
        doc = self.check("c", self.compile_c(dummy_code("c_seq"), openmp=False), tests)
        self.assert_pass(doc, tests)

    @unittest.skipUnless(HAVE_GCC, "gcc/nm 필요")
    def test_c_openmp_dummy_passes_with_4_3_1_threads(self) -> None:
        tests = self.all_tests()
        artifact = self.compile_c(dummy_code("c_openmp"), openmp=True)
        for threads in ("4", "3", "1"):
            with self.subTest(threads=threads):
                doc = self.check("c", artifact, tests, env_extra={"OMP_NUM_THREADS": threads, "OMP_DYNAMIC": "false"})
                self.assert_pass(doc, tests)


class PlantedBugTest(WorkerBase):
    """심은 버그가 설계 §10 의 패턴으로 분류된다(tiny, n = 7)."""

    def test_descending_order_is_wrong_output_mismatch(self) -> None:
        doc = self.check("python", self.write_py(DESCENDING), [self.tiny])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))
        run = doc["tests"][0]["runs"][0]
        self.assertEqual((run["classification"], run["pattern"], run["n_unwritten"], run["nan_inf"]), ("wrong_output", "mismatch", 0, False))
        self.assertIsNone(run["max_tol_ratio"])  # 허용오차 0: 틀린 원소의 배율은 inf → JSON 에는 null 로 기록된다(2026-11-08)

    def test_unwritten_last_element_is_partial_write(self) -> None:
        doc = self.check("python", self.write_py(LAST_UNWRITTEN), [self.tiny])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "partial_write:1/7"))
        self.assertEqual(doc["tests"][0]["runs"][0]["n_unwritten"], 1)

    def test_sorting_the_input_in_place_is_inputs_mutated(self) -> None:
        doc = self.check("python", self.write_py(INPLACE_INPUT), [self.tiny])
        self.assertTrue(doc["inputs_mutated"])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "inputs_mutated"))


if __name__ == "__main__":
    unittest.main()
