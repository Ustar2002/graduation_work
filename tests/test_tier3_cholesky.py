"""Tier 3 과제 cholesky 테스트(docs/harness_design.md §18).

- 생성기(tasks/cholesky/gen_tests.py)의 결정성: 같은 시드로 두 번 만든 npz 가 바이트 단위로 같다. A 는 float32·정확히 대칭, 참조는 float64.
- tiny 참조가 L·Lᵀ ≈ A 를 만족하고 하삼각·양의 대각이며, 순수 Python 배정밀도 Cholesky–Banachiewicz 와 일치한다.
- 더미 정답 커널(numpy k=0; gcc 가 있으면 c_seq·c_openmp k=0)이 worker check(tiny·rect)를 통과한다.
- 심은 버그: 엄격 상삼각을 쓰지 않는 커널 → wrong_output/partial_write:{n(n−1)/2}/{n²}; Lᵀ(상삼각)를 쓰는 커널 → arg_order/transposed_output.
테스트 입력은 임시 디렉터리에 작은 형태로 직접 만든다(tasks/cholesky/tests 불필요). 워커는 격리 실행한다.
"""
from __future__ import annotations

import importlib.util
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from harness.build.c import build_c
from harness.classify import load_tolerance
from harness.config import BuildCfg
from harness.extract import extract
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from harness.testgen import TEST_NAMES, generate

TASK = ROOT / "tasks" / "cholesky"
DUMMY = ROOT / "harness" / "dummy_responses" / "cholesky"

# 단위 테스트용 형태: tiny·rect 는 실제 평가 크기, small·large 는 빠르게 줄인다(이름은 TEST_NAMES 순서여야 한다)
UNIT_SHAPES = [("tiny", {"n": 6}), ("rect", {"n": 131}), ("small", {"n": 40}), ("large", {"n": 64})]

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

# 심은 버그 1: 하삼각만 쓰고 엄격 상삼각은 건드리지 않는다(센티널이 남는다) → wrong_output / partial_write
PY_LOWER_ONLY = """import numpy as np

def cholesky(A, L, n):
    f = np.linalg.cholesky(A.astype(np.float64))
    idx = np.tril_indices(n)
    L[idx] = f[idx]
"""

# 심은 버그 2: Lᵀ(상삼각 인자)를 쓴다 — 전체를 쓰지만 전치돼 있다 → arg_order / transposed_output
PY_UPPER = """import numpy as np

def cholesky(A, L, n):
    L[...] = np.linalg.cholesky(A.astype(np.float64)).T
"""

C_UPPER = """#include <math.h>
void cholesky(const float* A, float* L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            double s = 0.0;
            for (int k = 0; k < j; k++) s += (double)L[k * n + i] * (double)L[k * n + j];
            double v = (double)A[i * n + j] - s;
            L[j * n + i] = (float)((i == j) ? sqrt(v) : v / (double)L[j * n + j]);   /* U = L^T 로 저장 */
        }
        for (int j = i + 1; j < n; j++) L[j * n + i] = 0.0f;
    }
}
"""


def _load_gen_tests():
    spec = importlib.util.spec_from_file_location("cholesky_gen_tests", TASK / "gen_tests.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_gen_tests()


def _signature_spec() -> dict:
    return yaml.safe_load((TASK / "signatures.yaml").read_text(encoding="utf-8"))


def _dummy_code(lang: str, backend: str) -> str:
    text = (DUMMY / lang / "0.md").read_text(encoding="utf-8")
    res = extract(text, backend=backend, signature_spec=_signature_spec())
    assert res.ok and res.code is not None, f"{lang}: {res.detail}"
    return res.code


def _plain_cholesky(a: list[list[float]]) -> list[list[float]]:
    """순수 Python 배정밀도 Cholesky–Banachiewicz(참조 대조용)."""
    n = len(a)
    low = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(low[i][k] * low[j][k] for k in range(j))
            low[i][j] = math.sqrt(a[i][i] - s) if i == j else (a[i][j] - s) / low[j][j]
    return low


class CholeskyContractTest(unittest.TestCase):
    def test_taskspec_and_tolerance(self):
        spec = load_taskspec(TASK / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("cholesky", "cholesky", ["A", "L", "n"]))
        dims = {"n": 131}
        self.assertEqual(spec.shape_of(spec.args[0], dims), (131, 131))
        self.assertEqual(spec.shape_of(spec.output, dims), (131, 131))
        self.assertEqual((spec.args[0].kind, spec.output.name, spec.output.dtype), ("in", "L", "float32"))
        self.assertEqual(spec.work_of(dims), float(int(131 ** 3 / 3)))
        self.assertEqual(spec.baseline["backend"], "python")
        self.assertFalse(spec.baseline["openmp"])
        self.assertTrue(spec.baseline["gate_blas"])
        self.assertFalse(spec.baseline["gate_min_gflops"])
        self.assertTrue((TASK / spec.baseline["file"]).is_file())
        raw = _signature_spec()
        self.assertEqual(raw["languages"]["numpy"]["positional"], "def cholesky(A, L, n):")
        self.assertEqual(raw["languages"]["numpy"]["keyword"], "def cholesky(*, A, L, n):")
        for lang in ("c_seq", "c_openmp"):
            self.assertEqual(raw["languages"][lang]["positional"], "void cholesky(const float* A, float* L, int n)")
        tol = load_tolerance(TASK / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.atol_rms_rel, tol.numeric_pattern_tol_ratio), (1e-3, 0.0, 1e-3, 20.0))
        self.assertEqual(tol.sentinel, -2048.0)
        # 평가 형태와 문구가 spec 과 일치한다
        self.assertEqual([n for n, _ in GEN.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in GEN.SHAPES], [6, 131, 512, 2048])
        spec_text = (TASK / "spec.md").read_text(encoding="utf-8")
        for needle in ("rtol = 1e-3 and atol = 1e-3 * RMS(ref)", "6, 131 and 512", "2,048", "8 GiB", "must be set to 0",
                       "exactly three parameters, in this order: A, L, n"):
            self.assertIn(needle, spec_text)
        self.assertTrue((TASK / "spec.ko.md").is_file())

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            m1 = generate("cholesky", 20261105, Path(t1), UNIT_SHAPES, GEN.make_case, "test")
            m2 = generate("cholesky", 20261105, Path(t2), UNIT_SHAPES, GEN.make_case, "test")
            self.assertEqual(m1["test_hash"], m2["test_hash"])
            for name in TEST_NAMES:
                self.assertEqual((Path(t1) / f"{name}.npz").read_bytes(), (Path(t2) / f"{name}.npz").read_bytes(), name)
            with np.load(Path(t1) / "rect.npz") as npz:
                a, ref = npz["A"], npz["L_ref"]
                self.assertEqual((a.shape, a.dtype), ((131, 131), np.float32))
                self.assertEqual((ref.shape, ref.dtype), ((131, 131), np.float64))
                self.assertEqual(int(npz["n"]), 131)
                self.assertTrue(np.array_equal(a, a.T))  # 비트 단위 대칭
                self.assertTrue(np.all(np.isfinite(a)))
                self.assertTrue(np.all(np.triu(ref, 1) == 0.0))  # 참조의 엄격 상삼각은 정확히 0
                self.assertTrue(np.all(np.diag(ref) > 0))
                # 참조는 float32 로 캐스트한 A 의 float64 Cholesky: L·Lᵀ 가 저장된 A 를 거의 정확히 복원한다
                np.testing.assert_allclose(ref @ ref.T, a.astype(np.float64), rtol=1e-12, atol=1e-10)
                eig = np.linalg.eigvalsh(a.astype(np.float64))
                self.assertGreater(eig[0], 0.9 * 131)  # 고윳값 ≥ n 근처(양정치·조건 좋음)
                self.assertLess(eig[-1] / eig[0], 4.0)

    def test_tiny_reference_matches_plain_python_cholesky(self):
        rng = np.random.default_rng(20261105)
        dims, inputs, out_name, ref = GEN.make_case({"n": 6}, rng)
        self.assertEqual((dims, out_name, ref.dtype, ref.shape), ({"n": 6}, "L", np.float64, (6, 6)))
        a = inputs["A"]
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.array_equal(a, a.T))
        # 하삼각·양의 대각
        self.assertTrue(np.all(np.triu(ref, 1) == 0.0))
        self.assertTrue(np.all(np.diag(ref) > 0))
        # L·Lᵀ ≈ A (float64 비교)
        np.testing.assert_allclose(ref @ ref.T, a.astype(np.float64), rtol=1e-13, atol=1e-12)
        # 순수 Python 2중 루프와 일치
        expect = _plain_cholesky([[float(v) for v in row] for row in a])
        np.testing.assert_allclose(ref, np.array(expect), rtol=1e-12, atol=1e-12)
        # n = 1: L[0][0] = sqrt(A[0][0])
        one = GEN.make_spd(1, np.random.default_rng(1))
        self.assertEqual(GEN.reference(one)[0, 0], math.sqrt(float(one[0, 0])))


class WorkerBase(unittest.TestCase):
    """임시 디렉터리에 작은 테스트 npz 를 만들고 워커를 격리 실행한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="cholesky_test_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.tests_dir = cls.tmp / "tests"
        generate("cholesky", 20261105, cls.tests_dir, UNIT_SHAPES, GEN.make_case, "unit test")
        cls._n = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def workdir(self) -> Path:
        type(self)._n += 1
        d = self.tmp / f"case_{type(self)._n}"
        d.mkdir()
        return d

    def check(self, backend: str, artifact: Path, names: tuple[str, ...], *, openmp: bool = False) -> dict:
        out = artifact.parent / f"check_{'_'.join(names)}.json"
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", "check",
               "--tests", ",".join(str(self.tests_dir / f"{n}.npz") for n in names), "--tolerance", str(TASK / "tolerance.yaml"),
               "--signatures", str(TASK / "signatures.yaml"), "--out", str(out)]
        env = dict(launch_env_minimal())
        if openmp:
            env.update({"OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "true", "OMP_PLACES": "cores", "OMP_DYNAMIC": "false"})
        proc = run_isolated(cmd, timeout_s=180, cwd=artifact.parent, env=env)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["task"], "cholesky")
        return doc


class PythonKernelTest(WorkerBase):
    def test_numpy_dummy_passes_tiny_rect(self):
        k = self.workdir() / "kernel.py"
        k.write_text(_dummy_code("numpy", "python"), encoding="utf-8")
        doc = self.check("python", k, ("tiny", "rect"))
        self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect"])
        for t in doc["tests"]:
            self.assertEqual(t["classification"], "pass")
            self.assertLessEqual(t["max_tol_ratio"], 0.01, t)  # float32 LAPACK 분해: 반올림 수준의 오차만 남는다
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])

    def test_unwritten_upper_triangle_is_partial_write_on_rect(self):
        k = self.workdir() / "kernel.py"
        k.write_text(PY_LOWER_ONLY, encoding="utf-8")
        doc = self.check("python", k, ("rect",))
        n = 131
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", f"partial_write:{n * (n - 1) // 2}/{n * n}"))
        self.assertEqual(doc["tests"][0]["runs"][0]["n_unwritten"], n * (n - 1) // 2)

    def test_upper_factor_is_arg_order_transposed_on_rect(self):
        k = self.workdir() / "kernel.py"
        k.write_text(PY_UPPER, encoding="utf-8")
        doc = self.check("python", k, ("rect",))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("arg_order", "transposed_output"))
        self.assertEqual(doc["tests"][0]["pattern"], "transposed_output")


@unittest.skipIf(shutil.which("gcc") is None or shutil.which("nm") is None, "gcc/nm 필요")
class CKernelTest(WorkerBase):
    def build(self, code: str, *, openmp: bool) -> Path:
        res = build_c(code, self.workdir(), BCFG, openmp=openmp, launch_prefix=[], symbol="cholesky")
        self.assertTrue(res.ok, f"빌드 실패: {res.error_kind} {res.error_detail}\n{res.log}")
        self.assertEqual(res.forbidden_symbols, [])
        assert res.artifact is not None
        return res.artifact

    def test_c_dummies_compile_and_pass_tiny_rect(self):
        for lang, openmp in (("c_seq", False), ("c_openmp", True)):
            with self.subTest(lang=lang):
                text = (DUMMY / lang / "0.md").read_text(encoding="utf-8")
                res = extract(text, backend="c", signature_spec=_signature_spec())
                self.assertTrue(res.ok and res.signature_ok, res.detail)
                assert res.code is not None
                so = self.build(res.code, openmp=openmp)
                doc = self.check("c", so, ("tiny", "rect"), openmp=openmp)
                self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))
                for t in doc["tests"]:
                    self.assertLessEqual(t["max_tol_ratio"], 0.2, t)  # 직진 float32 커널의 계약(§18)

    def test_c_upper_factor_is_arg_order_transposed_on_rect(self):
        so = self.build(C_UPPER, openmp=False)
        doc = self.check("c", so, ("rect",))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("arg_order", "transposed_output"))


if __name__ == "__main__":
    unittest.main()
