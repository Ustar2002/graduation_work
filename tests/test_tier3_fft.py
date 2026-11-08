"""Tier 3 과제 fft 테스트(docs/harness_design.md §18).

- 생성기(tasks/fft/gen_tests.py)의 결정성: 같은 시드로 두 번 만든 npz 가 바이트 단위로 같다.
- tiny 참조를 순수 Python O(n²) DFT(배정밀도 cmath)와 대조한다.
- 더미 정답 커널(numpy k=0; gcc 가 있으면 c_seq·c_openmp k=0)이 worker check(tiny·rect)를 통과한다.
- 지수 부호가 반대인 커널(스케일 없는 역변환)은 wrong_output 으로 분류된다.
- 실수부만 쓰고 허수부를 센티널로 남긴 커널은 wrong_output/partial_write 로 분류된다.
테스트 입력은 임시 디렉터리에 작은 형태로 직접 만든다(tasks/fft/tests 불필요). 워커는 격리 실행한다.
"""
from __future__ import annotations

import cmath
import importlib.util
import json
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

TASK = ROOT / "tasks" / "fft"
DUMMY = ROOT / "harness" / "dummy_responses" / "fft"

# 단위 테스트용 형태: tiny·rect 는 실제 평가 크기, small·large 는 빠르게 줄인다(이름은 TEST_NAMES 순서, n 은 2 의 거듭제곱)
UNIT_SHAPES = [("tiny", {"n": 8}), ("rect", {"n": 1024}), ("small", {"n": 64}), ("large", {"n": 128})]

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

# 심은 버그 1: 지수 부호가 반대(스케일 없는 역변환) → wrong_output (허수부가 켤레로 뒤집힌다)
PY_CONJ = """import numpy as np

def fft(x, out, n):
    z = x.reshape(-1).view(np.complex64)
    out.reshape(-1).view(np.complex64)[...] = np.fft.ifft(z) * n
"""

# 심은 버그 2: 실수부만 기록, 허수부는 손대지 않음 → wrong_output/partial_write:n/2n
PY_REAL_ONLY = """import numpy as np

def fft(x, out, n):
    z = x.reshape(-1).view(np.complex64)
    out[:, 0] = np.fft.fft(z).real
"""

# 심은 버그 1 의 C 판: 트위들 각도의 부호만 양수(e^{+2πi jk/n})
C_CONJ = """#include <math.h>
#include <stdlib.h>
void fft(const float* x, float* out, int n) {
    int logn = 0;
    while ((1 << logn) < n) logn++;
    for (int i = 0; i < n; i++) {
        unsigned r = 0, v = (unsigned)i;
        for (int b = 0; b < logn; b++) { r = (r << 1) | (v & 1u); v >>= 1; }
        out[2 * r] = x[2 * i]; out[2 * r + 1] = x[2 * i + 1];
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2;
        double ang = 2.0 * M_PI / (double)len;
        for (int start = 0; start < n; start += len) {
            for (int j = 0; j < half; j++) {
                double wr = cos(ang * j), wi = sin(ang * j);
                int a = 2 * (start + j), b = a + 2 * half;
                double tr = wr * out[b] - wi * out[b + 1], ti = wr * out[b + 1] + wi * out[b];
                double ur = out[a], ui = out[a + 1];
                out[a] = (float)(ur + tr); out[a + 1] = (float)(ui + ti);
                out[b] = (float)(ur - tr); out[b + 1] = (float)(ui - ti);
            }
        }
    }
}
"""


def _load_gen_tests():
    spec = importlib.util.spec_from_file_location("fft_gen_tests", TASK / "gen_tests.py")
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


def _naive_dft(x: np.ndarray) -> list[complex]:
    """순수 Python O(n²) DFT(배정밀도). ``x`` 는 (n, 2) 교차 저장."""
    n = x.shape[0]
    z = [complex(float(x[j, 0]), float(x[j, 1])) for j in range(n)]
    return [sum(z[j] * cmath.exp(-2j * cmath.pi * j * k / n) for j in range(n)) for k in range(n)]


class FftContractTest(unittest.TestCase):
    def test_taskspec_and_tolerance(self):
        spec = load_taskspec(TASK / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("fft", "fft", ["x", "out", "n"]))
        dims = {"n": 1024}
        self.assertEqual(spec.shape_of(spec.args[0], dims), (1024, 2))
        self.assertEqual(spec.shape_of(spec.output, dims), (1024, 2))
        self.assertEqual(spec.output.dtype, "float32")
        self.assertEqual(spec.work_of(dims), 5 * 1024 * 10)  # 5·n·log2 n
        self.assertEqual(spec.baseline["backend"], "python")
        self.assertFalse(spec.baseline["openmp"])
        self.assertFalse(spec.baseline["gate_blas"])
        self.assertTrue((TASK / spec.baseline["file"]).is_file())
        tol = load_tolerance(TASK / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.atol_rms_rel, tol.numeric_pattern_tol_ratio), (1e-3, 0.0, 1e-3, 20.0))
        self.assertEqual(tol.sentinel, -1e9)
        self.assertTrue(np.isfinite(np.float32(tol.sentinel)))  # float32 로 표현 가능해야 한다
        # 평가 형태가 spec 과 일치하고 전부 2 의 거듭제곱이다
        self.assertEqual([n for n, _ in GEN.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in GEN.SHAPES], [8, 1024, 1 << 16, 1 << 22])
        for _, s in GEN.SHAPES + UNIT_SHAPES:
            self.assertEqual(s["n"] & (s["n"] - 1), 0, s)
        spec_text = (TASK / "spec.md").read_text(encoding="utf-8")
        for needle in ("exp(-2 * pi * i * j * k / n)", "No scaling is applied", "rtol = 1e-3 and atol = 1e-3 * RMS(ref)",
                       "8, 1,024 and 65,536", "4,194,304", "8 GiB", "exactly three parameters", "2*j + 1"):
            self.assertIn(needle, spec_text)

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            m1 = generate("fft", 20261105, Path(t1), UNIT_SHAPES, GEN.make_case, "test")
            m2 = generate("fft", 20261105, Path(t2), UNIT_SHAPES, GEN.make_case, "test")
            self.assertEqual(m1["test_hash"], m2["test_hash"])
            for name in TEST_NAMES:
                self.assertEqual((Path(t1) / f"{name}.npz").read_bytes(), (Path(t2) / f"{name}.npz").read_bytes(), name)
            with np.load(Path(t1) / "rect.npz") as npz:
                self.assertEqual(npz["x"].shape, (1024, 2))
                self.assertEqual(npz["x"].dtype, np.float32)
                self.assertEqual(npz["out_ref"].shape, (1024, 2))
                self.assertEqual(npz["out_ref"].dtype, np.float64)
                self.assertEqual(int(npz["n"]), 1024)
                self.assertTrue(np.all((npz["x"] >= -1) & (npz["x"] < 1)))
                self.assertTrue(np.any(npz["x"] < 0) and np.any(npz["x"] > 0))
                # X_0 는 입력의 합(배정밀도): 참조의 첫 원소로 확인
                x64 = npz["x"].astype(np.float64)
                np.testing.assert_allclose(npz["out_ref"][0], x64.sum(axis=0), rtol=1e-12, atol=1e-9)

    def test_rejects_non_power_of_two(self):
        with self.assertRaises(ValueError):
            GEN.make_case({"n": 12}, np.random.default_rng(0))

    def test_tiny_reference_matches_naive_dft(self):
        rng = np.random.default_rng(20261105)
        dims, inputs, out_name, ref = GEN.make_case({"n": 8}, rng)
        self.assertEqual((dims, out_name, ref.dtype, ref.shape), ({"n": 8}, "out", np.float64, (8, 2)))
        expect = _naive_dft(inputs["x"])
        np.testing.assert_allclose(ref[:, 0], [c.real for c in expect], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ref[:, 1], [c.imag for c in expect], rtol=1e-12, atol=1e-12)
        # 부호 규약: 켤레(역변환 × n)와는 허수부가 다르다(입력이 실수 전용이 아니므로 실수부도 다르다)
        conj = np.conj(np.fft.fft(np.conj(inputs["x"][:, 0].astype(np.float64) + 1j * inputs["x"][:, 1].astype(np.float64))))
        self.assertFalse(np.allclose(ref[:, 1], conj.imag, rtol=1e-3, atol=1e-3))


class WorkerBase(unittest.TestCase):
    """임시 디렉터리에 작은 테스트 npz 를 만들고 워커를 격리 실행한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="fft_test_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.tests_dir = cls.tmp / "tests"
        generate("fft", 20261105, cls.tests_dir, UNIT_SHAPES, GEN.make_case, "unit test")
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
        self.assertEqual(doc["task"], "fft")
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
            self.assertLessEqual(t["max_tol_ratio"], 0.01, t)  # complex64 FFT 의 오차는 허용오차의 1/10⁴ 수준
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        # atol_rms_rel 이 적용됐다: 형태마다 atol = 1e-3 × RMS(참조 성분)
        for t in doc["tests"]:
            with np.load(self.tests_dir / f"{t['name']}.npz") as npz:
                rms = float(np.sqrt(np.mean(npz["out_ref"] ** 2)))
            self.assertAlmostEqual(t["atol_effective"], 1e-3 * rms, places=12)

    def test_baseline_passes_tiny_rect(self):
        k = self.workdir() / "kernel.py"
        shutil.copy(TASK / "baseline" / "baseline.py", k)
        doc = self.check("python", k, ("tiny", "rect"))
        self.assertEqual(doc["classification"], "pass", doc.get("error_detail"))

    def test_opposite_sign_convention_is_wrong_output(self):
        k = self.workdir() / "kernel.py"
        k.write_text(PY_CONJ, encoding="utf-8")
        doc = self.check("python", k, ("rect",))
        self.assertEqual(doc["classification"], "wrong_output", doc)
        self.assertEqual(doc["error_detail"], "mismatch")
        self.assertEqual(doc["tests"][0]["runs"][0]["n_unwritten"], 0)  # 전부 썼지만 값이 틀리다

    def test_real_part_only_is_partial_write(self):
        k = self.workdir() / "kernel.py"
        k.write_text(PY_REAL_ONLY, encoding="utf-8")
        doc = self.check("python", k, ("rect",))
        self.assertEqual(doc["classification"], "wrong_output", doc)
        self.assertEqual(doc["error_detail"], "partial_write:1024/2048")
        self.assertEqual(doc["tests"][0]["runs"][0]["n_unwritten"], 1024)


@unittest.skipIf(shutil.which("gcc") is None or shutil.which("nm") is None, "gcc/nm 필요")
class CKernelTest(WorkerBase):
    def build(self, code: str, *, openmp: bool) -> Path:
        res = build_c(code, self.workdir(), BCFG, openmp=openmp, launch_prefix=[], symbol="fft")
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
                    self.assertLessEqual(t["max_tol_ratio"], 0.01, t)  # double 트위들 radix-2: 실측 ≤ 2e-4 (README)
                self.assertFalse(doc["inputs_mutated"])

    def test_c_opposite_sign_convention_is_wrong_output(self):
        so = self.build(C_CONJ, openmp=False)
        doc = self.check("c", so, ("rect",))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))


if __name__ == "__main__":
    unittest.main()
