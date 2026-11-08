"""Tier 2 과제 nbody 테스트(docs/harness_design.md §17).

- 생성기(tasks/nbody/gen_tests.py)의 결정성: 같은 시드로 두 번 만든 npz 가 바이트 단위로 같다.
- tiny 참조를 순수 Python 2중 루프(배정밀도)와 대조한다.
- 더미 정답 커널(numpy k=0; gcc 가 있으면 c_seq·c_openmp k=0)이 worker check(tiny·rect)를 통과한다.
- SoA 로 쓴 출력(acc[c*n + i])은 rect 에서 arg_order/transposed_output 으로 분류된다.
테스트 입력은 임시 디렉터리에 작은 형태로 직접 만든다(tasks/nbody/tests 불필요). 워커는 격리 실행한다.
"""
from __future__ import annotations

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

TASK = ROOT / "tasks" / "nbody"
DUMMY = ROOT / "harness" / "dummy_responses" / "nbody"
EPS2 = 1e-4

# 단위 테스트용 형태: tiny·rect 는 실제 평가 크기, small·large 는 빠르게 줄인다(이름은 TEST_NAMES 순서여야 한다)
UNIT_SHAPES = [("tiny", {"n": 5}), ("rect", {"n": 301}), ("small", {"n": 64}), ("large", {"n": 96})]

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

# 심은 버그: 결과는 맞지만 SoA(acc[c*n + i])로 저장 → arg_order/transposed_output
PY_SOA = """import numpy as np

def nbody(pos, mass, acc, n):
    p = pos.astype(np.float64)
    m = mass.astype(np.float64)
    d = p[None, :, :] - p[:, None, :]
    r2 = np.einsum("ijk,ijk->ij", d, d) + 1e-4
    w = m / (r2 * np.sqrt(r2))
    a = np.einsum("ij,ijk->ik", w, d)
    acc.reshape(3, n)[...] = a.T
"""

C_SOA = """#include <math.h>
void nbody(const float* pos, const float* mass, float* acc, int n) {
    for (int i = 0; i < n; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        for (int j = 0; j < n; j++) {
            float dx = pos[3*j] - pos[3*i], dy = pos[3*j+1] - pos[3*i+1], dz = pos[3*j+2] - pos[3*i+2];
            float r2 = dx*dx + dy*dy + dz*dz + 1e-4f;
            float inv = 1.0f / sqrtf(r2);
            float w = mass[j] * inv * inv * inv;
            ax += w * dx; ay += w * dy; az += w * dz;
        }
        acc[i] = ax; acc[n + i] = ay; acc[2*n + i] = az;
    }
}
"""


def _load_gen_tests():
    spec = importlib.util.spec_from_file_location("nbody_gen_tests", TASK / "gen_tests.py")
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


class NbodyContractTest(unittest.TestCase):
    def test_taskspec_and_tolerance(self):
        spec = load_taskspec(TASK / "signatures.yaml")
        self.assertEqual((spec.task, spec.function, spec.params), ("nbody", "nbody", ["pos", "mass", "acc", "n"]))
        dims = {"n": 301}
        self.assertEqual(spec.shape_of(spec.args[0], dims), (301, 3))
        self.assertEqual(spec.shape_of(spec.args[1], dims), (301,))
        self.assertEqual(spec.shape_of(spec.output, dims), (301, 3))
        self.assertEqual(spec.output.dtype, "float32")
        self.assertEqual(spec.work_of(dims), 20 * 301 * 301)
        self.assertEqual(spec.baseline["backend"], "c")
        self.assertTrue(spec.baseline["openmp"])
        tol = load_tolerance(TASK / "tolerance.yaml")
        self.assertEqual((tol.rtol, tol.atol, tol.atol_rms_rel, tol.numeric_pattern_tol_ratio), (1e-3, 0.0, 1e-3, 20.0))
        self.assertEqual(tol.sentinel, -1e6)
        self.assertTrue(np.isfinite(np.float32(tol.sentinel)))  # float32 로 표현 가능해야 한다
        # 평가 형태와 상수가 spec 과 일치한다
        self.assertEqual([n for n, _ in GEN.SHAPES], list(TEST_NAMES))
        self.assertEqual([s["n"] for _, s in GEN.SHAPES], [5, 301, 2048, 16384])
        self.assertEqual(GEN.EPS2, EPS2)
        spec_text = (TASK / "spec.md").read_text(encoding="utf-8")
        for needle in ("EPS2 = 1e-4", "rtol = 1e-3 and atol = 1e-3 * RMS(ref)", "5, 301 and 2,048", "16,384"):
            self.assertIn(needle, spec_text)
        self.assertTrue((TASK / "spec.ko.md").is_file())

    def test_generation_is_deterministic(self):
        with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
            m1 = generate("nbody", 20261105, Path(t1), UNIT_SHAPES, GEN.make_case, "test")
            m2 = generate("nbody", 20261105, Path(t2), UNIT_SHAPES, GEN.make_case, "test")
            self.assertEqual(m1["test_hash"], m2["test_hash"])
            for name in TEST_NAMES:
                self.assertEqual((Path(t1) / f"{name}.npz").read_bytes(), (Path(t2) / f"{name}.npz").read_bytes(), name)
            with np.load(Path(t1) / "rect.npz") as npz:
                self.assertEqual(npz["pos"].shape, (301, 3))
                self.assertEqual(npz["pos"].dtype, np.float32)
                self.assertEqual(npz["mass"].shape, (301,))
                self.assertEqual(npz["acc_ref"].shape, (301, 3))
                self.assertEqual(npz["acc_ref"].dtype, np.float64)
                self.assertEqual(int(npz["n"]), 301)
                self.assertTrue(np.all((npz["pos"] >= 0) & (npz["pos"] < 1)))
                self.assertTrue(np.all((npz["mass"] >= 0.5) & (npz["mass"] < 1.5)))

    def test_tiny_reference_matches_plain_python_double_loop(self):
        rng = np.random.default_rng(20261105)
        dims, inputs, out_name, ref = GEN.make_case({"n": 5}, rng)
        self.assertEqual((dims, out_name, ref.dtype, ref.shape), ({"n": 5}, "acc", np.float64, (5, 3)))
        pos = [[float(v) for v in row] for row in inputs["pos"]]
        mass = [float(v) for v in inputs["mass"]]
        n = 5
        expect = [[0.0, 0.0, 0.0] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue  # 자기 항은 정확히 0 — 참조는 포함하므로 둘이 같아야 한다
                d = [pos[j][c] - pos[i][c] for c in range(3)]
                r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + EPS2
                w = mass[j] / (r2 * r2 ** 0.5)
                for c in range(3):
                    expect[i][c] += w * d[c]
        np.testing.assert_allclose(ref, np.array(expect), rtol=1e-12, atol=1e-9)
        # 참조의 j = i 항은 d = 0 이라 정확히 0: 한 물체만 있으면 가속도는 0
        self.assertTrue(np.all(GEN.reference(inputs["pos"][:1], inputs["mass"][:1]) == 0.0))


class WorkerBase(unittest.TestCase):
    """임시 디렉터리에 작은 테스트 npz 를 만들고 워커를 격리 실행한다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="nbody_test_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.tests_dir = cls.tmp / "tests"
        generate("nbody", 20261105, cls.tests_dir, UNIT_SHAPES, GEN.make_case, "unit test")
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
        self.assertEqual(doc["task"], "nbody")
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
            self.assertLessEqual(t["max_tol_ratio"], 0.01, t)  # float64 계산 뒤 float32 반올림만 남는다
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])

    def test_soa_output_is_arg_order_transposed_on_rect(self):
        k = self.workdir() / "kernel.py"
        k.write_text(PY_SOA, encoding="utf-8")
        doc = self.check("python", k, ("rect",))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("arg_order", "transposed_output"))
        self.assertEqual(doc["tests"][0]["pattern"], "transposed_output")


@unittest.skipIf(shutil.which("gcc") is None or shutil.which("nm") is None, "gcc/nm 필요")
class CKernelTest(WorkerBase):
    def build(self, code: str, *, openmp: bool) -> Path:
        res = build_c(code, self.workdir(), BCFG, openmp=openmp, launch_prefix=[], symbol="nbody")
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
                    self.assertLessEqual(t["max_tol_ratio"], 0.2, t)  # 직진 float32 커널의 계약(§17)

    def test_c_soa_output_is_arg_order_transposed_on_rect(self):
        so = self.build(C_SOA, openmp=False)
        doc = self.check("c", so, ("rect",))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("arg_order", "transposed_output"))


if __name__ == "__main__":
    unittest.main()
