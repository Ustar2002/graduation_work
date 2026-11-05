"""worker 테스트: 임시 디렉터리에 작은 C 커널(정상/전치/OOB/크래시)과 python 커널을 만들어
``harness.worker`` 를 ``run_isolated`` 로 실행하고 분류·결과 JSON 키·종료 코드를 확인한다.

gcc 가 없으면 C 테스트는 건너뛴다. 테스트 입력은 여기서 작은 형태로 직접 만든다(tasks/gemm/tests 불필요).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.build.c import build_c
from harness.config import BuildCfg
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import ProcResult, run_isolated
from harness.protocol import launch_env_minimal

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

TOLERANCE_YAML = """task: gemm
dtype: float32
rtol: 1.0e-3
atol: 1.0e-3
reference_dtype: float64
numeric_pattern_tol_ratio: 20
sentinel: -2048.0
"""

C_CORRECT = """
void gemm(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}
"""

C_TRANSPOSED = """
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[j * M + i] = sum;
        }
}
"""

C_OOB = C_CORRECT.replace("void gemm(", "static void gemm_ok(").replace("restrict", "") + """
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    gemm_ok(A, B, C, M, N, K);
    C[M * N] = 1.0f;   /* 가드 영역 한 칸 */
}
"""

C_CRASH = """
#include <stddef.h>
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    volatile float* far = C + ((size_t)1 << 40);  /* 매핑되지 않은 주소 → SIGSEGV */
    *far = 1.0f;
}
"""

C_NO_GEMM = """
void gemm2(const float* A, const float* B, float* C, int M, int N, int K) { (void)A; (void)B; (void)C; }
"""

# M/N 교환: C[i*M + j]. M≠N 이면 view 안에 센티널이 남고 가드에도 쓴다(M>N: 충돌 없음) → §10 v2.1 규칙 3 swapped_MN_dims.
# 정사각(small)에서는 C[i*N + j] 와 같아 pass 다.
C_SWAPPED_MN = """
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) sum += A[i * K + k] * B[k * N + j];
            C[i * M + j] = sum;
        }
}
"""

C_OPENMP = """
void gemm(const float* restrict A, const float* restrict B, float* restrict C, int M, int N, int K) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) C[i * N + j] = 0.0f;
        for (int k = 0; k < K; k++) {
            float a = A[i * K + k];
            for (int j = 0; j < N; j++) C[i * N + j] += a * B[k * N + j];
        }
    }
}
"""

PY_CORRECT = """
import numpy as np
def gemm(A, B, C, M, N, K):
    print("hello from kernel")
    np.matmul(A, B, out=C)
"""

PY_RETURN_ONLY = """
def gemm(A, B, C, M, N, K):
    return A @ B
"""

PY_RAISES = """
def gemm(A, B, C, M, N, K):
    raise RuntimeError("boom")
"""

# __str__ 자체가 예외를 던지는 예외 객체: BaseException 핸들러의 str(e) 가 보호되지 않으면 핸들러가 죽어
# worker_error(rc=3) → run.py 가 harness_error 로 run 전체를 중단한다.
PY_RAISES_UNPRINTABLE = """
class Weird(Exception):
    def __str__(self):
        raise ValueError("no str")
def gemm(A, B, C, M, N, K):
    raise Weird()
"""

PY_IMPORT_RAISES_UNPRINTABLE = """
class Weird(Exception):
    def __str__(self):
        raise ValueError("no str")
raise Weird()
"""

PY_ACCUMULATE = """
import numpy as np
def gemm(A, B, C, M, N, K):
    C += A @ B
"""

PY_NONDET = """
import numpy as np
_calls = [0]
def gemm(A, B, C, M, N, K):
    np.matmul(A, B, out=C)
    if _calls[0] % 2 == 1:
        C[0, 0] += 1.0
    _calls[0] += 1
"""

PY_MUTATES = """
import numpy as np
def gemm(A, B, C, M, N, K):
    np.matmul(A, B, out=C)
    A[0, 0] += 1.0
"""

PY_SLEEP = """
import time, numpy as np
def gemm(A, B, C, M, N, K):
    time.sleep(SLEEP_S)
    np.matmul(A, B, out=C)
"""

PY_IMPORT_FAILS = """
import this_module_does_not_exist_xyz
def gemm(A, B, C, M, N, K):
    pass
"""

PY_NO_GEMM = """
def matmul(A, B, C, M, N, K):
    pass
"""

PY_ASYNC = """
async def gemm(A, B, C, M, N, K):
    pass
"""

# OpenMP 커널 로드 전후의 주 스레드 친화도(test_bench_pinned_cpus_survive_openmp_proc_bind 의 전제 확인용)
AFFINITY_PROBE = """
import ctypes, json, os, sys
before = sorted(os.sched_getaffinity(0))
ctypes.CDLL(sys.argv[1])
after = sorted(os.sched_getaffinity(0))
print(json.dumps({"before": before, "after": after}))
"""

SYS_CPU = Path("/sys/devices/system/cpu")


def _one_cpu_per_core(cpus: list[int]) -> list[int]:
    """물리 코어마다 CPU 하나(가장 작은 번호)를 고른다(``topology/thread_siblings_list`` 로 묶음; 못 읽으면 CPU 자체를 코어로 본다).

    같은 코어의 SMT 형제 쌍(이 머신의 0,1)만 고정하면 ``OMP_PLACES=cores`` 아래 place 가 하나뿐이라 libgomp 가 주 스레드를
    place 0 에 묶어도 마스크가 {0,1} 그대로여서 '로드 뒤 친화도 축소' 회귀를 관측할 수 없다(실측: taskset 0,1 → 변화 없음,
    0,2 → {0}). 서로 다른 코어의 CPU 를 골라야 단언이 비지 않는다.
    """
    seen: set[str] = set()
    out: list[int] = []
    for cpu in cpus:
        try:
            key = (SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list").read_text(encoding="utf-8").strip()
        except OSError:
            key = str(cpu)
        if key not in seen:
            seen.add(key)
            out.append(cpu)
    return out


def _core_pattern_is_pipe() -> bool:
    """``kernel.core_pattern`` 이 파이프(apport 등)면 덤프는 cwd 가 아니라 핸들러로 가므로 ``core*`` 파일 검사는 공허하다."""
    try:
        return Path("/proc/sys/kernel/core_pattern").read_text(encoding="utf-8").strip().startswith("|")
    except OSError:
        return False


def _make_npz(path: Path, M: int, N: int, K: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    A = (rng.random((M, K), dtype=np.float32) * np.float32(2) - np.float32(1))
    B = (rng.random((K, N), dtype=np.float32) * np.float32(2) - np.float32(1))
    C_ref = A.astype(np.float64) @ B.astype(np.float64)
    with open(path, "wb") as fh:
        np.savez(fh, A=A, B=B, C_ref=C_ref, M=np.int64(M), N=np.int64(N), K=np.int64(K), seed=np.int64(seed))


class WorkerTestBase(unittest.TestCase):
    """공통 준비: 임시 디렉터리, 테스트 npz, tolerance.yaml, 워커 실행 헬퍼."""

    tmp: Path
    tests: list[Path]
    large: Path
    tolerance: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="worker_test_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.tests = []
        for name, (M, N, K), seed in (("tiny", (7, 5, 3), 1), ("rect", (20, 13, 31), 2), ("small", (16, 16, 16), 3)):
            p = cls.tmp / f"{name}.npz"
            _make_npz(p, M, N, K, seed)
            cls.tests.append(p)
        cls.large = cls.tmp / "large.npz"
        _make_npz(cls.large, 48, 48, 48, 4)
        cls.tolerance = cls.tmp / "tolerance.yaml"
        cls.tolerance.write_text(TOLERANCE_YAML, encoding="utf-8")
        cls._n = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def workdir(self) -> Path:
        type(self)._n += 1
        d = self.tmp / f"case_{type(self)._n}"
        d.mkdir()
        return d

    def write_py(self, code: str, name: str = "kernel.py") -> Path:
        d = self.workdir()
        p = d / name
        p.write_text(code, encoding="utf-8")
        return p

    def build_c_kernel(self, code: str) -> Path:
        d = self.workdir()
        res = build_c(code, d, BCFG, openmp=False, launch_prefix=[])
        self.assertTrue(res.ok, f"빌드 실패: {res.error_kind} {res.error_detail}\n{res.log}")
        assert res.artifact is not None
        return res.artifact

    def run_worker(self, backend: str, artifact: Path, mode: str, tests: list[Path] | None = None, *,
                   extra: list[str] | None = None, timeout_s: float = 120.0,
                   workdir: Path | None = None) -> tuple[ProcResult, dict | None]:
        """워커를 격리 실행한다. 결과 JSON·cwd 는 ``workdir`` (기본: 임시 커널의 디렉터리; 저장소 안 파일이면 반드시 지정)."""
        wd = workdir if workdir is not None else artifact.parent
        assert self.tmp in wd.parents or wd == self.tmp, "테스트는 임시 디렉터리 밖에 쓰지 않는다"
        out = wd / f"{mode}.json"
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", mode,
               "--tolerance", str(self.tolerance), "--out", str(out)]
        if tests:
            cmd += ["--tests", ",".join(str(t) for t in tests)]
        cmd += extra or []
        proc = run_isolated(cmd, timeout_s=timeout_s, cwd=wd, env=launch_env_minimal())
        doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return proc, doc


@unittest.skipIf(shutil.which("gcc") is None or shutil.which("nm") is None, "gcc/nm 필요")
class CKernelTest(WorkerTestBase):
    def test_correct_kernel_passes_check(self) -> None:
        so = self.build_c_kernel(C_CORRECT)
        proc, doc = self.run_worker("c", so, "check", self.tests, extra=["--check-repeats", "2"])
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        self.assertEqual(proc.stdout_tail, "", "워커는 stdout 에 아무것도 쓰지 않는다")
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertIsNone(doc["in_progress"])
        self.assertTrue(doc["load_ok"])
        self.assertEqual(doc["classification"], "pass")
        self.assertEqual(doc["error_detail"], "")
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["nondeterministic"])
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect", "small"])
        for t in doc["tests"]:
            self.assertEqual(t["classification"], "pass")
            self.assertEqual(len(t["runs"]), 2)
            for r in t["runs"]:
                self.assertEqual(r["classification"], "pass")
                self.assertIsNone(r["return_type"])  # c 백엔드
                self.assertEqual(r["n_unwritten"], 0)
                self.assertGreaterEqual(r["elapsed_ms"], 0.0)
        self.assertIn("rlimit_as_bytes", doc["limits"])
        self.assertEqual(doc["limits"]["dumpable"], 0)
        self.assertIn("name", doc["blas"])
        self.assertLess(doc["max_tol_ratio"], 1.0)

    def test_transposed_store_is_arg_order(self) -> None:
        so = self.build_c_kernel(C_TRANSPOSED)
        proc, doc = self.run_worker("c", so, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "arg_order")
        self.assertEqual(doc["error_detail"], "transposed_output")
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("arg_order", "transposed_output"), t["name"])
        self.assertFalse(doc["oob_write"])

    def test_out_of_bounds_write_is_detected(self) -> None:
        so = self.build_c_kernel(C_OOB)
        proc, doc = self.run_worker("c", so, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["oob_write"])
        self.assertEqual(doc["classification"], "wrong_output")
        self.assertEqual(doc["error_detail"], "out_of_bounds_write")
        # 출력 자체는 맞으므로 classify_output 은 pass(§10 v2.1 규칙 2)지만 워커가 호출 수준에서 OOB 플래그를 반영한다
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("wrong_output", "out_of_bounds_write"), t["name"])
            for r in t["runs"]:
                self.assertTrue(r["oob_write"])
                self.assertEqual(r["n_unwritten"], 0)

    def test_out_of_bounds_write_in_bench_keeps_error_detail(self) -> None:
        so = self.build_c_kernel(C_OOB)
        proc, doc = self.run_worker("c", so, "bench", [self.large], extra=["--warmup", "1", "--repeats", "2", "--min-repeats", "1", "--budget-s", "60"])
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertTrue(doc["oob_write"])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "out_of_bounds_write"))
        self.assertEqual(doc["final_check"]["classification"], "wrong_output")
        self.assertTrue(doc["final_check"]["oob_write"])

    def test_swapped_mn_dims_kernel_is_arg_order(self) -> None:
        """C[i*M + j] 커널: tiny(7,5)·rect(20,13) 에서 센티널·OOB 가 남지만 워커가 out_full 을 넘기므로 swapped_MN_dims 다.
        워커가 out_full 을 넘기지 않으면 partial_write/out_of_bounds_write 로 떨어져 샘플이 wrong_output 이 된다(회귀 탐지)."""
        so = self.build_c_kernel(C_SWAPPED_MN)
        proc, doc = self.run_worker("c", so, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("arg_order", "swapped_MN_dims"))
        self.assertTrue(doc["oob_write"], "가드에 쓴 사실은 플래그로 남는다")
        by_name = {t["name"]: t for t in doc["tests"]}
        for name in ("tiny", "rect"):
            t = by_name[name]
            self.assertEqual((t["classification"], t["pattern"]), ("arg_order", "swapped_MN_dims"), name)
            for r in t["runs"]:
                self.assertGreater(r["n_unwritten"], 0, name)
                self.assertTrue(r["oob_write"], name)
        self.assertEqual(by_name["small"]["classification"], "pass", "정사각에서는 C[i*M+j] == C[i*N+j]")

    def test_segfault_leaves_partial_result_with_in_progress(self) -> None:
        so = self.build_c_kernel(C_CRASH)
        proc, doc = self.run_worker("c", so, "check", self.tests)
        self.assertLess(proc.rc, 0)
        self.assertEqual(proc.signal_name, "SIGSEGV")
        self.assertFalse(proc.timed_out)
        assert doc is not None, "호출 직전에 쓴 부분 결과가 있어야 한다"
        self.assertFalse(doc["complete"])
        self.assertEqual(doc["in_progress"], {"test": "tiny", "call_idx": 0})
        self.assertTrue(doc["load_ok"])
        # 코어 덤프 차단의 실제 증거는 부분 결과에 기록된 prctl(PR_SET_DUMPABLE) 적용값이다. core_pattern 이 파이프(apport)면
        # 덤프는 cwd 에 생기지 않으므로 core* 검사는 아무것도 검증하지 못한다 — 파이프가 아닐 때만 보조로 본다.
        self.assertEqual(doc["limits"]["dumpable"], 0, "세그폴트 경로에서도 dumpable=0 이어야 한다")
        if not _core_pattern_is_pipe():
            self.assertFalse(list(so.parent.glob("core*")), "코어 덤프가 남으면 안 된다")

    def test_import_mode_missing_symbol_is_signature_mismatch(self) -> None:
        d = self.workdir()
        (d / "kernel.c").write_text(C_NO_GEMM, encoding="utf-8")
        r = run_isolated(["gcc", "-std=gnu11", "-fPIC", "-shared", "-o", "kernel.so", "kernel.c"], timeout_s=60, cwd=d, env=launch_env_minimal())
        self.assertEqual(r.rc, 0, r.stderr_tail)
        proc, doc = self.run_worker("c", d / "kernel.so", "import")
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertEqual(doc["mode"], "import")
        self.assertFalse(doc["load_ok"])
        self.assertEqual(doc["classification"], "signature_mismatch")
        self.assertIn("gemm", doc["load_error"])
        self.assertTrue(doc["complete"])

    def test_import_mode_ok(self) -> None:
        so = self.build_c_kernel(C_CORRECT)
        proc, doc = self.run_worker("c", so, "import")
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["load_ok"])
        self.assertIsNone(doc["load_error"])
        self.assertEqual(doc["classification"], "pass")

    def test_bench_pinned_cpus_survive_openmp_proc_bind(self) -> None:
        """OMP_PROC_BIND=true 면 libgomp 가 로드 시점에 주 스레드를 place 0 에 묶어 친화도가 CPU 하나로 좁아진다.
        워커는 커널 로드 전의 집합(taskset 결과)을 pinned_cpus 로 기록해야 한다(실측 회귀 테스트).

        단언이 비지 않도록 (1) 서로 다른 물리 코어의 CPU 두 개를 고르고(SMT 형제 쌍은 place 가 하나라 축소가 안 일어난다),
        (2) 같은 .so·환경·taskset 으로 '로드 뒤 친화도가 실제로 좁아지는가'를 먼저 확인해 아니면 건너뛴다."""
        if shutil.which("taskset") is None:
            self.skipTest("taskset 필요")
        import os

        cpus = _one_cpu_per_core(sorted(os.sched_getaffinity(0)))[:2]
        if len(cpus) < 2:
            self.skipTest("서로 다른 물리 코어의 CPU 2개 필요")
        d = self.workdir()
        res = build_c(C_OPENMP, d, BCFG, openmp=True, launch_prefix=[])
        self.assertTrue(res.ok, f"빌드 실패: {res.error_kind} {res.error_detail}\n{res.log}")
        assert res.artifact is not None
        env = dict(launch_env_minimal())
        env.update({"OMP_NUM_THREADS": "2", "OMP_PROC_BIND": "true", "OMP_PLACES": "cores", "OMP_DYNAMIC": "false"})
        taskset = ["taskset", "-c", ",".join(str(c) for c in cpus)]

        # 전제 확인: 이 환경에서 로드 뒤 친화도가 좁아지는가(아니면 워커가 로드 뒤에 읽어도 같은 값이라 회귀를 못 본다)
        probe = d / "affinity_probe.py"
        probe.write_text(AFFINITY_PROBE, encoding="utf-8")
        pre = run_isolated(taskset + [str(PYTHON_EXE), "-I", str(probe), str(res.artifact)], timeout_s=60.0, cwd=d, env=env)
        self.assertEqual(pre.rc, 0, pre.stderr_tail)
        aff = json.loads(pre.stdout_tail.strip().splitlines()[-1])
        self.assertEqual(aff["before"], cpus, "taskset 이 적용돼야 한다")
        if aff["after"] == aff["before"]:
            self.skipTest(f"OpenMP 커널 로드 뒤 친화도가 좁아지지 않아({aff}) 회귀를 관측할 수 없다")
        self.assertNotEqual(aff["after"], cpus)

        out = d / "bench.json"
        cmd = taskset + [str(PYTHON_EXE), "-m", "harness.worker",
                         "--backend", "c", "--artifact", str(res.artifact), "--mode", "bench", "--tests", str(self.large),
                         "--tolerance", str(self.tolerance), "--out", str(out),
                         "--warmup", "1", "--repeats", "2", "--min-repeats", "1", "--budget-s", "60"]
        proc = run_isolated(cmd, timeout_s=120.0, cwd=d, env=env)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        doc = json.loads(out.read_text(encoding="utf-8"))
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "pass")
        self.assertEqual(doc["pinned_cpus"], cpus, f"커널 로드 전의 taskset 집합이어야 한다(로드 뒤는 {aff['after']})")
        self.assertEqual(len(doc["cpu_mhz_samples"]), 0 if doc["cpu_mhz_median"] is None else 2 * len(cpus))


class PythonKernelTest(WorkerTestBase):
    BASELINE = ROOT / "tasks" / "gemm" / "baseline" / "baseline.py"

    def test_correct_kernel_with_stdout(self) -> None:
        py = self.write_py(PY_CORRECT)
        proc, doc = self.run_worker("python", py, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        self.assertIn("hello from kernel", proc.stdout_tail)  # 커널의 stdout 은 그대로 흘러나온다(run.py 가 파일로 받는다)
        assert doc is not None
        self.assertEqual(doc["classification"], "pass")
        self.assertEqual(doc["tests"][0]["runs"][0]["return_type"], "NoneType")

    def test_returned_array_not_written(self) -> None:
        py = self.write_py(PY_RETURN_ONLY)
        proc, doc = self.run_worker("python", py, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "returned_array_not_written"))
        run0 = doc["tests"][0]["runs"][0]
        self.assertEqual(run0["return_type"], "ndarray")
        self.assertEqual(run0["n_unwritten"], 7 * 5)

    def test_exception_is_runtime_crash_and_stops(self) -> None:
        py = self.write_py(PY_RAISES)
        proc, doc = self.run_worker("python", py, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "runtime_crash")
        self.assertIn("RuntimeError: boom", doc["error_detail"])
        self.assertEqual(len(doc["tests"]), 1, "첫 테스트 크래시 뒤 이후 테스트는 중단")
        run0 = doc["tests"][0]["runs"][0]
        self.assertEqual(run0["classification"], "runtime_crash")
        self.assertEqual(run0["exception"]["type"], "RuntimeError")
        self.assertIn("Traceback", run0["exception"]["traceback"])

    def test_exception_with_unprintable_str_is_still_runtime_crash(self) -> None:
        py = self.write_py(PY_RAISES_UNPRINTABLE)
        proc, doc = self.run_worker("python", py, "check", self.tests)
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        self.assertNotIn("worker_error", proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertNotIn("worker_error", doc)
        self.assertEqual(doc["classification"], "runtime_crash")
        self.assertIn("Weird", doc["error_detail"])
        self.assertIn("unprintable", doc["error_detail"])
        run0 = doc["tests"][0]["runs"][0]
        self.assertEqual(run0["exception"]["type"], "Weird")
        self.assertEqual(run0["exception"]["message"], "<unprintable Weird>")
        self.assertIn("Traceback", run0["exception"]["traceback"])

    def test_import_exception_with_unprintable_str_is_compile_error(self) -> None:
        py = self.write_py(PY_IMPORT_RAISES_UNPRINTABLE)
        proc, doc = self.run_worker("python", py, "import")
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertFalse(doc["load_ok"])
        self.assertEqual(doc["classification"], "compile_error")
        self.assertEqual(doc["load_error"], "Weird: <unprintable Weird>")

    def test_accumulate_into_c(self) -> None:
        py = self.write_py(PY_ACCUMULATE)
        _, doc = self.run_worker("python", py, "check", self.tests)
        assert doc is not None
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "accumulated_into_c"))
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("wrong_output", "accumulated_into_c"), t["name"])

    def test_accumulate_into_c_on_real_small_shape(self) -> None:
        """실제 small(256³) 입력은 |ref| < 2⁻¹³ 인 원소가 있어 누적 결과 하나가 정확히 센티널로 되돌아간다 —
        누적 검사가 partial_write 보다 먼저여야 accumulated_into_c 다(설계 §10 v2.1 규칙 4)."""
        small = ROOT / "tasks" / "gemm" / "tests" / "small.npz"
        if not small.exists():
            self.skipTest("tasks/gemm/tests/small.npz 없음")
        py = self.write_py(PY_ACCUMULATE)
        proc, doc = self.run_worker("python", py, "check", [small], extra=["--check-repeats", "1"])
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "accumulated_into_c"))
        run0 = doc["tests"][0]["runs"][0]
        self.assertEqual(run0["pattern"], "accumulated_into_c")
        self.assertGreaterEqual(run0["n_unwritten"], 1, "전제: 센티널로 되돌아간 원소가 있다")

    def test_nondeterministic_is_race_condition(self) -> None:
        py = self.write_py(PY_NONDET)
        _, doc = self.run_worker("python", py, "check", self.tests, extra=["--check-repeats", "2"])
        assert doc is not None
        self.assertTrue(doc["nondeterministic"])
        self.assertEqual(doc["classification"], "race_condition")
        self.assertTrue(doc["error_detail"].startswith("nondeterministic:"))
        self.assertTrue(all(t["nondeterministic"] for t in doc["tests"]))

    def test_inputs_mutated(self) -> None:
        py = self.write_py(PY_MUTATES)
        _, doc = self.run_worker("python", py, "check", self.tests)
        assert doc is not None
        self.assertTrue(doc["inputs_mutated"])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "inputs_mutated"))

    def test_bench_on_numpy_baseline(self) -> None:
        if not self.BASELINE.exists():
            self.skipTest("tasks/gemm/baseline/baseline.py 없음")
        proc, doc = self.run_worker("python", self.BASELINE, "bench", [self.large],
                                    extra=["--warmup", "2", "--repeats", "5", "--min-repeats", "3", "--budget-s", "60"],
                                    workdir=self.workdir())
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        self.assertEqual(proc.stdout_tail, "")
        assert doc is not None
        for key in ("mode", "test", "M", "N", "K", "load_ok", "blas", "limits", "alignment", "warmup", "repeats", "repeats_reduced",
                    "warmup_ms", "timed_ms", "median_ms", "mean_ms", "min_ms", "max_ms", "cv", "gflops", "nondeterministic_runs",
                    "nondeterministic_bitwise_runs", "cpu_mhz_samples", "cpu_mhz_median", "cpu_mhz_min", "throttle_events",
                    "temp_c_start", "temp_c_end", "loadavg_start", "loadavg_end", "final_check", "inputs_mutated", "oob_write",
                    "classification", "error_detail", "exception", "working_set_mb", "complete"):
            self.assertIn(key, doc, key)
        self.assertEqual(doc["mode"], "bench")
        self.assertEqual(doc["test"], "large")
        self.assertEqual((doc["M"], doc["N"], doc["K"]), (48, 48, 48))
        self.assertEqual(doc["alignment"], 64)
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "pass")
        self.assertEqual(doc["final_check"]["classification"], "pass")
        self.assertEqual((doc["warmup"], doc["repeats"]), (2, 5))
        self.assertFalse(doc["repeats_reduced"])
        self.assertEqual(len(doc["warmup_ms"]), 2)
        self.assertEqual(len(doc["timed_ms"]), 5)
        self.assertAlmostEqual(doc["median_ms"], float(np.median(doc["timed_ms"])))
        self.assertAlmostEqual(doc["gflops"], 2 * 48**3 / (doc["median_ms"] / 1000) / 1e9)
        self.assertEqual(doc["nondeterministic_runs"], 0)
        self.assertIsInstance(doc["cpu_mhz_samples"], list)
        self.assertFalse(doc["inputs_mutated"])
        self.assertFalse(doc["oob_write"])
        self.assertAlmostEqual(doc["working_set_mb"], 3 * 48 * 48 * 4 / 2**20)

    def test_bench_early_exit_is_timeout(self) -> None:
        py = self.write_py(PY_SLEEP.replace("SLEEP_S", "0.3"))
        proc, doc = self.run_worker("python", py, "bench", [self.large],
                                    extra=["--warmup", "3", "--repeats", "10", "--min-repeats", "3", "--budget-s", "1"])
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "timeout")
        self.assertTrue(doc["error_detail"].startswith("first_call_"), doc["error_detail"])
        self.assertEqual(len(doc["warmup_ms"]), 1)
        self.assertEqual(doc["timed_ms"], [])
        self.assertIsNone(doc["median_ms"])
        self.assertIsNone(doc["final_check"])

    def test_bench_adaptive_reduction(self) -> None:
        py = self.write_py(PY_SLEEP.replace("SLEEP_S", "0.05"))
        proc, doc = self.run_worker("python", py, "bench", [self.large],
                                    extra=["--warmup", "3", "--repeats", "10", "--min-repeats", "3", "--budget-s", "0.5"])
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        assert doc is not None
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "pass")
        self.assertTrue(doc["repeats_reduced"])
        self.assertEqual(doc["warmup"], 1)
        self.assertGreaterEqual(doc["repeats"], 3)
        self.assertLess(doc["repeats"], 10)
        self.assertEqual(len(doc["warmup_ms"]), 1)
        self.assertEqual(len(doc["timed_ms"]), doc["repeats"])

    def test_import_mode_failures(self) -> None:
        for code, expected, needle in ((PY_IMPORT_FAILS, "compile_error", "ModuleNotFoundError"),
                                       (PY_NO_GEMM, "signature_mismatch", "gemm"),
                                       (PY_ASYNC, "signature_mismatch", "coroutine")):
            py = self.write_py(code)
            proc, doc = self.run_worker("python", py, "import")
            self.assertEqual(proc.rc, 0, proc.stderr_tail)
            assert doc is not None
            self.assertFalse(doc["load_ok"])
            self.assertEqual(doc["classification"], expected)
            self.assertIn(needle, doc["load_error"])
            self.assertTrue(doc["complete"])

    def test_harness_bug_exits_rc3_with_worker_error(self) -> None:
        py = self.write_py(PY_CORRECT)
        proc, doc = self.run_worker("python", py, "check", [self.tmp / "does_not_exist.npz"])
        self.assertEqual(proc.rc, 3)
        self.assertIn("worker_error", proc.stderr_tail)
        assert doc is not None
        self.assertFalse(doc["complete"])
        self.assertIn("Traceback", doc["worker_error"])


if __name__ == "__main__":
    unittest.main()
