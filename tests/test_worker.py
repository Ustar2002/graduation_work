"""worker 테스트: 임시 디렉터리에 작은 커널을 만들어 ``harness.worker`` 를 ``run_isolated`` 로 실행하고 분류·결과 JSON 키·
종료 코드를 확인한다. 테스트 입력은 여기서 작은 형태로 직접 만든다(``tasks/<task>/tests`` 불필요).

- ``CKernelTest``: gcc 로 만든 C 커널(정상/전치/OOB/크래시). ELF 공유 라이브러리와 POSIX 종료 코드를 전제하므로 Linux 전용이다.
- ``PythonKernelTest``: python 백엔드의 판정·bench·종료 규칙(GEMM).
- ``PythonTaskTest`` / ``NumbaKernelTest`` / ``HalideKernelTest`` / ``HalideFailureTest``: 설계 v3 §5 의 백엔드. 손으로 쓴 실제
  Numba·Halide 커널(``tests/fixtures/kernels``)을 gemm·jacobi2d·dot 과 int32 출력 과제(histogram)에서 import·check·bench,
  positional·keyword 로 돌리고, 계약이 정한 실패(서명 없는 numba, 틀린 Func, 정의 오류, 형태별 빌드 실패, 없는 오토스케줄러…)와
  ``--inspect`` 의 관찰값, 워커 stdout 이 비어 있음을 확인한다.
- ``CAbiTest``: C ABI 의 positional 호출과 키워드 모드의 매개변수 구조체(§5.2)를 이 머신의 C 컴파일러로 만든 커널로 확인한다
  (Linux 는 gcc, 그 밖은 ``tools/devcheck.py`` 의 zig 직렬 빌드; 둘 다 없으면 건너뛴다). ``RustAbiTest``: ``rustc`` 로 만든
  ``cdylib`` 를 같은 ``--backend c`` 로 돌린다(``rustc`` 가 없으면 건너뛴다).
- ``BuffersTest`` / ``WorkerCliTest`` / ``ResultDocumentTest``: 길이 0 배열의 버퍼, 명령줄 규칙, 결과 문서 보호 장치(프로세스
  안에서 도는 순수 부분).
"""

from __future__ import annotations

import contextlib
import ctypes
import gc
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from harness import kernels, sigs, testgen, worker
from harness.build.c import build_c
from harness.config import BuildCfg
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import ProcResult, run_isolated
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from tests._support import IS_LINUX, requires_linux, requires_modules, requires_tool

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

PY_EXITS = """
import os
def gemm(A, B, C, M, N, K):
    os._exit(7)
"""

# 첫 호출은 입력을 바꾸고 끝나며(검사됨), 둘째 호출에서 프로세스가 죽는다(시그널·시간 초과의 대역)
PY_MUTATES_THEN_EXITS = """
import os
import numpy as np
_calls = [0]
def gemm(A, B, C, M, N, K):
    _calls[0] += 1
    if _calls[0] == 2:
        os._exit(7)
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


# ----------------------------------------------------------------------------- 설계 v3 §5: 과제 입력·픽스처 커널·C 빌드

FIXTURES = ROOT / "tests" / "fixtures" / "kernels"
THREADS = 2
# Numba·Halide 워커의 환경: 오케스트레이터가 언어 설정의 ``env`` (설계 v3 §8)로 거는 것과 같은 변수다
NUMBA_THREADING_LAYER = "omp"
JIT_ENV = {"NUMBA_NUM_THREADS": str(THREADS), "NUMBA_THREADING_LAYER": NUMBA_THREADING_LAYER, "HL_NUM_THREADS": str(THREADS)}
# Numba·Halide 워커는 임포트와 JIT 에 수 초가 걸리고, 다른 작업이 도는 머신에서는 수십 초까지 늘어난다
JIT_TIMEOUT_S = 600.0
BENCH_ARGS = ["--warmup", "1", "--repeats", "3", "--min-repeats", "1", "--budget-s", "120"]
# 백엔드 인자(설계 v3 §5.1): numba 는 스레드 수만, halide 는 모드와 스레드 수(auto 에서는 오토스케줄러의 parallelism)
NUMBA_ARGS = ["--threads", str(THREADS)]
HALIDE_AUTO = ["--halide-mode", "auto", "--threads", str(THREADS)]
HALIDE_MANUAL = ["--halide-mode", "manual", "--threads", str(THREADS)]

# 형태 이름은 testgen 이 요구하는 네 개다. 첫 세 개는 정확성, large 는 bench 에 쓴다.
TASK_SHAPES: dict[str, list[tuple[str, dict[str, int]]]] = {
    "gemm": [("tiny", {"M": 7, "N": 5, "K": 3}), ("rect", {"M": 20, "N": 13, "K": 31}),
             ("small", {"M": 16, "N": 16, "K": 16}), ("large", {"M": 48, "N": 48, "K": 48})],
    "jacobi2d": [("tiny", {"H": 5, "W": 7, "T": 3}), ("rect", {"H": 13, "W": 20, "T": 4}),
                 ("small", {"H": 32, "W": 32, "T": 5}), ("large", {"H": 64, "W": 64, "T": 6})],
    "dot": [("tiny", {"n": 7}), ("rect", {"n": 1003}), ("small", {"n": 4096}), ("large", {"n": 20000})],
    "histogram": [("tiny", {"n": 10, "n_bins": 4}), ("rect", {"n": 1003, "n_bins": 37}),
                  ("small", {"n": 4096, "n_bins": 16}), ("large", {"n": 20000, "n_bins": 64})],
    "gather": [("tiny", {"n": 5, "m": 7}), ("rect", {"n": 100, "m": 33}),
               ("small", {"n": 64, "m": 256}), ("large", {"n": 1000, "m": 5000})],
}
CORRECTNESS = ("tiny", "rect", "small")

# 손으로 만든 과제 둘(저장소의 tasks/ 와 무관). histogram: int32 입력·int32 출력·정확 비교.
HISTOGRAM_SIGNATURES_YAML = """task: histogram
function: histogram
args:
  - {name: keys, kind: in,  dtype: int32, shape: [n]}
  - {name: hist, kind: out, dtype: int32, shape: [n_bins]}
  - {name: n, kind: dim}
  - {name: n_bins, kind: dim}
"""
HISTOGRAM_TOLERANCE_YAML = "task: histogram\ndtype: int32\ncompare: exact\nsentinel: -2048\n"
# gather: out[j] = x[idx[j]]. 스칼라와 포인터가 번갈아 나와 키워드 모드의 C 구조체에 패딩이 생기고(64비트: int 4 + 패딩 4 +
# 포인터 8 …), float32·int32 포인터가 섞여 있다. m = 0 이면 입력 idx 와 출력 out 이 길이 0 이다.
GATHER_SIGNATURES_YAML = """task: gather
function: gather
args:
  - {name: n, kind: dim}
  - {name: x, kind: in,  dtype: float32, shape: [n]}
  - {name: m, kind: dim}
  - {name: idx, kind: in,  dtype: int32, shape: [m]}
  - {name: out, kind: out, dtype: float32, shape: [m]}
"""
GATHER_TOLERANCE_YAML = ("task: gather\ndtype: float32\nrtol: 1.0e-6\natol: 1.0e-6\nreference_dtype: float64\n"
                         "numeric_pattern_tol_ratio: 20\nsentinel: -2048.0\n")
EMPTY_DIMS = {"histogram": {"n": 0, "n_bins": 4}, "gather": {"n": 4, "m": 0}}


def _gemm_case(shape, rng):
    A = rng.random((shape["M"], shape["K"]), dtype=np.float32) * np.float32(2) - np.float32(1)
    B = rng.random((shape["K"], shape["N"]), dtype=np.float32) * np.float32(2) - np.float32(1)
    return dict(shape), {"A": A, "B": B}, "C", A.astype(np.float64) @ B.astype(np.float64)


def _jacobi2d_case(shape, rng):
    src = rng.random((shape["H"], shape["W"]), dtype=np.float32)
    cur = src.astype(np.float64)
    nxt = cur.copy()
    for _ in range(shape["T"]):
        nxt[1:-1, 1:-1] = 0.25 * (cur[:-2, 1:-1] + cur[2:, 1:-1] + cur[1:-1, :-2] + cur[1:-1, 2:])
        cur, nxt = nxt, cur
    return dict(shape), {"src": src}, "dst", cur


def _dot_case(shape, rng):
    x = rng.random(shape["n"], dtype=np.float32)
    y = rng.random(shape["n"], dtype=np.float32)
    return dict(shape), {"x": x, "y": y}, "out", np.array([np.dot(x.astype(np.float64), y.astype(np.float64))])


def _histogram_case(shape, rng):
    keys = rng.integers(0, shape["n_bins"], size=shape["n"], dtype=np.int32)
    return dict(shape), {"keys": keys}, "hist", np.bincount(keys, minlength=shape["n_bins"]).astype(np.int32)


def _gather_case(shape, rng):
    x = rng.random(shape["n"], dtype=np.float32)
    idx = rng.integers(0, shape["n"], size=shape["m"], dtype=np.int32)
    return dict(shape), {"x": x, "idx": idx}, "out", x[idx].astype(np.float64)


@dataclass(frozen=True)
class TaskFiles:
    """워커에 넘기는 과제 하나의 파일: 명세, 허용오차, 테스트 입력 디렉터리."""

    signatures: Path
    tolerance: Path
    tests_dir: Path

    def tests(self, *names: str) -> list[Path]:
        return [self.tests_dir / f"{name}.npz" for name in names]


def make_tasks(root: Path) -> dict[str, TaskFiles]:
    """gemm·jacobi2d·dot(저장소의 명세와 허용오차)과 손으로 만든 histogram·gather 의 작은 테스트 입력을 ``root`` 아래에 만든다.
    histogram 과 gather 에는 길이 0 배열이 있는 ``empty.npz`` 도 만든다."""
    cases = {"gemm": (_gemm_case, "float32", None), "jacobi2d": (_jacobi2d_case, "float32", None), "dot": (_dot_case, "float32", None),
             "histogram": (_histogram_case, "int32", -2048), "gather": (_gather_case, "float32", None)}
    handmade = {"histogram": (HISTOGRAM_SIGNATURES_YAML, HISTOGRAM_TOLERANCE_YAML), "gather": (GATHER_SIGNATURES_YAML, GATHER_TOLERANCE_YAML)}
    tasks: dict[str, TaskFiles] = {}
    for index, (task, (make_case, out_dtype, sentinel)) in enumerate(cases.items()):
        task_dir = root / task
        tests_dir = task_dir / "tests"
        with contextlib.redirect_stderr(io.StringIO()):      # testgen 의 진행 로그
            testgen.generate(task, 20261122 + 10 * index, tests_dir, TASK_SHAPES[task], make_case, "numpy", out_dtype=out_dtype,
                             sentinel=sentinel)
        if task in handmade:
            signatures, tolerance = task_dir / "signatures.yaml", task_dir / "tolerance.yaml"
            signatures.write_text(handmade[task][0], encoding="utf-8", newline="\n")
            tolerance.write_text(handmade[task][1], encoding="utf-8", newline="\n")
        else:
            signatures, tolerance = ROOT / "tasks" / task / "signatures.yaml", ROOT / "tasks" / task / "tolerance.yaml"
        tasks[task] = TaskFiles(signatures=signatures, tolerance=tolerance, tests_dir=tests_dir)
    with open(tasks["histogram"].tests_dir / "empty.npz", "wb") as fh:
        np.savez(fh, keys=np.zeros(0, np.int32), hist_ref=np.zeros(4, np.int32), n=np.int64(0), n_bins=np.int64(4))
    with open(tasks["gather"].tests_dir / "empty.npz", "wb") as fh:
        np.savez(fh, x=np.linspace(0.0, 1.0, 4, dtype=np.float32), idx=np.zeros(0, np.int32), out_ref=np.zeros(0, np.float64),
                 n=np.int64(4), m=np.int64(0))
    return tasks


def keyword_only(code: str, function: str) -> str:
    """``def fn(a, b):`` 를 키워드 전용 ``def fn(*, a, b):`` 로 바꾼다(args=keyword 조건의 시그니처; 설계 v3 §3)."""
    out, count = re.subn(rf"^def {re.escape(function)}\(", f"def {function}(*, ", code, flags=re.MULTILINE)
    if count != 1:
        raise AssertionError(f"{function} 의 정의가 정확히 하나여야 한다(찾은 수 {count})")
    return out


def patched(code: str, old: str, new: str) -> str:
    """``code`` 의 ``old`` 한 곳을 ``new`` 로 바꾼 커널. 바꿀 곳이 정확히 하나가 아니면 테스트 작성 오류다."""
    if code.count(old) != 1:
        raise AssertionError(f"바꿀 곳이 정확히 하나여야 한다({code.count(old)}곳): {old!r}")
    return code.replace(old, new)


def c_toolchain() -> str | None:
    """이 머신에서 작은 C 커널을 공유 라이브러리로 만드는 방법: Linux 의 ``gcc``, 아니면 ``zig``(ziglang 패키지), 없으면 None."""
    if IS_LINUX and shutil.which("gcc") is not None:
        return "gcc"
    if importlib.util.find_spec("ziglang") is not None:
        return "zig"
    return None


def _devcheck() -> Any:
    """``tools/devcheck.py`` (패키지가 아니다)를 파일 경로로 올린다: zig 로 C 커널을 빌드하는 ``build_c_zig`` 를 빌려 쓴다."""
    name = "devcheck"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / "devcheck.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def build_shared_library(code: str, build_dir: Path, args_header: str | None = None) -> Path:
    """``code`` 를 ``build_dir`` 에서 공유 라이브러리로 만든다(``c_toolchain()``). ``args_header`` 는 키워드 모드의 ``kernel_args.h``."""
    if c_toolchain() == "gcc":
        (build_dir / "kernel.c").write_text(code, encoding="utf-8", newline="\n")
        if args_header is not None:
            (build_dir / sigs.C_ARGS_HEADER_NAME).write_text(args_header, encoding="utf-8", newline="\n")
        proc = run_isolated(["gcc", "-std=gnu11", "-O2", "-fPIC", "-shared", "-Wall", "-o", "kernel.so", "kernel.c"],
                            timeout_s=120.0, cwd=build_dir, env=launch_env_minimal())
        if proc.rc != 0:
            raise AssertionError(f"gcc 빌드 실패: {proc.stderr_tail}")
        return build_dir / "kernel.so"
    built = _devcheck().build_c_zig(code, build_dir, args_header)
    if not built.ok or built.artifact is None:
        raise AssertionError(f"zig 빌드 실패: {built.error_detail}")
    return Path(built.artifact)


PY_GATHER = """
def gather(n, x, m, idx, out):
    out[:] = x[idx]
"""

# 다섯 인자를 전부 읽는다: 구조체의 필드 하나라도 어긋나면(순서·패딩) 출력이 틀리거나 죽는다
C_GATHER = """
void gather(int n, const float* x, int m, const int* idx, float* out)
{
    for (int j = 0; j < m; j++) {
        int i = idx[j];
        out[j] = (i >= 0 && i < n) ? x[i] : -1.0f;
    }
}
"""

C_GATHER_KEYWORD = """
#include "kernel_args.h"

void gather(const gather_args* args)
{
    for (int j = 0; j < args->m; j++) {
        int i = args->idx[j];
        args->out[j] = (i >= 0 && i < args->n) ? args->x[i] : -1.0f;
    }
}
"""

# C_GATHER_KEYWORD 와 같은 커널: 구조체는 하네스가 주는 src/kernel_args.rs 의 #[repr(C)] GatherArgs
RS_GATHER_KEYWORD = """#![allow(non_snake_case)]
mod kernel_args;
use kernel_args::GatherArgs;

#[no_mangle]
pub unsafe extern "C" fn gather(args: *const GatherArgs) {
    let a = &*args;
    for j in 0..a.m as usize {
        let i = *a.idx.add(j);
        *a.out.add(j) = if i >= 0 && i < a.n { *a.x.add(i as usize) } else { -1.0 };
    }
}
"""

NUMBA_LAZY = """
from numba import njit


@njit
def gemm(A, B, C, M, N, K):
    for i in range(M):
        for j in range(N):
            s = 0.0
            for k in range(K):
                s += A[i, k] * B[k, j]
            C[i, j] = s
"""

GEMM_NUMBA_SIGNATURE = "void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)"
# 서명의 배열 타입이 float64: 임포트(컴파일)는 되지만 하네스가 넘기는 float32 배열에 맞는 정의가 없다
NUMBA_FLOAT64_SIGNATURE = patched(NUMBA_LAZY, "@njit\n", f'@njit("{GEMM_NUMBA_SIGNATURE.replace("float32", "float64")}")\n')
# 정의되지 않은 이름: 명시 서명이라 임포트 때 컴파일이 실패한다
NUMBA_TYPING_ERROR = patched(patched(NUMBA_LAZY, "@njit\n", f'@njit("{GEMM_NUMBA_SIGNATURE}")\n'),
                             "s = 0.0", "s = undefined_helper(i, j)")

HL_GEMM = (FIXTURES / "halide" / "gemm.py").read_text(encoding="utf-8")
HL_GEMM_UPDATE = "    C[j, i] += A[k.x, i] * B[j, k.x]\n"
HL_GEMM_FIRST_LINE = '    i, j = hl.Var("i"), hl.Var("j")\n'

# 출력(C: 2차원 float32)과 차원 수·타입이 다른 Func, Func 이 아닌 반환값
HL_WRONG_NDIM = 'import halide as hl\n\n\ndef gemm(A, B, M, N, K):\n    i = hl.Var("i")\n    C = hl.Func("C")\n    C[i] = A[0, i]\n    return C\n'
HL_WRONG_TYPE = patched(patched(HL_GEMM, "hl.f32(0.0)", "hl.f64(0.0)"), HL_GEMM_UPDATE,
                        "    C[j, i] += hl.f64(A[k.x, i]) * hl.f64(B[j, k.x])\n")
HL_RETURNS_NONE = patched(HL_GEMM, "    return C\n", "")
# 2-D ImageParam 을 첨자 세 개로 읽는다: Halide 가 정의 시점에 거부한다
HL_DEFINITION_ERROR = patched(HL_GEMM, HL_GEMM_UPDATE, "    C[j, i] += A[k.x, i, 0] * B[j, k.x]\n")
# 입력을 범위 밖에서 읽는다: 정의와 JIT 는 되고 realize 가 실패한다
HL_READS_OUT_OF_BOUNDS = patched(HL_GEMM, HL_GEMM_UPDATE, "    C[j, i] += A[k.x + 1, i] * B[j, k.x]\n")
# 스케줄 호출이 있는 커널(halide_mode=auto 에서는 오토스케줄러가 거부한다)
HL_SCHEDULED = patched(HL_GEMM, "    return C\n", "    C.update().parallel(i)\n    return C\n")
# 한 형태(rect: M == 20)에서만 빌드 함수가 실패한다
HL_FAILS_FOR_ONE_SHAPE = patched(HL_GEMM, HL_GEMM_FIRST_LINE,
                                 '    if M == 20:\n        raise ValueError("no pipeline for M == 20")\n' + HL_GEMM_FIRST_LINE)
# 큰 형태에서만 빌드 함수가 실패하거나 워커 프로세스를 끝낸다: --inspect-dims(성능 형태)의 빌드에서만 드러난다
HL_FAILS_WHEN_LARGE = patched(HL_GEMM, HL_GEMM_FIRST_LINE,
                              '    if M > 100:\n        raise ValueError("too large to build")\n' + HL_GEMM_FIRST_LINE)
HL_EXITS_WHEN_LARGE = patched(HL_GEMM, HL_GEMM_FIRST_LINE, "    if M > 100:\n        __import__(\"os\")._exit(7)\n" + HL_GEMM_FIRST_LINE)


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
                   extra: list[str] | None = None, timeout_s: float = 120.0, workdir: Path | None = None,
                   task: TaskFiles | None = None, env: dict[str, str] | None = None) -> tuple[ProcResult, dict | None]:
        """워커를 격리 실행한다. 결과 JSON·cwd 는 ``workdir`` (기본: 임시 커널의 디렉터리; 저장소 안 파일이면 반드시 지정).
        ``task`` 를 주면 그 과제의 명세·허용오차를 쓴다(없으면 gemm 과 이 클래스의 tolerance.yaml). ``env`` 는 최소 환경 위에 덮어쓴다."""
        wd = workdir if workdir is not None else artifact.parent
        assert self.tmp in wd.parents or wd == self.tmp, "테스트는 임시 디렉터리 밖에 쓰지 않는다"
        out = wd / f"{mode}.json"
        out.unlink(missing_ok=True)      # 같은 디렉터리의 앞선 실행 결과를 이번 결과로 읽지 않게
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", backend, "--artifact", str(artifact), "--mode", mode,
               "--tolerance", str(task.tolerance if task is not None else self.tolerance), "--out", str(out)]
        if task is not None:
            cmd += ["--signatures", str(task.signatures)]
        if tests:
            cmd += ["--tests", ",".join(str(t) for t in tests)]
        cmd += extra or []
        proc = run_isolated(cmd, timeout_s=timeout_s, cwd=wd, env={**launch_env_minimal(), **(env or {})})
        doc = json.loads(out.read_text(encoding="utf-8")) if out.exists() else None
        return proc, doc


@requires_linux("ELF 공유 라이브러리(kernel.so, nm -D)와 POSIX 종료 코드(시그널)를 전제한다")
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
        # 호출 기록에도 bench final_check 와 같은 이름으로 틀린 원소 수가 있다(repair 의 "k of n elements")
        self.assertEqual((run0["n_fail"], run0["n_total"]), (7 * 5, 7 * 5))

    def test_passing_runs_record_the_element_counts(self) -> None:
        py = self.write_py(PY_CORRECT)
        _, doc = self.run_worker("python", py, "check", self.tests)
        assert doc is not None
        for t, (M, N) in zip(doc["tests"], ((7, 5), (20, 13), (16, 16)), strict=True):
            self.assertEqual({(r["n_fail"], r["n_total"]) for r in t["runs"]}, {(0, M * N)}, t["name"])

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
        self.assertEqual((run0["n_fail"], run0["n_total"]), (None, None))
        # 검사한 호출이 없다: "검사했고 없음"(False)이 아니라 None(행에서는 NA)
        self.assertEqual((doc["inputs_mutated"], doc["oob_write"]), (None, None))

    def test_process_death_in_the_first_call_leaves_the_contract_flags_unknown(self) -> None:
        py = self.write_py(PY_EXITS)
        proc, doc = self.run_worker("python", py, "check", self.tests)
        self.assertEqual(proc.rc, 7, proc.stderr_tail)
        assert doc is not None
        self.assertFalse(doc["complete"])
        self.assertEqual(doc["in_progress"], {"test": "tiny", "call_idx": 0})
        self.assertEqual((doc["inputs_mutated"], doc["oob_write"]), (None, None))

    def test_flags_seen_before_the_process_died_stay_in_the_document(self) -> None:
        py = self.write_py(PY_MUTATES_THEN_EXITS)
        proc, doc = self.run_worker("python", py, "check", self.tests, extra=["--check-repeats", "2"])
        self.assertEqual(proc.rc, 7, proc.stderr_tail)
        assert doc is not None
        self.assertFalse(doc["complete"])
        self.assertEqual(doc["in_progress"], {"test": "tiny", "call_idx": 1})
        self.assertEqual((doc["inputs_mutated"], doc["oob_write"]), (True, False))

    def test_sanitize_mode_still_checks_the_results(self) -> None:
        """``--sanitize`` 는 버퍼 배치만 바꾼다: 판정·입력 변조 검사는 그대로다."""
        _, doc = self.run_worker("python", self.write_py(PY_CORRECT), "check", self.tests, extra=["--sanitize"])
        assert doc is not None
        self.assertEqual((doc["sanitize"], doc["classification"], doc["inputs_mutated"], doc["oob_write"]),
                         (True, "pass", False, False))
        _, doc = self.run_worker("python", self.write_py(PY_MUTATES), "check", self.tests, extra=["--sanitize"])
        assert doc is not None
        self.assertEqual((doc["classification"], doc["error_detail"], doc["inputs_mutated"]),
                         ("wrong_output", "inputs_mutated", True))
        _, doc = self.run_worker("python", self.write_py(PY_CORRECT), "check", self.tests)
        assert doc is not None
        self.assertFalse(doc["sanitize"])

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


# ----------------------------------------------------------------------------- 설계 v3 §5 의 백엔드


class TaskWorkerTestBase(WorkerTestBase):
    """공통 준비(테스트 메서드 없음): ``make_tasks`` 의 과제 입력, 픽스처 커널, 결과 문서의 공통 단언."""

    tasks: dict[str, TaskFiles]

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.tasks = make_tasks(cls.tmp / "tasks")

    def fixture(self, relpath: str, function: str, *, keyword: bool = False) -> Path:
        """``tests/fixtures/kernels/<relpath>`` 를 임시 디렉터리의 ``kernel.py`` 로 쓴다(keyword 면 키워드 전용 시그니처로 바꿔서)."""
        code = (FIXTURES / relpath).read_text(encoding="utf-8")
        return self.write_py(keyword_only(code, function) if keyword else code)

    def run_task(self, task: str, backend: str, artifact: Path, mode: str, tests: tuple[str, ...] = (), *,
                 extra: list[str] | None = None, env: dict[str, str] | None = None,
                 workdir: Path | None = None) -> tuple[ProcResult, dict[str, Any]]:
        """과제 ``task`` 의 입력으로 워커를 돌린다. 결과 문서는 반드시 있어야 한다."""
        files = self.tasks[task]
        proc, doc = self.run_worker(backend, artifact, mode, files.tests(*tests), extra=extra, timeout_s=JIT_TIMEOUT_S,
                                    workdir=workdir, task=files, env={**JIT_ENV, **(env or {})})
        self.assertIsNotNone(doc, f"결과 문서가 없다 (rc={proc.rc}): {proc.stderr_tail}")
        assert doc is not None
        return proc, doc

    def finished(self, proc: ProcResult, doc: dict[str, Any]) -> dict[str, Any]:
        """정상 종료한 워커: rc 0, 완전한 결과 문서, 워커가 stdout 에 쓴 것 없음."""
        self.assertEqual(proc.rc, 0, proc.stderr_tail)
        self.assertEqual(proc.stdout_tail, "", "워커는 stdout 에 아무것도 쓰지 않는다")
        self.assertNotIn("worker_error", doc)
        self.assertTrue(doc["complete"])
        self.assertIsNone(doc["in_progress"])
        return doc

    def assert_load_failed(self, doc: dict[str, Any], classification: str, detail: str) -> None:
        self.assertEqual((doc["load_ok"], doc["classification"], doc["load_error"], doc["error_detail"]),
                         (False, classification, detail, detail))
        self.assertNotIn("tests", doc, "로드에 실패한 커널은 호출하지 않는다")
        self.assertNotIn("inspect", doc)

    def assert_check_passed(self, doc: dict[str, Any], names: tuple[str, ...], *, per_shape_build: bool,
                            return_type: str | None) -> None:
        self.assertEqual((doc["load_ok"], doc["classification"], doc["error_detail"]), (True, "pass", ""), doc.get("tests"))
        self.assertEqual((doc["inputs_mutated"], doc["oob_write"], doc["nondeterministic"]), (False, False, False))
        self.assertEqual([t["name"] for t in doc["tests"]], list(names))
        for t in doc["tests"]:
            self.assertEqual((t["classification"], len(t["runs"])), ("pass", 2), t["name"])
            self.assertEqual({r["return_type"] for r in t["runs"]}, {return_type})
            if per_shape_build:      # 형태마다 파이프라인을 새로 만든다
                self.assertGreater(t["build_ms"], 0.0)
            else:
                self.assertNotIn("build_ms", t)
        if per_shape_build:
            # JIT 는 빌드에 들어가고 첫 호출에는 섞이지 않는다(프로세스의 첫 JIT 는 0.1 초 이상, 가장 작은 형태의 호출은 ms 이하)
            first = doc["tests"][0]
            self.assertGreater(first["build_ms"], first["runs"][0]["elapsed_ms"])

    def assert_bench_passed(self, doc: dict[str, Any], task: str, *, per_shape_build: bool) -> None:
        self.assertEqual((doc["load_ok"], doc["classification"], doc["error_detail"]), (True, "pass", ""), doc.get("exception"))
        self.assertEqual((doc["test"], doc["dims"]), ("large", TASK_SHAPES[task][3][1]))
        self.assertEqual(doc["final_check"]["classification"], "pass")
        self.assertEqual((len(doc["warmup_ms"]), len(doc["timed_ms"])), (1, 3))
        self.assertGreater(doc["median_ms"], 0.0)
        self.assertEqual((doc["nondeterministic_runs"], doc["inputs_mutated"], doc["oob_write"]), (0, False, False))
        if per_shape_build:
            self.assertGreater(doc["build_ms"], doc["warmup_ms"][0], "빌드·JIT 시간은 첫 호출에도 섞이지 않는다")
        else:
            self.assertNotIn("build_ms", doc)


class PythonTaskTest(TaskWorkerTestBase):
    """python 백엔드: int32 출력 과제, 길이 0 배열, 상대 경로 산출물."""

    def test_int32_task_with_a_zero_length_input(self) -> None:
        py = self.fixture("python/histogram.py", "histogram")
        doc = self.finished(*self.run_task("histogram", "python", py, "check", (*CORRECTNESS, "empty")))
        self.assert_check_passed(doc, (*CORRECTNESS, "empty"), per_shape_build=False, return_type="NoneType")
        self.assertEqual(doc["backend_detail"], {})
        self.assertEqual(doc["tests"][-1]["dims"], EMPTY_DIMS["histogram"])
        self.assertEqual(doc["max_abs_err"], 0.0)

    def test_zero_length_input_and_output(self) -> None:
        py = self.write_py(PY_GATHER)
        doc = self.finished(*self.run_task("gather", "python", py, "check", ("tiny", "empty")))
        self.assert_check_passed(doc, ("tiny", "empty"), per_shape_build=False, return_type="NoneType")
        empty = doc["tests"][-1]
        self.assertEqual(empty["dims"], EMPTY_DIMS["gather"])
        self.assertEqual({r["n_unwritten"] for r in empty["runs"]}, {0})

    def test_keyword_mode_with_a_keyword_only_definition(self) -> None:
        py = self.fixture("python/histogram.py", "histogram", keyword=True)
        doc = self.finished(*self.run_task("histogram", "python", py, "check", ("tiny",), extra=["--args-mode", "keyword"]))
        self.assert_check_passed(doc, ("tiny",), per_shape_build=False, return_type="NoneType")
        # 같은 커널을 positional 로 부르면 호출이 TypeError 로 실패한다
        doc = self.finished(*self.run_task("histogram", "python", py, "check", ("tiny",)))
        self.assertEqual(doc["classification"], "runtime_crash")
        self.assertEqual(doc["tests"][0]["runs"][0]["exception"]["type"], "TypeError")

    def test_relative_artifact_path(self) -> None:
        py = self.write_py(PY_GATHER)
        doc = self.finished(*self.run_task("gather", "python", Path(py.name), "check", ("tiny",), workdir=py.parent))
        self.assertEqual((doc["artifact"], doc["classification"]), ("kernel.py", "pass"))

    def test_inspect_has_nothing_to_report(self) -> None:
        py = self.write_py(PY_GATHER)
        doc = self.finished(*self.run_task("gather", "python", py, "import", extra=["--inspect", "--inspect-dims", '{"n": 9, "m": 9}']))
        self.assertEqual((doc["load_ok"], doc["classification"], doc["inspect"]), (True, "pass", {}))
        self.assertNotIn("build_ms", doc)


@unittest.skipIf(c_toolchain() is None, "C 컴파일러가 없다(Linux 의 gcc 또는 ziglang 패키지 필요)")
class CAbiTest(TaskWorkerTestBase):
    """C ABI 커널의 두 호출 방식(설계 v3 §5.2)을 이 머신의 C 컴파일러로 만든 ``gather`` 로 확인한다.

    과제 ``gather(n, x, m, idx, out)`` 는 스칼라와 포인터가 번갈아 나오고 포인터 타입이 둘이다: 키워드 모드에서 워커가 만든
    ``ctypes.Structure`` 의 필드 순서·패딩이 C 컴파일러가 본 ``kernel_args.h`` 와 어긋나면 출력이 틀린다."""

    positional: Path
    keyword: Path

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        spec = load_taskspec(cls.tasks["gather"].signatures)
        cls.header = sigs.render_args_struct(spec, "c")
        (cls.tmp / "c_positional").mkdir()
        (cls.tmp / "c_keyword").mkdir()
        cls.positional = build_shared_library(C_GATHER, cls.tmp / "c_positional")
        cls.keyword = build_shared_library(C_GATHER_KEYWORD, cls.tmp / "c_keyword", cls.header)

    def test_positional_call_with_float_and_int32_pointers(self) -> None:
        tests = (*CORRECTNESS, "empty")
        doc = self.finished(*self.run_task("gather", "c", self.positional, "check", tests, workdir=self.workdir()))
        self.assert_check_passed(doc, tests, per_shape_build=False, return_type=None)
        self.assertEqual(doc["backend_detail"], {})

    def test_sanitize_mode_passes_exact_allocations_to_a_c_kernel(self) -> None:
        """``--sanitize`` 의 정확 크기 C 할당(길이 0 배열 포함)으로도 맞는 C 커널은 그대로 통과한다."""
        tests = (*CORRECTNESS, "empty")
        extra = ["--sanitize"]
        doc = self.finished(*self.run_task("gather", "c", self.positional, "check", tests, extra=extra, workdir=self.workdir()))
        self.assert_check_passed(doc, tests, per_shape_build=False, return_type=None)
        self.assertTrue(doc["sanitize"])

    def test_keyword_call_passes_a_pointer_to_the_argument_structure(self) -> None:
        self.assertIn("int n;\n    const float* x;\n    int m;\n    const int* idx;\n    float* out;", self.header)
        extra = ["--args-mode", "keyword"]
        doc = self.finished(*self.run_task("gather", "c", self.keyword, "import", extra=extra, workdir=self.workdir()))
        self.assertEqual((doc["load_ok"], doc["classification"]), (True, "pass"))
        tests = (*CORRECTNESS, "empty")
        doc = self.finished(*self.run_task("gather", "c", self.keyword, "check", tests, extra=extra, workdir=self.workdir()))
        self.assert_check_passed(doc, tests, per_shape_build=False, return_type=None)
        doc = self.finished(*self.run_task("gather", "c", self.keyword, "bench", ("large",), extra=[*extra, *BENCH_ARGS],
                                           workdir=self.workdir()))
        self.assert_bench_passed(doc, "gather", per_shape_build=False)

    def test_missing_symbol_is_a_signature_mismatch_in_both_modes(self) -> None:
        for extra in ([], ["--args-mode", "keyword"]):
            # 과제 명세는 histogram 인데 라이브러리에는 gather 만 있다
            doc = self.finished(*self.run_task("histogram", "c", self.keyword, "import", extra=extra, workdir=self.workdir()))
            self.assertEqual((doc["load_ok"], doc["classification"]), (False, "signature_mismatch"))
            self.assertIn("symbol histogram not found", doc["load_error"])

    def test_relative_artifact_path_is_resolved_against_the_working_directory(self) -> None:
        """파일 이름만 준 산출물: ``dlopen``/``LoadLibrary`` 는 이름만 있으면 현재 디렉터리가 아니라 검색 경로에서 찾는다."""
        doc = self.finished(*self.run_task("gather", "c", Path(self.positional.name), "check", ("tiny",),
                                           workdir=self.positional.parent))
        self.assertEqual((doc["artifact"], doc["load_ok"], doc["classification"]), (self.positional.name, True, "pass"))

    def test_file_that_is_not_a_library_is_a_compile_error(self) -> None:
        bogus = self.workdir() / self.positional.name
        bogus.write_text("this is not a shared library\n", encoding="utf-8")
        doc = self.finished(*self.run_task("gather", "c", bogus, "import"))
        self.assert_load_failed(doc, "compile_error", doc["load_error"])
        self.assertTrue(doc["load_error"].startswith("OSError"), doc["load_error"])


@requires_tool("rustc")
class RustAbiTest(TaskWorkerTestBase):
    """Rust 산출물(``cdylib``)은 C ABI 공유 라이브러리라 ``--backend c`` 로 실행한다(설계 v3 §5.1). 키워드 모드에서는 C 와 같은
    ``ctypes.Structure`` 의 포인터가 ``#[repr(C)]`` 구조체(``sigs`` 가 주는 ``kernel_args.rs``)로 읽혀야 한다."""

    def test_cdylib_reads_the_argument_structure(self) -> None:
        build_dir = self.workdir()
        spec = load_taskspec(self.tasks["gather"].signatures)
        (build_dir / "kernel.rs").write_text(RS_GATHER_KEYWORD, encoding="utf-8", newline="\n")
        (build_dir / sigs.RUST_ARGS_MODULE_NAME).write_text(sigs.render_args_struct(spec, "rust"), encoding="utf-8", newline="\n")
        library = build_dir / ("kernel.dll" if os.name == "nt" else "kernel.so")
        # 컴파일러는 현재 환경을 그대로 물려받는다(MSVC 툴체인은 링커를 환경변수로 찾는다)
        built = run_isolated(["rustc", "--edition", "2021", "--crate-type", "cdylib", "-C", "opt-level=2", "-C", "panic=abort",
                              "-o", library.name, "kernel.rs"], timeout_s=JIT_TIMEOUT_S, cwd=build_dir)
        self.assertEqual(built.rc, 0, built.stderr_tail)
        tests = (*CORRECTNESS, "empty")
        doc = self.finished(*self.run_task("gather", "c", library, "check", tests, extra=["--args-mode", "keyword"]))
        self.assert_check_passed(doc, tests, per_shape_build=False, return_type=None)


@requires_modules("numba")
class NumbaKernelTest(TaskWorkerTestBase):
    """손으로 쓴 Numba 커널(``tests/fixtures/kernels/numba``)을 실제 워커로 돌린다(설계 v3 §5.1–5.2)."""

    # (과제, 인자 모드, 워커 모드): 과제마다, 인자 모드마다 세 워커 모드를 한 번씩은 지난다
    MATRIX = (
        ("gemm", "positional", ("import", "check", "bench")), ("gemm", "keyword", ("check",)),
        ("jacobi2d", "keyword", ("import", "check", "bench")), ("jacobi2d", "positional", ("check",)),
        ("dot", "positional", ("import", "check", "bench")), ("dot", "keyword", ("check",)),
    )
    PARALLEL = frozenset({"gemm", "dot"})       # parallel=True 로 컴파일되는 픽스처

    def assert_inspect(self, doc: dict[str, Any], *, parallel: bool) -> None:
        info = doc["inspect"]
        self.assertEqual(sorted(info), ["num_threads", "parallel_diagnostics", "signatures", "threading_layer"])
        self.assertEqual(len(info["signatures"]), 1)
        self.assertIn("float32", info["signatures"][0])
        self.assertEqual(info["num_threads"], THREADS)
        self.assertEqual(info["threading_layer"], NUMBA_THREADING_LAYER)
        if parallel:
            self.assertIn("prange", info["parallel_diagnostics"])
            self.assertLessEqual(len(info["parallel_diagnostics"]), 8000)
        else:
            self.assertEqual(info["parallel_diagnostics"], "", "parallel=True 가 아닌 함수에는 병렬 진단이 없다")

    def test_real_kernels_in_every_mode(self) -> None:
        for task, args_mode, modes in self.MATRIX:
            artifact = self.fixture(f"numba/{task}.py", task, keyword=args_mode == "keyword")
            extra = ["--args-mode", args_mode, *NUMBA_ARGS]
            for mode in modes:
                with self.subTest(task=task, args_mode=args_mode, mode=mode):
                    if mode == "import":
                        # gemm 에는 성능 형태의 dim 도 준다: numba 의 관찰값은 크기와 무관하다
                        dims = ["--inspect-dims", json.dumps(TASK_SHAPES[task][3][1])] if task == "gemm" else []
                        doc = self.finished(*self.run_task(task, "numba", artifact, "import", extra=[*extra, "--inspect", *dims]))
                        self.assertEqual((doc["backend"], doc["backend_detail"]), ("numba", {"threads": THREADS}))
                        self.assertEqual((doc["load_ok"], doc["classification"]), (True, "pass"))
                        self.assertNotIn("build_ms", doc)
                        self.assert_inspect(doc, parallel=task in self.PARALLEL)
                    elif mode == "check":
                        doc = self.finished(*self.run_task(task, "numba", artifact, "check", CORRECTNESS, extra=extra))
                        self.assert_check_passed(doc, CORRECTNESS, per_shape_build=False, return_type="NoneType")
                    else:
                        doc = self.finished(*self.run_task(task, "numba", artifact, "bench", ("large",), extra=[*extra, *BENCH_ARGS]))
                        self.assert_bench_passed(doc, task, per_shape_build=False)

    def test_int32_task_with_a_zero_length_input(self) -> None:
        artifact = self.fixture("numba/histogram.py", "histogram")
        tests = (*CORRECTNESS, "empty")
        doc = self.finished(*self.run_task("histogram", "numba", artifact, "check", tests, extra=NUMBA_ARGS))
        self.assert_check_passed(doc, tests, per_shape_build=False, return_type="NoneType")
        self.assertEqual((doc["tests"][-1]["dims"], doc["max_abs_err"]), (EMPTY_DIMS["histogram"], 0.0))
        doc = self.finished(*self.run_task("histogram", "numba", artifact, "bench", ("large",), extra=[*NUMBA_ARGS, *BENCH_ARGS]))
        self.assert_bench_passed(doc, "histogram", per_shape_build=False)

    def test_function_without_an_eagerly_compiled_signature_is_a_signature_mismatch(self) -> None:
        """``@njit`` 만 쓴 지연 컴파일(첫 호출에 JIT 가 섞인다)과 Numba 함수가 아닌 것: ``numba_signature_missing``."""
        for name, code in (("lazy @njit", NUMBA_LAZY), ("plain Python function", PY_CORRECT)):
            with self.subTest(kernel=name):
                py = self.write_py(code)
                doc = self.finished(*self.run_task("gemm", "numba", py, "import", extra=[*NUMBA_ARGS, "--inspect"]))
                self.assert_load_failed(doc, "signature_mismatch", "numba_signature_missing")
                self.assertNotIn("exception", doc)
        # check 모드에서도 같은 판정이다: 지연 컴파일 커널을 호출해 첫 호출에서 컴파일되게 두지 않는다
        doc = self.finished(*self.run_task("gemm", "numba", self.write_py(NUMBA_LAZY), "check", CORRECTNESS, extra=NUMBA_ARGS))
        self.assert_load_failed(doc, "signature_mismatch", "numba_signature_missing")

    def test_signature_that_does_not_accept_the_arguments_is_a_runtime_crash(self) -> None:
        py = self.write_py(NUMBA_FLOAT64_SIGNATURE)
        doc = self.finished(*self.run_task("gemm", "numba", py, "import", extra=NUMBA_ARGS))
        self.assertEqual((doc["load_ok"], doc["classification"]), (True, "pass"))
        doc = self.finished(*self.run_task("gemm", "numba", py, "check", CORRECTNESS, extra=NUMBA_ARGS))
        self.assertEqual(doc["classification"], "runtime_crash")
        self.assertTrue(doc["error_detail"].startswith("tiny: TypeError: "), doc["error_detail"])
        self.assertEqual(len(doc["tests"]), 1, "첫 테스트의 실패 뒤 나머지는 돌리지 않는다")
        self.assertEqual(doc["tests"][0]["runs"][0]["exception"]["type"], "TypeError")

    def test_compilation_failure_at_import_is_a_compile_error(self) -> None:
        py = self.write_py(NUMBA_TYPING_ERROR)
        doc = self.finished(*self.run_task("gemm", "numba", py, "import", extra=[*NUMBA_ARGS, "--inspect"]))
        self.assert_load_failed(doc, "compile_error", doc["load_error"])
        self.assertTrue(doc["load_error"].startswith("TypingError: "), doc["load_error"])
        self.assertIn("undefined_helper", doc["load_error"], "여러 줄 오류의 원인 줄까지 한 줄에 들어간다")
        self.assertNotIn("\n", doc["load_error"])
        self.assertEqual(doc["exception"]["type"], "TypingError")
        self.assertIn("undefined_helper", doc["exception"]["message"])

    def test_unusable_threading_layer_is_a_harness_error_not_a_kernel_error(self) -> None:
        """스레딩 계층을 로드할 수 없으면 parallel=True 커널의 임포트가 ValueError 로 실패한다. 도구 문제이므로 커널을
        임포트하기 전에 worker_error 로 끝나야 한다(compile_error 행이 되면 안 된다)."""
        py = self.fixture("numba/gemm.py", "gemm")
        proc, doc = self.run_task("gemm", "numba", py, "import", extra=NUMBA_ARGS, env={"NUMBA_THREADING_LAYER": "no_such_layer"})
        self.assertEqual(proc.rc, 3, proc.stderr_tail)
        self.assertEqual((doc["complete"], doc["load_ok"], doc["classification"]), (False, None, None))
        self.assertIn("no_such_layer", doc["worker_error"])


@requires_modules("halide")
class HalideKernelTest(TaskWorkerTestBase):
    """손으로 쓴 Halide 파이프라인 빌드 함수(``tests/fixtures/kernels/halide``)를 실제 워커로 돌린다(설계 v3 §5.3)."""

    # (과제, 픽스처, halide_mode, 인자 모드, 워커 모드): 과제·halide_mode·인자 모드마다 세 워커 모드를 한 번씩은 지난다
    MATRIX = (
        ("gemm", "gemm", "auto", "positional", ("import", "check", "bench")), ("gemm", "gemm", "manual", "keyword", ("check",)),
        ("jacobi2d", "jacobi2d", "auto", "keyword", ("import", "check", "bench")),
        ("jacobi2d", "jacobi2d_manual", "manual", "positional", ("check",)),
        ("dot", "dot", "manual", "positional", ("import", "check", "bench")), ("dot", "dot", "auto", "keyword", ("check",)),
    )

    @staticmethod
    def options(halide_mode: str, args_mode: str = "positional") -> list[str]:
        return ["--halide-mode", halide_mode, "--threads", str(THREADS), "--args-mode", args_mode]

    def assert_inspect(self, info: dict[str, Any], halide_mode: str, dims: dict[str, int]) -> None:
        self.assertEqual(sorted(info), ["autoschedule_ms", "autoscheduler", "dims", "jit_ms", "loop_nest", "schedule_source", "target"])
        self.assertEqual(info["dims"], dims)
        self.assertIn("produce ", info["loop_nest"])
        self.assertTrue(info["target"])
        self.assertGreater(info["jit_ms"], 0.0)
        if halide_mode == "auto":
            self.assertEqual(info["autoscheduler"], "Mullapudi2016")
            self.assertIn("pipeline.get_func", info["schedule_source"])
            self.assertGreater(info["autoschedule_ms"], 0.0)
        else:
            self.assertEqual((info["autoscheduler"], info["schedule_source"], info["autoschedule_ms"]), (None, "", None))

    def test_real_kernels_in_every_mode(self) -> None:
        for task, name, halide_mode, args_mode, modes in self.MATRIX:
            artifact = self.fixture(f"halide/{name}.py", task, keyword=args_mode == "keyword")
            extra = self.options(halide_mode, args_mode)
            detail: dict[str, Any] = {"threads": THREADS, "halide_mode": halide_mode, "autoscheduler": None}
            if halide_mode == "auto":
                detail.update({"autoscheduler": "Mullapudi2016", "autoscheduler_params": {"parallelism": str(THREADS)}})
            for mode in modes:
                with self.subTest(task=task, halide_mode=halide_mode, args_mode=args_mode, mode=mode):
                    if mode == "import":
                        tiny = TASK_SHAPES[task][0][1]
                        doc = self.finished(*self.run_task(task, "halide", artifact, "import", ("tiny",), extra=[*extra, "--inspect"]))
                        self.assertEqual((doc["backend"], doc["backend_detail"]), ("halide", detail))
                        self.assertEqual((doc["load_ok"], doc["classification"], doc["test"], doc["dims"]), (True, "pass", "tiny", tiny))
                        self.assertGreater(doc["build_ms"], 0.0)
                        self.assert_inspect(doc["inspect"], halide_mode, tiny)      # --inspect-dims 가 없으면 첫 정확성 테스트 크기
                    elif mode == "check":
                        doc = self.finished(*self.run_task(task, "halide", artifact, "check", CORRECTNESS, extra=extra))
                        self.assert_check_passed(doc, CORRECTNESS, per_shape_build=True, return_type=None)
                        self.assertEqual(doc["backend_detail"], detail)
                    else:
                        doc = self.finished(*self.run_task(task, "halide", artifact, "bench", ("large",), extra=[*extra, *BENCH_ARGS]))
                        self.assert_bench_passed(doc, task, per_shape_build=True)

    def test_int32_task_with_a_zero_length_input(self) -> None:
        artifact = self.fixture("halide/histogram.py", "histogram")
        tests = (*CORRECTNESS, "empty")
        doc = self.finished(*self.run_task("histogram", "halide", artifact, "check", tests, extra=self.options("auto")))
        self.assert_check_passed(doc, tests, per_shape_build=True, return_type=None)
        self.assertEqual((doc["tests"][-1]["dims"], doc["max_abs_err"]), (EMPTY_DIMS["histogram"], 0.0))
        doc = self.finished(*self.run_task("histogram", "halide", artifact, "bench", ("large",),
                                           extra=[*self.options("manual"), *BENCH_ARGS]))
        self.assert_bench_passed(doc, "histogram", per_shape_build=True)

    def test_compile_warnings_go_to_stderr_not_stdout(self) -> None:
        """Halide 는 컴파일 경고를 파이썬 stdout 으로 낸다(실측). 워커의 stdout 은 커널이 쓴 것으로 기록되므로(contract_violation)
        하네스가 부른 컴파일 단계의 경고는 stderr 로 돌린다."""
        # 수동 스케줄에서 순수 정의만 스케줄하고 update 정의를 빠뜨린 커널: 결과는 맞고 Halide 가 경고한다
        py = self.write_py(patched(HL_GEMM, "    return C\n", "    C.parallel(i)\n    return C\n"))
        proc, doc = self.run_task("gemm", "halide", py, "import", ("tiny",), extra=[*HALIDE_MANUAL, "--inspect"])
        doc = self.finished(proc, doc)      # stdout 이 비어 있다
        self.assertEqual((doc["load_ok"], doc["classification"]), (True, "pass"))
        self.assertIn("Warning: Update definition 0 of function C has not been scheduled", proc.stderr_tail)
        self.assertIn("parallel i", doc["inspect"]["loop_nest"])

    def test_inspect_dims_describe_the_pipeline_built_for_the_performance_shape(self) -> None:
        """오토스케줄러의 스케줄은 estimate(크기)에 따라 달라진다: ``--inspect-dims`` 를 주면 그 크기로 만든 파이프라인을 기술한다."""
        artifact = self.fixture("halide/gemm.py", "gemm")
        large = {"M": 512, "N": 512, "K": 512}
        extra = [*self.options("auto"), "--halide-params", '{"last_level_cache_size": 8388608}', "--inspect",
                 "--inspect-dims", json.dumps(large)]
        doc = self.finished(*self.run_task("gemm", "halide", artifact, "import", ("tiny",), extra=extra))
        self.assertEqual(doc["backend_detail"]["autoscheduler_params"], {"parallelism": str(THREADS), "last_level_cache_size": "8388608"})
        self.assertEqual((doc["load_ok"], doc["classification"], doc["dims"]), (True, "pass", TASK_SHAPES["gemm"][0][1]))
        info = doc["inspect"]
        self.assert_inspect(info, "auto", large)
        self.assertIn("for k in [0, 511]", info["loop_nest"], "루프 중첩은 K = 512 로 만든 파이프라인의 것이어야 한다")
        self.assertIn("vectorize", info["schedule_source"])


@requires_modules("halide")
class HalideFailureTest(TaskWorkerTestBase):
    """Halide 백엔드의 실패 규칙(설계 v3 §5.3): 어느 것이 커널의 오류이고 어느 것이 하네스·도구의 문제인가."""

    LARGE = ("--inspect", "--inspect-dims", '{"M": 512, "N": 512, "K": 512}')       # 성능 형태로 다시 빌드하는 관찰

    def import_doc(self, code: str, extra: list[str]) -> dict[str, Any]:
        return self.finished(*self.run_task("gemm", "halide", self.write_py(code), "import", ("tiny",), extra=extra))

    def test_returned_object_that_is_not_the_output_func_is_a_signature_mismatch(self) -> None:
        for code, detail in ((HL_WRONG_NDIM, "gemm returned an hl.Func with 1 dimension(s); the output C has 2"),
                             (HL_WRONG_TYPE, "gemm returned an hl.Func of type float64; the output C is float32"),
                             (HL_RETURNS_NONE, "gemm returned NoneType; expected hl.Func")):
            with self.subTest(detail=detail):
                doc = self.import_doc(code, [*HALIDE_AUTO, "--inspect"])
                self.assert_load_failed(doc, "signature_mismatch", detail)
                self.assertNotIn("exception", doc)

    def test_error_while_defining_the_pipeline_is_a_compile_error(self) -> None:
        doc = self.import_doc(HL_DEFINITION_ERROR, HALIDE_MANUAL)
        self.assert_load_failed(doc, "compile_error", doc["load_error"])
        self.assertTrue(doc["load_error"].startswith("HalideError: "), doc["load_error"])
        self.assertEqual((doc["test"], doc["exception"]["type"]), ("tiny", "HalideError"))
        self.assertIn("kernel.py", doc["exception"]["traceback"])

    def test_schedule_under_auto_mode_is_a_compile_error_with_the_autoscheduler_message(self) -> None:
        # update 정의를 병렬로 표시한 gemm, 스텝마다 compute_root 한 jacobi2d(수동 모드의 픽스처)
        for task, code in (("gemm", HL_SCHEDULED), ("jacobi2d", (FIXTURES / "halide" / "jacobi2d_manual.py").read_text(encoding="utf-8"))):
            with self.subTest(task=task):
                doc = self.finished(*self.run_task(task, "halide", self.write_py(code), "import", ("tiny",), extra=HALIDE_AUTO))
                self.assert_load_failed(doc, "compile_error", doc["load_error"])
                self.assertTrue(doc["load_error"].startswith("autoscheduler Mullapudi2016 failed: HalideError: "), doc["load_error"])
                self.assertIn("cannot auto-schedule function", doc["load_error"])
                self.assertEqual(doc["exception"]["type"], "HalideError")
        # 같은 커널이 수동 모드에서는 로드된다
        doc = self.import_doc(HL_SCHEDULED, HALIDE_MANUAL)
        self.assertEqual((doc["load_ok"], doc["classification"]), (True, "pass"))

    def test_build_failure_for_one_shape_is_a_runtime_crash_of_that_test(self) -> None:
        py = self.write_py(HL_FAILS_FOR_ONE_SHAPE)
        doc = self.finished(*self.run_task("gemm", "halide", py, "check", CORRECTNESS, extra=HALIDE_AUTO))
        self.assertEqual((doc["load_ok"], doc["classification"], doc["error_detail"]),
                         (True, "runtime_crash", "rect: ValueError: no pipeline for M == 20"))
        self.assertEqual([(t["name"], t["classification"]) for t in doc["tests"]], [("tiny", "pass"), ("rect", "runtime_crash")])
        rect = doc["tests"][1]
        self.assertNotIn("build_ms", rect)
        self.assertEqual(len(rect["runs"]), 1)
        run = rect["runs"][0]
        self.assertEqual((run["classification"], run["pattern"]), ("runtime_crash", "ValueError: no pipeline for M == 20"))
        self.assertEqual((run["exception"]["type"], run["exception"]["message"]), ("ValueError", "no pipeline for M == 20"))
        self.assertIn("kernel.py", run["exception"]["traceback"])

    def test_build_failure_in_bench_mode_is_a_runtime_crash(self) -> None:
        py = self.write_py(HL_FAILS_FOR_ONE_SHAPE.replace("M == 20", "M == 48"))      # 조건과 메시지 두 곳: bench 형태(large)에서 실패
        doc = self.finished(*self.run_task("gemm", "halide", py, "bench", ("large",), extra=[*HALIDE_MANUAL, *BENCH_ARGS]))
        self.assertEqual((doc["classification"], doc["error_detail"]), ("runtime_crash", "ValueError: no pipeline for M == 48"))
        self.assertEqual((doc["exception"]["type"], doc["warmup_ms"], doc["timed_ms"], doc["final_check"]), ("ValueError", [], [], None))
        self.assertNotIn("build_ms", doc)

    def test_error_while_running_the_pipeline_is_a_runtime_crash(self) -> None:
        py = self.write_py(HL_READS_OUT_OF_BOUNDS)
        doc = self.finished(*self.run_task("gemm", "halide", py, "check", CORRECTNESS, extra=HALIDE_MANUAL))
        self.assertEqual(doc["classification"], "runtime_crash")
        # 한 줄이어야 한다(행의 error_detail 로 간다): \Z 는 끝의 줄바꿈도 허용하지 않는다
        self.assertRegex(doc["error_detail"], r"\Atiny: HalideError: Error: Input buffer A is accessed at .* in dimension 0\Z")
        tiny = doc["tests"][0]
        self.assertGreater(tiny["build_ms"], 0.0, "빌드는 됐고 호출이 실패했다")
        self.assertEqual(tiny["runs"][0]["exception"]["type"], "HalideError")
        self.assertTrue(tiny["runs"][0]["exception"]["message"].endswith("\n"), "전제: Halide 의 메시지는 줄바꿈으로 끝난다")

    def test_unknown_autoscheduler_is_reported_before_the_kernel_is_imported(self) -> None:
        py = self.write_py("raise RuntimeError('the kernel must not be imported')\n")
        doc = self.finished(*self.run_task("gemm", "halide", py, "import", ("tiny",),
                                           extra=[*HALIDE_AUTO, "--halide-autoscheduler", "Nope2020"]))
        self.assert_load_failed(doc, "compile_error", "halide_autoscheduler_unavailable:Nope2020")
        self.assertEqual(doc["backend_detail"]["autoscheduler"], "Nope2020")
        self.assertNotIn("exception", doc)
        # 수동 모드는 오토스케줄러를 쓰지 않는다: 같은 이름을 줘도 커널까지 간다
        doc = self.finished(*self.run_task("gemm", "halide", py, "import", ("tiny",),
                                           extra=[*HALIDE_MANUAL, "--halide-autoscheduler", "Nope2020"]))
        self.assert_load_failed(doc, "compile_error", "RuntimeError: the kernel must not be imported")

    def test_bad_autoscheduler_parameters_are_a_harness_error_not_a_kernel_error(self) -> None:
        """틀린 오토스케줄러 매개변수(설정 오류)는 LLM 코드를 실행하기 전의 시험 적용에서 드러난다. Halide 22.1 은 예외 대신
        프로세스를 중단시키기도 한다 — 어느 쪽이든 결과 문서에 worker_error 가 남아야 한다(커널의 크래시로 읽히면 안 된다)."""
        py = self.write_py("raise RuntimeError('the kernel must not be imported')\n")
        proc, doc = self.run_task("gemm", "halide", py, "import", ("tiny",), extra=[*HALIDE_AUTO, "--halide-params", '{"no_such_parameter": 1}'])
        self.assertNotEqual(proc.rc, 0)
        self.assertEqual((doc["complete"], doc["load_ok"], doc["classification"]), (False, None, None))
        self.assertTrue(doc["worker_error"])

    def test_import_without_a_test_shape_is_a_harness_error(self) -> None:
        """halide 의 import 는 첫 정확성 테스트 크기로 빌드해 본다: ``--tests`` 없이 부른 것은 오케스트레이터의 잘못이다."""
        proc, doc = self.run_task("gemm", "halide", self.write_py(HL_GEMM), "import", extra=HALIDE_MANUAL)
        self.assertEqual(proc.rc, 3, proc.stderr_tail)
        self.assertFalse(doc["complete"])
        self.assertIn("--tests", doc["worker_error"])

    def test_failure_while_inspecting_never_fails_the_load(self) -> None:
        for name, code, extra, error in (
                ("the build function raises for the performance shape", HL_FAILS_WHEN_LARGE, list(self.LARGE),
                 "KernelError: compile_error: ValueError: too large to build"),
                ("the dims do not name every dim argument", HL_GEMM, ["--inspect", "--inspect-dims", '{"M": 512}'],
                 "ValueError: inspect dims lack ['N', 'K']")):
            with self.subTest(case=name):
                doc = self.import_doc(code, [*HALIDE_AUTO, *extra])
                self.assertEqual((doc["load_ok"], doc["classification"], doc["load_error"]), (True, "pass", None))
                info = doc["inspect"]
                self.assertEqual(info["inspect_error"], error)
                self.assertEqual((info["loop_nest"], info["autoscheduler"]), ("", "Mullapudi2016"))
                self.assertGreater(info["jit_ms"], 0.0)

    def test_process_death_while_inspecting_leaves_a_loaded_document(self) -> None:
        """성능 형태의 빌드가 워커를 끝내면(여기서는 ``os._exit``; 실제로는 시간 초과·중단) 문서에는 로드 성공과
        "수집이 끝나지 않았다"가 남는다: 오케스트레이터가 로드 실패와 구별할 수 있어야 한다."""
        proc, doc = self.run_task("gemm", "halide", self.write_py(HL_EXITS_WHEN_LARGE), "import", ("tiny",), extra=[*HALIDE_AUTO, *self.LARGE])
        self.assertEqual(proc.rc, 7)
        self.assertEqual((doc["complete"], doc["load_ok"], doc["classification"]), (False, True, "pass"))
        self.assertEqual(doc["in_progress"], {"test": worker.INSPECT_STEP, "call_idx": None})
        self.assertEqual(doc["inspect"], {"inspect_error": worker.INSPECT_UNFINISHED})


# ----------------------------------------------------------------------------- 프로세스 안에서 확인하는 순수 부분


class BuffersTest(unittest.TestCase):
    def test_zero_length_arrays_are_aligned_and_guarded(self) -> None:
        """길이 0 인 조각은 NumPy 가 데이터 포인터를 옮기지 않는다: 정렬된 주소에서 시작해야 하고 가드는 바로 뒤에 온다."""
        for dtype in (np.float32, np.int32):
            for n in (0, 1, 7):
                arr = worker.alloc_aligned(n, 8, -2048.0, dtype)
                self.assertEqual((arr.view.size, arr.view.dtype, arr.guard.size), (n, np.dtype(dtype), 8))
                self.assertEqual(arr.view.ctypes.data % worker.ALIGNMENT_BYTES, 0)
                self.assertEqual(arr.guard.ctypes.data, arr.view.ctypes.data + n * np.dtype(dtype).itemsize)
                self.assertEqual((arr.full.size, arr.full.ctypes.data), (n + 8, arr.view.ctypes.data))
                self.assertTrue(np.all(arr.guard == -2048))

    def test_buffers_of_a_task_with_zero_length_arrays(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worker_test_") as tmp:
            tasks = make_tasks(Path(tmp))
            spec = load_taskspec(tasks["gather"].signatures)
            path = tasks["gather"].tests("empty")[0]
            self.assertEqual(worker.load_dims(path, spec), EMPTY_DIMS["gather"])
            bufs = worker.Buffers(spec, worker.load_test(path, spec), -2048.0)
        self.assertEqual({name: view.shape for name, view in bufs.views.items()}, {"x": (4,), "idx": (0,), "out": (0,)})
        self.assertEqual((bufs.dims, bufs.ref.shape, bufs.ref.dtype), (EMPTY_DIMS["gather"], (0,), np.dtype(np.float64)))
        self.assertTrue(all(view.ctypes.data % worker.ALIGNMENT_BYTES == 0 for view in bufs.views.values()))
        bufs.restore_inputs()
        bufs.fill_sentinel()
        self.assertEqual((bufs.inputs_mutated(), bufs.oob_write()), (False, False))
        bufs.out.full[0] = 1.0      # 길이 0 인 출력에 한 칸이라도 쓰면 가드를 넘은 것이다
        self.assertTrue(bufs.oob_write())

    def test_sanitize_arrays_are_exact_aligned_c_allocations_without_a_guard(self) -> None:
        """``--sanitize``: 배열마다 정확히 n 원소의 C 할당(64바이트 정렬 시작). 블록의 끝 = 마지막 원소의 끝이라
        ASan 이 한 칸 넘는 접근을 본다. 마지막 참조가 사라지면 블록을 해제한다."""
        alloc, free = worker.c_aligned_allocator()
        freed: list[int] = []

        def spy_free(ptr: int) -> None:
            freed.append(ptr)
            free(ptr)

        with mock.patch.object(worker, "c_aligned_allocator", return_value=(alloc, spy_free)):
            for dtype in (np.float32, np.int32):
                for n in (0, 1, 7, 1000):
                    arr = worker.alloc_exact(n, dtype)
                    nbytes = n * np.dtype(dtype).itemsize
                    ptr = arr.view.ctypes.data
                    self.assertEqual((arr.view.size, arr.view.dtype, arr.guard.size, arr.offset), (n, np.dtype(dtype), 0, 0))
                    self.assertEqual(ptr % worker.ALIGNMENT_BYTES, 0)
                    self.assertTrue(arr.view.flags.writeable and arr.view.flags.c_contiguous)
                    block = arr.view.base           # 할당 블록 그 자체(앞 여유 없음, 크기 = n 원소)
                    self.assertEqual((ctypes.addressof(block), ctypes.sizeof(block)), (ptr, max(nbytes, 1)))
                    self.assertEqual((arr.full.size, arr.full.ctypes.data), (n, ptr))
                    arr.view[...] = 3
                    del arr, block
                    gc.collect()
                    self.assertEqual(freed[-1], ptr, "마지막 참조가 사라지면 해제한다")
        self.assertEqual(len(freed), 8)

    def test_sanitize_buffers_keep_the_checks_that_do_not_need_a_guard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="worker_test_") as tmp:
            path = Path(tmp) / "rect.npz"
            _make_npz(path, 20, 13, 31, 2)
            spec = load_taskspec(worker.DEFAULT_SIGNATURES)
            bufs = worker.Buffers(spec, worker.load_test(path, spec), -2048.0, sanitize=True)
        self.assertEqual({name: arr.view.size for name, arr in bufs.arrays.items()}, {"A": 20 * 31, "B": 31 * 13, "C": 20 * 13})
        self.assertTrue(all(arr.guard.size == 0 and arr.full.size == arr.n for arr in bufs.arrays.values()))
        self.assertTrue(all(view.ctypes.data % worker.ALIGNMENT_BYTES == 0 for view in bufs.views.values()))
        bufs.restore_inputs()
        bufs.fill_sentinel()
        self.assertTrue(np.all(bufs.out.view == -2048.0), "출력의 센티널 채움은 그대로다")
        self.assertEqual((bufs.inputs_mutated(), bufs.oob_write()), (False, False))
        bufs.views["A"][0, 0] += 1.0
        self.assertTrue(bufs.inputs_mutated())
        bufs.restore_inputs()
        self.assertFalse(bufs.inputs_mutated())


class WorkerCliTest(unittest.TestCase):
    """명령줄 규칙(설계 v3 §5.1). 틀린 조합은 argparse 의 사용법 오류(rc=2)다: 결과 문서를 쓰기 전에 끝난다."""

    BASE = ("--artifact", "kernel.py", "--mode", "import", "--out", "import.json")

    def parse(self, *argv: str) -> Any:
        return worker.parse_args([*self.BASE, *argv])

    def usage_error(self, *argv: str) -> str:
        said = io.StringIO()
        with self.assertRaises(SystemExit) as ctx, contextlib.redirect_stderr(said):
            self.parse(*argv)
        self.assertEqual(ctx.exception.code, 2)
        return said.getvalue()

    def test_backends_and_defaults(self) -> None:
        for backend in ("c", "python", "numba"):
            ns = self.parse("--backend", backend)
            self.assertEqual((ns.backend, ns.args_mode, ns.threads, ns.inspect, ns.inspect_dims), (backend, "positional", None, False, None))
            self.assertEqual((ns.halide_mode, ns.halide_autoscheduler, ns.halide_params), (None, "Mullapudi2016", {}))
        self.assertIn("--backend", self.usage_error("--backend", "rust"))      # rust 산출물은 --backend c 로 실행한다
        self.assertEqual(self.parse("--backend", "c", "--args-mode", "keyword").args_mode, "keyword")
        self.assertIn("--args-mode", self.usage_error("--backend", "c", "--args-mode", "named"))

    def test_halide_needs_a_mode_and_auto_needs_the_thread_count(self) -> None:
        self.assertIn("--halide-mode", self.usage_error("--backend", "halide"))
        self.assertIn("--threads", self.usage_error("--backend", "halide", "--halide-mode", "auto"))
        self.assertIn("--halide-mode", self.usage_error("--backend", "halide", "--halide-mode", "fast", "--threads", "4"))
        self.assertIsNone(self.parse("--backend", "halide", "--halide-mode", "manual").threads)
        ns = self.parse("--backend", "halide", "--halide-mode", "auto", "--threads", "4", "--halide-autoscheduler", "Adams2019")
        self.assertEqual((ns.halide_mode, ns.threads, ns.halide_autoscheduler), ("auto", 4, "Adams2019"))
        self.assertIn("--threads", self.usage_error("--backend", "numba", "--threads", "0"))

    def test_json_options_must_be_objects_of_the_right_shape(self) -> None:
        ns = self.parse("--backend", "halide", "--halide-mode", "auto", "--threads", "2", "--halide-params",
                        '{"balance": 40, "last_level_cache_size": "16777216", "x": 1.5}', "--inspect", "--inspect-dims", '{"M": 1024, "N": 8}')
        self.assertEqual(ns.halide_params, {"balance": 40, "last_level_cache_size": "16777216", "x": 1.5})
        self.assertEqual((ns.inspect, ns.inspect_dims), (True, {"M": 1024, "N": 8}))
        for bad in ("[1, 2]", "balance=40", '{"balance": true}', '{"balance": [40]}', '{"balance": null}'):
            self.assertIn("--halide-params", self.usage_error("--backend", "halide", "--halide-mode", "manual", "--halide-params", bad))
        for bad in ('"M"', "{M: 3}", '{"M": 1.5}', '{"M": true}', '{"M": "1024"}'):
            self.assertIn("--inspect-dims", self.usage_error("--backend", "numba", "--inspect", "--inspect-dims", bad))


class _RecordingHandle(kernels.KernelHandle):
    """``inspect`` 가 불린 시점에 디스크에 있던 결과 문서를 관찰값으로 돌려준다."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def prepare(self, dims: Any, arrays: Any = None) -> float | None:
        return None

    def invoke(self) -> Any:
        return None

    def _collect(self, info: dict[str, Any], dims: Any) -> None:
        info["on_disk"] = json.loads(self._path.read_text(encoding="utf-8"))
        info["dims"] = dims


class ResultDocumentTest(unittest.TestCase):
    """결과 문서의 보호 장치: 워커가 도중에 끝나도 디스크의 문서가 "누구 탓인가"를 틀리게 말하지 않는다."""

    def setUp(self) -> None:
        tmpdir = tempfile.TemporaryDirectory(prefix="worker_test_")
        self.addCleanup(tmpdir.cleanup)
        self.path = Path(tmpdir.name) / "import.json"
        self.writer = worker.ResultWriter(self.path, {"complete": False, "load_ok": None, "classification": None})

    def on_disk(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def test_backend_preparation_is_marked_as_a_worker_error_until_it_returns(self) -> None:
        with worker.harness_only(self.writer, "백엔드를 준비하는"):
            self.assertIn("백엔드를 준비하는", self.on_disk()["worker_error"])
        self.assertNotIn("worker_error", self.on_disk())
        # 계약이 정한 커널 쪽 판정(오토스케줄러 없음)은 그대로 올라가고 표시는 지워진다
        with self.assertRaises(kernels.KernelError), worker.harness_only(self.writer, "백엔드를 준비하는"):
            raise kernels.KernelError("compile_error", "halide_autoscheduler_unavailable:Nope2020")
        self.assertNotIn("worker_error", self.writer.doc)
        self.assertNotIn("worker_error", self.on_disk())

    def test_inspect_is_recorded_as_unfinished_until_it_returns(self) -> None:
        self.writer.doc.update({"load_ok": True, "classification": "pass"})
        worker.collect_inspect(_RecordingHandle(self.path), {"n": 3}, self.writer)
        seen = self.writer.doc["inspect"].pop("on_disk")
        self.assertEqual(self.writer.doc["inspect"], {"dims": {"n": 3}})
        self.assertEqual((seen["load_ok"], seen["classification"], seen["complete"]), (True, "pass", False))
        self.assertEqual(seen["in_progress"], {"test": "inspect", "call_idx": None})
        self.assertEqual(seen["inspect"], {"inspect_error": worker.INSPECT_UNFINISHED})


if __name__ == "__main__":
    unittest.main()
