"""워커 (설계 §8, v3 §8): LLM 코드를 실행하는 **유일한** 곳. run.py 가 ``procutil.run_isolated`` 로 띄운다.

    PYTHON_EXE -m harness.worker --backend {c,python,numba,halide} --artifact PATH --mode {import,check,bench}
        --tests tiny.npz,rect.npz,small.npz --tolerance tasks/<task>/tolerance.yaml --signatures tasks/<task>/signatures.yaml --out result.json
        [--check-repeats 2] [--warmup 3 --repeats 10 --min-repeats 3 --budget-s 300] [--args-mode positional|keyword]
        [--halide-mode auto|manual] [--halide-autoscheduler Li2018] [--halide-params JSON] [--threads N]
        [--inspect] [--inspect-dims JSON] [--max-mem-gb 8] [--sanitize]
        [--baseline-backend c|python --baseline-artifact PATH] [--kernel-cpus auto|none|LIST] [--baseline-cpus auto|none|LIST]
        [--no-loop-fsync]

- 시작: ``prctl(PR_SET_DUMPABLE=0)`` (코어 덤프/apport 차단), ``RLIMIT_AS`` (max_mem_gb). 적용값은 ``limits``,
  런타임 BLAS 정체는 ``blas`` 에 기록한다.
- 결과 파일은 매 커널 호출 직전과 매 테스트/호출 뒤에 원자적으로 갱신한다(``complete``, ``in_progress``).
  워커 자체의 예외(하네스 버그)는 ``worker_error`` 로 기록하고 rc=3 으로 끝난다.
- 커널을 로드하고 부르는 방법은 ``harness.kernels`` 가 백엔드별로 감춘다(c: ctypes — Rust 산출물도 C ABI 라 ``--backend c``,
  python·numba: 모듈의 함수, halide: 형태마다 빌드하는 파이프라인). 이 모듈은 프로세스 수준의 일(제한, 버퍼, 결과 문서,
  측정 루프)만 한다. 백엔드 준비(``kernels.open_backend``: LLM 코드를 실행하기 전의 도구 임포트·numba 스레딩 계층 시작·
  오토스케줄러 시험)에서 프로세스가 죽으면 도구·설정 문제이므로, 그 구간 동안 결과 문서에 ``worker_error`` 를 미리 써 둔다
  (``harness_only``).
- 모든 배열은 64바이트 정렬 + 가드 영역(센티널) 위에 만든다. 호출 전 입력을 복원하고 출력을 센티널로 채운다.
  앞·뒤 가드(앞 ``GUARD_FRONT``), 입력 가드 값은 출력 센티널과 다르다(``INPUT_GUARD_*``).
  ``--sanitize`` (ASan 재빌드 실행용)면 배열마다 C 할당자로 정확히 n 원소만 잡는다(64바이트 정렬 시작, 앞 여유·뒤 가드
  없음): 가드가 같은 malloc 블록 안이면 하네스 배열을 조금 넘는 접근이 ASan 에게 정상 접근으로 보이기 때문이다.
  이 모드에서는 가드 검사(``oob_write``)가 늘 False 다 — 판정은 쓰지 않고 새니타이저 보고만 센다.
- stdout 에는 아무것도 쓰지 않는다(커널의 stdout 은 run.py 가 파일로 받아 contract_violation 을 본다). 입력 배열은 덤프하지 않는다.
- 정수(int32) 출력: 참조는 다른 과제처럼 float64 로 읽고 rtol = atol = 0 으로 판정한다(classify, 설계 §17). 가드·센티널
  채움은 dtype 에 맞춘다.
- bench 교차 측정(2026-11-09)·호출 친화도(2026-11-11): ``--baseline-artifact`` 면 기준선을 같은 프로세스에서 번갈아 부르고
  (``baseline_interleaved``), 호출마다 ``--kernel-cpus``/``--baseline-cpus`` 의 집합을 주 스레드에 적용·복원한다(``call_affinity``).
  형태별 빌드(halide)도 커널의 집합에서 한다. 매핑된 libgomp 경로는 전부 ``call_affinity.libgomp_paths`` 에 남긴다.
- 형태별 빌드가 있는 백엔드(halide; v3 §8.4): import 모드는 ``--tests`` 의 첫 테스트 크기로 빌드·JIT 까지 해 보고(실패 →
  ``load_ok=false``), check/bench 는 테스트마다 빌드한 뒤 호출한다. 빌드 시간은 측정에 넣지 않고 ``build_ms`` 로 남긴다.
  check/bench 에서 빌드가 실패하면 그 테스트의 ``runtime_crash`` 다(커널이 예외를 올린 호출과 같은 꼴의 기록).
- ``--inspect`` (import 모드): 컴파일 결과의 관찰값을 ``inspect`` 에 넣는다(numba: 서명·병렬 진단·스레딩 계층·스레드 수,
  halide: 루프 중첩·스케줄 — ``--inspect-dims`` 가 있으면 그 크기(성능 형태)의 것). 수집 실패는 ``inspect.inspect_error`` 에
  남기고 워커는 계속한다. 수집 중에는 ``in_progress.test == "inspect"`` 이고 문서에 이미 ``load_ok=true`` 와 판정이 있다:
  그 상태로 프로세스가 끝났다면(성능 형태의 빌드가 죽거나 시간 상한에 걸림) 로드 실패가 아니라 관찰 실패다.
- 결과 문서의 추가 키(v3): ``backend_detail`` (받은 백엔드 설정), ``inspect``, ``build_ms`` 와 ``ignored_schedule_calls``
  (형태별 빌드가 있는 백엔드만; check 는 테스트별, import·bench 는 최상위 — halide auto 에서 빌드 함수가 부른 스케줄 메서드는
  무시·기록되고 판정에 영향이 없다), 로드 실패가 예외에서 왔으면 ``exception`` (``{type, message, traceback}``), ``sanitize``.
- 호출 예외의 설명(``error_detail``): c·python 은 ``Type: message`` 그대로, numba·halide 는 여러 줄 메시지를 한 줄로 잇는다
  (``crash_detail``). numba 는 로드 때 디스패처가 하네스의 인자 타입을 컴파일 없이 받는지 확인한다(아니면 ``signature_mismatch``).
- ``--max-mem-gb`` 가 0 이하면 ``RLIMIT_AS`` 를 걸지 않는다(ASan 실행용; ``limits.disabled``).
- 비-Linux(개발용 Windows; 설계 v3 §0): ``resource``·``prctl``·``os.getloadavg``·``os.sched_getaffinity`` 가 없어도
  임포트·실행된다(제한은 걸리지 않고 ``limits.unsupported``, 값은 None/빈 리스트). Linux 동작은 그대로다.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import functools
import json
import math
import os
import sys
import time
import traceback
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import resource
except ImportError:  # 비-Linux(Windows): RLIMIT_AS 를 걸 수 없다 → apply_memory_limit 이 unsupported 로 기록한다
    resource = None  # type: ignore[assignment]

from harness import kernels
from harness.classify import Tolerance, aggregate_sample, classify_task, effective_tolerance, load_tolerance
from harness.paths import atomic_write_json
from harness.protocol import blas_identity
from harness.taskspec import TaskSpec, load_taskspec

_LOG = "[worker]"

WORKER_ERROR_RC = 3
PR_SET_DUMPABLE = 4
ALIGNMENT_BYTES = 64
GUARD_INPUT = 1024
GUARD_OUTPUT_EXTRA = 1024
GUARD_FRONT = 4096  # 앞쪽 가드(원소 수): 음수 인덱스 쓰기(예: hist[-2048]) 탐지용 (2026-11-08)
# 입력 가드 값은 출력 센티널과 다르게 둔다: 흘러든 가드가 센티널로 오인되지 않고, 정수 입력 가드는 인덱스로 쓰이면 반드시 매핑되지 않은
# 주소를 쳐 결정적으로 크래시하게(off-by-one 읽기 → hist[data[n]]) 큰 음수를 쓴다.
INPUT_GUARD_FLOAT = -3.0e9
INPUT_GUARD_INT = -1073741824  # INT32_MIN / 2
# worker_error 트레이스백 꼬리(호출 예외의 꼬리 2000 자는 kernels.TRACEBACK_TAIL_CHARS; 이 값은 그 두 배)
WORKER_ERROR_TAIL_CHARS = 4000
BUDGET_FRACTION = 0.8
# 교차 측정 기준선의 모듈 이름(커널 모듈 "kernel" 과 달라야 한다: 둘이 한 프로세스에 함께 로드된다; 2026-11-09)
BASELINE_MODULE_NAME = "baseline_kernel"
# --kernel-cpus/--baseline-cpus 'auto' 에서 고정 코어 전체를 받는 백엔드(OpenMP 런타임을 쓰지 않는 스레드 코드).
# c 와 numba 는 None(건드리지 않는다): OpenMP 런타임(libgomp)이 주 스레드를 묶은 고정을 그대로 둔다(2026-11-11, 2026-11-12).
AUTO_FULL_MASK_BACKENDS: tuple[str, ...] = ("python", "halide")
# import 모드에서 ``--inspect`` 수집 중일 때의 ``in_progress.test`` 와, 수집이 끝나기 전에 문서에 써 두는 ``inspect_error``
INSPECT_STEP = "inspect"
INSPECT_UNFINISHED = "inspect did not finish: the worker process ended while collecting"

SYS_CPU = Path("/sys/devices/system/cpu")
THERMAL_DIR = Path("/sys/class/thermal")
PKG_TEMP_TYPE = "x86_pkg_temp"
DEFAULT_SIGNATURES = Path(__file__).resolve().parents[1] / "tasks" / "gemm" / "signatures.yaml"


# ----------------------------------------------------------------------------- 프로세스 제한


def disable_core_dumps() -> int | None:
    """``prctl(PR_SET_DUMPABLE, 0)``. 성공하면 0, 실패하면 None(+경고)."""
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        rc = int(libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0))
    except (OSError, AttributeError, TypeError) as e:  # TypeError: Windows 의 ctypes 는 CDLL(None) 을 받지 않는다
        print(f"{_LOG} 경고: prctl(PR_SET_DUMPABLE) 실패: {e}", file=sys.stderr)
        return None
    if rc != 0:
        print(f"{_LOG} 경고: prctl(PR_SET_DUMPABLE) rc={rc} errno={ctypes.get_errno()}", file=sys.stderr)
        return None
    return 0


def apply_memory_limit(max_mem_gb: float) -> dict[str, Any]:
    """``RLIMIT_AS`` 를 max_mem_gb(GiB)로 제한한다(하드 한도보다 크면 하드 한도). 적용값을 돌려준다.

    - ``max_mem_gb <= 0``: 제한을 걸지 않는다(ASan 은 수 TiB 의 가상 주소 공간을 예약하므로 RLIMIT_AS 아래에서 뜨지 않는다).
      ``setrlimit`` 을 부르지 않으며 ``{"rlimit_as_bytes": <실제로 걸려 있는 soft 한도 | None>, "rlimit_as_requested_bytes": None,
      …, "disabled": True}``. ``rlimit_as_bytes`` 는 다른 분기와 같이 **지금 걸려 있는 값**이다: 오케스트레이터를 띄운 셸의
      ``ulimit -v`` 처럼 물려받은 soft 한도가 있으면 그 값이 남는다(없으면 None) — "제한 없음"이라고 잘못 기록하지 않는다.
    - ``resource`` 모듈이 없는 플랫폼(Windows): ``{"rlimit_as_bytes": None, "unsupported": True}`` + 경고.
    """
    if resource is None:
        print(f"{_LOG} 경고: 이 플랫폼({sys.platform})에는 resource 모듈이 없어 RLIMIT_AS 를 걸지 않는다", file=sys.stderr)
        return {"rlimit_as_bytes": None, "unsupported": True}
    if max_mem_gb <= 0:
        soft0, hard0 = resource.getrlimit(resource.RLIMIT_AS)
        return {
            "rlimit_as_bytes": None if soft0 == resource.RLIM_INFINITY else soft0,
            "rlimit_as_requested_bytes": None,
            "rlimit_as_hard_bytes": None if hard0 == resource.RLIM_INFINITY else hard0,
            "disabled": True,
        }
    requested = int(max_mem_gb * (1024**3))
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    target = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    applied: int | None
    try:
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
        applied = target
    except (ValueError, OSError) as e:
        print(f"{_LOG} 경고: RLIMIT_AS 설정 실패({e}); 제한 없이 계속한다", file=sys.stderr)
        applied = None if soft == resource.RLIM_INFINITY else soft
    return {
        "rlimit_as_bytes": applied,
        "rlimit_as_requested_bytes": requested,
        "rlimit_as_hard_bytes": None if hard == resource.RLIM_INFINITY else hard,
    }


# ----------------------------------------------------------------------------- 버퍼


@dataclass
class AlignedArray:
    """64바이트 정렬 뷰 + 가드 영역. ``full`` 은 뷰와 가드를 잇는 전체(길이 n+guard)."""

    view: np.ndarray
    base: np.ndarray
    guard: np.ndarray
    offset: int
    n: int
    front: np.ndarray | None = None
    guard_value: float = 0.0

    @property
    def full(self) -> np.ndarray:
        """뷰 + 뒤쪽 가드(앞쪽 가드는 제외; swapped_MN_dims 패턴은 out[0] 부터 본다)."""
        return self.base[self.offset : self.offset + self.n + self.guard.size]

    def guards_intact(self) -> bool:
        gv = self.guard.dtype.type(self.guard_value)
        if np.any(self.guard != gv):
            return False
        return self.front is None or not np.any(self.front != gv)


NP_TYPES = {"float32": np.float32, "int32": np.int32}


def alloc_aligned(n: int, guard: int, sentinel: float, dtype: Any = np.float32, *, front: int = GUARD_FRONT,
                  guard_value: float | None = None) -> AlignedArray:
    """``base = np.empty(front + n + slack + guard)``, 뷰를 64바이트 경계에 맞추고 앞·뒤 가드를 ``guard_value``(기본 센티널)로 채운다."""
    dt = np.dtype(dtype)
    gv = sentinel if guard_value is None else guard_value
    slack = ALIGNMENT_BYTES // dt.itemsize
    base = np.empty(front + n + slack + guard, dtype=dt)
    addr0 = base.ctypes.data + front * dt.itemsize
    off = front + ((-addr0) % ALIGNMENT_BYTES) // dt.itemsize
    # 먼저 오프셋만큼 옮기고 길이를 자른다: 길이 0 인 조각(base[off:off])은 NumPy 가 데이터 포인터를 옮기지 않아 정렬되지 않는다.
    view = base[off:][:n]
    front_view = base[off - front : off]
    guard_view = base[off + n : off + n + guard]
    front_view[...] = dt.type(gv)
    guard_view[...] = dt.type(gv)
    if view.ctypes.data % ALIGNMENT_BYTES != 0:
        raise RuntimeError("alloc_aligned: 정렬 실패")
    return AlignedArray(view=view, base=base, guard=guard_view, offset=off, n=n, front=front_view, guard_value=gv)


@functools.cache
def c_aligned_allocator() -> tuple[Callable[[int], int], Callable[[int], None]]:
    """C 할당자의 ``(alloc(size) -> 주소, free(주소))`` (정렬 ALIGNMENT_BYTES).

    POSIX 는 프로세스 전역 심볼의 ``posix_memalign``/``free`` 다: LD_PRELOAD 로 올린 ASan 이 가로채 블록 크기를 정확히
    안다(``aligned_alloc`` 은 ASan 이 크기가 정렬의 배수가 아니면 오류로 멈추므로 쓰지 않는다). Windows(개발 머신의 단위
    테스트)는 ucrtbase 의 ``_aligned_malloc``/``_aligned_free`` 다."""
    if sys.platform == "win32":
        crt = ctypes.CDLL("ucrtbase")
        aligned_malloc = crt._aligned_malloc
        aligned_malloc.restype = ctypes.c_void_p
        aligned_malloc.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
        aligned_free = crt._aligned_free
        aligned_free.restype = None
        aligned_free.argtypes = [ctypes.c_void_p]

        def alloc_win(size: int) -> int:
            return aligned_malloc(size, ALIGNMENT_BYTES) or 0

        return alloc_win, aligned_free
    libc = ctypes.CDLL(None)
    posix_memalign = libc.posix_memalign
    posix_memalign.restype = ctypes.c_int
    posix_memalign.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_size_t]
    c_free = libc.free
    c_free.restype = None
    c_free.argtypes = [ctypes.c_void_p]

    def alloc_posix(size: int) -> int:
        ptr = ctypes.c_void_p()
        rc = posix_memalign(ctypes.byref(ptr), ALIGNMENT_BYTES, size)
        return (ptr.value or 0) if rc == 0 else 0

    return alloc_posix, c_free


def alloc_exact(n: int, dtype: Any = np.float32) -> AlignedArray:
    """``--sanitize`` 모드의 배열: C 할당자로 정확히 n 원소를 잡는다(시작은 64바이트 정렬, 앞 여유·뒤 가드 없음).

    마지막 원소가 할당 블록의 끝이라 하네스 배열을 한 칸이라도 넘는 읽기·쓰기를 ASan 이 heap-buffer-overflow 로 본다.
    가드는 빈 배열이다(``oob_write`` 는 늘 False). 블록은 이 뷰를 잡은 마지막 참조가 사라질 때 해제한다(테스트가 끝나
    ``Buffers`` 를 버리고 커널 핸들이 다음 테스트의 뷰로 바꾼 뒤). 길이 0 배열도 유효한 주소를 받도록 1바이트를 잡는다."""
    dt = np.dtype(dtype)
    size = max(n * dt.itemsize, 1)
    alloc, free = c_aligned_allocator()
    ptr = alloc(size)
    if not ptr:
        raise MemoryError(f"alloc_exact: C 할당 실패 ({size} bytes)")
    block = type("CBlock", (ctypes.c_char * size,), {}).from_address(ptr)
    weakref.finalize(block, free, ptr)
    view = np.frombuffer(block, dtype=dt, count=n)
    if view.ctypes.data % ALIGNMENT_BYTES != 0:
        raise RuntimeError("alloc_exact: 정렬 실패")
    return AlignedArray(view=view, base=view, guard=view[n:], offset=0, n=n)


class Buffers:
    """테스트 하나의 입력·출력 버퍼(정렬 + 가드)와 참조, 복원·검사 헬퍼. 인자 구성은 ``TaskSpec`` 이 정한다.

    ``sanitize`` 면 배열마다 ``alloc_exact`` (정확 크기, 가드 없음)로 만든다. 출력의 센티널 채움은 그대로다."""

    def __init__(self, spec: TaskSpec, data: dict[str, Any], sentinel: float, *, sanitize: bool = False) -> None:
        self.spec = spec
        self.sentinel = float(sentinel)
        self.dims: dict[str, int] = {n: int(data[n]) for n in spec.dim_names}
        self.arrays: dict[str, AlignedArray] = {}
        # 커널에 넘어가는 배열: 인자 이름 → 과제 형태의 N-D 뷰(복사 없음). 호출 방식은 harness.kernels 의 백엔드가 정한다.
        self.views: dict[str, np.ndarray] = {}
        self.inputs0: dict[str, np.ndarray] = {}
        for arg in spec.args:
            if arg.kind == "dim":
                continue
            shape = spec.shape_of(arg, self.dims)
            n = int(np.prod(shape))
            dt = NP_TYPES[arg.dtype]
            if arg.kind == "in":
                # --sanitize: 정확 크기 C 할당이라 가드도 가드 값도 없다(경계 넘는 접근은 ASan 이 본다)
                arr = (alloc_exact(n, dt) if sanitize else
                       alloc_aligned(n, GUARD_INPUT, sentinel, dt,
                                     guard_value=INPUT_GUARD_INT if np.dtype(dt).kind in "iu" else INPUT_GUARD_FLOAT))
                arr.view[...] = np.asarray(data[arg.name], dtype=dt).reshape(-1)
                self.inputs0[arg.name] = arr.view.copy()
            elif sanitize:
                arr = alloc_exact(n, dt)
            else:
                arr = alloc_aligned(n, spec.guard_extra_of(self.dims) + GUARD_OUTPUT_EXTRA, sentinel, dt)
            self.arrays[arg.name] = arr
            self.views[arg.name] = arr.view.reshape(shape)
        out_arg = spec.output
        self.out = self.arrays[out_arg.name]
        self.ref = np.ascontiguousarray(data[f"{out_arg.name}_ref"], dtype=np.float64).reshape(spec.shape_of(out_arg, self.dims))

    def restore_inputs(self) -> None:
        for name, orig in self.inputs0.items():
            self.arrays[name].view[...] = orig

    def fill_sentinel(self) -> None:
        self.out.view[...] = self.out.view.dtype.type(self.sentinel)

    def inputs_mutated(self) -> bool:
        return not all(np.array_equal(self.arrays[name].view, orig) for name, orig in self.inputs0.items())

    def oob_write(self) -> bool:
        """어느 배열이든 앞·뒤 가드가 바뀌었으면 True."""
        return any(not arr.guards_intact() for arr in self.arrays.values())

    def working_set_mb(self) -> float:
        return sum(arr.view.nbytes for arr in self.arrays.values()) / (1024**2)


def load_test(path: Path, spec: TaskSpec) -> dict[str, Any]:
    """테스트 npz 를 읽는다: 인자 이름의 배열·dim 스칼라와 ``<out>_ref``. 형태를 과제 명세와 대조한다."""
    out_arg = spec.output
    data: dict[str, Any] = {}
    with np.load(path) as npz:
        for name in spec.dim_names:
            data[name] = int(npz[name])
        for arg in spec.inputs:
            data[arg.name] = np.array(npz[arg.name], dtype=NP_TYPES[arg.dtype])
        data[f"{out_arg.name}_ref"] = np.array(npz[f"{out_arg.name}_ref"], dtype=np.float64)
    dims = {n: data[n] for n in spec.dim_names}
    for arg in [*spec.inputs, out_arg]:
        key = arg.name if arg.kind == "in" else f"{arg.name}_ref"
        want = spec.shape_of(arg, dims)
        if tuple(data[key].shape) != want:
            raise ValueError(f"{path}: {key} 형태 {tuple(data[key].shape)} != 명세 {want} (dims={dims})")
    return data


def load_dims(path: Path, spec: TaskSpec) -> dict[str, int]:
    """테스트 npz 에서 dim 스칼라만 읽는다(import 모드의 형태별 빌드: 입력 데이터는 필요 없다)."""
    with np.load(path) as npz:
        return {name: int(npz[name]) for name in spec.dim_names}


# ----------------------------------------------------------------------------- 시스템 샘플링


def prewarm_blas_threads() -> None:
    """OpenBLAS 스레드 풀을 **커널 로드 전에** 생성한다(2026-11-09).

    libgomp 는 ``OMP_PROC_BIND=true`` 면 ``kernel.so`` 로드 시점에 주 스레드를 첫 코어에 묶는다. OpenBLAS(pthread 빌드)는
    스레드를 처음 큰 호출 때 게으르게 만들므로, 그 호출이 커널 로드 뒤에 오면 워커 스레드들이 주 스레드의 1코어 마스크를
    물려받아 기준선(np.linalg 등)이 한 코어에서 돌아 2배 느려졌다(cholesky 기준선 351 ms vs 190 ms 실측).
    여기서 충분히 큰 sgemm 을 한 번 호출해 스레드 풀이 전체 고정 코어 마스크로 만들어지게 한다."""
    try:
        a = np.ones((512, 512), dtype=np.float32)
        np.matmul(a, a)
    except Exception as e:  # BLAS 가 없어도 측정은 계속한다
        print(f"{_LOG} 경고: BLAS 스레드 예열 실패: {e}", file=sys.stderr)


def pinned_cpus() -> list[int]:
    """현재 프로세스의 CPU 친화도(taskset 결과)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


def loaded_library(stem: str) -> str | None:
    """이 프로세스에 매핑된 공유 객체 가운데 이름이 ``stem`` 으로 시작하는 것의 경로(없으면 None). 기록용:
    libgomp 는 판(版)과 CPU 종류에 따라 병렬 구간 뒤 스핀 대기의 기본값이 다르다(decisions.md 2026-11-11)."""
    try:
        with open("/proc/self/maps", encoding="utf-8") as fh:
            for line in fh:
                path = line.rsplit(" ", 1)[-1].strip()
                if path.startswith("/") and os.path.basename(path).startswith(stem):
                    return os.path.realpath(path)
    except OSError:
        pass
    return None


def loaded_libraries(stem: str) -> list[str]:
    """``loaded_library`` 와 같은 규칙으로 매핑된 공유 객체 **전부**의 서로 다른 실제 경로(정렬; 못 읽으면 빈 리스트).

    numba 의 ``omp`` 스레딩 계층은 C+OpenMP 기준선과 같은 libgomp 를 써야 한다(2026-11-12 결정의 전제). 둘이 다른 libgomp 를
    올리면 OpenMP 런타임이 둘이 되어 스레드 풀·고정이 따로 논다 — 결과 문서에 경로를 모두 남겨 run.py 가 플래그로 단다."""
    found: set[str] = set()
    try:
        with open("/proc/self/maps", encoding="utf-8") as fh:
            for line in fh:
                path = line.rsplit(" ", 1)[-1].strip()
                if path.startswith("/") and os.path.basename(path).startswith(stem):
                    found.add(os.path.realpath(path))
    except OSError:
        pass
    return sorted(found)


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def cpu_mhz_samples(cpus: list[int]) -> list[float]:
    """각 CPU 의 ``scaling_cur_freq`` (kHz → MHz). 못 읽는 CPU 는 건너뛴다."""
    out: list[float] = []
    for cpu in cpus:
        khz = _read_int(SYS_CPU / f"cpu{cpu}" / "cpufreq" / "scaling_cur_freq")
        if khz is not None:
            out.append(khz / 1000.0)
    return out


def throttle_count(cpus: list[int]) -> int | None:
    """``thermal_throttle/core_throttle_count`` 합. 하나도 못 읽으면 None."""
    vals = [_read_int(SYS_CPU / f"cpu{cpu}" / "thermal_throttle" / "core_throttle_count") for cpu in cpus]
    vals = [v for v in vals if v is not None]
    return sum(vals) if vals else None


def package_temp_c() -> float | None:
    """``x86_pkg_temp`` 열 영역의 온도(℃). 없으면 None."""
    try:
        zones = sorted(THERMAL_DIR.glob("thermal_zone*"))
    except OSError:
        return None
    for zone in zones:
        try:
            if (zone / "type").read_text(encoding="utf-8").strip() != PKG_TEMP_TYPE:
                continue
        except OSError:
            continue
        milli = _read_int(zone / "temp")
        if milli is not None:
            return milli / 1000.0
    return None


def loadavg_1m() -> float | None:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # AttributeError: os.getloadavg 가 없는 플랫폼(Windows)
        return None


# ----------------------------------------------------------------------------- 결과 파일


class ResultWriter:
    """결과 문서를 통째로 원자적으로 쓴다(tmp → os.replace)."""

    def __init__(self, path: Path, doc: dict[str, Any]) -> None:
        self.path = Path(path)
        self.doc = doc

    def flush(self, *, sync: bool = True) -> None:
        """``sync=False``: 디스크 동기화(fsync) 없이 쓴다(bench 루프 안의 진행 기록용; ``run_bench`` 의 ``loop_fsync``)."""
        atomic_write_json(self.path, self.doc, fsync=sync)


@contextlib.contextmanager
def harness_only(writer: ResultWriter, what: str) -> Iterator[None]:
    """LLM 코드가 돌지 않는 구간. 여기서 프로세스가 죽으면(중단·세그폴트·타임아웃) 원인은 하네스 쪽(도구·설정)이다.

    들어가기 전에 결과 문서에 ``worker_error`` 를 써 두고 빠져나오면(예외로든) 지운다 — 문서가 그대로 남으면 run.py 가
    harness_error 로 run 을 멈추고 행을 쓰지 않는다. 구간 안에서 난 예외는 평소처럼 올라간다."""
    writer.doc["worker_error"] = f"워커가 {what} 중에 끝났다(LLM 코드를 실행하기 전: 도구·설정 문제)"
    writer.flush()
    try:
        yield
    finally:
        writer.doc.pop("worker_error", None)
        writer.flush()


def set_main_affinity(cpus: Iterable[int]) -> tuple[set[int] | None, str | None]:
    """호출 스레드(워커의 주 스레드)의 CPU 친화도를 ``cpus`` 로 바꾼다 → ``(이전 집합, 오류 문자열)``.

    왜 필요한가(2026-11-11): 교차 측정에서는 커널과 기준선이 한 프로세스에 함께 로드된다. 둘 중 하나가 OpenMP 공유
    객체면 libgomp 가 ``OMP_PROC_BIND=true`` 아래 로드 시점에 주 스레드를 place 0(CPU 하나)에 묶는다. 그 상태에서
    파이썬 커널을 부르면 커널이 만든 스레드(``threading``, ``ThreadPoolExecutor``)가 1-CPU 마스크를 물려받아 4스레드
    코드가 한 코어에서 돈다(실측: 스레드를 쓰는 numpy 커널의 bench 에서 cpu0 만 3.7 GHz, 나머지 고정 코어는 400 MHz).
    그래서 OpenMP 가 아닌 코드를 부르는 동안에는 주 스레드의 친화도를 그 언어의 규약 집합(numpy: 고정 코어 전체,
    c_seq: 고정 코어의 첫 CPU)으로 두고, 호출이 끝나면 직전의 집합(libgomp 가 정한 고정)으로 되돌린다. libgomp 는 주
    스레드를 다시 묶지 않으므로 되돌리지 않으면 OpenMP 주 스레드가 떠돌아 병렬 구간이 느려진다(검토 실측 1.9배)."""
    try:
        prev = set(os.sched_getaffinity(0))
        want = set(cpus)
        if want and prev != want:
            os.sched_setaffinity(0, want)
        return prev, None
    except (AttributeError, OSError, ValueError, OverflowError) as e:
        return None, f"{type(e).__name__}: {e}"


def _restore_main_affinity(prev: set[int] | None) -> str | None:
    """``set_main_affinity`` 가 돌려준 이전 집합으로 되돌린다 → 오류 문자열 또는 None."""
    if prev:
        try:
            if set(os.sched_getaffinity(0)) != prev:
                os.sched_setaffinity(0, prev)
        except (AttributeError, OSError, ValueError, OverflowError) as e:
            return f"restore {type(e).__name__}: {e}"
    return None


# ----------------------------------------------------------------------------- 커널 로드·준비·호출


def load_kernel(
    ns: argparse.Namespace,
    options: kernels.KernelOptions,
    spec: TaskSpec,
    tests: list[tuple[str, Path]],
    writer: ResultWriter,
) -> kernels.KernelHandle | None:
    """백엔드를 열고 커널을 로드한다. import 모드에서 형태별 빌드가 있는 백엔드는 첫 테스트 크기로 빌드까지 해 본다
    (정의 오류가 빌드 단계의 ``compile_error`` 로 드러나게). 커널 쪽 실패면 결과 문서에 ``load_ok=false`` 와 판정을 적고 None."""
    doc = writer.doc
    try:
        with harness_only(writer, "백엔드를 준비하는"):
            load = kernels.open_backend(ns.backend, options)
        handle = load(ns.artifact, spec)
        if ns.mode == "import" and handle.per_shape_build:
            if not tests:
                raise ValueError(f"--tests(첫 정확성 테스트)가 필요하다 (import, --backend {ns.backend})")
            name, path = tests[0]
            dims = load_dims(path, spec)
            doc.update({"test": name, "dims": dims, "in_progress": {"test": name, "call_idx": None}})
            writer.flush()
            doc["build_ms"] = handle.prepare(dims)
            doc["ignored_schedule_calls"] = list(handle.ignored_schedule_calls)
            doc["in_progress"] = None
    except kernels.KernelError as e:
        doc.update({"load_ok": False, "load_error": e.detail, "classification": e.classification, "error_detail": e.detail})
        if e.exception is not None:
            doc["exception"] = e.exception
        return None
    doc.update({"load_ok": True, "load_error": None})
    return handle


def load_baseline(backend: str, artifact: Path, spec: TaskSpec) -> tuple[kernels.KernelHandle | None, str, str]:
    """bench 교차 측정용 기준선을 로드한다 → ``(handle, classification, detail)``. 실패하면 handle=None,
    classification ∈ ``compile_error`` | ``signature_mismatch`` (2026-11-09).

    기준선은 언제나 c 또는 python 백엔드다. 모듈 이름을 커널(``kernel``)과 달리 해(``BASELINE_MODULE_NAME``) ``sys.modules`` 의
    커널 모듈을 덮지 않는다. ``harness_only`` 로 감싸지 않는다: 기준선 로드 실패는 치명적이지 않다(교차 측정 없이 진행).
    주 스레드 친화도(파이썬 기준선은 고정 코어 전체에서 임포트)는 부르는 쪽(``main``)이 맡는다."""
    try:
        load = kernels.open_backend(backend, kernels.KernelOptions(args_mode="positional", module_name=BASELINE_MODULE_NAME))
        return load(artifact, spec), "", ""
    except kernels.KernelError as e:
        return None, e.classification, e.detail


def collect_inspect(handle: kernels.KernelHandle, dims: dict[str, int] | None, writer: ResultWriter) -> None:
    """``--inspect``: 로드된 커널의 관찰값을 결과 문서의 ``inspect`` 에 넣는다. 수집 실패는 ``inspect.inspect_error`` 로 남을 뿐
    로드 판정(``load_ok``·``classification``)을 바꾸지 않는다.

    ``dims`` (성능 형태)로 다시 빌드하는 백엔드는 여기서 LLM 코드를 한 번 더 실행한다. 그 도중에 프로세스가 끝나도(중단·시간
    초과) 문서에 "로드는 됐고 수집이 끝나지 않았다"가 남도록 먼저 써 둔다."""
    doc = writer.doc
    doc["inspect"] = {"inspect_error": INSPECT_UNFINISHED}
    doc["in_progress"] = {"test": INSPECT_STEP, "call_idx": None}
    writer.flush()
    doc["inspect"] = handle.inspect(dims)


def prepare_kernel(
    handle: kernels.KernelHandle, bufs: Buffers, test_name: str, writer: ResultWriter
) -> tuple[float | None, dict[str, Any] | None]:
    """테스트 하나에 대해 커널을 준비한다(인자 묶기, 형태별 빌드) → ``(build_ms, crash)``.

    ``crash`` 는 준비가 커널 쪽 원인으로 실패했을 때의 기록 ``{"pattern", "exception", "elapsed_ms"}`` 다: 그 테스트의
    ``runtime_crash`` 가 된다. 형태별 빌드는 LLM 코드를 실행하므로 그 전에 ``in_progress`` 를 써 둔다(빌드 중 프로세스가
    죽어도 어느 테스트였는지 남는다).

    ``handle.call_cpus`` 가 있으면(bench) 준비도 그 집합에서 한다: 빌드 함수가 만드는 스레드(예: 빌드 중의 ``realize`` 가 띄우는
    Halide 스레드 풀)가 OpenMP 기준선이 좁혀 둔 1-CPU 마스크를 물려받지 않게 한다(2026-11-11 과 같은 경로). 적용·복원은
    ``build_ms`` 밖이다. c·python 은 인자만 묶으므로 영향이 없다."""
    if handle.per_shape_build:
        writer.doc["in_progress"] = {"test": test_name, "call_idx": None}
        writer.flush()
    prev: set[int] | None = None
    if handle.call_cpus is not None:
        prev, err = set_main_affinity(handle.call_cpus)
        if err is not None:
            handle.affinity_error = err
    t0 = time.perf_counter_ns()
    try:
        return handle.prepare(bufs.dims, bufs.views), None
    except kernels.KernelError as e:
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        return None, {"pattern": e.detail, "exception": e.record(), "elapsed_ms": elapsed_ms}
    finally:
        handle.affinity_error = _restore_main_affinity(prev) or handle.affinity_error


def call_kernel(handle: kernels.KernelHandle) -> tuple[float, Any, dict[str, Any] | None]:
    """커널을 한 번 호출한다 → ``(elapsed_ms, return_value, exception|None)``. 어떤 BaseException 도 잡는다.

    ``handle.call_cpus`` 가 있으면 호출 동안만 주 스레드 친화도를 그 집합으로 둔다(적용·복원은 시간 측정 구간 밖)."""
    prev: set[int] | None = None
    if handle.call_cpus is not None:
        prev, err = set_main_affinity(handle.call_cpus)
        if err is not None:
            handle.affinity_error = err
    t0 = time.perf_counter_ns()
    try:
        ret = handle.invoke()
    except BaseException as e:  # 커널의 예외·SystemExit·KeyboardInterrupt 모두 runtime_crash
        t1 = time.perf_counter_ns()
        handle.affinity_error = _restore_main_affinity(prev) or handle.affinity_error
        return (t1 - t0) / 1e6, None, kernels.exception_record(e)  # except 블록 안: 트레이스백은 이 예외의 것
    t1 = time.perf_counter_ns()
    handle.affinity_error = _restore_main_affinity(prev) or handle.affinity_error
    return (t1 - t0) / 1e6, ret, None


def crash_detail(handle: kernels.KernelHandle, exc: dict[str, Any]) -> str:
    """커널 호출 예외의 한 줄 설명(행의 ``error_detail``). c·python 은 ``Type: message`` 그대로(500자; 2026-11-23 이전 행과
    같은 문자열), 여러 줄 오류를 내는 백엔드(numba·halide: ``handle.join_error_lines``)는 메시지의 줄을 한 줄로 잇는다."""
    return kernels.crash_detail(exc, join_lines=handle.join_error_lines)


def outputs_close(a: np.ndarray, b: np.ndarray, tol: Tolerance) -> bool:
    """두 출력이 허용오차 안에서 같은가(NaN 은 NaN 과 같다; 유한 마스크가 다르면 다르다)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.allclose(a, b, rtol=tol.rtol, atol=tol.atol, equal_nan=True))


def _return_type(handle: kernels.KernelHandle, ret: Any) -> str | None:
    return type(ret).__name__ if handle.has_return_value else None


def _run_record(handle: kernels.KernelHandle, elapsed_ms: float, exc: dict[str, Any] | None,
                return_type: str | None) -> dict[str, Any]:
    """check 모드의 호출 기록 하나(판정 전). 예외가 있으면 ``runtime_crash`` 로 채운다(설명은 ``crash_detail``)."""
    run: dict[str, Any] = {
        "classification": None, "pattern": "", "max_abs_err": None, "max_tol_ratio": None,
        "n_unwritten": None, "nan_inf": None, "n_fail": None, "n_total": None, "elapsed_ms": elapsed_ms, "exception": exc,
        "return_type": return_type, "inputs_mutated": None, "oob_write": None,
    }
    if exc is not None:
        run["classification"] = "runtime_crash"
        run["pattern"] = crash_detail(handle, exc)
    return run


def _apply_contract_flags(status: str, pattern: str, *, mutated: bool, oob: bool) -> tuple[str, str]:
    """호출 후 공통 검사 플래그를 출력 판정에 반영한다.

    입력 변조는 어떤 판정(pass/arg_order/numeric)이든 ``wrong_output``/``inputs_mutated`` 로 덮는다(기존 규칙).
    §10 v2.1 에서는 허용오차 내(규칙 2)가 계약 규칙(규칙 4)보다 먼저라 출력이 맞으면 ``classify_output`` 은 OOB 가
    있어도 ``pass`` 를 돌려준다 — 그 호출은 ``wrong_output``/``out_of_bounds_write`` 로 기록해 bench 의
    ``error_detail`` 과 check_tests 가 비지 않게 한다. arg_order 판정은 OOB 가 남아 있어도 그대로 둔다(§10-3).
    """
    if mutated and status != "wrong_output":
        return "wrong_output", "inputs_mutated"
    if oob and status == "pass":
        return "wrong_output", "out_of_bounds_write"
    return status, pattern


# ----------------------------------------------------------------------------- check 모드


def run_check(
    handle: kernels.KernelHandle,
    spec: TaskSpec,
    tests: list[tuple[str, Path]],
    tol_base: Tolerance,
    check_repeats: int,
    writer: ResultWriter,
    *,
    sanitize: bool = False,
) -> None:
    """정확성 테스트마다 ``check_repeats`` 회 호출해 판정하고 실행 간 불일치를 본다. 결과는 writer.doc 에 쓴다.

    ``sanitize`` 는 ``Buffers`` 의 ``--sanitize`` 배치(정확 크기, 가드 없음)다. 형태별 빌드가 있는 백엔드(halide)는 테스트마다
    ``prepare_kernel`` 로 빌드하고(``build_ms``, auto 면 ``ignored_schedule_calls``), 빌드 실패는 그 테스트의 ``runtime_crash`` 다."""
    doc = writer.doc
    doc["tests"] = []
    doc["inputs_mutated"] = False
    doc["oob_write"] = False
    doc["nondeterministic"] = False
    doc["max_abs_err"] = None
    doc["max_tol_ratio"] = None
    any_mutated = False
    any_oob = False
    crashed = False

    for name, path in tests:
        bufs = Buffers(spec, load_test(path, spec), tol_base.sentinel, sanitize=sanitize)
        tol = effective_tolerance(tol_base, bufs.ref)  # atol_rms_rel 이 있는 과제는 형태마다 atol 이 다르다
        test_doc: dict[str, Any] = {"name": name, **bufs.dims, "dims": dict(bufs.dims), "runs": [], "classification": None,
                                    "atol_effective": tol.atol,
                                    "pattern": "", "nondeterministic": False}
        doc["tests"].append(test_doc)
        outputs: list[np.ndarray] = []

        build_ms, crash = prepare_kernel(handle, bufs, name, writer)
        if build_ms is not None:
            test_doc["build_ms"] = build_ms
        if handle.per_shape_build and crash is None:
            test_doc["ignored_schedule_calls"] = list(handle.ignored_schedule_calls)
        if crash is not None:
            run = _run_record(handle, crash["elapsed_ms"], crash["exception"], _return_type(handle, None))
            run["pattern"] = crash["pattern"]
            test_doc["runs"].append(run)
            crashed = True
            writer.flush()

        for call_idx in range(0 if crashed else check_repeats):
            bufs.restore_inputs()
            bufs.fill_sentinel()
            doc["in_progress"] = {"test": name, "call_idx": call_idx}
            writer.flush()

            elapsed_ms, ret, exc = call_kernel(handle)
            run = _run_record(handle, elapsed_ms, exc, _return_type(handle, ret))
            if exc is not None:
                test_doc["runs"].append(run)
                crashed = True
                writer.flush()
                break

            mutated = bufs.inputs_mutated()
            oob = bufs.oob_write()
            status, pattern, metrics = classify_task(
                spec.task, bufs.out.view, bufs.ref, bufs.inputs0, bufs.dims, tol,
                oob_write=oob, return_type=run["return_type"], out_full=bufs.out.full,
            )
            status, pattern = _apply_contract_flags(status, pattern, mutated=mutated, oob=oob)
            run.update({
                "classification": status, "pattern": pattern, "max_abs_err": metrics["max_abs_err"],
                "max_tol_ratio": metrics["max_tol_ratio"], "n_unwritten": metrics["n_unwritten"],
                "nan_inf": metrics["nan_inf"], "n_fail": metrics["n_fail"], "n_total": metrics["n_total"],
                "inputs_mutated": mutated, "oob_write": oob,
            })
            any_mutated = any_mutated or mutated
            any_oob = any_oob or oob
            test_doc["runs"].append(run)
            outputs.append(bufs.out.view.copy())
            writer.flush()

        # 테스트 수준 요약
        if len(outputs) >= 2:
            test_doc["nondeterministic"] = any(not outputs_close(outputs[0], o, tol) for o in outputs[1:])
        failing = [r for r in test_doc["runs"] if r["classification"] != "pass"]
        if failing:
            test_doc["classification"] = failing[0]["classification"]
            test_doc["pattern"] = failing[0]["pattern"]
        else:
            test_doc["classification"] = "pass"
            test_doc["pattern"] = ""
        errs = [r["max_abs_err"] for r in test_doc["runs"] if r["max_abs_err"] is not None]
        ratios = [r["max_tol_ratio"] for r in test_doc["runs"] if r["max_tol_ratio"] is not None]
        test_doc["max_abs_err"] = max(errs) if errs else None
        test_doc["max_tol_ratio"] = max(ratios) if ratios else None

        doc["inputs_mutated"] = any_mutated
        doc["oob_write"] = any_oob
        doc["nondeterministic"] = any(t["nondeterministic"] for t in doc["tests"])
        all_errs = [t["max_abs_err"] for t in doc["tests"] if t.get("max_abs_err") is not None]
        all_ratios = [t["max_tol_ratio"] for t in doc["tests"] if t.get("max_tol_ratio") is not None]
        doc["max_abs_err"] = max(all_errs) if all_errs else None
        doc["max_tol_ratio"] = max(all_ratios) if all_ratios else None
        doc["in_progress"] = None
        writer.flush()
        del bufs
        if crashed:
            break  # 이후 테스트 중단

    classification, detail = aggregate_sample(doc["tests"], inputs_mutated=any_mutated, oob_write=any_oob)
    doc["classification"] = classification
    doc["error_detail"] = detail


# ----------------------------------------------------------------------------- bench 모드


def _stats(timed_ms: list[float]) -> dict[str, float | None]:
    if not timed_ms:
        return {"median_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None, "cv": None}
    arr = np.asarray(timed_ms, dtype=np.float64)
    mean = float(arr.mean())
    cv = float(arr.std(ddof=1) / mean) if arr.size >= 2 and mean > 0 else None
    return {"median_ms": float(np.median(arr)), "mean_ms": mean, "min_ms": float(arr.min()), "max_ms": float(arr.max()), "cv": cv}


def run_bench(
    handle: kernels.KernelHandle,
    spec: TaskSpec,
    test: tuple[str, Path],
    tol: Tolerance,
    *,
    warmup: int,
    repeats: int,
    min_repeats: int,
    budget_s: float,
    writer: ResultWriter,
    cpus: list[int] | None = None,
    baseline: kernels.KernelHandle | None = None,
    loop_fsync: bool = True,
    sanitize: bool = False,
) -> None:
    """성능 테스트 1개: 센티널 채움 1회 → warmup → timed. 적응·조기 종료 규칙은 설계 §8.

    ``loop_fsync=False`` 면 호출 쌍 사이의 진행 기록을 디스크 동기화 없이 쓴다(마지막 기록은 언제나 동기화한다). 기본(True)은
    2026-11-11 이전과 같은 동작이다: 쌍 사이에 fsync 두 번(이 머신에서 합계 13~20 ms 추정)이 끼어 쌍의 첫 호출만 유휴 뒤에
    시작한다(검토 실측; 규약 변경 후보 — decisions.md 2026-11-11).

    ``baseline`` 이 있으면(2026-11-09, 교차 측정) 커널과 기준선을 **같은 프로세스에서 번갈아 호출**한다: 호출 쌍마다 순서를
    바꿔(k,b / b,k) 두 커널이 같은 클럭·열 상태를 보게 하고, 기준선 결과는 ``doc["baseline_interleaved"]`` 에 쓴다.
    적응·조기 종료 규칙은 한 쌍(커널 + 기준선)의 시간으로 투영한다.

    ``cpus`` 는 주파수·스로틀 샘플링 대상 CPU(= taskset 으로 고정된 집합). **커널 로드 전에** 읽은 값을 넘겨야 한다:
    OpenMP 커널(libgomp)은 ``OMP_PROC_BIND=true`` 면 로드 시점에 주 스레드를 place 0(첫 코어)에 묶어
    ``os.sched_getaffinity(0)`` 이 CPU 하나로 좁아진다(실측). None 이면 지금 읽는다.
    ``sanitize`` 는 ``Buffers`` 의 ``--sanitize`` 배치다(커널·기준선 버퍼 모두).

    형태별 빌드가 있는 커널(halide)은 루프 전에 한 번 빌드한다(``prepare_kernel``: ``handle.call_cpus`` 아래, 측정 밖,
    ``build_ms``). 빌드가 실패하면 ``runtime_crash`` 로 기록하고 루프에 들어가지 않는다(기준선 기록은 호출 없이 남는다:
    ``timed_ms`` 가 비고 ``classification`` 은 None — 커널이 첫 호출에서 죽은 경우와 같다). 기준선(c·python)은 그 뒤 인자만 묶는다.
    """
    doc = writer.doc
    name, path = test
    data = load_test(path, spec)
    bufs = Buffers(spec, data, tol.sentinel, sanitize=sanitize)
    bbufs = Buffers(spec, data, tol.sentinel, sanitize=sanitize) if baseline is not None else None
    tol = effective_tolerance(tol, bufs.ref)
    cpus = pinned_cpus() if cpus is None else list(cpus)

    doc.update({
        "test": name, **bufs.dims, "dims": dict(bufs.dims), "task": spec.task, "alignment": ALIGNMENT_BYTES, "atol_effective": tol.atol,
        "warmup": warmup, "repeats": repeats, "repeats_reduced": False,
        "warmup_ms": [], "timed_ms": [],
        "median_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None, "cv": None, "gflops": None,
        "nondeterministic_runs": 0, "nondeterministic_bitwise_runs": 0,
        "cpu_mhz_samples": [], "cpu_mhz_median": None, "cpu_mhz_min": None,
        "throttle_events": None, "temp_c_start": package_temp_c(), "temp_c_end": None,
        "final_check": None, "inputs_mutated": None, "oob_write": None,
        "classification": None, "error_detail": "", "exception": None,
        "working_set_mb": bufs.working_set_mb(), "pinned_cpus": cpus,
        "interleaved": baseline is not None, "loop_fsync": bool(loop_fsync),
    })
    bdoc: dict[str, Any] | None = None
    if baseline is not None and bbufs is not None:
        bdoc = {"backend": baseline.backend, "warmup_ms": [], "timed_ms": [], "median_ms": None, "mean_ms": None, "min_ms": None,
                "max_ms": None, "cv": None, "cpu_mhz_samples": [], "final_check": None, "inputs_mutated": None, "oob_write": None,
                "classification": None, "error_detail": "", "exception": None}
        doc["baseline_interleaved"] = bdoc
    throttle_start = throttle_count(cpus)
    doc["throttle_start"] = throttle_start

    bufs.restore_inputs()
    bufs.fill_sentinel()
    if bbufs is not None:
        bbufs.restore_inputs()
        bbufs.fill_sentinel()
    n_warm, n_timed = warmup, repeats
    warm_done = timed_done = 0
    call_idx = 0
    C_first: np.ndarray | None = None
    last_ret: Any = None
    last_bret: Any = None
    crashed = False
    early_exit = False

    build_ms, crash = prepare_kernel(handle, bufs, name, writer)
    if build_ms is not None:
        doc["build_ms"] = build_ms
    if handle.per_shape_build and crash is None:
        doc["ignored_schedule_calls"] = list(handle.ignored_schedule_calls)
    if crash is not None:
        doc["exception"] = crash["exception"]
        doc["classification"] = "runtime_crash"
        doc["error_detail"] = crash["pattern"]
        crashed = True
    if baseline is not None and bbufs is not None:
        baseline.prepare(bbufs.dims, bbufs.views)  # c·python 기준선: 인자 묶기뿐(형태별 빌드 없음)
    writer.flush()

    def _call_baseline(is_warm: bool) -> bool:
        """기준선을 한 번 호출해 기록한다. 예외가 나면 bdoc 에 적고 False."""
        assert baseline is not None and bbufs is not None and bdoc is not None
        nonlocal last_bret
        b_ms, bret, bexc = call_kernel(baseline)
        if bexc is not None:
            bdoc["exception"] = bexc
            bdoc["classification"] = "runtime_crash"
            bdoc["error_detail"] = f"{bexc['type']}: {bexc['message']}".rstrip(": ")[:500]
            return False
        last_bret = bret
        (bdoc["warmup_ms"] if is_warm else bdoc["timed_ms"]).append(b_ms)
        if not is_warm:
            bdoc["cpu_mhz_samples"].extend(cpu_mhz_samples(list(baseline.call_cpus) if baseline.call_cpus else cpus))
        return True

    while not crashed and (warm_done < n_warm or timed_done < n_timed):  # crashed: 형태별 빌드가 실패했다
        is_warm = warm_done < n_warm
        doc["in_progress"] = {"test": name, "call_idx": call_idx}
        writer.flush(sync=loop_fsync)
        # 교차 측정: 쌍마다 순서를 바꾼다(짝수 쌍: 기준선 먼저, 홀수 쌍: 커널 먼저)
        baseline_first = baseline is not None and call_idx % 2 == 0
        b_ms_this = 0.0
        if baseline_first:
            t_b0 = time.perf_counter_ns()
            if not _call_baseline(is_warm):
                break
            b_ms_this = (time.perf_counter_ns() - t_b0) / 1e6
        elapsed_ms, ret, exc = call_kernel(handle)
        if exc is not None:
            doc["exception"] = exc
            doc["classification"] = "runtime_crash"
            doc["error_detail"] = crash_detail(handle, exc)
            crashed = True
            break
        last_ret = ret
        if is_warm:
            doc["warmup_ms"].append(elapsed_ms)
            warm_done += 1
        else:
            doc["timed_ms"].append(elapsed_ms)
            timed_done += 1
            # 주파수는 그 코드가 실제로 쓴 CPU 에서 읽는다(순차 C 는 CPU 하나; 나머지 고정 코어는 유휴라 최솟값을 끌어내렸다)
            doc["cpu_mhz_samples"].extend(cpu_mhz_samples(list(handle.call_cpus) if handle.call_cpus else cpus))
            if C_first is None:
                C_first = bufs.out.view.copy()
            else:
                if not np.array_equal(bufs.out.view, C_first):
                    doc["nondeterministic_bitwise_runs"] += 1
                    if not outputs_close(bufs.out.view, C_first, tol):
                        doc["nondeterministic_runs"] += 1
        if baseline is not None and not baseline_first:
            t_b0 = time.perf_counter_ns()
            if not _call_baseline(is_warm):
                break
            b_ms_this = (time.perf_counter_ns() - t_b0) / 1e6

        if call_idx == 0:
            # 적응·조기 종료: 첫 쌍(커널 + 기준선)의 시간으로 예산을 투영한다
            t1 = (elapsed_ms + b_ms_this) / 1000.0
            projected = t1 * (1 + min_repeats)
            if projected > BUDGET_FRACTION * budget_s:
                doc["classification"] = "timeout"
                doc["error_detail"] = f"first_call_{elapsed_ms:.0f}ms_projected_{projected:.0f}s"
                early_exit = True
                break
            if t1 * (n_warm + n_timed) > BUDGET_FRACTION * budget_s:
                n_warm = min(1, warmup)
                n_timed = max(min_repeats, min(repeats, math.floor(BUDGET_FRACTION * budget_s / t1) - n_warm))
                doc["warmup"] = n_warm
                doc["repeats"] = n_timed
                doc["repeats_reduced"] = True
                print(f"{_LOG} 첫 호출 {elapsed_ms:.0f} ms → warmup={n_warm}, repeats={n_timed} 로 축소", file=sys.stderr)
        call_idx += 1
        aff_err = handle.affinity_error or (baseline.affinity_error if baseline is not None else None)
        if aff_err and isinstance(doc.get("call_affinity"), dict):
            doc["call_affinity"]["error"] = aff_err
        writer.flush(sync=loop_fsync)

    doc["in_progress"] = None
    samples = doc["cpu_mhz_samples"]
    doc["cpu_mhz_median"] = float(np.median(samples)) if samples else None
    doc["cpu_mhz_min"] = float(min(samples)) if samples else None
    throttle_end = throttle_count(cpus)
    doc["throttle_events"] = (throttle_end - throttle_start) if (throttle_start is not None and throttle_end is not None) else None
    doc["temp_c_end"] = package_temp_c()

    if bdoc is not None and bbufs is not None and baseline is not None and bdoc.get("classification") != "runtime_crash":
        bdoc.update(_stats(bdoc["timed_ms"]))
        if bdoc["timed_ms"]:
            bmut = bbufs.inputs_mutated()
            boob = bbufs.oob_write()
            bst, bpat, bmetrics = classify_task(spec.task, bbufs.out.view, bbufs.ref, bbufs.inputs0, bbufs.dims, tol,
                                                oob_write=boob, return_type=_return_type(baseline, last_bret), out_full=bbufs.out.full)
            bst, bpat = _apply_contract_flags(bst, bpat, mutated=bmut, oob=boob)
            bdoc["final_check"] = {"classification": bst, "pattern": bpat, **bmetrics}
            bdoc["inputs_mutated"] = bmut
            bdoc["oob_write"] = boob
            bdoc["classification"] = bst
            bdoc["error_detail"] = bpat
    elif bdoc is not None and bdoc.get("classification") != "runtime_crash":
        bdoc["classification"] = "incomplete"

    if crashed or early_exit:
        writer.flush()
        return

    doc.update(_stats(doc["timed_ms"]))
    work = spec.work_of(bufs.dims)
    if doc["median_ms"] and work is not None:
        doc["gflops"] = work / (doc["median_ms"] / 1000.0) / 1e9

    mutated = bufs.inputs_mutated()
    oob = bufs.oob_write()
    status, pattern, metrics = classify_task(
        spec.task, bufs.out.view, bufs.ref, bufs.inputs0, bufs.dims, tol,
        oob_write=oob, return_type=_return_type(handle, last_ret), out_full=bufs.out.full,
    )
    status, pattern = _apply_contract_flags(status, pattern, mutated=mutated, oob=oob)
    doc["final_check"] = {"classification": status, "pattern": pattern, **metrics, "return_type": _return_type(handle, last_ret)}
    doc["inputs_mutated"] = mutated
    doc["oob_write"] = oob
    if doc["nondeterministic_runs"] > 0:
        doc["classification"] = "race_condition"
        doc["error_detail"] = f"nondeterministic_runs={doc['nondeterministic_runs']}/{len(doc['timed_ms'])}"
    else:
        doc["classification"] = status
        doc["error_detail"] = pattern
    writer.flush()


# ----------------------------------------------------------------------------- main


def parse_tests(spec: str | None) -> list[tuple[str, Path]]:
    """``a.npz,b.npz`` → ``[(stem, Path), ...]``."""
    if not spec:
        return []
    out: list[tuple[str, Path]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        p = Path(item)
        out.append((p.stem, p))
    return out


def _json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"JSON 이 아니다: {e}") from None
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("JSON 객체여야 한다")
    return value


def parse_halide_params(text: str) -> dict[str, Any]:
    """``--halide-params``: 오토스케줄러 추가 매개변수(JSON 객체; 값은 문자열·정수·실수 — Halide 에는 문자열로 넘어간다)."""
    value = _json_object(text)
    bad = [k for k, v in value.items() if isinstance(v, bool) or not isinstance(v, (str, int, float))]
    if bad:
        raise argparse.ArgumentTypeError(f"값은 문자열·정수·실수여야 한다: {bad}")
    return value


def parse_dims(text: str) -> dict[str, int]:
    """``--inspect-dims``: dim 이름 → 정수(JSON 객체)."""
    value = _json_object(text)
    bad = [k for k, v in value.items() if isinstance(v, bool) or not isinstance(v, int)]
    if bad:
        raise argparse.ArgumentTypeError(f"dim 값은 정수여야 한다: {bad}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.worker", description="LLM 커널 실행 워커 (설계 §8, v3 §8)")
    parser.add_argument("--backend", choices=kernels.BACKENDS, required=True, help="rust 산출물은 C ABI 라 c 로 실행한다")
    parser.add_argument("--artifact", type=Path, required=True, help="kernel.so 또는 kernel.py")
    parser.add_argument("--mode", choices=("import", "check", "bench"), required=True)
    parser.add_argument("--tests", default=None,
                        help="npz 파일 경로(쉼표 구분); check 는 여러 개, bench 는 1개, halide 의 import 는 첫 정확성 테스트")
    parser.add_argument("--tolerance", type=Path, default=None, help="tolerance.yaml (check/bench 필수)")
    parser.add_argument("--signatures", type=Path, default=DEFAULT_SIGNATURES, help="과제 signatures.yaml (기본 gemm)")
    parser.add_argument("--out", type=Path, required=True, help="결과 JSON 경로")
    parser.add_argument("--check-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--args-mode", choices=kernels.ARGS_MODES, default="positional")
    parser.add_argument("--halide-mode", choices=kernels.HALIDE_MODES, default=None,
                        help="halide: auto(하네스가 오토스케줄러를 적용; 커널의 스케줄 호출은 무시·기록) | manual(커널이 스케줄을 쓴다)")
    parser.add_argument("--halide-autoscheduler", default=kernels.DEFAULT_AUTOSCHEDULER,
                        help=f"halide auto 의 오토스케줄러 이름(기본 {kernels.DEFAULT_AUTOSCHEDULER}; Mullapudi2016·Adams2019 도 된다)")
    parser.add_argument("--halide-params", type=parse_halide_params, default={}, metavar="JSON",
                        help="halide auto: 오토스케줄러 추가 매개변수(JSON 객체; Li2018 은 받지 않는다 — 모르는 키는 Halide 가 프로세스를 끝낸다)")
    parser.add_argument("--threads", type=int, default=None,
                        help="측정 스레드 수(오케스트레이터가 환경변수로 건 값); halide auto 에서는 오토스케줄러의 parallelism")
    parser.add_argument("--inspect", action="store_true", help="import 모드: 컴파일 결과 관찰값을 결과 문서의 inspect 에 넣는다")
    parser.add_argument("--inspect-dims", type=parse_dims, default=None, metavar="JSON",
                        help="--inspect: 성능 형태의 dim 값(JSON 객체); halide 는 이 크기로 만든 파이프라인을 기술한다")
    parser.add_argument("--max-mem-gb", type=float, default=8.0, help="RLIMIT_AS (GiB); 0 이하면 제한을 걸지 않는다")
    parser.add_argument("--sanitize", action="store_true",
                        help="ASan 실행용 버퍼: 배열마다 정확한 크기의 C 할당(64바이트 정렬, 가드 없음)")
    parser.add_argument("--baseline-backend", choices=("c", "python"), default=None, help="bench 교차 측정용 기준선 백엔드")
    parser.add_argument("--baseline-artifact", type=Path, default=None, help="bench 교차 측정용 기준선(kernel .so 또는 .py)")
    parser.add_argument("--kernel-cpus", default="auto",
                        help="bench: 커널을 호출(halide 는 빌드도)하는 동안 주 스레드의 CPU 집합. 'auto'(python·halide 백엔드는 "
                             "taskset 집합 전체, c·numba 는 그대로 — OpenMP 런타임의 고정), 'none'(건드리지 않는다; OpenMP 커널), "
                             "또는 CPU 목록(예: '0,2,4,6', '2')")
    parser.add_argument("--baseline-cpus", default="auto", help="bench: 기준선 호출 동안의 주 스레드 CPU 집합(--kernel-cpus 와 같은 형식)")
    parser.add_argument("--no-loop-fsync", action="store_true",
                        help="bench: 호출 쌍 사이의 진행 기록을 fsync 없이 쓴다(기본은 fsync; measurement.bench_loop_fsync)")
    return parser


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """명령줄을 읽고 백엔드 옵션의 조합을 확인한다(틀리면 argparse 의 사용법 오류: rc=2, 결과 문서 없음)."""
    parser = build_parser()
    ns = parser.parse_args(argv)
    if ns.threads is not None and ns.threads < 1:
        parser.error("--threads 는 1 이상이어야 한다")
    if ns.backend == "halide":
        if ns.halide_mode is None:
            parser.error("--backend halide 에는 --halide-mode 가 필요하다")
        if ns.halide_mode == "auto" and ns.threads is None:
            parser.error("--halide-mode auto 에는 --threads 가 필요하다(오토스케줄러의 parallelism)")
    return ns


def resolve_call_cpus(text: str | None, backend: str, pinned: list[int]) -> tuple[int, ...] | None:
    """``--kernel-cpus`` / ``--baseline-cpus`` 값을 CPU 튜플로. 'auto': python·halide 는 ``pinned``(커널 로드 전 taskset 집합),
    c·numba 는 None(OpenMP 런타임을 쓸 수 있는 코드: libgomp 의 고정을 그대로 둔다; ``AUTO_FULL_MASK_BACKENDS``).
    실제 측정에서는 run.py 가 언제나 목록이나 'none' 을 준다(``protocol.call_cpus``)."""
    t = (text or "auto").strip().lower()
    if t == "none":
        return None
    if t == "auto":
        return tuple(pinned) if (backend in AUTO_FULL_MASK_BACKENDS and pinned) else None
    cpus: set[int] = set()
    for part in t.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    if not cpus:
        raise ValueError(f"빈 CPU 목록: {text!r}")
    return tuple(sorted(cpus))


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    options = kernels.KernelOptions(args_mode=ns.args_mode, threads=ns.threads, halide_mode=ns.halide_mode,
                                    halide_autoscheduler=ns.halide_autoscheduler, halide_params=ns.halide_params)
    doc: dict[str, Any] = {
        "mode": ns.mode, "backend": ns.backend, "backend_detail": kernels.backend_detail(ns.backend, options),
        "artifact": str(ns.artifact),
        "complete": False, "in_progress": None, "load_ok": None, "load_error": None,
        "classification": None, "error_detail": "", "blas": None, "limits": None,
        "loadavg_start": None, "loadavg_end": None, "sanitize": ns.sanitize,
    }
    writer = ResultWriter(ns.out, doc)
    sys.dont_write_bytecode = True  # 커널(kernel.py/baseline.py) 옆에 __pycache__ 를 남기지 않는다
    try:
        dumpable = disable_core_dumps()
        limits = apply_memory_limit(ns.max_mem_gb)
        limits["dumpable"] = dumpable
        doc["limits"] = limits
        doc["blas"] = blas_identity()
        prewarm_blas_threads()  # OpenBLAS 스레드 풀을 커널 로드 전에 만든다(아래 docstring)
        doc["loadavg_start"] = loadavg_1m()
        # taskset 으로 고정된 CPU 집합. 커널 로드 전에 읽는다(OpenMP 런타임이 주 스레드 친화도를 좁히기 전).
        affinity = pinned_cpus()
        doc["pinned_cpus"] = affinity
        writer.flush()

        tests = parse_tests(ns.tests)
        if ns.mode in ("check", "bench"):
            if ns.tolerance is None:
                raise ValueError("--tolerance 가 필요하다 (check/bench)")
            if not tests:
                raise ValueError("--tests 가 필요하다 (check/bench)")
            if ns.mode == "bench" and len(tests) != 1:
                raise ValueError(f"bench 모드는 테스트 1개만 받는다 (현재 {len(tests)}개)")
            tol = load_tolerance(ns.tolerance)
        else:
            tol = None

        spec = load_taskspec(ns.signatures)
        doc["task"] = spec.task
        handle = load_kernel(ns, options, spec, tests, writer)
        if handle is None:
            pass  # load_kernel 이 판정을 적었다
        elif ns.mode == "import":
            doc["classification"] = "pass"
            if ns.inspect:
                collect_inspect(handle, ns.inspect_dims, writer)
        elif ns.mode == "check":
            assert tol is not None
            run_check(handle, spec, tests, tol, max(1, ns.check_repeats), writer, sanitize=ns.sanitize)
        else:
            assert tol is not None
            baseline_handle = None
            if ns.baseline_artifact is not None:
                bbackend = ns.baseline_backend or ("c" if ns.baseline_artifact.suffix == ".so" else "python")
                # 파이썬 기준선은 고정 코어 전체에서 임포트한다: 커널이 OpenMP 면 주 스레드가 이미 CPU 하나에 묶여 있어, 기준선이
                # 임포트 때 만드는 스레드가 그 마스크를 물려받는다(현재 기준선에는 그런 스레드가 없지만 순서에 기대지 않는다).
                # C 기준선은 건드리지 않는다: OpenMP 면 libgomp 가 로드 시점에 주 스레드를 묶는데, 그 고정을 되돌리면 안 된다.
                prev_load = None
                if bbackend == "python" and affinity:
                    prev_load, _ = set_main_affinity(affinity)
                baseline_handle, bcls, bdetail = load_baseline(bbackend, ns.baseline_artifact, spec)
                _restore_main_affinity(prev_load)
                doc["baseline_load"] = {"backend": bbackend, "artifact": str(ns.baseline_artifact), "ok": baseline_handle is not None,
                                        "classification": bcls, "detail": bdetail}
                if baseline_handle is None:
                    print(f"{_LOG} 경고: 기준선 로드 실패({bcls}: {bdetail}); 교차 측정 없이 진행", file=sys.stderr)
            # 호출 동안의 주 스레드 친화도(2026-11-11): OpenMP 가 아닌 코드가 libgomp 의 1-CPU 고정을 물려받지 않게 한다.
            # 요청한 집합은 커널 로드 전의 taskset 집합 안에 있어야 한다(밖이면 적용하지 않고 오류로 남긴다).
            aff_errors: list[str] = []

            def _checked(cpus: tuple[int, ...] | None, who: str) -> tuple[int, ...] | None:
                if cpus is not None and affinity and not set(cpus) <= set(affinity):
                    aff_errors.append(f"{who}: 요청 CPU {list(cpus)} 가 고정 집합 {affinity} 밖")
                    return None
                return cpus

            handle.call_cpus = _checked(resolve_call_cpus(ns.kernel_cpus, ns.backend, affinity), "kernel")
            if baseline_handle is not None:
                baseline_handle.call_cpus = _checked(resolve_call_cpus(ns.baseline_cpus, baseline_handle.backend, affinity), "baseline")
            main_after_load = pinned_cpus()        # 두 코드를 로드한 뒤의 주 스레드 집합(libgomp 가 좁혔으면 CPU 하나)
            doc["call_affinity"] = {
                "main_after_load": main_after_load,
                "kernel": list(handle.call_cpus) if handle.call_cpus is not None else None,
                "baseline": (list(baseline_handle.call_cpus) if baseline_handle is not None and baseline_handle.call_cpus is not None
                             else None),
                "main_after_bench": None,
                "libgomp": loaded_library("libgomp"),
                # 매핑된 libgomp 전부(numba 의 omp 계층과 C+OpenMP 기준선이 같은 런타임을 쓰는가; 둘 이상이면 run.py 의 플래그)
                "libgomp_paths": loaded_libraries("libgomp"),
                "error": "; ".join(aff_errors) or None,
            }
            run_bench(handle, spec, tests[0], tol, warmup=max(0, ns.warmup), repeats=max(1, ns.repeats),
                      min_repeats=max(1, min(ns.min_repeats, ns.repeats)), budget_s=ns.budget_s, writer=writer,
                      cpus=affinity, baseline=baseline_handle, loop_fsync=not ns.no_loop_fsync, sanitize=ns.sanitize)
            main_after_bench = pinned_cpus()
            doc["call_affinity"]["main_after_bench"] = main_after_bench
            errs = [e for e in (doc["call_affinity"]["error"], handle.affinity_error,
                                baseline_handle.affinity_error if baseline_handle is not None else None) if e]
            if main_after_bench != main_after_load:
                errs.append(f"주 스레드 친화도가 bench 뒤에 달라졌다: {main_after_load} → {main_after_bench}")
            doc["call_affinity"]["error"] = "; ".join(dict.fromkeys(errs)) or None

        doc["loadavg_end"] = loadavg_1m()
        doc["in_progress"] = None
        doc["complete"] = True
        writer.flush()
        return 0
    except Exception:  # 하네스 버그: 기록하고 rc=3
        tb = traceback.format_exc()
        doc["worker_error"] = tb[-WORKER_ERROR_TAIL_CHARS:]
        doc["complete"] = False
        try:
            writer.flush()
        except Exception as e2:  # 결과 파일조차 못 쓰면 stderr 에만 남긴다
            print(f"{_LOG} 결과 파일 쓰기 실패: {e2}", file=sys.stderr)
        print(f"{_LOG} worker_error:\n{tb}", file=sys.stderr)
        return WORKER_ERROR_RC


if __name__ == "__main__":
    sys.exit(main())
