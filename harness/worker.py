"""워커 (설계 §8): LLM 코드를 실행하는 **유일한** 곳. run.py 가 ``procutil.run_isolated`` 로 띄운다.

    PYTHON_EXE -m harness.worker --backend {c,python} --artifact PATH --mode {import,check,bench}
        --tests tiny.npz,rect.npz,small.npz --tolerance tasks/<task>/tolerance.yaml --signatures tasks/<task>/signatures.yaml --out result.json
        [--check-repeats 2] [--warmup 3 --repeats 10 --min-repeats 3 --budget-s 300] [--args-mode positional] [--max-mem-gb 8]

- 시작: ``prctl(PR_SET_DUMPABLE=0)`` (코어 덤프/apport 차단), ``RLIMIT_AS`` (max_mem_gb). 적용값은 ``limits``,
  런타임 BLAS 정체는 ``blas`` 에 기록한다.
- 결과 파일은 매 커널 호출 직전과 매 테스트/호출 뒤에 원자적으로 갱신한다(``complete``, ``in_progress``).
  워커 자체의 예외(하네스 버그)는 ``worker_error`` 로 기록하고 rc=3 으로 끝난다.
- 모든 배열은 64바이트 정렬 + 가드 영역(센티널) 위에 만든다. 호출 전 입력을 복원하고 C 를 센티널로 채운다.
- stdout 에는 아무것도 쓰지 않는다(커널의 stdout 은 run.py 가 파일로 받아 contract_violation 을 본다). 입력 배열은 덤프하지 않는다.
"""

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import inspect
import math
import os
import resource
import sys
import time
import traceback
from ctypes import POINTER, c_float, c_int, c_int32
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from harness.classify import Tolerance, aggregate_sample, classify_task, load_tolerance
from harness.paths import atomic_write_json
from harness.protocol import blas_identity
from harness.taskspec import TaskSpec, load_taskspec

_LOG = "[worker]"

WORKER_ERROR_RC = 3
PR_SET_DUMPABLE = 4
ALIGNMENT_BYTES = 64
ALIGN_SLACK_FLOATS = 16  # 64 바이트 정렬을 위한 여유(float32 16개 = 64 B)
GUARD_INPUT = 1024
GUARD_OUTPUT_EXTRA = 1024
TRACEBACK_TAIL_CHARS = 2000
BUDGET_FRACTION = 0.8

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
    except (OSError, AttributeError) as e:
        print(f"{_LOG} 경고: prctl(PR_SET_DUMPABLE) 실패: {e}", file=sys.stderr)
        return None
    if rc != 0:
        print(f"{_LOG} 경고: prctl(PR_SET_DUMPABLE) rc={rc} errno={ctypes.get_errno()}", file=sys.stderr)
        return None
    return 0


def apply_memory_limit(max_mem_gb: float) -> dict[str, Any]:
    """``RLIMIT_AS`` 를 max_mem_gb(GiB)로 제한한다(하드 한도보다 크면 하드 한도). 적용값을 돌려준다."""
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


# ----------------------------------------------------------------------------- 커널 로드


@dataclass
class KernelHandle:
    """로드된 커널: 백엔드별 호출 방식을 감춘다."""

    backend: str
    fn: Callable[..., Any]
    args_mode: str

    def invoke(self, bufs: "Buffers") -> Any:
        """커널을 한 번 호출한다. 반환값은 python 백엔드의 것(그 외 None)."""
        if self.backend == "c":
            self.fn(*bufs.c_args)
            return None
        if self.args_mode == "keyword":
            return self.fn(**dict(zip(bufs.names, bufs.py_args)))
        return self.fn(*bufs.py_args)


def _safe_str(e: BaseException) -> str:
    """``str(e)``. 커널이 올린 예외 객체의 ``__str__`` 자체가 예외를 던지면(LLM 코드) ``<unprintable Type>`` —
    BaseException 핸들러가 다시 죽어 worker_error(rc=3, run 중단)가 되지 않도록 한다."""
    try:
        return str(e)
    except BaseException:  # __str__ 이 SystemExit 등을 올려도 워커는 살아야 한다
        return f"<unprintable {type(e).__name__}>"


def _safe_format_exc() -> str:
    """``traceback.format_exc()``; 포맷 중 예외(커널 예외 객체의 속성이 던지는 경우)면 자리표시 문자열."""
    try:
        return traceback.format_exc()
    except BaseException:
        return "<traceback unavailable>"


def _one_line(e: BaseException) -> str:
    """예외를 한 줄 ``Type: message`` 로."""
    msg = _safe_str(e).strip().splitlines()
    return f"{type(e).__name__}: {msg[0] if msg else ''}".rstrip(": ")[:500]


def load_kernel(backend: str, artifact: Path, args_mode: str, spec: TaskSpec) -> tuple[KernelHandle | None, str, str]:
    """커널을 로드한다 → ``(handle, classification, detail)``. 실패하면 handle=None,
    classification ∈ ``compile_error`` | ``signature_mismatch``."""
    if backend == "c":
        try:
            lib = ctypes.CDLL(str(artifact))
        except OSError as e:
            return None, "compile_error", _one_line(e)
        try:
            fn = getattr(lib, spec.function)
        except AttributeError as e:
            return None, "signature_mismatch", f"symbol {spec.function} not found: {e}"[:500]
        fn.argtypes = [c_int if a.kind == "dim" else POINTER(C_TYPES[a.dtype]) for a in spec.args]
        fn.restype = None
        return KernelHandle("c", fn, args_mode), "", ""

    mspec = importlib.util.spec_from_file_location("kernel", str(artifact))
    if mspec is None or mspec.loader is None:
        return None, "compile_error", f"cannot create module spec for {artifact}"
    module = importlib.util.module_from_spec(mspec)
    sys.modules["kernel"] = module
    try:
        mspec.loader.exec_module(module)
    except BaseException as e:  # SystemExit/KeyboardInterrupt 도 임포트 실패로 본다
        return None, "compile_error", _one_line(e)
    name = spec.function
    fn = getattr(module, name, None)
    if fn is None:
        return None, "signature_mismatch", f"module has no attribute {name}"
    if not callable(fn):
        return None, "signature_mismatch", f"{name} is not callable ({type(fn).__name__})"
    if inspect.iscoroutinefunction(fn):
        return None, "signature_mismatch", f"{name} is a coroutine function (async def)"
    return KernelHandle("python", fn, args_mode), "", ""


# ----------------------------------------------------------------------------- 버퍼


@dataclass
class AlignedArray:
    """64바이트 정렬 뷰 + 가드 영역. ``full`` 은 뷰와 가드를 잇는 전체(길이 n+guard)."""

    view: np.ndarray
    base: np.ndarray
    guard: np.ndarray
    offset: int
    n: int

    @property
    def full(self) -> np.ndarray:
        return self.base[self.offset : self.offset + self.n + self.guard.size]


C_TYPES = {"float32": c_float, "int32": c_int32}
NP_TYPES = {"float32": np.float32, "int32": np.int32}


def alloc_aligned(n: int, guard: int, sentinel: float, dtype: Any = np.float32) -> AlignedArray:
    """``base = np.empty(n + slack + guard)``, 64바이트 경계로 오프셋을 잡고 가드를 센티널로 채운다(정수 배열은 정수 센티널)."""
    dt = np.dtype(dtype)
    base = np.empty(n + ALIGNMENT_BYTES // dt.itemsize + guard, dtype=dt)
    off = (-base.ctypes.data % ALIGNMENT_BYTES) // dt.itemsize
    view = base[off : off + n]
    guard_view = base[off + n : off + n + guard]
    guard_view[...] = dt.type(sentinel)
    if view.ctypes.data % ALIGNMENT_BYTES != 0:
        raise RuntimeError("alloc_aligned: 정렬 실패")
    return AlignedArray(view=view, base=base, guard=guard_view, offset=off, n=n)


class Buffers:
    """테스트 하나의 입력·출력 버퍼(정렬 + 가드)와 참조, 복원·검사 헬퍼. 인자 구성은 ``TaskSpec`` 이 정한다."""

    def __init__(self, spec: TaskSpec, data: dict[str, Any], sentinel: float) -> None:
        self.spec = spec
        self.sentinel = float(sentinel)
        self.dims: dict[str, int] = {n: int(data[n]) for n in spec.dim_names}
        self.names = spec.params
        self.arrays: dict[str, AlignedArray] = {}
        self.inputs0: dict[str, np.ndarray] = {}
        self.py_args: list[Any] = []
        self.c_args: list[Any] = []
        for arg in spec.args:
            if arg.kind == "dim":
                self.py_args.append(self.dims[arg.name])
                self.c_args.append(self.dims[arg.name])
                continue
            shape = spec.shape_of(arg, self.dims)
            n = int(np.prod(shape))
            dt = NP_TYPES[arg.dtype]
            if arg.kind == "in":
                arr = alloc_aligned(n, GUARD_INPUT, sentinel, dt)
                arr.view[...] = np.asarray(data[arg.name], dtype=dt).reshape(-1)
                self.inputs0[arg.name] = arr.view.copy()
            else:
                arr = alloc_aligned(n, spec.guard_extra_of(self.dims) + GUARD_OUTPUT_EXTRA, sentinel, dt)
            self.arrays[arg.name] = arr
            # python 백엔드용 다차원 뷰(복사 없음)와 c 백엔드용 포인터
            self.py_args.append(arr.view.reshape(shape))
            self.c_args.append(arr.view.ctypes.data_as(POINTER(C_TYPES[arg.dtype])))
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
        return any(bool(np.any(arr.guard != arr.guard.dtype.type(self.sentinel))) for arr in self.arrays.values())

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


# ----------------------------------------------------------------------------- 시스템 샘플링


def pinned_cpus() -> list[int]:
    """현재 프로세스의 CPU 친화도(taskset 결과)."""
    try:
        return sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return []


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
    except OSError:
        return None


# ----------------------------------------------------------------------------- 결과 파일


class ResultWriter:
    """결과 문서를 통째로 원자적으로 쓴다(tmp → os.replace)."""

    def __init__(self, path: Path, doc: dict[str, Any]) -> None:
        self.path = Path(path)
        self.doc = doc

    def flush(self) -> None:
        atomic_write_json(self.path, self.doc)


# ----------------------------------------------------------------------------- 호출


def call_kernel(handle: KernelHandle, bufs: Buffers) -> tuple[float, Any, dict[str, Any] | None]:
    """커널을 한 번 호출한다 → ``(elapsed_ms, return_value, exception|None)``. 어떤 BaseException 도 잡는다."""
    exc: dict[str, Any] | None = None
    ret: Any = None
    t0 = time.perf_counter_ns()
    try:
        ret = handle.invoke(bufs)
    except BaseException as e:  # 커널의 예외·SystemExit·KeyboardInterrupt 모두 runtime_crash
        t1 = time.perf_counter_ns()
        tb = _safe_format_exc()
        exc = {"type": type(e).__name__, "message": _safe_str(e)[:1000], "traceback": tb[-TRACEBACK_TAIL_CHARS:]}
        return (t1 - t0) / 1e6, None, exc
    t1 = time.perf_counter_ns()
    return (t1 - t0) / 1e6, ret, exc


def outputs_close(a: np.ndarray, b: np.ndarray, tol: Tolerance) -> bool:
    """두 출력이 허용오차 안에서 같은가(NaN 은 NaN 과 같다; 유한 마스크가 다르면 다르다)."""
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return bool(np.allclose(a, b, rtol=tol.rtol, atol=tol.atol, equal_nan=True))


def _return_type(handle: KernelHandle, ret: Any) -> str | None:
    return type(ret).__name__ if handle.backend == "python" else None


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
    handle: KernelHandle,
    spec: TaskSpec,
    tests: list[tuple[str, Path]],
    tol: Tolerance,
    check_repeats: int,
    writer: ResultWriter,
) -> None:
    """정확성 테스트마다 ``check_repeats`` 회 호출해 판정하고 실행 간 불일치를 본다. 결과는 writer.doc 에 쓴다."""
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
        bufs = Buffers(spec, load_test(path, spec), tol.sentinel)
        test_doc: dict[str, Any] = {"name": name, **bufs.dims, "dims": dict(bufs.dims), "runs": [], "classification": None,
                                    "pattern": "", "nondeterministic": False}
        doc["tests"].append(test_doc)
        outputs: list[np.ndarray] = []

        for call_idx in range(check_repeats):
            bufs.restore_inputs()
            bufs.fill_sentinel()
            doc["in_progress"] = {"test": name, "call_idx": call_idx}
            writer.flush()

            elapsed_ms, ret, exc = call_kernel(handle, bufs)
            run: dict[str, Any] = {
                "classification": None, "pattern": "", "max_abs_err": None, "max_tol_ratio": None,
                "n_unwritten": None, "nan_inf": None, "elapsed_ms": elapsed_ms, "exception": exc,
                "return_type": _return_type(handle, ret), "inputs_mutated": None, "oob_write": None,
            }
            if exc is not None:
                run["classification"] = "runtime_crash"
                run["pattern"] = f"{exc['type']}: {exc['message']}".rstrip(": ")[:500]
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
                "nan_inf": metrics["nan_inf"], "inputs_mutated": mutated, "oob_write": oob,
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
    handle: KernelHandle,
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
) -> None:
    """성능 테스트 1개: 센티널 채움 1회 → warmup → timed. 적응·조기 종료 규칙은 설계 §8.

    ``cpus`` 는 주파수·스로틀 샘플링 대상 CPU(= taskset 으로 고정된 집합). **커널 로드 전에** 읽은 값을 넘겨야 한다:
    OpenMP 커널(libgomp)은 ``OMP_PROC_BIND=true`` 면 로드 시점에 주 스레드를 place 0(첫 코어)에 묶어
    ``os.sched_getaffinity(0)`` 이 CPU 하나로 좁아진다(실측). None 이면 지금 읽는다.
    """
    doc = writer.doc
    name, path = test
    bufs = Buffers(spec, load_test(path, spec), tol.sentinel)
    cpus = pinned_cpus() if cpus is None else list(cpus)

    doc.update({
        "test": name, **bufs.dims, "dims": dict(bufs.dims), "task": spec.task, "alignment": ALIGNMENT_BYTES,
        "warmup": warmup, "repeats": repeats, "repeats_reduced": False,
        "warmup_ms": [], "timed_ms": [],
        "median_ms": None, "mean_ms": None, "min_ms": None, "max_ms": None, "cv": None, "gflops": None,
        "nondeterministic_runs": 0, "nondeterministic_bitwise_runs": 0,
        "cpu_mhz_samples": [], "cpu_mhz_median": None, "cpu_mhz_min": None,
        "throttle_events": None, "temp_c_start": package_temp_c(), "temp_c_end": None,
        "final_check": None, "inputs_mutated": None, "oob_write": None,
        "classification": None, "error_detail": "", "exception": None,
        "working_set_mb": bufs.working_set_mb(), "pinned_cpus": cpus,
    })
    throttle_start = throttle_count(cpus)
    doc["throttle_start"] = throttle_start

    bufs.restore_inputs()
    bufs.fill_sentinel()
    n_warm, n_timed = warmup, repeats
    warm_done = timed_done = 0
    call_idx = 0
    C_first: np.ndarray | None = None
    last_ret: Any = None
    crashed = False
    early_exit = False
    writer.flush()

    while warm_done < n_warm or timed_done < n_timed:
        is_warm = warm_done < n_warm
        doc["in_progress"] = {"test": name, "call_idx": call_idx}
        writer.flush()
        elapsed_ms, ret, exc = call_kernel(handle, bufs)
        if exc is not None:
            doc["exception"] = exc
            doc["classification"] = "runtime_crash"
            doc["error_detail"] = f"{exc['type']}: {exc['message']}".rstrip(": ")[:500]
            crashed = True
            break
        last_ret = ret
        if is_warm:
            doc["warmup_ms"].append(elapsed_ms)
            warm_done += 1
        else:
            doc["timed_ms"].append(elapsed_ms)
            timed_done += 1
            doc["cpu_mhz_samples"].extend(cpu_mhz_samples(cpus))
            if C_first is None:
                C_first = bufs.out.view.copy()
            else:
                if not np.array_equal(bufs.out.view, C_first):
                    doc["nondeterministic_bitwise_runs"] += 1
                    if not outputs_close(bufs.out.view, C_first, tol):
                        doc["nondeterministic_runs"] += 1

        if call_idx == 0:
            # 적응·조기 종료: 첫 호출 시간으로 예산을 투영한다
            t1 = elapsed_ms / 1000.0
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
        writer.flush()

    doc["in_progress"] = None
    samples = doc["cpu_mhz_samples"]
    doc["cpu_mhz_median"] = float(np.median(samples)) if samples else None
    doc["cpu_mhz_min"] = float(min(samples)) if samples else None
    throttle_end = throttle_count(cpus)
    doc["throttle_events"] = (throttle_end - throttle_start) if (throttle_start is not None and throttle_end is not None) else None
    doc["temp_c_end"] = package_temp_c()

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.worker", description="LLM 커널 실행 워커 (설계 §8)")
    parser.add_argument("--backend", choices=("c", "python"), required=True)
    parser.add_argument("--artifact", type=Path, required=True, help="kernel.so 또는 kernel.py")
    parser.add_argument("--mode", choices=("import", "check", "bench"), required=True)
    parser.add_argument("--tests", default=None, help="npz 파일 경로(쉼표 구분); check 는 여러 개, bench 는 1개")
    parser.add_argument("--tolerance", type=Path, default=None, help="tolerance.yaml (check/bench 필수)")
    parser.add_argument("--signatures", type=Path, default=DEFAULT_SIGNATURES, help="과제 signatures.yaml (기본 gemm)")
    parser.add_argument("--out", type=Path, required=True, help="결과 JSON 경로")
    parser.add_argument("--check-repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--min-repeats", type=int, default=3)
    parser.add_argument("--budget-s", type=float, default=300.0)
    parser.add_argument("--args-mode", choices=("positional", "keyword"), default="positional")
    parser.add_argument("--max-mem-gb", type=float, default=8.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    doc: dict[str, Any] = {
        "mode": ns.mode, "backend": ns.backend, "artifact": str(ns.artifact),
        "complete": False, "in_progress": None, "load_ok": None, "load_error": None,
        "classification": None, "error_detail": "", "blas": None, "limits": None,
        "loadavg_start": None, "loadavg_end": None,
    }
    writer = ResultWriter(ns.out, doc)
    sys.dont_write_bytecode = True  # 커널(kernel.py/baseline.py) 옆에 __pycache__ 를 남기지 않는다
    try:
        dumpable = disable_core_dumps()
        limits = apply_memory_limit(ns.max_mem_gb)
        limits["dumpable"] = dumpable
        doc["limits"] = limits
        doc["blas"] = blas_identity()
        doc["loadavg_start"] = loadavg_1m()
        # taskset 으로 고정된 CPU 집합. 커널 로드 전에 읽는다(OpenMP 런타임이 주 스레드 친화도를 좁히기 전).
        affinity = pinned_cpus()
        doc["pinned_cpus"] = affinity
        writer.flush()

        if ns.mode in ("check", "bench"):
            if ns.tolerance is None:
                raise ValueError("--tolerance 가 필요하다 (check/bench)")
            tests = parse_tests(ns.tests)
            if not tests:
                raise ValueError("--tests 가 필요하다 (check/bench)")
            if ns.mode == "bench" and len(tests) != 1:
                raise ValueError(f"bench 모드는 테스트 1개만 받는다 (현재 {len(tests)}개)")
            tol = load_tolerance(ns.tolerance)
        else:
            tests = []
            tol = None

        spec = load_taskspec(ns.signatures)
        doc["task"] = spec.task
        handle, cls, detail = load_kernel(ns.backend, ns.artifact, ns.args_mode, spec)
        doc["load_ok"] = handle is not None
        doc["load_error"] = detail or None
        if handle is None:
            doc["classification"] = cls
            doc["error_detail"] = detail
        elif ns.mode == "import":
            doc["classification"] = "pass"
        elif ns.mode == "check":
            assert tol is not None
            run_check(handle, spec, tests, tol, max(1, ns.check_repeats), writer)
        else:
            assert tol is not None
            run_bench(handle, spec, tests[0], tol, warmup=max(0, ns.warmup), repeats=max(1, ns.repeats),
                      min_repeats=max(1, min(ns.min_repeats, ns.repeats)), budget_s=ns.budget_s, writer=writer,
                      cpus=affinity)

        doc["loadavg_end"] = loadavg_1m()
        doc["in_progress"] = None
        doc["complete"] = True
        writer.flush()
        return 0
    except Exception:  # 하네스 버그: 기록하고 rc=3
        tb = traceback.format_exc()
        doc["worker_error"] = tb[-TRACEBACK_TAIL_CHARS * 2 :]
        doc["complete"] = False
        try:
            writer.flush()
        except Exception as e2:  # 결과 파일조차 못 쓰면 stderr 에만 남긴다
            print(f"{_LOG} 결과 파일 쓰기 실패: {e2}", file=sys.stderr)
        print(f"{_LOG} worker_error:\n{tb}", file=sys.stderr)
        return WORKER_ERROR_RC


if __name__ == "__main__":
    sys.exit(main())
