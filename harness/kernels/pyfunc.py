"""Python 함수 커널 (설계 v3 §5.1–5.2): ``kernel.py`` 를 모듈로 임포트해 최상위 함수를 부른다. ``python`` 과 ``numba`` 가 같은
로드·호출 경로를 쓴다.

- positional: ``fn(*args)``, keyword: ``fn(**kwargs)`` — 배열 인자는 과제 형태의 N-D 뷰, dim 인자는 Python ``int``.
- numba: 모듈 임포트 때 명시 서명으로 **즉시 컴파일**돼 있어야 한다. 로드한 객체가 Numba 디스패처가 아니거나 ``signatures`` 가
  비어 있으면(``@njit`` 만 쓴 지연 컴파일) ``signature_mismatch``/``numba_signature_missing`` — 지연 컴파일이면 첫 측정 호출에
  JIT 시간이 섞인다. 읽기 전용 배열은 쓰기 가능한 배열 타입의 서명과 맞지 않으므로(``TypeError``) 워커는 쓰기 가능한 뷰를
  넘기고 호출 뒤에 입력 변조를 검사한다.
- numba 의 스레딩 계층(``NUMBA_THREADING_LAYER``)은 커널보다 먼저 띄운다(``make_loader``). 계층을 로드할 수 없으면
  ``parallel=True`` 커널의 **임포트**가 ``ValueError`` 로 실패한다(실측) — 하네스가 고른 도구의 문제가 커널의
  ``compile_error`` 로 기록되지 않게, LLM 코드를 실행하기 전에 같은 실패를 낸다.
- ``inspect`` (numba): ``signatures``, ``parallel_diagnostics`` (``parallel=True`` 로 컴파일한 함수만; stdout 을 가로챈다),
  ``threading_layer``, ``num_threads``.

로드 실패: 모듈 임포트 중의 예외(``SystemExit``·``KeyboardInterrupt`` 포함) → ``compile_error``; 함수가 없거나 호출할 수
없거나 코루틴 함수 → ``signature_mismatch``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import inspect
import io
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from harness.kernels import KernelError, KernelHandle, KernelOptions, Loader, exception_record, one_line
from harness.taskspec import TaskSpec

_LOG = "[kernels.pyfunc]"

MODULE_NAME = "kernel"
NUMBA_SIGNATURE_MISSING = "numba_signature_missing"
# 병렬 진단의 상세도(1–4). 4 여야 융합 시도("fusion succeeded"/"fusion failed")가 나온다 — harness.perf_summary 가 센다.
NUMBA_DIAGNOSTICS_LEVEL = 4
NUMBA_DIAGNOSTICS_CHARS = 8000


def load_function(artifact: Path, name: str, *, join_error_lines: bool = False) -> Callable[..., Any]:
    """``artifact`` 를 모듈 ``kernel`` 로 임포트하고 최상위 호출 가능 객체 ``name`` 을 돌려준다(python·numba·halide 공용).

    ``join_error_lines``: 임포트 예외의 한 줄 설명에 메시지의 모든 줄을 잇는다(Numba·Halide 의 여러 줄 오류)."""
    mspec = importlib.util.spec_from_file_location(MODULE_NAME, str(artifact))
    if mspec is None or mspec.loader is None:
        raise KernelError("compile_error", f"cannot create module spec for {artifact}")
    module = importlib.util.module_from_spec(mspec)
    sys.modules[MODULE_NAME] = module
    try:
        mspec.loader.exec_module(module)
    except BaseException as e:  # SystemExit/KeyboardInterrupt 도 임포트 실패로 본다
        raise KernelError("compile_error", one_line(e, join_lines=join_error_lines), exception_record(e)) from None
    fn = getattr(module, name, None)
    if fn is None:
        raise KernelError("signature_mismatch", f"module has no attribute {name}")
    if not callable(fn):
        raise KernelError("signature_mismatch", f"{name} is not callable ({type(fn).__name__})")
    if inspect.iscoroutinefunction(fn):
        raise KernelError("signature_mismatch", f"{name} is a coroutine function (async def)")
    return fn


def _not_prepared() -> Any:
    raise RuntimeError("prepare(dims, arrays) 를 먼저 불러야 한다")


class PyKernel(KernelHandle):
    has_return_value = True

    def __init__(self, fn: Callable[..., Any], spec: TaskSpec, args_mode: str) -> None:
        self._fn = fn
        self._spec = spec
        self._keyword = args_mode == "keyword"
        self._call: Callable[[], Any] = _not_prepared

    def prepare(self, dims: Mapping[str, int], arrays: Mapping[str, np.ndarray] | None = None) -> float | None:
        if arrays is None:
            return None
        values = [int(dims[arg.name]) if arg.kind == "dim" else arrays[arg.name] for arg in self._spec.args]
        if self._keyword:
            self._call = functools.partial(self._fn, **dict(zip(self._spec.params, values)))
        else:
            self._call = functools.partial(self._fn, *values)
        return None

    def invoke(self) -> Any:
        return self._call()


def start_threading_layer() -> int:
    """Numba 의 스레딩 계층을 띄우고(이미 떠 있으면 그대로) 스레드 수를 돌려준다. 설정된 계층(``NUMBA_THREADING_LAYER``)을
    로드할 수 없으면 ``ValueError`` — ``numba.get_num_threads()`` 가 스레드 풀을 띄우면서 낸다."""
    import numba

    return int(numba.get_num_threads())


class NumbaKernel(PyKernel):
    def _collect(self, info: dict[str, Any], dims: Mapping[str, int] | None) -> None:
        import numba

        fn: Any = self._fn
        info.update({"signatures": [str(sig) for sig in fn.signatures], "parallel_diagnostics": "",
                     "threading_layer": None, "num_threads": None})
        info["num_threads"] = start_threading_layer()
        info["threading_layer"] = str(numba.threading_layer())
        if fn.targetoptions.get("parallel"):  # parallel=True 가 아닌 함수에는 진단이 없다(RuntimeError)
            captured = io.StringIO()
            with contextlib.redirect_stdout(captured):  # 진단은 stdout 에 찍힌다: 워커의 stdout 으로 새면 안 된다
                fn.parallel_diagnostics(level=NUMBA_DIAGNOSTICS_LEVEL)
            info["parallel_diagnostics"] = captured.getvalue()[:NUMBA_DIAGNOSTICS_CHARS]


def load(artifact: Path, spec: TaskSpec, *, backend: str, args_mode: str) -> PyKernel:
    if backend != "numba":
        return PyKernel(load_function(artifact, spec.function), spec, args_mode)
    from numba.extending import is_jitted

    fn = load_function(artifact, spec.function, join_error_lines=True)
    if not is_jitted(fn):
        print(f"{_LOG} {spec.function} 은 Numba 디스패처가 아니다({type(fn).__name__})", file=sys.stderr)
        raise KernelError("signature_mismatch", NUMBA_SIGNATURE_MISSING)
    if not fn.signatures:
        print(f"{_LOG} {spec.function} 에 컴파일된 서명이 없다(명시 서명 없는 지연 컴파일)", file=sys.stderr)
        raise KernelError("signature_mismatch", NUMBA_SIGNATURE_MISSING)
    return NumbaKernel(fn, spec, args_mode)


def make_loader(backend: str, options: KernelOptions) -> Loader:
    if backend == "numba":
        # 커널보다 먼저 Numba 를 임포트하고 스레딩 계층을 띄운다: 여기서 나는 예외는 커널의 compile_error 가 아니라 도구 문제다
        start_threading_layer()
    return functools.partial(load, backend=backend, args_mode=options.args_mode)
