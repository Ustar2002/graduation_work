"""Python 함수 커널 (설계 v3 §8.2–8.3): ``kernel.py`` 를 모듈로 임포트해 최상위 함수를 부른다. ``python`` 과 ``numba`` 가 같은
로드·호출 경로를 쓴다.

- positional: ``fn(*args)``, keyword: ``fn(**kwargs)`` — 배열 인자는 과제 형태의 N-D 뷰, dim 인자는 Python ``int``.
- numba: 모듈 임포트 때 명시 서명으로 **즉시 컴파일**돼 있어야 한다. 로드한 객체가 Numba 디스패처가 아니거나 ``signatures`` 가
  비어 있으면(``@njit`` 만 쓴 지연 컴파일) ``signature_mismatch``/``numba_signature_missing`` — 지연 컴파일이면 첫 측정 호출에
  JIT 시간이 섞인다. 그다음 디스패처의 새 컴파일을 끄고(``disable_compile``), 하네스가 넘길 인자 타입을 컴파일된 서명 가운데
  하나가 **컴파일 없이** 받는지 본다(``check_accepts_harness_args``; 2026-11-12 결정). 받지 않으면(예: float64 서명)
  ``signature_mismatch``/``numba_signature_rejects_args: …`` 로 로드가 실패한다 — 측정 호출에서 컴파일이 일어날 수 없다.
  읽기 전용 배열은 쓰기 가능한 배열 타입의 서명과 맞지 않으므로(``TypeError``) 워커는 쓰기 가능한 뷰를 넘기고 호출 뒤에 입력
  변조를 검사한다.
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

from harness.kernels import MODULE_NAME_DEFAULT, KernelError, KernelHandle, KernelOptions, Loader, exception_record, one_line
from harness.taskspec import TaskSpec

_LOG = "[kernels.pyfunc]"

MODULE_NAME = MODULE_NAME_DEFAULT
NUMBA_SIGNATURE_MISSING = "numba_signature_missing"
NUMBA_SIGNATURE_REJECTS_ARGS = "numba_signature_rejects_args"
# 과제 dtype → 하네스가 넘기는 배열의 NumPy dtype (worker.NP_TYPES 와 같다)
_NP_DTYPES: dict[str, Any] = {"float32": np.float32, "int32": np.int32}
# 병렬 진단의 상세도(1–4). 4 여야 융합 시도("fusion succeeded"/"fusion failed")가 나온다 — harness.perf_summary 가 센다.
NUMBA_DIAGNOSTICS_LEVEL = 4
NUMBA_DIAGNOSTICS_CHARS = 8000


def load_function(artifact: Path, name: str, *, join_error_lines: bool = False,
                  module_name: str = MODULE_NAME) -> Callable[..., Any]:
    """``artifact`` 를 모듈 ``module_name`` (기본 ``kernel``)으로 임포트하고 최상위 호출 가능 객체 ``name`` 을 돌려준다
    (python·numba·halide 공용). 교차 측정의 기준선은 ``baseline_kernel`` 로 올려 커널 모듈을 덮지 않는다.

    ``join_error_lines``: 임포트 예외의 한 줄 설명에 메시지의 모든 줄을 잇는다(Numba·Halide 의 여러 줄 오류)."""
    mspec = importlib.util.spec_from_file_location(module_name, str(artifact))
    if mspec is None or mspec.loader is None:
        raise KernelError("compile_error", f"cannot create module spec for {artifact}")
    module = importlib.util.module_from_spec(mspec)
    sys.modules[module_name] = module
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
    backend = "python"
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
    backend = "numba"
    join_error_lines = True

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


def harness_arg_types(spec: TaskSpec) -> dict[str, Any]:
    """하네스가 넘길 인자의 Numba 타입(인자 이름 → 타입). 배열은 ``Buffers`` 의 뷰와 같은 꼴(C-연속·정렬·쓰기 가능, 과제의
    차원 수)의 길이 1 배열, dim 은 Python ``int`` 로 구한다 — 값과 크기는 타입에 들어가지 않는다."""
    import numba

    return {a.name: numba.typeof(1 if a.kind == "dim" else np.zeros((1,) * len(a.shape), dtype=_NP_DTYPES[str(a.dtype)]))
            for a in spec.args}


def check_accepts_harness_args(fn: Any, spec: TaskSpec, args_mode: str) -> None:
    """디스패처가 하네스가 넘길 인자 타입을 **컴파일 없이** 받는지 본다(2026-11-12 결정). 아니면 ``KernelError``.

    먼저 새 서명의 컴파일을 끈다(명시 서명으로 데코레이트한 함수는 Numba 가 이미 꺼 두었다; ``njit`` 뒤 ``fn.compile(sig)`` 로
    서명을 더한 함수도 측정 호출에서 컴파일하지 않게 한다). 그 상태의 디스패처와 같은 규칙(안전하지 않은 변환 허용, 동률은
    오류)으로 컴파일된 서명 가운데 맞는 것을 찾는다(``Dispatcher.fold_argument_types`` 는 키워드 전용 정의에서 IndexError 를
    내므로 쓰지 않는다). 키워드 모드에서는 인자가 ``py_func`` 의 매개변수 순서로 접힌다. 매개변수 이름·개수가 과제와 맞지
    않으면 판정하지 않는다: 그 호출은 TypeError(runtime_crash)로 끝난다(python 백엔드와 같다)."""
    fn.disable_compile()
    types = harness_arg_types(spec)
    params = list(inspect.signature(fn.py_func).parameters)
    if args_mode == "keyword":
        if sorted(params) != sorted(types):
            return
        argtypes = tuple(types[p] for p in params)
    else:
        if len(params) != len(spec.args):
            return
        argtypes = tuple(types[a.name] for a in spec.args)
    shown = ", ".join(map(str, argtypes))
    try:
        sig = fn.typingctx.resolve_overload(fn.py_func, fn.nopython_signatures, argtypes, {},
                                            allow_ambiguous=False, unsafe_casting=True)
    except TypeError:
        raise KernelError("signature_mismatch",
                          f"{NUMBA_SIGNATURE_REJECTS_ARGS}: ambiguous compiled signatures for ({shown})") from None
    if sig is None:
        print(f"{_LOG} {spec.function} 의 컴파일된 서명 {[str(s) for s in fn.signatures]} 가운데 ({shown}) 를 받는 것이 없다",
              file=sys.stderr)
        raise KernelError("signature_mismatch",
                          f"{NUMBA_SIGNATURE_REJECTS_ARGS}: no compiled signature of {spec.function} accepts ({shown})")


def load(artifact: Path, spec: TaskSpec, *, backend: str, args_mode: str, module_name: str = MODULE_NAME) -> PyKernel:
    if backend != "numba":
        return PyKernel(load_function(artifact, spec.function, module_name=module_name), spec, args_mode)
    from numba.extending import is_jitted

    fn = load_function(artifact, spec.function, join_error_lines=True, module_name=module_name)
    if not is_jitted(fn):
        print(f"{_LOG} {spec.function} 은 Numba 디스패처가 아니다({type(fn).__name__})", file=sys.stderr)
        raise KernelError("signature_mismatch", NUMBA_SIGNATURE_MISSING)
    if not fn.signatures:
        print(f"{_LOG} {spec.function} 에 컴파일된 서명이 없다(명시 서명 없는 지연 컴파일)", file=sys.stderr)
        raise KernelError("signature_mismatch", NUMBA_SIGNATURE_MISSING)
    check_accepts_harness_args(fn, spec, args_mode)
    return NumbaKernel(fn, spec, args_mode)


def make_loader(backend: str, options: KernelOptions) -> Loader:
    if backend == "numba":
        # 커널보다 먼저 Numba 를 임포트하고 스레딩 계층을 띄운다: 여기서 나는 예외는 커널의 compile_error 가 아니라 도구 문제다.
        # 계층을 띄우면 libgomp 가 올라와 주 스레드를 묶을 수 있다 — 워커는 그 전에 BLAS 예열과 고정 집합 읽기를 끝낸다.
        start_threading_layer()
    return functools.partial(load, backend=backend, args_mode=options.args_mode, module_name=options.module_name)
