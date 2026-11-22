"""C ABI 커널 (설계 v3 §5.2): 공유 라이브러리(C 산출물, Rust ``cdylib``)를 ctypes 로 열어 부른다.

- positional: ``fn(*args)`` — 배열 인자는 ``POINTER(c_float | c_int32)``, dim 인자는 ``c_int``, 순서는 과제 명세의 ``args``.
- keyword: ``fn(byref(ArgsStruct(...)))`` — 매개변수 구조체(C ``kernel_args.h`` 의 ``{fn}_args``, Rust ``#[repr(C)]`` 구조체;
  ``harness.sigs.render_args_struct``)와 같은 순서·같은 타입의 ``ctypes.Structure`` 를 만들어 그 포인터 하나를 넘긴다.

로드 실패: 라이브러리를 열 수 없으면(``OSError``: 미정의 심벌 등) ``compile_error``, 함수 심벌이 없으면
``signature_mismatch``. 반환값은 쓰지 않는다(``restype=None``).
"""

from __future__ import annotations

import ctypes
import functools
from collections.abc import Mapping
from ctypes import POINTER, c_float, c_int, c_int32
from pathlib import Path
from typing import Any

import numpy as np

from harness.kernels import KernelError, KernelHandle, KernelOptions, Loader, one_line
from harness.taskspec import ArgSpec, TaskSpec

C_TYPES: dict[str, Any] = {"float32": c_float, "int32": c_int32}


def c_arg_type(arg: ArgSpec) -> Any:
    """인자 하나의 ctypes 타입: dim 은 ``c_int``, 배열은 원소 타입의 포인터(``const`` 는 ABI 에 나타나지 않는다)."""
    return c_int if arg.kind == "dim" else POINTER(C_TYPES[str(arg.dtype)])


def args_struct_type(spec: TaskSpec) -> type[ctypes.Structure]:
    """키워드 모드의 매개변수 구조체: 필드는 ``args`` 순서·이름 그대로, 정렬은 플랫폼 C ABI(ctypes 기본)."""
    fields = [(arg.name, c_arg_type(arg)) for arg in spec.args]
    return type(f"{spec.function}_args", (ctypes.Structure,), {"_fields_": fields})


class CKernel(KernelHandle):
    def __init__(self, fn: Any, spec: TaskSpec, struct_type: type[ctypes.Structure] | None) -> None:
        self._fn = fn
        self._spec = spec
        self._struct_type = struct_type  # None 이면 positional
        self._struct: ctypes.Structure | None = None
        self._call_args: tuple[Any, ...] = ()

    def prepare(self, dims: Mapping[str, int], arrays: Mapping[str, np.ndarray] | None = None) -> float | None:
        if arrays is None:
            return None
        values = [int(dims[arg.name]) if arg.kind == "dim" else arrays[arg.name].ctypes.data_as(c_arg_type(arg))
                  for arg in self._spec.args]
        if self._struct_type is None:
            self._call_args = tuple(values)
        else:
            self._struct = self._struct_type(*values)
            self._call_args = (ctypes.byref(self._struct),)
        return None

    def invoke(self) -> Any:
        self._fn(*self._call_args)
        return None


def load(artifact: Path, spec: TaskSpec, *, args_mode: str) -> CKernel:
    """공유 라이브러리를 열고 과제 함수를 찾는다. 경로는 절대 경로로 바꿔 연다 — 상대 경로는 ``dlopen``/``LoadLibrary`` 가
    현재 디렉터리가 아니라 라이브러리 검색 경로에서 찾는다."""
    try:
        lib = ctypes.CDLL(str(Path(artifact).resolve()))
    except OSError as e:
        raise KernelError("compile_error", one_line(e)) from None
    try:
        fn = getattr(lib, spec.function)
    except AttributeError as e:
        raise KernelError("signature_mismatch", f"symbol {spec.function} not found: {e}") from None
    struct_type = args_struct_type(spec) if args_mode == "keyword" else None
    fn.argtypes = [POINTER(struct_type)] if struct_type is not None else [c_arg_type(arg) for arg in spec.args]
    fn.restype = None
    return CKernel(fn, spec, struct_type)


def make_loader(options: KernelOptions) -> Loader:
    return functools.partial(load, args_mode=options.args_mode)
