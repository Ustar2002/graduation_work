"""시그니처 렌더러 (설계 v3 §3): 과제 명세(``TaskSpec.args``)에서 언어별 함수 시그니처·매개변수 설명·매개변수 구조체를 만든다.

과제 12개 × 언어 7개의 문자열을 ``signatures.yaml`` 에 손으로 쓰지 않기 위한 것이다. YAML ``languages.<language>`` 에
해당 문자열이 있으면 그것이 우선한다(``signature_bundle``; 기존 4개 과제의 프롬프트가 바이트 단위로 같게 유지된다).

``render_signature`` 의 출력(정확한 문자열; ``{P}`` = ``", ".join(params)``)::

    backend   positional                                              keyword
    python    def {fn}({P}):                                          def {fn}(*, {P}):
    numba     @njit("{numba_signature}")\\ndef {fn}({P}):              @njit("{numba_signature}")\\ndef {fn}(*, {P}):
    halide    def {fn}({halide P}):                                   def {fn}(*, {halide P}):
    c         void {fn}({c_type name, ...})                           void {fn}(const {fn}_args* args)
    rust      #[no_mangle]\\npub unsafe extern "C" fn {fn}({name}: {rust_type}, ...)
                                                                      #[no_mangle]\\npub unsafe extern "C" fn {fn}(args: *const {Camel}Args)

- 매개변수 이름은 모든 언어에서 ``args`` 의 이름 그대로다(Rust 도 ``A``, ``M``).
- halide 의 매개변수는 입력 배열과 dim 뿐이다(출력 인자 없음; 빌드 함수가 출력 ``Func`` 을 돌려준다 — 설계 v3 §5.3).
- ``render_param_docs`` 는 결정적인 영어 문장이다. 사실만 적고(dtype·형태·읽기 전용/출력·``doc``·반환 규칙) 수치 라이브러리
  이름을 쓰지 않는다 — 라이브러리 사용 여부는 관찰 대상이다(CLAUDE.md §4).
- C/Rust 의 keyword 모드에서는 매개변수 설명 맨 앞에 ``keyword_struct_intro`` 문단("``args`` 가 가리키는 구조체의 필드")이
  붙는다. 설명 본문이 렌더링한 것이든 YAML ``param_docs`` 든 같다(``signature_bundle``) — args=keyword 조건의 프롬프트가
  12개 과제에서 같은 틀이 된다. Rust 문단은 타입을 스코프로 가져오는 ``mod``/``use`` 문도 사실로 적는다.

이 모듈은 순수 함수만 있다(파일·서브프로세스 없음). generate·extract·run·devcheck 가 같이 쓴다.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from harness.taskspec import ArgSpec, TaskSpec, load_taskspec

_LOG = "[sigs]"

BACKENDS: tuple[str, ...] = ("python", "numba", "halide", "c", "rust")
ARGS_MODES: tuple[str, ...] = ("positional", "keyword")

# dim 인자에 ``doc`` 이 없을 때의 설명(설계 v3 §3)
DIM_DEFAULT_DOC = "always consistent with the array shapes above"
# ``doc`` 이 없고 어떤 배열 형태 식에도 나오지 않는 스칼라 인자(예: jacobi2d 의 T)의 설명 — 위 문장은 그런 인자에 틀린 말이다
SCALAR_DEFAULT_DOC = "a scalar parameter described in the task specification"

C_ARGS_HEADER_NAME = "kernel_args.h"
C_ARGS_HEADER_GUARD = "KERNEL_ARGS_H"
RUST_ARGS_MODULE_NAME = "kernel_args.rs"
RUST_ARGS_MODULE = "kernel_args"  # ``src/kernel_args.rs`` 의 모듈 이름(``mod kernel_args;``)
# keyword 모드에서 매개변수를 구조체 하나로 받는 backend
STRUCT_BACKENDS: tuple[str, ...] = ("c", "rust")

_C_ELEM = {"float32": "float", "int32": "int"}
_RUST_ELEM = {"float32": "f32", "int32": "i32"}
_HALIDE_ELEM = {"float32": "hl.Float(32)", "int32": "hl.Int(32)"}
_ATOM_RE = re.compile(r"^(?:[A-Za-z_]\w*|\d+)$")
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")


# ----------------------------------------------------------------------------- 타입·이름


def camel(function: str) -> str:
    """snake_case → CamelCase: ``"spmv_csr"`` → ``"SpmvCsr"``, ``"jacobi2d"`` → ``"Jacobi2d"``."""
    return "".join(part[:1].upper() + part[1:] for part in function.split("_") if part)


def _check_backend(backend: str) -> None:
    if backend not in BACKENDS:
        raise ValueError(f"알 수 없는 backend: {backend!r} (허용: {BACKENDS})")


def _check_args_mode(args_mode: str) -> None:
    if args_mode not in ARGS_MODES:
        raise ValueError(f"알 수 없는 args_mode: {args_mode!r} (허용: {ARGS_MODES})")


def c_type(arg: ArgSpec) -> str:
    """C 매개변수 타입: in ``const float*``/``const int*``, out ``float*``/``int*``, dim ``int``."""
    if arg.kind == "dim":
        return "int"
    elem = _C_ELEM[str(arg.dtype)]
    return f"const {elem}*" if arg.kind == "in" else f"{elem}*"


def rust_type(arg: ArgSpec) -> str:
    """Rust 매개변수 타입: ``*const f32`` | ``*const i32`` | ``*mut f32`` | ``*mut i32`` | ``i32``."""
    if arg.kind == "dim":
        return "i32"
    elem = _RUST_ELEM[str(arg.dtype)]
    return f"*const {elem}" if arg.kind == "in" else f"*mut {elem}"


def numba_type(spec: TaskSpec, arg: ArgSpec) -> str:
    """Numba 서명의 인자 타입: C-연속 배열 ``float32[::1]``(1-D)·``float32[:, ::1]``(2-D)·…, int32 도 같은 규칙, dim ``int64``."""
    if arg.kind == "dim":
        return "int64"
    ndim = len(arg.shape)
    if ndim < 1:
        raise ValueError(f"{spec.task}: 배열 인자 {arg.name!r} 에 shape 가 없다")
    return f"{arg.dtype}[" + ", ".join([":"] * (ndim - 1) + ["::1"]) + "]"


def numba_signature(spec: TaskSpec) -> str:
    """Numba 명시 서명 문자열: ``"void(" + ", ".join(numba_type) + ")"`` (인자 순서 = ``args``)."""
    return "void(" + ", ".join(numba_type(spec, a) for a in spec.args) + ")"


def halide_params(spec: TaskSpec) -> list[str]:
    """Halide 파이프라인 빌드 함수의 매개변수: 입력 배열과 dim 만(출력 인자 없음), ``args`` 순서."""
    return [a.name for a in spec.args if a.kind in ("in", "dim")]


# ----------------------------------------------------------------------------- 시그니처


def render_signature(spec: TaskSpec, backend: str, args_mode: str) -> str:
    """언어별 함수 시그니처(모듈 docstring 의 표 그대로). ``args_mode`` ∈ positional | keyword."""
    _check_backend(backend)
    _check_args_mode(args_mode)
    fn = spec.function
    keyword = args_mode == "keyword"
    if backend in ("python", "numba", "halide"):
        names = halide_params(spec) if backend == "halide" else spec.params
        plist = ", ".join(names)
        head = f"def {fn}(*, {plist}):" if keyword else f"def {fn}({plist}):"
        if backend == "numba":
            return f'@njit("{numba_signature(spec)}")\n{head}'
        return head
    if backend == "c":
        if keyword:
            return f"void {fn}(const {fn}_args* args)"
        return f"void {fn}(" + ", ".join(f"{c_type(a)} {a.name}" for a in spec.args) + ")"
    # rust
    prefix = f'#[no_mangle]\npub unsafe extern "C" fn {fn}('
    if keyword:
        return f"{prefix}args: *const {camel(fn)}Args)"
    return prefix + ", ".join(f"{a.name}: {rust_type(a)}" for a in spec.args) + ")"


def render_args_struct(spec: TaskSpec, backend: str) -> str:
    """키워드 모드의 매개변수 구조체 파일 전문. c: ``kernel_args.h``, rust: ``kernel_args.rs``, 그 외 ``""``.

    필드는 ``args`` 순서·이름 그대로다. C 는 헤더 가드 ``KERNEL_ARGS_H``, Rust 는 ``#[repr(C)]``.
    """
    _check_backend(backend)
    fn = spec.function
    if backend == "c":
        fields = "".join(f"    {c_type(a)} {a.name};\n" for a in spec.args)
        return (f"#ifndef {C_ARGS_HEADER_GUARD}\n#define {C_ARGS_HEADER_GUARD}\n\n"
                f"typedef struct {{\n{fields}}} {fn}_args;\n\n"
                f"#endif /* {C_ARGS_HEADER_GUARD} */\n")
    if backend == "rust":
        fields = "".join(f"    pub {a.name}: {rust_type(a)},\n" for a in spec.args)
        return f"#[repr(C)]\n#[allow(non_snake_case)]\npub struct {camel(fn)}Args {{\n{fields}}}\n"
    return ""


# ----------------------------------------------------------------------------- 매개변수 설명


def _shape_text(arg: ArgSpec) -> str:
    """``(M, K)`` / ``(n,)`` / ``(n_rows + 1,)``."""
    dims = [str(s).strip() for s in arg.shape]
    return "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"


def _count_text(arg: ArgSpec) -> str:
    """원소 수 식: ``M*K`` / ``n_rows + 1`` / ``(n + 1)*3``."""
    factors = [str(s).strip() for s in arg.shape]
    if len(factors) == 1:
        return factors[0]
    return "*".join(f if _ATOM_RE.match(f) else f"({f})" for f in factors)


def _doc_clause(doc: str) -> str:
    """``doc`` 를 문장 끝에 붙일 구절로(끝의 마침표는 호출자가 찍는다)."""
    return doc.strip().rstrip(".").rstrip()


def _shape_dim_names(spec: TaskSpec) -> set[str]:
    """배열 형태 식에 나오는 dim 이름(예: gemm 의 M, N, K). jacobi2d 의 ``T``, bfs 의 ``source`` 같은 스칼라는 빠진다."""
    used: set[str] = set()
    for a in spec.args:
        if a.kind != "dim":
            for expr in a.shape:
                used.update(_IDENT_RE.findall(str(expr)))
    return used & set(spec.dim_names)


def _dim_bullets(spec: TaskSpec, type_text: str) -> list[str]:
    """dim 인자 설명: (1) ``doc`` 이 없고 형태를 정하는 dim 은 한 줄로 묶어 ``DIM_DEFAULT_DOC``, (2) ``doc`` 이 있는 dim 은
    한 줄씩(``args`` 순서), (3) ``doc`` 도 없고 어떤 형태 식에도 나오지 않는 스칼라는 ``SCALAR_DEFAULT_DOC`` — 형태와 무관한
    인자에 "형태와 일치한다"는 틀린 문장을 쓰지 않는다(그런 인자에는 signatures.yaml 에 ``doc`` 을 주는 것이 원칙이다)."""
    shape_dims = _shape_dim_names(spec)
    lines: list[str] = []
    plain = [a.name for a in spec.args if a.kind == "dim" and not a.doc and a.name in shape_dims]
    if plain:
        lines.append("- " + ", ".join(f"`{n}`" for n in plain) + f": {type_text}; {DIM_DEFAULT_DOC}.")
    for a in spec.args:
        if a.kind == "dim" and a.doc:
            lines.append(f"- `{a.name}`: {type_text}; {_doc_clause(a.doc)}.")
    scalars = [a.name for a in spec.args if a.kind == "dim" and not a.doc and a.name not in shape_dims]
    if scalars:
        lines.append("- " + ", ".join(f"`{n}`" for n in scalars) + f": {type_text}; {SCALAR_DEFAULT_DOC}.")
    return lines


def _row_major_line(spec: TaskSpec) -> str | None:
    """다차원 배열이 있을 때의 선형 인덱스 규칙 한 줄(C/Rust). 2-D 는 설계 v3 §3 의 문장 그대로."""
    ndim = max((len(a.shape) for a in spec.args if a.kind != "dim"), default=1)
    if ndim < 2:
        return None
    idx = ", ".join(f"i{k}" for k in range(ndim))
    shape = ", ".join(f"D{k}" for k in range(ndim))
    linear = "i0"
    for k in range(1, ndim):
        linear = f"{linear}*D{k} + i{k}" if k == 1 else f"({linear})*D{k} + i{k}"
    return (f"- Multi-dimensional arrays are stored row-major: element ({idx}) of an array of shape ({shape}) "
            f"is at linear index {linear}.")


def _array_bullets_python(spec: TaskSpec) -> list[str]:
    lines: list[str] = []
    for a in spec.args:
        if a.kind == "dim":
            continue
        line = f"- `{a.name}`: array, dtype {a.dtype}, shape {_shape_text(a)}, C-contiguous, 64-byte aligned"
        line += ", read-only" if a.kind == "in" else "; the output buffer that must be filled"
        if a.doc:
            line += f"; {_doc_clause(a.doc)}"
        lines.append(line + ".")
    return lines


def _array_bullets_native(spec: TaskSpec, backend: str) -> list[str]:
    """C/Rust: 원소 타입·원소 수·(다차원이면) 형태·읽기 전용/출력."""
    elem = {"c": {"float32": "`float` (float32)", "int32": "`int` (32-bit)"},
            "rust": {"float32": "`f32`", "int32": "`i32`"}}[backend]
    lines: list[str] = []
    for a in spec.args:
        if a.kind == "dim":
            continue
        count = _count_text(a)
        noun = "element" if count == "1" else "elements"
        line = f"- `{a.name}`: {elem[str(a.dtype)]} array with {count} {noun}"
        if len(a.shape) > 1:
            line += f", shape {_shape_text(a)}"
        line += ", read-only" if a.kind == "in" else "; the output buffer that must be filled"
        if a.doc:
            line += f"; {_doc_clause(a.doc)}"
        lines.append(line + ".")
    return lines


def keyword_struct_intro(spec: TaskSpec, backend: str) -> str:
    """C/Rust keyword 모드의 매개변수 설명 맨 앞에 붙는 안내 문단(한 줄, 영어): "``args`` 가 가리키는 구조체의 필드".
    구조체를 쓰지 않는 backend(python·numba·halide)는 ``""``.

    Rust 문단은 타입을 스코프로 가져오는 방법도 사실로 적는다. 시그니처(설계 v3 §3)의 ``*const {Camel}Args`` 는
    ``mod kernel_args;`` 만으로는 풀리지 않는다 — ``use kernel_args::{Camel}Args;`` 가 없으면
    ``error[E0412]: cannot find type`` 이다(rustc 1.92 실측; ``tests/test_sigs.py`` 가 rustc 로 확인한다).
    """
    _check_backend(backend)
    fn = spec.function
    if backend == "c":
        return (f"`args` points to one `{fn}_args` structure (declared in `{C_ARGS_HEADER_NAME}`); "
                "the structure itself is read-only. Its fields:")
    if backend == "rust":
        struct = f"{camel(fn)}Args"
        return (f"`args` points to one `{struct}` structure (declared in the module file `{RUST_ARGS_MODULE_NAME}`; "
                f"`mod {RUST_ARGS_MODULE};` followed by `use {RUST_ARGS_MODULE}::{struct};` brings the type into scope); "
                "the structure itself is read-only. Its fields:")
    return ""


def render_param_docs(spec: TaskSpec, backend: str, args_mode: str) -> str:
    """프롬프트의 매개변수 설명(영어, 결정적, 끝 개행 없음).

    배열마다 dtype·형태·읽기 전용/출력 여부·``doc``, dim 마다 ``doc``(없으면 ``DIM_DEFAULT_DOC``; 형태 식에 나오지 않는
    스칼라는 ``SCALAR_DEFAULT_DOC``), 마지막에 반환 규칙.
    C/Rust 는 다차원 배열이 있으면 행우선 선형 인덱스 규칙 한 줄을, halide 는 좌표 규칙(설계 v3 §5.3)을 넣는다.
    C/Rust 의 keyword 모드는 맨 앞에 ``keyword_struct_intro`` 문단과 빈 줄이 붙고, 그 뒤는 positional 설명과 같다.
    """
    _check_backend(backend)
    _check_args_mode(args_mode)
    fn = spec.function
    out = spec.output
    lines: list[str] = []

    if backend in ("python", "numba"):
        lines += _array_bullets_python(spec)
        lines += _dim_bullets(spec, "Python `int`" if backend == "python" else "`int64`")
        if backend == "python":
            lines.append(f"- Returns: `None` (results are written into `{out.name}`).")
        else:
            lines.append(f"- Returns: nothing (results are written into `{out.name}`).")
        return "\n".join(lines)

    if backend == "halide":
        for a in spec.args:
            if a.kind != "in":
                continue
            ndim = len(a.shape)
            line = (f"- `{a.name}`: `hl.ImageParam` of type `{_HALIDE_ELEM[str(a.dtype)]}` with {ndim} "
                    f"dimension{'s' if ndim != 1 else ''}; the input array of shape {_shape_text(a)}, read-only")
            if a.doc:
                line += f"; {_doc_clause(a.doc)}"
            lines.append(line + ".")
        lines += _dim_bullets(spec, "Python `int`, the actual value for this call")
        ndim = len(out.shape)
        ret = (f"- Returns: a `hl.Func` with {ndim} dimension{'s' if ndim != 1 else ''} and element type "
               f"`{_HALIDE_ELEM[str(out.dtype)]}` that defines the output array `{out.name}` of shape {_shape_text(out)}")
        if out.doc:
            ret += f" ({_doc_clause(out.doc)})"
        lines.append(ret + f". `{fn}` has no output parameter: the harness realizes the returned `hl.Func` into the output buffer.")
        lines.append("- Coordinates: every buffer is created with `hl.Buffer(array)` (default `reverse_axes=True`), so for a "
                     "row-major array of shape (D0, D1) the element at row i0 and column i1 is `X[i1, i0]` (the last array "
                     "axis is the first coordinate); a 1-D array is indexed as `x[i]`. Define the output `hl.Func` with the same rule.")
        return "\n".join(lines)

    # c / rust
    if args_mode == "keyword":
        lines.append(keyword_struct_intro(spec, backend))
        lines.append("")
    row_major = _row_major_line(spec)
    if row_major:
        lines.append(row_major)
    lines += _array_bullets_native(spec, backend)
    n_arrays = sum(1 for a in spec.args if a.kind != "dim")
    if n_arrays > 1:
        lines.append("- All array arguments start at 64-byte-aligned addresses and do not overlap.")
    else:
        lines.append("- The array argument starts at a 64-byte-aligned address.")
    lines += _dim_bullets(spec, "`int`" if backend == "c" else "`i32`")
    lines.append(f"- Return value: none (results are written into `{out.name}`).")
    return "\n".join(lines)


# ----------------------------------------------------------------------------- 묶음


def _yaml_string(entry: dict[str, Any], key: str) -> str | None:
    value = entry.get(key)
    return value if isinstance(value, str) and value else None


def _yaml_param_docs(entry: dict[str, Any]) -> str | None:
    """YAML ``param_docs`` 에서 끝의 개행을 뗀 문자열(비어 있으면 None).

    ``generate.load_signature`` 가 쓰던 규칙(``rstrip("\\n")``)과 같다: ``|-`` 대신 ``|`` 블록으로 써도 프롬프트에 빈 줄이
    하나 더 들어가지 않는다(렌더링한 설명도 끝 개행이 없다)."""
    value = entry.get("param_docs")
    if not isinstance(value, str):
        return None
    return value.rstrip("\n") or None


def signature_bundle(spec: TaskSpec, language: str, backend: str, args_mode: str) -> dict[str, Any]:
    """한 (과제, 언어, 인자 방식)의 시그니처 묶음. generate(프롬프트)·extract(검사)·run(빌드)이 같이 쓴다.

    반환 키:

    - ``function``: 함수 이름.
    - ``params``: 그 backend 의 매개변수 이름(halide 는 ``halide_params``, 그 밖은 ``args`` 전부).
    - ``signature``: YAML ``languages.<language>.<args_mode>`` 문자열이 있으면 **그대로**, 없으면 ``render_signature``.
    - ``param_docs``: YAML ``languages.<language>.param_docs`` 가 있으면 그 문자열(끝의 개행만 뗀다), 없으면
      ``render_param_docs``. YAML 의 설명은 인자별 설명이라 인자 방식과 무관하게 쓰인다 — positional 과 python 계열의
      keyword 는 YAML 문자열 그대로이고(기존 프롬프트의 바이트 동일성), 매개변수가 구조체 하나로 바뀌는 C/Rust 의 keyword
      모드에서만 그 앞에 ``keyword_struct_intro`` 문단과 빈 줄을 붙인다(렌더링한 설명과 같은 틀; 기존 4개 과제의
      ``keyword: null`` 항목이 여기 해당한다).
    - ``args_struct``: keyword 모드의 c/rust 는 ``render_args_struct`` 전문, 그 밖(positional·다른 backend)은 ``""``.
    - ``numba_signature``: numba backend 면 ``numba_signature(spec)``, 그 밖은 ``""``.
    - ``source``: **시그니처 기준**이다 — ``signature`` 가 YAML 에서 왔으면 ``"yaml"``, 렌더링했으면 ``"rendered"``
      (``param_docs`` 의 출처는 알려 주지 않는다).
    """
    _check_backend(backend)
    _check_args_mode(args_mode)
    entry = spec.languages_raw.get(language) or {}
    yaml_signature = _yaml_string(entry, args_mode)
    yaml_docs = _yaml_param_docs(entry)
    if yaml_docs is None:
        param_docs = render_param_docs(spec, backend, args_mode)
    elif args_mode == "keyword" and backend in STRUCT_BACKENDS:
        param_docs = f"{keyword_struct_intro(spec, backend)}\n\n{yaml_docs}"
    else:
        param_docs = yaml_docs
    return {
        "function": spec.function,
        "params": halide_params(spec) if backend == "halide" else spec.params,
        "signature": yaml_signature if yaml_signature is not None else render_signature(spec, backend, args_mode),
        "param_docs": param_docs,
        "args_struct": render_args_struct(spec, backend) if args_mode == "keyword" else "",
        "numba_signature": numba_signature(spec) if backend == "numba" else "",
        "source": "yaml" if yaml_signature is not None else "rendered",
    }


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """과제의 시그니처 묶음을 출력한다(진단용): ``python -m harness.sigs --task gemm --backend c --args-mode keyword``."""
    from harness.paths import TASKS_DIR

    parser = argparse.ArgumentParser(prog="python -m harness.sigs", description="시그니처·매개변수 설명 렌더러 (설계 v3 §3)")
    parser.add_argument("--task", default=None, help="과제 이름(tasks/<task>/signatures.yaml)")
    parser.add_argument("--signatures", type=Path, default=None, help="signatures.yaml 경로(--task 대신)")
    parser.add_argument("--language", default=None, help="YAML languages 의 키(기본: backend 이름; YAML 우선 규칙 확인용)")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--args-mode", choices=ARGS_MODES, default="positional")
    parser.add_argument("--json", action="store_true", help="묶음을 JSON 으로 출력")
    ns = parser.parse_args(argv)
    if (ns.task is None) == (ns.signatures is None):
        parser.error("--task 와 --signatures 중 하나만 준다")
    path = ns.signatures if ns.signatures is not None else TASKS_DIR / ns.task / "signatures.yaml"
    try:
        spec = load_taskspec(path)
    except (OSError, ValueError) as e:
        print(f"{_LOG} 과제 명세를 읽을 수 없다: {e}", file=sys.stderr)
        return 2
    bundle = signature_bundle(spec, ns.language or ns.backend, ns.backend, ns.args_mode)
    if ns.json:
        print(json.dumps(bundle, ensure_ascii=False, indent=2))
        return 0
    print(f"# source={bundle['source']} function={bundle['function']} params={', '.join(bundle['params'])}")
    print(bundle["signature"])
    print()
    print(bundle["param_docs"])
    if bundle["args_struct"]:
        print()
        print(bundle["args_struct"], end="")
    return 0


if __name__ == "__main__":
    # YAML 원문(param_docs·doc)의 임의 글자를 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
