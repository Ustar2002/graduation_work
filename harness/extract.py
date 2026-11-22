"""응답 텍스트에서 코드 블록을 추출하고 함수 시그니처를 검사한다 (설계 §6, v3 §7).

``backend`` ∈ python | numba | halide | c | rust. 기대 시그니처는 ``bundle``(``harness.sigs.signature_bundle`` 의 결과)이
있으면 거기서(함수 이름·매개변수·C 프로토타입·Rust 시그니처), 없으면 ``signature_spec``(signatures.yaml 사전)에서 읽는다.
``bundle`` 없이 부르면 python·c 는 v3 이전과 똑같이 동작한다. halide·rust 는 ``bundle`` 이 있어야 한다(halide 의 매개변수는
출력 인자가 빠진 목록이고 Rust 는 타입이 필요한데, 둘 다 ``signature_spec`` 만으로는 알 수 없다).

절차:
1. 펜스 블록(``` 또는 ~~~, CRLF 허용) 수집. 닫히지 않은 펜스는 그 뒤 텍스트 전체를 블록으로 보고 ``unterminated_fence``.
2. 후보: 무태그 또는 backend 의 태그(대소문자 무시) — python·numba·halide → {python, py, python3}; c → {c}(``cpp`` 등 제외);
   rust → {rust, rs}. 태그 있는 것 우선.
3. 함수 정의를 포함하는 후보만; 여러 개면 가장 긴 것(``multiple_blocks_chose:<전체 블록 인덱스>``).
   정의가 있는 후보가 없으면 후보 중 가장 긴 것. 펜스가 없고 본문에 정의가 있으면 본문 전체(``bare_code``). 아니면 ``no_code_block``.
4. 시그니처 검사:
   - python·numba·halide: ``ast.parse`` 실패 → ok=True, signature_ok=None, ``syntax_error_deferred``(빌드 단계가 잡는다).
     최상위 ``FunctionDef``(데코레이터 허용 — numba 의 데코레이터 검사는 빌드 단계). ``AsyncFunctionDef`` → ``param_mismatch:async_def``.
     positional: ``posonlyargs+args`` 이름 == params, kwonly/vararg/kwarg 없음(기본값 있는 추가 매개변수도 불허).
     keyword: ``kwonlyargs`` == params, positional 없음. 기본값·주석(annotation)이 있으면 signature_exact=False.
     halide 의 params 는 입력 배열과 dim 뿐이다(``bundle["params"]`` = ``sigs.halide_params``).
   - c: ``work_dir`` 가 있으면 ``gcc -E -std=gnu11 -fopenmp kernel.c`` 로 전처리하고(procutil, 10 s, 코드 실행 없음;
     컴파일러는 ``cc`` 인자 — 호출자가 설정의 ``build.cc`` 를 넘긴다 — 이고 없으면 gcc)
     ``# N "kernel.c"`` 라인 마커로 kernel.c 유래 줄만 남긴 텍스트를 검사(주석·매크로 해소). 실패하면 원문에서 주석을 제거하고
     객체형 ``#define`` 만 치환한 뒤 검사하고 ``preprocess_failed`` 를 병기한다. keyword 모드(``bundle["args_struct"]``)면
     ``#include "kernel_args.h"`` 가 풀리도록 전처리 전에 그 헤더를 ``work_dir`` 에 써 둔다(빌드 단계가 같은 내용을 다시 쓴다).
     정의(``)`` 뒤 공백·개행·``__attribute__((...))`` 허용 후 ``{``)를 찾는다; 선언만 있으면 ``function_not_found``.
     매개변수 목록 안의 괄호는 3단계 중첩까지 허용한다(``__attribute__((aligned(64)))`` 가 매개변수에 붙는 경우).
     ``static`` 판정은 그 정의의 지정자 토큰열(직전 ``;`` 또는 ``}`` 이후)에서만 → ``static_linkage``.
     매개변수: ``__attribute__`` 제거, 소스의 단순 ``typedef <타입> <이름>;`` 해소(``typedef float real;`` 등; 체인 허용),
     ``const/restrict/__restrict__/__restrict/volatile`` 제거 후 반환형·개수·이름·기본 타입·포인터 깊이를 기대 프로토타입과 비교한다.
     기본 타입은 ABI 가 같은 ``int`` 철자(``signed``, ``signed int``, ``int signed``, ``<stdint.h>`` 의 ``int32_t``)를 ``int`` 로
     모아 비교한다(``C_INT_ALIASES``; 크기·부호가 다른 ``unsigned``·``long``·``int64_t``·``size_t`` 등은 그대로 불일치).
     개수·순서·기본 타입·이름 불일치만 ``param_mismatch``; 한정자·typedef·ABI 동일 철자 차이는 ``signature_exact=False``.
     keyword 모드(``bundle`` 이 있을 때): 기대 프로토타입은 ``void {fn}(const {fn}_args* args)`` — 매개변수 하나, ``{fn}_args`` 의
     포인터(const 유무 무관). 매개변수 이름은 자유이고 ``args`` 가 아니면 signature_exact=False 일 뿐이다. ``bundle`` 없이
     keyword 로 부르면 v3 이전처럼 positional 로 검사한다.
   - rust: 주석(줄·블록·중첩 블록)과 문자열·문자 리터럴을 지운 텍스트에서 본문이 있는 정의
     ``pub [unsafe] extern "C" fn {fn}(...)`` 를 찾는다. 없으면(``extern`` 블록 안의 선언, 제네릭 함수 포함)
     ``function_not_found``. 정의 바로 앞의 속성 목록(``#[…]`` 여러 개; 다른 속성이 섞여도 된다)에 ``#[no_mangle]`` 이나
     ``#[unsafe(no_mangle)]`` 가 없으면 ``no_mangle_missing``. 매개변수 개수·이름·타입(공백 무시)과 반환형(없음 또는 ``()``)이
     기대 시그니처와 다르면 ``param_mismatch``. ``unsafe`` 생략, ``-> ()`` 표기, ``#[unsafe(no_mangle)]``, ``mut`` 바인딩,
     ABI 가 같은 FFI 별칭(``c_int`` = ``i32``, ``c_float`` = ``f32``; 접두 없음 또는 ``std::ffi::``·``core::ffi::``·
     ``std::os::raw::``·``libc::``; 값·포인터 자리)은 signature_exact=False 일 뿐이다(``RUST_FFI_ALIASES``). keyword 모드에서는 구조체 타입을 경로로 쓴 것(``*const kernel_args::DotArgs``,
     ``*const crate::kernel_args::DotArgs``)도 받고 signature_exact=False 로 둔다.
5. ``ok = code is not None and signature_ok is not False``.

LLM 코드를 임포트·실행하지 않는다(ast.parse / gcc -E 만).
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.paths import atomic_write_text
from harness.procutil import run_isolated
from harness.sigs import BACKENDS, C_ARGS_HEADER_NAME, signature_bundle
from harness.taskspec import load_taskspec

_LOG = "[extract]"

PYTHON_TAGS = frozenset({"python", "py", "python3"})
C_TAGS = frozenset({"c"})  # 소문자 비교: c, C. cpp/c++ 등은 제외
RUST_TAGS = frozenset({"rust", "rs"})
# backend → 후보로 받는 펜스 태그(소문자). numba·halide 커널은 파이썬 소스다.
FENCE_TAGS: dict[str, frozenset[str]] = {
    "python": PYTHON_TAGS, "numba": PYTHON_TAGS, "halide": PYTHON_TAGS, "c": C_TAGS, "rust": RUST_TAGS,
}
PYTHON_FAMILY = frozenset({"python", "numba", "halide"})  # 파이썬 시그니처 검사를 쓰는 backend
BUNDLE_REQUIRED = frozenset({"halide", "rust"})  # signature_spec 만으로는 기대 시그니처를 알 수 없는 backend
CC = "gcc"
PREPROCESS_TIMEOUT_S = 10.0
KERNEL_C = "kernel.c"

DEFAULT_FUNCTION = "gemm"
DEFAULT_PARAMS = ["A", "B", "C", "M", "N", "K"]
DEFAULT_C_PROTOTYPE = "void gemm(const float* A, const float* B, float* C, int M, int N, int K)"

C_QUALIFIERS = frozenset({"const", "restrict", "__restrict__", "__restrict", "volatile"})
C_STORAGE_SPECIFIERS = frozenset({"static", "inline", "__inline", "__inline__", "extern", "_Noreturn"})
# 하네스가 지원하는 대상(LP64 Linux: x86-64·aarch64, glibc)에서 ``int`` 와 같은 타입인 다른 철자. typedef 해소 뒤의 기본 타입에
# 적용한다 — 원문 토큰은 그대로이므로 signature_exact=False 로만 드러난다(typedef 와 같은 규칙). ``<stdint.h>`` 의 ``int32_t`` 는
# glibc 에서 ``signed int`` 이고, 전처리 경로는 kernel.c 유래 줄만 보므로 헤더의 typedef 사슬이 보이지 않는다. 크기나 부호가 다른
# 타입(unsigned, long, short, int64_t, size_t, ssize_t, int_fast32_t …)은 여기 넣지 않는다 — 하네스는 C ``int`` 를 넘긴다.
C_INT_ALIASES = frozenset({"int32_t"})
_C_INT_SPECIFIER_SETS = (["signed"], ["int", "signed"])  # 정렬한 지정자 목록: signed / signed int / int signed


@dataclass
class ExtractResult:
    """추출 결과(설계 §6)."""

    ok: bool
    code: str | None
    signature_ok: bool | None
    signature_exact: bool | None
    detail: str
    n_blocks: int


@dataclass
class FencedBlock:
    """펜스 코드 블록 하나."""

    index: int
    tag: str  # 정보 문자열의 첫 단어(원문); 없으면 ""
    code: str
    terminated: bool


# ----------------------------------------------------------------------------- 1. 펜스 수집

_FENCE_OPEN_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*([^\s`]*)")
_FENCE_CLOSE_RE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")


def parse_fenced_blocks(text: str) -> list[FencedBlock]:
    """CommonMark 펜스 블록을 순서대로 모은다. 닫히지 않은 블록은 파일 끝까지를 내용으로 하고 terminated=False."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[FencedBlock] = []
    open_fence: str | None = None
    tag = ""
    buf: list[str] = []
    for line in lines:
        if open_fence is None:
            m = _FENCE_OPEN_RE.match(line)
            if m:
                open_fence = m.group(1)
                tag = m.group(2)
                buf = []
            continue
        m = _FENCE_CLOSE_RE.match(line)
        if m and m.group(1)[0] == open_fence[0] and len(m.group(1)) >= len(open_fence):
            blocks.append(FencedBlock(index=len(blocks), tag=tag, code="\n".join(buf) + "\n", terminated=True))
            open_fence = None
            continue
        buf.append(line)
    if open_fence is not None:
        blocks.append(FencedBlock(index=len(blocks), tag=tag, code="\n".join(buf) + "\n", terminated=False))
    return blocks


# ----------------------------------------------------------------------------- 2·3. 후보 선택

_C_ATTR = r"(?:__attribute__\s*\(\((?:[^()]|\([^()]*\))*\)\)\s*)"
# 매개변수 목록 안의 괄호 묶음: 3단계 중첩까지(``__attribute__((aligned(64)))``; 함수 포인터 ``void (*cb)(int)`` 는 1단계).
# 각 대안의 첫 글자가 서로 다르므로(``(`` vs 그 외) 역추적 폭발이 없다.
_C_PAREN3 = r"\((?:[^;{}()]|\((?:[^;{}()]|\([^;{}()]*\))*\))*\)"


def _python_def_re(function: str) -> re.Pattern[str]:
    return re.compile(rf"^[ \t]*(?:async[ \t]+)?def[ \t]+{re.escape(function)}[ \t]*\(", re.M)


def _c_def_re(function: str) -> re.Pattern[str]:
    # 이름 ( 매개변수(괄호 3단계 중첩 허용) ) [공백/개행] [__attribute__((...))]* {
    return re.compile(
        rf"\b{re.escape(function)}\s*\(((?:[^;{{}}()]|{_C_PAREN3})*)\)\s*({_C_ATTR}*)\{{"
    )


def _has_definition(code: str, backend: str, function: str) -> bool:
    if backend in PYTHON_FAMILY:
        return _python_def_re(function).search(code) is not None
    if backend == "rust":
        return _rust_def_re(function).search(_strip_rust(code)) is not None
    return _c_def_re(function).search(_strip_c_comments(code)) is not None


def _is_candidate(block: FencedBlock, backend: str) -> bool:
    tag = block.tag.lower()
    if tag == "":
        return True
    return tag in FENCE_TAGS[backend]


def select_block(blocks: list[FencedBlock], backend: str, function: str) -> tuple[FencedBlock | None, list[str]]:
    """후보 중 하나를 고른다. (블록, 블록 수준 detail 목록). 후보가 없으면 (None, [])."""
    candidates = [b for b in blocks if _is_candidate(b, backend)]
    if not candidates:
        return None, []
    with_def = [b for b in candidates if _has_definition(b.code, backend, function)]
    pool = with_def if with_def else candidates
    tagged = [b for b in pool if b.tag != ""]
    if tagged:
        pool = tagged
    chosen = max(pool, key=lambda b: len(b.code))  # 가장 긴 것; 같으면 먼저 나온 것
    notes: list[str] = []
    if not chosen.terminated:
        notes.append("unterminated_fence")
    if len(candidates) > 1:
        notes.append(f"multiple_blocks_chose:{chosen.index}")
    return chosen, notes


# ----------------------------------------------------------------------------- 4a. python 시그니처

def _check_python(code: str, function: str, params: list[str], args_mode: str) -> tuple[bool | None, bool | None, str]:
    """(signature_ok, signature_exact, detail)."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None, None, "syntax_error_deferred"
    fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            fn = node  # 같은 이름이 여러 번이면 마지막 정의가 유효하다(파이썬 의미론)
    if fn is None:
        return False, False, "function_not_found"
    if isinstance(fn, ast.AsyncFunctionDef):
        return False, False, "param_mismatch:async_def"
    a = fn.args
    positional = [x.arg for x in a.posonlyargs + a.args]
    keyword_only = [x.arg for x in a.kwonlyargs]
    if args_mode == "keyword":
        expected_desc = "*, " + ", ".join(params)
        matched = not positional and keyword_only == params
    else:
        expected_desc = ", ".join(params)
        matched = positional == params and not keyword_only
    if a.vararg is not None or a.kwarg is not None:
        matched = False
    if not matched:
        return False, False, f"param_mismatch:got ({ast.unparse(a)}) expected ({expected_desc})"
    has_defaults = bool(a.defaults) or any(d is not None for d in a.kw_defaults)
    has_annotations = any(x.annotation is not None for x in a.posonlyargs + a.args + a.kwonlyargs) or fn.returns is not None
    return True, not has_defaults and not has_annotations, ""


# ----------------------------------------------------------------------------- 4b. C 시그니처

_C_COMMENT_RE = re.compile(r'//[^\n]*|/\*.*?\*/|("(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\')', re.S)
_C_OBJECT_DEFINE_RE = re.compile(r"^[ \t]*#[ \t]*define[ \t]+([A-Za-z_]\w*)[ \t]+([^\n]*?)[ \t]*$", re.M)
_C_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\*|\[|\]|\S")
_LINEMARKER_RE = re.compile(r'^#\s+\d+\s+"([^"]*)"')


def _strip_c_comments(code: str) -> str:
    """주석을 공백으로 바꾼다(문자열 리터럴은 보존)."""
    return _C_COMMENT_RE.sub(lambda m: m.group(1) if m.group(1) else " ", code)


def _expand_object_defines(code: str) -> str:
    """전처리 실패 시의 보정: 객체형 ``#define NAME value`` 를 한 단계 치환한다(함수형 매크로는 건드리지 않음)."""
    defines = {m.group(1): m.group(2) for m in _C_OBJECT_DEFINE_RE.finditer(code) if m.group(2)}
    if not defines:
        return code
    pattern = re.compile(r"\b(" + "|".join(re.escape(k) for k in defines) + r")\b")
    body_lines = [ln for ln in code.split("\n") if not _C_OBJECT_DEFINE_RE.match(ln)]
    return pattern.sub(lambda m: defines[m.group(1)], "\n".join(body_lines))


def _lines_from_source(preprocessed: str, source_name: str) -> str:
    """``gcc -E`` 출력에서 라인 마커를 따라 ``source_name`` 유래 줄만 남긴다."""
    out: list[str] = []
    current: str | None = None
    for line in preprocessed.split("\n"):
        m = _LINEMARKER_RE.match(line)
        if m:
            current = m.group(1)
            continue
        if current == source_name:
            out.append(line)
    return "\n".join(out)


def preprocess_c(code: str, work_dir: Path, args_header: str | None = None, cc: str | None = None) -> str | None:
    """``work_dir/kernel.c`` 에 저장하고 ``gcc -E`` 로 전처리해 kernel.c 유래 줄만 돌려준다. 실패하면 None(경고 출력).

    ``cc`` 는 전처리에 쓸 컴파일러(호출자가 설정의 ``build.cc`` 를 넘긴다). None 이면 모듈 상수 ``CC`` (gcc)다.

    ``args_header`` (keyword 모드의 ``kernel_args.h`` 전문)가 있으면 먼저 ``work_dir`` 에 쓴다 — 커널의
    ``#include "kernel_args.h"`` 가 풀리지 않으면 전처리가 항상 실패한다. 헤더 유래 줄은 라인 마커로 걸러진다.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(work_dir / KERNEL_C, code)
    if args_header is not None:
        atomic_write_text(work_dir / C_ARGS_HEADER_NAME, args_header)
    out_path = work_dir / "kernel.i"
    err_path = work_dir / "preprocess.log"
    compiler = cc or CC
    cmd = [compiler, "-E", "-std=gnu11", "-fopenmp", KERNEL_C]
    try:
        r = run_isolated(cmd, timeout_s=PREPROCESS_TIMEOUT_S, cwd=work_dir, stdout_path=out_path, stderr_path=err_path)
    except OSError as e:
        print(f"{_LOG} 경고: 전처리기 실행 실패({e}); 원문으로 검사한다", file=sys.stderr)
        return None
    if r.timed_out or r.rc != 0:
        why = "timeout" if r.timed_out else f"rc={r.rc}"
        print(f"{_LOG} 경고: {compiler} -E 실패({why}): {r.stderr_tail.strip()[-300:]}; 원문으로 검사한다", file=sys.stderr)
        return None
    try:
        text = out_path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        print(f"{_LOG} 경고: 전처리 출력 읽기 실패({e}); 원문으로 검사한다", file=sys.stderr)
        return None
    return _lines_from_source(text, KERNEL_C)


def _split_top_level(params: str) -> list[str]:
    """괄호 깊이를 고려해 최상위 쉼표로 나눈다."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in params:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


@dataclass
class CParam:
    """C 매개변수 하나의 해석."""

    tokens: list[str]  # 속성 제거·공백 정규화 토큰열(한정자 포함; exact 비교용)
    base: str  # 한정자 제거 후 기본 타입("float", "int", "unsigned int" ...)
    depth: int  # 포인터 깊이(* 개수 + 배열 1)
    name: str | None


# 단순 typedef: ``typedef <타입 토큰열> <이름>;`` / ``typedef <타입> *<이름>;`` 만(구조체·열거형·함수 포인터·배열·다중 선언자는 제외).
_C_TYPEDEF_RE = re.compile(r"^[ \t]*typedef[ \t]+([^;{}()\[\]=,]+?)[ \t]*\b([A-Za-z_]\w*)[ \t]*;", re.M)
_TYPEDEF_MAX_ROUNDS = 8  # 체인 해소 상한(순환 방지)


def collect_typedefs(code: str) -> dict[str, list[str]]:
    """주석이 제거된 C 텍스트에서 단순 typedef 를 ``{이름: 타입 토큰열}`` 로 모은다(``typedef float real;`` → ``{"real": ["float"]}``)."""
    out: dict[str, list[str]] = {}
    for m in _C_TYPEDEF_RE.finditer(re.sub(_C_ATTR, " ", code)):
        tokens = _C_TOKEN_RE.findall(m.group(1))
        name = m.group(2)
        if tokens and name not in tokens:
            out[name] = tokens
    return out


def _resolve_typedefs(tokens: list[str], typedefs: dict[str, list[str]]) -> list[str]:
    """토큰열의 typedef 이름을 정의 토큰열로 치환한다(체인은 ``_TYPEDEF_MAX_ROUNDS`` 회까지)."""
    for _ in range(_TYPEDEF_MAX_ROUNDS):
        if not any(t in typedefs for t in tokens):
            break
        expanded: list[str] = []
        for t in tokens:
            expanded.extend(typedefs.get(t, [t]))
        tokens = expanded
    return tokens


def parse_c_param(text: str, typedefs: dict[str, list[str]] | None = None) -> CParam:
    """``const float *restrict A`` / ``float A[]`` / ``int M`` 을 해석한다. ``typedefs`` 가 있으면 기본 타입·깊이 계산 전에
    typedef 이름을 해소한다(``tokens`` 는 원문 그대로 두어 exact 비교에서는 typedef 차이가 드러난다)."""
    text = re.sub(_C_ATTR, " ", text)
    tokens = _C_TOKEN_RE.findall(text)
    is_array = "[" in tokens
    head = tokens[: tokens.index("[")] if is_array else tokens
    name: str | None = None
    if head and re.fullmatch(r"[A-Za-z_]\w*", head[-1]) and head[-1] not in C_QUALIFIERS:
        name = head[-1]
        head = head[:-1]
    if typedefs:
        head = _resolve_typedefs(head, typedefs)
    core = [t for t in head if t not in C_QUALIFIERS]
    depth = core.count("*") + (1 if is_array else 0)
    return CParam(tokens=tokens, base=_canonical_c_base([t for t in core if t != "*"]), depth=depth, name=name)


def _canonical_c_base(words: list[str]) -> str:
    """기본 타입 단어열 → 비교용 이름. ABI 가 같은 ``int`` 철자(``signed``, ``signed int``, ``int signed``, ``int32_t``)는
    ``"int"`` 로 모은다(``C_INT_ALIASES``). 그 밖은 단어를 공백으로 이은 그대로다."""
    if sorted(words) in _C_INT_SPECIFIER_SETS or (len(words) == 1 and words[0] in C_INT_ALIASES):
        return "int"
    return " ".join(words)


_PROTOTYPE_RE = re.compile(r"^\s*(?P<ret>[^()]*?)\b(?P<name>[A-Za-z_]\w*)\s*\((?P<params>.*)\)\s*;?\s*$", re.S)


def parse_c_prototype(prototype: str) -> tuple[list[str], list[CParam]]:
    """기대 시그니처 문자열 → (반환형 토큰열, 매개변수 목록)."""
    m = _PROTOTYPE_RE.match(prototype)
    if m is None:
        raise ValueError(f"C 프로토타입을 해석할 수 없다: {prototype!r}")
    ret_tokens = _C_TOKEN_RE.findall(m.group("ret"))
    return ret_tokens, [parse_c_param(p) for p in _split_top_level(m.group("params"))]


def _expected_c_prototype(signature_spec: dict[str, Any]) -> str:
    """signature_spec 에서 C 기대 프로토타입을 찾는다(없으면 gemm 기본)."""
    direct = signature_spec.get("positional")
    if isinstance(direct, str) and not direct.lstrip().startswith("def "):
        return direct
    langs = signature_spec.get("languages")
    if isinstance(langs, dict):
        for entry in langs.values():
            if isinstance(entry, dict):
                s = entry.get("positional")
                if isinstance(s, str) and not s.lstrip().startswith("def "):
                    return s
    return DEFAULT_C_PROTOTYPE


def _specifier_tokens(text_before: str) -> list[str]:
    """정의 직전 ``;``/``}`` 이후의 지정자 토큰열(전처리 지시문 줄·속성 제거)."""
    cut = max(text_before.rfind(";"), text_before.rfind("}"))
    prefix = text_before[cut + 1:]
    prefix = "\n".join(ln for ln in prefix.split("\n") if not ln.lstrip().startswith("#"))
    prefix = re.sub(_C_ATTR, " ", prefix)
    return _C_TOKEN_RE.findall(prefix)


def _check_c(
    code: str, function: str, params: list[str], expected_prototype: str, *, free_names: bool = False
) -> tuple[bool | None, bool | None, str]:
    """주석·매크로가 해소된 C 텍스트에서 (signature_ok, signature_exact, detail).

    기대 이름은 ``params`` (없는 자리는 프로토타입의 이름)다. ``free_names`` (keyword 모드의 구조체 포인터 하나)면 이름
    불일치를 실패로 보지 않는다 — 토큰열이 달라지므로 signature_exact 만 False 가 된다.
    """
    exp_ret, exp_params = parse_c_prototype(expected_prototype)
    m = _c_def_re(function).search(code)
    if m is None:
        return False, False, "function_not_found"
    spec_tokens = _specifier_tokens(code[: m.start()])
    if "static" in spec_tokens:
        return False, False, "static_linkage"
    ret_tokens = [t for t in spec_tokens if t not in C_STORAGE_SPECIFIERS]
    if ret_tokens != exp_ret:
        got = " ".join(ret_tokens) or "missing"
        return False, False, f"param_mismatch:return_type {got} expected {' '.join(exp_ret)}"
    raw_params = _split_top_level(m.group(1))
    if raw_params == ["void"]:
        raw_params = []
    typedefs = collect_typedefs(code)
    got_params = [parse_c_param(p, typedefs) for p in raw_params]
    if len(got_params) != len(exp_params):
        return False, False, f"param_mismatch:{len(got_params)} params expected {len(exp_params)}"
    for idx, (got, exp) in enumerate(zip(got_params, exp_params)):
        exp_name = params[idx] if idx < len(params) else exp.name
        if not free_names and got.name != exp_name:
            return False, False, f"param_mismatch:param {idx + 1} name {got.name or 'missing'} expected {exp_name}"
        if got.base != exp.base or got.depth != exp.depth:
            got_desc = (got.base or "?") + "*" * got.depth
            exp_desc = exp.base + "*" * exp.depth
            return False, False, f"param_mismatch:param {idx + 1} ({exp_name}) type {got_desc} expected {exp_desc}"
    exact = spec_tokens == exp_ret and all(g.tokens == e.tokens for g, e in zip(got_params, exp_params))
    return True, exact, ""


# ----------------------------------------------------------------------------- 4c. Rust 시그니처

NO_MANGLE_ATTRIBUTES = frozenset({"no_mangle", "unsafe(no_mangle)"})  # 속성 내용(공백 제거)

_RUST_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_RUST_RAW_PREFIXES = frozenset({"r", "br", "cr"})  # 원시 문자열 접두: r"…", r#"…"#, br"…", cr"…"
_RUST_RAW_OPEN_RE = re.compile(r'(#*)"')
_RUST_MUT_RE = re.compile(r"mut\s+")
_RUST_BODY_RE = re.compile(r"\s*(?:->\s*([^{;]*?))?\s*\{")  # `)` 뒤: [-> 반환형] {
_RUST_EXTERN_BLOCK_RE = re.compile(r'\bextern\s*(?:"C"|"")?\s*\{')  # ABI 문자열은 _strip_rust 뒤에 "C" 아니면 ""
_RUST_POINTER_RE = re.compile(r"\*(?:const|mut)")
_RUST_PATH_RE = re.compile(r"(?:[A-Za-z_]\w*::)+")
# ABI 가 같은 FFI 별칭: 지원 대상(x86-64·aarch64 Linux, 개발용 Windows) 모두에서 c_int = i32, c_float = f32. 경로 접두는 없거나
# 아래 넷 중 하나이고 값·포인터 자리 모두 받는다(signature_exact=False). 크기·부호가 다를 수 있는 별칭(c_uint, c_long, usize,
# isize, c_double …)은 넣지 않는다 — 그대로 param_mismatch 다.
RUST_FFI_ALIASES: dict[str, str] = {"c_int": "i32", "c_float": "f32"}
_RUST_FFI_ALIAS_RE = re.compile(r"(\*(?:const|mut))?(?:std::ffi::|core::ffi::|std::os::raw::|libc::)?(c_int|c_float)")


def _rust_def_re(function: str) -> re.Pattern[str]:
    # pub [unsafe] extern "C" fn 이름 (   — 이름 바로 뒤가 `(` 여야 하므로 제네릭 함수(`이름<T>(`)는 맞지 않는다
    return re.compile(rf'\bpub\s+(?:(unsafe)\s+)?extern\s*"C"\s*fn\s+{re.escape(function)}\s*\(')


def _skip_rust_string(code: str, start: int) -> int:
    """``start`` 의 ``"`` 로 시작하는 문자열 리터럴(이스케이프 있음)의 끝 다음 위치. 닫히지 않으면 텍스트 끝."""
    i = start + 1
    while i < len(code):
        if code[i] == "\\":
            i += 2
        elif code[i] == '"':
            return i + 1
        else:
            i += 1
    return len(code)


def _skip_rust_char(code: str, start: int) -> int | None:
    """``start`` 의 ``'`` 가 문자 리터럴(``'x'``, ``'\\n'``, ``'"'``)이면 그 끝 다음 위치, 수명·레이블(``'a``)이면 None."""
    if code.startswith("\\", start + 1):
        end = code.find("'", start + 3)
        return len(code) if end < 0 else end + 1
    if code.startswith("'", start + 2):
        return start + 3
    return None


def _strip_rust(code: str) -> str:
    """Rust 소스의 주석(줄, 블록, 중첩 블록)을 공백으로, 문자열 리터럴을 ``""`` 로, 문자 리터럴을 ``''`` 로 바꾼다.

    시그니처 검사가 주석·문자열 안의 가짜 정의와 괄호에 속지 않게 한다. ABI 문자열 ``"C"`` 만은 그대로 둔다
    (``extern "C" fn`` 을 찾아야 한다). 원시 문자열(``r"…"``, ``r#"…"#``; 내용에 ``"`` 가 있을 수 있다)과 수명(``'a``)을
    구분한다. 바이트·C 문자열(``b"…"``, ``c"…"``)은 접두 글자 뒤의 보통 문자열로 처리된다.
    """
    out: list[str] = []
    i, n = 0, len(code)
    while i < n:
        ch = code[i]
        if code.startswith("//", i):
            end = code.find("\n", i)
            i = n if end < 0 else end
            out.append(" ")
        elif code.startswith("/*", i):
            depth, i = 1, i + 2
            while i < n and depth:
                if code.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif code.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            out.append(" ")
        elif ch == '"':
            end = _skip_rust_string(code, i)
            out.append('"C"' if code[i:end] == '"C"' else '""')
            i = end
        elif ch == "'":
            end = _skip_rust_char(code, i)
            out.append(ch if end is None else "''")
            i = i + 1 if end is None else end
        elif (m := _RUST_IDENT_RE.match(code, i)) is not None:
            i = m.end()
            raw = _RUST_RAW_OPEN_RE.match(code, i) if m.group(0) in _RUST_RAW_PREFIXES else None
            if raw is None:
                out.append(m.group(0))
            else:  # 원시 문자열: 여는 `#` 수만큼의 `#` 가 붙은 `"` 까지
                close = code.find('"' + raw.group(1), raw.end())
                i = n if close < 0 else close + 1 + len(raw.group(1))
                out.append('""')
        else:
            out.append(ch)
            i += 1
    return "".join(out)


@dataclass
class RustParam:
    """Rust 매개변수 하나. ``type`` 은 공백을 한 칸으로 줄인 원문이고 비교는 공백을 모두 지우고 한다."""

    name: str
    type: str
    mutable: bool = False  # ``mut x: …`` 바인딩(이름·타입은 그대로다)


@dataclass
class RustDefinition:
    """찾은 ``pub [unsafe] extern "C" fn`` 정의."""

    unsafe: bool
    params: list[RustParam]
    returns: str | None  # 반환형 표기(공백 제거). 표기가 없으면 None
    attributes: list[str]  # 정의 바로 앞 외부 속성들의 내용(공백 제거): "no_mangle", "inline(never)", …


def _squeeze(text: str) -> str:
    return "".join(text.split())


def _matching_close(text: str, open_pos: int) -> int | None:
    """``open_pos`` 의 여는 괄호(``(`` 또는 ``{``)와 짝이 맞는 닫는 괄호의 위치. 없으면 None."""
    opener = text[open_pos]
    closer = {"(": ")", "{": "}"}[opener]
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == opener:
            depth += 1
        elif text[i] == closer:
            depth -= 1
            if depth == 0:
                return i
    return None


def _rust_extern_blocks(text: str) -> list[tuple[int, int]]:
    """``extern "…" { … }`` 블록들의 (여는 중괄호, 닫는 중괄호) 위치. 그 안의 항목은 외부 함수 **선언**이지 정의가 아니다."""
    spans: list[tuple[int, int]] = []
    for m in _RUST_EXTERN_BLOCK_RE.finditer(text):
        close = _matching_close(text, m.end() - 1)
        spans.append((m.end() - 1, len(text) if close is None else close))
    return spans


def _split_rust_params(text: str) -> list[str]:
    """최상위 쉼표로 나눈다. 괄호·대괄호·꺾쇠 깊이를 세고 ``->`` 의 ``>`` 는 닫는 꺾쇠로 보지 않는다. 끝의 쉼표는 무시한다."""
    out: list[str] = []
    cur: list[str] = []
    depth = 0
    prev = ""
    for ch in text:
        if ch in "([<":
            depth += 1
        elif ch in ")]" or (ch == ">" and prev != "-"):
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        prev = ch
    out.append("".join(cur))
    return [p.strip() for p in out if p.strip()]


def parse_rust_params(text: str) -> list[RustParam]:
    """``x: *const f32, mut n: i32`` (주석·문자열이 지워진 매개변수 목록) → ``RustParam`` 목록."""
    params: list[RustParam] = []
    for part in _split_rust_params(text):
        pattern, _, type_text = part.partition(":")
        pattern = pattern.strip()
        mut = _RUST_MUT_RE.match(pattern)
        params.append(RustParam(name=pattern[mut.end():] if mut else pattern, type=" ".join(type_text.split()),
                                mutable=mut is not None))
    return params


def parse_rust_signature(signature: str, function: str) -> list[RustParam]:
    """기대 시그니처 문자열(``#[no_mangle]\\npub unsafe extern "C" fn {function}(…)``)의 매개변수 목록."""
    text = _strip_rust(signature)
    m = re.search(rf"\bfn\s+{re.escape(function)}\s*\(", text)
    close = _matching_close(text, m.end() - 1) if m is not None else None
    if m is None or close is None:
        raise ValueError(f"Rust 시그니처를 해석할 수 없다: {signature!r}")
    return parse_rust_params(text[m.end():close])


def _rust_attributes_before(text: str, pos: int) -> list[str]:
    """``pos`` 바로 앞에 연달아 붙은 외부 속성(``#[…]``)들의 내용(공백 제거; 가까운 것부터). 내부 속성(``#![…]``)에서 멈춘다."""
    attrs: list[str] = []
    head = text[:pos].rstrip()
    while head.endswith("]"):
        depth, start = 0, len(head) - 1
        while start >= 0:
            depth += {"]": 1, "[": -1}.get(head[start], 0)
            if depth == 0:
                break
            start -= 1
        before = head[:max(start, 0)].rstrip()
        if start < 0 or not before.endswith("#"):
            break
        attrs.append(_squeeze(head[start + 1:-1]))
        head = before[:-1].rstrip()
    return attrs


def find_rust_definition(text: str, function: str) -> RustDefinition | None:
    """주석·문자열이 지워진 텍스트에서 **본문이 있는** 첫 ``pub [unsafe] extern "C" fn {function}(…)`` 정의.

    본문이 없는 선언, ``extern "C" { … }`` 블록 안에 있는 것, 제네릭 함수는 정의로 치지 않는다.
    """
    extern_blocks = _rust_extern_blocks(text)
    for m in _rust_def_re(function).finditer(text):
        close = _matching_close(text, m.end() - 1)
        body = _RUST_BODY_RE.match(text, close + 1) if close is not None else None
        if close is None or body is None or any(a < m.start() < b for a, b in extern_blocks):
            continue
        returns = _squeeze(body.group(1)) if body.group(1) is not None else None
        return RustDefinition(unsafe=m.group(1) is not None, params=parse_rust_params(text[m.end():close]), returns=returns,
                              attributes=_rust_attributes_before(text, m.start()))
    return None


def _is_path_spelling(got: str, expected: str) -> bool:
    """``got`` 이 포인터 타입 ``expected`` 의 가리키는 타입을 경로로 쓴 것인가(``*const kernel_args::DotArgs`` ↔ ``*const DotArgs``)."""
    got, expected = _squeeze(got), _squeeze(expected)
    pointer = _RUST_POINTER_RE.match(expected)
    if pointer is None or not got.startswith(pointer.group(0)):
        return False
    path, sep, last = got[pointer.end():].rpartition("::")
    return last == expected[pointer.end():] and _RUST_PATH_RE.fullmatch(path + sep) is not None


def _canonical_rust_type(text: str) -> str:
    """공백을 지운 타입 표기에서 FFI 별칭(``RUST_FFI_ALIASES``)을 기본 타입으로 바꾼 것(``*const std::ffi::c_float`` → ``*constf32``)."""
    squeezed = _squeeze(text)
    m = _RUST_FFI_ALIAS_RE.fullmatch(squeezed)
    return (m.group(1) or "") + RUST_FFI_ALIASES[m.group(2)] if m else squeezed


def _check_rust(code: str, function: str, expected_signature: str, args_mode: str) -> tuple[bool | None, bool | None, str]:
    """Rust 소스에서 (signature_ok, signature_exact, detail). ``expected_signature`` 는 ``bundle["signature"]`` 다."""
    exp_params = parse_rust_signature(expected_signature, function)
    found = find_rust_definition(_strip_rust(code), function)
    if found is None:
        return False, False, "function_not_found"
    if not NO_MANGLE_ATTRIBUTES.intersection(found.attributes):
        return False, False, "no_mangle_missing"
    if found.returns not in (None, "()"):
        return False, False, f"param_mismatch:return_type {found.returns or 'missing'} expected ()"
    if len(found.params) != len(exp_params):
        return False, False, f"param_mismatch:{len(found.params)} params expected {len(exp_params)}"
    exact = found.unsafe and found.returns is None and "no_mangle" in found.attributes
    for idx, (got, exp) in enumerate(zip(found.params, exp_params)):
        if got.name != exp.name:
            return False, False, f"param_mismatch:param {idx + 1} name {got.name or 'missing'} expected {exp.name}"
        if _squeeze(got.type) != _squeeze(exp.type):
            alias = _canonical_rust_type(got.type) == _squeeze(exp.type)
            if not alias and not (args_mode == "keyword" and _is_path_spelling(got.type, exp.type)):
                return False, False, f"param_mismatch:param {idx + 1} ({exp.name}) type {got.type or 'missing'} expected {exp.type}"
            exact = False
        exact = exact and not got.mutable
    return True, exact, ""


# ----------------------------------------------------------------------------- 공개 API

def extract(
    response_text: str,
    *,
    backend: str,
    signature_spec: dict[str, Any],
    args_mode: str = "positional",
    work_dir: Path | None = None,
    bundle: dict[str, Any] | None = None,
    cc: str | None = None,
) -> ExtractResult:
    """응답에서 코드를 뽑고 시그니처를 검사한다(설계 §6, v3 §7). 예외를 던지지 않는 정상 경로는 ExtractResult 로 보고한다.

    ``bundle`` (``sigs.signature_bundle`` 의 결과)이 있으면 기대 함수 이름·매개변수·C 프로토타입·Rust 시그니처를 거기서
    읽고 ``signature_spec`` 은 보지 않는다. 없으면 ``signature_spec`` 으로 검사한다(python·numba·c; v3 이전과 같은 동작).
    ``cc`` 는 c backend 의 전처리 컴파일러(설정의 ``build.cc``; ``work_dir`` 가 있을 때만 쓴다). None 이면 ``CC`` (gcc)다.
    """
    if backend not in BACKENDS:
        raise ValueError(f"알 수 없는 backend: {backend!r}")
    if bundle is not None:
        function = str(bundle["function"])
        params = [str(p) for p in bundle["params"]]
    elif backend in BUNDLE_REQUIRED:
        raise ValueError(f"backend {backend!r} 의 추출에는 bundle(sigs.signature_bundle 의 결과)이 필요하다")
    else:
        function = str(signature_spec.get("function") or DEFAULT_FUNCTION)
        params_raw = signature_spec.get("params")
        params = [str(p) for p in params_raw] if isinstance(params_raw, list) and params_raw else list(DEFAULT_PARAMS)

    blocks = parse_fenced_blocks(response_text)
    chosen, notes = select_block(blocks, backend, function)
    if chosen is not None:
        code = chosen.code
    elif not blocks and _has_definition(response_text, backend, function):
        code = response_text
        notes = ["bare_code"]
    else:
        return ExtractResult(ok=False, code=None, signature_ok=None, signature_exact=None, detail="no_code_block", n_blocks=len(blocks))

    extra: list[str] = []
    if backend in PYTHON_FAMILY:
        sig_ok, exact, sig_detail = _check_python(code, function, params, args_mode)
    elif backend == "rust":
        assert bundle is not None  # BUNDLE_REQUIRED
        sig_ok, exact, sig_detail = _check_rust(code, function, str(bundle["signature"]), args_mode)
    else:
        keyword = bundle is not None and args_mode == "keyword"
        if bundle is None and args_mode != "positional":
            # bundle 이 없으면 매개변수 구조체의 기대 프로토타입을 모른다: v3 이전처럼 positional 검사로 대신한다
            print(f"{_LOG} 참고: bundle 없는 C backend 의 args_mode={args_mode!r} 는 positional 로 검사한다", file=sys.stderr)
        text: str | None = None
        if work_dir is not None:
            header = str(bundle["args_struct"]) if keyword and bundle.get("args_struct") else None
            text = preprocess_c(code, Path(work_dir), header, cc)
        if text is None:
            if work_dir is not None:
                extra.append("preprocess_failed")
            text = _expand_object_defines(_strip_c_comments(code))
        prototype = str(bundle["signature"]) if bundle is not None else _expected_c_prototype(signature_spec)
        # keyword: 매개변수는 구조체 포인터 하나이고 이름은 자유다(기대 이름 args 는 프로토타입에서 읽힌다)
        sig_ok, exact, sig_detail = _check_c(text, function, [] if keyword else params, prototype, free_names=keyword)

    parts = ([sig_detail] if sig_detail else []) + notes + extra
    return ExtractResult(
        ok=sig_ok is not False,
        code=code,
        signature_ok=sig_ok,
        signature_exact=exact,
        detail=";".join(parts),
        n_blocks=len(blocks),
    )


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    """응답 파일(텍스트 또는 raw JSON)에서 추출 결과를 JSON으로 출력한다."""
    from harness.paths import ROOT

    parser = argparse.ArgumentParser(prog="python -m harness.extract", description="응답에서 코드 블록 추출·시그니처 검사")
    parser.add_argument("--file", type=Path, required=True, help="응답 텍스트(.md/.txt) 또는 raw JSON(.json; response_text 키)")
    parser.add_argument("--backend", choices=BACKENDS, required=True)
    parser.add_argument("--language", default=None, help="signatures.yaml languages 의 키(기본: backend 이름 = 렌더링한 시그니처)")
    parser.add_argument("--signatures", type=Path, default=ROOT / "tasks" / "gemm" / "signatures.yaml", help="signatures.yaml 경로")
    parser.add_argument("--args-mode", default="positional", choices=("positional", "keyword"))
    parser.add_argument("--work-dir", type=Path, default=None, help="C: gcc -E 전처리 작업 디렉터리(kernel.c 저장)")
    parser.add_argument("--print-code", action="store_true", help="추출한 코드도 출력")
    ns = parser.parse_args(argv)

    text = ns.file.read_text(encoding="utf-8")
    if ns.file.suffix == ".json":
        text = str(json.loads(text).get("response_text", ""))
    bundle = signature_bundle(load_taskspec(ns.signatures), ns.language or ns.backend, ns.backend, ns.args_mode)
    res = extract(text, backend=ns.backend, signature_spec={}, args_mode=ns.args_mode, work_dir=ns.work_dir, bundle=bundle)
    out = asdict(res)
    if not ns.print_code:
        out["code"] = None if res.code is None else f"<{len(res.code)} chars>"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    # 컴파일러 출력·경로·한국어 진단을 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
