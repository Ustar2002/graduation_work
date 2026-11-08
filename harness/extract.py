"""응답 텍스트에서 코드 블록을 추출하고 함수 시그니처를 검사한다 (설계 §6).

절차:
1. 펜스 블록(``` 또는 ~~~, CRLF 허용) 수집. 닫히지 않은 펜스는 그 뒤 텍스트 전체를 블록으로 보고 ``unterminated_fence``.
2. 후보: python → 태그 ∈ {python, py, python3} 또는 무태그; c → 태그 ∈ {c, C} 또는 무태그(``cpp`` 등 제외). 태그 있는 것 우선.
3. 함수 정의를 포함하는 후보만; 여러 개면 가장 긴 것(``multiple_blocks_chose:<전체 블록 인덱스>``).
   정의가 있는 후보가 없으면 후보 중 가장 긴 것. 펜스가 없고 본문에 정의가 있으면 본문 전체(``bare_code``). 아니면 ``no_code_block``.
4. 시그니처 검사:
   - python: ``ast.parse`` 실패 → ok=True, signature_ok=None, ``syntax_error_deferred``(빌드 단계가 잡는다).
     최상위 ``FunctionDef``(데코레이터 허용). ``AsyncFunctionDef`` → ``param_mismatch:async_def``.
     positional: ``posonlyargs+args`` 이름 == params, kwonly/vararg/kwarg 없음(기본값 있는 추가 매개변수도 불허).
     keyword(M4): ``kwonlyargs`` == params, positional 없음. 기본값·주석(annotation)이 있으면 signature_exact=False.
   - c: ``work_dir`` 가 있으면 ``gcc -E -std=gnu11 -fopenmp kernel.c`` 로 전처리하고(procutil, 10 s, 코드 실행 없음)
     ``# N "kernel.c"`` 라인 마커로 kernel.c 유래 줄만 남긴 텍스트를 검사(주석·매크로 해소). 실패하면 원문에서 주석을 제거하고
     객체형 ``#define`` 만 치환한 뒤 검사하고 ``preprocess_failed`` 를 병기한다.
     정의(``)`` 뒤 공백·개행·``__attribute__((...))`` 허용 후 ``{``)를 찾는다; 선언만 있으면 ``function_not_found``.
     매개변수 목록 안의 괄호는 3단계 중첩까지 허용한다(``__attribute__((aligned(64)))`` 가 매개변수에 붙는 경우).
     ``static`` 판정은 그 정의의 지정자 토큰열(직전 ``;`` 또는 ``}`` 이후)에서만 → ``static_linkage``.
     매개변수: ``__attribute__`` 제거, 소스의 단순 ``typedef <타입> <이름>;`` 해소(``typedef float real;`` 등; 체인 허용),
     ``const/restrict/__restrict__/__restrict/volatile`` 제거 후 반환형 void; 1·2·3: float 포인터(``*``/``[]``) 이름 A,B,C;
     4·5·6: int 이름 M,N,K. 개수·순서·기본 타입·이름 불일치만 ``param_mismatch``; 한정자·typedef 차이는 ``signature_exact=False``.
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

_LOG = "[extract]"

PYTHON_TAGS = frozenset({"python", "py", "python3"})
C_TAGS = frozenset({"c"})  # 소문자 비교: c, C. cpp/c++ 등은 제외
CC = "gcc"
PREPROCESS_TIMEOUT_S = 10.0
KERNEL_C = "kernel.c"

DEFAULT_FUNCTION = "gemm"
DEFAULT_PARAMS = ["A", "B", "C", "M", "N", "K"]
DEFAULT_C_PROTOTYPE = "void gemm(const float* A, const float* B, float* C, int M, int N, int K)"

C_QUALIFIERS = frozenset({"const", "restrict", "__restrict__", "__restrict", "volatile"})
C_STORAGE_SPECIFIERS = frozenset({"static", "inline", "__inline", "__inline__", "extern", "_Noreturn"})


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
    if backend == "python":
        return _python_def_re(function).search(code) is not None
    return _c_def_re(function).search(_strip_c_comments(code)) is not None


def _is_candidate(block: FencedBlock, backend: str) -> bool:
    tag = block.tag.lower()
    if tag == "":
        return True
    return tag in (PYTHON_TAGS if backend == "python" else C_TAGS)


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


def preprocess_c(code: str, work_dir: Path) -> str | None:
    """``work_dir/kernel.c`` 에 저장하고 ``gcc -E`` 로 전처리해 kernel.c 유래 줄만 돌려준다. 실패하면 None(경고 출력)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(work_dir / KERNEL_C, code)
    out_path = work_dir / "kernel.i"
    err_path = work_dir / "preprocess.log"
    cmd = [CC, "-E", "-std=gnu11", "-fopenmp", KERNEL_C]
    try:
        r = run_isolated(cmd, timeout_s=PREPROCESS_TIMEOUT_S, cwd=work_dir, stdout_path=out_path, stderr_path=err_path)
    except OSError as e:
        print(f"{_LOG} 경고: 전처리기 실행 실패({e}); 원문으로 검사한다", file=sys.stderr)
        return None
    if r.timed_out or r.rc != 0:
        why = "timeout" if r.timed_out else f"rc={r.rc}"
        print(f"{_LOG} 경고: gcc -E 실패({why}): {r.stderr_tail.strip()[-300:]}; 원문으로 검사한다", file=sys.stderr)
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
    base = " ".join(t for t in core if t != "*")
    return CParam(tokens=tokens, base=base, depth=depth, name=name)


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


def _check_c(code: str, function: str, params: list[str], expected_prototype: str) -> tuple[bool | None, bool | None, str]:
    """주석·매크로가 해소된 C 텍스트에서 (signature_ok, signature_exact, detail)."""
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
        if got.name != exp_name:
            return False, False, f"param_mismatch:param {idx + 1} name {got.name or 'missing'} expected {exp_name}"
        if got.base != exp.base or got.depth != exp.depth:
            got_desc = (got.base or "?") + "*" * got.depth
            exp_desc = exp.base + "*" * exp.depth
            return False, False, f"param_mismatch:param {idx + 1} ({exp_name}) type {got_desc} expected {exp_desc}"
    exact = spec_tokens == exp_ret and all(g.tokens == e.tokens for g, e in zip(got_params, exp_params))
    return True, exact, ""


# ----------------------------------------------------------------------------- 공개 API

def extract(
    response_text: str,
    *,
    backend: str,
    signature_spec: dict[str, Any],
    args_mode: str = "positional",
    work_dir: Path | None = None,
) -> ExtractResult:
    """응답에서 코드를 뽑고 시그니처를 검사한다(설계 §6). 예외를 던지지 않는 정상 경로는 ExtractResult 로 보고한다."""
    if backend not in ("python", "c"):
        raise ValueError(f"알 수 없는 backend: {backend!r}")
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
    if backend == "python":
        sig_ok, exact, sig_detail = _check_python(code, function, params, args_mode)
    else:
        if args_mode != "positional":
            # C 의 keyword 조건(매개변수 구조체)은 M4; M1 은 positional 검사로 대신한다
            print(f"{_LOG} 참고: C backend 의 args_mode={args_mode!r} 는 M1 에서 positional 로 검사한다", file=sys.stderr)
        text: str | None = None
        if work_dir is not None:
            text = preprocess_c(code, Path(work_dir))
        if text is None:
            if work_dir is not None:
                extra.append("preprocess_failed")
            text = _expand_object_defines(_strip_c_comments(code))
        sig_ok, exact, sig_detail = _check_c(text, function, params, _expected_c_prototype(signature_spec))

    if "bare_code" in notes and sig_ok is False:
        # 펜스 없는 응답(설명문 + 코드 조각)은 시그니처 검사가 산문을 읽어 detail 이 길어진다 → 형식 위반으로 짧게 요약(2026-11-08)
        sig_detail = "no_code_block;" + sig_detail[:80]
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
    import yaml

    from harness.paths import ROOT

    parser = argparse.ArgumentParser(prog="python -m harness.extract", description="응답에서 코드 블록 추출·시그니처 검사")
    parser.add_argument("--file", type=Path, required=True, help="응답 텍스트(.md/.txt) 또는 raw JSON(.json; response_text 키)")
    parser.add_argument("--backend", choices=("python", "c"), required=True)
    parser.add_argument("--signatures", type=Path, default=ROOT / "tasks" / "gemm" / "signatures.yaml", help="signatures.yaml 경로")
    parser.add_argument("--args-mode", default="positional", choices=("positional", "keyword"))
    parser.add_argument("--work-dir", type=Path, default=None, help="C: gcc -E 전처리 작업 디렉터리(kernel.c 저장)")
    parser.add_argument("--print-code", action="store_true", help="추출한 코드도 출력")
    ns = parser.parse_args(argv)

    text = ns.file.read_text(encoding="utf-8")
    if ns.file.suffix == ".json":
        text = str(json.loads(text).get("response_text", ""))
    with ns.signatures.open("r", encoding="utf-8") as fh:
        spec = yaml.safe_load(fh) or {}
    res = extract(text, backend=ns.backend, signature_spec=spec, args_mode=ns.args_mode, work_dir=ns.work_dir)
    out = asdict(res)
    if not ns.print_code:
        out["code"] = None if res.code is None else f"<{len(res.code)} chars>"
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    sys.exit(main())
