"""성능 결정 요약 (설계 v3 §11, CLAUDE.md §5): 생성 커널 하나가 성능 결정을 어떻게 내렸는지를 JSON 하나로 줄인다.

용도는 둘이다. (1) 사람이 "이 코드가 성능 결정을 어떻게 내렸나"를 10초 안에 본다. (2) 가설 검증의 데이터 — 성능 결정
(루프 순서·타일링·벡터화·병렬화·메모리 배치)을 **LLM** 이 직접 내렸는가, **컴파일러** 가 내렸는가, **라이브러리** 에 넘겼는가.

``summarize`` 는 순수 함수다. 파일·서브프로세스를 쓰지 않고, 커널을 임포트·실행하지 않으며(텍스트와 ``ast`` 만 읽는다),
어떤 입력에도 예외를 올리지 않는다. 내부 실패는 전체 키 집합에 ``notes = "perf_summary error: <Type>: <msg>"`` 를 담아 돌려준다.

출력 (CLAUDE.md §5 의 키 전부 + 확장; 결과 CSV 의 ``perf_summary`` 칸 하나에 들어간다) ::

    language              입력 그대로
    loops                 [{"depth", "order": [바깥→안 유도 변수], "bounds": [상한 식 원문; 알 수 없으면 ""]}]
                          가장 안쪽 루프마다 하나(소스 순서), 최대 20개
    parallel              {"kind": none|openmp|threads|rayon|numba|halide|unknown, "regions": 병렬 구문 수,
                           "collapse": bool, "reduction": bool, "loops": [병렬화된 루프의 변수],
                           "placement": none|outer|inner|mixed|region|unknown,
                           "constructs": [소스의 병렬 구문 원문 — OpenMP 지시문 전부, rayon 진입점, prange, Halide parallel(…)]}
    vectorized            {"status": yes|partial|no|unknown, "source": compiler_remark|intrinsics|loop_nest|interpreter|none,
                           "loops": [벡터화된 루프의 변수]}
    tiling                {"detected": bool, "tile_sizes": [블록 크기 — 리터럴이거나 #define/const 로 풀리는 것만, 중복 없이]}
    memory                {"temporaries": 임시 버퍼·배열을 만드는 자리 수, "layout_transposed": bool}
    library_calls         [호출한 라이브러리 루틴 이름]
    decisions_by_llm      소스에 드러난 결정
    decisions_by_compiler 컴파일러(gcc 벡터화기, Numba, Halide 오토스케줄러)가 내렸음이 로그·출력으로 확인된 결정
    decisions_by_library  (확장) 라이브러리 호출에 넘긴 결정
    notes                 근거(영어, ``; `` 로 이음, 최대 500자)

규칙:

- **거짓 확신을 만들지 않는다.** 정할 수 없으면 문자열 값은 ``"unknown"``, 수·불리언 값은 ``null``, 리스트는 빈 리스트다.
  ``false``·``0``·``none`` 은 "찾아봤는데 소스에 없다"는 뜻이다(예: 빌드 로그 없는 C 의 ``vectorized.status`` 는 ``no`` 가
  아니라 ``unknown``; 오토스케줄러 출력을 못 본 halide_auto 의 ``parallel.kind`` 는 ``none`` 이 아니라 ``unknown``).
- ``parallel`` 은 소스(Halide 는 최종 루프 중첩)에 드러난 병렬 구문만 본다. 라이브러리 안의 스레드는 보이지 않는다.
  ``placement``: 병렬 구문이 붙은 루프가 그 중첩의 가장 바깥이면 ``outer``, 아니면 ``inner``, 섞였으면 ``mixed``,
  루프에 붙지 않은 병렬 구문(작업 공유 루프 없는 parallel 영역, 명시적 스레드, ``rayon::join``, Numba 가 병렬화한 배열 식)이면 ``region``.
  C·Python 의 ``while``·``do`` 문 안에 있는 병렬 루프는 가장 바깥이 아니다(반복마다 fork/join; notes 에 남긴다).
  Numba 의 중첩 ``prange`` 는 가장 바깥 것만 병렬이다(안쪽은 Numba 가 직렬로 돌린다; notes 에 남긴다).
- ``vectorized.status``: 판정 근거(리마크·루프 중첩·인트린식)가 있는 가장 안쪽 루프가 전부 벡터화됐으면 ``yes``, 일부면
  ``partial``, 하나도 없으면 ``no``. 근거 없는 루프는 판정에서 빼고 notes 에 남기며, 근거 있는 루프가 없으면 ``unknown``.
  이 파일의 다른 함수(루프가 있는)를 부르기만 하는 루프는 가장 안쪽 루프로 세지 않는다.
- 결정 어휘는 고정이다(``DECISIONS``). ``library_delegation`` 은 ``decisions_by_library`` 에만 싣는다: 계산을 라이브러리
  루틴(NumPy 배열 연산, BLAS, 정렬)에 넘겼다는 표시이며, 라이브러리 안에서 어떤 결정이 내려졌는지는 소스로 알 수 없으므로
  다른 낱말을 덧붙이지 않는다(무엇을 넘겼는지는 ``library_calls`` 가 말한다).

백엔드별 추출 소스:

- **c**: 주석·문자열을 지운 토큰열에서 ``for`` 중첩(괄호 짝 맞추기; C 파서가 아니다), ``#pragma omp``, 타일링(1보다 큰 상수·매크로
  증분 + 그 변수에 기대는 안쪽 루프나 ``min(…)`` 절단문), 인트린식, 알려진 라이브러리 호출, 할당 호출 수. 벡터화는 ``build_log``
  의 gcc ``-fopt-info-vec`` 줄을 루프의 줄 번호에 대응시킨다. gcc 가 ``ignoring '#pragma omp …'`` 라고 경고했으면(``-fopenmp``
  없이 빌드) OpenMP 지시문은 ``constructs`` 에만 남고 병렬·simd 로 세지 않는다. 괄호 짝이 안 맞거나 매크로가 루프를 숨기면
  루프를 줄여 보고하고, 루프에 기대는 판정(벡터화 상태, "타일링·전치 복사 없음")은 ``unknown``/``null`` 로 둔다.
- **rust**: ``for … in`` 중첩, rayon 병렬 반복자·``join``·``scope``, ``std::arch`` 인트린식. 컴파일러 리마크가 없어 벡터화 상태는
  손으로 쓴 SIMD 가 가장 안쪽 루프를 전부 덮을 때만 ``yes``, 아니면 ``unknown``.
- **python**(numpy)·**numba**: ``ast``. 배열을 루프 변수로 첨자하는 ``for``(원소 루프), NumPy 호출(``@`` 는 ``np.matmul``),
  ``out=``·임시 배열(새 배열을 돌려주는 호출과 배열 산술 식의 수), Numba 데코레이터 옵션·``prange``·병렬 진단.
- **halide**: ``inspect["loop_nest"]`` (``print_loop_nest()``)에서 루프·병렬·벡터화·따로 저장되는 Func 수. 스케줄 호출은 소스의
  것이 LLM 결정, ``inspect["schedule_source"]`` (오토스케줄러)의 것이 컴파일러 결정이다.

CLI: ``python -m harness.perf_summary --language L --backend B --file kernel.c [--build-log build.log] [--inspect import.json]``
"""

from __future__ import annotations

import argparse
import ast
import bisect
import functools
import json
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator

if TYPE_CHECKING:
    from harness.taskspec import TaskSpec

_LOG = "[perf_summary]"

BACKENDS: tuple[str, ...] = ("python", "numba", "halide", "c", "rust")

# 결정 어휘(v3 §11; 고정). 출력의 decisions_* 리스트는 이 순서를 따른다.
DECISIONS: tuple[str, ...] = (
    "loop_order", "tiling", "vectorization", "parallel_region", "reduction", "memory_layout",
    "library_delegation", "algorithm_choice", "accumulation_precision",
)

ERROR_NOTE_PREFIX = "perf_summary error:"

MAX_LOOPS = 20    # loops 항목 수 상한(CSV 칸 하나에 들어간다)
MAX_ITEMS = 20    # 그 밖의 리스트 항목 수 상한
MAX_NOTES = 500   # notes 글자 수 상한
MAX_TEXT = 80     # 식 텍스트(상한 식, 구문 원문) 하나의 글자 수 상한

# Halide 스케줄 호출 이름. 기준은 harness.build.halide.SCHEDULE_METHODS(v3 §6)이고, 이것은 그 모듈을 임포트할 수 없을 때의 같은 내용이다.
DEFAULT_SCHEDULE_METHODS: frozenset[str] = frozenset({
    "align_bounds", "align_extent", "align_storage", "allow_race_conditions", "always_partition", "always_partition_all",
    "async_", "atomic", "bound", "bound_extent", "bound_storage", "clone_in", "compute_at", "compute_inline", "compute_root",
    "compute_with", "copy_to_device", "copy_to_host", "fold_storage", "fuse", "gpu", "gpu_blocks", "gpu_lanes",
    "gpu_single_thread", "gpu_threads", "gpu_tile", "hexagon", "hoist_storage", "hoist_storage_root", "in_", "memoize",
    "never_partition", "never_partition_all", "parallel", "partition", "prefetch", "rename", "reorder", "reorder_storage",
    "rfactor", "ring_buffer", "serial", "specialize", "split", "store_at", "store_in", "store_root", "tile", "unroll", "vectorize",
})


# ----------------------------------------------------------------------------- 결과 모델


@dataclass
class _Facts:
    """백엔드별 분석이 채우는 관찰값. ``to_dict`` 가 출력 스키마로 바꾸고 크기 상한을 적용한다."""

    language: str
    nests: list[dict[str, Any]] = field(default_factory=list)
    par_kind: str = "none"
    par_regions: int | None = 0
    par_collapse: bool | None = False
    par_reduction: bool | None = False
    par_loops: list[str] = field(default_factory=list)
    par_placement: str = "none"
    par_constructs: list[str] = field(default_factory=list)
    vec_status: str = "unknown"
    vec_source: str = "none"
    vec_loops: list[str] = field(default_factory=list)
    tiling: bool | None = False
    tile_sizes: list[int] = field(default_factory=list)
    temporaries: int | None = 0
    transposed: bool | None = False
    library_calls: list[str] = field(default_factory=list)
    by_llm: set[str] = field(default_factory=set)
    by_compiler: set[str] = field(default_factory=set)
    by_library: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def parallel_unknown(self) -> None:
        """병렬 구문을 정할 수 없다(최종 스케줄이나 병렬 진단을 보지 못했다)."""
        self.par_kind, self.par_regions, self.par_collapse, self.par_reduction = "unknown", None, None, None
        self.par_placement = "unknown"

    def to_dict(self) -> dict[str, Any]:
        notes = [n for n in self.notes if n]
        if len(self.nests) > MAX_LOOPS:
            notes.append(f"loops truncated: first {MAX_LOOPS} of {len(self.nests)} nests kept")
        return {
            "language": self.language,
            "loops": self.nests[:MAX_LOOPS],
            "parallel": {
                "kind": self.par_kind,
                "regions": self.par_regions,
                "collapse": self.par_collapse,
                "reduction": self.par_reduction,
                "loops": _uniq(self.par_loops)[:MAX_ITEMS],
                "placement": self.par_placement,
                "constructs": _uniq(self.par_constructs)[:MAX_ITEMS],
            },
            "vectorized": {"status": self.vec_status, "source": self.vec_source, "loops": _uniq(self.vec_loops)[:MAX_ITEMS]},
            "tiling": {"detected": self.tiling, "tile_sizes": _uniq(self.tile_sizes)[:MAX_ITEMS]},
            "memory": {"temporaries": self.temporaries, "layout_transposed": self.transposed},
            "library_calls": _uniq(self.library_calls)[:MAX_ITEMS],
            "decisions_by_llm": [d for d in DECISIONS if d in self.by_llm],
            "decisions_by_compiler": [d for d in DECISIONS if d in self.by_compiler],
            "decisions_by_library": [d for d in DECISIONS if d in self.by_library],
            "notes": _clip("; ".join(notes), MAX_NOTES),
        }


def _unknown(language: str, note: str) -> dict[str, Any]:
    """아무것도 정하지 못했을 때의 출력: 전체 키 집합에 ``unknown``/``null``/빈 리스트."""
    facts = _Facts(language, tiling=None, temporaries=None, transposed=None, notes=[note])
    facts.parallel_unknown()
    return facts.to_dict()


def _clip(text: str, limit: int = MAX_TEXT) -> str:
    """공백을 하나로 줄이고 ``limit`` 자로 자른다(잘렸으면 끝이 ``...``)."""
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _uniq(items: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(items))


def _listing(items: Iterable[str], limit: int) -> str:
    """notes 용 나열: 앞의 ``limit`` 개와 나머지 수."""
    names = _uniq(items)
    more = f", +{len(names) - limit} more" if len(names) > limit else ""
    return ", ".join(names[:limit]) + more


# ----------------------------------------------------------------------------- 루프 모델과 언어 공통 휴리스틱


@dataclass
class _Loop:
    """``for`` 문 하나(언어 공통). 식은 원문 텍스트로 둔다."""

    var: str                          # 유도 변수("?" = 정할 수 없음)
    bound: str = ""                   # 상한 식("" = 없음)
    init: str = ""                    # 시작값 식
    step: str = ""                    # 단위가 아닌 증분의 식("" = 단위 증분이거나 등차가 아님)
    parent: int | None = None         # 감싸는 루프(목록 인덱스)
    line: int = 0                     # 루프를 가리키는 줄: C 는 gcc 가 루프 위치로 쓰는 조건식 비교 연산자의 줄
    col: int = 0                      # 같은 연산자의 열(gcc 표시 열)
    first_line: int = 0               # for 키워드의 줄
    last_line: int = 0                # 본문 마지막 줄
    head: int = -1                    # for 토큰의 인덱스 (C·Rust)
    body: tuple[int, int] = (0, 0)    # 본문 토큰 구간 [시작, 끝) (C·Rust)
    node: Any = None                  # ast.For (Python)
    owner: Any = None                 # 감싸는 함수 정의 (Python)
    prange: bool = False              # numba.prange 루프 (Python)
    wrapped: int = 0                  # 이 루프를 감싸는 while·do 문의 수(병렬 placement 의 중첩 단계에 더한다; C·Python)


def _chain(loops: list[_Loop], idx: int) -> list[int]:
    """가장 바깥 루프부터 ``idx`` 까지의 인덱스."""
    out: list[int] = []
    cur: int | None = idx
    while cur is not None:
        out.append(cur)
        cur = loops[cur].parent
    return out[::-1]


def _innermost(loops: list[_Loop]) -> list[int]:
    parents = {lp.parent for lp in loops if lp.parent is not None}
    return [i for i in range(len(loops)) if i not in parents]


def _nests(loops: list[_Loop]) -> list[dict[str, Any]]:
    """가장 안쪽 루프마다 바깥→안 경로 하나."""
    out: list[dict[str, Any]] = []
    for idx in _innermost(loops):
        chain = _chain(loops, idx)
        out.append({"depth": len(chain), "order": [loops[i].var for i in chain], "bounds": [loops[i].bound for i in chain]})
    return out


def _placement(levels: list[int], regions: int) -> str:
    """병렬 구문이 붙은 루프들의 중첩 단계(1 = 가장 바깥)에서 placement 값."""
    if not levels:
        return "region" if regions else "none"
    outer = [lv == 1 for lv in levels]
    return "outer" if all(outer) else "mixed" if any(outer) else "inner"


def _parse(code: str, mode: str = "exec") -> Any:
    """``ast.parse`` 를 경고 없이 부른다(잘못된 이스케이프·리터럴의 SyntaxWarning 이 stderr 로 새지 않게)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ast.parse(code, mode=mode)


_INT_SUFFIX_RE = re.compile(r"\b(\d[\d_]*?)_?(?:[ui](?:8|16|32|64|128|size)|[uUlL]{1,3})\b")
_INT_CAST_RE = re.compile(r"\(\s*(?:unsigned\s+|signed\s+)?(?:size_t|ptrdiff_t|int|long|unsigned|u?int\d+_t)\s*\)")
_INT_EXPR_RE = re.compile(r"[\w\s+\-*/%()<>]+")


def _const_int(text: str, consts: dict[str, int]) -> int | None:
    """정수 상수식(리터럴, ``consts`` 의 이름, ``+ - * / % << >>``)의 값. 풀 수 없으면 None."""
    expr = _INT_SUFFIX_RE.sub(r"\1", _INT_CAST_RE.sub(" ", text)).strip()
    if not expr or not _INT_EXPR_RE.fullmatch(expr):
        return None
    try:
        tree = _parse(expr, "eval")
    except (SyntaxError, ValueError):
        return None

    def ev(node: ast.AST) -> int | None:
        if isinstance(node, ast.Constant):
            return node.value if type(node.value) is int else None
        if isinstance(node, ast.Name):
            return consts.get(node.id)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
            val = ev(node.operand)
            return None if val is None else (-val if isinstance(node.op, ast.USub) else val)
        if isinstance(node, ast.BinOp):
            lhs, rhs = ev(node.left), ev(node.right)
            if lhs is None or rhs is None:
                return None
            if isinstance(node.op, ast.Add):
                return lhs + rhs
            if isinstance(node.op, ast.Sub):
                return lhs - rhs
            if isinstance(node.op, ast.Mult):
                return lhs * rhs
            if isinstance(node.op, (ast.Div, ast.FloorDiv)):
                return abs(lhs) // abs(rhs) * (1 if (lhs >= 0) == (rhs > 0) else -1) if rhs else None
            if isinstance(node.op, ast.Mod):
                return lhs % rhs if rhs else None
            if isinstance(node.op, ast.LShift):
                return lhs << rhs if 0 <= rhs < 64 else None
            if isinstance(node.op, ast.RShift):
                return lhs >> rhs if 0 <= rhs < 64 else None
        return None

    return ev(tree.body)


def _resolve_consts(raw: dict[str, str]) -> dict[str, int]:
    """이름 → 상수식 텍스트를 이름 → 정수로 푼다(서로 참조하는 정의는 여러 번 돌려 푼다)."""
    out: dict[str, int] = {}
    for _ in range(8):
        before = len(out)
        for name, expr in raw.items():
            if name not in out:
                value = _const_int(expr, out)
                if value is not None:
                    out[name] = value
        if len(out) == before:
            break
    return out


def _is_float_task(spec: TaskSpec | None) -> bool:
    """출력이 float32 인 과제인가(명세가 없으면 그렇다고 본다). 누적 정밀도 결정은 이런 과제에서만 뜻이 있다."""
    return spec is None or any(a.kind == "out" and a.dtype == "float32" for a in spec.args)


def _word_re(word: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w.]){re.escape(word)}(?!\w)")


def _detect_tiling(
    loops: list[_Loop], consts: dict[str, int], fragments: Callable[[int], list[str]], clamp: re.Pattern[str]
) -> tuple[bool, list[int]]:
    """블록화된 루프를 찾는다: 증분이 1보다 큰 상수·이름인 루프 가운데, 그 유도 변수에 기대는 안쪽 루프(시작값·상한에 변수가
    나온다)가 있거나 본문에 블록 크기 절단문(변수·증분·``min``/조건식이 한 문장에 있다)이 있는 것. 증분만 큰 루프(수동
    언롤, SIMD 폭 걸음)는 타일링이 아니다. 반환: (찾음, 타일 루프의 증분 가운데 값이 풀리는 것 — 소스 순서)."""
    children: dict[int, list[int]] = {}
    for idx, loop in enumerate(loops):
        if loop.parent is not None:
            children.setdefault(loop.parent, []).append(idx)

    def inner_loops(idx: int) -> Iterator[_Loop]:
        pending = list(children.get(idx, ()))
        while pending:
            cur = pending.pop()
            yield loops[cur]
            pending.extend(children.get(cur, ()))

    detected = False
    sizes: list[int] = []
    for idx, loop in enumerate(loops):
        if not loop.step or loop.var == "?":
            continue
        value = _const_int(loop.step, consts)
        if value is not None and value <= 1:
            continue
        var_re = _word_re(loop.var)
        depends = any(var_re.search(lp.init) or var_re.search(lp.bound) for lp in inner_loops(idx))
        if not depends:
            if re.fullmatch(r"\w+", loop.step) is None:
                continue
            step_re = _word_re(loop.step)
            if not any(var_re.search(s) and step_re.search(s) and clamp.search(s) for s in fragments(idx)):
                continue
        detected = True
        if value is not None:
            sizes.append(value)
    return detected, sizes


def _loop_decisions(
    facts: _Facts, loops: list[_Loop], consts: dict[str, int], fragments: Callable[[int], list[str]], clamp: re.Pattern[str],
    *, complete: bool = True,
) -> None:
    """루프 중첩에서 읽는 LLM 의 결정: 두 단계 이상 겹쳤으면 루프 순서, 블록화돼 있으면 타일링.
    루프를 다 읽지 못했으면(``complete=False``) 타일링이 없다고 말하지 않는다."""
    if any(nest["depth"] >= 2 for nest in facts.nests):
        facts.by_llm.add("loop_order")
    facts.tiling, facts.tile_sizes = _detect_tiling(loops, consts, fragments, clamp)
    if facts.tiling:
        facts.by_llm.add("tiling")
    elif not complete:
        facts.tiling = None


# ----------------------------------------------------------------------------- C·Rust 어휘 층


_OPEN = frozenset("([{")
_CLOSE = {")": "(", "]": "[", "}": "{"}
_TOKEN_RE = re.compile(
    r"[A-Za-z_]\w*|\d\w*(?:\.\d\w*)?|<<=|>>=|\.\.=|\.\.\.|\.\.|->|=>|::|\+\+|--|&&|\|\||<<|>>|[-+*/%&|^<>=!]=|\S"
)
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
_SIMD_RE = re.compile(r"_mm\d*_\w+|__m(?:64|128|256|512)[id]?|__builtin_ia32_\w+|__builtin_shuffle|__builtin_convertvector")
_NOT_SIMD = frozenset({"_mm_malloc", "_mm_free", "_mm_prefetch", "_mm_pause", "_mm_sfence", "_mm_lfence", "_mm_mfence",
                       "_mm_getcsr", "_mm_setcsr"})
_RAW_STRING_RE = re.compile(r'r(#*)"')
_RUST_CHAR_RE = re.compile(r"'(?:\\(?:x[0-9A-Fa-f]{2}|u\{[0-9A-Fa-f_]{1,8}\}|.)|[^\\'\n])'")
_CMP_OPS = frozenset({"<", "<=", ">", ">=", "!="})


def _quoted_end(code: str, start: int, *, multiline: bool) -> int:
    """``start`` 의 따옴표로 시작한 리터럴의 끝(닫는 따옴표 다음). 닫히지 않으면 줄 끝(C) 또는 파일 끝(Rust)."""
    quote = code[start]
    i = start + 1
    while i < len(code):
        ch = code[i]
        if ch == "\\":
            i += 2
            continue
        if ch == quote:
            return i + 1
        if ch == "\n" and not multiline:
            return i
        i += 1
    return len(code)


def _block_comment_end(code: str, start: int, *, nested: bool) -> int:
    """``/*`` 주석의 끝. 닫히지 않으면 파일 끝. Rust 는 중첩된다."""
    depth = 1
    i = start + 2
    while i < len(code):
        if code.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        elif nested and code.startswith("/*", i):
            depth += 1
            i += 2
        else:
            i += 1
    return len(code)


def _blank(code: str, *, rust: bool) -> str:
    """주석·문자열·문자 리터럴을 같은 길이의 공백으로 바꾼다(개행은 그대로 → 줄·열 번호가 원문과 같다)."""
    out = list(code)
    i = 0
    while i < len(code):
        ch = code[i]
        if code.startswith("//", i):
            end = code.find("\n", i)
            end = len(code) if end < 0 else end
        elif code.startswith("/*", i):
            end = _block_comment_end(code, i, nested=rust)
        elif ch == '"':
            end = _quoted_end(code, i, multiline=rust)
        elif rust and ch == "r" and (m := _RAW_STRING_RE.match(code, i)) and not (i and (code[i - 1].isalnum() or code[i - 1] == "_")):
            closing = '"' + m.group(1)
            end = code.find(closing, m.end())
            end = len(code) if end < 0 else end + len(closing)
        elif ch == "'" and rust:
            lit = _RUST_CHAR_RE.match(code, i)   # 아니면 수명(lifetime) 표기다
            if lit is None:
                i += 1
                continue
            end = lit.end()
        elif ch == "'":
            end = _quoted_end(code, i, multiline=False)
        else:
            i += 1
            continue
        for k in range(i, end):
            if out[k] != "\n":
                out[k] = " "
        i = end
    return "".join(out)


@dataclass(frozen=True)
class _Tok:
    text: str
    pos: int    # 소스 안 시작 오프셋
    line: int   # 1부터


@dataclass
class _Tokens:
    """주석·문자열을 지운 소스의 토큰열과 괄호 짝."""

    text: str
    toks: list[_Tok]
    match: dict[int, int]   # 여는 괄호 인덱스 ↔ 닫는 괄호 인덱스
    balanced: bool

    def span(self, a: int, b: int, limit: int | None = MAX_TEXT) -> str:
        """토큰 구간 [a, b) 의 원문(공백 정리; ``limit`` 가 있으면 자른다)."""
        if a >= b:
            return ""
        last = self.toks[b - 1]
        raw = self.text[self.toks[a].pos: last.pos + len(last.text)]
        return _clip(raw, limit) if limit else " ".join(raw.split())

    def col(self, idx: int) -> int:
        """토큰의 gcc 표시 열(1부터; 탭은 8칸 정지)."""
        pos = self.toks[idx].pos
        col = 0
        for ch in self.text[self.text.rfind("\n", 0, pos) + 1: pos]:
            col = (col // 8 + 1) * 8 if ch == "\t" else col + 1
        return col + 1

    def top_level(self, a: int, b: int) -> Iterator[int]:
        """[a, b) 에서 괄호 깊이 0 인 토큰 인덱스(여는 괄호 자신은 내고 그 안은 건너뛴다)."""
        i = a
        while i < b:
            yield i
            close = self.match.get(i) if self.toks[i].text in _OPEN else None
            i = close + 1 if close is not None else i + 1

    def split(self, a: int, b: int, sep: str) -> list[tuple[int, int]]:
        """[a, b) 를 깊이 0 의 ``sep`` 토큰에서 나눈 구간들."""
        parts: list[tuple[int, int]] = []
        start = a
        for i in self.top_level(a, b):
            if self.toks[i].text == sep:
                parts.append((start, i))
                start = i + 1
        parts.append((start, b))
        return parts

    def texts(self, a: int, b: int) -> list[str]:
        return [t.text for t in self.toks[a:b]]


def _tokenize(text: str) -> _Tokens:
    starts = [0] + [m.end() for m in re.finditer(r"\n", text)]
    toks = [_Tok(m.group(), m.start(), bisect.bisect_right(starts, m.start())) for m in _TOKEN_RE.finditer(text)]
    match: dict[int, int] = {}
    stack: list[int] = []
    balanced = True
    for i, tok in enumerate(toks):
        if tok.text in _OPEN:
            stack.append(i)
        elif tok.text in _CLOSE:
            if stack and toks[stack[-1]].text == _CLOSE[tok.text]:
                j = stack.pop()
                match[j] = i
                match[i] = j
            else:
                balanced = False
    return _Tokens(text, toks, match, balanced and not stack)


def _is_ident(text: str) -> bool:
    return _IDENT_RE.fullmatch(text) is not None


def _fragments(tk: _Tokens, body: tuple[int, int]) -> list[str]:
    """본문 토큰 구간을 ``;``·중괄호에서 끊은 문장 조각(타일링의 절단문 판정용)."""
    out: list[str] = []
    start = body[0]
    for i in range(body[0], body[1] + 1):
        if i == body[1] or tk.toks[i].text in (";", "{", "}"):
            if i > start:
                out.append(tk.span(start, i, None))
            start = i + 1
    return out


def _leaf_loops(tk: _Tokens, loops: list[_Loop], functions: dict[str, tuple[int, int]]) -> list[int]:
    """벡터화를 따질 가장 안쪽 루프: 안에 루프가 없고, 본문이 (루프를 가진) 이 파일의 함수를 부르지도 않는 루프.
    ``functions`` 는 이 파일에 정의된 함수 이름 → 본문 토큰 구간이다."""
    toks = tk.toks

    def callees(span: tuple[int, int]) -> set[str]:
        return {toks[i].text for i in range(span[0], span[1] - 1) if toks[i].text in functions and toks[i + 1].text == "("}

    loopy = {name for name, span in functions.items() if any(span[0] <= lp.head < span[1] for lp in loops)}
    grew = True
    while grew:
        grew = False
        for name, span in functions.items():
            if name not in loopy and callees(span) & loopy:
                loopy.add(name)
                grew = True
    return [i for i in _innermost(loops) if not callees(loops[i].body) & loopy]


def _simd_sites(tk: _Tokens) -> list[int]:
    """손으로 쓴 SIMD 의 증거가 되는 토큰 인덱스: 인트린식·벡터 타입 이름(``_mm256_*``, ``__m256``)과 GCC 벡터 확장
    (``vector_size(…)`` 속성). 프리페치·할당 같은 비-SIMD 인트린식은 뺀다."""
    toks = tk.toks
    return [
        i for i, tok in enumerate(toks)
        if (_SIMD_RE.fullmatch(tok.text) and tok.text not in _NOT_SIMD)
        or (tok.text == "vector_size" and 0 < i < len(toks) - 1 and toks[i - 1].text == "(" and toks[i + 1].text == "(")
    ]


def _index_roles(tk: _Tokens, a: int, b: int, loop_vars: set[str]) -> tuple[set[str], set[str]]:
    """첨자 식 [a, b) 에서 (곱해지는 항의 루프 변수, 더해지기만 하는 항의 루프 변수) — 행우선 첨자의 행·열 역할."""
    major: set[str] = set()
    minor: set[str] = set()
    start = a
    for i in [*tk.top_level(a, b), b]:
        if i == b or tk.toks[i].text in ("+", "-"):
            names = {t for t in tk.texts(start, i) if t in loop_vars}
            scaled = any(tk.toks[k].text == "*" for k in tk.top_level(start, i))
            (major if scaled else minor).update(names)
            start = i + 1
    return major, minor


def _transposed_copy(tk: _Tokens, loops: list[_Loop]) -> bool:
    """``T[i*P + j] = S[j*Q + i];`` 꼴(서로 다른 배열 사이, 두 루프 변수의 행·열 역할이 뒤바뀐 복사)이 있는가."""
    toks = tk.toks
    loop_vars = {lp.var for lp in loops}
    for i, tok in enumerate(toks):
        if tok.text != "=" or i < 3 or i + 2 >= len(toks) or toks[i - 1].text != "]" or toks[i + 2].text != "[":
            continue
        lo, hi = tk.match.get(i - 1), tk.match.get(i + 2)
        if lo is None or hi is None or lo < 1 or hi + 1 >= len(toks) or toks[hi + 1].text != ";":
            continue
        dst, src = toks[lo - 1].text, toks[i + 1].text
        if not _is_ident(dst) or not _is_ident(src) or dst == src:
            continue
        if lo >= 2 and toks[lo - 2].text not in (";", "{", "}", ")", "else"):
            continue
        dst_major, dst_minor = _index_roles(tk, lo + 1, i - 1, loop_vars)
        src_major, src_minor = _index_roles(tk, i + 3, hi, loop_vars)
        if dst_major & src_minor and dst_minor & src_major:
            return True
    return False


# ----------------------------------------------------------------------------- C


# 알려진 라이브러리 호출: 선형대수(cblas_*, LAPACKE_*), 정렬, 덩어리 메모리 연산, 할당, 스레드·OpenMP 런타임(omp_*)
_C_ALLOC = frozenset({"malloc", "calloc", "realloc", "aligned_alloc", "posix_memalign", "memalign", "valloc", "alloca", "_mm_malloc"})
_C_THREADS = frozenset({"pthread_create", "thrd_create"})
_C_LIBRARY = frozenset({"memcpy", "memmove", "memset", "qsort"}) | _C_ALLOC | _C_THREADS
_C_LIBRARY_PREFIXES = ("cblas_", "LAPACKE_", "omp_")
_C_DELEGATING_PREFIXES = ("cblas_", "LAPACKE_", "qsort")   # 커널의 계산 자체를 넘기는 호출
_C_TYPE_RE = re.compile(r"float|double|int|char|short|long|unsigned|signed|size_t|ptrdiff_t|u?int(?:8|16|32|64)_t|__m(?:128|256|512)[id]?")
_C_DECL_WORDS = frozenset({"static", "const", "volatile", "register", "restrict", "_Alignas", "_Thread_local"})
_C_DOUBLE_TYPES = frozenset({"double", "__m128d", "__m256d", "__m512d"})
_C_CONST_DECL_RE = re.compile(
    r"\bconst\s+(?:(?:unsigned|signed|long|short|int|size_t|ptrdiff_t|u?int\d+_t)\s+)+(\w+)\s*=\s*([^;,{}]+)[;,]"
)
_C_ENUM_RE = re.compile(r"\benum\b[^{;]*\{([^}]*)\}")
_C_CLAMP_RE = re.compile(r"\?|\b[a-z]?min(?:f|\d|_\w+)?\s*\(", re.I)
_HIDDEN_LOOP_RE = re.compile(r"\bfor\b|\bwhile\s*\((?!\s*0\s*\))")   # 매크로 본문의 루프(do { … } while (0) 관용구는 아니다)
_OMP_RE = re.compile(r"#\s*pragma\s+omp\s+(.+)")
_OMP_IGNORED_RE = re.compile(r"ignoring\s+\W{0,3}#pragma\s+omp\b")
_DEFINE_RE = re.compile(r"#\s*define\s+(\w+)(\(?)(.*)")
_GCC_REMARK_RE = re.compile(r"(?P<file>.*?):(?P<line>\d+):(?P<col>\d+): (?P<kind>optimized|missed): (?P<msg>.*)")


def _c_directives(blanked: str) -> tuple[str, list[tuple[int, str]]]:
    """전처리 지시문(역슬래시로 이어진 줄 포함)을 떼어 낸다. 반환: (지시문 줄을 공백으로 지운 소스, [(마지막 줄 번호, 지시문)])."""
    lines = blanked.split("\n")
    found: list[tuple[int, str]] = []
    i = 0
    while i < len(lines):
        if lines[i].lstrip().startswith("#"):
            parts: list[str] = []
            while True:
                cur = lines[i]
                continued = cur.rstrip().endswith("\\")
                parts.append(cur.rstrip()[:-1] if continued else cur)
                lines[i] = " " * len(cur)
                if not continued or i + 1 >= len(lines):
                    break
                i += 1
            found.append((i + 1, " ".join(" ".join(parts).split())))
        i += 1
    return "\n".join(lines), found


def _stmt_end(tk: _Tokens, i: int) -> int:
    """인덱스 ``i`` 에서 시작하는 문장의 끝(다음 토큰 인덱스). 괄호 짝이 맞는 토큰열에서만 부른다."""
    toks, n = tk.toks, len(tk.toks)
    if i >= n:
        return n
    head = toks[i].text
    if head == "{":
        return tk.match[i] + 1
    if head in ("for", "while", "if", "switch") and i + 1 < n and toks[i + 1].text == "(":
        end = _stmt_end(tk, tk.match[i + 1] + 1)
        if head == "if" and end < n and toks[end].text == "else":
            end = _stmt_end(tk, end + 1)
        return end
    if head == "do":
        end = _stmt_end(tk, i + 1)
        if end + 1 < n and toks[end].text == "while" and toks[end + 1].text == "(":
            end = tk.match[end + 1] + 1
            if end < n and toks[end].text == ";":
                end += 1
        return end
    if head == "else":
        return _stmt_end(tk, i + 1)
    j = i
    while j < n:
        text = toks[j].text
        if text == ";":
            return j + 1
        if text in _CLOSE:
            return j
        j = tk.match[j] + 1 if text in _OPEN else j + 1
    return n


def _c_functions(tk: _Tokens) -> dict[str, tuple[int, int]]:
    """파일 범위의 함수 정의(``이름 ( … ) {``): 이름 → 본문 토큰 구간."""
    toks, n = tk.toks, len(tk.toks)
    out: dict[str, tuple[int, int]] = {}
    for i in tk.top_level(0, n):
        if not _is_ident(toks[i].text) or i + 1 >= n or toks[i + 1].text != "(" or i + 1 not in tk.match:
            continue
        k = tk.match[i + 1] + 1
        while k + 1 < n and toks[k].text == "__attribute__" and k + 1 in tk.match:
            k = tk.match[k + 1] + 1
        if k < n and toks[k].text == "{" and k in tk.match:
            out[toks[i].text] = (k + 1, tk.match[k])
    return out


def _c_increment(tk: _Tokens, pieces: list[tuple[int, int]], var: str | None) -> tuple[str | None, str]:
    """증분 식에서 (유도 변수, 단위가 아닌 증분). ``var`` 를 모르면 첫 증분 조각의 변수를 쓴다."""
    for a, b in pieces:
        t = tk.texts(a, b)
        if len(t) == 2 and t[0] in ("++", "--") and _is_ident(t[1]):
            name, step = t[1], ""
        elif len(t) == 2 and t[1] in ("++", "--") and _is_ident(t[0]):
            name, step = t[0], ""
        elif len(t) >= 3 and _is_ident(t[0]) and t[1] in ("+=", "-="):
            name, step = t[0], tk.span(a + 2, b)
        elif len(t) >= 5 and _is_ident(t[0]) and t[1] == "=" and t[2] == t[0] and t[3] in ("+", "-"):
            name, step = t[0], tk.span(a + 4, b)
        elif len(t) >= 3 and _is_ident(t[0]) and t[1] in ("=", "*=", "/=", "<<=", ">>="):
            name, step = t[0], ""   # 등차가 아닌 갱신
        else:
            continue
        if var is None or name == var:
            return name, step
    return var, ""


def _c_header(tk: _Tokens, i_for: int, close: int) -> _Loop:
    """``for ( init ; cond ; incr )`` 머리에서 유도 변수·상한·증분을 읽는다. 세 부분이 아니면 변수는 ``?`` 다."""
    toks = tk.toks
    line = toks[i_for].line
    loop = _Loop(var="?", line=line, col=tk.col(i_for), first_line=line)
    parts = tk.split(i_for + 2, close, ";")
    if len(parts) != 3:
        return loop
    (ia, ib), (ca, cb), (sa, sb) = parts
    eq = next((k for k in tk.top_level(ia, ib) if toks[k].text == "="), None)
    var = toks[eq - 1].text if eq is not None and eq > ia and _is_ident(toks[eq - 1].text) else None
    if var is not None and eq is not None:
        init_end = next((k for k in tk.top_level(eq + 1, ib) if toks[k].text == ","), ib)
        loop.init = tk.span(eq + 1, init_end)
    var, loop.step = _c_increment(tk, tk.split(sa, sb, ","), var)
    if var is None:
        var = next((t for t in tk.texts(ca, cb) if _is_ident(t)), None)
    loop.var = var or "?"

    top = list(tk.top_level(ca, cb))
    op = next((k for k in top if toks[k].text in _CMP_OPS), None)
    if op is not None:
        loop.line, loop.col = toks[op].line, tk.col(op)
    whole = tk.span(ca, cb)
    if op is None or any(toks[k].text in ("&&", "||") for k in top):
        loop.bound = whole
    elif tk.texts(ca, op) == [loop.var]:
        loop.bound = tk.span(op + 1, cb) if toks[op].text in ("<", "<=", "!=") else (loop.init or whole)
    elif tk.texts(op + 1, cb) == [loop.var]:
        loop.bound = tk.span(ca, op) if toks[op].text in (">", ">=", "!=") else (loop.init or whole)
    else:
        loop.bound = whole
    return loop


def _c_while_bodies(tk: _Tokens) -> list[tuple[int, int]]:
    """``while`` 문과 ``do … while`` 문의 본문 토큰 구간. ``do { … } while (0)`` (한 번만 도는 매크로 관용구)는 뺀다."""
    toks, n = tk.toks, len(tk.toks)
    out: list[tuple[int, int]] = []
    tails: set[int] = set()     # do 문 끝의 while 토큰
    for i, tok in enumerate(toks):
        if tok.text == "do":
            end = _stmt_end(tk, i + 1)
            if end + 1 < n and toks[end].text == "while" and toks[end + 1].text == "(":
                tails.add(end)
                if tk.texts(end + 2, tk.match[end + 1]) == ["0"]:
                    continue
            out.append((i + 1, end))
        elif tok.text == "while" and i not in tails and i + 1 < n and toks[i + 1].text == "(":
            close = tk.match[i + 1]
            out.append((close + 1, _stmt_end(tk, close + 1)))
    return out


def _c_loops(tk: _Tokens) -> list[_Loop]:
    """``for`` 문과 그 중첩. 본문은 중괄호 블록이거나 문장 하나다. ``while``·``do`` 는 루프 목록에 넣지 않고 그 안의 ``for`` 의
    ``wrapped`` 로만 센다."""
    toks = tk.toks
    loops: list[_Loop] = []
    open_loops: list[int] = []
    whiles = _c_while_bodies(tk)
    for i, tok in enumerate(toks):
        if tok.text != "for" or i + 1 >= len(toks) or toks[i + 1].text != "(":
            continue
        close = tk.match[i + 1]
        loop = _c_header(tk, i, close)
        loop.head = i
        loop.body = (close + 1, _stmt_end(tk, close + 1))
        loop.last_line = toks[min(loop.body[1], len(toks)) - 1].line
        loop.wrapped = sum(a <= i < b for a, b in whiles)
        while open_loops and loops[open_loops[-1]].body[1] <= i:
            open_loops.pop()
        loop.parent = open_loops[-1] if open_loops else None
        loops.append(loop)
        open_loops.append(len(loops) - 1)
    return loops


@dataclass
class _Omp:
    text: str                 # "parallel for schedule(static)"
    words: frozenset[str]     # 절의 괄호 안을 뺀 낱말
    collapse: int             # collapse(n) 의 n (없으면 1)
    loop: int | None          # 지시문 바로 뒤의 for (없으면 None)

    @property
    def parallel(self) -> bool:
        return "parallel" in self.words

    @property
    def worksharing(self) -> bool:
        return bool(self.words & {"for", "taskloop", "loop", "distribute"})

    @property
    def simd(self) -> bool:
        return "simd" in self.words and "declare" not in self.words

    @property
    def reduction(self) -> bool:
        return "reduction" in self.words and "declare" not in self.words


def _c_openmp(directives: list[tuple[int, str]], tk: _Tokens, loops: list[_Loop]) -> list[_Omp]:
    """``#pragma omp`` 지시문. 지시문 다음 줄의 첫 토큰이 ``for`` 면 그 루프가 지시문이 붙은 루프다."""
    tok_lines = [t.line for t in tk.toks]
    loop_of_for = {lp.head: idx for idx, lp in enumerate(loops)}
    out: list[_Omp] = []
    for last_line, text in directives:
        m = _OMP_RE.fullmatch(text)
        if m is None:
            continue
        body = m.group(1).strip()
        bare, prev = body, None
        while prev != bare:
            prev, bare = bare, re.sub(r"\([^()]*\)", " ", bare)
        collapse = re.search(r"\bcollapse\s*\(\s*(\d+)\s*\)", body)
        out.append(_Omp(
            text=_clip(body), words=frozenset(re.findall(r"[A-Za-z_]+", bare)),
            collapse=int(collapse.group(1)) if collapse else 1,
            loop=loop_of_for.get(bisect.bisect_right(tok_lines, last_line)),
        ))
    return out


@dataclass
class _GccVec:
    usable: bool = False                                        # 이 파일에 대한 벡터화기 리마크가 하나라도 있다
    vectorized: dict[int, int] = field(default_factory=dict)    # 루프 → 가장 넓은 벡터 바이트 수(모르면 0)
    slp: set[int] = field(default_factory=set)                  # 본문에 basic block 벡터화 리마크가 걸린 루프
    versioned: set[int] = field(default_factory=set)            # 별칭 검사로 버전이 갈린 루프
    reasons: dict[int, str] = field(default_factory=dict)       # 루프 → 벡터화하지 못한 이유(첫 번째)
    remarked: set[int] = field(default_factory=set)             # 리마크가 하나라도 붙은 루프
    missed: set[int] = field(default_factory=set)               # 루프 머리에 missed 리마크가 붙은 루프
    unmatched: int = 0                                         # for 루프에 대응시키지 못한 "loop vectorized" 줄 수
    stray: set[int] = field(default_factory=set)                # 어느 for 에도 속하지 않는 "couldn't vectorize loop" 의 줄


def _gcc_vectorization(build_log: str, loops: list[_Loop]) -> _GccVec:
    """gcc ``-fopt-info-vec-optimized -fopt-info-vec-missed`` 줄을 루프에 대응시킨다. 같은 루프가 벡터 크기마다 여러 번 나온다."""
    info = _GccVec()
    by_line: dict[int, list[int]] = {}
    for idx, lp in enumerate(loops):
        by_line.setdefault(lp.line, []).append(idx)

    def at(line: int, col: int) -> int | None:
        cands = by_line.get(line, [])
        if len(cands) > 1:
            cands = [i for i in cands if loops[i].col == col]
        return cands[0] if len(cands) == 1 else None

    def enclosing(line: int) -> int | None:
        inside = [i for i, lp in enumerate(loops) if lp.first_line <= line <= lp.last_line]
        return max(inside, key=lambda i: (loops[i].first_line, i)) if inside else None

    for raw in build_log.splitlines():
        m = _GCC_REMARK_RE.fullmatch(raw.strip())
        if m is None or not m.group("file").lower().endswith(".c"):
            continue
        info.usable = True
        line, col, kind, msg = int(m.group("line")), int(m.group("col")), m.group("kind"), m.group("msg").strip()
        header = at(line, col)
        if kind == "optimized" and "loop vectorized" in msg:
            if header is None:
                info.unmatched += 1
            else:
                width = re.search(r"(\d+) byte vectors", msg)
                info.vectorized[header] = max(info.vectorized.get(header, 0), int(width.group(1)) if width else 0)
                info.remarked.add(header)
            continue
        loop = header if header is not None else enclosing(line)   # 루프 머리가 아니면 그 줄을 본문에 둔 가장 안쪽 루프
        if loop is None:
            if kind == "missed" and "couldn't vectorize loop" in msg:
                info.stray.add(line)                               # for 문 밖의 루프(while·do): 벡터화되지 않았다
            continue
        info.remarked.add(loop)
        if kind == "missed" and header is not None:
            info.missed.add(header)
        if kind == "optimized" and "basic block" in msg and "vectorized" in msg:
            info.slp.add(loop)
        elif kind == "optimized" and "versioned for vectorization" in msg:
            info.versioned.add(loop)
        elif kind == "missed" and msg.startswith("not vectorized:"):
            info.reasons.setdefault(loop, _clip(msg[len("not vectorized:"):].strip().rstrip(".")))
    return info


def _c_double_accumulation(tk: _Tokens, loops: list[_Loop]) -> bool:
    """``double`` (또는 배정밀도 벡터)로 선언한 변수를 루프 안에서 누적하는가."""
    toks, n = tk.toks, len(tk.toks)
    names: set[str] = set()
    for i, tok in enumerate(toks):
        if tok.text not in _C_DOUBLE_TYPES:
            continue
        j = i + 1
        while j < n and _is_ident(toks[j].text) and toks[j].text not in _C_DECL_WORDS and not _C_TYPE_RE.fullmatch(toks[j].text):
            if j + 1 < n and toks[j + 1].text == "(":
                break                                    # 함수 선언자
            names.add(toks[j].text)
            k = j + 1
            while k < n and toks[k].text not in (",", ";") and toks[k].text not in _CLOSE:
                k = tk.match.get(k, k) + 1 if toks[k].text in _OPEN else k + 1
            if k >= n or toks[k].text != ",":
                break
            j = k + 1
    for i in range(n - 1):
        if toks[i].text not in names or not any(lp.body[0] <= i < lp.body[1] for lp in loops):
            continue
        op = toks[i + 1].text
        if op in ("+=", "-="):
            return True
        if op == "=":
            end = next((k for k in range(i + 2, n) if toks[k].text == ";"), n)
            if toks[i].text in tk.texts(i + 2, end):
                return True
    return False


def _c_symbolic_arrays(tk: _Tokens) -> int:
    """크기가 이름으로 된(가변 길이·매크로 크기) 지역·정적 배열 정의의 수. 리터럴 크기의 작은 스크래치와 const 표는 세지 않는다."""
    toks = tk.toks
    count = 0
    depth = 0
    for i, tok in enumerate(toks):
        if tok.text == "(":
            depth += 1
        elif tok.text == ")":
            depth -= 1
        elif tok.text == "[" and depth == 0 and i >= 2 and _is_ident(toks[i - 1].text):
            j = i - 2
            while j >= 0 and toks[j].text == "*":
                j -= 1
            close = tk.match.get(i)
            if j < 0 or close is None or not _C_TYPE_RE.fullmatch(toks[j].text):
                continue
            if not any(_is_ident(t) for t in tk.texts(i + 1, close)):
                continue
            k = j
            is_const = False
            while k >= 0 and (toks[k].text in _C_DECL_WORDS or _C_TYPE_RE.fullmatch(toks[k].text)):
                is_const = is_const or toks[k].text == "const"
                k -= 1
            if not is_const:
                count += 1
    return count


def _c_constants(source: str, directives: list[tuple[int, str]]) -> tuple[dict[str, int], list[str]]:
    """(정수 상수 이름 → 값, 본문에 루프가 있는 매크로 이름). 상수는 객체형 ``#define``, ``const`` 정수 선언, ``enum`` 멤버에서 온다."""
    raw: dict[str, str] = {m.group(1): m.group(2) for m in _C_CONST_DECL_RE.finditer(source)}
    for m in _C_ENUM_RE.finditer(source):
        for member in m.group(1).split(","):
            name, eq, value = member.partition("=")
            if eq:
                raw[name.strip()] = value
    hidden: list[str] = []
    for _, text in directives:
        d = _DEFINE_RE.fullmatch(text)
        if d is None:
            continue
        if _HIDDEN_LOOP_RE.search(d.group(3)):
            hidden.append(d.group(1))
        elif not d.group(2) and d.group(3).strip():
            raw[d.group(1)] = d.group(3)
    return _resolve_consts(raw), hidden


def _c_parallel(
    facts: _Facts, tk: _Tokens, loops: list[_Loop], omps: list[_Omp], build_log: str, *, complete: bool
) -> list[_Omp]:
    """OpenMP 지시문과 명시적 스레드에서 ``parallel`` 을 채운다. 반환: 효력이 있는 지시문(gcc 가 무시했다고 경고했으면 없다)."""
    toks = tk.toks
    facts.par_constructs = [o.text for o in omps]
    active = omps
    if omps and _OMP_IGNORED_RE.search(build_log):
        facts.notes.append("OpenMP pragmas ignored by the compiler (built without -fopenmp)")
        active = []
    regions = sum(o.parallel for o in active)
    threads = [toks[i].text for i in range(len(toks) - 1) if toks[i].text in _C_THREADS and toks[i + 1].text == "("]
    if regions:
        shared = [o for o in active if o.worksharing and o.loop is not None]
        for o in shared:
            cur = o.loop
            for _ in range(o.collapse):          # collapse(n): 바로 안쪽으로 n 단계가 함께 나뉜다
                if cur is None:
                    break
                facts.par_loops.append(loops[cur].var)
                cur = next((i for i, lp in enumerate(loops) if lp.parent == cur), None)
        facts.par_kind, facts.par_regions = "openmp", regions
        facts.par_collapse = any(o.collapse > 1 for o in active)
        facts.par_reduction = any(o.reduction for o in active)
        # while·do 안의 병렬 루프는 그 문장이 한 단계 더 바깥에 있는 것으로 센다(반복마다 병렬 루프가 한 번씩 돈다)
        facts.par_placement = _placement([len(_chain(loops, o.loop)) + loops[o.loop].wrapped for o in shared if o.loop is not None],
                                         regions)
        if any(o.loop is not None and loops[o.loop].wrapped for o in shared):
            facts.notes.append("parallel loop inside a while/do loop (runs once per iteration)")
        if not shared and not complete:
            facts.par_placement = "unknown"      # 지시문이 붙은 루프를 읽지 못했다
        facts.by_llm.add("parallel_region")
    elif threads:
        facts.par_kind, facts.par_regions, facts.par_placement = "threads", len(threads), "region"
        facts.par_constructs += threads
        facts.by_llm.add("parallel_region")
    if any(o.reduction for o in active):     # 병렬 영역 밖의 simd reduction 도 LLM 이 쓴 리덕션이다
        facts.by_llm.add("reduction")
    return active


def _c_vectorization(
    facts: _Facts, tk: _Tokens, loops: list[_Loop], active: list[_Omp], build_log: str, *, complete: bool
) -> None:
    """``vectorized`` 와 벡터화 결정의 주체. 판정 근거가 있는 가장 안쪽 루프만으로 정한다: 전부 벡터화됐으면 ``yes``, 일부면
    ``partial``, 하나도 아니면 ``no``. 로그가 없거나 루프 모형이 불완전하면(``complete=False``) ``unknown``.
    루프는 자신이나 감싸는 루프가 "loop vectorized" 로 보고됐거나 본문에 인트린식이 있으면 벡터화된 것이고, 머리에 missed 리마크가
    있거나 "not vectorized:" 사유가 붙었으면 벡터화되지 않은 것이다(basic block 벡터화만 된 루프는 일부). 벡터화 판정 리마크가
    없는 가장 안쪽 루프(gcc 가 memset 같은 라이브러리 호출로 바꾸거나 완전히 풀어 벡터화기가 보지 못한 루프)는 판정에서 빼고
    notes 에 남긴다 — missed 리마크 없이 "벡터화되지 않았다"고 적지 않는다. 근거가 있는 가장 안쪽 루프가 없으면 ``unknown``."""
    toks = tk.toks
    leaves = _leaf_loops(tk, loops, _c_functions(tk))
    sites = _simd_sites(tk)
    hand = {i for i in leaves if any(loops[i].body[0] <= s < loops[i].body[1] for s in sites)}
    gcc = _gcc_vectorization(build_log, loops)
    covered = {i for i in leaves if i in hand or any(a in gcc.vectorized for a in _chain(loops, i))}
    slp = [i for i in leaves if i in gcc.slp and i not in covered]
    unvectorized = [i for i in leaves if i not in covered and (i in gcc.missed or i in gcc.reasons)]
    silent = [i for i in leaves if i not in covered and i not in slp and i not in unvectorized]   # 판정 근거 없음
    if not complete:
        facts.vec_status = "unknown"
    elif not gcc.usable:
        if leaves and len(covered) == len(leaves):
            facts.vec_status = "yes"            # 로그 없이도: 가장 안쪽 루프가 전부 인트린식
    elif not leaves:                            # for 루프가 없다: for 밖의 루프(while·do)에 대한 리마크만 본다
        facts.vec_status = ("partial" if gcc.unmatched and gcc.stray else "yes" if gcc.unmatched
                            else "no" if gcc.stray else "unknown")
    elif len(silent) == len(leaves):
        facts.vec_status = "unknown"
    elif not slp and not unvectorized and not gcc.stray:
        facts.vec_status = "yes"
    else:
        facts.vec_status = "partial" if covered or slp or gcc.unmatched else "no"
    facts.vec_source = "intrinsics" if sites else "compiler_remark" if gcc.usable else "none"
    facts.vec_loops = [lp.var for i, lp in enumerate(loops) if i in gcc.vectorized or i in hand]

    requested = {o.loop for o in active if o.simd}      # LLM 이 omp simd 로 벡터화를 요구한 루프
    if sites or requested:
        facts.by_llm.add("vectorization")
    if any(i not in requested for i in gcc.vectorized) or slp or (gcc.unmatched and not requested):
        facts.by_compiler.add("vectorization")

    def at(idx: int) -> str:
        return f"{loops[idx].var}@{loops[idx].first_line}"

    if gcc.vectorized:
        facts.notes.append("gcc: vectorized " + _listing((f"{at(i)} ({w}-byte)" if w else at(i) for i, w in sorted(gcc.vectorized.items())), 4))
    if slp:
        facts.notes.append("gcc: basic block vectorized in " + _listing((at(i) for i in slp), 4))
    missed = [i for i in leaves if i not in covered and i in gcc.reasons]
    if missed:
        facts.notes.append("gcc: not vectorized " + _listing((f"{at(i)} ({gcc.reasons[i]})" for i in missed), 3))
    if gcc.usable and silent:
        facts.notes.append("gcc: no vectorizer verdict for " + _listing((at(i) for i in silent), 4)
                           + " (not counted; possibly replaced by a library call such as memset)")
    if gcc.versioned:
        facts.notes.append("gcc: versioned for aliasing " + _listing((at(i) for i in sorted(gcc.versioned)), 4))
    if gcc.unmatched:
        facts.notes.append(f"gcc: {gcc.unmatched} 'loop vectorized' remark(s) not matched to a for loop")
    if gcc.stray:
        facts.notes.append(f"gcc: {len(gcc.stray)} loop(s) outside any for statement not vectorized")
    intrinsics = [toks[s].text for s in sites if toks[s].text.startswith("_mm")]
    if intrinsics:
        facts.notes.append(f"intrinsics: {len(set(intrinsics))} distinct ({_listing(intrinsics, 4)})")
    builtins = [t.text for t in toks if t.text.startswith("__builtin_") and not _SIMD_RE.fullmatch(t.text)]
    if builtins:
        facts.notes.append(f"builtins: {_listing(builtins, 4)}")


def _analyze_c(language: str, code: str, build_log: str, spec: TaskSpec | None) -> _Facts:
    facts = _Facts(language)
    source, directives = _c_directives(_blank(code, rust=False))
    tk = _tokenize(source)
    toks = tk.toks
    consts, loop_macros = _c_constants(source, directives)
    used = {t.text for t in toks}
    hidden = [name for name in loop_macros if name in used]      # 쓰이지 않는 매크로는 아무것도 숨기지 않는다
    loops = _c_loops(tk) if tk.balanced else []
    if not tk.balanced:
        facts.notes.append("unbalanced brackets: loop nests omitted")
    if hidden:
        facts.notes.append(f"macro may hide a loop: {_listing(hidden, 3)}")
    facts.nests = _nests(loops)

    complete = tk.balanced and not hidden

    active = _c_parallel(facts, tk, loops, _c_openmp(directives, tk, loops), build_log, complete=complete)
    _c_vectorization(facts, tk, loops, active, build_log, complete=complete)
    _loop_decisions(facts, loops, consts, lambda i: _fragments(tk, loops[i].body), _C_CLAMP_RE, complete=complete)
    facts.library_calls = [
        toks[i].text for i in range(len(toks) - 1)
        if toks[i + 1].text == "(" and (toks[i].text in _C_LIBRARY or toks[i].text.startswith(_C_LIBRARY_PREFIXES))
    ]
    facts.temporaries = sum(c in _C_ALLOC for c in facts.library_calls) + _c_symbolic_arrays(tk)
    facts.transposed = _transposed_copy(tk, loops) or (False if complete else None)
    if facts.temporaries or facts.transposed:
        facts.by_llm.add("memory_layout")
    if any(c.startswith(_C_DELEGATING_PREFIXES) for c in facts.library_calls):
        facts.by_library.add("library_delegation")
    if _is_float_task(spec) and _c_double_accumulation(tk, loops):
        facts.by_llm.add("accumulation_precision")
    return facts


# ----------------------------------------------------------------------------- Rust


_RUST_SORTS = frozenset({
    "sort", "sort_unstable", "sort_by", "sort_by_key", "sort_by_cached_key", "sort_unstable_by", "sort_unstable_by_key",
    "par_sort", "par_sort_unstable", "par_sort_by", "par_sort_by_key", "par_sort_unstable_by", "par_sort_unstable_by_key",
})
_RUST_BULK = frozenset({"copy_from_slice", "clone_from_slice", "fill", "copy_nonoverlapping", "write_bytes", "swap_with_slice"})
_RUST_REDUCERS = frozenset({"sum", "product", "reduce", "reduce_with", "fold", "min", "max", "count"})
_RAYON_METHOD_RE = re.compile(r"par_[a-z_]+|into_par_iter")
_RAYON_FUNCTIONS = frozenset({"join", "scope", "scope_fifo", "in_place_scope", "spawn"})
_RUST_CONST_RE = re.compile(r"\b(?:const|static|let)\s+(\w+)\s*(?::\s*[\w:<>]+\s*)?=\s*([^;{}]+);")
_RUST_CLAMP_RE = re.compile(r"\bmin\s*\(|\bif\b")
_RUST_USE_RAYON_RE = re.compile(r"\buse\s+rayon\s*::([^;]*);")


def _rust_loops(tk: _Tokens) -> list[_Loop]:
    """``for PAT in EXPR { … }`` 와 그 중첩(``impl … for``, ``for<'a>`` 는 ``in`` 이 없어 걸러진다)."""
    toks, n = tk.toks, len(tk.toks)
    loops: list[_Loop] = []
    open_loops: list[int] = []
    for i, tok in enumerate(toks):
        if tok.text != "for":
            continue
        k = i + 1
        in_idx = brace = None
        while k < n:
            text = toks[k].text
            if text == "{":
                brace = k
                break
            if text in (";", "}", ")", "]") or (in_idx is None and text in ("<", "=")):
                break
            if text == "in" and in_idx is None:
                in_idx = k
            k = tk.match.get(k, k) + 1 if text in _OPEN else k + 1
        if in_idx is None or brace is None or brace not in tk.match or in_idx == i + 1:
            continue
        loop = _Loop(var=tk.span(i + 1, in_idx), line=tok.line, first_line=tok.line, head=i,
                     body=(brace + 1, tk.match[brace]), last_line=toks[tk.match[brace]].line)
        a, b = in_idx + 1, brace
        if toks[a].text == "(" and tk.match.get(a, b) < b:      # (lo..hi).step_by(S) / .rev()
            tail = tk.texts(tk.match[a] + 1, b)
            if tail[:3] == [".", "step_by", "("]:
                loop.step = tk.span(tk.match[a] + 4, tk.match.get(tk.match[a] + 3, b))
            a, b = a + 1, tk.match[a]
        dots = next((k for k in tk.top_level(a, b) if toks[k].text in ("..", "..=")), None)
        if dots is None:
            loop.bound = tk.span(in_idx + 1, brace)
        else:
            loop.init, loop.bound = tk.span(a, dots), tk.span(dots + 1, b)
        while open_loops and loops[open_loops[-1]].body[1] <= i:
            open_loops.pop()
        loop.parent = open_loops[-1] if open_loops else None
        loops.append(loop)
        open_loops.append(len(loops) - 1)
    return loops


def _rust_functions(tk: _Tokens) -> dict[str, tuple[int, int]]:
    """``fn 이름 … {`` 정의: 이름 → 본문 토큰 구간."""
    toks, n = tk.toks, len(tk.toks)
    out: dict[str, tuple[int, int]] = {}
    for i in range(n - 1):
        if toks[i].text != "fn" or not _is_ident(toks[i + 1].text):
            continue
        brace = next((k for k in tk.top_level(i + 2, n) if toks[k].text in ("{", ";")), None)
        if brace is not None and toks[brace].text == "{" and brace in tk.match:
            out[toks[i + 1].text] = (brace + 1, tk.match[brace])
    return out


def _rust_chain_methods(tk: _Tokens, site: int) -> list[str]:
    """병렬 반복자 진입점 ``site`` 에서 같은 식에 이어진 메서드 이름들(인자 안의 호출은 건너뛴다)."""
    toks, n = tk.toks, len(tk.toks)
    names: list[str] = []
    k = site + 1
    while k < n:
        text = toks[k].text
        if text in (";", ",") or text in _CLOSE:
            break
        if text == "." and k + 1 < n and _is_ident(toks[k + 1].text):
            names.append(toks[k + 1].text)
        k = tk.match.get(k, k) + 1 if text in _OPEN else k + 1
    return names


def _rust_parallel(facts: _Facts, tk: _Tokens, loops: list[_Loop], source: str) -> None:
    """rayon 병렬 반복자(체인 하나가 영역 하나), ``rayon::join``·``scope``, std 스레드에서 ``parallel`` 을 채운다."""
    toks, n = tk.toks, len(tk.toks)
    imported = {name for m in _RUST_USE_RAYON_RE.finditer(source) for name in _IDENT_RE.findall(m.group(1))}
    placements: set[str] = set()
    regions = 0
    threads: list[str] = []
    for i, tok in enumerate(toks):
        before = toks[i - 1].text if i else ""
        owner = toks[i - 2].text if i >= 2 else ""
        if i + 1 >= n or toks[i + 1].text not in ("(", "::"):
            continue
        if _RAYON_METHOD_RE.fullmatch(tok.text) and before == ".":
            facts.par_constructs.append(tok.text)
            opener = next((k for k in range(i - 1, -1, -1) if toks[k].text in _OPEN and tk.match.get(k, -1) > i), None)
            if opener is not None and opener >= 2 and toks[opener - 1].text in ("zip", "zip_eq", "chain") and toks[opener - 2].text == ".":
                continue                             # 다른 병렬 체인의 zip 인자: 같은 영역이다
            regions += 1
            placements.add("inner" if any(lp.body[0] <= i < lp.body[1] for lp in loops) else "outer")
            if set(_rust_chain_methods(tk, i)) & _RUST_REDUCERS:
                facts.par_reduction = True
        elif tok.text in _RAYON_FUNCTIONS and ((before == "::" and owner == "rayon") or (before not in (".", "::") and tok.text in imported)):
            facts.par_constructs.append(f"rayon::{tok.text}")
            regions += 1
            placements.add("region")
        elif tok.text in ("spawn", "scope") and before == "::" and owner == "thread":
            threads.append(f"thread::{tok.text}")
    facts.par_constructs += threads
    if regions:
        facts.par_kind, facts.par_regions = "rayon", regions
        facts.par_placement = placements.pop() if len(placements) == 1 else "mixed"
        facts.by_llm.add("parallel_region")
        if facts.par_reduction:
            facts.by_llm.add("reduction")
    elif threads:
        facts.par_kind, facts.par_regions, facts.par_placement = "threads", len(threads), "region"
        facts.by_llm.add("parallel_region")


def _rust_wide_accumulation(tk: _Tokens) -> bool:
    """f64 로 누적하는가: ``let mut`` 로 선언한 f64 누산기에 ``+=``, ``sum::<f64>()``, ``fold(0.0f64, …)``."""
    toks, n = tk.toks, len(tk.toks)
    wide: set[str] = set()
    for i in range(n - 2):
        if toks[i].text == "let" and toks[i + 1].text == "mut" and _is_ident(toks[i + 2].text):
            end = next((k for k in range(i + 3, n) if toks[k].text == ";"), n)
            if any(t.endswith("f64") for t in tk.texts(i + 3, end)):
                wide.add(toks[i + 2].text)
    for i in range(n - 1):
        text, after = toks[i].text, toks[i + 1].text
        if text in wide and after in ("+=", "-="):
            return True
        if text == "sum" and tk.texts(i + 1, i + 5) == ["::", "<", "f64", ">"]:
            return True
        if text == "fold" and after == "(" and any(t.endswith("f64") for t in tk.texts(i + 2, i + 5)):
            return True
    return False


def _analyze_rust(language: str, code: str, spec: TaskSpec | None) -> _Facts:
    facts = _Facts(language)
    source = _blank(code, rust=True)
    tk = _tokenize(source)
    toks, n = tk.toks, len(tk.toks)
    loops = _rust_loops(tk) if tk.balanced else []
    if not tk.balanced:
        facts.notes.append("unbalanced brackets: loop nests omitted")
    facts.nests = _nests(loops)
    _rust_parallel(facts, tk, loops, source)

    # 벡터화: 컴파일러 리마크가 없으므로 손으로 쓴 SIMD(std::arch 인트린식, std::simd)만 확인된다
    sites = _simd_sites(tk)
    leaves = _leaf_loops(tk, loops, _rust_functions(tk))
    hand = [i for i in leaves if any(loops[i].body[0] <= s < loops[i].body[1] for s in sites)]
    portable = any(tk.texts(i, i + 3) in (["std", "::", "simd"], ["core", "::", "simd"]) for i in range(n))
    if sites or portable:
        facts.vec_source = "intrinsics"
        facts.by_llm.add("vectorization")
        if tk.balanced and leaves and len(hand) == len(leaves):
            facts.vec_status = "yes"
    facts.vec_loops = [loops[i].var for i in hand]
    intrinsics = [toks[s].text for s in sites if toks[s].text.startswith("_mm")]
    if intrinsics:
        facts.notes.append(f"intrinsics: {len(set(intrinsics))} distinct ({_listing(intrinsics, 4)})")

    consts = _resolve_consts({m.group(1): m.group(2) for m in _RUST_CONST_RE.finditer(source)})
    _loop_decisions(facts, loops, consts, lambda i: _fragments(tk, loops[i].body), _RUST_CLAMP_RE, complete=tk.balanced)

    temporaries = unsafe_blocks = exact_chunks = 0
    for i in range(n - 1):
        text, after = toks[i].text, toks[i + 1].text
        before = toks[i - 1].text if i else ""
        if after == "(" and before == "." and text in _RUST_SORTS:
            facts.library_calls.append(text)
            facts.by_library.add("library_delegation")       # 정렬 알고리즘을 표준 라이브러리·rayon 에 넘겼다
        elif after == "(" and before in (".", "::") and text in _RUST_BULK:
            facts.library_calls.append(text)
        if (text == "vec" and after == "!") or (before == "." and text in ("to_vec", "collect")) or (
            text in ("with_capacity", "new") and before == "::" and i >= 2 and toks[i - 2].text == "Vec"
        ):
            temporaries += 1
        unsafe_blocks += text == "unsafe" and after == "{"
        exact_chunks += before == "." and text in ("chunks_exact", "chunks_exact_mut")
    facts.temporaries = temporaries
    facts.transposed = _transposed_copy(tk, loops) or (False if tk.balanced else None)
    if temporaries or facts.transposed:
        facts.by_llm.add("memory_layout")
    if _is_float_task(spec) and _rust_wide_accumulation(tk):
        facts.by_llm.add("accumulation_precision")
    facts.notes.append(f"unsafe blocks: {unsafe_blocks}")
    if exact_chunks:
        facts.notes.append(f"chunks_exact: {exact_chunks} use(s)")
    return facts


# ----------------------------------------------------------------------------- Python (numpy · numba)


# NumPy 함수(numpy 모듈 아래의 이름) → 역할(_PyCall.role). 여기에 없는 함수는 "compute"(배열을 계산해 새 배열을 돌려준다)다.
_NP_ROLES: dict[str, str] = {
    **dict.fromkeys((     # 형·설정·조회: 계산이 아니다(호출 목록에도 싣지 않는다)
        "shape", "ndim", "size", "dtype", "errstate", "seterr", "geterr", "iinfo", "finfo", "isscalar", "result_type", "can_cast",
        "issubdtype", "may_share_memory", "shares_memory", "float16", "float32", "float64", "double", "single", "longdouble",
        "int8", "int16", "int32", "int64", "intp", "intc", "uint8", "uint16", "uint32", "uint64", "bool_", "complex64", "complex128",
    ), "meta"),
    **dict.fromkeys((     # 새 배열도 계산도 없다: 뷰·형 맞춤·덩어리 복사
        "asarray", "asanyarray", "reshape", "ravel", "transpose", "swapaxes", "moveaxis", "squeeze", "expand_dims", "broadcast_to",
        "atleast_1d", "atleast_2d", "require", "copyto",
    ), "plain"),
    **dict.fromkeys((     # 새 배열을 만들 뿐이다
        "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like", "array", "copy",
        "ascontiguousarray", "asfortranarray", "arange",
    ), "alloc"),
    **dict.fromkeys((     # 축약: 축을 남기지 않으면 스칼라를 돌려준다
        "dot", "vdot", "inner", "sum", "prod", "max", "min", "amax", "amin", "mean", "trace", "any", "all", "argmax", "argmin",
        "count_nonzero", "nansum", "linalg.norm",
    ), "reducer"),
    **dict.fromkeys((     # 원소별 수학 함수: 스칼라에도 쓰인다 → 인자가 배열일 때만 배열 계산이다
        "sqrt", "cbrt", "square", "abs", "absolute", "fabs", "sign", "exp", "exp2", "expm1", "log", "log2", "log10", "log1p",
        "sin", "cos", "tan", "arcsin", "arccos", "arctan", "arctan2", "sinh", "cosh", "tanh", "hypot", "power", "float_power",
        "reciprocal", "floor", "ceil", "trunc", "rint", "round", "minimum", "maximum", "fmin", "fmax", "add", "subtract", "multiply",
        "divide", "true_divide", "floor_divide", "mod", "remainder", "negative", "conj", "conjugate", "real", "imag", "isfinite",
        "isnan", "isinf",
    ), "elementwise"),
}
_NP_WORK_BUFFERS = frozenset({     # 작업 배열을 명시적으로 만드는 호출(메모리 배치 결정으로 본다); ndarray.copy 포함
    "empty", "zeros", "ones", "full", "empty_like", "zeros_like", "ones_like", "full_like", "copy", "ascontiguousarray", "asfortranarray",
})
_ND_METHODS: dict[str, str] = {    # ndarray 메서드 → 역할(_PyCall.role)
    **dict.fromkeys(("copy", "astype", "flatten"), "alloc"),
    **dict.fromkeys(("reshape", "ravel", "view", "transpose", "swapaxes", "squeeze"), "plain"),
    **dict.fromkeys(("dot", "sum", "prod", "max", "min", "mean", "any", "all", "argmax", "argmin", "trace"), "reducer"),
    **dict.fromkeys(("sort", "partition", "put"), "inplace"),
    **dict.fromkeys(("cumsum", "cumprod", "argsort", "clip", "repeat", "take", "round", "conj", "searchsorted", "argpartition",
                     "nonzero", "compress"), "compute"),
}
_PRODUCT_FUNCS = frozenset({"dot", "inner"})      # 인자가 2차원 이상이면 스칼라가 아니라 배열을 돌려주는 축약
_TRANSPOSERS = frozenset({"transpose", "swapaxes", "asfortranarray"})
_FLOAT64_NAMES = frozenset({"float64", "double", "longdouble"})
_THREAD_APIS = frozenset({
    "threading.Thread", "concurrent.futures.ThreadPoolExecutor", "concurrent.futures.ProcessPoolExecutor",
    "concurrent.futures.thread.ThreadPoolExecutor", "multiprocessing.Pool", "multiprocessing.Process", "multiprocessing.pool.ThreadPool",
})
_IGNORED_MODULES = ("numba", "halide")            # 컴파일러·DSL 이름공간: 라이브러리 호출로 세지 않는다
_PY_CLAMP_RE = re.compile(r"\bmin\(|\bminimum\(| if ")
_NUMBA_OPTIONS = ("parallel", "fastmath", "boundscheck", "nogil", "cache")
_NUMBA_LISTING_RE = re.compile(r"(?P<src>.*?)-+\|\s*#(?P<ids>\d+(?:,\s*\d+)*)\s*")


def _dotted(node: ast.AST) -> list[str] | None:
    """``np.linalg.cholesky`` → ["np", "linalg", "cholesky"]. 뿌리가 이름이 아니면 None."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return parts[::-1]
    return None


def _root_name(node: ast.AST) -> str | None:
    """``A[i].T``, ``A.ravel()[i]`` → "A"."""
    while True:
        if isinstance(node, (ast.Attribute, ast.Subscript, ast.Starred)):
            node = node.value
        elif isinstance(node, ast.Call):
            node = node.func
        else:
            return node.id if isinstance(node, ast.Name) else None


def _index_parts(sub: ast.Subscript) -> list[ast.expr]:
    return list(sub.slice.elts) if isinstance(sub.slice, ast.Tuple) else [sub.slice]


def _is_slice_part(node: ast.expr) -> bool:
    return isinstance(node, ast.Slice) or (isinstance(node, ast.Constant) and node.value in (Ellipsis, None))


def _assign_pairs(target: ast.expr, value: ast.expr) -> list[tuple[str, ast.expr]]:
    """대입문의 (이름, 값) 짝. ``a, b = b, a`` 같은 묶음 대입은 자리마다 푼다."""
    if isinstance(target, ast.Name):
        return [(target.id, value)]
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)) and len(target.elts) == len(value.elts):
        return [pair for t, v in zip(target.elts, value.elts) for pair in _assign_pairs(t, v)]
    return []


def _names_float64(node: ast.AST, imports: dict[str, str]) -> bool:
    """노드가 배정밀도 dtype 을 이름으로 가리키는가(``np.float64``, 임포트한 ``float64``, ``"float64"``)."""
    if isinstance(node, ast.Attribute):
        return node.attr in _FLOAT64_NAMES
    if isinstance(node, ast.Name):
        return imports.get(node.id, "").rsplit(".", 1)[-1] in _FLOAT64_NAMES
    return isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in ("float64", "f8")


def _py_imports(tree: ast.Module) -> dict[str, str]:
    """임포트로 묶인 이름 → 전체 이름(``np`` → "numpy", ``prange`` → "numba.prange")."""
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                out[alias.asname or top] = alias.name if alias.asname else top
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                out[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return out


def _py_consts(tree: ast.Module) -> dict[str, int]:
    """한 번만 대입되는 정수 상수 이름(``TILE = 64``)."""
    stores: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            stores[node.id] = stores.get(node.id, 0) + 1
    raw = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if stores.get(name) == 1:
                raw[name] = ast.unparse(node.value)
    return _resolve_consts(raw)


@dataclass(frozen=True)
class _PyCall:
    name: str   # library_calls 에 싣는 이름("np.matmul", "ndarray.copy", "scipy.sparse.csr_matrix")
    role: str   # alloc(새 배열, 계산 없음) | plain(둘 다 없음) | reducer(축약) | elementwise(원소별 수학) | compute(계산해 새 배열)
    #             | inplace(제자리 계산) | other(NumPy 밖)


class _PyModule:
    """Python 소스의 AST 관찰: 임포트 별칭, 배열로 보이는 이름, 루프, 라이브러리 호출, 임시 배열."""

    def __init__(self, tree: ast.Module, spec: TaskSpec | None) -> None:
        self.tree = tree
        self.imports = _py_imports(tree)
        self.functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.params = {a.arg for fn in self.functions for a in (*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs)}
        self.arrays: dict[str, int | None] = {}   # 배열 값을 가리키는 이름 → 차원 수(모르면 None)
        self.aliases: set[str] = set()            # 배열에서 꺼낸 값을 가리키는 이름(원소일 수도 있다; 원소 루프 판정용)
        self._infer_arrays(spec)

        self.calls: list[tuple[tuple[int, int], str]] = []
        self.temporaries = 0
        self.delegated = False        # 배열 계산을 라이브러리 루프에 넘긴 자리가 있다
        self.uses_out = False
        self.allocates = False        # 작업 배열을 명시적으로 만든다
        self.transposed = False
        self.float64 = False
        self.sort_kind = False        # 정렬 호출에 kind= 를 줬다
        self.thread_sites: list[str] = []
        self._scan(tree, False)

    # ------------------------------------------------------------------ 배열 추론

    def _infer_arrays(self, spec: TaskSpec | None) -> None:
        if spec is not None:
            self.arrays.update({a.name: len(a.shape) for a in spec.args if a.kind != "dim"})
        else:   # 명세가 없으면 배열로 쓰인 흔적(첨자·속성 접근·@ 피연산자)이 있는 매개변수만 배열로 본다
            used: set[str] = set()
            for node in ast.walk(self.tree):
                if isinstance(node, (ast.Subscript, ast.Attribute)) and isinstance(node.value, ast.Name):
                    used.add(node.value.id)
                elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
                    used.update(x.id for x in (node.left, node.right) if isinstance(x, ast.Name))
            self.arrays.update(dict.fromkeys(sorted(self.params & used)))
        pairs: list[tuple[str, ast.expr]] = []
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    pairs.extend(_assign_pairs(target, node.value))
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                pairs.extend(_assign_pairs(node.target, node.value))
        pairs.sort(key=lambda p: (p[1].lineno, p[1].col_offset))
        for _ in range(4):      # 대입 순서와 무관하게 퍼지도록 몇 번 돈다
            before = (len(self.arrays), len(self.aliases))
            for name, value in pairs:
                if name in self.arrays:
                    continue
                if self.is_array(value):
                    self.arrays[name] = self.rank(value)
                elif isinstance(value, (ast.Subscript, ast.Attribute, ast.Name)) and self.from_array(value):
                    self.aliases.add(name)
            if before == (len(self.arrays), len(self.aliases)):
                break

    def from_array(self, node: ast.AST) -> bool:
        """식의 뿌리 이름이 매개변수이거나 배열(또는 배열에서 꺼낸 값)을 가리키는 이름인가."""
        root = _root_name(node)
        return root in self.params or root in self.arrays or root in self.aliases

    def rank(self, node: ast.AST) -> int | None:
        """배열 식의 차원 수(알 수 있을 때만)."""
        if isinstance(node, ast.Name):
            return self.arrays.get(node.id)
        if isinstance(node, ast.Subscript):
            base = self.rank(node.value)
            if base is None:
                return None
            parts = _index_parts(node)
            scalars = sum(1 for p in parts if not _is_slice_part(p))
            added = sum(1 for p in parts if isinstance(p, ast.Constant) and p.value is None)
            return max(base - scalars + added, 0)
        if isinstance(node, ast.Attribute) and node.attr == "T":
            return self.rank(node.value)
        return None

    def is_array(self, node: ast.AST) -> bool:
        """식의 값이 (원소가 아니라) 배열인가. 확실한 근거가 있을 때만 True."""
        if isinstance(node, ast.Name):
            return node.id in self.arrays
        if isinstance(node, ast.Subscript):
            parts = _index_parts(node)
            if any(_is_slice_part(p) or self.is_array(p) for p in parts):
                return True
            base = self.rank(node.value)
            return base is not None and base - len(parts) >= 1
        if isinstance(node, ast.Attribute):
            return node.attr in ("T", "real", "imag") and self.is_array(node.value)
        if isinstance(node, ast.BinOp):
            return isinstance(node.op, ast.MatMult) or self.is_array(node.left) or self.is_array(node.right)
        if isinstance(node, ast.UnaryOp):
            return self.is_array(node.operand)
        if isinstance(node, ast.IfExp):
            return self.is_array(node.body) or self.is_array(node.orelse)
        if isinstance(node, ast.Call):
            call = self.classify(node)
            if call is None:
                return False
            if call.role == "plain":        # 뷰는 받은 것이 배열이면 배열이다
                source = node.func.value if call.name.startswith("ndarray.") else (node.args[0] if node.args else None)
                return source is not None and self.is_array(source)
            return self.produces_array(node, call)
        return False

    # ------------------------------------------------------------------ 호출 분류

    def qualified(self, func: ast.AST) -> str | None:
        """임포트한 모듈을 거친 호출의 전체 이름("numpy.linalg.cholesky")."""
        parts = _dotted(func)
        if not parts or parts[0] not in self.imports:
            return None
        return ".".join([self.imports[parts[0]], *parts[1:]])

    def classify(self, node: ast.Call) -> _PyCall | None:
        full = self.qualified(node.func)
        if full is not None:
            if full.split(".")[0] in _IGNORED_MODULES:
                return None
            if not full.startswith("numpy."):
                return _PyCall(full, "other")
            leaf = full[len("numpy."):]
            role = _NP_ROLES.get(leaf, "compute")
            return None if role == "meta" else _PyCall(f"np.{leaf}", role)
        if isinstance(node.func, ast.Attribute) and node.func.attr in _ND_METHODS:
            receiver = node.func.value      # 배열로 확인된 값이거나 산술 식의 결과((x * y).sum())일 때만 ndarray 메서드로 본다
            if self.is_array(receiver) or isinstance(receiver, ast.BinOp):
                return _PyCall(f"ndarray.{node.func.attr}", _ND_METHODS[node.func.attr])
        return None

    def computes_on_arrays(self, node: ast.Call, call: _PyCall, *, stored_to_slice: bool = False) -> bool:
        """호출이 배열 계산을 라이브러리 안에서 하는가. 원소별 수학 함수는 인자가 배열로 확인될 때(또는 ``out=`` 이 있거나 결과를
        슬라이스에 대입할 때)만 그렇다 — ``np.sqrt(r2)`` 처럼 스칼라에 쓰는 것은 위임이 아니다."""
        if call.role == "elementwise":
            return stored_to_slice or any(k.arg == "out" for k in node.keywords) or any(self.is_array(a) for a in node.args)
        return call.role in ("compute", "reducer", "inplace") or call.name.startswith("scipy.")

    def produces_array(self, node: ast.Call, call: _PyCall, *, stored_to_slice: bool = False) -> bool:
        """호출이 새 배열을 만드는가(``out=`` 이 있으면 아니다; 축약은 축을 남길 때만)."""
        if any(k.arg == "out" for k in node.keywords) or call.role in ("plain", "inplace", "other"):
            return False
        if call.role == "elementwise":
            return self.computes_on_arrays(node, call, stored_to_slice=stored_to_slice)
        if call.role != "reducer":
            return True
        if any(k.arg == "axis" and not (isinstance(k.value, ast.Constant) and k.value.value is None) for k in node.keywords):
            return True
        if call.name.rsplit(".", 1)[-1] not in _PRODUCT_FUNCS:
            return False
        operands = [*node.args] + ([node.func.value] if call.name.startswith("ndarray.") else [])
        ranks = [self.rank(a) for a in operands]
        if any(r is not None and r >= 2 for r in ranks):
            return True
        return stored_to_slice and any(r is None for r in ranks)

    # ------------------------------------------------------------------ 훑기

    def _scan(self, node: ast.AST, in_arith: bool, stored_to_slice: bool = False) -> None:
        if isinstance(node, ast.BinOp):
            array = self.is_array(node)
            if array:
                self.delegated = True
                if isinstance(node.op, ast.MatMult):
                    self.calls.append(((node.lineno, node.col_offset), "np.matmul"))
                if not in_arith:
                    self.temporaries += 1     # 산술 식 하나가 임시 배열 하나(가장 바깥 연산에서 한 번만 센다)
            self._scan(node.left, array)
            self._scan(node.right, array)
            return
        if isinstance(node, ast.UnaryOp):
            array = self.is_array(node.operand)
            if array and not in_arith:
                self.delegated = True
                self.temporaries += 1
            self._scan(node.operand, array or in_arith)
            return
        if isinstance(node, ast.Call):
            self._scan_call(node, stored_to_slice)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript) \
                and any(_is_slice_part(p) for p in _index_parts(node.targets[0])):
            self._scan(node.targets[0], False)
            self._scan(node.value, False, stored_to_slice=True)
            return
        elif isinstance(node, ast.AugAssign) and (self.is_array(node.target) or isinstance(node.op, ast.MatMult)):
            self.delegated = True             # 제자리 배열 연산: 임시 배열은 없지만 루프는 라이브러리 안에 있다
        elif isinstance(node, ast.Attribute) and node.attr == "T" and self.from_array(node.value):
            self.transposed = True
        if _names_float64(node, self.imports):
            self.float64 = True
        for child in ast.iter_child_nodes(node):
            self._scan(child, False)

    def _scan_call(self, node: ast.Call, stored_to_slice: bool) -> None:
        for kw in node.keywords:
            if kw.arg == "out":
                self.uses_out = True
            elif kw.arg == "order" and isinstance(kw.value, ast.Constant) and kw.value.value == "F":
                self.transposed = True
            elif kw.arg == "dtype" and isinstance(kw.value, ast.Name) and kw.value.id == "float":
                self.float64 = True
        full = self.qualified(node.func)
        if full == "math.fsum":
            self.float64 = True
        if full in _THREAD_APIS:
            self.thread_sites.append(full.rsplit(".", 1)[-1])
        call = self.classify(node)
        if call is None:
            return
        self.calls.append(((node.lineno, node.col_offset), call.name))
        leaf = call.name.rsplit(".", 1)[-1]
        if self.computes_on_arrays(node, call, stored_to_slice=stored_to_slice):
            self.delegated = True
        if call.role == "alloc" and leaf in _NP_WORK_BUFFERS:
            self.allocates = True
        if leaf in _TRANSPOSERS:
            self.transposed = True
        if leaf in ("sort", "argsort") and any(k.arg == "kind" for k in node.keywords):
            self.sort_kind = True
        if self.produces_array(node, call, stored_to_slice=stored_to_slice):
            self.temporaries += 1

    # ------------------------------------------------------------------ 루프

    def loops(self) -> list[_Loop]:
        """모든 ``for`` 문과 그 중첩(함수 정의를 넘어서는 중첩은 없다). ``while`` 은 루프 목록에 넣지 않고 그 안의 ``for`` 의
        ``wrapped`` 로만 센다."""
        out: list[_Loop] = []

        def walk(stmts: list[ast.stmt], parent: int | None, owner: ast.AST | None, wraps: int = 0) -> None:
            for stmt in stmts:
                if isinstance(stmt, (ast.For, ast.AsyncFor)):
                    out.append(self._loop(stmt, parent, owner))
                    out[-1].wrapped = wraps
                    walk(stmt.body, len(out) - 1, owner, wraps)
                    walk(stmt.orelse, parent, owner, wraps)
                elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    walk(stmt.body, None, owner if isinstance(stmt, ast.ClassDef) else stmt)
                else:   # if·while·with·try·match: 문장 목록을 가진 필드를 따라 내려간다
                    walk(getattr(stmt, "body", []), parent, owner, wraps + isinstance(stmt, ast.While))
                    for name in ("orelse", "finalbody"):
                        walk(getattr(stmt, name, []), parent, owner, wraps)
                    for block in [*getattr(stmt, "handlers", []), *getattr(stmt, "cases", [])]:
                        walk(block.body, parent, owner, wraps)

        walk(self.tree.body, None, None)
        return out

    def _loop(self, node: ast.For | ast.AsyncFor, parent: int | None, owner: ast.AST | None) -> _Loop:
        loop = _Loop(var=_clip(ast.unparse(node.target)), parent=parent, node=node, owner=owner,
                     line=node.lineno, first_line=node.lineno, last_line=node.end_lineno or node.lineno)
        it = node.iter
        parts = _dotted(it.func) if isinstance(it, ast.Call) else None
        is_prange = bool(parts) and parts[-1] == "prange" and self.imports.get(parts[0], "").startswith("numba")
        if isinstance(it, ast.Call) and (parts == ["range"] or is_prange) and not it.keywords and 1 <= len(it.args) <= 3:
            args = [_clip(ast.unparse(a)) for a in it.args]
            start, stop = ("0", args[0]) if len(args) == 1 else (args[0], args[1])
            step = args[2] if len(args) == 3 else ""
            loop.init, loop.bound = (stop, start) if step.startswith("-") else (start, stop)
            loop.step = "" if step.lstrip("-") in ("", "1") else step.lstrip("-")
            loop.prange = is_prange
        else:
            loop.bound = _clip(ast.unparse(it))
        return loop

    def is_element_loop(self, loop: _Loop) -> bool:
        """본문이 배열(매개변수나 거기서 온 이름)을 루프 변수로 첨자하는가. 슬라이스의 경계로만 쓰는 블록 루프는 아니다."""
        variables = {n.id for n in ast.walk(loop.node.target) if isinstance(n, ast.Name)}
        return any(
            isinstance(node, ast.Subscript) and self.from_array(node.value)
            and any(isinstance(n, ast.Name) and n.id in variables
                    for part in _index_parts(node) if not isinstance(part, ast.Slice) for n in ast.walk(part))
            for stmt in loop.node.body for node in ast.walk(stmt)
        )


def _py_fragments(loop: _Loop) -> list[str]:
    simple = (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.Return)
    return [ast.unparse(n) for stmt in loop.node.body for n in ast.walk(stmt) if isinstance(n, simple)]


def _numba_options(fn: ast.AST, imports: dict[str, str]) -> dict[str, Any] | None:
    """함수의 ``@njit``/``@jit`` 데코레이터 키워드 옵션(상수 값만). 그런 데코레이터가 없으면 None."""
    for dec in getattr(fn, "decorator_list", []):
        call = dec if isinstance(dec, ast.Call) else None
        parts = _dotted(call.func if call else dec)
        if not parts or parts[-1] not in ("njit", "jit") or not imports.get(parts[0], "").startswith("numba"):
            continue
        return {k.arg: k.value.value for k in (call.keywords if call else []) if k.arg and isinstance(k.value, ast.Constant)}
    return None


def _analyze_python(
    language: str, backend: str, code: str, task: str, spec: TaskSpec | None, inspect: dict | None, build_extra: dict | None
) -> _Facts | dict[str, Any]:
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError) as e:
        return _unknown(language, f"source does not parse: {type(e).__name__}: {e}")
    facts = _Facts(language)
    mod = _PyModule(tree, spec)
    loops = mod.loops()
    facts.nests = _nests(loops)
    element = [i for i, lp in enumerate(loops) if mod.is_element_loop(lp)]

    facts.library_calls = [name for _, name in sorted(mod.calls)]
    facts.temporaries = mod.temporaries
    facts.transposed = mod.transposed
    if mod.delegated:
        facts.by_library.add("library_delegation")
    if mod.allocates or mod.uses_out or mod.transposed:
        facts.by_llm.add("memory_layout")
    if mod.sort_kind:
        facts.by_llm.add("algorithm_choice")
    if mod.float64 and _is_float_task(spec):
        facts.by_llm.add("accumulation_precision")
    _loop_decisions(facts, loops, _py_consts(tree), lambda i: _py_fragments(loops[i]), _PY_CLAMP_RE)
    if element:
        facts.notes.append(f"element loops: {_listing((loops[i].var for i in element), 6)}")
    if mod.uses_out:
        facts.notes.append("out= used")

    if mod.thread_sites:
        facts.par_kind, facts.par_regions, facts.par_placement = "threads", len(mod.thread_sites), "region"
        facts.par_constructs = mod.thread_sites
        facts.by_llm.add("parallel_region")
    if backend == "numba":
        _numba_facts(facts, mod, loops, task, spec, inspect, build_extra)
    elif element and not mod.delegated:
        facts.vec_status, facts.vec_source = "no", "interpreter"     # CPython 이 원소를 하나씩 돈다
    return facts


def _prange_reduces(loop: ast.For) -> bool:
    """루프 밖에서 온 스칼라를 본문에서 ``+=`` 류로 누적하는가(본문 안에서 새로 대입되는 이름은 반복마다의 지역 변수다)."""
    body = [n for stmt in loop.body for n in ast.walk(stmt)]
    accumulated = [n.target for n in body if isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)]
    local = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store) and all(n is not t for t in accumulated)}
    return any(t.id not in local for t in accumulated)


def _inspect_note(inspect: dict | None) -> str | None:
    """워커 ``inspect`` 의 수집 실패(``inspect_error``)를 notes 한 줄로. 실패가 없으면 None.

    수집이 끝나기 전에 import 워커가 죽으면 ``inspect`` 는 ``{"inspect_error": …}`` 뿐이다(관찰값 없음). 도중에 실패하면 그때까지
    모은 값이 함께 있다 — 있는 값은 그대로 쓰고 없는 값은 "보지 못함"으로 다룬다."""
    error = inspect.get("inspect_error") if isinstance(inspect, dict) else None
    if not error:
        return None
    partial = any(key != "inspect_error" for key in inspect)
    return _clip(f"inspect {'incomplete' if partial else 'not available'}: {' '.join(str(error).split())}", 200)


def _numba_facts(
    facts: _Facts, mod: _PyModule, loops: list[_Loop], task: str, spec: TaskSpec | None, inspect: dict | None, build_extra: dict | None
) -> None:
    """Numba: 데코레이터 옵션, ``prange``(``parallel=True`` 일 때만 병렬), 병렬 진단."""
    options = {fn: opts for fn in mod.functions if (opts := _numba_options(fn, mod.imports)) is not None}
    kernel = spec.function if spec is not None else task
    observed = (build_extra or {}).get("options")
    if kernel and isinstance(observed, dict):     # 빌드 단계가 과제 함수의 데코레이터에서 읽은 값(v3 §6)
        for fn in options:
            if getattr(fn, "name", None) == kernel:
                options[fn] = {k: v for k, v in observed.items() if v is not None}
    for fn, opts in list(options.items())[:3]:
        shown = ", ".join(f"{k}={opts[k]}" for k in _NUMBA_OPTIONS if k in opts)
        facts.notes.append(f"numba {getattr(fn, 'name', '?')}: {shown or 'default options'}")

    parallel_on = any(opts.get("parallel") is True for opts in options.values())
    pranges = [i for i, lp in enumerate(loops) if lp.prange]
    active = [i for i in pranges if options.get(loops[i].owner, {}).get("parallel") is True]
    if len(active) < len(pranges):
        facts.notes.append("prange without parallel=True runs serially")
    # Numba 는 가장 바깥 prange 만 병렬로 돌리고 그 안의 prange 는 직렬 range 로 바꾼다
    outer = [i for i in active if not any(j in active for j in _chain(loops, i)[:-1])]
    if len(outer) < len(active):
        facts.notes.append(f"nested prange serialized ({len(active) - len(outer)} inner)")
    if pranges:
        facts.par_constructs.append("prange")
    inspect_note = _inspect_note(inspect)
    if inspect_note:
        facts.notes.append(inspect_note)

    diag = str((inspect or {}).get("parallel_diagnostics") or "")
    from_prange = from_arrays = 0
    for line in diag.splitlines():
        m = _NUMBA_LISTING_RE.fullmatch(line.rstrip())
        if m is not None:
            count = len(m.group("ids").split(","))
            if "prange" in m.group("src"):
                from_prange += count
            else:
                from_arrays += count
    if diag.strip():
        fused = len(re.findall(r"fusion succeeded", diag))
        facts.notes.append(f"numba diagnostics: {from_prange} parallel loop(s) from prange, {from_arrays} from array expressions, {fused} fused")
    layer = (inspect or {}).get("threading_layer")
    if layer and (outer or from_arrays):     # 직렬 커널에도 로드된 커널이면 값이 있다: 병렬 구문이 있을 때만 의미가 있다
        facts.notes.append(f"threading_layer={layer}")

    if outer or from_arrays:
        facts.par_kind, facts.par_regions = "numba", len(outer) + from_arrays
        facts.par_loops = [loops[i].var for i in outer]
        facts.par_placement = _placement([len(_chain(loops, i)) + loops[i].wrapped for i in outer], from_arrays)
        if any(loops[i].wrapped for i in outer):
            facts.notes.append("parallel loop inside a while loop (runs once per iteration)")
        if outer:
            facts.by_llm.add("parallel_region")
        if from_arrays:
            facts.by_compiler.add("parallel_region")       # Numba 가 배열 식을 스스로 병렬 루프로 바꿨다
        if any(_prange_reduces(loops[i].node) for i in outer):
            facts.par_reduction = True
            facts.by_compiler.add("reduction")             # prange 안의 스칼라 누적은 Numba 가 리덕션으로 추론한다
    elif parallel_on and not diag.strip():
        facts.parallel_unknown()      # parallel=True 인데 prange 가 없다: 배열 식의 자동 병렬화 여부는 진단 없이는 알 수 없다


# ----------------------------------------------------------------------------- Halide


_HL_LOOP_RE = re.compile(r"(?P<kind>for|parallel|vectorized|unrolled|gpu_block|gpu_thread|gpu_lane|extern)\s+(?P<var>[^\s:<]+)"
                         r"(?:\s+in\s+(?P<range>\[[^\]]*\]))?.*:")
_HL_BLOCK_RE = re.compile(r"(?P<kind>produce|consume|store)\s+(?P<name>\S+):")
_HL_PROVIDE_RE = re.compile(r"\S+\(\.\.\.\)\s*=\s*\.\.\.")
_HL_CXX_CALL_RE = re.compile(r"\.\s*(\w+)\s*\(((?:[^()]|\([^()]*\))*)\)")
_HL_DECISIONS: dict[str, str] = {   # 스케줄 호출 → 그것이 내리는 결정. split·compute_root 는 문맥을 봐야 해서 따로 판정한다
    "reorder": "loop_order", "tile": "tiling", "vectorize": "vectorization", "parallel": "parallel_region",
    "rfactor": "reduction", "atomic": "reduction",
    **dict.fromkeys(("compute_at", "store_at", "store_root", "store_in", "fold_storage", "reorder_storage", "align_storage",
                     "bound_storage", "hoist_storage", "hoist_storage_root"), "memory_layout"),
}


@functools.lru_cache(maxsize=1)
def _schedule_methods() -> frozenset[str]:
    """Halide 스케줄 호출 이름 집합(v3 §6: ``harness.build.halide.SCHEDULE_METHODS``). 임포트할 수 없으면 같은 내용의 기본값."""
    try:
        from harness.build.halide import SCHEDULE_METHODS
    except Exception:  # noqa: BLE001 - 임포트 사슬(procutil 등)이 이 플랫폼에서 깨져도 요약은 계속한다
        return DEFAULT_SCHEDULE_METHODS
    return frozenset(SCHEDULE_METHODS)


def _split_commas(text: str) -> list[str]:
    """괄호 깊이 0 의 쉼표에서 나눈다."""
    out: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        depth += ch in "([{"
        depth -= ch in ")]}"
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur).strip())
    return [part for part in out if part]


@dataclass(frozen=True)
class _HlCall:
    stage: str               # 같은 스테이지(Func·update)의 호출을 묶는 열쇠
    name: str
    args: tuple[str, ...]

    @property
    def text(self) -> str:
        return _clip(f"{self.name}({', '.join(self.args)})")


def _halide_source_calls(tree: ast.Module, methods: frozenset[str]) -> list[_HlCall]:
    """LLM 소스의 스케줄 호출(소스 순서). 모듈 함수(``np.split``)와 문자열 메서드는 스케줄 호출이 아니다."""
    imports = _py_imports(tree)
    found: list[tuple[tuple[int, int], _HlCall]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in methods):
            continue
        receiver = node.func.value
        if isinstance(receiver, ast.Constant) or (isinstance(receiver, ast.Name) and receiver.id in imports):
            continue
        while isinstance(receiver, ast.Call) and isinstance(receiver.func, ast.Attribute) and receiver.func.attr in methods:
            receiver = receiver.func.value
        args = [ast.unparse(a) for a in node.args] + [f"{k.arg}={ast.unparse(k.value)}" for k in node.keywords if k.arg]
        found.append(((node.func.end_lineno or 0, node.func.end_col_offset or 0), _HlCall(ast.unparse(receiver), node.func.attr, tuple(args))))
    return [call for _, call in sorted(found, key=lambda item: item[0])]


def _halide_cxx_calls(schedule_source: str, methods: frozenset[str]) -> list[_HlCall]:
    """오토스케줄러가 낸 C++ 풍 스케줄 텍스트의 스케줄 호출. 문장(``;``)마다 스테이지 하나다."""
    out: list[_HlCall] = []
    for statement in schedule_source.split(";"):
        calls = [(m.start(), m.group(1), m.group(2)) for m in _HL_CXX_CALL_RE.finditer(statement)]
        scheduled = [c for c in calls if c[1] in methods]
        if not scheduled:
            continue
        stage = " ".join(statement[: scheduled[0][0]].split())
        out.extend(_HlCall(stage, name, tuple(_split_commas(args))) for _, name, args in scheduled)
    return out


def _halide_tiling(calls: list[_HlCall], consts: dict[str, int]) -> tuple[bool, list[int]]:
    """``tile`` 과, 안쪽 변수가 같은 스테이지에서 벡터화·언롤되지 않는 ``split`` 이 타일링이다(벡터 폭으로 자르는 split 은 아니다)."""
    consumed = {(c.stage, c.args[0]) for c in calls if c.name in ("vectorize", "unroll") and c.args}
    detected = False
    sizes: list[int] = []
    for c in calls:
        if c.name == "tile":
            lists = [a for a in c.args if a.startswith("[")]
            factors = _split_commas(lists[-1][1:-1]) if lists else list(c.args[6:8] if len(c.args) >= 8 else c.args[4:6])
        elif c.name == "split" and len(c.args) >= 4 and (c.stage, c.args[2]) not in consumed:
            factors = [c.args[3]]
        else:
            continue
        detected = True
        sizes += [v for v in (_const_int(a, consts) for a in factors) if v is not None]
    return detected, sizes


@dataclass(frozen=True)
class _HlLoop:
    number: int   # 텍스트 안 줄 번호(루프를 구별하는 열쇠)
    kind: str     # for | parallel | vectorized | unrolled | …
    var: str
    range: str    # "[0, 255]" 또는 ""(상수가 아님)


@dataclass
class _HlNest:
    """``print_loop_nest()`` 텍스트에서 읽은 것."""

    nests: list[dict[str, Any]] = field(default_factory=list)
    vectorized: list[bool] = field(default_factory=list)            # nests 와 같은 순서: 그 경로에 vectorized 루프가 있는가
    parallel: list[tuple[str, int]] = field(default_factory=list)   # (변수, 중첩 단계)
    vector_vars: list[str] = field(default_factory=list)
    produced: list[str] = field(default_factory=list)               # produce 된(따로 계산·저장되는) Func 이름


def _halide_loop_nest(text: str) -> _HlNest:
    """들여쓰기로 중첩을 읽는다. 대입 줄(``F(...) = ...``)을 직접 감싸는 루프마다 바깥→안 경로 하나."""
    out = _HlNest()
    stack: list[tuple[int, _HlLoop | None]] = []   # (들여쓰기, 루프 줄이면 그 루프 / produce·consume 블록이면 None)
    emitted: set[int] = set()
    for number, raw in enumerate(text.splitlines()):
        line = raw.strip()
        if not line:
            continue
        indent = len(raw) - len(raw.lstrip())
        while stack and stack[-1][0] >= indent:
            stack.pop()
        block = _HL_BLOCK_RE.fullmatch(line)
        if block is not None:
            if block.group("kind") == "produce" and block.group("name") not in out.produced:
                out.produced.append(block.group("name"))
            stack.append((indent, None))
            continue
        m = _HL_LOOP_RE.fullmatch(line)
        if m is not None:
            loop = _HlLoop(number, m.group("kind"), m.group("var"), m.group("range") or "")
            stack.append((indent, loop))
            if loop.kind == "parallel":
                out.parallel.append((loop.var, sum(1 for _, lp in stack if lp is not None)))
            elif loop.kind == "vectorized":
                out.vector_vars.append(loop.var)
            continue
        chain = [lp for _, lp in stack if lp is not None]
        if _HL_PROVIDE_RE.fullmatch(line) and chain and chain[-1].number not in emitted:
            emitted.add(chain[-1].number)
            out.nests.append({"depth": len(chain), "order": [lp.var for lp in chain], "bounds": [lp.range for lp in chain]})
            out.vectorized.append(any(lp.kind == "vectorized" for lp in chain))
    return out


def _analyze_halide(language: str, code: str, inspect: dict | None, build_extra: dict | None) -> _Facts | dict[str, Any]:
    try:
        tree = _parse(code)
    except (SyntaxError, ValueError) as e:
        return _unknown(language, f"source does not parse: {type(e).__name__}: {e}")
    facts = _Facts(language)
    methods = _schedule_methods()
    info = inspect or {}
    autoscheduler = str(info.get("autoscheduler") or "")
    auto = bool(autoscheduler) or language == "halide_auto"
    consts = _py_consts(tree)

    llm_calls = _halide_source_calls(tree, methods)
    names = (build_extra or {}).get("schedule_calls")
    llm_names = [str(n) for n in names] if isinstance(names, list) else [c.name for c in llm_calls]
    auto_calls = _halide_cxx_calls(str(info.get("schedule_source") or ""), methods) if auto else []

    nest: _HlNest | None = _halide_loop_nest(str(info.get("loop_nest") or ""))
    if not nest.produced:       # produce 블록이 하나도 없으면 루프 중첩 텍스트가 아니다(없거나 깨진 출력)
        nest = None
    produced = len(nest.produced) if nest is not None else None

    for actor, call_names, calls in ((facts.by_llm, llm_names, llm_calls), (facts.by_compiler, [c.name for c in auto_calls], auto_calls)):
        actor.update(_HL_DECISIONS[n] for n in call_names if n in _HL_DECISIONS)
        if _halide_tiling(calls, consts)[0]:
            actor.add("tiling")
        if "compute_root" in call_names and produced is not None and produced > 1:
            actor.add("memory_layout")       # 출력이 아닌 Func 을 따로 계산해 저장한다

    all_calls = llm_calls + auto_calls
    facts.tiling, facts.tile_sizes = _halide_tiling(all_calls, consts)
    facts.transposed = any(c.name == "reorder_storage" for c in all_calls)
    facts.par_constructs = [c.text for c in all_calls if c.name == "parallel"]
    if llm_calls:
        facts.notes.append(_clip("schedule in source: " + ", ".join(_uniq(c.text for c in llm_calls)), 200))
    if auto_calls:
        facts.notes.append(_clip(f"schedule by {autoscheduler or 'autoscheduler'}: " + ", ".join(_uniq(c.text for c in auto_calls)), 260))

    if nest is not None:
        facts.nests = nest.nests
        facts.temporaries = max(len(nest.produced) - 1, 0)
        if nest.parallel:
            facts.par_kind, facts.par_regions = "halide", len(nest.parallel)
            facts.par_loops = [var for var, _ in nest.parallel]
            facts.par_placement = _placement([level for _, level in nest.parallel], len(nest.parallel))
        facts.vec_source = "loop_nest"
        facts.vec_loops = nest.vector_vars
        if nest.vectorized:
            facts.vec_status = "yes" if all(nest.vectorized) else "partial" if any(nest.vectorized) else "no"
    else:
        # 루프 중첩을 못 봤다: 구조는 모른다. 수동 스케줄이면 소스의 parallel 호출만 확실하다
        facts.temporaries = None
        facts.notes.append("loop nest not available")
        parallel_calls = [c for c in llm_calls if c.name == "parallel"]
        if auto:
            facts.parallel_unknown()
            facts.tiling, facts.transposed = None, None
        elif parallel_calls:
            facts.par_kind, facts.par_regions, facts.par_placement = "halide", len(parallel_calls), "unknown"
            facts.par_loops = [c.args[0] for c in parallel_calls if c.args]
    if auto and not auto_calls:
        facts.notes.append("autoscheduler schedule not available")
    inspect_note = _inspect_note(inspect)
    if inspect_note:
        facts.notes.append(inspect_note)
    return facts


# ----------------------------------------------------------------------------- 공개 API


def summarize(
    *,
    language: str,
    backend: str,
    code: str,
    build_log: str = "",
    task: str = "",
    spec: TaskSpec | None = None,
    inspect: dict | None = None,
    build_extra: dict | None = None,
) -> dict[str, Any]:
    """생성 커널 하나의 성능 결정 요약(모듈 docstring 의 스키마). 예외를 올리지 않는다.

    - ``build_log``: C 의 컴파일러 출력(gcc ``-fopt-info-vec`` 줄이 들어 있다).
    - ``task``·``spec``: 과제 이름과 명세. Python 계열에서 배열 인자와 그 차원 수, 누적 정밀도 판단(float32 출력 과제),
      Numba 의 과제 함수 이름(``spec`` 이 없으면 ``task``)에 쓴다. 없어도 된다.
    - ``inspect``: 워커 import 문서의 ``inspect`` (v3 §5.1) — numba ``parallel_diagnostics``, halide ``loop_nest``·``schedule_source``.
    - ``build_extra``: ``BuildResult.extra`` (v3 §6) — numba ``options``, halide ``schedule_calls``. 없으면 소스에서 직접 읽는다.
    """
    try:
        if backend == "c":
            result: _Facts | dict[str, Any] = _analyze_c(language, code, build_log, spec)
        elif backend == "rust":
            result = _analyze_rust(language, code, spec)
        elif backend in ("python", "numba"):
            result = _analyze_python(language, backend, code, task, spec, inspect, build_extra)
        elif backend == "halide":
            result = _analyze_halide(language, code, inspect, build_extra)
        else:
            return _unknown(language, f"unsupported backend: {backend}")
        return result.to_dict() if isinstance(result, _Facts) else result
    except Exception as e:  # noqa: BLE001 - 계약(v3 §10.2): 요약 실패가 결과 행을 막지 않도록 실패를 notes 로 돌려준다
        return _unknown(language, f"{ERROR_NOTE_PREFIX} {type(e).__name__}: {e}")


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """소스 파일 하나의 요약을 JSON 으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.perf_summary", description="성능 결정 요약 (설계 v3 §11)")
    parser.add_argument("--language", required=True, help="언어 이름(예: c_openmp, numpy, halide_auto)")
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--file", type=Path, required=True, help="커널 소스(kernel.c, kernel.py, src/lib.rs)")
    parser.add_argument("--build-log", type=Path, default=None, help="컴파일러 출력(build.log)")
    parser.add_argument("--inspect", type=Path, default=None, help="워커 import 결과 문서(import.json; 그 안의 inspect 를 쓴다)")
    ns = parser.parse_args(argv)
    try:
        code = ns.file.read_text(encoding="utf-8", errors="replace")
        build_log = ns.build_log.read_text(encoding="utf-8", errors="replace") if ns.build_log else ""
        inspect = json.loads(ns.inspect.read_text(encoding="utf-8")).get("inspect") if ns.inspect else None
    except (OSError, ValueError, AttributeError) as e:
        print(f"{_LOG} 입력을 읽을 수 없다: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    summary = summarize(language=ns.language, backend=ns.backend, code=code, build_log=build_log, inspect=inspect)
    # 키 하나에 한 줄: 유효한 JSON 이면서 한눈에 읽힌다
    print("{\n" + ",\n".join(f"  {json.dumps(k)}: {json.dumps(v, ensure_ascii=False)}" for k, v in summary.items()) + "\n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
