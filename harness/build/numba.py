"""Numba "빌드" (설계 v3 §7.2): ``kernel.py`` 저장 + ``compile()`` 구문 검사 + 데코레이터의 AST 검사.

Numba 조건은 **명시 서명·즉시 컴파일**을 요구한다(decisions.md 2026-11-12 "M3 사전 검증 결과와 언어별 통합 결정", 통합 규칙은
2026-11-23). 서명이 없는 함수는 첫 호출에 JIT 가 섞여 측정이 오염되므로, 모듈의 **모든** ``njit``/``jit`` 가 서명을 가져야 한다.
워커도 로드 때 디스패처가 하네스의 인자 타입을 컴파일 없이 받는지 본다(이중 검사). 여기서는 코드를 임포트·실행하지 않는다
(``compile()`` 과 ``ast`` 만).

판정(먼저 걸리는 것 하나; 모두 ``signature_mismatch``):

1. 최상위에 ``def {function}`` 이 없다 → ``function_not_found`` (추출 단계를 거쳤으면 나오지 않는다).
2. 과제 함수에 Numba 의 ``njit``/``jit`` 데코레이터가 없다 → ``numba_signature_missing``.
3. 모듈 어디서든(최상위·중첩 함수·클래스 본문) ``njit``/``jit`` 로 컴파일되는 함수가 **명시 서명을 첫 위치 인자로** 받지 않는다 →
   ``numba_unsigned_jit:<함수 이름>`` (소스 순서로 첫 번째). 데코레이터(``@njit``, ``@njit()``, ``@njit(parallel=True)``)와 호출 꼴
   (``njit(f)``, ``njit(parallel=True)(f)``) 모두. ``signature=`` 키워드는 서명으로 치지 않는다 — Numba 0.68 은 이 키워드를 쓰지
   않고 조용히 지연 컴파일한다(실측). Numba 의 실제 첫 매개변수 이름인 ``signature_or_function=`` 은 위치 인자와 같아 받는다.
4. 어느 ``njit``/``jit`` 에든 ``cache=`` 가 리터럴 ``False``/``None`` 이 아니다(``cache=True``, 식) → ``numba_cache_true``.
   캐시는 ``sys.dont_write_bytecode`` 와 무관하게 샘플 디렉터리에 기계어를 써서 다음 라운드·재실행이 컴파일된 코드를 읽는다(실측).
5. 과제 함수의 데코레이터가 ``jit`` 인데 ``nopython=True`` 리터럴이 없다 → ``numba_signature_missing``.

서명은 문자열, 문자열의 리스트·튜플, Numba 타입 식(``void(float32[::1], int64)``) 어느 것이든 된다. 호출 꼴 ``njit(X)`` 에서 ``X`` 가
모듈에 정의된 함수 이름이거나 ``lambda`` 면 서명 없이 그 함수를 컴파일하는 것으로, 그 밖의 식이면 서명으로 본다.

BLAS 를 부르는 호출(``np.dot``·``np.vdot``·``np.matmul``·``np.linalg.*``·``배열.dot(…)``·``@``)은 거절하지 않고 기록한다
(``extra["blas_calls"]`` → 성능 결정 요약). SciPy 가 없는 워커 인터프리터에서 Numba 는 이런 코드를 임포트 때 거부하고
(``compile_error``), run 시작 점검은 SciPy 가 있으면 numba run 을 멈춘다(decisions.md 2026-11-23).

``BuildResult.extra`` (판정이 실패여도 채운다; 구문 오류면 구문 트리가 없으므로 빈 사전)::

    {"decorator": "njit" | "jit" | None,                   # 과제 함수에 붙은 Numba 데코레이터
     "signature_literal": str | None,                     # 문자열 서명은 그 값, 그 밖의 식은 ast.unparse 한 원문
     "signature_matches_expected": bool | None,           # 기대 서명과 글자가 같은가(공백 무시). 판단할 수 없으면 None
     "options": {"parallel": bool | None, "fastmath": …, "cache": …, "boundscheck": …, "nogil": …},
     "uses_prange": bool,                                 # 모듈 어디서든 numba 의 prange 를 호출하는가
     "unsigned_jit_functions": [이름…],                    # 서명 없이 컴파일되는 함수(소스 순서)
     "blas_calls": [{"name": "numpy.dot" | "numpy.linalg.solve" | "ndarray.dot" | "@", "line": int}, …]}  # Numba 함수 본문 안

- ``signature_matches_expected``: 문자열 서명(리스트·튜플이면 그중 하나)이 기대 서명과 같으면 True, 다르면 False — 글자가
  달라도 같은 타입일 수 있으므로(``f4[::1]``) 빌드를 실패시키지 않는다(워커가 하네스의 인자 타입으로 확인한다).
  타입 식은 글자가 같을 때만 True 이고 그 밖에는 None 이다(이름이 무엇에 묶였는지는 실행해야 안다).
- ``options`` 의 값은 소스에 ``True``/``False`` 리터럴로 쓴 것만 싣는다. 쓰지 않았거나 식이면 None.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

from harness.build.c import BuildResult
from harness.build.python import build_python_source, dotted_name, imported_names, package_version, top_level_function

COMPILER_NAME = "numba"
NUMBA_OPTIONS: tuple[str, ...] = ("parallel", "fastmath", "cache", "boundscheck", "nogil")
# 서명으로 치는 키워드: Numba 의 첫 매개변수 이름뿐이다(``signature=`` 는 Numba 0.68 이 쓰지 않는다 — 모듈 docstring)
SIGNATURE_KEYWORDS: tuple[str, ...] = ("signature_or_function",)
JIT_DECORATORS: tuple[str, ...] = ("njit", "jit")
SIGNATURE_MISSING = "numba_signature_missing"
UNSIGNED_JIT = "numba_unsigned_jit"   # 상세는 "numba_unsigned_jit:<함수 이름>"
CACHE_TRUE = "numba_cache_true"
FUNCTION_NOT_FOUND = "function_not_found"
# Numba 가 BLAS(SciPy 의 cython_blas/lapack)로 구현하는 NumPy 함수(전체 이름). ``numpy.linalg.*`` 는 접두로 본다.
BLAS_FUNCTIONS: frozenset[str] = frozenset({"numpy.dot", "numpy.vdot", "numpy.matmul"})
BLAS_PREFIX = "numpy.linalg."

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def _squeeze(text: str) -> str:
    return "".join(text.split())


def numba_member(node: ast.AST, imports: dict[str, str]) -> str | None:
    """식이 numba 패키지의 이름을 가리키면 그 마지막 이름(``nb.njit`` → ``"njit"``), 아니면 None.

    임포트 없이 쓴 맨 이름(``from numba import *`` 뒤의 ``njit``)은 numba 의 것으로 본다.
    """
    parts = dotted_name(node)
    if parts is None:
        return None
    root = imports.get(parts[0], f"numba.{parts[0]}" if len(parts) == 1 else parts[0])
    full = [*root.split("."), *parts[1:]]
    return full[-1] if full[0] == "numba" else None


def _signature_node(call: ast.Call) -> ast.expr | None:
    """데코레이터 호출이 준 서명 식. 없으면(위치 인자·서명 키워드 없음, ``None``·빈 리스트) None."""
    node = call.args[0] if call.args else next((kw.value for kw in call.keywords if kw.arg in SIGNATURE_KEYWORDS), None)
    if node is None:
        return None
    if isinstance(node, ast.Constant) and not isinstance(node.value, str):
        return None
    if isinstance(node, (ast.List, ast.Tuple)) and not node.elts:
        return None
    return node


def _string_values(node: ast.expr) -> list[str] | None:
    """문자열 상수면 ``[값]``, 문자열 상수만 든 리스트·튜플이면 그 값들, 그 밖에는 None."""
    nodes = node.elts if isinstance(node, (ast.List, ast.Tuple)) else [node]
    values = [n.value for n in nodes if isinstance(n, ast.Constant) and isinstance(n.value, str)]
    return values if len(values) == len(nodes) else None


def _literal_true(call: ast.Call, name: str) -> bool:
    return any(kw.arg == name and isinstance(kw.value, ast.Constant) and kw.value.value is True for kw in call.keywords)


def _jit_kind(node: ast.AST, imports: dict[str, str]) -> str | None:
    """``node`` 가 numba 의 ``njit``/``jit`` 이름(``njit``, ``nb.njit``)이면 그 이름, 아니면 None."""
    kind = numba_member(node, imports)
    return kind if kind in JIT_DECORATORS else None


def _cache_enabled(call: ast.Call) -> bool:
    """``cache=`` 키워드가 리터럴 ``False``/``None`` 이 아닌가(식도 켜진 것으로 본다: 확인할 수 없다)."""
    return any(kw.arg == "cache" and not (isinstance(kw.value, ast.Constant) and not kw.value.value) for kw in call.keywords)


class _JitScan:
    """모듈 전체의 ``njit``/``jit`` 사용을 모은다: 서명 없이 컴파일되는 함수, ``cache`` 사용, 컴파일되는 함수 본문."""

    def __init__(self, tree: ast.Module, imports: dict[str, str]) -> None:
        self.imports = imports
        self.unsigned: list[tuple[int, str]] = []     # (줄, 함수 이름)
        self.cache_enabled = False
        self.compiled_names: set[str] = set()         # 호출 꼴로 컴파일되는 함수의 이름
        self.compiled_bodies: list[FunctionNode] = []
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        self.function_names = {f.name for f in functions}
        decorator_calls: set[int] = set()
        for fn in functions:
            for dec in fn.decorator_list:
                call = dec if isinstance(dec, ast.Call) else None
                if _jit_kind(call.func if call is not None else dec, imports) is None:
                    continue
                if call is not None:
                    decorator_calls.add(id(call))
                    self.cache_enabled |= _cache_enabled(call)
                if call is None or _signature_node(call) is None:
                    self.unsigned.append((fn.lineno, fn.name))
                self.compiled_bodies.append(fn)
                break
        outer_funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or id(node) in decorator_calls:
                continue
            inner = node.func if isinstance(node.func, ast.Call) else None
            if inner is not None and _jit_kind(inner.func, imports) is not None:
                # njit(...)(f): 안쪽 호출이 서명을 가져야 한다
                self.cache_enabled |= _cache_enabled(inner)
                target = self._target_name(node.args[0]) if node.args else None
                if _signature_node(inner) is None:
                    self.unsigned.append((node.lineno, target or f"<line {node.lineno}>"))
                if target:
                    self.compiled_names.add(target)
            elif _jit_kind(node.func, imports) is not None and id(node) not in outer_funcs:
                # njit(f) 또는 njit("sig") 단독: 함수를 받으면 서명 없이 컴파일한다(njit("sig") 만으로는 아무것도 컴파일하지 않는다)
                self.cache_enabled |= _cache_enabled(node)
                first = node.args[0] if node.args else None
                target = self._target_name(first) if first is not None else None
                if target is not None:
                    self.unsigned.append((node.lineno, target))
                    self.compiled_names.add(target)
        self.unsigned.sort()
        self.compiled_bodies += [f for f in functions if f.name in self.compiled_names and f not in self.compiled_bodies]

    def _target_name(self, node: ast.expr) -> str | None:
        """호출 꼴의 인자가 컴파일할 함수면 그 이름(``lambda`` 는 ``<lambda>``), 서명으로 볼 식이면 None."""
        if isinstance(node, ast.Lambda):
            return "<lambda>"
        if isinstance(node, ast.Name) and node.id in self.function_names:
            return node.id
        return None


def blas_calls(bodies: list[FunctionNode], imports: dict[str, str]) -> list[dict[str, Any]]:
    """Numba 로 컴파일되는 함수 본문 안의 BLAS 호출(모듈 docstring). 줄 순서, 같은 줄이면 이름순. 거절하지 않고 기록만 한다."""
    found: dict[tuple[int, int], dict[str, Any]] = {}
    for fn in bodies:
        for node in ast.walk(fn):
            name: str | None = None
            if isinstance(node, ast.Call) and isinstance(node.func, (ast.Name, ast.Attribute)):
                parts = dotted_name(node.func)
                if parts is not None:
                    root = imports.get(parts[0], parts[0])
                    full = ".".join([*root.split("."), *parts[1:]])
                    if full in BLAS_FUNCTIONS or full.startswith(BLAS_PREFIX):
                        name = full
                    elif len(parts) > 1 and parts[-1] == "dot" and parts[0] not in imports:
                        name = "ndarray.dot"
                elif isinstance(node.func, ast.Attribute) and node.func.attr == "dot":
                    name = "ndarray.dot"
            elif isinstance(node, (ast.BinOp, ast.AugAssign)) and isinstance(node.op, ast.MatMult):
                name = "@"
            if name is not None:
                found[(node.lineno, node.col_offset)] = {"name": name, "line": node.lineno}
    return [found[key] for key in sorted(found)]


def inspect_numba(tree: ast.Module, function: str, expected_signature: str) -> tuple[str, str, dict[str, Any]]:
    """과제 함수와 모듈 전체의 Numba 데코레이터를 검사한다 → ``(error_kind, error_detail, extra)`` (모듈 docstring)."""
    imports = imported_names(tree)
    scan = _JitScan(tree, imports)
    extra: dict[str, Any] = {
        "decorator": None,
        "signature_literal": None,
        "signature_matches_expected": None,
        "options": dict.fromkeys(NUMBA_OPTIONS),
        "uses_prange": any(isinstance(n, ast.Call) and numba_member(n.func, imports) == "prange" for n in ast.walk(tree)),
        "unsigned_jit_functions": [name for _, name in scan.unsigned],
        "blas_calls": blas_calls(scan.compiled_bodies, imports),
    }
    fn = top_level_function(tree, function)
    if fn is None:
        return "signature_mismatch", FUNCTION_NOT_FOUND, extra

    nopython_ok = False
    for dec in fn.decorator_list:
        kind = _jit_kind(dec.func if isinstance(dec, ast.Call) else dec, imports)
        if kind is None:
            continue
        extra["decorator"] = kind
        if isinstance(dec, ast.Call):
            for kw in dec.keywords:
                if kw.arg in NUMBA_OPTIONS and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                    extra["options"][kw.arg] = kw.value.value
            signature = _signature_node(dec)
            if signature is not None:
                strings = _string_values(signature)
                expected = _squeeze(expected_signature)
                if strings is not None:
                    extra["signature_literal"] = signature.value if isinstance(signature, ast.Constant) else ast.unparse(signature)
                    extra["signature_matches_expected"] = any(_squeeze(s) == expected for s in strings)
                else:
                    extra["signature_literal"] = ast.unparse(signature)
                    extra["signature_matches_expected"] = True if _squeeze(extra["signature_literal"]) == expected else None
            nopython_ok = kind == "njit" or _literal_true(dec, "nopython")
        break  # 과제 함수의 Numba 데코레이터는 하나다(첫 번째 것이 판정한다)
    if extra["decorator"] is None:
        return "signature_mismatch", SIGNATURE_MISSING, extra
    if scan.unsigned:
        return "signature_mismatch", f"{UNSIGNED_JIT}:{scan.unsigned[0][1]}", extra
    if scan.cache_enabled:
        return "signature_mismatch", CACHE_TRUE, extra
    if not nopython_ok:
        return "signature_mismatch", SIGNATURE_MISSING, extra
    return "", "", extra


def build_numba(code: str, out_dir: Path, *, function: str, expected_signature: str) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 구문과 Numba 데코레이터(과제 함수 ``function`` 과 모듈 전체)를 검사한다. ``build.json`` 도 저장.

    ``expected_signature`` 는 ``sigs.numba_signature(spec)`` (``"void(float32[::1], …, int64)"``)다.
    """
    return build_python_source(
        code, out_dir, compiler=COMPILER_NAME, compiler_version=package_version("numba"),
        inspect=lambda tree: inspect_numba(tree, function, expected_signature),
    )


def main(argv: list[str] | None = None) -> int:
    """Numba 커널 소스 파일 하나를 검사하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.numba", description="Numba 커널 구문·데코레이터 검사 (설계 v3 §7.2)")
    parser.add_argument("--src", type=Path, required=True, help="Python 소스 파일")
    parser.add_argument("--out-dir", type=Path, required=True, help="출력 디렉터리(kernel.py/build.json)")
    parser.add_argument("--function", required=True, help="과제 함수 이름")
    parser.add_argument("--expected-signature", required=True, help='기대 서명(예: "void(float32[::1], int64)")')
    ns = parser.parse_args(argv)
    res = build_numba(ns.src.read_text(encoding="utf-8"), ns.out_dir, function=ns.function, expected_signature=ns.expected_signature)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
    return 0 if res.ok else 1


if __name__ == "__main__":
    # 컴파일러 출력·경로·한국어 진단을 찍다가 로캘 코드페이지에서 죽지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
