"""Numba "빌드" (설계 v3 §6): ``kernel.py`` 저장 + ``compile()`` 구문 검사 + 데코레이터의 AST 검사.

Numba 조건은 **명시 서명·즉시 컴파일**을 요구한다(``@njit("<서명>")``; decisions 2026-11-20). 서명이 없으면 첫 호출에 JIT 가
섞여 측정이 오염되므로 빌드 단계에서 ``signature_mismatch``/``numba_signature_missing`` 으로 막는다(워커도 로드 뒤
``signatures`` 가 비어 있는지 본다 — 이중 검사). 여기서는 코드를 임포트·실행하지 않는다(``compile()`` 과 ``ast`` 만).

받아들이는 꼴: 최상위 ``def {function}`` 의 데코레이터가

- ``njit(...)`` — ``njit``, ``numba.njit``, ``nb.njit`` (numba 모듈의 어떤 별칭이든), 또는
- ``jit(..., nopython=True)`` — ``jit``, ``numba.jit``, …

의 **호출**이고 서명을 첫 위치 인자나 키워드(``signature=``, Numba 의 실제 매개변수 이름 ``signature_or_function=``)로 준다.
서명은 문자열, 문자열의 리스트·튜플, Numba 타입 식(``void(float32[::1], int64)``) 어느 것이든 된다. 데코레이터가 없거나,
``@njit`` 만 쓰거나, ``@njit(parallel=True)`` 처럼 서명이 없으면 ``numba_signature_missing`` 이다.

여기서 보는 것은 소스의 꼴이지 Numba 의 동작이 아니다. 실측(Numba 0.68.1): 계약이 받는 ``signature=`` 키워드를 Numba 는 서명으로
쓰지 않고 조용히 지연 컴파일하며, 문자열의 튜플은 임포트 때 TypeError 다. 그런 커널은 여기를 통과하고 워커가 판정한다
(``signatures`` 가 빈 디스패처 → ``numba_signature_missing``, 임포트 예외 → ``compile_error``).

최상위에 ``def {function}`` 자체가 없으면 ``signature_mismatch``/``function_not_found`` 다(추출 단계를 거쳤으면 나오지 않는다).

``BuildResult.extra`` (데코레이터 검사가 실패해도 채운다; 구문 오류면 구문 트리가 없으므로 빈 사전)::

    {"decorator": "njit" | "jit" | None,                   # 과제 함수에 붙은 Numba 데코레이터
     "signature_literal": str | None,                     # 문자열 서명은 그 값, 그 밖의 식은 ast.unparse 한 원문
     "signature_matches_expected": bool | None,           # 기대 서명과 글자가 같은가(공백 무시). 판단할 수 없으면 None
     "options": {"parallel": bool | None, "fastmath": …, "cache": …, "boundscheck": …, "nogil": …},
     "uses_prange": bool}                                 # 모듈 어디서든 numba 의 prange 를 호출하는가

- ``signature_matches_expected``: 문자열 서명(리스트·튜플이면 그중 하나)이 기대 서명과 같으면 True, 다르면 False — 글자가
  달라도 같은 타입일 수 있으므로(``f4[::1]``) 빌드를 실패시키지 않는다(타입이 맞지 않으면 호출이 TypeError 로 드러난다).
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
SIGNATURE_KEYWORDS: tuple[str, ...] = ("signature", "signature_or_function")
SIGNATURE_MISSING = "numba_signature_missing"
FUNCTION_NOT_FOUND = "function_not_found"


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


def inspect_numba(tree: ast.Module, function: str, expected_signature: str) -> tuple[str, str, dict[str, Any]]:
    """과제 함수의 데코레이터를 검사한다 → ``(error_kind, error_detail, extra)`` (모듈 docstring)."""
    imports = imported_names(tree)
    extra: dict[str, Any] = {
        "decorator": None,
        "signature_literal": None,
        "signature_matches_expected": None,
        "options": dict.fromkeys(NUMBA_OPTIONS),
        "uses_prange": any(isinstance(n, ast.Call) and numba_member(n.func, imports) == "prange" for n in ast.walk(tree)),
    }
    fn = top_level_function(tree, function)
    if fn is None:
        return "signature_mismatch", FUNCTION_NOT_FOUND, extra

    accepted = False
    for dec in fn.decorator_list:
        kind = numba_member(dec.func if isinstance(dec, ast.Call) else dec, imports)
        if kind not in ("njit", "jit"):
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
                accepted = kind == "njit" or _literal_true(dec, "nopython")
        break  # 과제 함수의 Numba 데코레이터는 하나다(첫 번째 것이 판정한다)
    if not accepted:
        return "signature_mismatch", SIGNATURE_MISSING, extra
    return "", "", extra


def build_numba(code: str, out_dir: Path, *, function: str, expected_signature: str) -> BuildResult:
    """``out_dir/kernel.py`` 를 쓰고 구문과 과제 함수 ``function`` 의 Numba 데코레이터를 검사한다. ``build.json`` 도 저장.

    ``expected_signature`` 는 ``sigs.numba_signature(spec)`` (``"void(float32[::1], …, int64)"``)다.
    """
    return build_python_source(
        code, out_dir, compiler=COMPILER_NAME, compiler_version=package_version("numba"),
        inspect=lambda tree: inspect_numba(tree, function, expected_signature),
    )


def main(argv: list[str] | None = None) -> int:
    """Numba 커널 소스 파일 하나를 검사하고 BuildResult 를 JSON으로 출력한다(진단용)."""
    parser = argparse.ArgumentParser(prog="python -m harness.build.numba", description="Numba 커널 구문·데코레이터 검사 (설계 v3 §6)")
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
