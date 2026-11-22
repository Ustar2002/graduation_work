"""build/numba.py 테스트 (설계 v3 §7.2): 구문 검사, 데코레이터의 AST 검사(받는 꼴·거절하는 꼴), ``BuildResult.extra``.

규칙(decisions.md 2026-11-12·2026-11-23): 모듈의 모든 ``njit``/``jit`` 가 명시 서명을 첫 위치 인자로 받는다(아니면
``numba_unsigned_jit:<이름>``; ``signature=`` 키워드는 치지 않는다), ``cache=True`` 는 어디서든 ``numba_cache_true``, BLAS 호출은
기록만 한다(``blas_calls``).

빌드 단계는 코드를 임포트·실행하지 않으므로 Numba 가 없어도 돈다. ``RealNumbaTest`` 만 Numba 를 **서브프로세스**에서 실제로 돌려,
여기서 받는 꼴이 정말 임포트 때 컴파일되는지(= 디스패처의 ``signatures`` 가 채워지는지) 확인한다 — 빌드 단계의 규칙이 막으려는
것이 "첫 호출에 섞이는 JIT" 이기 때문이다.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import io
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from harness import sigs
from harness.build.numba import NUMBA_OPTIONS, build_numba, main
from harness.paths import ROOT
from harness.taskspec import load_taskspec
from tests._support import requires_modules

SIG = "void(float32[::1], float32[::1], float32[::1], int64)"  # dot 의 기대 서명
OTHER_SIG = "void(float64[::1], float64[::1], float64[::1], int64)"
NO_OPTIONS: dict[str, Any] = {"parallel": None, "fastmath": None, "cache": None, "boundscheck": None, "nogil": None}
EXTRA_KEYS = {"decorator", "signature_literal", "signature_matches_expected", "options", "uses_prange", "unsigned_jit_functions",
              "blas_calls"}


def expected_signature(task: str) -> str:
    """run.py 가 넘기는 것과 같은 기대 서명(``bundle["numba_signature"]``)."""
    return sigs.numba_signature(load_taskspec(ROOT / "tasks" / task / "signatures.yaml"))


class BuildDirTest(unittest.TestCase):
    """임시 출력 디렉터리에 빌드하는 테스트의 공통 부분(빌드마다 새 ``sample_<n>`` 디렉터리)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_numba_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp

    def build(self, code: str, function: str = "dot", expected: str = SIG) -> Any:
        self.out = self.tmp / f"sample_{len(list(self.tmp.iterdir()))}"
        return build_numba(code, self.out, function=function, expected_signature=expected)


# ----------------------------------------------------------------------------- 실제 커널 꼴

NUMBA_GEMM = '''import numpy as np
from numba import njit, prange


@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)", parallel=True, fastmath=True)
def gemm(A, B, C, M, N, K):
    for i in prange(M):
        for j in range(N):
            C[i, j] = 0.0
        for k in range(K):
            a = A[i, k]
            for j in range(N):
                C[i, j] += a * B[k, j]
'''

NUMBA_DOT = '''import numba as nb
import numpy as np


@nb.njit("void(float32[::1], float32[::1], float32[::1], int64)", cache=False, nogil=True)
def dot(x, y, out, n):
    acc = 0.0
    for i in range(n):
        acc += np.float64(x[i]) * np.float64(y[i])
    out[0] = acc
'''

# 서명을 Numba 타입 식으로 쓰고, 도움 함수에도 서명 문자열을 준 꼴(모든 njit 가 서명을 가져야 한다)
NUMBA_JACOBI2D = '''import numpy as np
from numba import float32, int64, njit, prange, void


@njit("void(float32[:, ::1], float32[:, ::1], int64, int64)", fastmath=True, inline="always")
def _sweep(a, b, H, W):
    for y in prange(1, H - 1):
        for x in range(1, W - 1):
            b[y, x] = 0.25 * (a[y - 1, x] + a[y + 1, x] + a[y, x - 1] + a[y, x + 1])


@njit(void(float32[:, ::1], float32[:, ::1], int64, int64, int64), parallel=True, boundscheck=False)
def jacobi2d(src, dst, H, W, T):
    tmp = src.copy()
    dst[:, :] = src
    a, b = tmp, dst
    for _ in range(T):
        _sweep(a, b, H, W)
        a, b = b, a
    if T % 2 == 0:
        dst[:, :] = tmp
'''

NUMBA_SPMV = '''import numba


@numba.jit("void(int32[::1], int32[::1], float32[::1], float32[::1], float32[::1], int64, int64, int64)",
           nopython=True, parallel=True)
def spmv_csr(row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz):
    for i in numba.prange(n_rows):
        acc = 0.0
        for p in range(row_ptr[i], row_ptr[i + 1]):
            acc += vals[p] * x[col_idx[p]]
        y[i] = acc
'''


class RealisticKernelsTest(BuildDirTest):
    def test_expected_signatures_are_the_rendered_ones(self) -> None:
        self.assertEqual(expected_signature("gemm"), "void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)")
        self.assertEqual(expected_signature("dot"), SIG)

    def test_kernels_pass_with_the_observed_facts(self) -> None:
        cases = {
            "gemm": (NUMBA_GEMM, {
                "decorator": "njit", "signature_literal": expected_signature("gemm"), "signature_matches_expected": True,
                "options": {**NO_OPTIONS, "parallel": True, "fastmath": True}, "uses_prange": True}),
            "dot": (NUMBA_DOT, {
                "decorator": "njit", "signature_literal": expected_signature("dot"), "signature_matches_expected": True,
                "options": {**NO_OPTIONS, "cache": False, "nogil": True}, "uses_prange": False}),
            "jacobi2d": (NUMBA_JACOBI2D, {
                "decorator": "njit", "signature_literal": expected_signature("jacobi2d"), "signature_matches_expected": True,
                "options": {**NO_OPTIONS, "parallel": True, "boundscheck": False}, "uses_prange": True}),
            "spmv_csr": (NUMBA_SPMV, {
                "decorator": "jit", "signature_literal": expected_signature("spmv_csr"), "signature_matches_expected": True,
                "options": {**NO_OPTIONS, "parallel": True}, "uses_prange": True}),
        }
        for task, (code, extra) in cases.items():
            with self.subTest(task=task):
                res = self.build(code, function=task, expected=expected_signature(task))
                self.assertEqual((res.ok, res.error_kind, res.error_detail), (True, "", ""))
                self.assertEqual(res.extra, {**extra, "unsigned_jit_functions": [], "blas_calls": []})

    def test_files_and_result_fields(self) -> None:
        res = self.build(NUMBA_GEMM, function="gemm", expected=expected_signature("gemm"))
        self.assertEqual(res.artifact, self.out / "kernel.py")
        self.assertEqual({p.name for p in self.out.iterdir()}, {"kernel.py", "build.json"})
        self.assertEqual((self.out / "kernel.py").read_text(encoding="utf-8"), NUMBA_GEMM)
        self.assertEqual((res.compiler, res.command, res.n_warnings, res.log), ("numba", ["python", "compile", "kernel.py"], 0, ""))
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, json.loads(json.dumps(res.to_dict())))
        self.assertEqual(set(saved["extra"]), EXTRA_KEYS)
        self.assertEqual(tuple(saved["extra"]["options"]), NUMBA_OPTIONS)

    @requires_modules("numba")
    def test_compiler_version_is_the_installed_numba(self) -> None:
        self.assertEqual(self.build(NUMBA_DOT).compiler_version, importlib.metadata.version("numba"))

    def test_the_kernel_is_not_executed(self) -> None:
        marker = self.tmp / "executed.txt"
        code = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('x')\n" + NUMBA_DOT
        self.assertTrue(self.build(code).ok)
        self.assertFalse(marker.exists(), "빌드 단계는 커널을 임포트·실행하지 않는다")


# ----------------------------------------------------------------------------- 데코레이터의 꼴

IMPORTS = "import functools\n\nimport numba\nimport numba as nb\nfrom numba import float32, int64, jit, njit, prange, void"
BODY = "def dot(x, y, out, n):\n    acc = 0.0\n    for i in range(n):\n        acc += x[i] * y[i]\n    out[0] = acc\n"


def with_helpers(helpers: str, decorator: str = f'@njit("{SIG}")', imports: str = IMPORTS, body: str = BODY) -> str:
    """``helpers`` (도움 함수들의 소스) 뒤에 ``decorator`` 를 붙인 과제 함수 ``body`` 를 둔 모듈."""
    return f"{imports}\n\n\n{helpers}\n\n{decorator}\n{body}"


def module(decorator: str, imports: str = IMPORTS, body: str = BODY) -> str:
    """``decorator`` (빈 문자열이면 데코레이터 없음)를 ``body`` 의 첫 정의에 붙인 모듈."""
    return f"{imports}\n\n\n{decorator}\n{body}" if decorator else f"{imports}\n\n\n{body}"


class AcceptedFormsTest(BuildDirTest):
    def check(self, cases: dict[str, tuple[str, str, bool | None]], imports: str = IMPORTS) -> None:
        for decorator, (kind, literal, matches) in cases.items():
            with self.subTest(decorator=decorator):
                res = self.build(module(decorator, imports))
                self.assertEqual((res.ok, res.error_kind, res.error_detail), (True, "", ""))
                self.assertEqual((res.extra["decorator"], res.extra["signature_literal"], res.extra["signature_matches_expected"]),
                                 (kind, literal, matches))

    def test_njit_and_jit_nopython_with_a_string_signature(self) -> None:
        self.check({
            f'@njit("{SIG}")': ("njit", SIG, True),
            f'@numba.njit("{SIG}")': ("njit", SIG, True),
            f'@nb.njit("{SIG}", parallel=True)': ("njit", SIG, True),
            f'@jit("{SIG}", nopython=True)': ("jit", SIG, True),
            f'@numba.jit("{SIG}", nopython=True, cache=False)': ("jit", SIG, True),
            # 다른 데코레이터와 같이 써도 Numba 데코레이터가 판정한다
            f'@functools.wraps(print)\n@njit("{SIG}")': ("njit", SIG, True),
        })

    def test_signature_keywords(self) -> None:
        # Numba 의 실제 첫 매개변수 이름은 위치 인자와 같다(임포트 때 컴파일된다). `signature=` 는 RejectedFormsTest
        self.check({
            f'@njit(signature_or_function="{SIG}")': ("njit", SIG, True),
            f'@njit(fastmath=True, signature_or_function="{SIG}")': ("njit", SIG, True),
        })

    def test_signed_helpers_in_every_form(self) -> None:
        helpers = (
            f'@njit("{SIG}")\ndef _a(x, y, out, n):\n    out[0] = 0.0\n\n\n'
            f'@nb.njit(["{SIG}"], parallel=True)\ndef _b(x, y, out, n):\n    out[0] = 0.0\n\n\n'
            f'def _c(x, y, out, n):\n    out[0] = 0.0\n\n\n_c_fast = njit("{SIG}")(_c)\n'
            f'_d = numba.jit("{SIG}", nopython=True)(lambda x: x)\n'
            f'_e = njit("{SIG}")   # 데코레이터 공장만 만들고 아무것도 컴파일하지 않는다\n\n\n'
        )
        res = self.build(with_helpers(helpers))
        self.assertEqual((res.ok, res.error_detail, res.extra["unsigned_jit_functions"]), (True, "", []))

    def test_any_alias_of_the_numba_module(self) -> None:
        cases = {
            "import numba as whatever": f'@whatever.njit("{SIG}")',
            "from numba import njit as fast": f'@fast("{SIG}")',
            "from numba.core.decorators import njit": f'@njit("{SIG}")',
            "import numba.core.decorators as deco": f'@deco.njit("{SIG}")',
            "from numba import *": f'@njit("{SIG}")',                        # 묶인 곳을 알 수 없는 맨 이름은 Numba 것으로 본다
            "try:\n    from numba import njit\nexcept ImportError:\n    njit = None": f'@njit("{SIG}")',
        }
        for imports, decorator in cases.items():
            with self.subTest(imports=imports):
                self.check({decorator: ("njit", SIG, True)}, imports=imports)

    def test_signature_comparison_ignores_whitespace_only(self) -> None:
        compact = "void(float32[::1],float32[::1],float32[::1],int64)"
        spaced = "void( float32[::1], float32[::1],\\tfloat32[::1], int64 )"
        other_spelling = "void(f4[::1], f4[::1], f4[::1], i8)"
        self.check({
            f'@njit("{compact}")': ("njit", compact, True),
            f'@njit("{spaced}")': ("njit", spaced.replace("\\t", "\t"), True),
            # 글자가 다르면 False 로 적을 뿐 빌드는 통과한다(같은 타입의 다른 표기일 수 있다 — 틀린 타입이면 호출이 TypeError 로 드러난다)
            f'@njit("{OTHER_SIG}")': ("njit", OTHER_SIG, False),
            f'@njit("{other_spelling}")': ("njit", other_spelling, False),
        })

    def test_list_or_tuple_of_strings(self) -> None:
        self.check({
            f'@njit(["{SIG}"])': ("njit", f"['{SIG}']", True),
            f'@njit(("{SIG}",))': ("njit", f"('{SIG}',)", True),
            f'@njit(["{OTHER_SIG}", "{SIG}"], cache=None)': ("njit", f"['{OTHER_SIG}', '{SIG}']", True),
            f'@njit(["{OTHER_SIG}"])': ("njit", f"['{OTHER_SIG}']", False),
            f'@jit(["{SIG}"], nopython=True)': ("jit", f"['{SIG}']", True),
        })

    def test_type_expression_is_recorded_as_source_text(self) -> None:
        qualified = "nb.void(nb.float32[::1], nb.float32[::1], nb.float32[::1], nb.int64)"
        arguments_only = "(float32[::1], float32[::1], float32[::1], int64)"
        self.check({
            "@njit(void(float32[::1], float32[::1], float32[::1], int64))": ("njit", SIG, True),
            "@njit(void(float32[::1],float32[::1],float32[::1],int64), parallel=True)": ("njit", SIG, True),
            # 글자가 다른 식은 판단하지 않는다(None): 이름이 무엇에 묶였는지는 실행해야 안다
            f"@njit({qualified})": ("njit", qualified, None),
            f"@njit({arguments_only})": ("njit", arguments_only, None),
            "@njit(SIGNATURE)": ("njit", "SIGNATURE", None),
            f'@njit(["{SIG}", void(float32[::1], float32[::1], float32[::1], int64)])': ("njit", f"['{SIG}', {SIG}]", None),
        })


class RejectedFormsTest(BuildDirTest):
    def check(self, cases: dict[str, str | None], imports: str = IMPORTS, detail: str = "numba_signature_missing") -> None:
        for decorator, kind in cases.items():
            with self.subTest(decorator=decorator):
                res = self.build(module(decorator, imports))
                self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail),
                                 (False, None, "signature_mismatch", detail))
                self.assertEqual(res.extra["decorator"], kind)

    def test_no_numba_jit_decorator(self) -> None:
        self.check({
            "": None,
            "@functools.lru_cache(maxsize=None)": None,
            f'@numba.vectorize(["{SIG}"])': None,
            f'@numba.cfunc("{SIG}")': None,
            "@numba.stencil": None,
        })

    def test_decorator_without_a_signature(self) -> None:
        # 과제 함수도 "njit/jit 로 컴파일되는 함수" 다: 상세는 numba_unsigned_jit:<이름>
        self.check({
            "@njit": "njit",
            "@numba.njit": "njit",
            "@nb.njit()": "njit",
            "@njit(parallel=True)": "njit",
            "@njit(cache=False, fastmath=True, nogil=True)": "njit",
            "@njit(None)": "njit",
            "@njit([])": "njit",
            "@njit(())": "njit",
            "@njit(signature=None)": "njit",
            "@jit": "jit",
            "@jit(nopython=True)": "jit",
            "@numba.jit(nopython=True, parallel=True)": "jit",
        }, detail="numba_unsigned_jit:dot")

    def test_signature_keyword_does_not_count(self) -> None:
        # Numba 0.68 은 `signature=` 를 서명으로 쓰지 않고 조용히 지연 컴파일한다(RealNumbaTest 가 확인한다)
        self.check({
            f'@njit(signature="{SIG}")': "njit",
            f'@njit(fastmath=True, signature="{SIG}")': "njit",
            f'@nb.jit(nopython=True, signature="{SIG}")': "jit",
        }, detail="numba_unsigned_jit:dot")
        res = self.build(module(f'@njit(signature="{SIG}")'))
        self.assertEqual((res.extra["signature_literal"], res.extra["unsigned_jit_functions"]), (None, ["dot"]))

    def test_unsigned_helpers_are_rejected(self) -> None:
        signed = f'@njit("{SIG}")'
        helper_body = "(x, n):\n    x[0] = 0.0\n"
        cases = {
            "bare helper": (f"@njit\ndef _h{helper_body}", "_h"),
            "parallel helper": (f"@njit(parallel=True)\ndef _h{helper_body}", "_h"),
            "called without arguments": (f"@nb.njit()\ndef _h{helper_body}", "_h"),
            "jit helper": (f"@numba.jit(nopython=True)\ndef _h{helper_body}", "_h"),
            "fastmath inline helper": (f'@njit(fastmath=True, inline="always")\ndef _h{helper_body}', "_h"),
            "signature keyword helper": (f'@njit(signature="void(float32[::1], int64)")\ndef _h{helper_body}', "_h"),
            "call form": (f"def _h{helper_body}\n\nhelper = njit(_h)\n", "_h"),
            "call form with options": (f"def _h{helper_body}\n\nhelper = njit(parallel=True)(_h)\n", "_h"),
            "lambda": ("square = njit(lambda v: v * v)\n", "<lambda>"),
            "class body": (f"class Kernels:\n    @staticmethod\n    @njit\n    def _h{helper_body.replace(chr(10) + '    ', chr(10) + '        ')}",
                           "_h"),
        }
        for name, (helper, unsigned) in cases.items():
            with self.subTest(name):
                res = self.build(with_helpers(helper, signed))
                self.assertEqual((res.ok, res.error_kind, res.error_detail),
                                 (False, "signature_mismatch", f"numba_unsigned_jit:{unsigned}"))
                self.assertEqual(res.extra["unsigned_jit_functions"], [unsigned])
                self.assertEqual(res.extra["signature_literal"], SIG, "과제 함수의 관찰값은 그대로 채운다")

    def test_nested_unsigned_function_inside_the_task_function(self) -> None:
        body = BODY.replace("    acc = 0.0\n", "    @njit\n    def inner(v):\n        return v\n    acc = 0.0\n")
        res = self.build(module(f'@njit("{SIG}")', body=body))
        self.assertEqual((res.ok, res.error_detail), (False, "numba_unsigned_jit:inner"))

    def test_first_unsigned_function_in_source_order_is_reported(self) -> None:
        helpers = "@njit\ndef _b(x):\n    return x\n\n\n@njit\ndef _a(x):\n    return x\n\n\n"
        res = self.build(with_helpers(helpers))
        self.assertEqual((res.error_detail, res.extra["unsigned_jit_functions"]), ("numba_unsigned_jit:_b", ["_b", "_a"]))

    def test_cache_is_rejected_anywhere(self) -> None:
        cases = {
            "task function": module(f'@njit("{SIG}", cache=True)'),
            "jit task function": module(f'@numba.jit("{SIG}", nopython=True, cache=True)'),
            "list signature": module(f'@njit(["{OTHER_SIG}", "{SIG}"], cache=True)'),
            "non-literal": module(f'@njit("{SIG}", cache=CACHE)'),
            "helper": with_helpers(f'@njit("{SIG}", cache=True)\ndef _h(x, y, out, n):\n    out[0] = 0.0\n'),
            "call form": with_helpers(f'def _h(x, y, out, n):\n    out[0] = 0.0\n\n\n_fast = njit("{SIG}", cache=True)(_h)\n'),
        }
        for name, code in cases.items():
            with self.subTest(name):
                res = self.build(code)
                self.assertEqual((res.ok, res.error_kind, res.error_detail), (False, "signature_mismatch", "numba_cache_true"))
        res = self.build(module(f'@njit("{SIG}", cache=True)'))
        self.assertIs(res.extra["options"]["cache"], True, "옵션 관찰값은 그대로 기록한다")
        # 서명 없는 함수가 함께 있으면 그쪽이 먼저다
        res = self.build(module("@njit(cache=True)"))
        self.assertEqual(res.error_detail, "numba_unsigned_jit:dot")

    def test_jit_without_a_literal_nopython_true(self) -> None:
        self.check({
            f'@jit("{SIG}")': "jit",
            f'@numba.jit("{SIG}", nopython=False)': "jit",
            f'@jit("{SIG}", forceobj=True)': "jit",
            f'@jit("{SIG}", nopython=NOPYTHON)': "jit",
            f'@jit("{SIG}", nopython=1)': "jit",
        })

    def test_same_names_from_another_package_are_not_numba(self) -> None:
        cases = {
            "from fastlib import njit": f'@njit("{SIG}")',
            "from jax import jit as njit": f'@njit("{SIG}")',
            "import fastlib as numba": f'@numba.njit("{SIG}")',
        }
        for imports, decorator in cases.items():
            with self.subTest(imports=imports):
                self.check({decorator: None}, imports=imports)

    def test_only_the_task_function_counts(self) -> None:
        helper = f'@njit("{SIG}")\ndef _dot(x, y, out, n):\n    out[0] = 0.0\n'
        res = self.build(module("", body=f"{helper}\n\n{BODY}"))
        self.assertEqual((res.ok, res.error_detail, res.extra["decorator"]), (False, "numba_signature_missing", None))
        # 같은 이름을 두 번 정의하면 마지막 정의가 모듈에 남는다(파이썬 의미론)
        res = self.build(module(f'@njit("{SIG}")') + "\n\n" + BODY)
        self.assertEqual((res.ok, res.error_detail, res.extra["decorator"]), (False, "numba_signature_missing", None))
        # 서명 없는 Numba 데코레이터가 붙은 첫 정의가 남아 있어도(가려진 정의) 그 함수는 컴파일되는 함수다
        res = self.build(module("@njit") + f'\n\n@njit("{SIG}")\n' + BODY)
        self.assertEqual((res.ok, res.error_detail), (False, "numba_unsigned_jit:dot"))
        res = self.build(module("") + f'\n\n@njit("{SIG}")\n' + BODY)
        self.assertEqual((res.ok, res.extra["decorator"], res.extra["signature_literal"]), (True, "njit", SIG))

    def test_extra_is_filled_for_rejected_kernels_too(self) -> None:
        res = self.build(module("@njit(parallel=True, fastmath=False)", body=BODY.replace("range(n)", "prange(n)")))
        self.assertEqual(res.extra, {"decorator": "njit", "signature_literal": None, "signature_matches_expected": None,
                                     "options": {**NO_OPTIONS, "parallel": True, "fastmath": False}, "uses_prange": True,
                                     "unsigned_jit_functions": ["dot"], "blas_calls": []})
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["ok"], saved["artifact"], saved["error_kind"], saved["error_detail"], saved["extra"]),
                         (False, None, "signature_mismatch", "numba_unsigned_jit:dot", res.extra))
        self.assertTrue((self.out / "kernel.py").is_file(), "소스는 실패해도 남는다(린트가 읽는다)")


class OptionsAndPrangeTest(BuildDirTest):
    def options(self, decorator: str) -> dict[str, Any]:
        res = self.build(module(decorator))
        self.assertTrue(res.ok, res.error_detail)
        return res.extra["options"]

    def test_literal_booleans_are_recorded(self) -> None:
        self.assertEqual(self.options(f'@njit("{SIG}", parallel=True, fastmath=False, cache=False, boundscheck=False, nogil=True)'),
                         {"parallel": True, "fastmath": False, "cache": False, "boundscheck": False, "nogil": True})
        self.assertEqual(self.options(f'@njit("{SIG}")'), NO_OPTIONS)
        self.assertEqual(self.options(f'@jit("{SIG}", nopython=True, nogil=True)'), {**NO_OPTIONS, "nogil": True})

    def test_anything_but_a_literal_boolean_is_unknown(self) -> None:
        cases = [
            f'@njit("{SIG}", parallel=PARALLEL, fastmath=FAST or True)',
            f'@njit("{SIG}", fastmath={{"fast"}}, cache=None, boundscheck=None)',
            f'@njit("{SIG}", **OPTIONS)',
            f'@njit("{SIG}", error_model="numpy", inline="always", locals={{}})',
        ]
        for decorator in cases:
            with self.subTest(decorator=decorator):
                self.assertEqual(self.options(decorator), NO_OPTIONS)

    def test_uses_prange(self) -> None:
        decorated = f'@njit("{SIG}", parallel=True)'
        helper = '@njit("void(float32[::1], int64)", parallel=True)\ndef _clear(x, n):\n    for i in prange(n):\n        x[i] = 0.0\n'
        cases = {
            "range only": (module(decorated), False),
            "imported name": (module(decorated, body=BODY.replace("range(n)", "prange(n)")), True),
            "module attribute": (module(decorated, body=BODY.replace("range(n)", "numba.prange(n)")), True),
            "module alias": (module(decorated, body=BODY.replace("range(n)", "nb.prange(n)")), True),
            "in a helper": (module("", body=f"{helper}\n\n{decorated}\n{BODY}"), True),      # 모듈의 사실이다(과제 함수 밖이어도)
            "another package": (module(decorated, imports="from numba import njit\nfrom mylib import prange",
                                       body=BODY.replace("range(n)", "prange(n)")), False),
            "in a comment": (module(decorated, body=BODY.replace("acc = 0.0", "acc = 0.0  # prange(n) needs a reduction")), False),
        }
        for name, (code, expected) in cases.items():
            with self.subTest(name):
                res = self.build(code)
                self.assertTrue(res.ok, res.error_detail)
                self.assertIs(res.extra["uses_prange"], expected)


class BlasCallsTest(BuildDirTest):
    """Numba 함수 본문 안의 BLAS 호출은 거절하지 않고 기록한다(decisions.md 2026-11-23; 성능 결정 요약이 읽는다)."""

    def blas(self, body_line: str, imports: str = IMPORTS + "\nimport numpy as np", decorated: bool = True) -> list[dict[str, Any]]:
        body = BODY.replace("    acc = 0.0\n", f"    {body_line}\n    acc = 0.0\n")
        code = module(f'@njit("{SIG}")' if decorated else "", imports=imports, body=body)
        self.line = code.splitlines().index(f"    {body_line}") + 1
        res = self.build(code)
        if decorated:
            self.assertEqual((res.ok, res.error_detail), (True, ""))
        return res.extra["blas_calls"]

    def test_recorded_calls(self) -> None:
        cases = {
            "z = np.dot(x, y)": "numpy.dot",
            "z = np.vdot(x, y)": "numpy.vdot",
            "z = np.matmul(x, y)": "numpy.matmul",
            "z = np.linalg.solve(x, y)": "numpy.linalg.solve",
            "z = np.linalg.norm(x)": "numpy.linalg.norm",
            "z = x.dot(y)": "ndarray.dot",
            "z = (x + y).dot(y)": "ndarray.dot",
            "z = x @ y": "@",
        }
        for line, name in cases.items():
            with self.subTest(line):
                self.assertEqual(self.blas(line), [{"name": name, "line": self.line}])

    def test_aliases_and_from_imports(self) -> None:
        self.assertEqual(self.blas("z = xp.dot(x, y)", imports=IMPORTS + "\nimport numpy as xp"), [{"name": "numpy.dot", "line": self.line}])
        self.assertEqual(self.blas("z = dot(x, y)", imports=IMPORTS + "\nfrom numpy import dot"), [{"name": "numpy.dot", "line": self.line}])
        self.assertEqual(self.blas("z = la.inv(x)", imports=IMPORTS + "\nimport numpy.linalg as la"),
                         [{"name": "numpy.linalg.inv", "line": self.line}])

    def test_augmented_matmul_and_several_calls(self) -> None:
        self.assertEqual(self.blas("x @= y; z = np.dot(x, y) + np.vdot(x, y)"),
                         [{"name": "@", "line": self.line}, {"name": "numpy.dot", "line": self.line}, {"name": "numpy.vdot", "line": self.line}])

    def test_not_recorded(self) -> None:
        for line in ("z = np.sum(x * y)", "z = np.outer(x, y)", "z = math.fsum(x)", "z = dot(x, y)"):
            with self.subTest(line):
                self.assertEqual(self.blas(line), [])
        # Numba 함수 밖(파이썬 함수)의 호출은 Numba 가 컴파일하지 않는다
        self.assertEqual(self.blas("z = np.dot(x, y)", decorated=False), [])

    def test_helper_bodies_count(self) -> None:
        helper = f'@njit("{SIG}")\ndef _h(x, y, out, n):\n    out[0] = np.dot(x, y)\n\n\n'
        plain = "def _p(x, y):\n    return np.dot(x, y)\n\n\n_pf = njit(\"void(float32[::1])\")(_p)\n\n\n"
        code = with_helpers(helper + plain, imports=IMPORTS + "\nimport numpy as np")
        lines = [i for i, text in enumerate(code.splitlines(), 1) if "np.dot" in text]
        res = self.build(code)
        self.assertEqual(len(lines), 2)
        self.assertEqual((res.ok, res.extra["blas_calls"]), (True, [{"name": "numpy.dot", "line": line} for line in lines]))


class NotAKernelTest(BuildDirTest):
    def test_function_not_found(self) -> None:
        cases = {
            "other name": module(f'@njit("{SIG}")').replace("def dot(", "def dot_product("),
            "nested": f'{IMPORTS}\n\n\ndef make():\n    @njit("{SIG}")\n    def dot(x, y, out, n):\n        out[0] = 0.0\n    return dot\n',
            "method": f'{IMPORTS}\n\n\nclass Kernel:\n    @njit("{SIG}")\n    def dot(x, y, out, n):\n        out[0] = 0.0\n',
            "async": module(f'@njit("{SIG}")').replace("def dot(", "async def dot("),
            "assigned": f'{IMPORTS}\n\n\ndef _dot(x, y, out, n):\n    out[0] = 0.0\n\n\ndot = njit("{SIG}")(_dot)\n',
        }
        for name, code in cases.items():
            with self.subTest(name):
                res = self.build(code)
                self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "signature_mismatch", "function_not_found"))
                self.assertEqual((res.extra["decorator"], res.extra["options"]), (None, NO_OPTIONS))

    def test_syntax_error_is_a_compile_error(self) -> None:
        broken = module(f'@njit("{SIG}")').replace("def dot(x, y, out, n):", "def dot(x, y, out, n)")
        res = self.build(broken)
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.extra), (False, None, "compile_error", {}))
        self.assertRegex(res.error_detail, r"^SyntaxError: .+ \(line \d+\)$")
        self.assertEqual(res.log, res.error_detail)
        self.assertEqual((self.out / "kernel.py").read_text(encoding="utf-8"), broken)
        res = self.build("def dot(x, y, out, n):\n    out[0] = 0.0\n\x00")
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))


class CliTest(unittest.TestCase):
    def test_exit_code_and_json(self) -> None:
        unsigned = NUMBA_DOT.replace(f'"{SIG}", ', "")
        self.assertIn("@nb.njit(cache=False, nogil=True)", unsigned)
        with tempfile.TemporaryDirectory(prefix="build_numba_test_") as tmp:
            src = Path(tmp) / "dot.py"
            for code, rc, detail in ((NUMBA_DOT, 0, ""), (unsigned, 1, "numba_unsigned_jit:dot")):
                src.write_text(code, encoding="utf-8")
                stdout = io.StringIO()
                with self.subTest(rc=rc), contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["--src", str(src), "--out-dir", str(Path(tmp) / "out"), "--function", "dot",
                                           "--expected-signature", SIG]), rc)
                    self.assertEqual(json.loads(stdout.getvalue())["error_detail"], detail)


# ----------------------------------------------------------------------------- 실제 Numba 와의 일치

# 모듈 경로들을 받아 차례로 임포트하고 {경로: dot 의 컴파일된 서명 수 | "error: …"} 를 JSON 으로 찍는다
PROBE = r'''
import importlib.util
import json
import sys

out = {}
for path in sys.argv[1:]:
    spec = importlib.util.spec_from_file_location("kernel", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        out[path] = len(mod.dot.signatures)
    except Exception as e:
        out[path] = f"error: {type(e).__name__}: {e}"
print(json.dumps(out))
'''


@requires_modules("numba")
class RealNumbaTest(BuildDirTest):
    """빌드 단계가 받는 꼴은 임포트 때 컴파일되고(서명 1개), 거절하는 꼴은 그렇지 않다(지연 컴파일: 서명 0개).

    테스트용 커널을 Numba 로 실제 컴파일하므로 서브프로세스에서 돌린다(이 프로세스에 Numba 를 올리지 않는다).
    """

    def test_ast_verdict_agrees_with_eager_compilation(self) -> None:
        eager = [
            f'@njit("{SIG}")',
            f'@njit(["{SIG}"])',
            "@njit(void(float32[::1], float32[::1], float32[::1], int64), fastmath=True)",
            f'@jit("{SIG}", nopython=True)',
            f'@nb.njit(signature_or_function="{SIG}")',
        ]
        # `signature=` 키워드: Numba 0.68 은 이 키워드를 서명으로 쓰지 않고 조용히 지연 컴파일한다(실측) — 그래서 빌드가 거절한다
        lazy = ["@njit", "@njit(fastmath=True)", "@jit(nopython=True)", f'@njit(signature="{SIG}")']
        decorator_of: dict[str, str] = {}
        for decorator in [*eager, *lazy]:
            res = self.build(module(decorator))
            self.assertEqual(res.ok, decorator not in lazy, decorator)
            decorator_of[str(self.out / "kernel.py")] = decorator
        proc = subprocess.run([sys.executable, "-c", PROBE, *decorator_of], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        compiled = {decorator_of[path]: count for path, count in json.loads(proc.stdout).items()}
        for decorator in eager:
            with self.subTest(decorator=decorator):
                self.assertEqual(compiled[decorator], 1)
        for decorator in lazy:
            with self.subTest(decorator=decorator):
                self.assertEqual(compiled[decorator], 0)

    def test_signed_helpers_leave_nothing_to_compile_later(self) -> None:
        # 빌드를 통과하는 모듈(과제 함수·도움 함수 모두 서명)은 임포트 뒤 어떤 디스패처도 더 컴파일할 수 없다(_can_compile False)
        helper = f'@njit("{SIG}", parallel=True)\ndef _h(x, y, out, n):\n    out[0] = 0.0\n\n\n'
        res = self.build(with_helpers(helper, body=BODY.replace("    acc = 0.0\n", "    _h(x, y, out, n)\n    acc = 0.0\n")))
        self.assertTrue(res.ok, res.error_detail)
        probe = (
            "import importlib.util, json, sys\n"
            "spec = importlib.util.spec_from_file_location('kernel', sys.argv[1])\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            "print(json.dumps({name: [len(d.signatures), d._can_compile] for name, d in vars(mod).items() if hasattr(d, '_can_compile')}))\n"
        )
        proc = subprocess.run([sys.executable, "-c", probe, str(self.out / "kernel.py")], stdin=subprocess.DEVNULL, capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=600)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(json.loads(proc.stdout), {"_h": [1, False], "dot": [1, False]})


if __name__ == "__main__":
    unittest.main()
