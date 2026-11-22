"""extract 테스트: 펜스 수집·후보 선택(태그/무태그/다중/맨 코드/미종료), python·C 시그니처 검사, gcc -E 경로, 더미 응답 기대.

설계 v3 §6: backend 별 펜스 태그, ``bundle`` (``sigs.signature_bundle``) 경로 — numba·halide 의 파이썬 검사, C keyword 모드의
구조체 포인터, Rust 시그니처(``no_mangle_missing`` 포함) — 와 ``bundle`` 없이 부를 때의 동작 불변, 진단용 CLI.
"""

from __future__ import annotations

import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import yaml

import harness.extract as extract_mod
from harness import sigs
from harness.extract import (
    ExtractResult,
    _strip_rust,
    collect_typedefs,
    extract,
    main,
    parse_c_param,
    parse_c_prototype,
    parse_fenced_blocks,
    parse_rust_signature,
)
from harness.paths import ROOT
from harness.procutil import ProcResult
from harness.taskspec import load_taskspec
from tests._support import requires_tool

SIGNATURES = yaml.safe_load((ROOT / "tasks" / "gemm" / "signatures.yaml").read_text(encoding="utf-8"))

PY_OK = "def gemm(A, B, C, M, N, K):\n    C[:] = A @ B\n"
C_OK = (
    "void gemm(const float* A, const float* B, float* C, int M, int N, int K)\n"
    "{\n    for (int i = 0; i < M * N; i++) C[i] = 0.0f;\n}\n"
)


def py(text: str, **kw: object) -> ExtractResult:
    return extract(text, backend="python", signature_spec=SIGNATURES, **kw)  # type: ignore[arg-type]


def c(text: str, **kw: object) -> ExtractResult:
    return extract(text, backend="c", signature_spec=SIGNATURES, **kw)  # type: ignore[arg-type]


class FenceParsingTest(unittest.TestCase):
    def test_backtick_tilde_crlf_and_longer_closing(self) -> None:
        text = "intro\r\n```python\r\nx = 1\r\n```\r\n~~~\r\ny = 2\r\n~~~\r\n````c\r\nz\r\n```\r\nstill inside\r\n````\r\n"
        blocks = parse_fenced_blocks(text)
        self.assertEqual([(b.tag, b.code, b.terminated) for b in blocks], [
            ("python", "x = 1\n", True),
            ("", "y = 2\n", True),
            ("c", "z\n```\nstill inside\n", True),
        ])

    def test_unterminated_block_runs_to_end(self) -> None:
        blocks = parse_fenced_blocks("```c\nvoid f(void) {}\nmore text\n")
        self.assertEqual(len(blocks), 1)
        self.assertFalse(blocks[0].terminated)
        self.assertIn("more text", blocks[0].code)


class BlockSelectionTest(unittest.TestCase):
    def test_tagged_block_preferred_over_untagged(self) -> None:
        text = "Untagged first:\n```\n" + PY_OK + "```\nThen tagged:\n```python\n" + PY_OK.replace("A @ B", "A.dot(B)") + "```\n"
        res = py(text)
        self.assertTrue(res.ok)
        self.assertIn("A.dot(B)", res.code or "")
        self.assertEqual(res.n_blocks, 2)
        self.assertIn("multiple_blocks_chose:1", res.detail)

    def test_untagged_block_alone_is_accepted(self) -> None:
        res = py("```\n" + PY_OK + "```\n")
        self.assertTrue(res.ok)
        self.assertEqual(res.code, PY_OK)
        self.assertEqual(res.detail, "")

    def test_block_with_definition_beats_longer_block_without(self) -> None:
        helper = "```python\nimport numpy as np\n\n\ndef helper(x):\n    return x * 2  # long filler " + "." * 200 + "\n```\n"
        text = helper + "\nAnd the kernel:\n```python\n" + PY_OK + "```\n"
        res = py(text)
        self.assertTrue(res.ok)
        self.assertEqual(res.code, PY_OK)
        self.assertEqual(res.n_blocks, 2)
        self.assertEqual(res.detail, "multiple_blocks_chose:1")

    def test_wrong_language_tag_is_not_a_candidate(self) -> None:
        res = c("```cpp\n" + C_OK + "```\n")
        self.assertFalse(res.ok)
        self.assertIsNone(res.code)
        self.assertEqual(res.detail, "no_code_block")
        self.assertEqual(res.n_blocks, 1)
        res = c("```C\n" + C_OK + "```\n")
        self.assertTrue(res.ok)

    def test_bare_code_without_fences(self) -> None:
        res = py("import numpy as np\n\n" + PY_OK)
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "bare_code")
        self.assertEqual(res.n_blocks, 0)
        self.assertTrue((res.code or "").startswith("import numpy"))

    def test_prose_only_is_no_code_block(self) -> None:
        res = py("I cannot help with that request.")
        self.assertEqual(res, ExtractResult(ok=False, code=None, signature_ok=None, signature_exact=None, detail="no_code_block", n_blocks=0))

    def test_unterminated_fence(self) -> None:
        res = py("Here you go:\n```python\n" + PY_OK)
        self.assertTrue(res.ok)
        self.assertEqual(res.detail, "unterminated_fence")
        self.assertTrue(res.signature_ok)
        res = c("```c\nvoid gemm(const float* A, const float* B, float* C, int M, int N, int K);\n")
        self.assertFalse(res.ok)
        self.assertEqual(res.detail, "function_not_found;unterminated_fence")


class PythonSignatureTest(unittest.TestCase):
    def test_exact_positional(self) -> None:
        res = py("```python\n" + PY_OK + "```")
        self.assertTrue(res.ok)
        self.assertTrue(res.signature_ok)
        self.assertTrue(res.signature_exact)

    def test_decorator_and_annotations_allowed_but_not_exact(self) -> None:
        code = "import functools\n\n@functools.lru_cache(maxsize=None)\ndef gemm(A: object, B, C, M: int, N, K) -> None:\n    pass\n"
        res = py("```python\n" + code + "```")
        self.assertTrue(res.ok)
        self.assertTrue(res.signature_ok)
        self.assertFalse(res.signature_exact)

    def test_missing_parameter_fails(self) -> None:
        res = py("```python\ndef gemm(A, B, M, N, K):\n    return A @ B\n```")
        self.assertFalse(res.ok)
        self.assertFalse(res.signature_ok)
        self.assertTrue(res.detail.startswith("param_mismatch:"), res.detail)
        self.assertIn("got (A, B, M, N, K)", res.detail)

    def test_extra_default_parameter_fails(self) -> None:
        res = py("```python\ndef gemm(A, B, C, M, N, K, block=64):\n    pass\n```")
        self.assertFalse(res.ok)
        self.assertTrue(res.detail.startswith("param_mismatch:"))
        res = py("```python\ndef gemm(A, B, C, M, N, K, *, block=64):\n    pass\n```")
        self.assertFalse(res.ok)
        res = py("```python\ndef gemm(A, B, C, M, N, K, *args, **kwargs):\n    pass\n```")
        self.assertFalse(res.ok)

    def test_default_on_existing_parameter_is_inexact_only(self) -> None:
        res = py("```python\ndef gemm(A, B, C, M, N, K=None):\n    pass\n```")
        self.assertTrue(res.ok)
        self.assertFalse(res.signature_exact)

    def test_async_def_fails(self) -> None:
        res = py("```python\nasync def gemm(A, B, C, M, N, K):\n    pass\n```")
        self.assertFalse(res.ok)
        self.assertFalse(res.signature_ok)
        self.assertEqual(res.detail, "param_mismatch:async_def")

    def test_keyword_mode(self) -> None:
        kw = "```python\ndef gemm(*, A, B, C, M, N, K):\n    pass\n```"
        res = py(kw, args_mode="keyword")
        self.assertTrue(res.ok)
        self.assertTrue(res.signature_exact)
        self.assertFalse(py(kw).ok)  # positional 조건에서는 실패
        res = py("```python\n" + PY_OK + "```", args_mode="keyword")
        self.assertFalse(res.ok)
        self.assertIn("expected (*, A, B, C, M, N, K)", res.detail)

    def test_syntax_error_is_deferred(self) -> None:
        res = py("```python\ndef gemm(A, B, C, M, N, K)\n    C[:] = A @ B\n```")
        self.assertTrue(res.ok)
        self.assertIsNone(res.signature_ok)
        self.assertEqual(res.detail, "syntax_error_deferred")

    def test_function_not_found(self) -> None:
        res = py("```python\ndef matmul(A, B, C, M, N, K):\n    pass\n```")
        self.assertFalse(res.ok)
        self.assertEqual(res.detail, "function_not_found")
        # 중첩 정의는 최상위가 아니다
        res = py("```python\nclass X:\n    def gemm(A, B, C, M, N, K):\n        pass\n```")
        self.assertFalse(res.ok)
        self.assertEqual(res.detail, "function_not_found")


C_MESSY = r"""
/* Blocked GEMM.
   void gemm(...) { }  <- this is a comment, not a definition */
#include <stddef.h>
#include <string.h>
#define RESTRICT __restrict__
#define BLOCK 64   // tile size

static inline void zero_row(float *RESTRICT row, int n) { memset(row, 0, (size_t)n * sizeof *row); }

// forward declaration
void gemm(const float *A, const float *B, float *C, int M, int N, int K);

void
gemm(const float *RESTRICT A, const float *RESTRICT B,
     float *RESTRICT C, int M, int N, int K)
    __attribute__((hot))
{
    for (int i = 0; i < M; i++) {
        zero_row(C + (size_t)i * N, N);
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
}
"""


class CSignatureTest(unittest.TestCase):
    def test_exact_signature(self) -> None:
        res = c("```c\n" + C_OK + "```")
        self.assertTrue(res.ok)
        self.assertTrue(res.signature_ok)
        self.assertTrue(res.signature_exact)
        self.assertEqual(res.detail, "")

    def test_messy_source_without_preprocessor(self) -> None:
        res = c("```c\n" + C_MESSY + "```")
        self.assertTrue(res.ok, res.detail)
        self.assertTrue(res.signature_ok)
        self.assertFalse(res.signature_exact)  # restrict 한정자 차이
        self.assertEqual(res.detail, "")

    @unittest.skipIf(shutil.which("gcc") is None, "gcc 필요")
    def test_messy_source_with_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            work = Path(tmp) / "sample"
            res = c("```c\n" + C_MESSY + "```", work_dir=work)
            self.assertTrue(res.ok, res.detail)
            self.assertTrue(res.signature_ok)
            self.assertFalse(res.signature_exact)
            self.assertEqual(res.detail, "")
            self.assertTrue((work / "kernel.c").exists())
            self.assertTrue((work / "kernel.i").exists())
            # 매크로가 함수 이름을 만들어도 전처리 경로는 찾는다
            macro = "#define KERNEL(name) void name(const float* A, const float* B, float* C, int M, int N, int K)\nKERNEL(gemm) { }\n"
            res = c("```c\n" + macro + "```", work_dir=work)
            self.assertTrue(res.ok, res.detail)
            self.assertTrue(res.signature_exact)

    @unittest.skipIf(shutil.which("gcc") is None, "gcc 필요")
    def test_preprocess_failure_falls_back(self) -> None:
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            res = c("```c\n#include <no_such_header_xyz.h>\n" + C_OK + "```", work_dir=Path(tmp))
            self.assertTrue(res.ok, res.detail)
            self.assertEqual(res.detail, "preprocess_failed")

    def test_static_definition_rejected_but_static_helper_allowed(self) -> None:
        res = c("```c\nstatic void gemm(const float* A, const float* B, float* C, int M, int N, int K) { }\n```")
        self.assertFalse(res.ok)
        self.assertEqual(res.detail, "static_linkage")
        res = c("```c\nstatic void helper(void) { }\n" + C_OK + "```")
        self.assertTrue(res.ok)
        self.assertTrue(res.signature_exact)
        res = c("```c\nstatic int counter;\n#pragma GCC optimize(\"O3\")\n" + C_OK + "```")
        self.assertTrue(res.ok)

    def test_prototype_only_is_function_not_found(self) -> None:
        res = c("```c\nvoid gemm(const float* A, const float* B, float* C, int M, int N, int K);\n```")
        self.assertFalse(res.ok)
        self.assertFalse(res.signature_ok)
        self.assertEqual(res.detail, "function_not_found")
        # 호출만 있는 경우도 정의가 아니다
        res = c("```c\nvoid run(void) { gemm(0, 0, 0, 1, 1, 1); }\n```")
        self.assertEqual(res.detail, "function_not_found")

    def test_parameter_mismatches(self) -> None:
        cases = {
            "void gemm(const double* A, const float* B, float* C, int M, int N, int K) { }": "type",
            "void gemm(const float* A, const float* B, float* C, int M, int N) { }": "5 params",
            "void gemm(const float* A, const float* B, float* C, int N, int M, int K) { }": "name N expected M",
            "void gemm(const float* A, const float* B, float* C, size_t M, int N, int K) { }": "type",
            "void gemm(const float* A, const float* B, float** C, int M, int N, int K) { }": "type float** expected float*",
            "int gemm(const float* A, const float* B, float* C, int M, int N, int K) { return 0; }": "return_type int",
            "gemm(const float* A, const float* B, float* C, int M, int N, int K) { }": "return_type missing",
        }
        for src, expected in cases.items():
            with self.subTest(src=src):
                res = c("```c\n" + src + "\n```")
                self.assertFalse(res.ok)
                self.assertFalse(res.signature_ok)
                self.assertTrue(res.detail.startswith("param_mismatch:"), res.detail)
                self.assertIn(expected, res.detail)

    def test_qualifier_and_array_variants_are_ok_but_inexact(self) -> None:
        cases = [
            "void gemm(const float *restrict A, const float *restrict B, float *restrict C, int M, int N, int K) { }",
            "void gemm(const float A[], const float B[], float C[], int M, int N, int K) { }",
            "void gemm(const float *__restrict__ A, const float *__restrict B, float *C, const int M, const int N, const int K) { }",
            "void gemm(float const* A, const float* B, float* C, int M, int N, int K) { }",
        ]
        for src in cases:
            with self.subTest(src=src):
                res = c("```c\n" + src + "\n```")
                self.assertTrue(res.ok, res.detail)
                self.assertTrue(res.signature_ok)
                self.assertFalse(res.signature_exact)

    def test_param_parser(self) -> None:
        p = parse_c_param("const float *restrict A")
        self.assertEqual((p.base, p.depth, p.name), ("float", 1, "A"))
        p = parse_c_param("float B[]")
        self.assertEqual((p.base, p.depth, p.name), ("float", 1, "B"))
        p = parse_c_param("unsigned int M")
        self.assertEqual((p.base, p.depth, p.name), ("unsigned int", 0, "M"))
        ret, params = parse_c_prototype("void gemm(const float* A, const float* B, float* C, int M, int N, int K)")
        self.assertEqual(ret, ["void"])
        self.assertEqual([q.name for q in params], ["A", "B", "C", "M", "N", "K"])

    def test_attribute_on_parameter_is_still_a_definition(self) -> None:
        # 매개변수에 붙은 __attribute__((aligned(64))) 는 괄호 3단계: 정의를 찾아야 하고(예전엔 function_not_found),
        # 속성은 `)` 뒤의 속성과 마찬가지로 시그니처 비교에서 제외된다(실제 빌드는 성공하고 T gemm 이 내보내진다).
        cases = [
            "void gemm(const float* __attribute__((aligned(64))) A, const float* B, float* C, int M, int N, int K) { }",
            "void gemm(const float* __attribute__((aligned(64), unused)) A, const float* __attribute__((aligned(64))) B,\n"
            "          float* __attribute__((aligned(64))) C, int M, int N, int K) { }",
        ]
        for src in cases:
            with self.subTest(src=src):
                res = c("```c\n" + src + "\n```")
                self.assertTrue(res.ok, res.detail)
                self.assertTrue(res.signature_ok)
                self.assertTrue(res.signature_exact)
                self.assertEqual(res.detail, "")
        # 함수 포인터 매개변수(괄호 1단계)는 전과 같이 이름 불일치
        res = c("```c\nvoid gemm(const float* A, const float* B, float* C, int M, int N, void (*cb)(int)) { }\n```")
        self.assertFalse(res.ok)
        self.assertIn("param_mismatch:param 6", res.detail)

    def test_typedef_is_resolved_before_base_type_comparison(self) -> None:
        # typedef 로 쓴 같은 타입은 통과(예전엔 param_mismatch:... real* expected float*); 원문 토큰이 다르므로 exact 는 아니다
        cases = [
            "typedef float real;\nvoid gemm(const real* A, const real* B, real* C, int M, int N, int K) { }",
            "typedef float *fptr;\ntypedef const float *cfptr;\nvoid gemm(cfptr A, cfptr B, fptr C, int M, int N, int K) { }",
            "typedef float real;\ntypedef real scalar_t;\ntypedef int dim_t;\n"
            "void gemm(const scalar_t* A, const scalar_t* B, scalar_t* C, dim_t M, dim_t N, dim_t K) { }",
            "typedef float real __attribute__((aligned(4)));\nvoid gemm(const real A[], const real B[], real C[], int M, int N, int K) { }",
        ]
        for src in cases:
            with self.subTest(src=src):
                res = c("```c\n" + src + "\n```")
                self.assertTrue(res.ok, res.detail)
                self.assertTrue(res.signature_ok)
                self.assertFalse(res.signature_exact)
                self.assertEqual(res.detail, "")
        # 다른 기본 타입으로의 typedef 는 해소된 타입으로 불일치를 보고한다
        res = c("```c\ntypedef double real;\nvoid gemm(const real* A, const real* B, real* C, int M, int N, int K) { }\n```")
        self.assertFalse(res.ok)
        self.assertIn("type double* expected float*", res.detail)
        # 구조체·함수 포인터·배열 typedef 는 수집하지 않는다; 순환 typedef 도 멈춘다
        src = "typedef struct { int x; } ctx_t;\ntypedef void (*fn)(int);\ntypedef float vec[4];\ntypedef float real;\ntypedef enum { P, Q } e_t;\n"
        self.assertEqual(collect_typedefs(src), {"real": ["float"]})
        p = parse_c_param("const a* A", {"a": ["b"], "b": ["a"]})
        self.assertEqual((p.depth, p.name), (1, "A"))

    @unittest.skipIf(shutil.which("gcc") is None, "gcc 필요")
    def test_attribute_and_typedef_with_preprocessor(self) -> None:
        src = "typedef float real;\nvoid gemm(const real* __attribute__((aligned(64))) A, const real* B, real* C, int M, int N, int K) { }\n"
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            res = c("```c\n" + src + "```", work_dir=Path(tmp))
            self.assertTrue(res.ok, res.detail)
            self.assertTrue(res.signature_ok)
            self.assertFalse(res.signature_exact)
            self.assertEqual(res.detail, "")


class DummyResponsesTest(unittest.TestCase):
    """더미 응답(§13)은 numpy/1 만 추출 단계에서 실패해야 한다."""

    DUMMY = ROOT / "harness" / "dummy_responses" / "gemm"

    def test_all_dummy_responses(self) -> None:
        expected_fail = {("numpy", 1)}
        seen = 0
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            # 턴 0 응답(<k>.md)만 본다. 수정 턴의 더미(<k>.turn<t>.md; 설계 v3 §14)는 이 기대표의 대상이 아니다
            for md in sorted(p for p in (self.DUMMY / lang).glob("*.md") if p.stem.isdigit()):
                k = int(md.stem)
                res = extract(md.read_text(encoding="utf-8"), backend=backend, signature_spec=SIGNATURES)
                seen += 1
                with self.subTest(lang=lang, k=k):
                    self.assertEqual(res.n_blocks, 1)
                    if (lang, k) in expected_fail:
                        self.assertFalse(res.ok)
                        self.assertTrue(res.detail.startswith("param_mismatch:"), res.detail)
                    else:
                        self.assertTrue(res.ok, res.detail)
                        self.assertTrue(res.signature_ok)
                        self.assertEqual(res.detail, "")
        self.assertEqual(seen, 11)


# ----------------------------------------------------------------------------- 설계 v3 §6: backend·bundle


def bundle_for(task: str, language: str, backend: str, args_mode: str = "positional") -> dict[str, Any]:
    """run.py 가 넘기는 것과 같은 시그니처 묶음."""
    return sigs.signature_bundle(load_taskspec(ROOT / "tasks" / task / "signatures.yaml"), language, backend, args_mode)


def fenced(tag: str, code: str) -> str:
    return f"Here is the kernel.\n\n```{tag}\n{code}\n```\n"


def verdict(res: ExtractResult) -> tuple[bool, bool | None, bool | None, str]:
    return res.ok, res.signature_ok, res.signature_exact, res.detail


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

HALIDE_GEMM = '''import halide as hl


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]
    return C
'''


class BackendTest(unittest.TestCase):
    def test_fence_tags_per_backend(self) -> None:
        python_tags = ("python", "py", "python3", "Python", "")  # 태그는 대소문자를 가리지 않고, 무태그 블록도 후보다
        accepted = {"python": python_tags, "numba": python_tags, "halide": python_tags, "c": ("c", "C", ""), "rust": ("rust", "rs", "Rust", "")}
        sources = {"python": PY_OK, "numba": NUMBA_GEMM, "halide": HALIDE_GEMM, "c": C_OK, "rust": RS_GEMM}
        languages = {"python": "numpy", "numba": "numba", "halide": "halide_auto", "c": "c_seq", "rust": "rust"}
        for backend, tags in accepted.items():
            bundle = bundle_for("gemm", languages[backend], backend)
            for tag in ("python", "py", "python3", "Python", "c", "C", "rust", "rs", "Rust", "cpp", "numba", "halide", "toml", ""):
                with self.subTest(backend=backend, tag=tag):
                    res = extract(fenced(tag, sources[backend]), backend=backend, signature_spec={}, bundle=bundle)
                    if tag in tags:
                        self.assertEqual(verdict(res), (True, True, True, ""))
                    else:
                        self.assertEqual((res.ok, res.code, res.detail, res.n_blocks), (False, None, "no_code_block", 1))

    def test_unknown_backend_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            extract(fenced("cpp", C_OK), backend="cpp", signature_spec=SIGNATURES)

    def test_halide_and_rust_need_a_bundle(self) -> None:
        # halide 의 매개변수(출력 인자 없음)와 Rust 의 타입은 signature_spec 만으로는 알 수 없다
        for backend, tag, code in (("halide", "python", HALIDE_GEMM), ("rust", "rust", RS_GEMM)):
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                extract(fenced(tag, code), backend=backend, signature_spec=SIGNATURES)

    def test_numba_without_bundle_uses_the_python_check(self) -> None:
        res = extract(fenced("python", NUMBA_GEMM), backend="numba", signature_spec=SIGNATURES)
        self.assertEqual(verdict(res), (True, True, True, ""))
        res = extract(fenced("python", NUMBA_GEMM.replace("(A, B, C, M, N, K)", "(A, B, C, M, K, N)")), backend="numba",
                      signature_spec=SIGNATURES)
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:got (A, B, C, M, K, N) expected (A, B, C, M, N, K)"))


class BundleTest(unittest.TestCase):
    """``bundle`` 이 있으면 기대 함수 이름·매개변수·프로토타입은 거기서 온다(``signature_spec`` 은 보지 않는다)."""

    def test_bundle_wins_over_signature_spec(self) -> None:
        wrong_spec = {"function": "matmul", "params": ["X", "Y"], "positional": "void matmul(float* X, float* Y)"}
        res = extract(fenced("python", PY_OK), backend="python", signature_spec=wrong_spec, bundle=bundle_for("gemm", "numpy", "python"))
        self.assertEqual(verdict(res), (True, True, True, ""))
        res = extract(fenced("c", C_OK), backend="c", signature_spec=wrong_spec, bundle=bundle_for("gemm", "c_seq", "c"))
        self.assertEqual(verdict(res), (True, True, True, ""))
        # 같은 응답을 다른 과제의 bundle 로 보면 그 과제의 함수를 찾는다
        res = extract(fenced("c", C_OK), backend="c", signature_spec=SIGNATURES, bundle=bundle_for("dot", "c_seq", "c"))
        self.assertEqual(verdict(res), (False, False, False, "function_not_found"))

    def test_bundle_and_signature_spec_agree_on_the_gemm_dummies(self) -> None:
        # 기존 과제에서는 bundle 을 넘기든 안 넘기든(= v3 이전의 호출) 결과가 같다
        dummy = ROOT / "harness" / "dummy_responses" / "gemm"
        seen = 0
        for lang, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            bundle = bundle_for("gemm", lang, backend)
            for md in sorted(p for p in (dummy / lang).glob("*.md") if p.stem.isdigit()):
                text = md.read_text(encoding="utf-8")
                seen += 1
                with self.subTest(lang=lang, file=md.name):
                    self.assertEqual(extract(text, backend=backend, signature_spec={}, bundle=bundle),
                                     extract(text, backend=backend, signature_spec=SIGNATURES))
        self.assertEqual(seen, 11)

    def test_numba_is_checked_like_python(self) -> None:
        bundle = bundle_for("gemm", "numba", "numba")
        self.assertEqual(verdict(extract(fenced("python", NUMBA_GEMM), backend="numba", signature_spec={}, bundle=bundle)),
                         (True, True, True, ""))
        # 데코레이터가 없어도 추출은 통과한다(서명 검사는 빌드 단계)
        bare = "def gemm(A, B, C, M, N, K):\n    C[:] = A @ B\n"
        self.assertEqual(verdict(extract(fenced("py", bare), backend="numba", signature_spec={}, bundle=bundle)), (True, True, True, ""))
        res = extract(fenced("python", NUMBA_GEMM.replace("def gemm(A, B, C, M, N, K)", "def gemm(A, B, C)")), backend="numba",
                      signature_spec={}, bundle=bundle)
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:got (A, B, C) expected (A, B, C, M, N, K)"))
        keyword = bundle_for("gemm", "numba", "numba", "keyword")
        kw_code = NUMBA_GEMM.replace("def gemm(A, B, C, M, N, K)", "def gemm(*, A, B, C, M, N, K)")
        self.assertEqual(verdict(extract(fenced("python", kw_code), backend="numba", signature_spec={}, args_mode="keyword", bundle=keyword)),
                         (True, True, True, ""))
        res = extract(fenced("python", NUMBA_GEMM), backend="numba", signature_spec={}, args_mode="keyword", bundle=keyword)
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:got (A, B, C, M, N, K) expected (*, A, B, C, M, N, K)"))

    def test_halide_parameters_exclude_the_output(self) -> None:
        bundle = bundle_for("gemm", "halide_auto", "halide")
        self.assertEqual(bundle["params"], ["A", "B", "M", "N", "K"])
        self.assertEqual(verdict(extract(fenced("python", HALIDE_GEMM), backend="halide", signature_spec={}, bundle=bundle)),
                         (True, True, True, ""))
        # 출력 인자를 받는 꼴(다른 언어의 시그니처)은 halide 계약이 아니다
        with_output = HALIDE_GEMM.replace("def gemm(A, B, M, N, K)", "def gemm(A, B, C, M, N, K)")
        res = extract(fenced("python", with_output), backend="halide", signature_spec={}, bundle=bundle)
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:got (A, B, C, M, N, K) expected (A, B, M, N, K)"))
        keyword = bundle_for("gemm", "halide_manual", "halide", "keyword")
        kw_code = HALIDE_GEMM.replace("def gemm(A, B, M, N, K)", "def gemm(*, A, B, M, N, K)")
        self.assertEqual(verdict(extract(fenced("python", kw_code), backend="halide", signature_spec={}, args_mode="keyword", bundle=keyword)),
                         (True, True, True, ""))
        res = extract(fenced("python", HALIDE_GEMM), backend="halide", signature_spec={}, args_mode="keyword", bundle=keyword)
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:got (A, B, M, N, K) expected (*, A, B, M, N, K)"))

    def test_c_positional_prototype_comes_from_the_bundle(self) -> None:
        bundle = bundle_for("dot", "c_seq", "c")
        ok = "void dot(const float* x, const float* y, float* out, int n)\n{\n    out[0] = 0.0f;\n}\n"
        self.assertEqual(verdict(extract(fenced("c", ok), backend="c", signature_spec={}, bundle=bundle)), (True, True, True, ""))
        cases = {
            "void dot(const float* y, const float* x, float* out, int n) { }": "param_mismatch:param 1 name y expected x",
            "void dot(const float* x, const float* y, double* out, int n) { }": "param_mismatch:param 3 (out) type double* expected float*",
            "void dot(const float* x, const float* y, float* out) { }": "param_mismatch:3 params expected 4",
            "float dot(const float* x, const float* y, float* out, int n) { return 0; }": "param_mismatch:return_type float expected void",
        }
        for src, detail in cases.items():
            with self.subTest(src=src):
                self.assertEqual(verdict(extract(fenced("c", src), backend="c", signature_spec={}, bundle=bundle)), (False, False, False, detail))


C_KEYWORD_OK = '''#include "kernel_args.h"

void gemm(const gemm_args* args)
{
    const float *A = args->A, *B = args->B;
    float *C = args->C;
    for (int i = 0; i < args->M; i++)
        for (int j = 0; j < args->N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < args->K; k++)
                acc += A[i * args->K + k] * B[k * args->N + j];
            C[i * args->N + j] = acc;
        }
}
'''


class CKeywordTest(unittest.TestCase):
    """C keyword 모드: 매개변수는 ``{fn}_args`` 의 포인터 하나(const 유무 무관), 이름은 자유(``args`` 가 아니면 exact 만 False)."""

    BUNDLE = bundle_for("gemm", "c_seq", "c", "keyword")

    def kw(self, code: str, **kwargs: Any) -> ExtractResult:
        return extract(fenced("c", code), backend="c", signature_spec={}, args_mode="keyword", bundle=self.BUNDLE, **kwargs)

    def test_exact_form(self) -> None:
        self.assertEqual(self.BUNDLE["signature"], "void gemm(const gemm_args* args)")
        self.assertEqual(verdict(self.kw(C_KEYWORD_OK)), (True, True, True, ""))

    def test_accepted_but_inexact_forms(self) -> None:
        cases = [
            "void gemm(gemm_args* args) { }",                      # const 없음
            "void gemm(const gemm_args *restrict args) { }",
            "void gemm(gemm_args const * const args) { }",
            "void gemm(const gemm_args* a) { }",                   # 다른 이름
            "void gemm(const gemm_args* p_args) { }",
            "void gemm(const gemm_args args[]) { }",
            "typedef gemm_args params_t;\nvoid gemm(const params_t* args) { }",
        ]
        for src in cases:
            with self.subTest(src=src):
                self.assertEqual(verdict(self.kw(src)), (True, True, False, ""))

    def test_redefining_the_struct_is_not_checked(self) -> None:
        # 헤더와 충돌하면 컴파일 오류가 된다 — 추출 단계는 보지 않는다
        src = ("typedef struct { const float* A; const float* B; float* C; int M; int N; int K; } gemm_args;\n"
               "void gemm(const gemm_args* args) { }")
        self.assertEqual(verdict(self.kw(src)), (True, True, True, ""))

    def test_rejected_forms(self) -> None:
        cases = {
            "void gemm(const float* A, const float* B, float* C, int M, int N, int K) { }": "param_mismatch:6 params expected 1",
            "void gemm(void) { }": "param_mismatch:0 params expected 1",
            "void gemm(gemm_args args) { }": "param_mismatch:param 1 (args) type gemm_args expected gemm_args*",
            "void gemm(const gemm_args** args) { }": "param_mismatch:param 1 (args) type gemm_args** expected gemm_args*",
            "void gemm(const struct gemm_args* args) { }": "param_mismatch:param 1 (args) type struct gemm_args* expected gemm_args*",
            "void gemm(const void* args) { }": "param_mismatch:param 1 (args) type void* expected gemm_args*",
            "void gemm(const dot_args* args) { }": "param_mismatch:param 1 (args) type dot_args* expected gemm_args*",
            "int gemm(const gemm_args* args) { return 0; }": "param_mismatch:return_type int expected void",
            "static void gemm(const gemm_args* args) { }": "static_linkage",
            "void gemm(const gemm_args* args);": "function_not_found",
        }
        for src, detail in cases.items():
            with self.subTest(src=src):
                self.assertEqual(verdict(self.kw(src)), (False, False, False, detail))

    def test_keyword_without_bundle_is_checked_as_positional_like_before(self) -> None:
        # bundle 이 없으면 구조체의 기대 프로토타입을 모른다: v3 이전과 같이 positional 로 검사한다(stderr 에 그 사실을 알린다)
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            res = extract(fenced("c", C_OK), backend="c", signature_spec=SIGNATURES, args_mode="keyword")
            self.assertEqual(verdict(res), (True, True, True, ""))
            res = extract(fenced("c", C_KEYWORD_OK), backend="c", signature_spec=SIGNATURES, args_mode="keyword")
            self.assertEqual(verdict(res), (False, False, False, "param_mismatch:1 params expected 6"))
        self.assertEqual(stderr.getvalue().count("[extract] 참고:"), 2)

    def test_positional_bundle_does_not_accept_the_struct_form(self) -> None:
        res = extract(fenced("c", C_KEYWORD_OK), backend="c", signature_spec={}, bundle=bundle_for("gemm", "c_seq", "c"))
        self.assertEqual(verdict(res), (False, False, False, "param_mismatch:1 params expected 6"))

    def test_args_header_is_written_for_the_preprocessor(self) -> None:
        # gcc 가 없는 머신에서는 전처리 실패 경고가 stderr 로 나온다(판정은 원문 검사로 같다): 출력만 가린다
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp, contextlib.redirect_stderr(io.StringIO()):
            work = Path(tmp) / "sample"
            res = self.kw(C_KEYWORD_OK, work_dir=work)
            self.assertTrue(res.ok, res.detail)
            self.assertEqual((work / "kernel_args.h").read_bytes().decode("utf-8"), self.BUNDLE["args_struct"])
            self.assertIn("} gemm_args;", self.BUNDLE["args_struct"])
            # positional 에서는 헤더를 쓰지 않는다
            other = Path(tmp) / "positional"
            extract(fenced("c", C_OK), backend="c", signature_spec={}, bundle=bundle_for("gemm", "c_seq", "c"), work_dir=other)
            self.assertTrue((other / "kernel.c").exists())
            self.assertFalse((other / "kernel_args.h").exists())

    @requires_tool("gcc")
    def test_keyword_kernel_goes_through_the_preprocessor(self) -> None:
        # 헤더가 work_dir 에 있어야 `#include "kernel_args.h"` 가 풀린다(없으면 preprocess_failed 로 떨어진다)
        macro = '#include "kernel_args.h"\n#define KERNEL(name) void name(const gemm_args* args)\nKERNEL(gemm) { (void)args; }\n'
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            self.assertEqual(verdict(self.kw(C_KEYWORD_OK, work_dir=Path(tmp) / "a")), (True, True, True, ""))
            self.assertEqual(verdict(self.kw(macro, work_dir=Path(tmp) / "b")), (True, True, True, ""))


# glibc 의 gcc -E 출력 흉내: 헤더 유래 줄(int32_t 의 typedef 사슬)은 라인 마커로 걸러지고 kernel.c 유래 줄만 검사된다
PREPROCESSED_STDINT = '''# 1 "kernel.c"
# 1 "<built-in>"
# 1 "kernel.c"
# 1 "/usr/include/stdint.h" 1 3 4
# 1 "/usr/include/x86_64-linux-gnu/bits/types.h" 1 3 4
typedef signed int __int32_t;
# 2 "/usr/include/stdint.h" 2 3 4
typedef __int32_t int32_t;
# 2 "kernel.c" 2

{kernel}'''


class CIntSpellingTest(unittest.TestCase):
    """C 의 기본 타입은 한정자를 뺀 단어열 그대로 비교한다: ``int32_t``·``signed``·``signed int``·``int signed`` 는 ``int`` 와
    다른 철자이므로 param_mismatch(→ signature_mismatch)다(docs/decisions.md 2026-11-23 — 기존 c_seq·c_openmp 판정 유지).
    크기·부호가 다른 타입도 물론 param_mismatch 다."""

    SCAN = "void scan(const int32_t* restrict x, int32_t* restrict out, int n)\n{\n    out[0] = x[0] + n;\n}\n"
    SPMV = ("void spmv_csr(const int32_t* row_ptr, const signed int* col_idx, const float* vals, const float* x, float* y,\n"
            "              signed int n_rows, signed n_cols, int signed nnz)\n{\n    y[0] = vals[row_ptr[0] + col_idx[0]] * x[n_cols];\n}\n")
    EXPECTED = {
        "scan": (False, False, False, "param_mismatch:param 1 (x) type int32_t* expected int*"),
        "spmv_csr": (False, False, False, "param_mismatch:param 1 (row_ptr) type int32_t* expected int*"),
    }

    def check(self, task: str, code: str, args_mode: str = "positional", **kwargs: Any) -> tuple[bool, bool | None, bool | None, str]:
        bundle = bundle_for(task, "c_seq", "c", args_mode)
        return verdict(extract(fenced("c", code), backend="c", signature_spec={}, args_mode=args_mode, bundle=bundle, **kwargs))

    def test_other_int_spellings_are_mismatches(self) -> None:
        for task, code in (("scan", self.SCAN), ("spmv_csr", self.SPMV)):
            with self.subTest(task=task):
                self.assertEqual(self.check(task, "#include <stdint.h>\n" + code), self.EXPECTED[task])
        # 스칼라 매개변수의 철자 셋과 int32_t: 상세에는 한정자를 뺀 철자가 그대로 나온다
        for spelling, base in (("signed", "signed"), ("signed int", "signed int"), ("int signed", "int signed"),
                               ("const signed int", "signed int"), ("int32_t", "int32_t")):
            with self.subTest(spelling=spelling):
                code = f"void scan(const int* x, int* out, {spelling} n) {{ }}\n"
                self.assertEqual(self.check("scan", code), (False, False, False, f"param_mismatch:param 3 (n) type {base} expected int"))
        # typedef 는 해소되지만 해소 결과(int32_t)가 int 가 아니다
        code = "typedef int32_t idx_t;\nvoid scan(const idx_t* x, idx_t* out, idx_t n) { }\n"
        self.assertEqual(self.check("scan", code), (False, False, False, "param_mismatch:param 1 (x) type int32_t* expected int*"))
        p = parse_c_param("const signed int *restrict col_idx")
        self.assertEqual((p.base, p.depth, p.name), ("signed int", 1, "col_idx"))

    def test_const_placement_and_typedef_to_int_still_pass(self) -> None:
        # 한정자 위치·typedef 로 int 에 이르는 철자는 통과하고 exact 만 달라진다(기존 규칙)
        cases = {
            "void scan(const int* x, int* out, int n) { }\n": (True, True, True, ""),
            "void scan(int const* x, int* out, int n) { }\n": (True, True, False, ""),
            "void scan(const int* const x, int* out, const int n) { }\n": (True, True, False, ""),
            "void scan(const int* restrict x, int* restrict out, int n) { }\n": (True, True, False, ""),
            "typedef int idx_t;\nvoid scan(const idx_t* x, idx_t* out, idx_t n) { }\n": (True, True, False, ""),
        }
        for code, want in cases.items():
            with self.subTest(code=code):
                self.assertEqual(self.check("scan", code), want)

    def test_preprocessed_glibc_text_is_a_mismatch_too(self) -> None:
        # 전처리 경로: run_isolated 대역이 glibc 식 출력을 쓴다(헤더의 typedef 는 보이지 않으므로 int32_t 가 그대로 남는다)
        def fake_cpp(cmd: list[str], *, stdout_path: Path, **_kwargs: Any) -> ProcResult:
            kernel = (Path(_kwargs["cwd"]) / "kernel.c").read_text(encoding="utf-8").replace("#include <stdint.h>\n", "")
            Path(stdout_path).write_text(PREPROCESSED_STDINT.format(kernel=kernel), encoding="utf-8")
            return ProcResult(rc=0, timed_out=False, elapsed_s=0.0, stdout_tail="", stderr_tail="", signal_name=None)

        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp, mock.patch.object(extract_mod, "run_isolated", fake_cpp):
            for task, code in (("scan", self.SCAN), ("spmv_csr", self.SPMV)):
                with self.subTest(task=task):
                    self.assertEqual(self.check(task, "#include <stdint.h>\n" + code, work_dir=Path(tmp) / task), self.EXPECTED[task])

    @requires_tool("gcc")
    def test_real_preprocessor(self) -> None:
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            for task, code in (("scan", self.SCAN), ("spmv_csr", self.SPMV)):
                with self.subTest(task=task):
                    self.assertEqual(self.check(task, "#include <stdint.h>\n" + code, work_dir=Path(tmp) / task), self.EXPECTED[task])

    def test_other_sizes_and_signedness_stay_mismatches(self) -> None:
        for spelling in ("unsigned", "unsigned int", "uint32_t", "long", "long int", "int64_t", "size_t", "ssize_t", "short",
                         "int_fast32_t"):
            with self.subTest(spelling=spelling):
                code = f"void scan(const int* x, int* out, {spelling} n) {{ }}\n"
                self.assertEqual(self.check("scan", code), (False, False, False,
                                                                  f"param_mismatch:param 3 (n) type {spelling} expected int"))
        code = "void scan(const uint32_t* x, int* out, int n) { }\n"
        self.assertEqual(self.check("scan", code), (False, False, False, "param_mismatch:param 1 (x) type uint32_t* expected int*"))
        code = "typedef unsigned int u32;\nvoid scan(const u32* x, int* out, int n) { }\n"
        self.assertEqual(self.check("scan", code), (False, False, False, "param_mismatch:param 1 (x) type unsigned int* expected int*"))
        code = "typedef long int32_t;\nvoid scan(const int* x, int* out, int32_t n) { }\n"  # 소스의 typedef 가 먼저다
        self.assertEqual(self.check("scan", code), (False, False, False, "param_mismatch:param 3 (n) type long expected int"))

    def test_keyword_mode_is_unchanged(self) -> None:
        # keyword 모드의 매개변수는 하네스가 만든 구조체의 포인터 하나다
        ok = '#include "kernel_args.h"\nvoid scan(const scan_args* args) { args->out[0] = args->x[0] + args->n; }\n'
        self.assertEqual(self.check("scan", ok, "keyword"), (True, True, True, ""))
        bad = "void scan(const int32_t* args) { }\n"
        self.assertEqual(self.check("scan", bad, "keyword"),
                         (False, False, False, "param_mismatch:param 1 (args) type int32_t* expected scan_args*"))


class PreprocessorChoiceTest(unittest.TestCase):
    """``extract(cc=...)``: C 전처리에 설정의 컴파일러를 쓴다. 주지 않으면 예전처럼 ``gcc`` 다."""

    def run_with(self, **kwargs: Any) -> list[str]:
        seen: list[list[str]] = []

        def fake_cpp(cmd: list[str], *, stdout_path: Path, cwd: Path, **_kwargs: Any) -> ProcResult:
            seen.append(list(cmd))
            Path(stdout_path).write_text('# 1 "kernel.c"\n' + (Path(cwd) / "kernel.c").read_text(encoding="utf-8"), encoding="utf-8")
            return ProcResult(rc=0, timed_out=False, elapsed_s=0.0, stdout_tail="", stderr_tail="", signal_name=None)

        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp, mock.patch.object(extract_mod, "run_isolated", fake_cpp):
            res = c("```c\n" + C_OK + "```", work_dir=Path(tmp), **kwargs)
        self.assertEqual(verdict(res), (True, True, True, ""))
        self.assertEqual(len(seen), 1)
        return seen[0]

    def test_default_is_gcc_and_cc_overrides_it(self) -> None:
        self.assertEqual(self.run_with(), ["gcc", "-E", "-std=gnu11", "-fopenmp", "kernel.c"])
        self.assertEqual(self.run_with(cc=None)[0], "gcc")
        self.assertEqual(self.run_with(cc="clang-18"), ["clang-18", "-E", "-std=gnu11", "-fopenmp", "kernel.c"])


# ----------------------------------------------------------------------------- 설계 v3 §6: Rust

RS_SIG = 'pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32)'
RS_BODY = "{\n    *out = 0.0;\n}\n"

RS_DOT = '''use std::slice;

/// Dot product with a double-precision accumulator.
#[no_mangle]
pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
    let len = n as usize;
    let (xs, ys) = (slice::from_raw_parts(x, len), slice::from_raw_parts(y, len));
    let mut acc = 0.0f64;
    for (a, b) in xs.iter().zip(ys) {
        acc += *a as f64 * *b as f64;
    }
    *out = acc as f32;
}
'''

# rustfmt 가 긴 시그니처를 나누는 꼴(매개변수마다 한 줄, 끝에 쉼표) + 크레이트 속성 + rayon
RS_GEMM = '''#![allow(non_snake_case)]

use rayon::prelude::*;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn gemm(
    A: *const f32,
    B: *const f32,
    C: *mut f32,
    M: i32,
    N: i32,
    K: i32,
) {
    let (m, n, k) = (M as usize, N as usize, K as usize);
    let a = slice::from_raw_parts(A, m * k);
    let b = slice::from_raw_parts(B, k * n);
    let c = slice::from_raw_parts_mut(C, m * n);
    c.par_chunks_mut(n).zip(a.par_chunks(k)).for_each(|(c_row, a_row)| {
        c_row.fill(0.0);
        for (p, &a_ip) in a_row.iter().enumerate() {
            for (c_ij, &b_pj) in c_row.iter_mut().zip(&b[p * n..(p + 1) * n]) {
                *c_ij += a_ip * b_pj;
            }
        }
    });
}
'''


class RustStripTest(unittest.TestCase):
    def test_comments_strings_and_chars(self) -> None:
        cases = {
            "a // line\nb": "a  \nb",
            "a /* block */ b": "a   b",
            "a /* outer /* nested */ still */ b": "a   b",
            "a /* never closed": "a  ",
            'let s = "text // not a comment";': 'let s = "";',
            'let s = "quote \\" inside";': 'let s = "";',
            'let s = b"bytes"; let t = c"cstr";': 'let s = b""; let t = c"";',
            'let s = r"raw \\"; x': 'let s = ""; x',
            'let s = r#"raw "quoted" /* text */"#; x': 'let s = ""; x',
            'let s = br##"a"# b"##; x': 'let s = ""; x',
            "let c = '\"'; let d = '\\''; let e = '{';": "let c = ''; let d = ''; let e = '';",
            "fn first<'a>(v: &'a [f32]) -> &'a f32": "fn first<'a>(v: &'a [f32]) -> &'a f32",
            "'outer: loop { break 'outer; }": "'outer: loop { break 'outer; }",
            'extern "C" fn f() {}': 'extern "C" fn f() {}',          # ABI 문자열은 남는다
            'extern "system" fn f() {}': 'extern "" fn f() {}',
            "let r#type = 1;": "let r#type = 1;",                     # 원시 식별자는 문자열이 아니다
        }
        for src, expected in cases.items():
            with self.subTest(src=src):
                self.assertEqual(_strip_rust(src), expected)

    def test_expected_signature_parser(self) -> None:
        params = parse_rust_signature(bundle_for("dot", "rust", "rust")["signature"], "dot")
        self.assertEqual([(p.name, p.type, p.mutable) for p in params],
                         [("x", "*const f32", False), ("y", "*const f32", False), ("out", "*mut f32", False), ("n", "i32", False)])
        params = parse_rust_signature(bundle_for("dot", "rust", "rust", "keyword")["signature"], "dot")
        self.assertEqual([(p.name, p.type) for p in params], [("args", "*const DotArgs")])
        with self.assertRaises(ValueError):
            parse_rust_signature("def dot(x, y, out, n):", "dot")


class RustSignatureTest(unittest.TestCase):
    BUNDLE = bundle_for("dot", "rust", "rust")

    def rs(self, code: str, tag: str = "rust") -> ExtractResult:
        return extract(fenced(tag, code), backend="rust", signature_spec={}, bundle=self.BUNDLE)

    def test_expected_signature(self) -> None:
        self.assertEqual(self.BUNDLE["signature"], "#[no_mangle]\n" + RS_SIG)

    def test_realistic_kernels_are_exact(self) -> None:
        self.assertEqual(verdict(self.rs(RS_DOT)), (True, True, True, ""))
        res = extract(fenced("rust", RS_GEMM), backend="rust", signature_spec={}, bundle=bundle_for("gemm", "rust", "rust"))
        self.assertEqual(verdict(res), (True, True, True, ""))
        self.assertEqual(res.code, RS_GEMM + "\n")

    def test_whitespace_in_types_is_ignored(self) -> None:
        src = ('#[no_mangle]\npub  unsafe  extern "C"  fn  dot (\n    x : * const f32,\n    y:*const f32,\n'
               "    out: *mut   f32,\n    n: i32,\n) " + RS_BODY)
        self.assertEqual(verdict(self.rs(src)), (True, True, True, ""))

    def test_accepted_but_inexact_forms(self) -> None:
        cases = {
            "no unsafe": f'#[no_mangle]\npub extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "unit return": f"#[no_mangle]\n{RS_SIG} -> () {RS_BODY}",
            "unsafe(no_mangle)": f"#[unsafe(no_mangle)]\n{RS_SIG} {RS_BODY}",
            "unsafe(no_mangle) spaced": f"#[ unsafe( no_mangle ) ]\n{RS_SIG} {RS_BODY}",
            "mut bindings": ('#[no_mangle]\npub unsafe extern "C" fn dot(mut x: *const f32, mut y: *const f32, out: *mut f32, n: i32) '
                             + RS_BODY),
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (True, True, False, ""))

    def test_other_attributes_may_surround_no_mangle(self) -> None:
        cases = {
            "after": f"#[no_mangle]\n#[inline(never)]\n#[allow(clippy::missing_safety_doc)]\n{RS_SIG} {RS_BODY}",
            "before": f'#[cfg(target_arch = "x86_64")]\n#[doc = "bracket ] in a string"]\n#[no_mangle]\n{RS_SIG} {RS_BODY}',
            "doc comments between": f"#[no_mangle]\n/// Safety: the pointers are valid.\n// plain comment\n{RS_SIG} {RS_BODY}",
            "same line": f"#[no_mangle] {RS_SIG} {RS_BODY}",
            "crate attribute first": f"#![allow(non_snake_case)]\n#[no_mangle]\n{RS_SIG} {RS_BODY}",
            "nested attribute arguments": (f'#[target_feature(enable = "avx2,fma")]\n#[no_mangle]\n#[cfg_attr(all(unix, not(test)), inline)]\n'
                                           f"{RS_SIG} {RS_BODY}"),
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (True, True, True, ""))

    def test_no_mangle_missing(self) -> None:
        cases = {
            "no attribute": f"{RS_SIG} {RS_BODY}",
            "other attributes only": f"#[inline]\n#[allow(unused)]\n{RS_SIG} {RS_BODY}",
            "commented out": f"// #[no_mangle]\n{RS_SIG} {RS_BODY}",
            "on another function": f'#[no_mangle]\npub extern "C" fn helper() {{}}\n\n{RS_SIG} {RS_BODY}',
            "on another item": f"#[no_mangle]\nstatic CALLS: i32 = 0;\n{RS_SIG} {RS_BODY}",
            "crate attribute only": f"#![allow(non_snake_case)]\n{RS_SIG} {RS_BODY}",
            "inside cfg_attr": f"#[cfg_attr(unix, no_mangle)]\n{RS_SIG} {RS_BODY}",
            "export_name": f'#[export_name = "dot"]\n{RS_SIG} {RS_BODY}',
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (False, False, False, "no_mangle_missing"))

    def test_function_not_found(self) -> None:
        cases = {
            "extern block declaration": 'extern "C" {\n    pub fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32);\n}\n',
            "definition inside an extern block": f'extern "C" {{\n    #[no_mangle]\n    {RS_SIG} {RS_BODY}}}\n',
            "inside an unsafe extern block": f'unsafe extern "C" {{\n    #[no_mangle]\n    {RS_SIG} {RS_BODY}}}\n',
            "generic": f'#[no_mangle]\npub unsafe extern "C" fn dot<T>(x: *const T, y: *const T, out: *mut T, n: i32) {RS_BODY}',
            "declaration without a body": f"#[no_mangle]\n{RS_SIG};\n",
            "rust abi": f"#[no_mangle]\npub unsafe fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}",
            "other abi": f'#[no_mangle]\npub unsafe extern "system" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "implicit abi": f"#[no_mangle]\npub unsafe extern fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}",
            "unwind abi": f'#[no_mangle]\npub unsafe extern "C-unwind" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "not pub": f'#[no_mangle]\nunsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "restricted pub": f'#[no_mangle]\npub(crate) unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "other name": f'#[no_mangle]\npub unsafe extern "C" fn dot_product(x: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}',
            "in a line comment": f"// #[no_mangle]\n// {RS_SIG} {{ }}\nfn helper() {{}}\n",
            "in a nested block comment": f"/* outer /* inner */ #[no_mangle]\n{RS_SIG} {RS_BODY} */\nfn helper() {{}}\n",
            "in a string": ('const DOC: &str = "#[no_mangle] pub unsafe extern \\"C\\" fn dot(x: *const f32, y: *const f32, '
                            'out: *mut f32, n: i32) { }";\nfn helper() {}\n'),
            "in a raw string": f'const DOC: &str = r#"#[no_mangle]\n{RS_SIG} {{ }}"#;\nfn helper() {{}}\n',
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (False, False, False, "function_not_found"))

    def test_definition_after_an_extern_block_is_found(self) -> None:
        src = ('extern "C" {\n    fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;\n}\n\n'
               f"#[no_mangle]\n{RS_SIG} {RS_BODY}")
        self.assertEqual(verdict(self.rs(src)), (True, True, True, ""))

    def test_definition_among_other_items_is_found(self) -> None:
        cases = {
            "function pointer type before it": f'fn apply(f: extern "C" fn(i32) -> i32, v: i32) -> i32 {{ f(v) }}\n#[no_mangle]\n{RS_SIG} {RS_BODY}',
            "inside a module": f"pub mod ffi {{\n    #[no_mangle]\n    {RS_SIG} {RS_BODY}}}\n",
            "one definition per target": (f'#[cfg(target_arch = "x86_64")]\n#[no_mangle]\n{RS_SIG} {RS_BODY}\n'
                                          f'#[cfg(not(target_arch = "x86_64"))]\n#[no_mangle]\n{RS_SIG} {RS_BODY}'),
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (True, True, True, ""))

    def test_param_mismatch(self) -> None:
        head = '#[no_mangle]\npub unsafe extern "C" fn dot'
        cases = {
            f"{head}(x: *const f32, y: *const f32, out: *mut f32) {RS_BODY}": "param_mismatch:3 params expected 4",
            f"{head}() {RS_BODY}": "param_mismatch:0 params expected 4",
            f"{head}(a: *const f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}": "param_mismatch:param 1 name a expected x",
            f"{head}(y: *const f32, x: *const f32, out: *mut f32, n: i32) {RS_BODY}": "param_mismatch:param 1 name y expected x",
            f"{head}(x: *const f32, y: *const f32, out: *mut f32, _n: i32) {RS_BODY}": "param_mismatch:param 4 name _n expected n",
            f"{head}(x: *const f64, y: *const f32, out: *mut f32, n: i32) {RS_BODY}":
                "param_mismatch:param 1 (x) type *const f64 expected *const f32",
            f"{head}(x: *mut f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}":
                "param_mismatch:param 1 (x) type *mut f32 expected *const f32",
            f"{head}(x: &[f32], y: &[f32], out: *mut f32, n: i32) {RS_BODY}":
                "param_mismatch:param 1 (x) type &[f32] expected *const f32",
            f"{head}(x: *const f32, y: *const f32, out: *mut f32, n: usize) {RS_BODY}":
                "param_mismatch:param 4 (n) type usize expected i32",
            # c_int·c_float 밖의 별칭은 풀지 않는다(글자 비교; RustFfiAliasTest): 부호가 다른 `c_uint` 는 다른 타입이다
            f"{head}(x: *const f32, y: *const f32, out: *mut f32, n: std::ffi::c_uint) {RS_BODY}":
                "param_mismatch:param 4 (n) type std::ffi::c_uint expected i32",
            f"{head}(x: *const f32, y: *const f32, out: *mut f32, n: i32) -> f32 {{ 0.0 }}":
                "param_mismatch:return_type f32 expected ()",
        }
        for src, detail in cases.items():
            with self.subTest(detail=detail):
                self.assertEqual(verdict(self.rs(src)), (False, False, False, detail))

    def test_comments_and_literals_inside_the_source(self) -> None:
        cases = {
            "comments in the signature": ('#[no_mangle] // exported\npub unsafe extern "C" fn dot(x: *const f32, /* in */ y: *const f32,\n'
                                          "    out: *mut f32 /* out */, n: i32) " + RS_BODY),
            "quote characters": "fn q(c: char) -> bool { c == '\"' || c == '\\'' || c == '(' }\n#[no_mangle]\n" + RS_SIG + " " + RS_BODY,
            "lifetimes": "fn first<'a>(v: &'a [f32]) -> &'a f32 { &v[0] }\n#[no_mangle]\n" + RS_SIG + " " + RS_BODY,
            "braces and brackets in strings": 'const S: &str = "}{ ]( extern";\n#[no_mangle]\n' + RS_SIG + " " + RS_BODY,
            "fake definition in a comment first": f"// pub unsafe extern \"C\" fn dot() {{}}\n#[no_mangle]\n{RS_SIG} {RS_BODY}",
        }
        for name, src in cases.items():
            with self.subTest(name):
                self.assertEqual(verdict(self.rs(src)), (True, True, True, ""))

    def test_block_selection_and_bare_code(self) -> None:
        toml = '```toml\n[dependencies]\nrayon = "1"\n```\n'
        res = extract(toml + fenced("rust", RS_DOT), backend="rust", signature_spec={}, bundle=self.BUNDLE)
        self.assertEqual((verdict(res), res.n_blocks, res.code), ((True, True, True, ""), 2, RS_DOT + "\n"))
        # 정의가 있는 블록이 더 긴 블록(도움 함수만)보다 우선한다
        helper = "```rust\n// helpers\nfn unused_helper(v: &[f32]) -> f32 { v.iter().sum() }\n" + "// filler\n" * 40 + "```\n"
        res = extract(helper + fenced("rs", RS_DOT), backend="rust", signature_spec={}, bundle=self.BUNDLE)
        self.assertEqual((res.ok, res.detail, res.code), (True, "multiple_blocks_chose:1", RS_DOT + "\n"))
        res = extract(RS_DOT, backend="rust", signature_spec={}, bundle=self.BUNDLE)
        self.assertEqual((verdict(res), res.n_blocks), ((True, True, True, "bare_code"), 0))
        res = extract("```rust\n" + RS_DOT, backend="rust", signature_spec={}, bundle=self.BUNDLE)
        self.assertEqual(verdict(res), (True, True, True, "unterminated_fence"))
        res = extract("I cannot write this kernel.", backend="rust", signature_spec={}, bundle=self.BUNDLE)
        self.assertEqual((res.ok, res.code, res.detail), (False, None, "no_code_block"))

    def test_rust_start_tokens_count_only_for_the_rust_backend(self) -> None:
        # 펜스 없는 응답의 첫 줄 규칙(2026-11-08)은 c·python 에서 그대로다: Rust 토큰으로 시작하면 코드로 보지 않는다
        for first in ("use std::slice;", "#![allow(non_snake_case)]", "#[no_mangle]", "pub fn helper() {}", "fn helper() {}",
                      "unsafe fn helper() {}", "mod kernel_args;"):
            with self.subTest(first=first):
                self.assertEqual(c(first + "\n" + C_OK).detail, "no_code_block")
                self.assertEqual(py(first + "\n" + PY_OK).detail, "no_code_block")
                body = RS_DOT if first.startswith("use ") else first + "\n" + RS_DOT
                res = extract(body, backend="rust", signature_spec={}, bundle=self.BUNDLE)
                self.assertEqual((res.ok, res.detail), (True, "bare_code"))


class RustKeywordTest(unittest.TestCase):
    """Rust keyword 모드: ``args: *const {Camel}Args``. 타입을 경로로 쓴 것은 받되 exact 가 아니다."""

    BUNDLE = bundle_for("dot", "rust", "rust", "keyword")
    HEAD = '#[no_mangle]\npub unsafe extern "C" fn dot'

    def kw(self, code: str) -> ExtractResult:
        return extract(fenced("rust", code), backend="rust", signature_spec={}, args_mode="keyword", bundle=self.BUNDLE)

    def test_exact_form(self) -> None:
        self.assertEqual(self.BUNDLE["signature"], '#[no_mangle]\npub unsafe extern "C" fn dot(args: *const DotArgs)')
        src = ("mod kernel_args;\nuse kernel_args::DotArgs;\n\n" + self.HEAD + "(args: *const DotArgs) {\n"
               "    let a = &*args;\n    *a.out = 0.0;\n}\n")
        self.assertEqual(verdict(self.kw(src)), (True, True, True, ""))

    def test_path_spellings_are_accepted_but_inexact(self) -> None:
        for type_text in ("*const kernel_args::DotArgs", "*const crate::kernel_args::DotArgs", "*const self::kernel_args::DotArgs",
                          "* const kernel_args :: DotArgs"):
            with self.subTest(type_text=type_text):
                self.assertEqual(verdict(self.kw(f"mod kernel_args;\n{self.HEAD}(args: {type_text}) {RS_BODY}")), (True, True, False, ""))

    def test_rejected_forms(self) -> None:
        cases = {
            "(args: *const kernel_args::GemmArgs)": "param_mismatch:param 1 (args) type *const kernel_args::GemmArgs expected *const DotArgs",
            "(args: *mut DotArgs)": "param_mismatch:param 1 (args) type *mut DotArgs expected *const DotArgs",
            "(args: *mut kernel_args::DotArgs)": "param_mismatch:param 1 (args) type *mut kernel_args::DotArgs expected *const DotArgs",
            "(args: DotArgs)": "param_mismatch:param 1 (args) type DotArgs expected *const DotArgs",
            "(args: &DotArgs)": "param_mismatch:param 1 (args) type &DotArgs expected *const DotArgs",
            "(args: *const ::DotArgs)": "param_mismatch:param 1 (args) type *const ::DotArgs expected *const DotArgs",
            "(a: *const DotArgs)": "param_mismatch:param 1 name a expected args",
            "(x: *const f32, y: *const f32, out: *mut f32, n: i32)": "param_mismatch:4 params expected 1",
        }
        for params, detail in cases.items():
            with self.subTest(params=params):
                self.assertEqual(verdict(self.kw(f"{self.HEAD}{params} {RS_BODY}")), (False, False, False, detail))

    def test_path_spelling_is_not_accepted_in_positional_mode(self) -> None:
        src = f'#[no_mangle]\npub unsafe extern "C" fn dot(x: *const core::primitive::f32, y: *const f32, out: *mut f32, n: i32) {RS_BODY}'
        res = extract(fenced("rust", src), backend="rust", signature_spec={}, bundle=bundle_for("dot", "rust", "rust"))
        self.assertEqual(verdict(res), (False, False, False,
                                        "param_mismatch:param 1 (x) type *const core::primitive::f32 expected *const f32"))


class RustFfiAliasTest(unittest.TestCase):
    """ABI 가 같은 FFI 별칭 ``c_int`` = ``i32``, ``c_float`` = ``f32`` (접두 없음 또는 ``std::ffi::``·``core::ffi::``·
    ``std::os::raw::``·``libc::``; 값·포인터 자리)는 통과하고 exact 만 False 다. 다른 별칭은 그대로 param_mismatch 다."""

    HEAD = '#[no_mangle]\npub unsafe extern "C" fn '
    PREFIXES = ("", "std::ffi::", "core::ffi::", "std::os::raw::", "libc::")

    def check(self, task: str, params: str, args_mode: str = "positional") -> tuple[bool, bool | None, bool | None, str]:
        code = f"use std::ffi::{{c_float, c_int}};\n\n{self.HEAD}{task}({params}) {{\n}}\n"
        bundle = bundle_for(task, "rust", "rust", args_mode)
        return verdict(extract(fenced("rust", code), backend="rust", signature_spec={}, args_mode=args_mode, bundle=bundle))

    def test_aliases_pass_without_exact(self) -> None:
        for p in self.PREFIXES:
            with self.subTest(prefix=p):
                # scan: int32 배열(포인터 자리)과 dim(값 자리)
                self.assertEqual(self.check("scan", f"x: *const {p}c_int, out: *mut {p}c_int, n: {p}c_int"), (True, True, False, ""))
                # spmv_csr: int32·float32 배열이 섞인 서명, 공백이 있는 표기
                params = (f"row_ptr: *const {p}c_int, col_idx: * const {p}c_int, vals: *const {p}c_float, x: *const f32, "
                          f"y: *mut {p}c_float, n_rows: {p}c_int, n_cols: i32, nnz: {p} c_int")
                self.assertEqual(self.check("spmv_csr", params), (True, True, False, ""))
        self.assertEqual(self.check("scan", "x: *const i32, out: *mut i32, n: i32"), (True, True, True, ""))

    def test_other_aliases_and_wrong_pointers_stay_mismatches(self) -> None:
        cases = {
            "c_uint": "c_uint", "std::ffi::c_long": "std::ffi::c_long", "usize": "usize", "isize": "isize",
            "core::ffi::c_double": "core::ffi::c_double", "c_float": "c_float", "std::ffi::c_int32": "std::ffi::c_int32",
            "other::c_int": "other::c_int",
        }
        for spelling, shown in cases.items():
            with self.subTest(spelling=spelling):
                self.assertEqual(self.check("scan", f"x: *const i32, out: *mut i32, n: {spelling}"),
                                 (False, False, False, f"param_mismatch:param 3 (n) type {shown} expected i32"))
        # 포인터의 const/mut, 값과 포인터는 별칭이어도 구분한다
        self.assertEqual(self.check("scan", "x: *mut c_int, out: *mut i32, n: i32"),
                         (False, False, False, "param_mismatch:param 1 (x) type *mut c_int expected *const i32"))
        self.assertEqual(self.check("scan", "x: c_int, out: *mut i32, n: i32"),
                         (False, False, False, "param_mismatch:param 1 (x) type c_int expected *const i32"))
        self.assertEqual(self.check("scan", "x: *const c_float, out: *mut i32, n: i32"),
                         (False, False, False, "param_mismatch:param 1 (x) type *const c_float expected *const i32"))

    def test_keyword_mode(self) -> None:
        # keyword 모드의 매개변수는 구조체 포인터 하나다: 경로 표기 규칙은 그대로이고, 별칭은 구조체 타입이 아니다
        self.assertEqual(self.check("scan", "args: *const ScanArgs", "keyword"), (True, True, True, ""))
        self.assertEqual(self.check("scan", "args: *const kernel_args::ScanArgs", "keyword"), (True, True, False, ""))
        self.assertEqual(self.check("scan", "args: *const c_int", "keyword"),
                         (False, False, False, "param_mismatch:param 1 (args) type *const c_int expected *const ScanArgs"))


class CliTest(unittest.TestCase):
    """진단용 CLI 는 과제의 ``signatures.yaml`` 에서 시그니처 묶음을 만들어 run.py 와 같은 검사를 한다."""

    def run_cli(self, text: str, *args: str, suffix: str = ".md") -> tuple[int, dict[str, Any]]:
        with tempfile.TemporaryDirectory(prefix="extract_test_") as tmp:
            path = Path(tmp) / f"response{suffix}"
            path.write_text(text, encoding="utf-8")
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main(["--file", str(path), *args])
        return rc, json.loads(stdout.getvalue())

    def test_exit_code_follows_the_verdict(self) -> None:
        dot = str(ROOT / "tasks" / "dot" / "signatures.yaml")
        rc, doc = self.run_cli(fenced("rust", RS_DOT), "--backend", "rust", "--signatures", dot)
        self.assertEqual((rc, doc["ok"], doc["signature_exact"], doc["detail"], doc["code"]), (0, True, True, "", f"<{len(RS_DOT) + 1} chars>"))
        rc, doc = self.run_cli(fenced("rust", RS_DOT.replace("#[no_mangle]\n", "")), "--backend", "rust", "--signatures", dot)
        self.assertEqual((rc, doc["ok"], doc["signature_ok"], doc["detail"]), (1, False, False, "no_mangle_missing"))

    def test_task_language_and_args_mode_select_the_expected_signature(self) -> None:
        # 기본 과제는 gemm. --language 는 YAML 의 언어 항목을 고른다(없으면 backend 이름 = 렌더링한 시그니처)
        rc, doc = self.run_cli(fenced("c", C_OK), "--backend", "c", "--language", "c_seq", "--print-code")
        self.assertEqual((rc, doc["signature_exact"], doc["code"]), (0, True, C_OK + "\n"))
        rc, doc = self.run_cli(fenced("c", C_KEYWORD_OK), "--backend", "c", "--args-mode", "keyword")
        self.assertEqual((rc, doc["signature_exact"]), (0, True))
        rc, doc = self.run_cli(fenced("c", C_KEYWORD_OK), "--backend", "c")
        self.assertEqual((rc, doc["detail"]), (1, "param_mismatch:1 params expected 6"))
        rc, doc = self.run_cli(fenced("python", HALIDE_GEMM), "--backend", "halide", "--language", "halide_manual")
        self.assertEqual((rc, doc["signature_ok"]), (0, True))
        rc, doc = self.run_cli(fenced("python", NUMBA_GEMM), "--backend", "halide")
        self.assertEqual((rc, doc["detail"]), (1, "param_mismatch:got (A, B, C, M, N, K) expected (A, B, M, N, K)"))

    def test_raw_json_input(self) -> None:
        raw = json.dumps({"response_text": fenced("python", PY_OK), "stop_reason": "end_turn"})
        rc, doc = self.run_cli(raw, "--backend", "python", "--language", "numpy", suffix=".json")
        self.assertEqual((rc, doc["ok"], doc["n_blocks"]), (0, True, 1))


if __name__ == "__main__":
    unittest.main()
