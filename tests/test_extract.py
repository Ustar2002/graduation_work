"""extract 테스트: 펜스 수집·후보 선택(태그/무태그/다중/맨 코드/미종료), python·C 시그니처 검사, gcc -E 경로, 더미 응답 기대."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from harness.extract import ExtractResult, collect_typedefs, extract, parse_c_param, parse_c_prototype, parse_fenced_blocks
from harness.paths import ROOT

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
            for md in sorted((self.DUMMY / lang).glob("*.md")):
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


if __name__ == "__main__":
    unittest.main()
