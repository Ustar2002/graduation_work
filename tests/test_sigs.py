"""sigs 테스트 (설계 v3 §3): 시그니처 렌더링의 정확한 문자열(백엔드 5개 × 인자 방식 2개), YAML 우선 규칙,
기존 4개 과제의 C·Python 시그니처와의 문자 단위 일치, Numba 서명, 매개변수 구조체, 매개변수 설명,
keyword 모드의 구조체 안내 문단(YAML 설명에도 붙는다; Rust 는 rustc 로 스코프 문장을 확인), 진단용 CLI."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import yaml

from harness import sigs
from harness.paths import ROOT
from harness.procutil import ProcResult, run_isolated
from harness.sigs import (
    ARGS_MODES,
    BACKENDS,
    DIM_DEFAULT_DOC,
    SCALAR_DEFAULT_DOC,
    c_type,
    camel,
    halide_params,
    keyword_struct_intro,
    numba_signature,
    numba_type,
    render_args_struct,
    render_param_docs,
    render_signature,
    rust_type,
    signature_bundle,
)
from harness.taskspec import ArgSpec, TaskSpec, load_taskspec, parse_taskspec
from tests._support import requires_tool

TASKS = ROOT / "tasks"
EXISTING_TASKS = ("gemm", "jacobi2d", "spmv_csr", "dot")
BUNDLE_KEYS = {"function", "params", "signature", "param_docs", "args_struct", "numba_signature", "source"}

# 수치 라이브러리·런타임 이름: 매개변수 설명에 나오면 안 된다(라이브러리 사용 여부는 관찰 대상).
LIBRARY_NAMES = ("numpy", "scipy", "blas", "lapack", "mkl", "eigen", "fftw", "rayon", "openmp", "libgomp", "numba")


def spec_of(task: str):
    return load_taskspec(TASKS / task / "signatures.yaml")


def raw_yaml(task: str) -> dict:
    return yaml.safe_load((TASKS / task / "signatures.yaml").read_text(encoding="utf-8"))


# YAML languages 가 없는 정수 출력 과제(설계 v3 §2.1 의 예)
HISTOGRAM = parse_taskspec({
    "task": "histogram", "function": "histogram",
    "args": [
        {"name": "keys", "kind": "in", "dtype": "int32", "shape": ["n"], "doc": "each value is in the range 0 <= keys[i] < n_bins"},
        {"name": "hist", "kind": "out", "dtype": "int32", "shape": ["n_bins"]},
        {"name": "n", "kind": "dim"},
        {"name": "n_bins", "kind": "dim", "doc": "the number of bins."},
    ],
})
NBODY = parse_taskspec({
    "task": "nbody", "function": "nbody",
    "args": [
        {"name": "pos", "kind": "in", "dtype": "float32", "shape": ["n", 3]},
        {"name": "mass", "kind": "in", "dtype": "float32", "shape": ["n"]},
        {"name": "acc", "kind": "out", "dtype": "float32", "shape": ["n", 3]},
        {"name": "n", "kind": "dim"},
    ],
})

GEMM_SIGNATURES = {
    ("python", "positional"): "def gemm(A, B, C, M, N, K):",
    ("python", "keyword"): "def gemm(*, A, B, C, M, N, K):",
    ("numba", "positional"): '@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)")\n'
                             "def gemm(A, B, C, M, N, K):",
    ("numba", "keyword"): '@njit("void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)")\n'
                          "def gemm(*, A, B, C, M, N, K):",
    ("halide", "positional"): "def gemm(A, B, M, N, K):",
    ("halide", "keyword"): "def gemm(*, A, B, M, N, K):",
    ("c", "positional"): "void gemm(const float* A, const float* B, float* C, int M, int N, int K)",
    ("c", "keyword"): "void gemm(const gemm_args* args)",
    ("rust", "positional"): '#[no_mangle]\npub unsafe extern "C" fn gemm(A: *const f32, B: *const f32, C: *mut f32, M: i32, N: i32, K: i32)',
    ("rust", "keyword"): '#[no_mangle]\npub unsafe extern "C" fn gemm(args: *const GemmArgs)',
}

_SPMV_P = "row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz"
_SPMV_NUMBA = "void(int32[::1], int32[::1], float32[::1], float32[::1], float32[::1], int64, int64, int64)"
SPMV_SIGNATURES = {
    ("python", "positional"): f"def spmv_csr({_SPMV_P}):",
    ("python", "keyword"): f"def spmv_csr(*, {_SPMV_P}):",
    ("numba", "positional"): f'@njit("{_SPMV_NUMBA}")\ndef spmv_csr({_SPMV_P}):',
    ("numba", "keyword"): f'@njit("{_SPMV_NUMBA}")\ndef spmv_csr(*, {_SPMV_P}):',
    ("halide", "positional"): "def spmv_csr(row_ptr, col_idx, vals, x, n_rows, n_cols, nnz):",
    ("halide", "keyword"): "def spmv_csr(*, row_ptr, col_idx, vals, x, n_rows, n_cols, nnz):",
    ("c", "positional"): "void spmv_csr(const int* row_ptr, const int* col_idx, const float* vals, const float* x, float* y, "
                         "int n_rows, int n_cols, int nnz)",
    ("c", "keyword"): "void spmv_csr(const spmv_csr_args* args)",
    ("rust", "positional"): '#[no_mangle]\npub unsafe extern "C" fn spmv_csr(row_ptr: *const i32, col_idx: *const i32, '
                            "vals: *const f32, x: *const f32, y: *mut f32, n_rows: i32, n_cols: i32, nnz: i32)",
    ("rust", "keyword"): '#[no_mangle]\npub unsafe extern "C" fn spmv_csr(args: *const SpmvCsrArgs)',
}


class NamesAndTypesTest(unittest.TestCase):
    def test_constants(self) -> None:
        self.assertEqual(BACKENDS, ("python", "numba", "halide", "c", "rust"))
        self.assertEqual(ARGS_MODES, ("positional", "keyword"))
        self.assertEqual(DIM_DEFAULT_DOC, "always consistent with the array shapes above")

    def test_camel(self) -> None:
        self.assertEqual(camel("spmv_csr"), "SpmvCsr")
        self.assertEqual(camel("gemm"), "Gemm")
        self.assertEqual(camel("jacobi2d"), "Jacobi2d")
        self.assertEqual(camel("floyd_warshall"), "FloydWarshall")
        self.assertEqual(camel("prefix_sum"), "PrefixSum")

    def test_c_and_rust_types(self) -> None:
        cases = [
            (ArgSpec("A", "in", "float32", ("M", "K")), "const float*", "*const f32"),
            (ArgSpec("idx", "in", "int32", ("n",)), "const int*", "*const i32"),
            (ArgSpec("C", "out", "float32", ("M", "N")), "float*", "*mut f32"),
            (ArgSpec("hist", "out", "int32", ("n_bins",)), "int*", "*mut i32"),
            (ArgSpec("M", "dim"), "int", "i32"),
        ]
        for arg, c_expected, rust_expected in cases:
            with self.subTest(arg=arg.name):
                self.assertEqual(c_type(arg), c_expected)
                self.assertEqual(rust_type(arg), rust_expected)

    def test_numba_types_for_1d_2d_int_and_dim(self) -> None:
        spec = spec_of("gemm")
        self.assertEqual(numba_type(spec, ArgSpec("x", "in", "float32", ("n",))), "float32[::1]")
        self.assertEqual(numba_type(spec, ArgSpec("A", "in", "float32", ("M", "K"))), "float32[:, ::1]")
        self.assertEqual(numba_type(spec, ArgSpec("idx", "in", "int32", ("n",))), "int32[::1]")
        self.assertEqual(numba_type(spec, ArgSpec("G", "out", "int32", ("H", "W"))), "int32[:, ::1]")
        self.assertEqual(numba_type(spec, ArgSpec("V", "in", "float32", ("a", "b", "c"))), "float32[:, :, ::1]")
        self.assertEqual(numba_type(spec, ArgSpec("M", "dim")), "int64")

    def test_numba_signature(self) -> None:
        self.assertEqual(numba_signature(spec_of("gemm")), "void(float32[:, ::1], float32[:, ::1], float32[:, ::1], int64, int64, int64)")
        self.assertEqual(numba_signature(spec_of("spmv_csr")), _SPMV_NUMBA)
        self.assertEqual(numba_signature(spec_of("dot")), "void(float32[::1], float32[::1], float32[::1], int64)")
        self.assertEqual(numba_signature(spec_of("jacobi2d")), "void(float32[:, ::1], float32[:, ::1], int64, int64, int64)")
        self.assertEqual(numba_signature(HISTOGRAM), "void(int32[::1], int32[::1], int64, int64)")

    @unittest.skipIf(importlib.util.find_spec("numba") is None, "numba 가 설치돼 있지 않다")
    def test_numba_accepts_the_signature_strings(self) -> None:
        """렌더링한 서명 문자열을 Numba 가 해석한다(컴파일 없이 서명만 정규화)."""
        from numba.core import sigutils, types

        for spec in (spec_of("gemm"), spec_of("spmv_csr"), HISTOGRAM, NBODY):
            with self.subTest(task=spec.task):
                args, ret = sigutils.normalize_signature(numba_signature(spec))
                self.assertEqual(ret, types.void)
                self.assertEqual(len(args), len(spec.args))
                for nb_type, arg in zip(args, spec.args):
                    if arg.kind == "dim":
                        self.assertEqual(nb_type, types.int64)
                    else:
                        self.assertEqual((nb_type.ndim, nb_type.layout, str(nb_type.dtype)), (len(arg.shape), "C", arg.dtype))

    def test_halide_params_have_no_output(self) -> None:
        self.assertEqual(halide_params(spec_of("gemm")), ["A", "B", "M", "N", "K"])
        self.assertEqual(halide_params(spec_of("spmv_csr")), ["row_ptr", "col_idx", "vals", "x", "n_rows", "n_cols", "nnz"])
        self.assertEqual(halide_params(HISTOGRAM), ["keys", "n", "n_bins"])


class RenderSignatureTest(unittest.TestCase):
    def test_every_backend_and_args_mode_for_gemm(self) -> None:
        spec = spec_of("gemm")
        self.assertEqual(set(GEMM_SIGNATURES), {(b, m) for b in BACKENDS for m in ARGS_MODES})
        for (backend, mode), expected in GEMM_SIGNATURES.items():
            with self.subTest(backend=backend, mode=mode):
                self.assertEqual(render_signature(spec, backend, mode), expected)

    def test_every_backend_and_args_mode_for_spmv_csr(self) -> None:
        spec = spec_of("spmv_csr")
        self.assertEqual(set(SPMV_SIGNATURES), {(b, m) for b in BACKENDS for m in ARGS_MODES})
        for (backend, mode), expected in SPMV_SIGNATURES.items():
            with self.subTest(backend=backend, mode=mode):
                self.assertEqual(render_signature(spec, backend, mode), expected)

    def test_int_output_task(self) -> None:
        self.assertEqual(render_signature(HISTOGRAM, "c", "positional"), "void histogram(const int* keys, int* hist, int n, int n_bins)")
        self.assertEqual(render_signature(HISTOGRAM, "rust", "positional"),
                         '#[no_mangle]\npub unsafe extern "C" fn histogram(keys: *const i32, hist: *mut i32, n: i32, n_bins: i32)')
        self.assertEqual(render_signature(HISTOGRAM, "halide", "positional"), "def histogram(keys, n, n_bins):")
        self.assertEqual(render_signature(HISTOGRAM, "numba", "positional"),
                         '@njit("void(int32[::1], int32[::1], int64, int64)")\ndef histogram(keys, hist, n, n_bins):')

    def test_c_positional_equals_existing_yaml_character_for_character(self) -> None:
        """기존 4개 과제의 YAML C 시그니처(c_seq, c_openmp)와 렌더링 결과가 문자 단위로 같다."""
        for task in EXISTING_TASKS:
            langs = raw_yaml(task)["languages"]
            rendered = render_signature(spec_of(task), "c", "positional")
            for language in ("c_seq", "c_openmp"):
                with self.subTest(task=task, language=language):
                    self.assertEqual(rendered, langs[language]["positional"])

    def test_python_signatures_equal_existing_yaml(self) -> None:
        for task in EXISTING_TASKS:
            entry = raw_yaml(task)["languages"]["numpy"]
            with self.subTest(task=task):
                self.assertEqual(render_signature(spec_of(task), "python", "positional"), entry["positional"])
                self.assertEqual(render_signature(spec_of(task), "python", "keyword"), entry["keyword"])

    def test_unknown_backend_or_mode_raises(self) -> None:
        spec = spec_of("gemm")
        with self.assertRaises(ValueError):
            render_signature(spec, "fortran", "positional")
        with self.assertRaises(ValueError):
            render_signature(spec, "c", "named")
        with self.assertRaises(ValueError):
            render_param_docs(spec, "c", "named")
        with self.assertRaises(ValueError):
            render_args_struct(spec, "cpp")
        with self.assertRaises(ValueError):
            signature_bundle(spec, "numpy", "numpy", "positional")  # 언어 이름은 backend 가 아니다


class ArgsStructTest(unittest.TestCase):
    def test_c_header_for_gemm(self) -> None:
        self.assertEqual(render_args_struct(spec_of("gemm"), "c"), (
            "#ifndef KERNEL_ARGS_H\n"
            "#define KERNEL_ARGS_H\n"
            "\n"
            "typedef struct {\n"
            "    const float* A;\n"
            "    const float* B;\n"
            "    float* C;\n"
            "    int M;\n"
            "    int N;\n"
            "    int K;\n"
            "} gemm_args;\n"
            "\n"
            "#endif /* KERNEL_ARGS_H */\n"
        ))

    def test_rust_struct_for_gemm(self) -> None:
        self.assertEqual(render_args_struct(spec_of("gemm"), "rust"), (
            "#[repr(C)]\n"
            "#[allow(non_snake_case)]\n"
            "pub struct GemmArgs {\n"
            "    pub A: *const f32,\n"
            "    pub B: *const f32,\n"
            "    pub C: *mut f32,\n"
            "    pub M: i32,\n"
            "    pub N: i32,\n"
            "    pub K: i32,\n"
            "}\n"
        ))

    def test_structs_for_spmv_csr_keep_args_order_and_int_fields(self) -> None:
        header = render_args_struct(spec_of("spmv_csr"), "c")
        self.assertIn(
            "typedef struct {\n    const int* row_ptr;\n    const int* col_idx;\n    const float* vals;\n    const float* x;\n"
            "    float* y;\n    int n_rows;\n    int n_cols;\n    int nnz;\n} spmv_csr_args;\n", header)
        rs = render_args_struct(spec_of("spmv_csr"), "rust")
        self.assertEqual(rs, (
            "#[repr(C)]\n#[allow(non_snake_case)]\npub struct SpmvCsrArgs {\n"
            "    pub row_ptr: *const i32,\n    pub col_idx: *const i32,\n    pub vals: *const f32,\n    pub x: *const f32,\n"
            "    pub y: *mut f32,\n    pub n_rows: i32,\n    pub n_cols: i32,\n    pub nnz: i32,\n}\n"
        ))

    def test_int_output_struct(self) -> None:
        self.assertIn("    const int* keys;\n    int* hist;\n    int n;\n    int n_bins;\n} histogram_args;", render_args_struct(HISTOGRAM, "c"))
        self.assertIn("    pub keys: *const i32,\n    pub hist: *mut i32,\n", render_args_struct(HISTOGRAM, "rust"))

    def test_other_backends_have_no_struct(self) -> None:
        for backend in ("python", "numba", "halide"):
            self.assertEqual(render_args_struct(spec_of("gemm"), backend), "")


class ParamDocsTest(unittest.TestCase):
    def test_python_docs_for_int_task(self) -> None:
        self.assertEqual(render_param_docs(HISTOGRAM, "python", "positional"), "\n".join([
            "- `keys`: array, dtype int32, shape (n,), C-contiguous, 64-byte aligned, read-only; "
            "each value is in the range 0 <= keys[i] < n_bins.",
            "- `hist`: array, dtype int32, shape (n_bins,), C-contiguous, 64-byte aligned; the output buffer that must be filled.",
            "- `n`: Python `int`; always consistent with the array shapes above.",
            "- `n_bins`: Python `int`; the number of bins.",
            "- Returns: `None` (results are written into `hist`).",
        ]))

    def test_numba_docs_use_signature_types(self) -> None:
        docs = render_param_docs(spec_of("gemm"), "numba", "positional")
        self.assertIn("- `A`: array, dtype float32, shape (M, K), C-contiguous, 64-byte aligned, read-only.", docs)
        self.assertIn("- `M`, `N`, `K`: `int64`; always consistent with the array shapes above.", docs)
        self.assertTrue(docs.endswith("- Returns: nothing (results are written into `C`)."))

    def test_scalar_that_is_not_a_shape_dimension_gets_no_shape_claim(self) -> None:
        # doc 이 없고 어떤 형태 식에도 나오지 않는 스칼라(jacobi2d 의 T 같은 것): "형태와 일치한다"는 틀린 문장 대신 명세를 가리킨다.
        # 저장소의 jacobi2d 는 T 에 doc 을 주었으므로(2026-11-21) 기본 문구는 doc 없는 같은 모양의 명세로 본다.
        self.assertEqual(SCALAR_DEFAULT_DOC, "a scalar parameter described in the task specification")
        spec = parse_taskspec({"task": "j", "args": [
            {"name": "src", "kind": "in", "dtype": "float32", "shape": ["H", "W"]},
            {"name": "dst", "kind": "out", "dtype": "float32", "shape": ["H", "W"]},
            {"name": "H", "kind": "dim"}, {"name": "W", "kind": "dim"}, {"name": "T", "kind": "dim"}]})
        self.assertIn("- `T`: Python `int`; the number of steps.", render_param_docs(spec_of("jacobi2d"), "python", "positional"))
        expected = {
            "python": "Python `int`", "numba": "`int64`", "c": "`int`", "rust": "`i32`",
            "halide": "Python `int`, the actual value for this call",
        }
        for backend, type_text in expected.items():
            with self.subTest(backend=backend):
                docs = render_param_docs(spec, backend, "positional")
                self.assertIn(f"- `H`, `W`: {type_text}; always consistent with the array shapes above.", docs)
                self.assertIn(f"- `T`: {type_text}; a scalar parameter described in the task specification.", docs)
        # 형태 식 안의 dim(n_rows + 1 의 n_rows)은 형태 dim 이다
        self.assertIn("- `n_rows`, `n_cols`, `nnz`: Python `int`; always consistent with the array shapes above.",
                      render_param_docs(spec_of("spmv_csr"), "python", "positional"))
        # doc 을 주면 그 문장이 쓰인다(권장)
        with_doc = parse_taskspec({"task": "j", "args": [
            {"name": "src", "kind": "in", "dtype": "float32", "shape": ["H", "W"]},
            {"name": "dst", "kind": "out", "dtype": "float32", "shape": ["H", "W"]},
            {"name": "H", "kind": "dim"}, {"name": "W", "kind": "dim"}, {"name": "T", "kind": "dim", "doc": "the number of steps"}]})
        self.assertIn("- `T`: `int`; the number of steps.", render_param_docs(with_doc, "c", "positional"))

    def test_c_docs_for_2d_task(self) -> None:
        self.assertEqual(render_param_docs(NBODY, "c", "positional"), "\n".join([
            "- Multi-dimensional arrays are stored row-major: element (i0, i1) of an array of shape (D0, D1) "
            "is at linear index i0*D1 + i1.",
            "- `pos`: `float` (float32) array with n*3 elements, shape (n, 3), read-only.",
            "- `mass`: `float` (float32) array with n elements, read-only.",
            "- `acc`: `float` (float32) array with n*3 elements, shape (n, 3); the output buffer that must be filled.",
            "- All array arguments start at 64-byte-aligned addresses and do not overlap.",
            "- `n`: `int`; always consistent with the array shapes above.",
            "- Return value: none (results are written into `acc`).",
        ]))

    def test_row_major_sentence_only_when_multidimensional(self) -> None:
        sentence = "row-major: element (i0, i1) of an array of shape (D0, D1) is at linear index i0*D1 + i1"
        for backend in ("c", "rust"):
            self.assertIn(sentence, render_param_docs(spec_of("gemm"), backend, "positional"))
            self.assertIn(sentence, render_param_docs(spec_of("jacobi2d"), backend, "positional"))
            self.assertNotIn("row-major", render_param_docs(spec_of("spmv_csr"), backend, "positional"))
            self.assertNotIn("row-major", render_param_docs(spec_of("dot"), backend, "positional"))
        for backend in ("python", "numba", "halide"):
            self.assertNotIn(sentence, render_param_docs(spec_of("gemm"), backend, "positional"))

    def test_c_and_rust_docs_for_1d_int_arrays(self) -> None:
        c_docs = render_param_docs(spec_of("spmv_csr"), "c", "positional")
        self.assertIn("- `row_ptr`: `int` (32-bit) array with n_rows + 1 elements, read-only.", c_docs)
        self.assertIn("- `y`: `float` (float32) array with n_rows elements; the output buffer that must be filled.", c_docs)
        self.assertIn("- `n_rows`, `n_cols`, `nnz`: `int`; always consistent with the array shapes above.", c_docs)
        rust_docs = render_param_docs(spec_of("spmv_csr"), "rust", "positional")
        self.assertIn("- `row_ptr`: `i32` array with n_rows + 1 elements, read-only.", rust_docs)
        self.assertIn("- `vals`: `f32` array with nnz elements, read-only.", rust_docs)
        self.assertIn("- `n_rows`, `n_cols`, `nnz`: `i32`; always consistent with the array shapes above.", rust_docs)
        self.assertIn("- `A`: `f32` array with M*K elements, shape (M, K), read-only.", render_param_docs(spec_of("gemm"), "rust", "positional"))
        # 원소가 하나인 출력(dot), 식이 섞인 다차원 형태
        self.assertIn("- `out`: `float` (float32) array with 1 element; the output buffer that must be filled.",
                      render_param_docs(spec_of("dot"), "c", "positional"))
        odd = parse_taskspec({"task": "t", "args": [{"name": "g", "kind": "out", "dtype": "int32", "shape": ["n + 1", 3, "2*m"]},
                                                    {"name": "n", "kind": "dim"}, {"name": "m", "kind": "dim"}]})
        odd_docs = render_param_docs(odd, "c", "positional")
        self.assertIn("- `g`: `int` (32-bit) array with (n + 1)*3*(2*m) elements, shape (n + 1, 3, 2*m); the output buffer that must be filled.", odd_docs)
        self.assertIn("element (i0, i1, i2) of an array of shape (D0, D1, D2) is at linear index (i0*D1 + i1)*D2 + i2.", odd_docs)
        self.assertIn("- The array argument starts at a 64-byte-aligned address.", odd_docs)

    def test_keyword_mode_explains_the_struct_for_c_and_rust_only(self) -> None:
        gemm, spmv = spec_of("gemm"), spec_of("spmv_csr")
        self.assertEqual(keyword_struct_intro(gemm, "c"),
                         "`args` points to one `gemm_args` structure (declared in `kernel_args.h`); "
                         "the structure itself is read-only. Its fields:")
        self.assertEqual(keyword_struct_intro(spmv, "rust"),
                         "`args` points to one `SpmvCsrArgs` structure (declared in the module file `kernel_args.rs`; "
                         "`mod kernel_args;` followed by `use kernel_args::SpmvCsrArgs;` brings the type into scope); "
                         "the structure itself is read-only. Its fields:")
        # keyword 설명 = 안내 문단 + 빈 줄 + positional 설명
        for spec, backend in ((gemm, "c"), (spmv, "rust"), (HISTOGRAM, "c"), (HISTOGRAM, "rust")):
            with self.subTest(task=spec.task, backend=backend):
                self.assertEqual(render_param_docs(spec, backend, "keyword"),
                                 keyword_struct_intro(spec, backend) + "\n\n" + render_param_docs(spec, backend, "positional"))
        for backend in ("python", "numba", "halide"):
            self.assertEqual(keyword_struct_intro(gemm, backend), "")
            self.assertEqual(render_param_docs(gemm, backend, "keyword"), render_param_docs(gemm, backend, "positional"))
        with self.assertRaises(ValueError):
            keyword_struct_intro(gemm, "cpp")

    def test_halide_docs_state_types_return_and_coordinate_rule(self) -> None:
        docs = render_param_docs(spec_of("gemm"), "halide", "positional")
        self.assertIn("- `A`: `hl.ImageParam` of type `hl.Float(32)` with 2 dimensions; the input array of shape (M, K), read-only.", docs)
        self.assertIn("- `M`, `N`, `K`: Python `int`, the actual value for this call; always consistent with the array shapes above.", docs)
        self.assertIn("- Returns: a `hl.Func` with 2 dimensions and element type `hl.Float(32)` that defines the output array `C` of shape (M, N).", docs)
        self.assertIn("`gemm` has no output parameter", docs)
        self.assertIn("the element at row i0 and column i1 is `X[i1, i0]`", docs)
        self.assertIn("a 1-D array is indexed as `x[i]`", docs)
        self.assertNotIn("- `C`:", docs, "halide 빌드 함수에는 출력 인자가 없다")
        int_docs = render_param_docs(HISTOGRAM, "halide", "positional")
        self.assertIn("- `keys`: `hl.ImageParam` of type `hl.Int(32)` with 1 dimension; the input array of shape (n,), read-only; "
                      "each value is in the range 0 <= keys[i] < n_bins.", int_docs)
        self.assertIn("a `hl.Func` with 1 dimension and element type `hl.Int(32)`", int_docs)
        self.assertIn("- `n_bins`: Python `int`, the actual value for this call; the number of bins.", int_docs)

    def test_docs_are_deterministic_english_without_library_names(self) -> None:
        for spec in (spec_of("gemm"), spec_of("jacobi2d"), spec_of("spmv_csr"), spec_of("dot"), HISTOGRAM, NBODY):
            for backend in BACKENDS:
                for mode in ARGS_MODES:
                    with self.subTest(task=spec.task, backend=backend, mode=mode):
                        docs = render_param_docs(spec, backend, mode)
                        self.assertEqual(docs, render_param_docs(spec, backend, mode))
                        self.assertTrue(docs.isascii(), "영어(ASCII) 문장이어야 한다")
                        self.assertFalse(docs.endswith("\n"))
                        self.assertNotIn("..", docs)
                        lowered = docs.lower()
                        for name in LIBRARY_NAMES:
                            self.assertNotIn(name, lowered)
                        if backend != "halide":
                            self.assertNotIn("halide", lowered)
                            self.assertNotIn("hl.", docs)
                        # 모든 인자가 설명된다(halide 는 출력 인자가 없고 반환 규칙에 이름이 나온다)
                        for arg in spec.args:
                            self.assertIn(f"`{arg.name}`", docs)


@requires_tool("rustc")
class RustKeywordScopeTest(unittest.TestCase):
    """Rust keyword 시그니처(``args: *const {Camel}Args``)는 설명이 적어 둔 ``mod``/``use`` 문이 있어야 컴파일된다.

    회귀: 안내 문단이 "declared in ``kernel_args.rs``" 뿐이면 글자 그대로 따른 커널이 ``error[E0412]: cannot find type`` 로
    떨어진다(rustc 1.92 실측). 링크 없이 타입 검사만 한다(``--emit=metadata``) — 링커가 없는 머신에서도 돈다."""

    def typecheck(self, spec: TaskSpec, statements: list[str]) -> ProcResult:
        with tempfile.TemporaryDirectory(prefix="sigs_rust_") as tmp:
            src = Path(tmp) / "src"
            src.mkdir()
            (src / "kernel_args.rs").write_text(render_args_struct(spec, "rust"), encoding="utf-8", newline="\n")
            lib = "".join(f"{s}\n" for s in statements) + "\n" + render_signature(spec, "rust", "keyword") + " {\n    let _ = args;\n}\n"
            (src / "lib.rs").write_text(lib, encoding="utf-8", newline="\n")
            cmd = ["rustc", "--edition", "2021", "--crate-type", "cdylib", "--crate-name", "kernel", "--emit=metadata",
                   "-o", str(Path(tmp) / "kernel.rmeta"), str(src / "lib.rs")]
            return run_isolated(cmd, timeout_s=300, cwd=Path(tmp), tail_chars=4000)

    def test_signature_compiles_with_the_statements_the_docs_state(self) -> None:
        for spec in (spec_of("gemm"), HISTOGRAM):
            with self.subTest(task=spec.task):
                intro = render_param_docs(spec, "rust", "keyword").split("\n\n", 1)[0]
                statements = re.findall(r"`((?:mod|use) [^`]*;)`", intro)
                self.assertEqual(statements, ["mod kernel_args;", f"use kernel_args::{camel(spec.function)}Args;"])
                res = self.typecheck(spec, statements)
                self.assertEqual(res.rc, 0, res.stderr_tail)

    def test_signature_does_not_compile_without_the_use_statement(self) -> None:
        """``mod kernel_args;`` 만으로는 타입이 풀리지 않는다 — 그래서 설명에 ``use`` 문이 있어야 한다."""
        res = self.typecheck(spec_of("gemm"), ["mod kernel_args;"])
        self.assertNotEqual(res.rc, 0)
        self.assertIn("E0412", res.stderr_tail)
        self.assertIn("GemmArgs", res.stderr_tail)


class SignatureBundleTest(unittest.TestCase):
    def test_existing_tasks_use_yaml_strings_unchanged(self) -> None:
        """기존 4개 과제의 numpy/c_seq/c_openmp positional 은 source == "yaml" 이고 YAML 문자열과 같다."""
        for task in EXISTING_TASKS:
            spec = spec_of(task)
            langs = raw_yaml(task)["languages"]
            for language, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
                with self.subTest(task=task, language=language):
                    b = signature_bundle(spec, language, backend, "positional")
                    self.assertEqual(set(b), BUNDLE_KEYS)
                    self.assertEqual(b["source"], "yaml")
                    self.assertEqual(b["signature"], langs[language]["positional"])
                    self.assertEqual(b["param_docs"], langs[language]["param_docs"])
                    self.assertEqual(b["function"], task)
                    self.assertEqual(b["params"], spec.params)
                    self.assertEqual(b["args_struct"], "")
                    self.assertEqual(b["numba_signature"], "")

    def test_yaml_keyword_string_is_used_when_present(self) -> None:
        b = signature_bundle(spec_of("gemm"), "numpy", "python", "keyword")
        self.assertEqual((b["source"], b["signature"]), ("yaml", "def gemm(*, A, B, C, M, N, K):"))
        self.assertEqual(b["param_docs"], raw_yaml("gemm")["languages"]["numpy"]["param_docs"])

    def test_null_yaml_signature_falls_back_to_rendering(self) -> None:
        # c_seq 의 keyword 는 YAML 에 null → 시그니처는 렌더링한다(source 는 시그니처 기준). param_docs 는 YAML 본문을 쓰되
        # 매개변수가 구조체 하나로 바뀌므로 렌더링한 설명과 같은 구조체 안내 문단이 앞에 붙는다
        spec = spec_of("gemm")
        b = signature_bundle(spec, "c_seq", "c", "keyword")
        self.assertEqual(b["source"], "rendered")
        self.assertEqual(b["signature"], "void gemm(const gemm_args* args)")
        yaml_docs = raw_yaml("gemm")["languages"]["c_seq"]["param_docs"]
        self.assertEqual(b["param_docs"],
                         "`args` points to one `gemm_args` structure (declared in `kernel_args.h`); "
                         "the structure itself is read-only. Its fields:\n\n" + yaml_docs)
        self.assertEqual(b["args_struct"], render_args_struct(spec, "c"))

    def test_c_keyword_docs_have_the_same_frame_for_yaml_and_rendered_docs(self) -> None:
        """args=keyword 의 C 설명은 본문이 YAML 이든 렌더링이든 같은 구조체 안내 문단으로 시작한다.

        회귀: 기존 4개 과제는 YAML 의 positional 용 설명이 그대로 나가 ``args`` 가 무엇인지 말하는 문단이 없었고, YAML 항목이
        없는 새 과제에만 있었다 — 같은 조건(args=keyword)의 프롬프트 구조가 과제 묶음에 따라 달랐다."""
        for task in EXISTING_TASKS:
            spec = spec_of(task)
            langs = raw_yaml(task)["languages"]
            intro = (f"`args` points to one `{task}_args` structure (declared in `kernel_args.h`); "
                     "the structure itself is read-only. Its fields:")
            for language in ("c_seq", "c_openmp"):
                with self.subTest(task=task, language=language):
                    keyword = signature_bundle(spec, language, "c", "keyword")
                    positional = signature_bundle(spec, language, "c", "positional")
                    self.assertEqual(positional["param_docs"], langs[language]["param_docs"], "positional 은 YAML 그대로")
                    self.assertEqual(keyword["param_docs"], intro + "\n\n" + positional["param_docs"])
                    self.assertEqual((keyword["source"], keyword["signature"]), ("rendered", f"void {task}(const {task}_args* args)"))
                    # YAML 항목이 없을 때의 렌더링도 같은 문단 + 빈 줄로 시작한다
                    self.assertTrue(render_param_docs(spec, "c", "keyword").startswith(intro + "\n\n"))
            # python 계열의 keyword 에는 구조체가 없다: YAML 문자열 그대로
            self.assertEqual(signature_bundle(spec, "numpy", "python", "keyword")["param_docs"], langs["numpy"]["param_docs"])
        rendered = signature_bundle(HISTOGRAM, "c_seq", "c", "keyword")["param_docs"]
        self.assertTrue(rendered.startswith("`args` points to one `histogram_args` structure (declared in `kernel_args.h`); "
                                            "the structure itself is read-only. Its fields:\n\n- `keys`:"), rendered)

    def test_keyword_struct_intro_is_added_to_yaml_docs_for_rust_too(self) -> None:
        raw = {
            "task": "dot", "function": "dot",
            "args": [{"name": "x", "kind": "in", "dtype": "float32", "shape": ["n"]},
                     {"name": "out", "kind": "out", "dtype": "float32", "shape": [1]},
                     {"name": "n", "kind": "dim"}],
            "languages": {"rust": {"param_docs": "- `x`: hand-written.\n- `out`, `n`: hand-written."},
                          "c_seq": {"keyword": "void dot(const dot_args *args)", "param_docs": "- hand-written C docs."}},
        }
        spec = parse_taskspec(raw)
        self.assertEqual(signature_bundle(spec, "rust", "rust", "positional")["param_docs"], "- `x`: hand-written.\n- `out`, `n`: hand-written.")
        b = signature_bundle(spec, "rust", "rust", "keyword")
        self.assertEqual(b["param_docs"], keyword_struct_intro(spec, "rust") + "\n\n- `x`: hand-written.\n- `out`, `n`: hand-written.")
        self.assertIn("`use kernel_args::DotArgs;`", b["param_docs"])
        # YAML 에 keyword 시그니처까지 있어도(source == "yaml") 설명의 틀은 같다: 안내 문단 + YAML 본문
        c = signature_bundle(spec, "c_seq", "c", "keyword")
        self.assertEqual((c["source"], c["signature"]), ("yaml", "void dot(const dot_args *args)"))
        self.assertEqual(c["param_docs"], keyword_struct_intro(spec, "c") + "\n\n- hand-written C docs.")

    def test_yaml_param_docs_lose_only_their_trailing_newlines(self) -> None:
        """``|`` (clip)·``|+`` (keep) 블록으로 쓴 param_docs 도 ``|-`` 로 쓴 것과 같은 문자열이 된다.

        회귀: 묶음이 YAML 문자열을 그대로 주면 ``|`` 블록의 끝 개행이 프롬프트에 빈 줄 하나를 더 넣어, 끝 개행을 떼던
        생성기(``generate.load_signature`` 의 ``rstrip("\\n")``)와 프롬프트 바이트·``prompt_hash`` 가 달라진다."""
        text = textwrap.dedent(
            """
            task: t
            function: t
            args:
              - {name: x, kind: in, dtype: float32, shape: [n]}
              - {name: out, kind: out, dtype: float32, shape: [1]}
              - {name: n, kind: dim}
            languages:
              numpy:
                positional: "def t(x, out, n):"
                param_docs: |
                  First paragraph.

                  - `x`: input
                  - `out`: output
              c_seq:
                param_docs: |-
                  First paragraph.

                  - `x`: input
                  - `out`: output
              c_openmp:
                param_docs: |+
                  First paragraph.

                  - `x`: input
                  - `out`: output


              rust:
                param_docs: "\\n\\n"
            """
        )
        raw = yaml.safe_load(text)
        self.assertTrue(raw["languages"]["numpy"]["param_docs"].endswith("output\n"))
        self.assertTrue(raw["languages"]["c_openmp"]["param_docs"].endswith("output\n\n\n"))
        spec = parse_taskspec(raw)
        expected = "First paragraph.\n\n- `x`: input\n- `out`: output"
        for language, backend in (("numpy", "python"), ("c_seq", "c"), ("c_openmp", "c")):
            with self.subTest(language=language):
                self.assertEqual(signature_bundle(spec, language, backend, "positional")["param_docs"], expected)
        self.assertEqual(signature_bundle(spec, "c_openmp", "c", "keyword")["param_docs"],
                         keyword_struct_intro(spec, "c") + "\n\n" + expected)
        # 개행뿐인 설명은 없는 것과 같다 → 렌더링한다
        self.assertEqual(signature_bundle(spec, "rust", "rust", "positional")["param_docs"], render_param_docs(spec, "rust", "positional"))
        # 기존 과제의 YAML 설명은 전부 ``|-`` 블록이라 이 규칙으로 바뀌지 않는다(끝 개행이 없다)
        for task in EXISTING_TASKS:
            for language, entry in raw_yaml(task)["languages"].items():
                self.assertFalse(entry["param_docs"].endswith("\n"), f"{task}/{language}")

    def test_language_without_yaml_entry_is_rendered(self) -> None:
        spec = spec_of("spmv_csr")
        expected_params = {"numba": spec.params, "halide": halide_params(spec), "rust": spec.params}
        for language, backend in (("numba", "numba"), ("halide_auto", "halide"), ("halide_manual", "halide"), ("rust", "rust")):
            for mode in ARGS_MODES:
                with self.subTest(language=language, mode=mode):
                    b = signature_bundle(spec, language, backend, mode)
                    self.assertEqual(set(b), BUNDLE_KEYS)
                    self.assertEqual(b["source"], "rendered")
                    self.assertEqual(b["signature"], SPMV_SIGNATURES[(backend, mode)])
                    self.assertEqual(b["param_docs"], render_param_docs(spec, backend, mode))
                    self.assertEqual(b["params"], expected_params[backend])
                    self.assertEqual(b["numba_signature"], _SPMV_NUMBA if backend == "numba" else "")
                    self.assertEqual(b["args_struct"], render_args_struct(spec, backend) if (backend == "rust" and mode == "keyword") else "")

    def test_task_without_languages_is_rendered_for_every_backend(self) -> None:
        self.assertEqual(HISTOGRAM.languages_raw, {})
        for language, backend in (("numpy", "python"), ("numba", "numba"), ("halide_auto", "halide"), ("c_seq", "c"), ("c_openmp", "c"), ("rust", "rust")):
            for mode in ARGS_MODES:
                with self.subTest(language=language, mode=mode):
                    b = signature_bundle(HISTOGRAM, language, backend, mode)
                    self.assertEqual(b["source"], "rendered")
                    self.assertEqual(b["signature"], render_signature(HISTOGRAM, backend, mode))
                    self.assertEqual(b["param_docs"], render_param_docs(HISTOGRAM, backend, mode))

    def test_yaml_override_wins_over_renderer(self) -> None:
        raw = {
            "task": "histogram", "function": "histogram",
            "args": [{"name": "keys", "kind": "in", "dtype": "int32", "shape": ["n"]},
                     {"name": "hist", "kind": "out", "dtype": "int32", "shape": ["n_bins"]},
                     {"name": "n", "kind": "dim"}, {"name": "n_bins", "kind": "dim"}],
            "languages": {"c_seq": {"positional": "void histogram(const int *keys, int *hist, int n, int n_bins)",
                                    "param_docs": "hand-written docs\nline two"},
                          "numpy": {"positional": "def histogram(keys, hist, n, n_bins):"}},
        }
        spec = parse_taskspec(raw)
        b = signature_bundle(spec, "c_seq", "c", "positional")
        self.assertEqual((b["source"], b["signature"], b["param_docs"]),
                         ("yaml", "void histogram(const int *keys, int *hist, int n, int n_bins)", "hand-written docs\nline two"))
        # 같은 backend 라도 YAML 항목이 없는 언어(c_openmp)는 렌더링한다
        b2 = signature_bundle(spec, "c_openmp", "c", "positional")
        self.assertEqual((b2["source"], b2["signature"]), ("rendered", "void histogram(const int* keys, int* hist, int n, int n_bins)"))
        # 시그니처만 YAML 에 있으면 param_docs 는 렌더링한다(source 는 시그니처 기준)
        b3 = signature_bundle(spec, "numpy", "python", "positional")
        self.assertEqual(b3["source"], "yaml")
        self.assertEqual(b3["param_docs"], render_param_docs(spec, "python", "positional"))


class CliTest(unittest.TestCase):
    """``python -m harness.sigs`` (진단용): 출력 형식, JSON, 종료 코드, 인코딩할 수 없는 글자가 있어도 죽지 않는 진입점."""

    def run_main(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = sigs.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_text_output_for_c_keyword(self) -> None:
        rc, out, _ = self.run_main("--task", "gemm", "--language", "c_seq", "--backend", "c", "--args-mode", "keyword")
        self.assertEqual(rc, 0)
        bundle = signature_bundle(spec_of("gemm"), "c_seq", "c", "keyword")
        self.assertEqual(out, "# source=rendered function=gemm params=A, B, C, M, N, K\n"
                              f"{bundle['signature']}\n\n{bundle['param_docs']}\n\n{bundle['args_struct']}")
        self.assertIn("\n} gemm_args;\n", out)

    def test_language_defaults_to_the_backend_name_and_signatures_path(self) -> None:
        # --language 를 주지 않으면 YAML languages 의 키로 backend 이름을 쓴다(그런 항목이 없으므로 렌더링)
        rc, out, _ = self.run_main("--signatures", str(TASKS / "dot" / "signatures.yaml"), "--backend", "rust")
        self.assertEqual(rc, 0)
        self.assertTrue(out.startswith("# source=rendered function=dot params=x, y, out, n\n#[no_mangle]\n"
                                       'pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32)\n\n'), out)
        self.assertTrue(out.endswith("- Return value: none (results are written into `out`).\n"), out)

    def test_json_output_is_the_bundle(self) -> None:
        rc, out, _ = self.run_main("--task", "gemm", "--language", "numpy", "--backend", "python", "--json")
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), signature_bundle(spec_of("gemm"), "numpy", "python", "positional"))

    def test_usage_errors_and_unreadable_spec_exit_2(self) -> None:
        for argv in (["--backend", "c"],  # --task 도 --signatures 도 없다
                     ["--task", "gemm", "--signatures", str(TASKS / "gemm" / "signatures.yaml"), "--backend", "c"],
                     ["--task", "gemm", "--backend", "fortran"]):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as cm:
                self.run_main(*argv)
            self.assertEqual(cm.exception.code, 2)
        rc, out, err = self.run_main("--task", "no_such_task", "--backend", "c")
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("[sigs]", err)
        with tempfile.TemporaryDirectory(prefix="sigs_cli_") as tmp:
            bad = Path(tmp) / "signatures.yaml"
            bad.write_text("task: t\nargs: []\n", encoding="utf-8", newline="\n")
            rc, _, err = self.run_main("--signatures", str(bad), "--backend", "c")
        self.assertEqual(rc, 2)
        self.assertIn("[sigs]", err)

    def test_entry_point_prints_text_that_the_stream_encoding_cannot_encode(self) -> None:
        """회귀(공용 ``setup_stdio``): 로캘 코드페이지로 인코딩할 수 없는 글자가 YAML 설명에 있어도 진입점이 죽지 않는다.

        ``PYTHONIOENCODING=ascii`` 로 어느 플랫폼에서나 같은 상황을 만든다(cp949 콘솔의 ``—`` 와 같다)."""
        with tempfile.TemporaryDirectory(prefix="sigs_cli_") as tmp:
            path = Path(tmp) / "signatures.yaml"
            path.write_text('task: t\nargs:\n  - {name: o, kind: out, dtype: float32, shape: [1], doc: "the result — one value"}\n',
                            encoding="utf-8", newline="\n")
            out_path = Path(tmp) / "out.txt"
            env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "ascii"}
            res = run_isolated([sys.executable, "-m", "harness.sigs", "--signatures", str(path), "--backend", "python"],
                               timeout_s=120, cwd=ROOT, env=env, stdout_path=out_path)
            self.assertEqual(res.rc, 0, res.stderr_tail)
            self.assertIn("the result — one value.", out_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
