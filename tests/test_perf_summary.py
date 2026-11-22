"""perf_summary 테스트 (설계 v3 §11): 실제 입력을 픽스처로 써서 요약 JSON 전체를 단언한다.

픽스처(``tests/fixtures/perf_summary/``)의 출처:

- 더미 응답의 코드 블록을 그대로 옮긴 것 — ``c/gemm_ikj.c`` (gemm/c_seq/0), ``gemm_ijk_missing_semicolon.c`` (c_seq/1),
  ``gemm_ijk.c`` (c_seq/2), ``gemm_omp_outer.c`` (c_openmp/0), ``gemm_omp_inner.c`` (c_openmp/2), ``dot_unrolled.c`` (dot/c_seq/0),
  ``dot_omp_simd_reduction.c`` (dot/c_openmp/0), ``jacobi2d_omp.c`` (jacobi2d/c_openmp/0), ``spmv_csr_omp.c`` (spmv_csr/c_openmp/0),
  ``python/gemm_matmul_out.py`` (gemm/numpy/0), ``gemm_matmul_scaled.py`` (numpy/2), ``gemm_triple_loop.py`` (numpy/3),
  ``gemm_return_product.py`` (numpy/4), ``dot_np_dot.py``, ``jacobi2d_slices.py``, ``spmv_csr_bincount.py`` (각 과제의 numpy/0).
- 손으로 쓴 커널 — ``c/gemm_blis_avx2.c`` (블록화 + 패킹 + AVX2/FMA 6×16 마이크로커널: 첫 실제 실행에서 모델이 낸 형태),
  ``gemm_blocked_omp_collapse.c``, ``histogram_omp_private.c``, ``gemm_transposed_b.c``, ``numba/*.py``, ``halide/*.py``, ``rust/*.rs``.
  C 는 zig cc(OpenMP pragma 는 무시된다), ``rust/dot_avx2.rs``·``gemm_blocked.rs`` 는 rustc 로 빌드하고 Numba·Halide 커널은
  그대로 실행해 NumPy 참조와 맞는 것을 확인했다. rayon 을 쓰는 두 Rust 파일은 이 머신에 크레이트가 없어 rustfmt 구문 검사만 했다.
- ``c/*.build.log`` 는 **실제 gcc 출력이 아니다.** 이 머신에 gcc 가 없어, 실험 머신의 gcc 15 가 ``-fopt-info-vec-optimized
  -fopt-info-vec-missed`` 로 내는 줄 모양(``kernel.c:14:27: optimized: loop vectorized using 32 byte vectors`` 등; 열은 조건식의
  비교 연산자)을 픽스처 소스의 줄·열에 맞춰 손으로 조립했다. Linux 에서 실제 로그로 다시 확인한다.
- ``*.import.json`` 은 ``make_inspect_fixtures.py`` 가 실제 Numba 0.68.1·Halide 22.1.0(Mullapudi2016)으로 만든 ``inspect`` 다.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

from harness import perf_summary
from harness.perf_summary import DECISIONS, DEFAULT_SCHEDULE_METHODS, summarize
from harness.taskspec import parse_taskspec

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "perf_summary"

KEYS = ["language", "loops", "parallel", "vectorized", "tiling", "memory", "library_calls",
        "decisions_by_llm", "decisions_by_compiler", "decisions_by_library", "notes"]

GEMM = parse_taskspec({"task": "gemm", "function": "gemm", "args": [
    {"name": "A", "kind": "in", "dtype": "float32", "shape": ["M", "K"]}, {"name": "B", "kind": "in", "dtype": "float32", "shape": ["K", "N"]},
    {"name": "C", "kind": "out", "dtype": "float32", "shape": ["M", "N"]}, {"name": "M", "kind": "dim"}, {"name": "N", "kind": "dim"},
    {"name": "K", "kind": "dim"}]})
DOT = parse_taskspec({"task": "dot", "function": "dot", "args": [
    {"name": "x", "kind": "in", "dtype": "float32", "shape": ["n"]}, {"name": "y", "kind": "in", "dtype": "float32", "shape": ["n"]},
    {"name": "out", "kind": "out", "dtype": "float32", "shape": [1]}, {"name": "n", "kind": "dim"}]})
HISTOGRAM = parse_taskspec({"task": "histogram", "function": "histogram", "args": [
    {"name": "keys", "kind": "in", "dtype": "int32", "shape": ["n"]}, {"name": "hist", "kind": "out", "dtype": "int32", "shape": ["n_bins"]},
    {"name": "n", "kind": "dim"}, {"name": "n_bins", "kind": "dim"}]})


def read(rel: str) -> str:
    return (FIXTURES / rel).read_text(encoding="utf-8")


def inspect_of(rel: str) -> dict[str, Any]:
    return json.loads(read(rel))["inspect"]


def expect(language: str, **over: Any) -> dict[str, Any]:
    """아무것도 찾지 못한 요약(전체 키)에 ``over`` 를 덮어쓴 기대값. 값이 dict 인 키는 준 하위 키만 덮어쓴다."""
    doc: dict[str, Any] = {
        "language": language,
        "loops": [],
        "parallel": {"kind": "none", "regions": 0, "collapse": False, "reduction": False, "loops": [], "placement": "none", "constructs": []},
        "vectorized": {"status": "unknown", "source": "none", "loops": []},
        "tiling": {"detected": False, "tile_sizes": []},
        "memory": {"temporaries": 0, "layout_transposed": False},
        "library_calls": [],
        "decisions_by_llm": [],
        "decisions_by_compiler": [],
        "decisions_by_library": [],
        "notes": "",
    }
    for key, value in over.items():
        doc[key] = {**doc[key], **value} if isinstance(doc[key], dict) else value
    return doc


UNKNOWN_PARALLEL = {"kind": "unknown", "regions": None, "collapse": None, "reduction": None, "placement": "unknown"}
# 루프를 다 읽지 못한 소스(괄호 짝이 안 맞거나 매크로가 루프를 숨긴다): 루프에 기대는 판정은 "없음"이 아니라 "모름"이다
UNREAD_LOOPS: dict[str, Any] = {"tiling": {"detected": None}, "memory": {"layout_transposed": None}}


def nest(*levels: tuple[str, str]) -> dict[str, Any]:
    """``nest(("i", "M"), ("j", "N"))`` → loops 항목 하나."""
    return {"depth": len(levels), "order": [var for var, _ in levels], "bounds": [bound for _, bound in levels]}


def c(code: str, build_log: str = "", language: str = "c_seq", **kw: Any) -> dict[str, Any]:
    return summarize(language=language, backend="c", code=code, build_log=build_log, **kw)


def c_fixture(name: str, language: str, *, log: bool = False, **kw: Any) -> dict[str, Any]:
    return c(read(f"c/{name}.c"), read(f"c/{name}.build.log") if log else "", language, **kw)


def py(code: str, **kw: Any) -> dict[str, Any]:
    return summarize(language="numpy", backend="python", code=code, **kw)


def numba(code: str, **kw: Any) -> dict[str, Any]:
    return summarize(language="numba", backend="numba", code=code, **kw)


def halide(name: str, language: str, inspect: str | None = None, **kw: Any) -> dict[str, Any]:
    return summarize(language=language, backend="halide", code=read(f"halide/{name}.py"),
                     inspect=inspect_of(f"halide/{inspect}.import.json") if inspect else None, **kw)


def rust(code: str, **kw: Any) -> dict[str, Any]:
    return summarize(language="rust", backend="rust", code=code, **kw)


GEMM_IKJ_LOOPS = [nest(("i", "M"), ("j", "N")), nest(("i", "M"), ("k", "K"), ("j", "N"))]
GEMM_IJK_LOOPS = [nest(("i", "M"), ("j", "N"), ("k", "K"))]

BLIS_LOOPS = [
    nest(("i", "mc"), ("p", "kc"), ("r", "rows")),
    nest(("i", "mc"), ("p", "kc"), ("r", "MR")),
    nest(("j", "nc"), ("p", "kc"), ("c", "cols")),
    nest(("j", "nc"), ("p", "kc"), ("c", "NR")),
    nest(("p", "kc")),
    nest(("jc", "N"), ("pc", "K"), ("ic", "M"), ("jr", "nc"), ("ir", "mc"), ("r", "rows"), ("q", "cols")),
]
BLIS_INTRINSICS_NOTE = "intrinsics: 7 distinct (_mm256_setzero_ps, _mm256_load_ps, _mm256_broadcast_ss, _mm256_fmadd_ps, +3 more)"


class SchemaTest(unittest.TestCase):
    def all_fixture_summaries(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted((FIXTURES / "c").glob("*.c")):
            log = path.with_suffix(".build.log")
            out.append(c(path.read_text(encoding="utf-8"), log.read_text(encoding="utf-8") if log.exists() else "", "c_openmp"))
        for path in sorted((FIXTURES / "python").glob("*.py")):
            out.append(py(path.read_text(encoding="utf-8")))
        for path in sorted((FIXTURES / "numba").glob("*.py")):
            out.append(numba(path.read_text(encoding="utf-8"), inspect=inspect_of(f"numba/{path.stem}.import.json")))
        for path in sorted((FIXTURES / "halide").glob("*.import.json")):
            source = path.name.split(".")[0]
            language = "halide_auto" if "auto" in source else "halide_manual"
            out.append(halide(source, language, path.name.removesuffix(".import.json")))
        for path in sorted((FIXTURES / "rust").glob("*.rs")):
            out.append(rust(path.read_text(encoding="utf-8")))
        return out

    def test_keys_are_the_claude_md_schema_plus_decisions_by_library(self) -> None:
        for backend, language in (("c", "c_seq"), ("rust", "rust"), ("python", "numpy"), ("numba", "numba"), ("halide", "halide_auto"), ("zig", "zig")):
            out = summarize(language=language, backend=backend, code="")
            self.assertEqual(list(out), KEYS, backend)
            self.assertEqual(list(out["parallel"])[:4], ["kind", "regions", "collapse", "reduction"])
            self.assertEqual(list(out["parallel"])[4:], ["loops", "placement", "constructs"])
            self.assertEqual(list(out["vectorized"]), ["status", "source", "loops"])
            self.assertEqual(list(out["tiling"]), ["detected", "tile_sizes"])
            self.assertEqual(list(out["memory"]), ["temporaries", "layout_transposed"])
            self.assertEqual(out["language"], language)

    def test_fixture_summaries_fit_one_csv_cell_and_use_the_fixed_vocabulary(self) -> None:
        summaries = self.all_fixture_summaries()
        self.assertGreater(len(summaries), 35)
        for out in summaries:
            json.dumps(out)
            self.assertEqual(list(out), KEYS)
            self.assertLessEqual(len(out["loops"]), 20)
            self.assertLessEqual(len(out["notes"]), 500)
            self.assertNotIn(perf_summary.ERROR_NOTE_PREFIX, out["notes"])
            for key in ("decisions_by_llm", "decisions_by_compiler", "decisions_by_library"):
                self.assertEqual(out[key], [d for d in DECISIONS if d in out[key]], key)
            self.assertIn(out["vectorized"]["status"], ("yes", "partial", "no", "unknown"))
            self.assertIn(out["parallel"]["kind"], ("none", "openmp", "threads", "rayon", "numba", "halide", "unknown"))
            for loop in out["loops"]:
                self.assertEqual(loop["depth"], len(loop["order"]))
                self.assertEqual(loop["depth"], len(loop["bounds"]))

    def test_library_delegation_only_appears_in_decisions_by_library(self) -> None:
        for out in self.all_fixture_summaries():
            self.assertNotIn("library_delegation", out["decisions_by_llm"] + out["decisions_by_compiler"])
            self.assertLessEqual(set(out["decisions_by_library"]), {"library_delegation"})

    def test_loops_are_capped_at_20_and_the_cut_is_noted(self) -> None:
        code = "void f(float* a, int n) {\n" + "".join(f"    for (int i{k} = 0; i{k} < n; i{k}++) a[i{k}] = {k};\n" for k in range(30)) + "}\n"
        out = c(code)
        self.assertEqual(len(out["loops"]), 20)
        self.assertEqual(out["loops"][0], nest(("i0", "n")))
        self.assertEqual(out["loops"][-1], nest(("i19", "n")))
        self.assertEqual(out["notes"], "loops truncated: first 20 of 30 nests kept")

    def test_notes_are_capped_at_500_characters(self) -> None:
        names = [f"__builtin_{'x' * 150}_{k}" for k in range(6)]
        out = c("void f(float* a, int n) { a[0] = " + " + ".join(f"{name}(n)" for name in names) + "; }\n")
        self.assertEqual(len(out["notes"]), 500)
        self.assertTrue(out["notes"].startswith("builtins: __builtin_xxx"))
        self.assertTrue(out["notes"].endswith("..."))

    def test_a_long_compiler_reason_does_not_crowd_out_the_other_notes(self) -> None:
        code = "void f(float* a, int n) {\n    for (int i = 0; i < n; i++) a[i] = __builtin_expect(n, 1);\n}\n"
        out = c(code, "kernel.c:2:23: missed: not vectorized: " + "statement dump " * 40 + ".\n")
        self.assertEqual(out["notes"], "gcc: not vectorized i@2 (" + ("statement dump " * 6)[:77] + "...); builtins: __builtin_expect")

    def test_long_expression_text_is_clipped(self) -> None:
        bound = " + ".join(f"n{k}" for k in range(40))
        out = c(f"void f(float* a) {{ for (int i = 0; i < {bound}; i++) a[i] = 0; }}")
        self.assertEqual(len(out["loops"][0]["bounds"][0]), 80)
        self.assertTrue(out["loops"][0]["bounds"][0].endswith("..."))


class DummyResponsesTest(unittest.TestCase):
    """``harness/dummy_responses`` 의 모든 응답(다른 언어·과제가 더해져도)을 내부 오류 없이 요약한다."""

    BACKENDS = {"numpy": "python", "numba": "numba", "halide_auto": "halide", "halide_manual": "halide",
                "c_seq": "c", "c_openmp": "c", "rust": "rust"}

    def test_every_dummy_response_is_summarized_without_an_internal_error(self) -> None:
        root = Path(perf_summary.__file__).resolve().parent / "dummy_responses"
        seen = 0
        for path in sorted(root.glob("*/*/*.md")):
            language = path.parent.name
            block = re.search(r"^```\w*\n(.*?)^```", path.read_text(encoding="utf-8"), re.S | re.M)
            if language not in self.BACKENDS or block is None:
                continue
            with self.subTest(response=path.relative_to(root).as_posix()):
                out = summarize(language=language, backend=self.BACKENDS[language], code=block.group(1), task=path.parts[-3])
                self.assertEqual(list(out), KEYS)
                self.assertNotIn(perf_summary.ERROR_NOTE_PREFIX, out["notes"])
                self.assertLessEqual(len(json.dumps(out)), 4000)
                seen += 1
        self.assertGreaterEqual(seen, 20)


class NeverRaisesTest(unittest.TestCase):
    def test_internal_failure_returns_the_full_key_set_with_an_error_note(self) -> None:
        with mock.patch.object(perf_summary, "_analyze_c", side_effect=RuntimeError("boom")):
            out = c(read("c/gemm_ikj.c"))
        self.assertEqual(out, expect("c_seq", parallel=UNKNOWN_PARALLEL, tiling={"detected": None},
                                     memory={"temporaries": None, "layout_transposed": None},
                                     notes="perf_summary error: RuntimeError: boom"))

    def test_wrong_input_types_are_reported_not_raised(self) -> None:
        out = summarize(language="c_seq", backend="c", code=None)  # type: ignore[arg-type]
        self.assertEqual(list(out), KEYS)
        self.assertTrue(out["notes"].startswith("perf_summary error: TypeError: "), out["notes"])
        self.assertEqual(out["vectorized"]["status"], "unknown")
        out = summarize(language="numpy", backend="python", code=b"def f(): pass", spec=object())  # type: ignore[arg-type]
        self.assertTrue(out["notes"].startswith("perf_summary error: "), out["notes"])

    def test_unsupported_backend_is_unknown(self) -> None:
        self.assertEqual(summarize(language="opencl", backend="opencl", code="kernel void f() {}"),
                         expect("opencl", parallel=UNKNOWN_PARALLEL, tiling={"detected": None},
                                memory={"temporaries": None, "layout_transposed": None}, notes="unsupported backend: opencl"))

    def test_empty_and_garbage_input(self) -> None:
        unknown = dict(parallel=UNKNOWN_PARALLEL, tiling={"detected": None}, memory={"temporaries": None, "layout_transposed": None})
        for text in ("", "lorem ipsum dolor sit amet"):
            self.assertEqual(c(text), expect("c_seq"))
            self.assertEqual(rust(text), expect("rust", notes="unsafe blocks: 0"))
        self.assertEqual(c("@@@ ]]] {{{ ((( for"), expect("c_seq", **UNREAD_LOOPS, notes="unbalanced brackets: loop nests omitted"))
        self.assertEqual(py(""), expect("numpy"))
        self.assertEqual(numba(""), expect("numba"))
        self.assertEqual(py("lorem ipsum dolor sit amet"), expect(
            "numpy", **unknown, notes="source does not parse: SyntaxError: invalid syntax (<unknown>, line 1)"))
        # Halide: 소스에 스케줄이 없다는 것까지만 안다 — 수동이면 병렬 없음, 오토스케줄러가 붙으면 그 출력을 보기 전에는 모른다
        self.assertEqual(summarize(language="halide_manual", backend="halide", code=""),
                         expect("halide_manual", memory={"temporaries": None}, notes="loop nest not available"))
        self.assertEqual(summarize(language="halide_auto", backend="halide", code=""),
                         expect("halide_auto", **unknown, notes="loop nest not available; autoscheduler schedule not available"))

    def test_adversarial_text_never_raises(self) -> None:
        texts = [
            "", "\n\n", "\x00\x01\x02", "lorem ipsum dolor sit amet", "}}}}]]]))))", "((((([[[[{{{{", "for", "for (", "for (;;", "for (;;) {",
            "#pragma omp parallel for", "#define", "#define X \\", "/* never closed", '"never closed', "'", "r#\"never closed",
            "for i in", "for x in 0..", "@njit\ndef", "def f(:\n", "import", "﻿void f() {}", "void f() { for (int i = 0; i < n; i++) }",
            "void f(){" + "for(;;)" * 3000 + ";}", "x = " + "(" * 400 + "1" + ")" * 400, "x = " + " + ".join(["a"] * 3000),
            "for " * 5000, "{" * 5000, "a[" * 2000,
        ]
        for backend, language in (("c", "c_openmp"), ("rust", "rust"), ("python", "numpy"), ("numba", "numba"), ("halide", "halide_manual")):
            for text in texts:
                with self.subTest(backend=backend, text=text[:30]):
                    out = summarize(language=language, backend=backend, code=text, build_log=text,
                                    inspect={"loop_nest": text, "schedule_source": text, "parallel_diagnostics": text}, build_extra={"options": text})
                    self.assertEqual(list(out), KEYS)
                    json.dumps(out)
                    self.assertLessEqual(len(out["notes"]), 500)

    def test_summarize_opens_no_file_and_starts_no_process(self) -> None:
        cases = [
            dict(language="c_openmp", backend="c", code=read("c/jacobi2d_omp.c"), build_log=read("c/jacobi2d_omp.build.log")),
            dict(language="rust", backend="rust", code=read("rust/gemm_rayon.rs")),
            dict(language="numpy", backend="python", code=read("python/spmv_csr_bincount.py")),
            dict(language="numba", backend="numba", code=read("numba/gemm_prange.py"), inspect=inspect_of("numba/gemm_prange.import.json")),
            dict(language="halide_auto", backend="halide", code=read("halide/jacobi2d_auto.py"), inspect=inspect_of("halide/jacobi2d_auto.import.json")),
        ]
        summarize(language="halide_manual", backend="halide", code="")     # 스케줄 메서드 이름 집합의 지연 임포트를 미리 끝낸다
        expected = [summarize(**case) for case in cases]
        forbidden = AssertionError("summarize 는 순수 함수여야 한다")
        with mock.patch("builtins.open", side_effect=forbidden), mock.patch.object(Path, "open", side_effect=forbidden), \
                mock.patch.object(subprocess, "Popen", side_effect=forbidden), mock.patch.object(os, "system", side_effect=forbidden):
            self.assertEqual([summarize(**case) for case in cases], expected)

    def test_summarize_prints_nothing_and_raises_no_warning(self) -> None:
        # "8and" 같은 토막은 Python 파서에 넘기면 SyntaxWarning 을 낸다(상수식 풀이·Python 소스 파싱)
        c_code = "void f(float* a, int n) { for (int i = 0; i < n; i += 8and) { for (int j = i; j < n; j++) a[j] = 0; } }"
        py_code = "def f(x, n):\n    for i in range(0, n, 8):\n        x[i] = 0 if 1else 2\n    return '\\d'\n"
        out, err = io.StringIO(), io.StringIO()
        with warnings.catch_warnings(record=True) as caught, contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            warnings.simplefilter("always")
            self.assertTrue(c(c_code)["tiling"]["detected"])
            self.assertEqual(py(py_code)["loops"], [nest(("i", "n"))])
            self.assertEqual(numba(py_code)["loops"], [nest(("i", "n"))])
            summarize(language="halide_manual", backend="halide", code=py_code)
        self.assertEqual((caught, out.getvalue(), err.getvalue()), ([], "", ""))

    def test_kernel_code_is_never_executed(self) -> None:
        marker = "PERF_SUMMARY_TEST_EXECUTED"
        code = f"import os\nos.environ[{marker!r}] = '1'\nraise SystemExit(3)\n"
        self.assertNotIn(marker, os.environ)
        for backend in ("python", "numba", "halide"):
            summarize(language=backend, backend=backend, code=code)
        self.assertNotIn(marker, os.environ)


class CLoopNestTest(unittest.TestCase):
    def test_dummy_gemm_ikj_without_a_build_log(self) -> None:
        self.assertEqual(c_fixture("gemm_ikj", "c_seq"), expect("c_seq", loops=GEMM_IKJ_LOOPS, decisions_by_llm=["loop_order"]))

    def test_source_that_does_not_compile_is_still_summarized(self) -> None:
        self.assertEqual(c_fixture("gemm_ijk_missing_semicolon", "c_seq"), expect("c_seq", loops=GEMM_IJK_LOOPS, decisions_by_llm=["loop_order"]))

    def test_induction_variable_and_bound_forms(self) -> None:
        out = c("""
void f(float* a, const float* end, const int* row_ptr, int n) {
    for (int i = n - 1; i >= 0; i--) a[i] = 0;
    for (float* p = a; p != end; ++p) *p = 1;
    for (int i = 0, j = n - 1; i < j; i++, j--) a[i] = a[j];
    for (;;) { break; }
    for (int i = 0; i < n && a[i] > 0; i += 2) a[i] = 2;
    int k = 0;
    for (; k + 3 < n; k += 4) a[k] = 3;
    for (size_t q = row_ptr[0]; row_ptr[n] > q; q = q + 1) a[q] = 4;
    for (int len = 1; len <= n; len <<= 1) a[len] = 5;
}""")
        self.assertEqual(out["loops"], [
            nest(("i", "n - 1")),                # 내림차순: 상한은 시작값
            nest(("p", "end")),
            nest(("i", "j")),
            nest(("?", "")),                     # 머리가 비었다: 변수도 상한도 모른다
            nest(("i", "i < n && a[i] > 0")),    # 복합 조건은 조건식 전체
            nest(("k", "k + 3 < n")),            # 변수만 있는 비교가 아니다
            nest(("q", "row_ptr[n]")),           # 변수가 오른쪽에 있는 비교
            nest(("len", "n")),
        ])
        self.assertEqual(out["tiling"], {"detected": False, "tile_sizes": []})
        self.assertEqual(out["decisions_by_llm"], [])

    def test_braceless_bodies_nest_and_siblings_do_not(self) -> None:
        out = c("""
void f(float* a, int n, int m) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            if (a[j] > 0) a[i] += a[j]; else
                for (int k = 0; k < 3; k++) a[k] = 0;
    for (int t = 0; t < n; t++) a[t] = 1;
    do { for (int u = 0; u < m; u++) a[u] = 2; } while (0);
}""")
        self.assertEqual(out["loops"], [nest(("i", "n"), ("j", "m"), ("k", "3")), nest(("t", "n")), nest(("u", "m"))])

    def test_for_inside_strings_and_comments_is_not_a_loop(self) -> None:
        code = ('/* for (int i = 0; i < n; i++) */\n'
                'void f(float* x, int n) {\n'
                '    const char* s = "for (int j = 0; j < n; j++) {";\n'
                "    char q = '{'; // for (k = 0; k < n; k++)\n"
                '    x[0] = (float)n; (void)s; (void)q;\n'
                '}\n')
        self.assertEqual(c(code), expect("c_seq"))

    def test_macro_that_hides_a_loop_makes_the_structure_unknown(self) -> None:
        code = "#define FOR_EACH(i, n) for (int i = 0; i < (n); i++)\nvoid f(float* x, int n) {\n    FOR_EACH(i, n) x[i] = 0.0f;\n}\n"
        self.assertEqual(c(code), expect("c_seq", **UNREAD_LOOPS, notes="macro may hide a loop: FOR_EACH"))
        # 그 줄의 루프가 벡터화됐다는 리마크가 있어도 어느 루프인지 모른다: 상태는 unknown, 컴파일러가 벡터화했다는 사실만 남는다
        log = "kernel.c:3:5: optimized: loop vectorized using 32 byte vectors\n"
        self.assertEqual(c(code, log), expect(
            "c_seq", **UNREAD_LOOPS, vectorized={"source": "compiler_remark"}, decisions_by_compiler=["vectorization"],
            notes="macro may hide a loop: FOR_EACH; gcc: 1 'loop vectorized' remark(s) not matched to a for loop"))
        # 지시문이 그 매크로에 붙어 있으면 병렬 영역이 있다는 것까지만 안다
        pragma = code.replace("    FOR_EACH", "    #pragma omp parallel for\n    FOR_EACH")
        self.assertEqual(c(pragma, language="c_openmp"), expect(
            "c_openmp", **UNREAD_LOOPS,
            parallel={"kind": "openmp", "regions": 1, "placement": "unknown", "constructs": ["parallel for"]},
            decisions_by_llm=["parallel_region"], notes="macro may hide a loop: FOR_EACH"))

    def test_do_while_zero_macro_and_unused_loop_macro_hide_nothing(self) -> None:
        code = ("#define SWAP(a, b) do { float t_ = (a); (a) = (b); (b) = t_; } while (0)\n"
                "#define UNUSED_LOOP(i, n) for (int i = 0; i < (n); i++)\n"
                "void f(float* a, int n) { for (int i = 0; i < n / 2; i++) SWAP(a[i], a[n - 1 - i]); }\n")
        log = "kernel.c:3:45: missed: couldn't vectorize loop\n"
        self.assertEqual(c(code, log), expect("c_seq", loops=[nest(("i", "n / 2"))], vectorized={"status": "no", "source": "compiler_remark"}))

    def test_unbalanced_brackets_omit_the_loops(self) -> None:
        code = "void f(float* x, int n) {\n    for (int i = 0; i < n; i++) {\n        x[i] = 0;\n"
        self.assertEqual(c(code), expect("c_seq", **UNREAD_LOOPS, notes="unbalanced brackets: loop nests omitted"))
        self.assertEqual(rust("pub fn f(x: &mut [f32]) {\n    for i in 0..x.len() {\n        x[i] = 0.0;\n"), expect(
            "rust", **UNREAD_LOOPS, notes="unbalanced brackets: loop nests omitted; unsafe blocks: 0"))

    def test_loops_in_helper_functions_are_separate_nests(self) -> None:
        self.assertEqual(c_fixture("jacobi2d_omp", "c_openmp")["loops"], [nest(("y", "H - 1"), ("x", "W - 1")), nest(("t", "T"))])

    def test_crlf_line_endings_change_nothing(self) -> None:
        code, log = read("c/jacobi2d_omp.c"), read("c/jacobi2d_omp.build.log")
        self.assertEqual(c(code.replace("\n", "\r\n"), log.replace("\n", "\r\n"), "c_openmp"), c(code, log, "c_openmp"))
        blis = read("c/gemm_blis_avx2.c")
        self.assertEqual(c(blis.replace("\n", "\r\n")), c(blis))


class COpenMPTest(unittest.TestCase):
    def test_parallel_for_on_the_outer_loop(self) -> None:
        self.assertEqual(c_fixture("gemm_omp_outer", "c_openmp"), expect(
            "c_openmp", loops=GEMM_IKJ_LOOPS,
            parallel={"kind": "openmp", "regions": 1, "loops": ["i"], "placement": "outer", "constructs": ["parallel for schedule(static)"]},
            decisions_by_llm=["loop_order", "parallel_region"]))

    def test_parallel_for_inside_a_while_loop_is_not_outer(self) -> None:
        # 단계 동기식 BFS: 병렬 for 가 순차 while 안에서 단계마다 한 번씩 돈다
        inner = ("        #pragma omp parallel for\n        for (int f = 0; f < fn; f++) next[f] = cur[f] + 1;\n"
                 "        fn = fn / 2;\n")
        head = "void bfs(int* cur, int* next, int fn) {\n"
        variants = {
            "while": head + "    while (fn > 0) {\n" + inner + "    }\n}\n",
            "do while": head + "    do {\n" + inner + "    } while (fn > 0);\n}\n",
            "for(;;)": head + "    for (;;) {\n        if (fn <= 0) break;\n" + inner + "    }\n}\n",
        }
        for name, code in variants.items():
            with self.subTest(name):
                out = c(code, language="c_openmp")
                self.assertEqual((out["parallel"]["placement"], out["parallel"]["loops"]), ("inner", ["f"]))
        out = c(variants["while"], language="c_openmp")
        self.assertEqual(out["loops"], [nest(("f", "fn"))], "while 은 루프 목록에 넣지 않는다")
        self.assertEqual(out["notes"], "parallel loop inside a while/do loop (runs once per iteration)")
        # do { … } while (0) 는 한 번만 도는 관용구다: placement 를 바꾸지 않는다
        once = c(head + "    do {\n" + inner + "    } while (0);\n}\n", language="c_openmp")
        self.assertEqual((once["parallel"]["placement"], once["notes"]), ("outer", ""))

    def test_parallel_for_on_the_inner_loop(self) -> None:
        self.assertEqual(c_fixture("gemm_omp_inner", "c_openmp"), expect(
            "c_openmp", loops=GEMM_IJK_LOOPS,
            parallel={"kind": "openmp", "regions": 1, "loops": ["k"], "placement": "inner", "constructs": ["parallel for"]},
            decisions_by_llm=["loop_order", "parallel_region"]))

    def test_collapse_and_simd_on_a_blocked_nest(self) -> None:
        self.assertEqual(c_fixture("gemm_blocked_omp_collapse", "c_openmp"), expect(
            "c_openmp",
            loops=[nest(("ii", "M"), ("jj", "N"), ("kk", "K"), ("i", "MIN(ii + BLOCK, M)"), ("k", "MIN(kk + BLOCK, K)"), ("j", "MIN(jj + BLOCK, N)"))],
            parallel={"kind": "openmp", "regions": 1, "collapse": True, "loops": ["ii", "jj"], "placement": "outer",
                      "constructs": ["parallel for collapse(2) schedule(static)", "simd"]},
            tiling={"detected": True, "tile_sizes": [64]},
            library_calls=["memset"],
            decisions_by_llm=["loop_order", "tiling", "vectorization", "parallel_region"]))

    def test_parallel_region_with_worksharing_loop_and_critical(self) -> None:
        self.assertEqual(c_fixture("histogram_omp_private", "c_openmp", spec=HISTOGRAM), expect(
            "c_openmp", loops=[nest(("b", "n_bins")), nest(("i", "n")), nest(("b", "n_bins"))],
            parallel={"kind": "openmp", "regions": 1, "loops": ["i"], "placement": "outer",
                      "constructs": ["parallel", "for schedule(static) nowait", "critical"]},
            memory={"temporaries": 1}, library_calls=["calloc"],
            decisions_by_llm=["parallel_region", "memory_layout"]))

    def test_simd_reduction_on_one_loop(self) -> None:
        self.assertEqual(c_fixture("dot_omp_simd_reduction", "c_openmp", log=True, spec=DOT), expect(
            "c_openmp", loops=[nest(("i", "n"))],
            parallel={"kind": "openmp", "regions": 1, "reduction": True, "loops": ["i"], "placement": "outer",
                      "constructs": ["parallel for simd reduction(+:s) schedule(static)"]},
            vectorized={"status": "yes", "source": "compiler_remark", "loops": ["i"]},
            decisions_by_llm=["vectorization", "parallel_region", "reduction", "accumulation_precision"],
            notes="gcc: vectorized i@4 (32-byte)"))

    def test_regions_clauses_continuation_lines_and_mixed_placement(self) -> None:
        out = c("""
#include <omp.h>
void f(float* a, int n, int m, int k) {
    omp_set_num_threads(4);
    #pragma omp parallel num_threads(4) default(none) shared(a, n, m, k)
    {
        #pragma omp for collapse(3) schedule(dynamic, 16) nowait
        for (int i = 0; i < n; i++)
            for (int j = 0; j < m; j++)
                for (int l = 0; l < k; l++) {
                    #pragma omp atomic
                    a[i] += 1.0f;
                }
    }
    for (int t = 0; t < 10; t++) {
        #pragma omp parallel for \\
            schedule(static) reduction(max: a[0])
        for (int i = 0; i < n; i++) a[i] *= 2;
    }
}""", language="c_openmp")
        self.assertEqual(out["parallel"], {
            "kind": "openmp", "regions": 2, "collapse": True, "reduction": True, "loops": ["i", "j", "l"], "placement": "mixed",
            "constructs": ["parallel num_threads(4) default(none) shared(a, n, m, k)", "for collapse(3) schedule(dynamic, 16) nowait",
                           "atomic", "parallel for schedule(static) reduction(max: a[0])"]})
        self.assertEqual(out["library_calls"], ["omp_set_num_threads"])
        self.assertEqual(out["decisions_by_llm"], ["loop_order", "parallel_region", "reduction"])

    def test_pragmas_ignored_by_the_compiler_are_not_counted(self) -> None:
        # c_seq 로 빌드한 로그: -fopenmp 가 없으면 gcc 는 -Wall 에서 지시문을 무시한다고 경고한다
        log = ("kernel.c:6: warning: ignoring '#pragma omp parallel' [-Wunknown-pragmas]\n"
               "    6 | #pragma omp parallel for schedule(static)\n"
               "kernel.c:10: warning: ignoring '#pragma omp simd' [-Wunknown-pragmas]\n"
               "   10 | #pragma omp simd\n" + read("c/jacobi2d_omp.build.log"))
        self.assertEqual(c(read("c/jacobi2d_omp.c"), log), expect(
            "c_seq", loops=[nest(("y", "H - 1"), ("x", "W - 1")), nest(("t", "T"))],
            parallel={"constructs": ["parallel for schedule(static)", "simd"]},
            vectorized={"status": "yes", "source": "compiler_remark", "loops": ["x"]},
            memory={"temporaries": 1}, library_calls=["memcpy", "aligned_alloc"],
            decisions_by_llm=["loop_order", "memory_layout"],
            decisions_by_compiler=["vectorization"],       # simd 요청이 무효라 벡터화는 컴파일러가 스스로 내린 결정이다
            notes="OpenMP pragmas ignored by the compiler (built without -fopenmp); gcc: vectorized x@11 (32-byte)"))
        utf8_quotes = "kernel.c:6: warning: ignoring ‘#pragma omp parallel’ [-Wunknown-pragmas]\n"
        self.assertEqual(c(read("c/jacobi2d_omp.c"), utf8_quotes)["parallel"]["kind"], "none")

    def test_simd_reduction_without_a_parallel_region(self) -> None:
        out = c("void f(const float* x, float* out, int n) {\n    float s = 0;\n    #pragma omp simd reduction(+:s)\n"
                "    for (int i = 0; i < n; i++) s += x[i];\n    out[0] = s;\n}\n", language="c_openmp")
        self.assertEqual(out["parallel"], expect("c")["parallel"] | {"constructs": ["simd reduction(+:s)"]})
        self.assertEqual(out["decisions_by_llm"], ["vectorization", "reduction"])

    def test_explicit_threads(self) -> None:
        out = c("#include <pthread.h>\nstatic void* work(void* arg) { return arg; }\n"
                "void f(float* a) { pthread_t t[4]; for (int i = 0; i < 4; i++) pthread_create(&t[i], 0, work, a); }\n")
        self.assertEqual(out["parallel"], {"kind": "threads", "regions": 1, "collapse": False, "reduction": False, "loops": [],
                                           "placement": "region", "constructs": ["pthread_create"]})
        self.assertEqual(out["library_calls"], ["pthread_create"])
        self.assertEqual(out["decisions_by_llm"], ["parallel_region"])


class CVectorizationTest(unittest.TestCase):
    def test_compiler_vectorized_the_main_loop_but_said_nothing_about_the_fill_loop(self) -> None:
        # 채우기 루프(j@7)는 gcc 가 memset 으로 바꿔 벡터화기 리마크가 없다: 판정 근거가 없으므로 not vectorized 로 세지 않는다
        self.assertEqual(c_fixture("gemm_ikj", "c_seq", log=True), expect(
            "c_seq", loops=GEMM_IKJ_LOOPS,
            vectorized={"status": "yes", "source": "compiler_remark", "loops": ["j"]},
            decisions_by_llm=["loop_order"], decisions_by_compiler=["vectorization"],
            notes="gcc: vectorized j@12 (32-byte); "
                  "gcc: no vectorizer verdict for j@7 (not counted; possibly replaced by a library call such as memset)"))

    def test_a_loop_is_not_vectorized_only_with_a_missed_remark_and_no_evidence_is_unknown(self) -> None:
        code = ("void f(float* restrict a, const float* restrict b, int n) {\n"
                "    for (int i = 0; i < n; i++)\n        a[i] = 0.0f;\n"
                "    for (int j = 0; j < n; j++) a[j] += b[j];\n}\n")
        silent = "gcc: no vectorizer verdict for i@2 (not counted; possibly replaced by a library call such as memset)"
        # 근거 있는 루프(j)가 벡터화되지 않았다 → no; 근거 없는 i 는 빠진다
        out = c(code, "kernel.c:4:23: missed: couldn't vectorize loop\n")
        self.assertEqual((out["vectorized"]["status"], out["notes"]), ("no", silent))
        # 루프 머리의 missed 리마크도 "not vectorized:" 사유도 아닌 본문 줄의 리마크(memset 으로 바뀐 문장)는 근거가 아니다
        out = c(code, "kernel.c:4:23: optimized: loop vectorized using 32 byte vectors\n"
                      "kernel.c:3:14: missed: statement clobbers memory: __builtin_memset (a_5(D), 0, _3);\n")
        self.assertEqual(out["vectorized"]["status"], "yes")
        # 루프 머리의 missed 리마크는 근거다 → 일부만 벡터화
        out = c(code, "kernel.c:4:23: optimized: loop vectorized using 32 byte vectors\nkernel.c:2:23: missed: couldn't vectorize loop\n")
        self.assertEqual(out["vectorized"]["status"], "partial")
        # 가장 안쪽 루프 어디에도 판정 리마크가 없으면 unknown (no·partial 이 아니다)
        out = c(code, "kernel.c:1:6: optimized: basic block part vectorized using 16 byte vectors\n")
        self.assertEqual(out["vectorized"], {"status": "unknown", "source": "compiler_remark", "loops": []})

    def test_remarks_without_any_vectorized_loop_mean_no(self) -> None:
        self.assertEqual(c_fixture("gemm_ijk", "c_seq", log=True), expect(
            "c_seq", loops=GEMM_IJK_LOOPS, vectorized={"status": "no", "source": "compiler_remark"},
            decisions_by_llm=["loop_order"], notes="gcc: not vectorized k@6 (unsupported control flow in loop)"))
        self.assertEqual(c_fixture("spmv_csr_omp", "c_openmp", log=True), expect(
            "c_openmp", loops=[nest(("i", "n_rows"), ("p", "row_ptr[i + 1]"))],
            parallel={"kind": "openmp", "regions": 1, "loops": ["i"], "placement": "outer", "constructs": ["parallel for schedule(static)"]},
            vectorized={"status": "no", "source": "compiler_remark"},
            decisions_by_llm=["loop_order", "parallel_region"],
            notes="gcc: not vectorized p@8 (complicated access pattern)"))

    def test_without_a_usable_log_the_status_is_unknown_not_no(self) -> None:
        code = read("c/gemm_ijk.c")
        for log in ("", "kernel.c: In function 'gemm':\nkernel.c:5:19: warning: unused variable 'x' [-Wunused-variable]\n",
                    "/usr/lib/gcc/x86_64-linux-gnu/15/include/avxintrin.h:900:1: missed: not vectorized: complicated access pattern.\n"):
            self.assertEqual(c(code, log)["vectorized"], {"status": "unknown", "source": "none", "loops": []}, log)

    def test_omp_simd_request_is_the_llm_decision_and_call_loops_are_not_leaves(self) -> None:
        self.assertEqual(c_fixture("jacobi2d_omp", "c_openmp", log=True), expect(
            "c_openmp", loops=[nest(("y", "H - 1"), ("x", "W - 1")), nest(("t", "T"))],
            parallel={"kind": "openmp", "regions": 1, "loops": ["y"], "placement": "outer", "constructs": ["parallel for schedule(static)", "simd"]},
            vectorized={"status": "yes", "source": "compiler_remark", "loops": ["x"]},     # t 루프는 step() 을 부를 뿐이다
            memory={"temporaries": 1}, library_calls=["memcpy", "aligned_alloc"],
            decisions_by_llm=["loop_order", "vectorization", "parallel_region", "memory_layout"],
            notes="gcc: vectorized x@11 (32-byte)"))

    def test_basic_block_vectorization_inside_an_unrolled_loop_is_partial(self) -> None:
        self.assertEqual(c_fixture("dot_unrolled", "c_seq", log=True, spec=DOT), expect(
            "c_seq", loops=[nest(("i", "i + 3 < n")), nest(("i", "n"))],
            vectorized={"status": "partial", "source": "compiler_remark"},
            decisions_by_llm=["accumulation_precision"], decisions_by_compiler=["vectorization"],
            notes="gcc: basic block vectorized in i@4"))

    def test_the_same_loop_reported_several_times_counts_once(self) -> None:
        code = "void f(float* restrict a, const float* restrict b, int n) {\n    for (int i = 0; i < n; i++)\n        a[i] += b[i];\n}\n"
        log = ("kernel.c:2:23: missed: couldn't vectorize loop\n"
               "kernel.c:2:23: missed: not vectorized: unsupported control flow in loop.\n"
               "kernel.c:2:23: optimized: loop vectorized using 32 byte vectors\n"
               "kernel.c:2:23: optimized:  loop vectorized using 16 byte vectors\r\n")
        self.assertEqual(c(code, log), expect(
            "c_seq", loops=[nest(("i", "n"))], vectorized={"status": "yes", "source": "compiler_remark", "loops": ["i"]},
            decisions_by_compiler=["vectorization"], notes="gcc: vectorized i@2 (32-byte)"))

    def test_remarks_that_point_into_a_loop_body_belong_to_that_loop(self) -> None:
        code = "void f(float* a, const float* b, int n) {\n    for (int i = 0; i < n; i++)\n        a[i] += b[i];\n}\n"
        log = ("kernel.c:3:9: optimized: loop versioned for vectorization because of possible aliasing\n"
               "kernel.c:2:23: optimized: loop vectorized using 32 byte vectors\n")
        self.assertEqual(c(code, log)["notes"], "gcc: vectorized i@2 (32-byte); gcc: versioned for aliasing i@2")
        # "loop vectorized" 만은 루프 머리의 줄에 있어야 그 루프의 것으로 센다: 대응시키지 못했으므로 i 는 판정 근거가 없다
        out = c(code, "kernel.c:3:9: optimized: loop vectorized using 32 byte vectors\n")
        self.assertEqual(out["vectorized"], {"status": "unknown", "source": "compiler_remark", "loops": []})

    def test_a_vectorized_outer_loop_covers_its_inner_loops(self) -> None:
        code = ("void f(float* restrict a, const float* restrict b, int n) {\n    for (int i = 0; i < n; i++)\n"
                "        for (int k = 0; k < 4; k++)\n            a[4 * i + k] += b[k];\n}\n")
        out = c(code, "kernel.c:2:23: optimized: loop vectorized using 32 byte vectors\n")
        self.assertEqual(out["vectorized"], {"status": "yes", "source": "compiler_remark", "loops": ["i"]})

    def test_an_unvectorized_while_loop_keeps_the_status_from_being_yes(self) -> None:
        code = ("void reverse_copy(const int* keys, int* out, int n) {\n    for (int i = 0; i < n; i++) out[i] = keys[i];\n"
                "    int lo = 0, hi = n - 1;\n    while (lo < hi) {\n        int t = out[lo]; out[lo] = out[hi]; out[hi] = t;\n"
                "        lo++; hi--;\n    }\n}\n")
        log = "kernel.c:2:23: optimized: loop vectorized using 32 byte vectors\nkernel.c:4:15: missed: couldn't vectorize loop\n"
        out = c(code, log)
        self.assertEqual(out["loops"], [nest(("i", "n"))])              # while 은 루프 목록에 없다
        self.assertEqual(out["vectorized"], {"status": "partial", "source": "compiler_remark", "loops": ["i"]})
        self.assertEqual(out["notes"], "gcc: vectorized i@2 (32-byte); gcc: 1 loop(s) outside any for statement not vectorized")
        self.assertEqual(c(code, log.splitlines()[0])["vectorized"]["status"], "yes")

    def test_loops_sharing_a_line_are_told_apart_by_column(self) -> None:
        code = "void f(float* a, float* b, int n) {\n    for (int i = 0; i < n; i++) a[i] = 0; for (int j = 0; j < n; j++) b[j] = 1;\n}\n"
        out = c(code, "kernel.c:2:61: optimized: loop vectorized using 32 byte vectors\nkernel.c:2:23: missed: couldn't vectorize loop\n")
        self.assertEqual(out["vectorized"], {"status": "partial", "source": "compiler_remark", "loops": ["j"]})
        self.assertEqual(out["notes"], "gcc: vectorized j@2 (32-byte)")
        # 열이 어느 루프와도 맞지 않으면 대응시키지 않는다(그러면 두 루프 모두 판정 근거가 없다)
        out = c(code, "kernel.c:2:40: optimized: loop vectorized using 32 byte vectors\n")
        self.assertEqual(out["vectorized"], {"status": "unknown", "source": "compiler_remark", "loops": []})
        self.assertEqual(out["notes"], "gcc: no vectorizer verdict for i@2, j@2 (not counted; possibly replaced by a library call "
                                       "such as memset); gcc: 1 'loop vectorized' remark(s) not matched to a for loop")

    def test_gcc_columns_count_a_tab_as_eight(self) -> None:
        code = "void f(float* a, float* b, int n) {\n\tfor (int i = 0; i < n; i++) a[i] = 0; for (int j = 0; j < n; j++) b[j] = 1;\n}\n"
        out = c(code, "kernel.c:2:27: optimized: loop vectorized using 32 byte vectors\n")
        self.assertEqual(out["vectorized"]["loops"], ["i"])

    def test_hand_written_intrinsics_without_a_log(self) -> None:
        self.assertEqual(c_fixture("gemm_blis_avx2", "c_seq"), expect(
            "c_seq", loops=BLIS_LOOPS,
            vectorized={"source": "intrinsics", "loops": ["p"]},      # 나머지 루프는 로그 없이 알 수 없다
            tiling={"detected": True, "tile_sizes": [6, 16, 4080, 256, 96]},
            memory={"temporaries": 3}, library_calls=["memset", "aligned_alloc"],
            decisions_by_llm=["loop_order", "tiling", "vectorization", "memory_layout"],
            notes=BLIS_INTRINSICS_NOTE))

    def test_hand_written_intrinsics_with_a_log(self) -> None:
        self.assertEqual(c_fixture("gemm_blis_avx2", "c_seq", log=True, spec=GEMM), expect(
            "c_seq", loops=BLIS_LOOPS,
            vectorized={"status": "partial", "source": "intrinsics", "loops": ["c", "p", "q"]},
            tiling={"detected": True, "tile_sizes": [6, 16, 4080, 256, 96]},
            memory={"temporaries": 3}, library_calls=["memset", "aligned_alloc"],
            decisions_by_llm=["loop_order", "tiling", "vectorization", "memory_layout"],
            decisions_by_compiler=["vectorization"],
            notes="gcc: vectorized c@36 (32-byte), q@125 (32-byte); gcc: not vectorized r@20 (complicated access pattern); "
                  "gcc: no vectorizer verdict for r@22, c@38 (not counted; possibly replaced by a library call such as memset); "
                  "gcc: versioned for aliasing q@125; " + BLIS_INTRINSICS_NOTE))

    def test_every_leaf_loop_hand_vectorized_is_yes_even_without_a_log(self) -> None:
        code = ("#include <immintrin.h>\nvoid f(const float* x, float* y, int n) {\n"
                "    for (int i = 0; i + 8 <= n; i += 8) _mm256_storeu_ps(y + i, _mm256_loadu_ps(x + i));\n}\n")
        out = c(code)
        self.assertEqual(out["vectorized"], {"status": "yes", "source": "intrinsics", "loops": ["i"]})
        self.assertEqual(out["decisions_by_llm"], ["vectorization"])
        self.assertEqual(out["tiling"], {"detected": False, "tile_sizes": []})       # SIMD 폭 걸음은 타일링이 아니다

    def test_gcc_vector_extension_counts_but_a_variable_named_vector_size_does_not(self) -> None:
        self.assertEqual(c("typedef float v8sf __attribute__((vector_size(32)));\nvoid f(v8sf* a) { a[0] = a[0] + a[0]; }\n")["decisions_by_llm"],
                         ["vectorization"])
        self.assertEqual(c("void f(float* a, int n) { int vector_size = 8; a[0] = vector_size * n; }\n")["decisions_by_llm"], [])


class CTilingMemoryTest(unittest.TestCase):
    def test_strided_loops_without_a_dependent_inner_loop_are_not_tiling(self) -> None:
        fft = """
void fft(float* out, int n) {
    for (int len = 2; len <= n; len <<= 1)
        for (int i = 0; i < n; i += len)
            for (int j = 0; j < len / 2; j++) {
                float u = out[2 * (i + j)], v = out[2 * (i + j + len / 2)];
                out[2 * (i + j)] = u + v;
            }
}"""
        self.assertEqual(c(fft)["tiling"], {"detected": False, "tile_sizes": []})
        self.assertEqual(c_fixture("dot_unrolled", "c_seq")["tiling"], {"detected": False, "tile_sizes": []})

    def test_block_size_from_const_enum_and_runtime_variable(self) -> None:
        out = c("""
static const int BS = 32;
enum { NB = 4 * 16 };
void f(float* C, int n, int bs) {
    for (int ii = 0; ii < n; ii += BS) {
        int iend = ii + BS < n ? ii + BS : n;
        for (int i = ii; i < iend; i++) C[i] = 0;
    }
    for (int b = 0; b < n; b += NB) { for (int i = b; i < (b + NB < n ? b + NB : n); i++) C[i] = 1; }
    for (int b = 0; b < n; b += bs) { int e = b + bs; for (int i = b; i < e && i < n; i++) C[i] = 2; }
    for (int b = 0; b < n; b += 16UL) { for (int i = b; i < b + 16; i++) C[i] = 3; }
}""")
        self.assertEqual(out["tiling"], {"detected": True, "tile_sizes": [32, 64, 16]})     # 실행 시간에 정해지는 bs 는 값을 싣지 않는다
        self.assertEqual(out["decisions_by_llm"], ["loop_order", "tiling"])

    def test_transposed_copy_and_double_accumulation(self) -> None:
        self.assertEqual(c_fixture("gemm_transposed_b", "c_seq", spec=GEMM), expect(
            "c_seq", loops=[nest(("k", "K"), ("j", "N")), nest(("i", "M"), ("j", "N"), ("k", "K"))],
            memory={"temporaries": 1, "layout_transposed": True}, library_calls=["malloc"],
            decisions_by_llm=["loop_order", "memory_layout", "accumulation_precision"]))

    def test_plain_copy_and_same_array_mirror_are_not_a_transposed_layout(self) -> None:
        packed = "void f(const float* B, float* Bp, int n, int k) { for (int p = 0; p < k; p++) for (int j = 0; j < n; j++) Bp[p * n + j] = B[p * n + j]; }"
        mirror = "void f(float* L, int n) { for (int i = 0; i < n; i++) for (int j = 0; j < i; j++) L[j * n + i] = L[i * n + j]; }"
        self.assertFalse(c(packed)["memory"]["layout_transposed"])
        self.assertFalse(c(mirror)["memory"]["layout_transposed"])

    def test_accumulation_precision_needs_a_float_task(self) -> None:
        code = "void f(const int* keys, int* hist, int n) { double s = 0; for (int i = 0; i < n; i++) s += keys[i]; hist[0] = (int)s; }"
        self.assertEqual(c(code)["decisions_by_llm"], ["accumulation_precision"])
        self.assertEqual(c(code, spec=HISTOGRAM)["decisions_by_llm"], [])
        float_only = "void f(const float* x, float* out, int n) { float s = 0; for (int i = 0; i < n; i++) s += x[i]; out[0] = s; }"
        self.assertEqual(c(float_only, spec=DOT)["decisions_by_llm"], [])

    def test_temporaries_count_allocations_and_symbolic_arrays(self) -> None:
        out = c("""
#define MC 96
void f(const float* a, float* out, int n) {
    float scratch[8];
    float pack[MC * 4];
    static const float table[MC] = {0};
    float vla[n];
    float *heap = malloc(n * sizeof(float)), *zero = calloc(n, sizeof(float));
    out[0] = a[0] + scratch[0] + pack[0] + table[0] + vla[0] + heap[0] + zero[0];
    free(heap); free(zero);
}""")
        self.assertEqual(out["memory"], {"temporaries": 4, "layout_transposed": False})     # pack, vla, malloc, calloc
        self.assertEqual(out["library_calls"], ["malloc", "calloc"])
        self.assertEqual(out["decisions_by_llm"], ["memory_layout"])

    def test_blas_and_qsort_are_library_delegation(self) -> None:
        out = c("""
void gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, A, K, B, N, 0.0f, C, N);
}
void sort(const int* keys, int* out, int n) { memcpy(out, keys, n * sizeof(int)); qsort(out, n, sizeof(int), cmp); }""")
        self.assertEqual(out, expect("c_seq", library_calls=["cblas_sgemm", "memcpy", "qsort"], decisions_by_library=["library_delegation"]))
        self.assertEqual(c_fixture("jacobi2d_omp", "c_openmp")["decisions_by_library"], [])      # memcpy 만으로는 위임이 아니다


class PythonTest(unittest.TestCase):
    def test_matmul_into_the_output_buffer(self) -> None:
        want = expect("numpy", library_calls=["np.matmul"], decisions_by_llm=["memory_layout"],
                      decisions_by_library=["library_delegation"], notes="out= used")
        self.assertEqual(py(read("python/gemm_matmul_out.py"), spec=GEMM), want)
        self.assertEqual(py(read("python/gemm_matmul_out.py")), want)

    def test_operator_matmul_assigned_to_a_slice_is_one_temporary(self) -> None:
        want = expect("numpy", memory={"temporaries": 1}, library_calls=["np.matmul"], decisions_by_library=["library_delegation"])
        self.assertEqual(py(read("python/gemm_matmul_scaled.py"), spec=GEMM), want)        # C[:] = (A @ B) * 1.01
        self.assertEqual(py(read("python/gemm_return_product.py"), spec=GEMM), want)       # return A @ B
        self.assertEqual(py("def gemm(A, B, C, M, N, K):\n    C[:] = A @ B\n"), want)

    def test_pure_python_triple_loop(self) -> None:
        want = expect("numpy", loops=GEMM_IJK_LOOPS, vectorized={"status": "no", "source": "interpreter"},
                      decisions_by_llm=["loop_order"], notes="element loops: i, j, k")
        self.assertEqual(py(read("python/gemm_triple_loop.py"), spec=GEMM), want)
        self.assertEqual(py(read("python/gemm_triple_loop.py")), want)

    def test_scalar_reduction_makes_no_temporary(self) -> None:
        want = expect("numpy", library_calls=["np.dot"], decisions_by_library=["library_delegation"])
        self.assertEqual(py(read("python/dot_np_dot.py"), spec=DOT), want)
        self.assertEqual(py(read("python/dot_np_dot.py")), want)
        # 같은 np.dot 이라도 2차원 인자면 배열을 돌려준다(차원 수는 과제 명세에서 안다)
        self.assertEqual(py("import numpy as np\ndef gemm(A, B, C, M, N, K):\n    C[:] = np.dot(A, B)\n", spec=GEMM)["memory"]["temporaries"], 1)
        self.assertEqual(py("import numpy as np\ndef gemm(A, B, C, M, N, K):\n    np.dot(A, B, out=C)\n", spec=GEMM)["memory"]["temporaries"], 0)

    def test_slice_arithmetic_in_a_time_loop(self) -> None:
        self.assertEqual(py(read("python/jacobi2d_slices.py")), expect(
            "numpy", loops=[nest(("_", "T"))], memory={"temporaries": 3}, library_calls=["ndarray.copy"],
            decisions_by_llm=["memory_layout"], decisions_by_library=["library_delegation"]))

    def test_gather_scatter_formulation(self) -> None:
        self.assertEqual(py(read("python/spmv_csr_bincount.py")), expect(
            "numpy", memory={"temporaries": 6},
            library_calls=["np.repeat", "np.arange", "np.diff", "np.bincount", "ndarray.astype"],
            decisions_by_llm=["accumulation_precision"], decisions_by_library=["library_delegation"]))

    def test_row_loop_with_a_library_body_and_a_thread_pool(self) -> None:
        out = py("""
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def gemm(A, B, C, M, N, K):
    def rows(lo, hi):
        for i in range(lo, hi):
            C[i, :] = A[i, :] @ B
    step = (M + 3) // 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda lo: rows(lo, min(lo + step, M)), range(0, M, step)))
""")
        self.assertEqual(out, expect(
            "numpy", loops=[nest(("i", "hi"))],
            parallel={"kind": "threads", "regions": 1, "placement": "region", "constructs": ["ThreadPoolExecutor"]},
            memory={"temporaries": 1}, library_calls=["np.matmul", "concurrent.futures.ThreadPoolExecutor"],
            decisions_by_llm=["parallel_region"], decisions_by_library=["library_delegation"],
            notes="element loops: i"))            # 라이브러리에 넘긴 부분이 있어 벡터화 상태는 unknown 이다

    def test_blocked_slices_are_tiling_not_element_loops(self) -> None:
        out = py("""
import numpy as np
BS = 128

def gemm(A, B, C, M, N, K):
    C[...] = 0.0
    for ii in range(0, M, BS):
        ie = min(ii + BS, M)
        for kk in range(0, K, BS):
            ke = min(kk + BS, K)
            C[ii:ie, :] += A[ii:ie, kk:ke] @ B[kk:ke, :]
""")
        self.assertEqual(out, expect(
            "numpy", loops=[nest(("ii", "M"), ("kk", "K"))], tiling={"detected": True, "tile_sizes": [128]},
            memory={"temporaries": 1}, library_calls=["np.matmul"],
            decisions_by_llm=["loop_order", "tiling"], decisions_by_library=["library_delegation"]))

    def test_explicit_memory_precision_and_algorithm_choices(self) -> None:
        out = py("""
import numpy as np

def sort(keys, out, n):
    out[:] = np.sort(keys, kind="stable")

def gemm(A, B, C, M, N, K):
    Bt = np.ascontiguousarray(B.T)
    np.einsum("ik,jk->ij", A, Bt, out=C, dtype=np.float64, casting="unsafe")
""")
        self.assertEqual(out, expect(
            "numpy", memory={"temporaries": 2, "layout_transposed": True},
            library_calls=["np.sort", "np.ascontiguousarray", "np.einsum"],
            decisions_by_llm=["memory_layout", "algorithm_choice", "accumulation_precision"],
            decisions_by_library=["library_delegation"], notes="out= used"))

    def test_elementwise_math_is_delegation_only_when_applied_to_arrays(self) -> None:
        out = py("""
import numpy as np

def nbody(pos, mass, acc, n):
    d = pos[None, :, :] - pos[:, None, :]
    r2 = (d * d).sum(axis=2) + 1e-4
    acc[:] = (d * (mass[None, :] / (r2 * np.sqrt(r2)))[:, :, None]).sum(axis=1)
""")
        self.assertEqual(out["library_calls"], ["ndarray.sum", "np.sqrt"])
        self.assertEqual(out["decisions_by_library"], ["library_delegation"])
        self.assertEqual(out["memory"]["temporaries"], 8)      # 새 배열을 돌려주는 호출 3 + 배열 산술 식 5
        scalar = py("""
import numpy as np

def nbody(pos, mass, acc, n):
    for i in range(n):
        ax = 0.0
        for j in range(n):
            dx = pos[j, 0] - pos[i, 0]
            r2 = dx * dx + 1e-4
            ax += mass[j] * dx / (r2 * np.sqrt(r2))
        acc[i, 0] = ax
""")
        self.assertEqual(scalar, expect(
            "numpy", loops=[nest(("i", "n"), ("j", "n"))], vectorized={"status": "no", "source": "interpreter"},
            library_calls=["np.sqrt"], decisions_by_llm=["loop_order"], notes="element loops: i, j"))
        self.assertEqual(py("import numpy as np\ndef f(x, y, n):\n    np.sqrt(x, out=y)\n")["decisions_by_library"], ["library_delegation"])
        self.assertEqual(py("import numpy as np\ndef f(x, y, n):\n    y[:] = np.sqrt(x)\n")["memory"]["temporaries"], 1)

    def test_range_forms_and_non_range_iterables(self) -> None:
        out = py("def f(x, y, n):\n    for i in range(n - 1, -1, -1):\n        x[i] = 0\n    for i in range(2, n, 3):\n        x[i] = 1\n"
                 "    for a, b in zip(x, y):\n        pass\n")
        self.assertEqual(out["loops"], [nest(("i", "n - 1")), nest(("i", "n")), nest(("(a, b)", "zip(x, y)"))])

    def test_integer_arithmetic_is_not_array_arithmetic(self) -> None:
        out = py("def gemm(A, B, C, M, N, K):\n    total = M * N + K\n    for i in range(total):\n        pass\n", spec=GEMM)
        self.assertEqual(out, expect("numpy", loops=[nest(("i", "total"))]))

    def test_source_that_does_not_parse_is_unknown(self) -> None:
        out = py("import numpy as np\ndef gemm(A, B, C, M, N, K):\n    for i in range(M):\n        C[i, :] = A[i, :] @")
        self.assertEqual(out, expect("numpy", parallel=UNKNOWN_PARALLEL, tiling={"detected": None},
                                     memory={"temporaries": None, "layout_transposed": None},
                                     notes="source does not parse: SyntaxError: invalid syntax (<unknown>, line 4)"))


class NumbaTest(unittest.TestCase):
    def test_prange_on_the_outer_loop(self) -> None:
        want = expect("numba", loops=GEMM_IKJ_LOOPS,
                      parallel={"kind": "numba", "regions": 1, "loops": ["i"], "placement": "outer", "constructs": ["prange"]},
                      decisions_by_llm=["loop_order", "parallel_region"],
                      notes="element loops: i, j, k; numba gemm: parallel=True, fastmath=True")
        self.assertEqual(numba(read("numba/gemm_prange.py")), want)
        want["notes"] += "; numba diagnostics: 1 parallel loop(s) from prange, 0 from array expressions, 0 fused; threading_layer=omp"
        self.assertEqual(numba(read("numba/gemm_prange.py"), inspect=inspect_of("numba/gemm_prange.import.json"), spec=GEMM), want)

    def test_scalar_accumulation_in_prange_is_a_reduction_inferred_by_numba(self) -> None:
        self.assertEqual(numba(read("numba/dot_prange.py"), inspect=inspect_of("numba/dot_prange.import.json"), spec=DOT), expect(
            "numba", loops=[nest(("i", "n"))],
            parallel={"kind": "numba", "regions": 1, "reduction": True, "loops": ["i"], "placement": "outer", "constructs": ["prange"]},
            decisions_by_llm=["parallel_region", "accumulation_precision"], decisions_by_compiler=["reduction"],
            notes="element loops: i; numba dot: parallel=True; numba diagnostics: 1 parallel loop(s) from prange, "
                  "0 from array expressions, 0 fused; threading_layer=omp"))

    def test_an_accumulator_assigned_inside_the_prange_body_is_not_a_reduction(self) -> None:
        code = ("from numba import njit, prange\n\n@njit(parallel=True)\ndef gemm(A, B, C, M, N, K):\n    for i in prange(M):\n"
                "        for j in range(N):\n            acc = 0.0\n            for k in range(K):\n                acc += A[i, k] * B[k, j]\n"
                "            C[i, j] = acc\n")
        out = numba(code)
        self.assertFalse(out["parallel"]["reduction"])
        self.assertEqual(out["decisions_by_compiler"], [])

    def test_array_expressions_parallelized_by_numba(self) -> None:
        code = read("numba/jacobi2d_array_expr.py")
        base = dict(loops=[nest(("_", "T"))], memory={"temporaries": 3}, library_calls=["ndarray.copy"],
                    decisions_by_llm=["memory_layout"], decisions_by_library=["library_delegation"])
        self.assertEqual(numba(code, inspect=inspect_of("numba/jacobi2d_array_expr.import.json")), expect(
            "numba", **base, parallel={"kind": "numba", "regions": 3, "placement": "region"},
            decisions_by_compiler=["parallel_region"],
            notes="numba jacobi2d: parallel=True; numba diagnostics: 0 parallel loop(s) from prange, 3 from array expressions, 0 fused; "
                  "threading_layer=omp"))
        # 진단이 없으면 parallel=True 가 무엇을 병렬화했는지 알 수 없다
        self.assertEqual(numba(code), expect("numba", **base, parallel=UNKNOWN_PARALLEL, notes="numba jacobi2d: parallel=True"))

    def test_blocked_serial_kernel(self) -> None:
        self.assertEqual(numba(read("numba/gemm_blocked.py"), inspect=inspect_of("numba/gemm_blocked.import.json")), expect(
            "numba", loops=[nest(("ii", "M"), ("kk", "K"), ("i", "min(ii + BLOCK, M)"), ("k", "min(kk + BLOCK, K)"), ("j", "N"))],
            tiling={"detected": True, "tile_sizes": [64]},
            decisions_by_llm=["loop_order", "tiling"], notes="element loops: i, k, j; numba gemm: fastmath=True"))

    def test_nested_prange_only_the_outermost_runs_in_parallel(self) -> None:
        code = ("from numba import njit, prange\n\n@njit(parallel=True)\ndef gemm(A, B, C, M, N, K):\n    for i in prange(M):\n"
                "        for j in prange(N):\n            acc = 0.0\n            for k in range(K):\n                acc += A[i, k] * B[k, j]\n"
                "            C[i, j] = acc\n")
        out = numba(code)
        self.assertEqual(out["parallel"], {"kind": "numba", "regions": 1, "collapse": False, "reduction": False, "loops": ["i"],
                                           "placement": "outer", "constructs": ["prange"]})
        self.assertIn("nested prange serialized (1 inner)", out["notes"])

    def test_prange_inside_a_while_loop_is_not_outer(self) -> None:
        body = ("        for i in prange(n):\n            nxt[i] = 0.5 * cur[i]\n        cur, nxt = nxt, cur\n")
        head = "from numba import njit, prange\n\n@njit(parallel=True)\ndef step(cur, nxt, n, T):\n"
        out = numba(head + "    t = 0\n    while t < T:\n" + body + "        t += 1\n")
        self.assertEqual((out["parallel"]["placement"], out["parallel"]["loops"]), ("inner", ["i"]))
        self.assertIn("parallel loop inside a while loop (runs once per iteration)", out["notes"])
        self.assertEqual(out["loops"], [nest(("i", "n"))], "while 은 루프 목록에 넣지 않는다")
        # 같은 구조를 for 로 쓰면 같은 placement 다
        self.assertEqual(numba(head + "    for t in range(T):\n" + body)["parallel"]["placement"], "inner")

    def test_threading_layer_is_noted_only_for_a_kernel_with_a_parallel_construct(self) -> None:
        loaded = {"parallel_diagnostics": "", "threading_layer": "omp", "num_threads": 16}   # 로드된 커널이면 항상 문자열이다
        serial = numba(read("numba/gemm_blocked.py"), inspect=loaded)
        self.assertNotIn("threading_layer", serial["notes"])
        parallel = numba(read("numba/gemm_prange.py"), inspect=loaded)
        self.assertTrue(parallel["notes"].endswith("threading_layer=omp"), parallel["notes"])

    def test_inspect_that_did_not_finish_is_no_data_and_is_noted(self) -> None:
        code = read("numba/jacobi2d_array_expr.py")
        out = numba(code, inspect={"inspect_error": "inspect did not finish: worker exited"})
        self.assertEqual(out["parallel"], {**expect("numba")["parallel"], **UNKNOWN_PARALLEL}, "진단을 못 봤으므로 모른다")
        self.assertEqual(out["notes"], "numba jacobi2d: parallel=True; inspect not available: inspect did not finish: worker exited")

    def test_prange_without_parallel_true_runs_serially(self) -> None:
        code = read("numba/gemm_prange.py").replace("parallel=True, ", "")
        self.assertEqual(numba(code), expect(
            "numba", loops=GEMM_IKJ_LOOPS, parallel={"constructs": ["prange"]}, decisions_by_llm=["loop_order"],
            notes="element loops: i, j, k; numba gemm: fastmath=True; prange without parallel=True runs serially"))

    def test_build_extra_options_describe_the_task_function(self) -> None:
        # 빌드 단계(v3 §6)가 과제 함수의 데코레이터에서 읽은 옵션이 오면 그것을 쓴다
        extra = {"decorator": "njit", "options": {"parallel": None, "fastmath": None, "cache": True, "boundscheck": None, "nogil": None}, "uses_prange": True}
        out = numba(read("numba/gemm_prange.py"), task="gemm", build_extra=extra)
        self.assertEqual(out["parallel"]["kind"], "none")
        self.assertEqual(out["notes"], "element loops: i, j, k; numba gemm: cache=True; prange without parallel=True runs serially")
        self.assertEqual(numba(read("numba/gemm_prange.py"), task="gemm", build_extra={"options": None})["parallel"]["kind"], "numba")

    def test_prange_in_a_jitted_helper_follows_the_helper_decorator(self) -> None:
        out = numba("""
import numba as nb

@nb.njit(parallel=True)
def _core(x, y, n):
    s = 0.0
    for i in nb.prange(n):
        s += x[i] * y[i]
    return s

@nb.njit("void(float32[::1], float32[::1], float32[::1], int64)")
def dot(x, y, out, n):
    out[0] = _core(x, y, n)
""", spec=DOT)
        self.assertEqual(out["parallel"], {"kind": "numba", "regions": 1, "collapse": False, "reduction": True, "loops": ["i"],
                                           "placement": "outer", "constructs": ["prange"]})
        self.assertEqual(out["notes"], "element loops: i; numba _core: parallel=True; numba dot: default options")


class HalideTest(unittest.TestCase):
    AUTO_GEMM_NOTE = ("schedule by Mullapudi2016: compute_root(), split(j, j_vo, j_vi, 8), vectorize(j_vi), parallel(i), "
                      "reorder(j, k$x, i), split(j, j_vo, j_vi, 8, TailStrategy::GuardWithIf)")

    def test_autoscheduled_gemm_decisions_belong_to_the_compiler(self) -> None:
        self.assertEqual(halide("gemm_auto", "halide_auto", "gemm_auto.large"), expect(
            "halide_auto",
            loops=[nest(("i", ""), ("j.j_vo", ""), ("j.j_vi", "[0, 7]")), nest(("i", ""), ("k", "[0, 1023]"), ("j.j_vo", ""), ("j.j_vi", "[0, 7]"))],
            parallel={"kind": "halide", "regions": 2, "loops": ["i"], "placement": "outer", "constructs": ["parallel(i)"]},
            vectorized={"status": "yes", "source": "loop_nest", "loops": ["j.j_vi"]},
            decisions_by_compiler=["loop_order", "vectorization", "parallel_region"],       # 벡터 폭으로 자른 split 은 타일링이 아니다
            notes=self.AUTO_GEMM_NOTE))

    def test_the_same_source_at_the_tiny_shape_gets_a_different_schedule(self) -> None:
        self.assertEqual(halide("gemm_auto", "halide_auto", "gemm_auto.tiny"), expect(
            "halide_auto", loops=[nest(("i", ""), ("j", "")), nest(("i", ""), ("k", "[0, 2]"), ("j", ""))],
            parallel={"kind": "halide", "regions": 2, "loops": ["i"], "placement": "outer", "constructs": ["parallel(i)"]},
            vectorized={"status": "no", "source": "loop_nest"},
            decisions_by_compiler=["loop_order", "parallel_region"],
            notes="schedule by Mullapudi2016: compute_root(), parallel(i), reorder(j, k$x, i)"))

    def test_autoscheduler_output_missing_means_unknown(self) -> None:
        want = expect("halide_auto", parallel=UNKNOWN_PARALLEL, tiling={"detected": None},
                      memory={"temporaries": None, "layout_transposed": None},
                      notes="loop nest not available; autoscheduler schedule not available")
        self.assertEqual(halide("gemm_auto", "halide_auto"), want)
        self.assertEqual(summarize(language="halide_auto", backend="halide", code=read("halide/gemm_auto.py"), inspect={}), want)
        broken = {"loop_nest": "Error: something went wrong\n  at Foo", "schedule_source": "", "autoscheduler": "Mullapudi2016"}
        self.assertEqual(summarize(language="halide_auto", backend="halide", code=read("halide/gemm_auto.py"), inspect=broken), want)

    def test_inspect_that_did_not_finish_is_no_data_and_is_noted(self) -> None:
        # import 워커가 수집 중에 죽으면 inspect 에는 inspect_error 하나뿐이다: 관찰값이 없는 것으로 다루고 notes 에 남긴다
        failed = {"inspect_error": "inspect did not finish: worker killed\n  (signal SIGKILL)"}
        want = expect("halide_auto", parallel=UNKNOWN_PARALLEL, tiling={"detected": None},
                      memory={"temporaries": None, "layout_transposed": None},
                      notes="loop nest not available; autoscheduler schedule not available; "
                            "inspect not available: inspect did not finish: worker killed (signal SIGKILL)")
        self.assertEqual(summarize(language="halide_auto", backend="halide", code=read("halide/gemm_auto.py"), inspect=failed), want)
        partial = {**inspect_of("halide/gemm_manual.import.json"), "inspect_error": "schedule_source: RuntimeError"}
        out = halide("gemm_manual", "halide_manual")
        with_error = summarize(language="halide_manual", backend="halide", code=read("halide/gemm_manual.py"), inspect=partial)
        self.assertTrue(with_error["notes"].endswith("inspect incomplete: schedule_source: RuntimeError"), with_error["notes"])
        self.assertNotEqual(with_error["parallel"], out["parallel"], "모은 값(루프 중첩)은 그대로 쓴다")

    def test_manual_schedule_decisions_belong_to_the_llm(self) -> None:
        self.assertEqual(halide("gemm_manual", "halide_manual", "gemm_manual"), expect(
            "halide_manual",
            loops=[nest(("i", ""), ("j.j", ""), ("j.v3", "[0, 7]")), nest(("i", ""), ("k", "[0, 1023]"), ("j.j", ""), ("j.v5", "[0, 7]"))],
            parallel={"kind": "halide", "regions": 2, "loops": ["i"], "placement": "outer", "constructs": ["parallel(i)"]},
            vectorized={"status": "yes", "source": "loop_nest", "loops": ["j.v3", "j.v5"]},
            decisions_by_llm=["loop_order", "vectorization", "parallel_region"],
            notes="schedule in source: vectorize(j, 8, tail), parallel(i), reorder(j, k.x, i)"))

    def test_manual_schedule_without_the_loop_nest(self) -> None:
        self.assertEqual(halide("gemm_manual", "halide_manual"), expect(
            "halide_manual",
            parallel={"kind": "halide", "regions": 2, "loops": ["i"], "placement": "unknown", "constructs": ["parallel(i)"]},
            memory={"temporaries": None},
            decisions_by_llm=["loop_order", "vectorization", "parallel_region"],
            notes="schedule in source: vectorize(j, 8, tail), parallel(i), reorder(j, k.x, i); loop nest not available"))

    def test_manual_tile_with_a_named_constant(self) -> None:
        self.assertEqual(halide("gemm_manual_tiled", "halide_manual", "gemm_manual_tiled"), expect(
            "halide_manual",
            loops=[nest(("i", ""), ("j", "")),
                   nest(("i.io", ""), ("j.jo", ""), ("i.ii", "[0, 31]"), ("k", "[0, 1023]"), ("j.ji.ji", "[0, 7]"), ("j.ji.v8", "[0, 7]"))],
            parallel={"kind": "halide", "regions": 1, "loops": ["i.io"], "placement": "outer", "constructs": ["parallel(io)"]},
            vectorized={"status": "partial", "source": "loop_nest", "loops": ["j.ji.v8"]},       # 초기화 스테이지는 스케줄하지 않았다
            tiling={"detected": True, "tile_sizes": [64, 32]},
            decisions_by_llm=["loop_order", "tiling", "vectorization", "parallel_region"],
            notes="schedule in source: tile(j, i, jo, io, ji, ii, TILE, 32, tail), reorder(ji, k.x, ii, jo, io), vectorize(ji, 8, tail), parallel(io)"))

    def test_autoscheduled_stencil_is_tiled_and_keeps_intermediate_buffers(self) -> None:
        step = nest(("y.y_o", ""), ("x.x_o", ""), ("y", ""), ("x.x_vo", "[0, 32]"), ("x.x_vi", "[0, 7]"))
        self.assertEqual(halide("jacobi2d_auto", "halide_auto", "jacobi2d_auto"), expect(
            "halide_auto",
            loops=[step, step, step, nest(("y.y_o", ""), ("x.x_o", ""), ("y.y_i", "[0, 255]"), ("x.x_i.x_i_vo", "[0, 31]"), ("x.x_i.x_i_vi", "[0, 7]"))],
            parallel={"kind": "halide", "regions": 1, "loops": ["y.y_o"], "placement": "outer", "constructs": ["parallel(y_o)"]},
            vectorized={"status": "yes", "source": "loop_nest", "loops": ["x.x_vi", "x.x_i.x_i_vi"]},
            tiling={"detected": True, "tile_sizes": [256]},
            memory={"temporaries": 3},
            decisions_by_compiler=["loop_order", "tiling", "vectorization", "parallel_region", "memory_layout"],
            notes="schedule by Mullapudi2016: compute_at(step3, x_o), split(x, x_vo, x_vi, 8), vectorize(x_vi), compute_root(), "
                  "split(x, x_o, x_i, 256), split(y, y_o, y_i, 256), reorder(x_i, y_i, x_o, y_o), split(x_i, x_i_vo, x_i_vi, 8), "
                  "vectorize(x_i_vi), parallel(y_o)"))

    def test_manual_stencil_computes_every_step_at_root(self) -> None:
        self.assertEqual(halide("jacobi2d_manual", "halide_manual", "jacobi2d_manual"), expect(
            "halide_manual",
            loops=[nest(("y", ""), ("x.x", ""), (f"x.v{k}", "[0, 7]")) for k in (9, 10, 11, 12)],
            parallel={"kind": "halide", "regions": 4, "loops": ["y"], "placement": "outer", "constructs": ["parallel(y)"]},
            vectorized={"status": "yes", "source": "loop_nest", "loops": ["x.v9", "x.v10", "x.v11", "x.v12"]},
            memory={"temporaries": 3},
            decisions_by_llm=["vectorization", "parallel_region", "memory_layout"],
            notes="schedule in source: compute_root(), parallel(y), vectorize(x, 8, hl.TailStrategy.GuardWithIf)"))

    def test_autoscheduled_reduction_stays_serial_and_scalar(self) -> None:
        copy = nest(("_0._0_vo", "[0, 4194303]"), ("_0._0_vi", "[0, 7]"))
        self.assertEqual(halide("dot_auto", "halide_auto", "dot_auto"), expect(
            "halide_auto", loops=[copy, copy, nest(("i", "")), nest(("i", ""), ("r", "[0, 33554431]"))],
            parallel={"kind": "halide", "regions": 4, "loops": ["_0._0_vo", "i"], "placement": "outer", "constructs": ["parallel(i)", "parallel(_0_vo)"]},
            vectorized={"status": "partial", "source": "loop_nest", "loops": ["_0._0_vi"]},      # 입력 복사만 벡터화됐고 누적 루프 r 은 아니다
            memory={"temporaries": 2},
            decisions_by_compiler=["loop_order", "vectorization", "parallel_region", "memory_layout"],
            notes="schedule by Mullapudi2016: compute_root(), parallel(i), reorder(i, r$x), reorder(r$x, i), split(_0, _0_vo, _0_vi, 8), "
                  "vectorize(_0_vi), parallel(_0_vo)"))

    def test_manual_rfactor_reduction(self) -> None:
        self.assertEqual(halide("dot_manual", "halide_manual", "dot_manual"), expect(
            "halide_manual",
            loops=[nest(("u", "[0, 4095]")), nest(("u", "[0, 4095]"), ("ri", "[0, 8191]")), nest(("i", "")), nest(("i", ""), ("ro", "[0, 4095]"))],
            parallel={"kind": "halide", "regions": 1, "loops": ["u"], "placement": "outer", "constructs": ["parallel(u)"]},
            vectorized={"status": "no", "source": "loop_nest"},
            tiling={"detected": True, "tile_sizes": [8192]},
            memory={"temporaries": 2},
            decisions_by_llm=["tiling", "parallel_region", "reduction", "memory_layout"],
            notes="schedule in source: split(r.x, ro, ri, 8192), rfactor(ro, u), compute_root(), parallel(u)"))

    def test_schedule_calls_written_in_a_halide_auto_source_are_llm_decisions(self) -> None:
        code = read("halide/gemm_auto.py").replace("    return C\n", "    C.update(0).parallel(i)\n    return C\n")
        out = summarize(language="halide_auto", backend="halide", code=code, inspect=inspect_of("halide/gemm_auto.large.import.json"))
        self.assertEqual(out["decisions_by_llm"], ["parallel_region"])
        self.assertEqual(out["decisions_by_compiler"], ["loop_order", "vectorization", "parallel_region"])
        self.assertEqual(out["notes"], "schedule in source: parallel(i); " + self.AUTO_GEMM_NOTE)

    def test_build_extra_schedule_calls_are_the_llm_schedule(self) -> None:
        out = summarize(language="halide_manual", backend="halide", code=read("halide/gemm_auto.py"),
                        build_extra={"schedule_calls": ["reorder", "store_at", "rfactor"], "imports_halide": True})
        self.assertEqual(out["decisions_by_llm"], ["loop_order", "reduction", "memory_layout"])

    def test_storage_directives_list_form_tile_and_false_friends(self) -> None:
        out = summarize(language="halide_manual", backend="halide", code="""
import numpy as np
import halide as hl
T = 32

def f(src, H, W):
    names = "x y xo yo xi yi".split()
    halves = np.split(np.arange(8), 2)
    x, y, xo, yo, xi, yi = [hl.Var(n) for n in names]
    g = hl.Func("g"); g[x, y] = src[x, y] * 2
    out = hl.Func("out"); out[x, y] = g[x, y] + g[x, y + 1]
    out.tile([x, y], [xo, yo], [xi, yi], [T, 2 * T]).parallel(yo).unroll(xi, 4)
    g.store_at(out, yo).compute_at(out, xo).reorder_storage(y, x).vectorize(x, 8)
    out.split(xo, xo, xi, 4)
    return out
""")
        self.assertEqual(out, expect(
            "halide_manual",
            parallel={"kind": "halide", "regions": 1, "loops": ["yo"], "placement": "unknown", "constructs": ["parallel(yo)"]},
            tiling={"detected": True, "tile_sizes": [32, 64]},      # unroll 되는 xi 로 자른 split(…, 4) 는 타일이 아니다
            memory={"temporaries": None, "layout_transposed": True},
            decisions_by_llm=["tiling", "vectorization", "parallel_region", "memory_layout"],
            notes="schedule in source: tile([x, y], [xo, yo], [xi, yi], [T, 2 * T]), parallel(yo), unroll(xi, 4), store_at(out, yo), "
                  "compute_at(out, xo), reorder_storage(y, x), vectorize(x, 8), split(xo, xo, xi, 4); loop nest not available"))

    def test_schedule_call_to_decision_table(self) -> None:
        table = {
            "reorder": "loop_order", "tile": "tiling", "vectorize": "vectorization", "parallel": "parallel_region",
            "rfactor": "reduction", "atomic": "reduction",
            **dict.fromkeys(("compute_at", "store_at", "store_root", "store_in", "fold_storage", "reorder_storage", "align_storage",
                             "bound_storage", "hoist_storage", "hoist_storage_root"), "memory_layout"),
        }
        for name, decision in table.items():
            out = summarize(language="halide_manual", backend="halide", code=f"def f(g, x):\n    g.{name}(x)\n    return g\n")
            self.assertEqual(out["decisions_by_llm"], [decision], name)
            self.assertEqual(out["notes"], f"schedule in source: {name}(x); loop nest not available")
        # 그 자체로는 어휘의 결정이 아닌 스케줄 호출: 기록만 된다. 출력 Func 하나뿐인 compute_root 도 그렇다
        for name in ("compute_root", "unroll", "fuse", "serial", "bound", "split"):
            out = summarize(language="halide_manual", backend="halide", code=f"def f(g, x):\n    g.{name}(x)\n    return g\n",
                            inspect={"loop_nest": "produce g:\n  for x:\n    g(...) = ...\n", "schedule_source": "", "autoscheduler": None})
            self.assertEqual((out["decisions_by_llm"], out["notes"]), ([], f"schedule in source: {name}(x)"), name)

    def test_default_schedule_methods_equal_the_build_module(self) -> None:
        try:
            from harness.build.halide import SCHEDULE_METHODS
        except Exception as e:  # noqa: BLE001 - 빌드 모듈이 없거나 이 플랫폼에서 임포트되지 않으면 기본값이 쓰인다
            self.skipTest(f"harness.build.halide 를 임포트할 수 없다: {type(e).__name__}: {e}")
        self.assertEqual(frozenset(SCHEDULE_METHODS), DEFAULT_SCHEDULE_METHODS)


class RustTest(unittest.TestCase):
    def test_rayon_rows_with_serial_loops_inside(self) -> None:
        self.assertEqual(rust(read("rust/gemm_rayon.rs")), expect(
            "rust", loops=[nest(("(p, &a_ip)", "a_row.iter().enumerate()"), ("(c_ij, &b_pj)", "c_row.iter_mut().zip(b_row)"))],
            parallel={"kind": "rayon", "regions": 1, "placement": "outer", "constructs": ["par_chunks_mut", "par_chunks"]},
            library_calls=["fill"], decisions_by_llm=["loop_order", "parallel_region"], notes="unsafe blocks: 0"))

    def test_rayon_reduction_in_f64(self) -> None:
        self.assertEqual(rust(read("rust/dot_rayon_f64.rs"), spec=DOT), expect(
            "rust", parallel={"kind": "rayon", "regions": 1, "reduction": True, "placement": "outer", "constructs": ["par_chunks"]},
            decisions_by_llm=["parallel_region", "reduction", "accumulation_precision"], notes="unsafe blocks: 0"))

    def test_std_arch_intrinsics_with_a_scalar_remainder_loop(self) -> None:
        self.assertEqual(rust(read("rust/dot_avx2.rs")), expect(
            "rust", loops=[nest(("(a, b)", "xc.zip(yc)")), nest(("(p, q)", "xr.iter().zip(yr)"))],
            vectorized={"source": "intrinsics", "loops": ["(a, b)"]},        # 리마크가 없어 나머지 루프는 알 수 없다
            decisions_by_llm=["vectorization", "accumulation_precision"],
            notes="intrinsics: 8 distinct (_mm256_setzero_pd, _mm256_loadu_ps, _mm256_mul_pd, _mm256_cvtps_pd, +4 more); "
                  "unsafe blocks: 3; chunks_exact: 2 use(s)"))

    def test_step_by_blocks(self) -> None:
        self.assertEqual(rust(read("rust/gemm_blocked.rs")), expect(
            "rust", loops=[nest(("ii", "m"), ("pp", "k"), ("i", "(ii + BLOCK).min(m)"), ("p", "(pp + BLOCK).min(k)"), ("j", "n"))],
            tiling={"detected": True, "tile_sizes": [48]}, library_calls=["fill"],
            decisions_by_llm=["loop_order", "tiling"], notes="unsafe blocks: 0"))

    def test_for_in_other_syntax_is_not_a_loop(self) -> None:
        out = rust("""
use std::fmt;
struct K<'a> { x: &'a [f32] }
impl<'a> fmt::Debug for K<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "for i in 0..n {{ }}") }
}
/* for i in 0..n { /* nested for j in 0..m { */ } */
// for q in 0..n {
pub fn f(x: &[f32], out: &mut [f32]) {
    let s = r#"for k in 0..n { "quoted" }"#;
    let c = '{';
    let d = '\\'';
    for (i, v) in x.iter().enumerate() {
        out[i] = *v + (s.len() + c as usize + d as usize) as f32;
    }
}
""")
        self.assertEqual(out, expect("rust", loops=[nest(("(i, v)", "x.iter().enumerate()"))], notes="unsafe blocks: 0"))

    def test_join_scope_parallel_sort_and_placement(self) -> None:
        out = rust("""
use rayon::join;
use rayon::prelude::*;
pub fn sort(keys: &[i32], out: &mut [i32]) {
    out.copy_from_slice(keys);
    out.par_sort_unstable();
}
pub fn f(a: &mut [f32], b: &mut [f32], t: usize) {
    let (x, y) = join(|| a.iter().sum::<f32>(), || b.iter().sum::<f32>());
    for _ in 0..t {
        (0..a.len()).into_par_iter().for_each(|_i| {});
    }
    rayon::scope(|s| { s.spawn(|_| {}); });
    let h = std::thread::spawn(move || x + y);
    h.join().unwrap();
}
""")
        self.assertEqual(out["parallel"], {
            "kind": "rayon", "regions": 4, "collapse": False, "reduction": False, "loops": [], "placement": "mixed",
            "constructs": ["par_sort_unstable", "rayon::join", "into_par_iter", "rayon::scope", "thread::spawn"]})
        self.assertEqual(out["library_calls"], ["copy_from_slice", "par_sort_unstable"])
        self.assertEqual(out["decisions_by_library"], ["library_delegation"])       # 정렬 알고리즘을 rayon 에 넘겼다

    def test_std_threads_allocation_transposed_copy_and_f64_accumulator(self) -> None:
        out = rust("""
use std::thread;
pub fn gemm(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
    let mut bt = vec![0.0f32; n * k];
    for p in 0..k {
        for j in 0..n {
            bt[j * k + p] = b[p * n + j];
        }
    }
    thread::scope(|s| {
        for (i, row) in c.chunks_mut(n).enumerate() {
            let bt = &bt;
            s.spawn(move || {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for p in 0..k {
                        acc += (a[i * k + p] * bt[j * k + p]) as f64;
                    }
                    row[j] = acc as f32;
                }
            });
        }
    });
}
""", spec=GEMM)
        self.assertEqual(out, expect(
            "rust", loops=[nest(("p", "k"), ("j", "n")), nest(("(i, row)", "c.chunks_mut(n).enumerate()"), ("j", "n"), ("p", "k"))],
            parallel={"kind": "threads", "regions": 1, "placement": "region", "constructs": ["thread::scope"]},
            memory={"temporaries": 1, "layout_transposed": True},
            decisions_by_llm=["loop_order", "parallel_region", "memory_layout", "accumulation_precision"],
            notes="unsafe blocks: 0"))

    def test_a_loop_that_only_calls_a_hand_vectorized_helper_is_not_a_leaf(self) -> None:
        code = read("rust/dot_avx2.rs").replace("    for (p, q) in xr.iter().zip(yr) {\n        total += *p as f64 * *q as f64;\n    }\n", "")
        code += "\npub unsafe fn blocks(x: &[f32], y: &[f32]) -> f64 {\n    let mut s = 0.0f64;\n" \
                "    for (a, b) in x.chunks(4096).zip(y.chunks(4096)) {\n        s += dot_avx2(a, b);\n    }\n    s\n}\n"
        out = rust(code)
        self.assertEqual(len(out["loops"]), 2)
        self.assertEqual(out["vectorized"], {"status": "yes", "source": "intrinsics", "loops": ["(a, b)"]})


class CliTest(unittest.TestCase):
    def run_cli(self, *argv: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = perf_summary.main(list(argv))
        return rc, out.getvalue(), err.getvalue()

    def test_c_with_build_log(self) -> None:
        rc, out, err = self.run_cli("--language", "c_openmp", "--backend", "c", "--file", str(FIXTURES / "c" / "jacobi2d_omp.c"),
                                    "--build-log", str(FIXTURES / "c" / "jacobi2d_omp.build.log"))
        self.assertEqual((rc, err), (0, ""))
        self.assertEqual(json.loads(out), c_fixture("jacobi2d_omp", "c_openmp", log=True))

    def test_halide_with_import_document(self) -> None:
        rc, out, _ = self.run_cli("--language", "halide_auto", "--backend", "halide", "--file", str(FIXTURES / "halide" / "jacobi2d_auto.py"),
                                  "--inspect", str(FIXTURES / "halide" / "jacobi2d_auto.import.json"))
        self.assertEqual(rc, 0)
        self.assertEqual(json.loads(out), halide("jacobi2d_auto", "halide_auto", "jacobi2d_auto"))

    def test_unreadable_input_is_an_error(self) -> None:
        rc, out, err = self.run_cli("--language", "numpy", "--backend", "python", "--file", str(FIXTURES / "python" / "no_such_file.py"))
        self.assertEqual((rc, out), (1, ""))
        self.assertTrue(err.startswith("[perf_summary] "), err)
        rc, out, err = self.run_cli("--language", "numba", "--backend", "numba", "--file", str(FIXTURES / "numba" / "gemm_prange.py"),
                                    "--inspect", str(FIXTURES / "numba" / "gemm_prange.py"))
        self.assertEqual((rc, out), (1, ""))


if __name__ == "__main__":
    unittest.main()
