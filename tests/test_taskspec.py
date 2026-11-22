"""과제 명세(harness/taskspec.py)와 과제 일반 판정·워커 경로(M2) 테스트.

v3(2026-11-20): ``doc``·``unsupported``·``languages`` 원문(§2). 정수 출력 과제는 저장소 과제(histogram·scan·sort·bfs)와
같은 규칙이다: float64 참조 + rtol = atol = 0 의 정확 일치(docs/decisions.md 2026-11-08). 손으로 만든 int32 과제(작은 히스토그램)를
실제 워커로 돌리는 경로(pass / 누적 / 미기록 / 불일치 / 부분 기록 / 실행 간 불일치 / bench)도 이 규칙으로 확인한다."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from harness import testgen, worker
from harness.classify import Tolerance, classify_task, load_tolerance
from harness.config import load_config
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.taskspec import ArgSpec, TaskSpecError, load_taskspec, parse_taskspec

TASKS = ROOT / "tasks"
TOL = Tolerance(rtol=1e-3, atol=1e-3, numeric_pattern_tol_ratio=20, sentinel=-2048.0)
# config/experiment.yaml run.tasks 에 들어 있어야 하는 과제(Tier 1 = M2a, Tier 2 = M2b; 설계 §16·§17)
TIER1 = ("gemm", "jacobi2d", "spmv_csr", "dot")
TIER2 = ("histogram", "scan", "nbody", "sort")


class TaskSpecTest(unittest.TestCase):
    def test_all_configured_tasks_parse(self):
        """run.tasks 의 모든 과제가 signatures/tolerance/spec/baseline 파일을 갖고 명세가 파싱된다(설정과 과제 디렉터리의 어긋남 탐지)."""
        tasks = load_config().run.tasks
        for expected in (*TIER1, *TIER2):
            self.assertIn(expected, tasks)
        for task in tasks:
            with self.subTest(task=task):
                spec = load_taskspec(TASKS / task / "signatures.yaml")
                self.assertEqual(spec.task, task)
                self.assertEqual(spec.function, task)
                self.assertEqual(len([a for a in spec.args if a.kind == "out"]), 1)
                self.assertTrue((TASKS / task / spec.baseline["file"]).is_file(), spec.baseline)
                self.assertTrue((TASKS / task / "spec.md").is_file())
                self.assertTrue((TASKS / task / "gen_tests.py").is_file())
                tol = load_tolerance(TASKS / task / "tolerance.yaml")
                self.assertEqual(tol.task, task)
                if spec.output.dtype == "int32":  # 정수 출력은 정확 일치(§17)
                    self.assertEqual((tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio), (0.0, 0.0, 0.0))

    def test_shapes_work_and_guard(self):
        gemm = load_taskspec(TASKS / "gemm" / "signatures.yaml")
        dims = {"M": 200, "N": 130, "K": 312}
        self.assertEqual(gemm.shape_of(gemm.output, dims), (200, 130))
        self.assertEqual(gemm.work_of(dims), 2 * 200 * 130 * 312)
        self.assertEqual(gemm.guard_extra_of(dims), 200 * 200 - 200 * 130)
        spmv = load_taskspec(TASKS / "spmv_csr" / "signatures.yaml")
        self.assertEqual(spmv.shape_of(spmv.args[0], {"n_rows": 5, "n_cols": 7, "nnz": 9}), (6,))
        self.assertEqual(spmv.baseline["backend"], "c")
        self.assertFalse(spmv.baseline["gate_min_gflops"])

    def test_rejects_bad_specs(self):
        base = {"task": "t", "args": [{"name": "x", "kind": "in", "dtype": "float32", "shape": ["n"]},
                                      {"name": "o", "kind": "out", "dtype": "float32", "shape": [1]}, {"name": "n", "kind": "dim"}]}
        parse_taskspec(base)
        with self.assertRaises(TaskSpecError):
            parse_taskspec({**base, "params": ["o", "x", "n"]})  # 순서 불일치
        with self.assertRaises(TaskSpecError):
            parse_taskspec({"task": "t", "args": [{"name": "x", "kind": "in", "dtype": "float32", "shape": ["n"]}]})  # 출력 없음
        spec = parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "int32", "shape": [1]}]})  # 정수 출력(Tier 2)
        self.assertEqual(spec.output.dtype, "int32")
        with self.assertRaises(TaskSpecError):  # 출력 dtype 은 float32·int32 뿐이다(설계 §17; M2 에서는 int32 도 거부했다)
            parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "float64", "shape": [1]}]})
        with self.assertRaises(TaskSpecError):
            parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "int64", "shape": [1]}]})
        with self.assertRaises(TaskSpecError):  # 출력은 정확히 1개
            parse_taskspec({"task": "t", "args": [{"name": "a", "kind": "out", "dtype": "int32", "shape": [1]},
                                                  {"name": "b", "kind": "out", "dtype": "int32", "shape": [1]}]})
        with self.assertRaises(TaskSpecError):  # 이름 중복
            parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "int32", "shape": [1]}, {"name": "o", "kind": "dim"}]})

    def test_int32_output_is_accepted(self):
        spec = parse_taskspec(HISTOGRAM_RAW)
        self.assertEqual(spec.output.name, "hist")
        self.assertEqual(spec.output.dtype, "int32")
        self.assertEqual(spec.shape_of(spec.output, {"n": 10, "n_bins": 4}), (4,))
        self.assertEqual([a.name for a in spec.inputs], ["keys"])
        self.assertEqual(spec.dim_names, ["n", "n_bins"])
        self.assertEqual(spec.params, ["keys", "hist", "n", "n_bins"])
        # 출력 하나만 있는 최소 명세도 된다
        one = parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "int32", "shape": [1]}]})
        self.assertEqual(one.output, ArgSpec(name="o", kind="out", dtype="int32", shape=("1",)))

    def test_doc_is_optional_and_kept_for_arrays_and_dims(self):
        spec = parse_taskspec(HISTOGRAM_RAW)
        by_name = {a.name: a for a in spec.args}
        self.assertEqual(by_name["keys"].doc, "each value is in the range 0 <= keys[i] < n_bins")
        self.assertEqual(by_name["hist"].doc, "")
        self.assertEqual(by_name["n"].doc, "")
        self.assertEqual(by_name["n_bins"].doc, "the number of bins")
        # 여러 줄·연속 공백은 한 구절로 정리된다
        raw = {"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "float32", "shape": [1], "doc": "first\n   second  "}]}
        self.assertEqual(parse_taskspec(raw).output.doc, "first second")
        with self.assertRaises(TaskSpecError):
            parse_taskspec({"task": "t", "args": [{"name": "o", "kind": "out", "dtype": "float32", "shape": [1], "doc": 3}]})
        # doc 을 주지 않은 과제(gemm)는 기본값(빈 문자열)이다
        self.assertTrue(all(a.doc == "" for a in load_taskspec(TASKS / "gemm" / "signatures.yaml").args))
        # 저장소 과제의 doc 은 한 구절로 읽힌다(새 언어의 매개변수 설명에 쓰인다; harness/sigs.py)
        hist = {a.name: a for a in load_taskspec(TASKS / "histogram" / "signatures.yaml").args}
        self.assertEqual(hist["data"].doc, "every value satisfies 0 <= data[i] < nbins")

    def test_unsupported_and_supports(self):
        spec = parse_taskspec(HISTOGRAM_RAW)
        self.assertEqual(spec.unsupported, {"halide_auto": "data-dependent writes are not expressible as a Halide pipeline",
                                            "halide_manual": "data-dependent writes are not expressible as a Halide pipeline"})
        self.assertFalse(spec.supports("halide_auto"))
        self.assertFalse(spec.supports("halide_manual"))
        for language in ("numpy", "numba", "c_seq", "c_openmp", "rust", "anything_else"):
            self.assertTrue(spec.supports(language), language)
        # 기존 과제: unsupported 없음 → 모든 언어 지원
        gemm = load_taskspec(TASKS / "gemm" / "signatures.yaml")
        self.assertEqual(gemm.unsupported, {})
        self.assertTrue(gemm.supports("halide_auto"))
        # Halide 는 10개 과제만 지원한다: sort·bfs 는 halide_auto·halide_manual 을 지원하지 않는다(이유는 영어 한 줄)
        for task in ("sort", "bfs"):
            spec_t = load_taskspec(TASKS / task / "signatures.yaml")
            self.assertEqual(sorted(spec_t.unsupported), ["halide_auto", "halide_manual"], task)
            for language in ("halide_auto", "halide_manual"):
                self.assertFalse(spec_t.supports(language), f"{task}/{language}")
                self.assertTrue(spec_t.unsupported[language].isascii() and "\n" not in spec_t.unsupported[language])
            for language in ("numpy", "c_seq", "c_openmp", "numba", "rust"):
                self.assertTrue(spec_t.supports(language), f"{task}/{language}")
        supported = [t for t in load_config().run.tasks if load_taskspec(TASKS / t / "signatures.yaml").supports("halide_auto")]
        self.assertEqual(len(supported), 10)
        for bad in (["halide_auto"], {"halide_auto": ""}, {"halide_auto": None}, {"halide_auto": 3}):
            with self.assertRaises(TaskSpecError, msg=repr(bad)):
                parse_taskspec({**HISTOGRAM_RAW, "unsupported": bad})

    def test_languages_raw_keeps_yaml_entries(self):
        gemm = load_taskspec(TASKS / "gemm" / "signatures.yaml")
        self.assertEqual(sorted(gemm.languages_raw), ["c_openmp", "c_seq", "numpy"])
        self.assertEqual(gemm.languages_raw["numpy"]["positional"], "def gemm(A, B, C, M, N, K):")
        self.assertEqual(gemm.languages_raw["numpy"]["keyword"], "def gemm(*, A, B, C, M, N, K):")
        self.assertIsNone(gemm.languages_raw["c_seq"]["keyword"])
        self.assertTrue(gemm.languages_raw["c_seq"]["param_docs"].startswith("Element access: A[i*K + k]"))
        # languages 가 없으면 빈 dict, 매핑이 아니면 오류
        self.assertEqual(parse_taskspec(HISTOGRAM_RAW).languages_raw, {})
        with self.assertRaises(TaskSpecError):
            parse_taskspec({**HISTOGRAM_RAW, "languages": ["numpy"]})
        with self.assertRaises(TaskSpecError):
            parse_taskspec({**HISTOGRAM_RAW, "languages": {"numpy": "def histogram(keys, hist, n, n_bins):"}})

    def test_existing_properties_and_eval_rules_are_unchanged(self):
        spec = parse_taskspec({**HISTOGRAM_RAW, "work": "n", "out_guard_extra": "max(n, n_bins) - n_bins"})
        dims = {"n": 10, "n_bins": 4}
        self.assertEqual(spec.work_of(dims), 10.0)
        self.assertEqual(spec.guard_extra_of(dims), 6)
        self.assertEqual(spec.eval_expr("min(n, n_bins) + 1", dims), 5)
        self.assertEqual(spec.eval_expr(7, dims), 7)
        with self.assertRaises(TaskSpecError):
            spec.eval_expr("__import__('os')", dims)  # 내장 함수 없음
        with self.assertRaises(TaskSpecError):
            spec.eval_expr("unknown_dim * 2", dims)
        self.assertEqual(spec.baseline["backend"], "python")  # 기본 기준선


# 정수 출력 과제의 시험용 명세(정수 출력 규칙은 설계 §17): 작은 히스토그램
HISTOGRAM_RAW = {
    "task": "histogram", "function": "histogram",
    "params": ["keys", "hist", "n", "n_bins"],
    "args": [
        {"name": "keys", "kind": "in", "dtype": "int32", "shape": ["n"], "doc": "each value is in the range 0 <= keys[i] < n_bins"},
        {"name": "hist", "kind": "out", "dtype": "int32", "shape": ["n_bins"]},
        {"name": "n", "kind": "dim"},
        {"name": "n_bins", "kind": "dim", "doc": "the number of bins"},
    ],
    "unsupported": {"halide_auto": "data-dependent writes are not expressible as a Halide pipeline",
                    "halide_manual": "data-dependent writes are   not expressible\nas a Halide pipeline"},
}


class IntOutputTest(unittest.TestCase):
    """정수 출력(histogram/scan/sort)은 rtol = atol = 0 으로 정확 일치를 요구한다."""

    EXACT = Tolerance(rtol=0.0, atol=0.0, numeric_pattern_tol_ratio=0.0, sentinel=-2048.0)

    def test_exact_match_and_mismatch(self):
        ref = np.array([3, 0, 7, 1], dtype=np.float64)
        self.assertEqual(classify_task("histogram", np.array([3, 0, 7, 1], np.int32), ref, {}, {}, self.EXACT)[0], "pass")
        st, pat, m = classify_task("histogram", np.array([3, 0, 8, 1], np.int32), ref, {}, {}, self.EXACT)
        self.assertEqual((st, pat), ("wrong_output", "mismatch"))
        self.assertEqual(m["n_fail"], 1)
        self.assertTrue(np.isinf(m["max_tol_ratio"]))
        st, pat, _ = classify_task("histogram", np.array([-2048, -2048, -2048, -2048], np.int32), ref, {}, {}, self.EXACT)
        self.assertEqual((st, pat), ("wrong_output", "output_not_written"))
        st, pat, _ = classify_task("histogram", np.array([3, 0, -2048, 1], np.int32), ref, {}, {}, self.EXACT)
        self.assertEqual((st, pat), ("wrong_output", "partial_write:1/4"))


class ClassifyTaskTest(unittest.TestCase):
    def test_generic_pass_transposed_and_scalar_ratio(self):
        ref = np.arange(12, dtype=np.float64).reshape(3, 4) + 1.0
        ok = classify_task("jacobi2d", ref.astype(np.float32), ref, {}, {}, TOL)
        self.assertEqual(ok[0], "pass")
        tr = classify_task("jacobi2d", np.ascontiguousarray(ref.T).astype(np.float32).reshape(-1), ref, {}, {}, TOL)
        self.assertEqual(tr[:2], ("arg_order", "transposed_output"))
        # 스칼라 출력: 7 % 틀린 값은 dot 의 넓은 배율(500)에서는 numeric, 기본 배율(20)에서는 wrong_output
        dot_tol = Tolerance(rtol=1e-3, atol=1e-3, numeric_pattern_tol_ratio=500, sentinel=-2048.0)
        r1 = np.array([1000.0])
        self.assertEqual(classify_task("dot", np.array([930.0], np.float32), r1, {}, {}, dot_tol)[:2], ("numeric_instability", "imprecise"))
        self.assertEqual(classify_task("dot", np.array([930.0], np.float32), r1, {}, {}, TOL)[:2], ("wrong_output", "mismatch"))
        self.assertEqual(classify_task("dot", np.array([-2048.0], np.float32), r1, {}, {}, dot_tol)[:2], ("wrong_output", "output_not_written"))


KERNELS = {
    "jacobi2d": "import numpy as np\ndef jacobi2d(src, dst, H, W, T):\n    cur = src.copy(); nxt = src.copy()\n    for _ in range(T):\n"
                "        nxt[1:-1,1:-1] = 0.25*(cur[:-2,1:-1]+cur[2:,1:-1]+cur[1:-1,:-2]+cur[1:-1,2:])\n        cur, nxt = nxt, cur\n    dst[...] = cur\n",
    "spmv_csr": "import numpy as np\ndef spmv_csr(row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz):\n"
                "    rows = np.repeat(np.arange(n_rows), np.diff(row_ptr))\n    y[...] = np.bincount(rows, weights=vals.astype(np.float64)*x[col_idx], minlength=n_rows)\n",
    "dot": "import numpy as np\ndef dot(x, y, out, n):\n    out[0] = np.dot(x, y)\n",
}
MUTATING_SPMV = ("import numpy as np\ndef spmv_csr(row_ptr, col_idx, vals, x, y, n_rows, n_cols, nnz):\n"
                 "    col_idx[...] = 0\n    y[...] = 0.0\n")


class WorkerOnNewTasksTest(unittest.TestCase):
    """python 백엔드 커널을 실제 tiny/rect 테스트로 check 한다(테스트 npz 가 없으면 건너뜀)."""

    def _check(self, task: str, code: str) -> dict:
        tdir = TASKS / task
        tests = [tdir / "tests" / f"{n}.npz" for n in ("tiny", "rect")]
        if not all(t.exists() for t in tests):
            self.skipTest(f"{task} 테스트 npz 없음 (make tests)")
        with tempfile.TemporaryDirectory() as tmp:
            # 한글이 든 디렉터리에서 돌린다(회귀): 워커의 결과 JSON 은 UTF-8 이고 산출물 경로를 담는다. 인코딩을 지정하지
            # 않고 읽으면 로캘 코드페이지(Windows 의 cp949)로 풀다가 깨진다 — %TEMP% 에 한글이 있는 머신에서 실측.
            work = Path(tmp) / "커널 작업"
            work.mkdir()
            k = work / "kernel.py"
            k.write_text(code, encoding="utf-8", newline="\n")
            out = work / "check.json"
            cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", "python", "--artifact", str(k), "--mode", "check",
                   "--tests", ",".join(map(str, tests)), "--tolerance", str(tdir / "tolerance.yaml"),
                   "--signatures", str(tdir / "signatures.yaml"), "--out", str(out)]
            proc = run_isolated(cmd, timeout_s=120, cwd=ROOT)
            self.assertEqual(proc.rc, 0, proc.stderr_tail)
            doc = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(Path(doc["artifact"]), k, "결과 문서의 경로가 그대로 되읽힌다")
            return doc

    def test_correct_kernels_pass(self):
        for task, code in KERNELS.items():
            doc = self._check(task, code)
            self.assertEqual(doc["classification"], "pass", f"{task}: {doc.get('error_detail')}")
            self.assertEqual(doc["task"], task)

    def test_int_input_mutation_is_detected(self):
        doc = self._check("spmv_csr", MUTATING_SPMV)
        self.assertTrue(doc["inputs_mutated"])
        self.assertEqual(doc["classification"], "wrong_output")


# ----------------------------------------------------------------------------- 정수 출력 과제를 실제 워커로 (설계 §17)

HISTOGRAM_SIGNATURES_YAML = """task: histogram
function: histogram
params: [keys, hist, n, n_bins]
args:
  - {name: keys, kind: in,  dtype: int32, shape: [n], doc: "each value is in the range 0 <= keys[i] < n_bins"}
  - {name: hist, kind: out, dtype: int32, shape: [n_bins]}
  - {name: n, kind: dim}
  - {name: n_bins, kind: dim}
baseline: {backend: python, file: baseline/baseline.py, openmp: false, gate_blas: false, gate_min_gflops: false}
"""
# 저장소 tasks/histogram/tolerance.yaml 과 같은 꼴: 정수 출력 = float64 참조 + rtol = atol = 0
HISTOGRAM_TOLERANCE_YAML = ("task: histogram\ndtype: int32\nrtol: 0.0\natol: 0.0\nreference_dtype: float64\n"
                            "numeric_pattern_tol_ratio: 0.0\nsentinel: -2048.0\n")
HISTOGRAM_SHAPES = [("tiny", {"n": 10, "n_bins": 4}), ("rect", {"n": 1003, "n_bins": 37}),
                    ("small", {"n": 4096, "n_bins": 16}), ("large", {"n": 20000, "n_bins": 64})]

HIST_CORRECT = "import numpy as np\ndef histogram(keys, hist, n, n_bins):\n    hist[:] = np.bincount(keys, minlength=n_bins)\n"
# 출력 버퍼를 0 으로 지우지 않고 더한다: 초기 내용(센티널) 위에 누적된다
HIST_ACCUMULATE = ("import numpy as np\ndef histogram(keys, hist, n, n_bins):\n"
                   "    counts = np.bincount(keys, minlength=n_bins).astype(np.int32)\n    hist += counts\n")
HIST_NOT_WRITTEN = "import numpy as np\ndef histogram(keys, hist, n, n_bins):\n    return np.bincount(keys, minlength=n_bins)\n"
HIST_OFF_BY_ONE = ("import numpy as np\ndef histogram(keys, hist, n, n_bins):\n"
                   "    hist[:] = np.bincount(keys, minlength=n_bins)\n    hist[0] += 1\n")
HIST_PARTIAL = ("import numpy as np\ndef histogram(keys, hist, n, n_bins):\n"
                "    hist[1:] = np.bincount(keys, minlength=n_bins)[1:]\n")
HIST_NONDET = ("import numpy as np\n_calls = [0]\ndef histogram(keys, hist, n, n_bins):\n"
               "    hist[:] = np.bincount(keys, minlength=n_bins)\n    hist[-1] += _calls[0] % 2\n    _calls[0] += 1\n")


def _histogram_case(shape, rng):
    n, n_bins = shape["n"], shape["n_bins"]
    keys = rng.integers(0, n_bins - 1, size=n, dtype=np.int32)  # 마지막 칸은 항상 0 개(누적 버그에서 센티널 그대로 남는 칸)
    ref = np.bincount(keys, minlength=n_bins).astype(np.float64)  # 정수 값을 담은 float64 참조(저장소 과제와 같다)
    return {"n": n, "n_bins": n_bins}, {"keys": keys}, "hist", ref


class WorkerIntOutputTest(unittest.TestCase):
    """손으로 만든 int32 과제를 testgen 으로 생성하고 실제 워커(check/bench)로 돌린다."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="int_task_")
        cls.tmp = Path(cls._tmpdir.name)
        cls.sig = cls.tmp / "signatures.yaml"
        cls.tol = cls.tmp / "tolerance.yaml"
        cls.sig.write_text(HISTOGRAM_SIGNATURES_YAML, encoding="utf-8", newline="\n")
        cls.tol.write_text(HISTOGRAM_TOLERANCE_YAML, encoding="utf-8", newline="\n")
        cls.tests_dir = cls.tmp / "tests"
        testgen.generate("histogram", 1234, cls.tests_dir, HISTOGRAM_SHAPES, _histogram_case, "numpy bincount")
        cls._n = 0

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _run(self, code: str, mode: str = "check", tests: list[str] | None = None, extra: list[str] | None = None,
             tests_dir: Path | None = None) -> tuple[int, dict]:
        type(self)._n += 1
        wd = self.tmp / f"case_{type(self)._n}"
        wd.mkdir()
        kernel = wd / "kernel.py"
        kernel.write_text(code, encoding="utf-8", newline="\n")
        out = wd / f"{mode}.json"
        names = tests if tests is not None else ["tiny", "rect", "small"]
        paths = [str((tests_dir or self.tests_dir) / f"{n}.npz") for n in names]
        cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", "python", "--artifact", str(kernel), "--mode", mode,
               "--tests", ",".join(paths), "--tolerance", str(self.tol), "--signatures", str(self.sig), "--out", str(out),
               *(extra or [])]
        proc = run_isolated(cmd, timeout_s=120, cwd=ROOT)
        self.assertTrue(out.exists(), proc.stderr_tail)
        return proc.rc, json.loads(out.read_text(encoding="utf-8"))

    def test_tolerance_file_is_the_int_output_form(self):
        tol = load_tolerance(self.tol)
        self.assertEqual((tol.dtype, tol.reference_dtype, tol.sentinel, tol.rtol, tol.atol, tol.numeric_pattern_tol_ratio),
                         ("int32", "float64", -2048.0, 0.0, 0.0, 0.0))

    def test_reference_is_float64_holding_integers(self):
        with np.load(self.tests_dir / "tiny.npz") as npz:
            self.assertEqual(npz["hist_ref"].dtype, np.float64)
            self.assertTrue(np.array_equal(npz["hist_ref"], np.round(npz["hist_ref"])))
            self.assertEqual(npz["keys"].dtype, np.int32)
            self.assertEqual(int(npz["n_bins"]), 4)

    def test_correct_kernel_passes(self):
        rc, doc = self._run(HIST_CORRECT)
        self.assertEqual(rc, 0)
        self.assertTrue(doc["complete"])
        self.assertEqual((doc["classification"], doc["error_detail"]), ("pass", ""))
        self.assertEqual(doc["task"], "histogram")
        self.assertEqual([t["name"] for t in doc["tests"]], ["tiny", "rect", "small"])
        self.assertFalse(doc["nondeterministic"])
        self.assertFalse(doc["oob_write"])
        self.assertFalse(doc["inputs_mutated"])
        self.assertEqual(doc["max_abs_err"], 0.0)
        # rtol = atol = 0: 오차가 없는 원소의 배율은 0 이다(틀린 원소는 inf — IntOutputTest)
        self.assertEqual(doc["max_tol_ratio"], 0.0)
        for t in doc["tests"]:
            for r in t["runs"]:
                self.assertEqual((r["classification"], r["n_unwritten"], r["nan_inf"]), ("pass", 0, False))

    def test_accumulating_without_zeroing_is_accumulated_into_c(self):
        rc, doc = self._run(HIST_ACCUMULATE)
        self.assertEqual(rc, 0)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "accumulated_into_c"))
        for t in doc["tests"]:
            self.assertEqual((t["classification"], t["pattern"]), ("wrong_output", "accumulated_into_c"), t["name"])
            # 개수가 0 인 칸은 센티널 그대로다 → n_unwritten 에 세어지지만 부분 기록이 아니라 누적으로 판정된다
            self.assertGreaterEqual(t["runs"][0]["n_unwritten"], 1)
            self.assertEqual(t["runs"][0]["max_abs_err"], 2048.0)

    def test_output_not_written(self):
        rc, doc = self._run(HIST_NOT_WRITTEN)
        self.assertEqual(rc, 0)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "returned_array_not_written"))
        self.assertEqual(doc["tests"][0]["runs"][0]["n_unwritten"], 4)

    def test_wrong_count_is_mismatch_not_numeric(self):
        rc, doc = self._run(HIST_OFF_BY_ONE)
        self.assertEqual(rc, 0)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "mismatch"))
        self.assertEqual(doc["max_abs_err"], 1.0)

    def test_partial_write(self):
        rc, doc = self._run(HIST_PARTIAL, tests=["tiny"])
        self.assertEqual(rc, 0)
        self.assertEqual((doc["classification"], doc["error_detail"]), ("wrong_output", "partial_write:1/4"))

    def test_run_to_run_difference_is_race_condition(self):
        rc, doc = self._run(HIST_NONDET, extra=["--check-repeats", "2"])
        self.assertEqual(rc, 0)
        self.assertTrue(doc["nondeterministic"])
        self.assertEqual(doc["classification"], "race_condition")
        self.assertTrue(all(t["nondeterministic"] for t in doc["tests"]))

    def test_bench_mode_on_int_output(self):
        rc, doc = self._run(HIST_CORRECT, mode="bench", tests=["large"],
                            extra=["--warmup", "1", "--repeats", "3", "--min-repeats", "1", "--budget-s", "60"])
        self.assertEqual(rc, 0)
        self.assertTrue(doc["complete"])
        self.assertEqual(doc["classification"], "pass")
        self.assertEqual(doc["final_check"]["classification"], "pass")
        self.assertEqual((doc["nondeterministic_runs"], doc["nondeterministic_bitwise_runs"]), (0, 0))
        self.assertEqual(len(doc["timed_ms"]), 3)
        self.assertEqual(doc["dims"], {"n": 20000, "n_bins": 64})
        self.assertIsNone(doc["gflops"], "work 식이 없는 과제는 gflops 를 내지 않는다")


class _FakeResource:
    """``resource`` 모듈 대역: 물려받은 ``RLIMIT_AS`` (soft, hard)와 ``setrlimit`` 호출 기록.

    실제 ``resource`` 가 있는 플랫폼에서도 테스트 프로세스의 한도를 건드리지 않고, 없는 플랫폼(Windows)에서도 워커의
    Linux 분기를 돌려 본다."""

    RLIMIT_AS = 9
    RLIM_INFINITY = -1

    def __init__(self, soft: int, hard: int, *, setrlimit_error: Exception | None = None) -> None:
        self.limits = (soft, hard)
        self.calls: list[tuple[int, tuple[int, int]]] = []
        self.setrlimit_error = setrlimit_error

    def getrlimit(self, which: int) -> tuple[int, int]:
        assert which == self.RLIMIT_AS
        return self.limits

    def setrlimit(self, which: int, limits: tuple[int, int]) -> None:
        self.calls.append((which, limits))
        if self.setrlimit_error is not None:
            raise self.setrlimit_error
        self.limits = limits


class WorkerMemoryLimitTest(unittest.TestCase):
    """``apply_memory_limit`` 의 Linux 분기를 ``resource`` 대역으로 확인한다(v3 §8.1: 0 이하면 RLIMIT_AS 를 걸지 않는다)."""

    GIB = 1024**3
    INF = _FakeResource.RLIM_INFINITY

    def apply(self, fake: _FakeResource, max_mem_gb: float) -> tuple[dict, str]:
        err = io.StringIO()
        with mock.patch.object(worker, "resource", fake), contextlib.redirect_stderr(err):
            limits = worker.apply_memory_limit(max_mem_gb)
        return limits, err.getvalue()

    def test_disabled_limit_reports_the_inherited_soft_limit(self):
        """회귀: 0 이하일 때 물려받은 soft 한도(오케스트레이터를 띄운 셸의 ``ulimit -v``)가 남아 있어도 적용값을 None 으로
        보고했다 — sanitizer 워커는 그 한도 아래에서 도는데 결과 문서는 제한이 없다고 말했다."""
        for value in (0, 0.0, -1):
            fake = _FakeResource(2 * self.GIB, self.INF)
            limits, err = self.apply(fake, value)
            self.assertEqual(limits, {"rlimit_as_bytes": 2 * self.GIB, "rlimit_as_requested_bytes": None,
                                      "rlimit_as_hard_bytes": None, "disabled": True})
            self.assertEqual(fake.calls, [], "0 이하면 setrlimit 을 부르지 않는다")
            self.assertEqual(err, "")

    def test_disabled_limit_without_an_inherited_limit_is_none(self):
        fake = _FakeResource(self.INF, 16 * self.GIB)
        limits, _ = self.apply(fake, 0)
        self.assertEqual(limits, {"rlimit_as_bytes": None, "rlimit_as_requested_bytes": None,
                                  "rlimit_as_hard_bytes": 16 * self.GIB, "disabled": True})
        self.assertEqual(fake.calls, [])

    def test_positive_limit_is_applied_and_capped_by_the_hard_limit(self):
        fake = _FakeResource(2 * self.GIB, self.INF)
        limits, _ = self.apply(fake, 8.0)
        self.assertEqual(limits, {"rlimit_as_bytes": 8 * self.GIB, "rlimit_as_requested_bytes": 8 * self.GIB, "rlimit_as_hard_bytes": None})
        self.assertEqual(fake.calls, [(fake.RLIMIT_AS, (8 * self.GIB, self.INF))])
        capped = _FakeResource(self.INF, 4 * self.GIB)
        limits, _ = self.apply(capped, 8.0)
        self.assertEqual(limits, {"rlimit_as_bytes": 4 * self.GIB, "rlimit_as_requested_bytes": 8 * self.GIB,
                                  "rlimit_as_hard_bytes": 4 * self.GIB})
        self.assertEqual(capped.calls, [(capped.RLIMIT_AS, (4 * self.GIB, 4 * self.GIB))])

    def test_failed_setrlimit_reports_the_limit_still_in_force(self):
        fake = _FakeResource(2 * self.GIB, self.INF, setrlimit_error=ValueError("not allowed"))
        limits, err = self.apply(fake, 8.0)
        self.assertEqual(limits, {"rlimit_as_bytes": 2 * self.GIB, "rlimit_as_requested_bytes": 8 * self.GIB, "rlimit_as_hard_bytes": None})
        self.assertIn("[worker]", err)


class WorkerPortabilityTest(unittest.TestCase):
    """워커의 프로세스 제한·시스템 샘플링 헬퍼가 없는 기능을 예외 없이 건너뛴다(v3 §8.1 --max-mem-gb, §0)."""

    def test_non_positive_max_mem_skips_rlimit(self):
        for value in (0, 0.0, -1):
            limits = worker.apply_memory_limit(value)
            if worker.resource is None:  # Windows: resource 모듈 없음
                self.assertEqual(limits, {"rlimit_as_bytes": None, "unsupported": True})
            else:
                # 이 프로세스가 물려받은 한도는 그대로다(setrlimit 을 부르지 않는다). 적용값은 실제로 걸려 있는 soft 한도다.
                soft, hard = worker.resource.getrlimit(worker.resource.RLIMIT_AS)
                inf = worker.resource.RLIM_INFINITY
                self.assertEqual(limits, {"rlimit_as_bytes": None if soft == inf else soft, "rlimit_as_requested_bytes": None,
                                          "rlimit_as_hard_bytes": None if hard == inf else hard, "disabled": True})
                self.assertEqual(worker.resource.getrlimit(worker.resource.RLIMIT_AS), (soft, hard))

    def test_sampling_helpers_never_raise(self):
        load = worker.loadavg_1m()
        self.assertTrue(load is None or load >= 0.0)
        self.assertIsInstance(worker.pinned_cpus(), list)
        self.assertIsInstance(worker.cpu_mhz_samples([0]), list)
        self.assertTrue(worker.package_temp_c() is None or worker.package_temp_c() > -100.0)

    @unittest.skipUnless(worker.resource is None,
                         f"resource 모듈이 있는 플랫폼({sys.platform}): RLIMIT_AS·prctl 이 실제로 적용되므로 테스트 프로세스에서 부르지 않는다")
    def test_platform_without_resource_reports_unsupported_limits(self):
        self.assertEqual(worker.apply_memory_limit(8.0), {"rlimit_as_bytes": None, "unsupported": True})
        self.assertIsNone(worker.disable_core_dumps())
        self.assertEqual(worker.pinned_cpus(), [])
        self.assertIsNone(worker.loadavg_1m())


if __name__ == "__main__":
    unittest.main()
