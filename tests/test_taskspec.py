"""과제 명세(harness/taskspec.py)와 과제 일반 판정·워커 경로(M2) 테스트."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from harness.classify import Tolerance, classify_task, load_tolerance
from harness.config import load_config
from harness.paths import PYTHON_EXE, ROOT
from harness.procutil import run_isolated
from harness.taskspec import TaskSpecError, load_taskspec, parse_taskspec

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
            k = Path(tmp) / "kernel.py"; k.write_text(code)
            out = Path(tmp) / "check.json"
            cmd = [str(PYTHON_EXE), "-m", "harness.worker", "--backend", "python", "--artifact", str(k), "--mode", "check",
                   "--tests", ",".join(map(str, tests)), "--tolerance", str(tdir / "tolerance.yaml"),
                   "--signatures", str(tdir / "signatures.yaml"), "--out", str(out)]
            proc = run_isolated(cmd, timeout_s=120, cwd=ROOT)
            self.assertEqual(proc.rc, 0, proc.stderr_tail)
            return json.loads(out.read_text())

    def test_correct_kernels_pass(self):
        for task, code in KERNELS.items():
            doc = self._check(task, code)
            self.assertEqual(doc["classification"], "pass", f"{task}: {doc.get('error_detail')}")
            self.assertEqual(doc["task"], task)

    def test_int_input_mutation_is_detected(self):
        doc = self._check("spmv_csr", MUTATING_SPMV)
        self.assertTrue(doc["inputs_mutated"])
        self.assertEqual(doc["classification"], "wrong_output")


if __name__ == "__main__":
    unittest.main()
