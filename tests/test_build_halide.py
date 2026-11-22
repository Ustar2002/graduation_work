"""build/halide.py 테스트 (설계 v3 §6): 구문 검사, AST 관찰(``schedule_calls``·``imports_halide``), ``SCHEDULE_METHODS``.

빌드 단계는 코드를 임포트·실행하지 않으므로 Halide 가 없어도 돈다(파이프라인 구성·스케줄·JIT 는 워커가 한다).
아래 커널들은 Halide 22.1 에서 실제로 맞는 결과를 내는 것을 확인한 꼴이다.
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

from harness.build.halide import HALIDE_MODES, SCHEDULE_METHODS, build_halide, main
from harness.perf_summary import DEFAULT_SCHEDULE_METHODS
from tests._support import requires_modules

# 설계 v3 §6 이 고정한 스케줄 메서드 이름 50개(문서의 목록 그대로)
CONTRACT_SCHEDULE_METHODS = (
    "align_bounds, align_extent, align_storage, allow_race_conditions, always_partition, always_partition_all, async_, atomic, "
    "bound, bound_extent, bound_storage, clone_in, compute_at, compute_inline, compute_root, compute_with, copy_to_device, "
    "copy_to_host, fold_storage, fuse, gpu, gpu_blocks, gpu_lanes, gpu_single_thread, gpu_threads, gpu_tile, hexagon, "
    "hoist_storage, hoist_storage_root, in_, memoize, never_partition, never_partition_all, parallel, partition, prefetch, "
    "rename, reorder, reorder_storage, rfactor, ring_buffer, serial, specialize, split, store_at, store_in, store_root, tile, "
    "unroll, vectorize"
).split(", ")

# 알고리즘만 쓴 꼴(halide_auto 가 요구하는 것)
HALIDE_GEMM = '''import halide as hl


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]
    return C
'''

# 같은 알고리즘에 스케줄을 직접 쓴 꼴(halide_manual): 메서드 연쇄
HALIDE_GEMM_MANUAL = '''import halide as hl


def gemm(A, B, M, N, K):
    i, j = hl.Var("i"), hl.Var("j")
    k = hl.RDom([(0, K)], "k")
    C = hl.Func("C")
    C[j, i] = hl.f32(0.0)
    C[j, i] += A[k.x, i] * B[j, k.x]
    C.parallel(i).vectorize(j, 8, hl.TailStrategy.GuardWithIf)
    C.update(0).reorder(j, k.x, i).parallel(i).vectorize(j, 8, hl.TailStrategy.GuardWithIf)
    return C
'''

# T 스텝마다 Func 을 하나씩 만드는 꼴: 루프 안의 스케줄 호출은 소스에 쓴 횟수만큼만 센다
HALIDE_JACOBI2D = '''import halide as hl


def jacobi2d(src, H, W, T):
    x, y = hl.Var("x"), hl.Var("y")
    xo, yo, xi, yi = hl.Var("xo"), hl.Var("yo"), hl.Var("xi"), hl.Var("yi")
    cur = hl.Func("step_0")
    cur[x, y] = src[x, y]
    cur.compute_root()
    for t in range(T):
        clamped = hl.BoundaryConditions.repeat_edge(cur, [(0, W), (0, H)])
        interior = (x > 0) & (x < W - 1) & (y > 0) & (y < H - 1)
        stencil = hl.f32(0.25) * (clamped[x, y - 1] + clamped[x, y + 1] + clamped[x - 1, y] + clamped[x + 1, y])
        nxt = hl.Func(f"step_{t + 1}")
        nxt[x, y] = hl.select(interior, stencil, cur[x, y])
        nxt.compute_root().tile(x, y, xo, yo, xi, yi, 64, 16, hl.TailStrategy.GuardWithIf).parallel(yo).vectorize(
            xi, 8, hl.TailStrategy.GuardWithIf)
        cur = nxt
    return cur
'''

HALIDE_DOT = '''from halide import Func, RDom, Var, f32, f64


def dot(x, y, n):
    r = RDom([(0, n)], "r")
    total = Func("total")
    total[()] = f64(0.0)
    total[()] += f64(x[r.x]) * f64(y[r.x])
    out = Func("out")
    i = Var("i")
    out[i] = f32(total[()])
    total.compute_root()
    return out
'''


class BuildDirTest(unittest.TestCase):
    """임시 출력 디렉터리에 빌드하는 테스트의 공통 부분(빌드마다 새 ``sample_<n>`` 디렉터리)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_halide_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp

    def build(self, code: str, function: str = "gemm", halide_mode: str = "manual") -> Any:
        self.out = self.tmp / f"sample_{len(list(self.tmp.iterdir()))}"
        return build_halide(code, self.out, function=function, halide_mode=halide_mode)


class ScheduleMethodsTest(unittest.TestCase):
    def test_exactly_the_contract_names(self) -> None:
        self.assertEqual(len(CONTRACT_SCHEDULE_METHODS), 50)
        self.assertIsInstance(SCHEDULE_METHODS, frozenset)
        self.assertEqual(sorted(SCHEDULE_METHODS), sorted(CONTRACT_SCHEDULE_METHODS))

    def test_same_set_as_the_perf_summary_default(self) -> None:
        # perf_summary 는 이 집합을 임포트해 쓰고, 임포트할 수 없을 때를 위해 같은 내용의 사본을 갖는다
        self.assertEqual(SCHEDULE_METHODS, DEFAULT_SCHEDULE_METHODS)

    @requires_modules("halide")
    def test_names_are_methods_of_the_python_bindings(self) -> None:
        # 이 프로세스에 Halide 를 올리지 않는다: 바인딩의 메서드 이름만 서브프로세스에서 읽어 온다
        probe = "import json, halide as hl; print(json.dumps(sorted(set(dir(hl.Func)) | set(dir(hl.Stage)))))"
        proc = subprocess.run([sys.executable, "-c", probe], stdin=subprocess.DEVNULL, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=300)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(sorted(SCHEDULE_METHODS - set(json.loads(proc.stdout))), [])


class RealisticKernelsTest(BuildDirTest):
    def test_kernels_pass_with_the_observed_facts(self) -> None:
        cases = {
            "algorithm only": (HALIDE_GEMM, "gemm", []),
            "manual schedule": (HALIDE_GEMM_MANUAL, "gemm", ["parallel", "parallel", "reorder", "vectorize", "vectorize"]),
            "funcs in a loop": (HALIDE_JACOBI2D, "jacobi2d", ["compute_root", "compute_root", "parallel", "tile", "vectorize"]),
            "from-import": (HALIDE_DOT, "dot", ["compute_root"]),
        }
        for name, (code, function, calls) in cases.items():
            for mode in HALIDE_MODES:
                with self.subTest(name, mode=mode):
                    res = self.build(code, function=function, halide_mode=mode)
                    self.assertEqual((res.ok, res.error_kind, res.error_detail), (True, "", ""))
                    self.assertEqual(res.extra, {"schedule_calls": calls, "imports_halide": True})

    def test_schedule_calls_under_auto_do_not_fail_the_build(self) -> None:
        # halide_auto 조건의 위반은 run.py 가 extra 를 보고 contract_violation 으로 관찰만 한다: 두 모드의 결과가 같다
        auto = self.build(HALIDE_GEMM_MANUAL, halide_mode="auto")
        manual = self.build(HALIDE_GEMM_MANUAL, halide_mode="manual")
        self.assertTrue(auto.ok)
        self.assertEqual((auto.ok, auto.error_kind, auto.extra), (manual.ok, manual.error_kind, manual.extra))
        self.assertEqual(auto.extra["schedule_calls"], ["parallel", "parallel", "reorder", "vectorize", "vectorize"])

    def test_unknown_mode_is_a_caller_error(self) -> None:
        self.assertEqual(HALIDE_MODES, ("auto", "manual"))
        for mode in ("Auto", "none", "", "None"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                build_halide(HALIDE_GEMM, self.tmp / "never", function="gemm", halide_mode=mode)
        self.assertFalse((self.tmp / "never").exists())

    def test_files_and_result_fields(self) -> None:
        res = self.build(HALIDE_GEMM_MANUAL)
        self.assertEqual(res.artifact, self.out / "kernel.py")
        self.assertEqual({p.name for p in self.out.iterdir()}, {"kernel.py", "build.json"})
        self.assertEqual((self.out / "kernel.py").read_text(encoding="utf-8"), HALIDE_GEMM_MANUAL)
        self.assertEqual((res.compiler, res.command, res.n_warnings, res.log), ("halide", ["python", "compile", "kernel.py"], 0, ""))
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, json.loads(json.dumps(res.to_dict())))
        self.assertEqual(saved["extra"], {"schedule_calls": ["parallel", "parallel", "reorder", "vectorize", "vectorize"],
                                          "imports_halide": True})

    @requires_modules("halide")
    def test_compiler_version_is_the_installed_halide(self) -> None:
        self.assertEqual(self.build(HALIDE_GEMM).compiler_version, importlib.metadata.version("halide"))

    def test_the_kernel_is_not_executed(self) -> None:
        marker = self.tmp / "executed.txt"
        code = f"import pathlib\npathlib.Path({str(marker)!r}).write_text('x')\n" + HALIDE_GEMM
        self.assertTrue(self.build(code).ok)
        self.assertFalse(marker.exists(), "빌드 단계는 커널을 임포트·실행하지 않는다")


def kernel(body: str, imports: str = "import halide as hl") -> str:
    """본문 줄들(들여쓰기 없이)을 ``def gemm`` 안에 넣은 모듈."""
    lines = "".join(f"    {line}\n" for line in body.splitlines())
    return f"{imports}\n\n\ndef gemm(A, B, M, N, K):\n    f = hl.Func('f')\n{lines}    return f\n"


class ScheduleCallsTest(BuildDirTest):
    def calls(self, code: str) -> list[str]:
        res = self.build(code)
        self.assertTrue(res.ok, res.error_detail)
        return res.extra["schedule_calls"]

    def test_one_entry_per_call_site_sorted_by_name(self) -> None:
        body = ("f.vectorize(x, 8)\nf.parallel(y)\nf.update().vectorize(x, 8).unroll(c)\n"
                "f.compute_root().split(x, xo, xi, 64).reorder(xi, y, xo)\ng.in_(f).compute_at(f, y)\ng.store_at(f, yo)")
        self.assertEqual(self.calls(kernel(body)), ["compute_at", "compute_root", "in_", "parallel", "reorder", "split", "store_at",
                                                    "unroll", "vectorize", "vectorize"])

    def test_every_contract_name_is_recognised(self) -> None:
        body = "\n".join(f"f.{name}()" for name in CONTRACT_SCHEDULE_METHODS)
        self.assertEqual(self.calls(kernel(body)), sorted(CONTRACT_SCHEDULE_METHODS))

    def test_other_halide_methods_are_not_schedule_calls(self) -> None:
        body = ("f.update(0)\nf.set_estimates([(0, N), (0, M)])\nA.dim(0).set_estimate(0, K)\nf.realize([N, M])\n"
                "f.compile_jit(hl.get_host_target())\nf.print_loop_nest()\nf.trace_stores()\nhl.Pipeline(f).apply_autoscheduler(t, p)")
        self.assertEqual(self.calls(kernel(body)), [])

    def test_calls_on_string_literals_and_imported_modules_are_not_schedule_calls(self) -> None:
        body = ('names = "i,j".split(",")\nlabel = f"{M}x{N}".rename if False else None\nparts = f"{M}_{N}".split("_")\n'
                'head = b"a b".partition(b" ")\ntiles = np.tile(np.arange(4), 2)\nhalves = np.split(np.arange(4), 2)\n'
                'folder = os.path.split(__file__)\nos.rename("a", "b")\nf.compute_root()')
        self.assertEqual(self.calls(kernel(body, imports="import os\n\nimport halide as hl\nimport numpy as np")), ["compute_root"])

    def test_receiver_types_are_not_known_to_the_ast(self) -> None:
        # 변수에 담긴 문자열의 메서드는 Func 의 메서드와 구별되지 않는다: 리터럴과 임포트한 모듈이 아닌 수신 객체의 호출은 센다
        self.assertEqual(self.calls(kernel('text = "i,j"\nnames = text.split(",")')), ["split"])
        # 호출 결과(``hl.Func("g")``, ``….update()``)에 대한 호출은 식의 뿌리가 임포트한 모듈이어도 센다
        self.assertEqual(self.calls(kernel('hl.Func("g").compute_root()\nhl.Func("h").update().parallel(y)')), ["compute_root", "parallel"])

    def test_schedule_outside_the_task_function_is_still_seen(self) -> None:
        code = ("import halide as hl\n\n\ndef _schedule(f, y):\n    f.parallel(y)\n\n\ndef gemm(A, B, M, N, K):\n"
                "    f = hl.Func('f')\n    _schedule(f, hl.Var('y'))\n    return f\n")
        self.assertEqual(self.calls(code), ["parallel"])

    def test_names_alone_are_not_calls(self) -> None:
        body = "method = f.parallel\nvectorize = 8\nsplit(x)\n# f.vectorize(x, 8)\nnote = 'f.reorder(x, y)'"
        self.assertEqual(self.calls(kernel(body)), [])


class ImportsHalideTest(BuildDirTest):
    def test_import_forms(self) -> None:
        cases = {
            "import halide as hl": True,
            "import halide": True,
            "import numpy as np, halide as hl": True,
            "from halide import Func, Var": True,
            "import halide.imageparam": True,
            "try:\n    import halide as hl\nexcept ImportError:\n    hl = None": True,
            "import numpy as np": False,
            "import halide_tools as hl": False,
            "from . import halide as hl": False,
            "from .halide import Func": False,
            "hl = None  # import halide as hl": False,
        }
        for imports, expected in cases.items():
            with self.subTest(imports=imports):
                res = self.build(kernel("f.compute_root()", imports=imports))
                self.assertTrue(res.ok, res.error_detail)
                self.assertIs(res.extra["imports_halide"], expected)

    def test_import_inside_the_function(self) -> None:
        code = "def gemm(A, B, M, N, K):\n    import halide as hl\n    return hl.Func('f')\n"
        self.assertEqual(self.build(code).extra, {"schedule_calls": [], "imports_halide": True})


class NotAKernelTest(BuildDirTest):
    def test_function_not_found(self) -> None:
        cases = {
            "other name": HALIDE_GEMM.replace("def gemm(", "def matmul("),
            "nested": "import halide as hl\n\n\ndef make():\n    def gemm(A, B, M, N, K):\n        return hl.Func('f')\n    return gemm\n",
            "async": HALIDE_GEMM.replace("def gemm(", "async def gemm("),
            "assigned": "import halide as hl\n\n\ndef _gemm(A, B, M, N, K):\n    return hl.Func('f')\n\n\ngemm = _gemm\n",
        }
        for name, code in cases.items():
            with self.subTest(name):
                res = self.build(code)
                self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "signature_mismatch", "function_not_found"))
                self.assertEqual(set(res.extra), {"schedule_calls", "imports_halide"})

    def test_syntax_error_is_a_compile_error(self) -> None:
        broken = HALIDE_GEMM_MANUAL.replace("C.update(0).reorder(j, k.x, i)", "C.update(0).reorder(j, k.x, i")
        res = self.build(broken)
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.extra), (False, None, "compile_error", {}))
        self.assertRegex(res.error_detail, r"^SyntaxError: .+ \(line \d+\)$")
        self.assertEqual((self.out / "kernel.py").read_text(encoding="utf-8"), broken)
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["ok"], saved["error_kind"], saved["extra"]), (False, "compile_error", {}))


class CliTest(unittest.TestCase):
    def test_exit_code_and_json(self) -> None:
        with tempfile.TemporaryDirectory(prefix="build_halide_test_") as tmp:
            src = Path(tmp) / "gemm.py"
            src.write_text(HALIDE_GEMM_MANUAL, encoding="utf-8")
            for function, rc in (("gemm", 0), ("dot", 1)):
                stdout = io.StringIO()
                with self.subTest(function=function), contextlib.redirect_stdout(stdout):
                    self.assertEqual(main(["--src", str(src), "--out-dir", str(Path(tmp) / "out"), "--function", function,
                                           "--halide-mode", "auto"]), rc)
                    doc = json.loads(stdout.getvalue())
                    self.assertEqual(doc["extra"]["schedule_calls"], ["parallel", "parallel", "reorder", "vectorize", "vectorize"])
                    self.assertEqual(doc["error_detail"], "" if rc == 0 else "function_not_found")


if __name__ == "__main__":
    unittest.main()
