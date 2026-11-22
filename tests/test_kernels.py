"""harness.kernels 단위 테스트 (설계 v3 §8): 옵션·예외 기록·inspect 규약·핸들 속성(백엔드 이름, 호출 친화도), C ABI 의 인자
타입과 매개변수 구조체, python 함수 로드·호출 규칙(기준선의 모듈 이름 포함), Halide 어댑터의 순수 부분(플러그인 파일 찾기, 출력
가로채기, auto 모드의 스케줄 호출 무시·기록 — 가짜 클래스로).

이 프로세스에서 도는 것은 여기서 임시 디렉터리에 쓴 몇 줄짜리 파이썬 모듈뿐이다. 공유 라이브러리 로드(ctypes)와 Numba·Halide
커널의 실행은 실제 워커 서브프로세스로 확인한다(``tests/test_worker.py``) — 테스트 프로세스에 라이브러리·JIT 상태를 남기지 않는다.
"""

from __future__ import annotations

import contextlib
import ctypes
import io
import os
import sys
import tempfile
import unittest
from collections.abc import Mapping
from ctypes import POINTER, c_float, c_int, c_int32
from pathlib import Path
from typing import Any

import numpy as np

from harness import kernels, sigs
from harness.kernels import cabi, halide_pipe, pyfunc
from harness.paths import ROOT
from harness.taskspec import load_taskspec, parse_taskspec

GEMM = load_taskspec(ROOT / "tasks" / "gemm" / "signatures.yaml")
# dim 과 포인터가 번갈아 나오는 과제: 구조체에 패딩이 생긴다(64비트: int 4 + 패딩 4 + 포인터 8 …)
INTERLEAVED = parse_taskspec({
    "task": "interleaved", "function": "interleaved",
    "args": [{"name": "n", "kind": "dim"}, {"name": "x", "kind": "in", "dtype": "float32", "shape": ["n"]},
             {"name": "m", "kind": "dim"}, {"name": "idx", "kind": "in", "dtype": "int32", "shape": ["m"]},
             {"name": "out", "kind": "out", "dtype": "float32", "shape": ["n"]}],
})


class ErrorRecordTest(unittest.TestCase):
    def test_one_line_uses_the_first_line_unless_asked_to_join(self) -> None:
        e = ValueError("first line  \n   second line\n\nthird")
        self.assertEqual(kernels.one_line(e), "ValueError: first line")
        self.assertEqual(kernels.one_line(e, join_lines=True), "ValueError: first line second line third")
        self.assertEqual(kernels.one_line(KeyError()), "KeyError")
        self.assertEqual(kernels.one_line(RuntimeError("  \n ")), "RuntimeError")
        self.assertEqual(len(kernels.one_line(RuntimeError("x" * 2000))), kernels.ONE_LINE_CHARS)

    def test_exception_whose_str_raises_is_still_recorded(self) -> None:
        class Weird(Exception):
            def __str__(self) -> str:
                raise SystemExit("no str")

        self.assertEqual(kernels.safe_str(Weird()), "<unprintable Weird>")
        self.assertEqual(kernels.one_line(Weird()), "Weird: <unprintable Weird>")
        try:
            raise Weird()
        except Weird as e:
            record = kernels.exception_record(e)
        self.assertEqual((record["type"], record["message"]), ("Weird", "<unprintable Weird>"))
        self.assertIn("Traceback", record["traceback"])

    def test_exception_record_is_bounded(self) -> None:
        try:
            raise RuntimeError("m" * 5000)
        except RuntimeError as e:
            record = kernels.exception_record(e)
        self.assertEqual(sorted(record), ["message", "traceback", "type"])
        self.assertEqual(len(record["message"]), kernels.MESSAGE_CHARS)
        self.assertLessEqual(len(record["traceback"]), kernels.TRACEBACK_TAIL_CHARS)

    def test_crash_detail_is_one_line_built_from_the_record(self) -> None:
        self.assertEqual(kernels.single_line("  first \n\n   second line\r\nthird\n"), "first second line third")
        self.assertEqual(kernels.single_line(""), "")
        try:
            raise RuntimeError("Error: Input buffer A is accessed at 3,\n  which is beyond the max (2) in dimension 0\n")
        except RuntimeError as e:
            record = kernels.exception_record(e)
        self.assertTrue(record["message"].endswith("dimension 0\n"), "기록의 메시지는 원문 그대로다")
        self.assertEqual(kernels.crash_detail(record),
                         "RuntimeError: Error: Input buffer A is accessed at 3, which is beyond the max (2) in dimension 0")
        self.assertEqual(kernels.crash_detail({"type": "KeyError", "message": "", "traceback": ""}), "KeyError")
        self.assertEqual(kernels.crash_detail({"type": "Weird", "message": "<unprintable Weird>", "traceback": ""}), "Weird: <unprintable Weird>")
        self.assertEqual(len(kernels.crash_detail({"type": "E", "message": "x" * 2000, "traceback": ""})), kernels.ONE_LINE_CHARS)

    def test_crash_detail_without_joining_is_the_plain_type_and_message(self) -> None:
        """c·python 커널(``join_error_lines`` False)의 설명은 ``f"{type}: {message}".rstrip(": ")[:500]`` 그대로다(줄바꿈 유지)."""
        for message in ("boom", "first\nsecond", "ends with newline\n", "", "x" * 2000):
            record = {"type": "RuntimeError", "message": message, "traceback": ""}
            self.assertEqual(kernels.crash_detail(record, join_lines=False), f"RuntimeError: {message}".rstrip(": ")[:500], message)
        self.assertEqual(kernels.crash_detail({"type": "E", "message": "a\nb", "traceback": ""}), "E: a b")

    def test_kernel_error_record(self) -> None:
        plain = kernels.KernelError("signature_mismatch", "gemm returned NoneType; expected hl.Func")
        self.assertEqual((plain.classification, plain.detail, plain.exception),
                         ("signature_mismatch", "gemm returned NoneType; expected hl.Func", None))
        self.assertEqual(plain.record(), {"type": "signature_mismatch", "message": "gemm returned NoneType; expected hl.Func", "traceback": ""})
        raised = {"type": "ValueError", "message": "boom", "traceback": "Traceback ..."}
        self.assertIs(kernels.KernelError("compile_error", "ValueError: boom", raised).record(), raised)
        long = kernels.KernelError("compile_error", "JIT compilation failed: " + "x" * 2000)
        self.assertEqual((len(long.detail), long.detail[:24]), (kernels.ONE_LINE_CHARS, "JIT compilation failed: "))


class OptionsTest(unittest.TestCase):
    def test_autoscheduler_params_are_strings_with_parallelism_from_threads(self) -> None:
        options = kernels.KernelOptions(threads=4, halide_mode="auto", halide_params={"balance": 40, "last_level_cache_size": "16777216"})
        self.assertEqual(kernels.autoscheduler_params(options), {"parallelism": "4", "balance": "40", "last_level_cache_size": "16777216"})
        # 추가 매개변수가 뒤에 오므로 설정이 parallelism 을 덮어쓸 수 있다(§8.3)
        override = kernels.KernelOptions(threads=4, halide_mode="auto", halide_params={"parallelism": 2})
        self.assertEqual(kernels.autoscheduler_params(override), {"parallelism": "2"})
        with self.assertRaises(ValueError):
            kernels.autoscheduler_params(kernels.KernelOptions(halide_mode="auto"))

    def test_backend_detail(self) -> None:
        self.assertEqual(kernels.backend_detail("c", kernels.KernelOptions()), {})
        self.assertEqual(kernels.backend_detail("python", kernels.KernelOptions(args_mode="keyword")), {})
        self.assertEqual(kernels.backend_detail("numba", kernels.KernelOptions(threads=4)), {"threads": 4})
        self.assertEqual(kernels.backend_detail("halide", kernels.KernelOptions(threads=2, halide_mode="manual")),
                         {"threads": 2, "halide_mode": "manual", "autoscheduler": None})
        self.assertEqual(kernels.backend_detail("halide", kernels.KernelOptions(threads=2, halide_mode="auto", halide_params={"balance": 40})),
                         {"threads": 2, "halide_mode": "auto", "autoscheduler": "Li2018",
                          "autoscheduler_params": {"parallelism": "2", "balance": "40"}})
        self.assertEqual(kernels.backend_detail("halide", kernels.KernelOptions(threads=1, halide_mode="auto", halide_autoscheduler="Adams2019"))
                         ["autoscheduler"], "Adams2019")
        # module_name (교차 측정 기준선의 모듈 이름)은 받은 백엔드 설정이 아니다
        self.assertEqual(kernels.backend_detail("python", kernels.KernelOptions(module_name="baseline_kernel")), {})

    def test_defaults(self) -> None:
        """기본 오토스케줄러는 Li2018 (2026-11-12 결정), 모듈 이름은 ``kernel``."""
        options = kernels.KernelOptions()
        self.assertEqual((kernels.DEFAULT_AUTOSCHEDULER, options.halide_autoscheduler, options.module_name, options.args_mode),
                         ("Li2018", "Li2018", "kernel", "positional"))
        self.assertEqual(pyfunc.MODULE_NAME, "kernel")

    def test_open_backend_rejects_unknown_names_before_touching_any_tool(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend"):
            kernels.open_backend("rust", kernels.KernelOptions())
        with self.assertRaisesRegex(ValueError, "args_mode"):
            kernels.open_backend("python", kernels.KernelOptions(args_mode="named"))
        # halide_mode 검사는 halide 임포트보다 먼저다(패키지가 없는 머신에서도 사용법 오류는 사용법 오류다)
        with self.assertRaisesRegex(ValueError, "halide_mode"):
            kernels.open_backend("halide", kernels.KernelOptions(threads=2))
        with self.assertRaisesRegex(ValueError, "halide_mode"):
            halide_pipe.make_loader(kernels.KernelOptions(threads=2, halide_mode="fast"))


class _Handle(kernels.KernelHandle):
    """``inspect`` 규약 확인용: 값을 하나 넣고 실패한다."""

    def __init__(self, error: Exception | None) -> None:
        self._error = error

    def prepare(self, dims: Mapping[str, int], arrays: Mapping[str, np.ndarray] | None = None) -> float | None:
        return None

    def invoke(self) -> Any:
        return None

    def _collect(self, info: dict[str, Any], dims: Mapping[str, int] | None) -> None:
        info["collected"] = dict(dims or {})
        if self._error is not None:
            raise self._error


class InspectContractTest(unittest.TestCase):
    def test_inspect_keeps_what_was_collected_and_never_raises(self) -> None:
        self.assertEqual(_Handle(None).inspect({"n": 3}), {"collected": {"n": 3}})
        info = _Handle(RuntimeError("no diagnostics\nfor this function")).inspect()
        self.assertEqual(info, {"collected": {}, "inspect_error": "RuntimeError: no diagnostics for this function"})
        kernel_side = _Handle(kernels.KernelError("compile_error", "autoscheduler X failed: HalideError: nope")).inspect()
        self.assertEqual(kernel_side["inspect_error"], "KernelError: compile_error: autoscheduler X failed: HalideError: nope")

    def test_default_flags(self) -> None:
        self.assertFalse(kernels.KernelHandle.per_shape_build)
        self.assertFalse(kernels.KernelHandle.has_return_value)
        self.assertFalse(kernels.KernelHandle.join_error_lines)
        self.assertEqual((kernels.KernelHandle.backend, kernels.KernelHandle.call_cpus, kernels.KernelHandle.affinity_error,
                          kernels.KernelHandle.ignored_schedule_calls), ("", None, None, ()))
        classes = (cabi.CKernel, pyfunc.PyKernel, pyfunc.NumbaKernel, halide_pipe.HalideKernel)
        # 형태별 빌드는 halide 만, 커널의 반환값을 돌려주는 것은 python 계열만, 오류 줄을 잇는 것은 numba·halide 만
        self.assertEqual({cls.__name__: (cls.per_shape_build, cls.has_return_value, cls.join_error_lines) for cls in classes},
                         {"CKernel": (False, False, False), "PyKernel": (False, True, False), "NumbaKernel": (False, True, True),
                          "HalideKernel": (True, False, True)})
        # 워커의 --backend 값(교차 측정 문서의 baseline_interleaved.backend)
        self.assertEqual({cls.__name__: cls.backend for cls in classes},
                         {"CKernel": "c", "PyKernel": "python", "NumbaKernel": "numba", "HalideKernel": "halide"})
        # 호출 친화도 속성은 모든 핸들에 있다(인스턴스에 덮어써도 클래스 기본값은 그대로)
        handle = _Handle(None)
        self.assertEqual((handle.call_cpus, handle.affinity_error), (None, None))
        handle.call_cpus, handle.affinity_error = (0, 2), "OSError: x"
        self.assertEqual((_Handle(None).call_cpus, _Handle(None).affinity_error, kernels.KernelHandle.call_cpus), (None, None, None))


class CAbiTypesTest(unittest.TestCase):
    """C ABI 의 인자 타입과 키워드 모드 구조체(§8.2). 실제 호출은 tests/test_worker.py 가 컴파일한 커널로 확인한다."""

    def test_argument_types_follow_the_task_spec(self) -> None:
        self.assertEqual([cabi.c_arg_type(a) for a in GEMM.args],
                         [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_int, c_int])
        self.assertEqual([cabi.c_arg_type(a) for a in INTERLEAVED.args],
                         [c_int, POINTER(c_float), c_int, POINTER(c_int32), POINTER(c_float)])

    def test_struct_fields_are_the_args_in_order(self) -> None:
        struct = cabi.args_struct_type(GEMM)
        self.assertTrue(issubclass(struct, ctypes.Structure))
        self.assertEqual(struct.__name__, "gemm_args")
        self.assertEqual([(name, ctype) for name, ctype in struct._fields_],
                         [("A", POINTER(c_float)), ("B", POINTER(c_float)), ("C", POINTER(c_float)), ("M", c_int), ("N", c_int), ("K", c_int)])
        # 필드 이름·순서는 sigs 가 프롬프트·빌드에 주는 헤더(kernel_args.h)와 같다
        header = sigs.render_args_struct(GEMM, "c")
        positions = [header.index(f" {name};") for name, _ in struct._fields_]
        self.assertEqual(positions, sorted(positions))

    def test_struct_layout_is_the_platform_c_layout(self) -> None:
        struct = cabi.args_struct_type(INTERLEAVED)
        ptr, integer = ctypes.sizeof(ctypes.c_void_p), ctypes.sizeof(c_int)
        # int n; float* x; int m; int* idx; float* out;  — 포인터는 포인터 크기 경계에 놓인다
        self.assertEqual([getattr(struct, name).offset for name in ("n", "x", "m", "idx", "out")],
                         [0, ptr, 2 * ptr, 3 * ptr, 4 * ptr])
        self.assertEqual(ctypes.sizeof(struct), 5 * ptr)
        self.assertGreater(ptr, integer, "전제: 64비트 ABI 에서 int 뒤에 패딩이 생긴다")

    def test_struct_carries_the_values_it_was_built_from(self) -> None:
        struct = cabi.args_struct_type(INTERLEAVED)
        x = np.arange(4, dtype=np.float32)
        idx = np.arange(3, dtype=np.int32)
        out = np.zeros(4, dtype=np.float32)
        value = struct(4, x.ctypes.data_as(POINTER(c_float)), 3, idx.ctypes.data_as(POINTER(c_int32)), out.ctypes.data_as(POINTER(c_float)))
        self.assertEqual((value.n, value.m), (4, 3))
        self.assertEqual(ctypes.addressof(value.x.contents), x.ctypes.data)
        self.assertEqual(ctypes.addressof(value.idx.contents), idx.ctypes.data)
        self.assertEqual(ctypes.addressof(value.out.contents), out.ctypes.data)

    def test_a_file_that_is_not_a_library_is_a_compile_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kernels_test_") as tmp:
            bogus = Path(tmp) / "kernel.so"
            bogus.write_text("this is not a shared library\n", encoding="utf-8")
            with self.assertRaises(kernels.KernelError) as ctx:
                cabi.load(bogus, GEMM, args_mode="positional")
            self.assertEqual(ctx.exception.classification, "compile_error")
            self.assertTrue(ctx.exception.detail.startswith("OSError"), ctx.exception.detail)
            with self.assertRaises(kernels.KernelError) as ctx:
                cabi.load(Path(tmp) / "missing.so", GEMM, args_mode="keyword")
            self.assertEqual(ctx.exception.classification, "compile_error")


PY_GEMM = "import numpy as np\n\n\ndef gemm(A, B, C, M, N, K):\n    np.matmul(A, B, out=C)\n    return (M, N, K)\n"
BASELINE_MODULE = "baseline_kernel"  # harness.worker.BASELINE_MODULE_NAME
PY_GEMM_KEYWORD_ONLY = PY_GEMM.replace("def gemm(A, B, C, M, N, K):", "def gemm(*, A, B, C, M, N, K):")


class PyFuncTest(unittest.TestCase):
    """python 함수 커널의 로드·호출 규칙(§8.2). 로드하는 모듈은 여기서 쓴 몇 줄짜리 파일이다."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="kernels_test_")
        self.tmp = Path(self._tmpdir.name)
        self.addCleanup(self._tmpdir.cleanup)
        for name in (pyfunc.MODULE_NAME, BASELINE_MODULE):
            saved = sys.modules.get(name)
            self.addCleanup(lambda name=name, saved=saved: sys.modules.pop(name, None) if saved is None
                            else sys.modules.__setitem__(name, saved))

    def kernel(self, code: str, name: str = "kernel.py") -> Path:
        path = self.tmp / name
        path.write_text(code, encoding="utf-8", newline="\n")
        return path

    def test_module_name_keeps_the_kernel_module(self) -> None:
        """교차 측정: 기준선을 ``baseline_kernel`` 로 올리면 커널 모듈(``kernel``)을 덮지 않는다(둘 다 ``gemm`` 을 정의한다)."""
        kernel = kernels.open_backend("python", kernels.KernelOptions())(self.kernel(PY_GEMM), GEMM)
        baseline_code = PY_GEMM.replace("return (M, N, K)", "return 'baseline'")
        baseline = kernels.open_backend("python", kernels.KernelOptions(module_name=BASELINE_MODULE))(
            self.kernel(baseline_code, "baseline.py"), GEMM)
        self.assertEqual(Path(sys.modules[pyfunc.MODULE_NAME].__file__).name, "kernel.py")
        self.assertEqual(Path(sys.modules[BASELINE_MODULE].__file__).name, "baseline.py")
        for handle, expected in ((kernel, (4, 5, 3)), (baseline, "baseline")):
            handle.prepare({"M": 4, "N": 5, "K": 3}, self.arrays())
            self.assertEqual(handle.invoke(), expected)
        self.assertEqual((kernel.backend, baseline.backend), ("python", "python"))

    def load_error(self, code: str, **kwargs: Any) -> kernels.KernelError:
        with self.assertRaises(kernels.KernelError) as ctx:
            pyfunc.load_function(self.kernel(code), "gemm", **kwargs)
        return ctx.exception

    @staticmethod
    def arrays() -> dict[str, np.ndarray]:
        rng = np.random.default_rng(0)
        return {"A": rng.random((4, 3), dtype=np.float32), "B": rng.random((3, 5), dtype=np.float32),
                "C": np.zeros((4, 5), dtype=np.float32)}

    def test_positional_call_passes_arrays_and_python_ints_in_spec_order(self) -> None:
        handle = kernels.open_backend("python", kernels.KernelOptions())(self.kernel(PY_GEMM), GEMM)
        self.assertIsInstance(handle, pyfunc.PyKernel)
        arrays = self.arrays()
        self.assertIsNone(handle.prepare({"M": 4, "N": 5, "K": 3}, arrays))
        ret = handle.invoke()
        self.assertEqual(ret, (4, 5, 3))
        self.assertTrue(all(type(v) is int for v in ret))
        np.testing.assert_allclose(arrays["C"], arrays["A"] @ arrays["B"], rtol=1e-6)
        self.assertEqual(handle.inspect(), {})

    def test_keyword_call_works_with_a_keyword_only_definition(self) -> None:
        handle = kernels.open_backend("python", kernels.KernelOptions(args_mode="keyword"))(self.kernel(PY_GEMM_KEYWORD_ONLY), GEMM)
        arrays = self.arrays()
        handle.prepare({"M": 4, "N": 5, "K": 3}, arrays)
        self.assertEqual(handle.invoke(), (4, 5, 3))
        np.testing.assert_allclose(arrays["C"], arrays["A"] @ arrays["B"], rtol=1e-6)

    def test_positional_call_of_a_keyword_only_definition_raises_from_invoke(self) -> None:
        handle = kernels.open_backend("python", kernels.KernelOptions())(self.kernel(PY_GEMM_KEYWORD_ONLY), GEMM)
        handle.prepare({"M": 4, "N": 5, "K": 3}, self.arrays())
        with self.assertRaises(TypeError):      # 워커가 runtime_crash 로 기록한다
            handle.invoke()

    def test_prepare_without_arrays_binds_nothing(self) -> None:
        handle = kernels.open_backend("python", kernels.KernelOptions())(self.kernel(PY_GEMM), GEMM)
        self.assertIsNone(handle.prepare({"M": 4, "N": 5, "K": 3}))
        with self.assertRaisesRegex(RuntimeError, "prepare"):
            handle.invoke()

    def test_import_failures_are_compile_errors_with_the_exception_record(self) -> None:
        e = self.load_error("raise RuntimeError('boom\\nsecond line')\n")
        self.assertEqual((e.classification, e.detail), ("compile_error", "RuntimeError: boom"))
        assert e.exception is not None
        self.assertEqual((e.exception["type"], e.exception["message"]), ("RuntimeError", "boom\nsecond line"))
        self.assertIn("kernel.py", e.exception["traceback"])
        joined = self.load_error("raise RuntimeError('boom\\nsecond line')\n", join_error_lines=True)
        self.assertEqual(joined.detail, "RuntimeError: boom second line")
        self.assertEqual(self.load_error("import sys\nsys.exit(3)\n").detail, "SystemExit: 3")
        self.assertEqual(self.load_error("import this_module_does_not_exist_xyz\n").classification, "compile_error")

    def test_missing_or_unusable_function_is_a_signature_mismatch(self) -> None:
        for code, needle in (("def matmul(A, B, C, M, N, K):\n    pass\n", "module has no attribute gemm"),
                             ("gemm = 3\n", "gemm is not callable (int)"),
                             ("async def gemm(A, B, C, M, N, K):\n    pass\n", "gemm is a coroutine function (async def)")):
            e = self.load_error(code)
            self.assertEqual((e.classification, e.detail, e.exception), ("signature_mismatch", needle, None))


class HalidePureTest(unittest.TestCase):
    """Halide 어댑터에서 halide 패키지 없이 확인할 수 있는 부분."""

    def test_plugin_file_lookup(self) -> None:
        with tempfile.TemporaryDirectory(prefix="kernels_test_") as tmp:
            pkg = Path(tmp) / "halide"
            (pkg / "lib" / "cmake").mkdir(parents=True)
            (pkg / "lib64").mkdir()
            for rel in ("lib/autoschedule_mullapudi2016.dll", "lib64/libautoschedule_adams2019.so", "lib/libautoschedule_li2018.dylib",
                        "lib/cmake/autoschedule_mullapudi2016.cmake", "lib/autoschedule_anderson2021.txt"):
                (pkg / rel).write_bytes(b"")
            (pkg / "lib" / "autoschedule_fake2020.so").mkdir()      # 디렉터리는 플러그인이 아니다
            find = halide_pipe.find_autoscheduler_plugin
            self.assertEqual(find(pkg, "Mullapudi2016"), (pkg / "lib" / "autoschedule_mullapudi2016.dll").resolve())
            self.assertEqual(find(pkg, "Adams2019"), (pkg / "lib64" / "libautoschedule_adams2019.so").resolve())
            self.assertEqual(find(pkg, "Li2018"), (pkg / "lib" / "libautoschedule_li2018.dylib").resolve())
            for missing in ("Anderson2021", "Fake2020", "Nope2020", "", "*", "../lib/autoschedule_mullapudi2016", "Mullapudi2016.dll"):
                self.assertIsNone(find(pkg, missing), missing)
            found = find(pkg, "Mullapudi2016")
            assert found is not None
            self.assertTrue(found.is_absolute())

    def test_fd_level_stderr_capture_restores_the_descriptor(self) -> None:
        before = os.fstat(2)
        text = halide_pipe.capture_stderr_fd(lambda: os.write(2, b"produce C:\n  for i:\n"))
        self.assertEqual(text, "produce C:\n  for i:\n")
        with self.assertRaises(ZeroDivisionError):
            halide_pipe.capture_stderr_fd(lambda: 1 // 0)
        after = os.fstat(2)
        self.assertEqual((before.st_dev, before.st_ino, before.st_mode), (after.st_dev, after.st_ino, after.st_mode))
        self.assertEqual(halide_pipe.capture_stderr_fd(lambda: None), "")

    def test_compile_messages_go_to_stderr_not_stdout(self) -> None:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with halide_pipe.compile_messages_to_stderr():
                print("Warning: something was not scheduled")
            with self.assertRaises(ValueError), halide_pipe.compile_messages_to_stderr():
                print("printed before the failure")
                raise ValueError("compile step failed")
            print("after")
        self.assertEqual(out.getvalue(), "after\n")
        self.assertEqual(err.getvalue(), "Warning: something was not scheduled\nprinted before the failure\n")


class _FakeBase:
    def inherited(self, *args: Any) -> str:
        return "real inherited"


class _FakeFunc(_FakeBase):
    def parallel(self, *args: Any) -> str:
        return "real parallel"

    def compute_root(self) -> str:
        return "real compute_root"

    def specialize_fail(self, message: str) -> str:
        return "real specialize_fail"

    def update(self, index: int = 0) -> _FakeStage:      # 스케줄 지시가 아니다: 바꾸지 않는다
        return _FakeStage()


class _FakeStage:
    def reorder(self, *args: Any) -> str:
        return "real reorder"

    def rfactor(self, *args: Any) -> str:
        return "real rfactor"


class _FakePipeline:
    def apply_autoscheduler(self, *args: Any) -> str:
        return "real apply_autoscheduler"


class _FakeHalide:
    Func = _FakeFunc
    Stage = _FakeStage
    Pipeline = _FakePipeline


class HalideIgnoreScheduleTest(unittest.TestCase):
    """halide auto 의 '무시·기록' (2026-11-12 결정): 빌드 함수가 도는 동안만 스케줄 메서드가 기록만 하는 무동작이다.
    halide 패키지 없이 같은 모양의 가짜 클래스로 확인한다(실제 Halide 는 tests/test_worker.py 의 HalideFailureTest)."""

    METHODS = frozenset({"parallel", "compute_root", "specialize_fail", "reorder", "rfactor", "inherited", "not_a_method_here"})

    def test_calls_are_recorded_and_chain_while_inside(self) -> None:
        record: list[str] = []
        f = _FakeFunc()
        with halide_pipe.ignore_schedule_calls(_FakeHalide, self.METHODS, record):
            self.assertIs(f.parallel("i"), f)
            self.assertIs(f.compute_root().parallel("y"), f)
            self.assertIsNone(f.specialize_fail("no"))
            stage = f.update(0)
            self.assertIs(stage.reorder("j", "k"), stage)
            proxy = stage.rfactor([("r", "v")])
            self.assertIs(proxy.compute_root().update(0).vectorize("v", 8), proxy)
            self.assertIsNone(_FakePipeline().apply_autoscheduler("target", "params"))
            self.assertIs(f.inherited(), f)
        self.assertEqual(record, ["Func.parallel", "Func.compute_root", "Func.parallel", "Func.specialize_fail", "Stage.reorder",
                                  "Stage.rfactor", "Stage.rfactor->compute_root", "Stage.rfactor->update", "Stage.rfactor->vectorize",
                                  "Pipeline.apply_autoscheduler", "Func.inherited"])

    def test_original_methods_are_restored_even_after_an_exception(self) -> None:
        originals = {name: _FakeFunc.__dict__.get(name) for name in ("parallel", "compute_root", "specialize_fail")}
        record: list[str] = []
        with self.assertRaises(ValueError), halide_pipe.ignore_schedule_calls(_FakeHalide, self.METHODS, record):
            _FakeFunc().parallel("x")
            raise ValueError("the build function failed")
        self.assertEqual(record, ["Func.parallel"])
        self.assertEqual({name: _FakeFunc.__dict__.get(name) for name in originals}, originals)
        self.assertEqual((_FakeFunc().parallel(), _FakeStage().rfactor(), _FakePipeline().apply_autoscheduler()),
                         ("real parallel", "real rfactor", "real apply_autoscheduler"))
        # 상속한 메서드는 하위 클래스에 남지 않는다(덮어쓴 자리를 지운다)
        self.assertNotIn("inherited", _FakeFunc.__dict__)
        self.assertEqual(_FakeFunc().inherited(), "real inherited")
        self.assertEqual(_FakeFunc().update(0).reorder(), "real reorder")

    def test_method_list_covers_the_static_list_and_halide_22_additions(self) -> None:
        from harness.build.halide import SCHEDULE_METHODS

        self.assertTrue({"parallel", "vectorize", "compute_root", "split", "reorder", "rfactor"} <= SCHEDULE_METHODS)
        self.assertEqual(halide_pipe.EXTRA_SCHEDULE_METHODS & {"update", "set_estimate", "set_estimates", "realize", "compile_jit",
                                                               "print_loop_nest"}, set())
        self.assertIn("aligned_split", halide_pipe.EXTRA_SCHEDULE_METHODS)


if __name__ == "__main__":
    unittest.main()
