"""build/c.py 테스트 (설계 §7, v3 §6): 컴파일 명령(기본 인자에서는 v3 이전과 같다), ``args_header``·``extra_flags``·
``artifact_name``·``write_json``, ``ToolchainMissing``, 심벌 분류, ``BuildResult.extra``.

gcc 가 없는 개발 머신에서도 돌도록 대부분은 **가짜 컴파일러**(파이썬 스크립트: ``-o`` 파일을 만들거나 ``error:`` 를 내고 실패)와
고정된 ``nm`` 출력으로 ``build_c`` 의 흐름을 검사한다. 진짜 gcc·nm 빌드는 Linux 에서만 돈다(``RealGccTest``).
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import harness.build.c as build_c_mod
from harness.build.c import (
    BuildResult,
    ToolchainMissing,
    build_c,
    compile_command,
    count_warnings,
    error_lines,
    nm_dynamic,
    symbol_exported,
    undefined_symbols_of,
)
from harness.procutil import ProcResult
from tests._support import requires_linux, requires_tool

# 설계 §2 의 build 절(이 저장소의 기본 설정)과 같은 값. 설정 로더를 거치지 않는다(다른 그룹이 설정을 고치는 중에도 돈다).
GCC_CFG = SimpleNamespace(
    cc="gcc",
    cflags=["-O3", "-march=native"],
    base_flags=["-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition"],
    openmp_flag="-fopenmp",
    ldflags=["-lm", "-Wl,--no-undefined"],
    diag_flags=["-fopt-info-vec-optimized", "-fopt-info-vec-missed"],
    compile_timeout_s=60.0,
    forbidden_symbols=["pthread_create", "clone", "fork", "vfork", "system", "popen", "execve", "execv", "execl"],
    thread_symbols_prefix=["pthread_create", "GOMP_", "omp_", "__kmpc_"],
)

# 가짜 컴파일러: `python fake_cc.py <플래그…> -o <산출물> kernel.c <링크 플래그…>` (cwd = out_dir)
FAKE_CC = r'''
import pathlib
import sys
import time

args = sys.argv[1:]
out = pathlib.Path(args[args.index("-o") + 1])
src = pathlib.Path("kernel.c").read_text(encoding="utf-8")
if "SLOW" in src:
    time.sleep(60)
if "kernel_args.h" in src and not pathlib.Path("kernel_args.h").is_file():
    print("kernel.c:1:10: fatal error: kernel_args.h: No such file or directory", file=sys.stderr)
    sys.exit(1)
if "BROKEN" in src:
    print("kernel.c: In function 'gemm':", file=sys.stderr)
    print("kernel.c:3:12: error: expected ';' before '}' token", file=sys.stderr)
    print("kernel.c:4:1: error: expected declaration or statement at end of input", file=sys.stderr)
    sys.exit(1)
if "unused" in src:
    print("kernel.c:2:9: warning: unused variable 'unused' [-Wunused-variable]", file=sys.stderr)
out.write_bytes(b"\x7fELF fake shared object")
'''

C_OK = "void gemm(const float* A, const float* B, float* C, int M, int N, int K)\n{\n    C[0] = A[0] * B[0];\n}\n"

NM_DEFINED = """0000000000001139 T gemm
0000000000001100 t helper
0000000000004028 B counter
"""
NM_UNDEFINED = """                 U memset@GLIBC_2.2.5
                 U GOMP_parallel@GOMP_4.0
                 U omp_get_num_threads@OMP_1.0
                 U system@GLIBC_2.2.5
                 U memset@GLIBC_2.2.5
                 w __cxa_finalize@GLIBC_2.2.5
"""


def legacy_command(bcfg: Any, *, openmp: bool) -> list[str]:
    """v3 이전 ``compile_command`` 의 정의(설계 §7)를 그대로 옮긴 것: 기본 인자의 명령이 바뀌지 않았는지의 기준."""
    cmd = [bcfg.cc, *bcfg.base_flags, *bcfg.cflags]
    if openmp:
        cmd.append(bcfg.openmp_flag)
    cmd += [*bcfg.diag_flags, "-o", "kernel.so", "kernel.c", *bcfg.ldflags]
    return cmd


class CompileCommandTest(unittest.TestCase):
    def test_default_command_is_the_documented_one(self) -> None:
        self.assertEqual(compile_command(GCC_CFG, openmp=True), [
            "gcc", "-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition", "-O3", "-march=native", "-fopenmp",
            "-fopt-info-vec-optimized", "-fopt-info-vec-missed", "-o", "kernel.so", "kernel.c", "-lm", "-Wl,--no-undefined",
        ])
        for openmp in (False, True):
            with self.subTest(openmp=openmp):
                self.assertEqual(compile_command(GCC_CFG, openmp=openmp), legacy_command(GCC_CFG, openmp=openmp))
                self.assertEqual(compile_command(GCC_CFG, openmp=openmp, extra_flags=None, artifact_name="kernel.so"),
                                 legacy_command(GCC_CFG, openmp=openmp))
                self.assertEqual(compile_command(GCC_CFG, openmp=openmp, extra_flags=[]), legacy_command(GCC_CFG, openmp=openmp))

    def test_extra_flags_follow_cflags(self) -> None:
        self.assertEqual(compile_command(GCC_CFG, openmp=True, extra_flags=["-DN=4", "-g"]), [
            "gcc", "-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition", "-O3", "-march=native", "-DN=4", "-g",
            "-fopenmp", "-fopt-info-vec-optimized", "-fopt-info-vec-missed", "-o", "kernel.so", "kernel.c", "-lm", "-Wl,--no-undefined",
        ])

    def test_sanitizer_flags_drop_no_undefined(self) -> None:
        flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
        self.assertEqual(compile_command(GCC_CFG, openmp=False, extra_flags=flags, artifact_name="kernel_san.so"), [
            "gcc", "-std=gnu11", "-fPIC", "-shared", "-Wall", "-fno-semantic-interposition", "-O3", "-march=native",
            "-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g",
            "-fopt-info-vec-optimized", "-fopt-info-vec-missed", "-o", "kernel_san.so", "kernel.c", "-lm",
        ])
        for flags in (["-fsanitize=thread"], ["-g", "-fsanitize=undefined"]):
            with self.subTest(flags=flags):
                self.assertNotIn("-Wl,--no-undefined", compile_command(GCC_CFG, openmp=True, extra_flags=flags))
        # -fsanitize= 로 시작하는 플래그가 없으면 그대로 둔다(-fno-sanitize=…, -fsanitize-recover=… 는 계측을 켜는 플래그가 아니다)
        for flags in (["-g"], ["-fno-sanitize=all"], ["-fsanitize-recover=all"], ["-DX=-fsanitize=address"]):
            with self.subTest(flags=flags):
                self.assertEqual(compile_command(GCC_CFG, openmp=False, extra_flags=flags)[-2:], ["-lm", "-Wl,--no-undefined"])
        self.assertEqual(GCC_CFG.ldflags, ["-lm", "-Wl,--no-undefined"], "설정의 리스트를 고치면 안 된다")


class LogHelpersTest(unittest.TestCase):
    def test_nm_parsing(self) -> None:
        self.assertTrue(symbol_exported(NM_DEFINED, "gemm"))
        self.assertFalse(symbol_exported(NM_DEFINED, "helper"))     # 지역 심벌(t)
        self.assertFalse(symbol_exported(NM_DEFINED, "counter"))    # 함수가 아니다(B)
        self.assertFalse(symbol_exported(NM_DEFINED, "gem"))
        self.assertTrue(symbol_exported("0000000000001139 W dot\n", "dot"))
        # GNU 간접 함수(target_clones·ifunc): nm 은 소문자 i 로 찍는다. 대문자 I(간접 참조)는 정의된 함수가 아니다
        self.assertTrue(symbol_exported("0000000000001200 i gemm\n0000000000001100 t gemm.resolver\n", "gemm"))
        self.assertFalse(symbol_exported("0000000000001200 I gemm\n", "gemm"))
        self.assertEqual(undefined_symbols_of(NM_UNDEFINED), ["memset", "GOMP_parallel", "omp_get_num_threads", "system"])

    def test_error_and_warning_lines(self) -> None:
        log = ("kernel.c: In function 'gemm':\nkernel.c:3:12: error: expected ';' before '}' token\n"
               "kernel.c:2:9: warning: unused variable 'x' [-Wunused-variable]\nkernel.c:4:1: error: expected declaration\n")
        self.assertEqual(error_lines(log), ["kernel.c:3:12: error: expected ';' before '}' token", "kernel.c:4:1: error: expected declaration"])
        self.assertEqual(error_lines("collect2: ld returned 1 exit status\n\n"), ["collect2: ld returned 1 exit status"])
        self.assertEqual(count_warnings(log), 1)

    def test_build_result_extra_defaults_to_an_empty_dict(self) -> None:
        res = BuildResult(ok=True, artifact=Path("kernel.so"), error_kind="", error_detail="", log="", build_time_ms=1.0,
                          command=["gcc"], compiler="gcc", compiler_version="gcc 15", n_warnings=0)
        self.assertEqual(res.extra, {})
        self.assertEqual(res.to_dict()["extra"], {})
        self.assertEqual(res.to_dict()["artifact"], "kernel.so")
        other = BuildResult(ok=True, artifact=None, error_kind="", error_detail="", log="", build_time_ms=1.0, command=[],
                            compiler="numba", compiler_version="", n_warnings=0, extra={"decorator": "njit"})
        self.assertEqual(other.to_dict()["extra"], {"decorator": "njit"})
        self.assertEqual(res.extra, {}, "기본값 dict 를 인스턴스끼리 나눠 쓰면 안 된다")


class FakeCompilerTest(unittest.TestCase):
    """가짜 컴파일러와 고정된 nm 출력으로 ``build_c`` 의 흐름을 본다(플랫폼 무관)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_c_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fake_cc = self.tmp / "fake_cc.py"
        self.fake_cc.write_text(FAKE_CC, encoding="utf-8")
        self.out = self.tmp / "sample_0"
        # cc = 파이썬 인터프리터, 첫 "플래그" = 가짜 컴파일러 스크립트 → `python fake_cc.py -O3 … -o kernel.so kernel.c -lm …`
        self.bcfg = SimpleNamespace(**{**vars(GCC_CFG), "cc": sys.executable, "base_flags": [str(self.fake_cc), "-shared"]})
        self.nm_calls: list[tuple[str, str]] = []
        self.nm_defined = NM_DEFINED
        patcher = mock.patch.object(build_c_mod, "nm_dynamic", side_effect=self.fake_nm)
        patcher.start()
        self.addCleanup(patcher.stop)

    def fake_nm(self, out_dir: Path, mode_flag: str, artifact_name: str = "kernel.so") -> str:
        self.assertEqual(Path(out_dir), self.out)
        self.assertTrue((self.out / artifact_name).is_file(), "nm 은 산출물이 생긴 뒤에 불린다")
        self.nm_calls.append((mode_flag, artifact_name))
        return self.nm_defined if mode_flag == "--defined-only" else NM_UNDEFINED

    def files(self) -> set[str]:
        return {p.name for p in self.out.iterdir()}

    def test_default_arguments_keep_the_command_and_the_files(self) -> None:
        res = build_c(C_OK, self.out, self.bcfg, openmp=True, launch_prefix=[])
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.command, legacy_command(self.bcfg, openmp=True))
        self.assertEqual(res.artifact, self.out / "kernel.so")
        self.assertEqual(self.files(), {"kernel.c", "kernel.so", "build.log", "build.json"})
        self.assertEqual((self.out / "kernel.c").read_text(encoding="utf-8"), C_OK)
        self.assertEqual(self.nm_calls, [("--defined-only", "kernel.so"), ("--undefined-only", "kernel.so")])
        self.assertEqual((res.error_kind, res.error_detail, res.n_warnings, res.compiler, res.extra), ("", "", 0, sys.executable, {}))
        self.assertEqual(res.undefined_symbols, ["memset", "GOMP_parallel", "omp_get_num_threads", "system"])
        self.assertEqual(res.forbidden_symbols, ["system"])
        self.assertEqual(res.thread_symbols, ["GOMP_parallel", "omp_get_num_threads"])
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, json.loads(json.dumps(res.to_dict())))
        self.assertEqual(saved["extra"], {})
        self.assertEqual(saved["artifact"], str(self.out / "kernel.so"))

    def test_launch_prefix_comes_first(self) -> None:
        prefix = [sys.executable, "-c", "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))"]
        res = build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=prefix)
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.command, [*prefix, *legacy_command(self.bcfg, openmp=False)])

    def test_args_header_is_written_next_to_the_source(self) -> None:
        header = "#ifndef KERNEL_ARGS_H\n#define KERNEL_ARGS_H\n\ntypedef struct {\n    int M;\n} gemm_args;\n\n#endif /* KERNEL_ARGS_H */\n"
        code = '#include "kernel_args.h"\nvoid gemm(const gemm_args* args) { (void)args; }\n'
        res = build_c(code, self.out, self.bcfg, openmp=False, launch_prefix=[], args_header=header)
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual((self.out / "kernel_args.h").read_bytes().decode("utf-8"), header)
        self.assertEqual(self.files(), {"kernel.c", "kernel_args.h", "kernel.so", "build.log", "build.json"})
        self.assertEqual(res.command, legacy_command(self.bcfg, openmp=False), "헤더는 명령을 바꾸지 않는다")
        # 헤더를 주지 않으면 쓰지 않는다 → 그 헤더를 포함하는 커널은 컴파일 오류다
        other = self.tmp / "sample_1"
        self.out = other
        res = build_c(code, other, self.bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertIn("kernel_args.h: No such file or directory", res.error_detail)
        self.assertFalse((other / "kernel_args.h").exists())

    def test_extra_flags_artifact_name_and_no_json(self) -> None:
        flags = ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"]
        res = build_c(C_OK, self.out, self.bcfg, openmp=True, launch_prefix=[], extra_flags=flags, artifact_name="kernel_san.so",
                      write_json=False)
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.command, [
            sys.executable, str(self.fake_cc), "-shared", "-O3", "-march=native", *flags, "-fopenmp",
            "-fopt-info-vec-optimized", "-fopt-info-vec-missed", "-o", "kernel_san.so", "kernel.c", "-lm",
        ])
        self.assertEqual(res.artifact, self.out / "kernel_san.so")
        self.assertEqual(self.files(), {"kernel.c", "kernel_san.so", "build.log"})
        self.assertEqual(self.nm_calls, [("--defined-only", "kernel_san.so"), ("--undefined-only", "kernel_san.so")])

    def test_write_json_false_on_failure_too(self) -> None:
        res = build_c("BROKEN", self.out, self.bcfg, openmp=False, launch_prefix=[], write_json=False)
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertEqual(self.files(), {"kernel.c", "build.log"})

    def test_stale_artifact_is_removed_before_compiling(self) -> None:
        self.assertTrue(build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=[]).ok)
        res = build_c("BROKEN", self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.artifact, res.error_kind), (False, None, "compile_error"))
        self.assertFalse((self.out / "kernel.so").exists(), "이전 빌드의 산출물이 남으면 안 된다")
        self.assertEqual(json.loads((self.out / "build.json").read_text(encoding="utf-8"))["ok"], False)

    def test_compile_error_detail_and_warnings(self) -> None:
        res = build_c("BROKEN", self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.artifact, res.error_kind), (False, None, "compile_error"))
        self.assertEqual(res.error_detail, "kernel.c:3:12: error: expected ';' before '}' token | "
                                           "kernel.c:4:1: error: expected declaration or statement at end of input")
        self.assertIn("In function 'gemm'", res.log)
        self.assertEqual(self.nm_calls, [], "컴파일이 실패하면 nm 을 부르지 않는다")
        self.assertEqual((res.undefined_symbols, res.forbidden_symbols, res.thread_symbols), ([], [], []))
        self.out = self.tmp / "sample_1"
        res = build_c(C_OK.replace("C[0] =", "int unused; C[0] ="), self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.n_warnings), (True, 1))

    def test_timeout(self) -> None:
        bcfg = SimpleNamespace(**{**vars(self.bcfg), "compile_timeout_s": 0.5})
        res = build_c("SLOW", self.out, bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "timeout", "compile timeout after 0.5s"))
        self.assertEqual(json.loads((self.out / "build.json").read_text(encoding="utf-8"))["error_kind"], "timeout")

    def test_symbol_not_exported(self) -> None:
        self.nm_defined = "0000000000001100 t gemm\n0000000000001139 T gemm_impl\n"
        res = build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "signature_mismatch", "symbol gemm not exported"))
        self.assertEqual(res.forbidden_symbols, ["system"], "심벌 관찰값은 실패한 결과에도 남는다")
        self.nm_defined = NM_DEFINED
        res = build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=[], symbol="dot")
        self.assertEqual((res.error_kind, res.error_detail), ("signature_mismatch", "symbol dot not exported"))


class ToolchainMissingTest(unittest.TestCase):
    """없는 도구는 ``compile_error`` 결과가 아니라 ``ToolchainMissing`` 예외다(모델 오류로 기록되면 안 된다)."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_c_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.fake_cc = self.tmp / "fake_cc.py"
        self.fake_cc.write_text(FAKE_CC, encoding="utf-8")
        self.out = self.tmp / "sample_0"
        self.bcfg = SimpleNamespace(**{**vars(GCC_CFG), "cc": sys.executable, "base_flags": [str(self.fake_cc)]})
        self.missing = SimpleNamespace(**{**vars(GCC_CFG), "cc": "no-such-compiler-for-build-c-test"})

    def test_is_a_runtime_error_but_not_an_os_error(self) -> None:
        self.assertTrue(issubclass(ToolchainMissing, RuntimeError))
        self.assertFalse(issubclass(ToolchainMissing, OSError))

    def test_missing_compiler(self) -> None:
        with self.assertRaises(ToolchainMissing) as cm:
            build_c(C_OK, self.out, self.missing, openmp=False, launch_prefix=[])
        self.assertIn("no-such-compiler-for-build-c-test", str(cm.exception))
        self.assertFalse(self.out.exists(), "도구가 없으면 아무것도 쓰지 않는다")

    def test_missing_compiler_behind_a_launch_prefix(self) -> None:
        # taskset 같은 접두가 있으면 없는 컴파일러는 예외가 아니라 접두 프로그램의 rc=127 로 나타난다: 예전에는 compile_error 였다
        prefix = [sys.executable, "-c", "import sys; print('taskset: failed to execute gcc', file=sys.stderr); sys.exit(127)"]
        with self.assertRaises(ToolchainMissing):
            build_c(C_OK, self.out, self.missing, openmp=False, launch_prefix=prefix)
        self.assertFalse((self.out / "build.json").exists())

    def test_command_that_cannot_be_started(self) -> None:
        with self.assertRaises(ToolchainMissing) as cm:
            build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=["no-such-launcher-for-build-c-test"])
        self.assertIn("no-such-launcher-for-build-c-test", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, OSError)
        self.assertFalse((self.out / "build.json").exists())

    def test_missing_nm(self) -> None:
        with mock.patch.object(build_c_mod, "NM", "no-such-nm-for-build-c-test"), self.assertRaises(ToolchainMissing) as cm:
            build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertIn("no-such-nm-for-build-c-test", str(cm.exception))
        self.assertFalse((self.out / "build.json").exists(), "심벌 검사를 못 했으면 결과를 쓰지 않는다")

    def test_failing_nm_is_not_a_missing_symbol(self) -> None:
        # nm 이 뜨기는 했는데 실패한 경우(여기서는 `python -D …` 가 rc=2 로 끝난다): 빈 심벌 목록으로 넘기면
        # `symbol gemm not exported` — 커널 탓 — 이 된다
        with mock.patch.object(build_c_mod, "NM", sys.executable), self.assertRaises(ToolchainMissing) as cm:
            build_c(C_OK, self.out, self.bcfg, openmp=False, launch_prefix=[])
        self.assertIn("-D --defined-only kernel.so` 실패(rc=2)", str(cm.exception))
        self.assertFalse((self.out / "build.json").exists())

    def test_nm_dynamic_result_rules(self) -> None:
        def proc(rc: int, timed_out: bool = False, stdout: str = "", stderr: str = "") -> ProcResult:
            return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.01, stdout_tail=stdout, stderr_tail=stderr, signal_name=None)

        with mock.patch.object(build_c_mod, "run_isolated", return_value=proc(0, stdout=NM_DEFINED)) as run:
            self.assertEqual(nm_dynamic(self.tmp, "--defined-only", "kernel_san.so"), NM_DEFINED)
        self.assertEqual(run.call_args.args[0], ["nm", "-D", "--defined-only", "kernel_san.so"])
        self.assertEqual(run.call_args.kwargs["cwd"], self.tmp)
        # 내보낸 심벌이 없는 라이브러리: GNU nm 은 `no symbols` 를 알리고 rc=0 으로 끝난다 → 실패가 아니라 빈 목록이다
        with mock.patch.object(build_c_mod, "run_isolated", return_value=proc(0, stderr="nm: kernel.so: no symbols\n")):
            self.assertEqual(nm_dynamic(self.tmp, "--defined-only"), "")
        failures = {
            "rc=1": proc(1, stderr="nm: kernel.so: file format not recognized\n"),
            "30s 안에 끝나지 않았다": proc(-9, timed_out=True),
        }
        for why, result in failures.items():
            with self.subTest(why=why):
                with mock.patch.object(build_c_mod, "run_isolated", return_value=result), self.assertRaises(ToolchainMissing) as cm:
                    nm_dynamic(self.tmp, "--undefined-only")
                self.assertIn(why, str(cm.exception))
                self.assertIn("nm -D --undefined-only kernel.so", str(cm.exception))


C_KEYWORD = """#include "kernel_args.h"

void gemm(const gemm_args* args)
{
    for (int i = 0; i < args->M * args->N; i++)
        args->C[i] = SCALE * args->A[0] * args->B[0];
}
"""
C_KEYWORD_HEADER = """#ifndef KERNEL_ARGS_H
#define KERNEL_ARGS_H

typedef struct {
    const float* A;
    const float* B;
    float* C;
    int M;
    int N;
    int K;
} gemm_args;

#endif /* KERNEL_ARGS_H */
"""
C_UNDEFINED_CALL = """extern void defined_nowhere_for_build_c_test(float* C);

void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    C[0] = A[0] * B[0];
    defined_nowhere_for_build_c_test(C);
}
"""


C_TARGET_CLONES = """__attribute__((target_clones("avx2", "default")))
void gemm(const float* A, const float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}
"""

# 라이브러리를 ctypes 로 열어 2x2 GEMM 을 한 번 부른다(argv[1] = 라이브러리 경로)
CALL_GEMM_2X2 = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
F4 = ctypes.c_float * 4
A, B, C = F4(1, 2, 3, 4), F4(5, 6, 7, 8), F4()
lib.gemm(A, B, C, 2, 2, 2)
print(*C)
"""


@requires_linux("gcc 로 ELF 공유 라이브러리를 만들고 nm -D 로 읽는다")
@requires_tool("gcc", "nm")
class RealGccTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_c_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_default_build(self) -> None:
        out = self.tmp / "a"
        res = build_c(C_OK, out, GCC_CFG, openmp=True, launch_prefix=[])
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.command, legacy_command(GCC_CFG, openmp=True))
        self.assertEqual({p.name for p in out.iterdir()}, {"kernel.c", "kernel.so", "build.log", "build.json"})
        self.assertEqual((res.artifact, res.forbidden_symbols, res.extra), (out / "kernel.so", [], {}))
        self.assertTrue(res.compiler_version.startswith("gcc"), res.compiler_version)
        self.assertIn("-march=", res.march_resolved)
        res = build_c(C_OK.replace("gemm", "other"), self.tmp / "b", GCC_CFG, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.error_kind, res.error_detail), (False, "signature_mismatch", "symbol gemm not exported"))

    def test_keyword_header_extra_flags_and_artifact_name(self) -> None:
        out = self.tmp / "kw"
        res = build_c(C_KEYWORD, out, GCC_CFG, openmp=False, launch_prefix=[], args_header=C_KEYWORD_HEADER,
                      extra_flags=["-DSCALE=2.0f"], artifact_name="kernel_alt.so")
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(res.artifact, out / "kernel_alt.so")
        self.assertEqual({p.name for p in out.iterdir()}, {"kernel.c", "kernel_args.h", "kernel_alt.so", "build.log", "build.json"})
        # 헤더 없이는 포함 실패, 매크로 없이는 미선언 식별자 — 둘 다 모델의 컴파일 오류다
        res = build_c(C_KEYWORD, self.tmp / "no_header", GCC_CFG, openmp=False, launch_prefix=[], extra_flags=["-DSCALE=2.0f"])
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertIn("kernel_args.h", res.error_detail)
        res = build_c(C_KEYWORD, self.tmp / "no_macro", GCC_CFG, openmp=False, launch_prefix=[], args_header=C_KEYWORD_HEADER)
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertIn("SCALE", res.error_detail)

    @unittest.skipUnless(platform.machine() in ("x86_64", "AMD64"), 'target_clones("avx2") 는 x86-64 전용')
    def test_target_clones_ifunc_is_an_exported_kernel(self) -> None:
        # gcc 는 target_clones 함수를 STT_GNU_IFUNC 로 내보낸다(nm: i). 정상적으로 내보낸 커널이다 — signature_mismatch 가 아니다
        out = self.tmp / "ifunc"
        res = build_c(C_TARGET_CLONES, out, GCC_CFG, openmp=False, launch_prefix=[])
        self.assertTrue(res.ok, f"{res.error_kind}: {res.error_detail}")
        nm = build_c_mod.run_isolated(["nm", "-D", "--defined-only", "kernel.so"], timeout_s=30, cwd=out).stdout_tail
        self.assertRegex(nm, r"(?m)^\S+ i gemm$")
        proc = subprocess.run([sys.executable, "-c", CALL_GEMM_2X2, str(res.artifact)], capture_output=True, text=True, timeout=60)
        self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["19.0", "22.0", "43.0", "50.0"]), proc.stderr[-2000:])

    def test_undefined_symbol_is_a_link_error_unless_sanitizing(self) -> None:
        res = build_c(C_UNDEFINED_CALL, self.tmp / "plain", GCC_CFG, openmp=False, launch_prefix=[])
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertIn("defined_nowhere_for_build_c_test", res.log)
        libasan = build_c_mod.run_isolated(["gcc", "-print-file-name=libasan.so"], timeout_s=30, cwd=self.tmp).stdout_tail.strip()
        if not Path(libasan).is_absolute():
            self.skipTest("libasan 이 없다(gcc -print-file-name=libasan.so)")
        res = build_c(C_UNDEFINED_CALL, self.tmp / "san", GCC_CFG, openmp=False, launch_prefix=[],
                      extra_flags=["-fsanitize=address", "-g"], artifact_name="kernel_san.so")
        self.assertTrue(res.ok, res.error_detail)   # -Wl,--no-undefined 가 빠졌다: 미정의 심벌은 로드 시점의 일이다
        self.assertNotIn("-Wl,--no-undefined", res.command)
        self.assertIn("defined_nowhere_for_build_c_test", res.undefined_symbols)


if __name__ == "__main__":
    unittest.main()
