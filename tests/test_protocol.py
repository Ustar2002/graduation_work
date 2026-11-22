"""protocol 테스트: /proc/stat 파서·사용률 계산, launch_prefix/launch_env 규칙, 플래그 판정, 실제 동적 검사,
플랫폼 플래그, 선택 설정 기본값, 툴체인 정체(가짜 run_isolated)와 환경 지문."""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from harness import config as config_module
from harness import protocol
from harness.config import load_config, parse_cpulist
from harness.paths import PYTHON_EXE
from harness.procutil import ProcResult
from harness.protocol import (
    DynamicStatus,
    StaticStatus,
    blas_flag_label,
    busy_pct,
    check_dynamic,
    check_static,
    config_option,
    dropped_env_overrides,
    environment_fingerprint,
    flags_ok,
    is_informational,
    launch_env,
    launch_prefix,
    parse_proc_stat,
    platform_flag,
    sibling_cpus,
    toolchain_identity,
)
from tests._support import requires_modules

# 지문·프로브 테스트가 쓰는 고정 툴체인(``toolchain_identity`` 의 결과 형태)
TOOLCHAIN = {
    "cc": "gcc", "cc_version": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0", "cargo": "cargo 1.92.0 (344c4567c 2025-10-21)",
    "rustc": "rustc 1.92.0 (ded5c06cf 2025-12-08)", "numba_version": "0.68.1", "halide_version": "22.1.0",
    "halide_autoscheduler": "Li2018", "halide_autoscheduler_loadable": True,
    "lint_tools": {"ruff": "0.17.6", "pyflakes": "4.1.0", "cppcheck": "2.13.0", "clang-tidy": None}, "errors": {},
}

# 통합 전(6f02d16)의 환경 지문 키 — 측정 규약의 일부라 고정이다(docs/decisions.md 2026-11-12·2026-11-23): 키가 바뀌면 같은 머신·설정의 env_hash 가 바뀐다
UP_FINGERPRINT_KEYS = frozenset({
    "cpu_model", "kernel", "glibc", "cc", "cc_version", "march_resolved", "python_version", "numpy_version", "numpy_blas",
    "sdk_versions", "flags", "cores", "single_core", "threads", "omp_env", "call_affinity_policy", "bench_loop_fsync",
    "governor", "epp", "no_turbo",
})

PROC_STAT_BEFORE = """cpu  100 0 100 800 0 0 0 0 0 0
cpu0 10 0 10 80 0 0 0 0 0 0
cpu1 10 0 10 80 0 0 0 0 0 0
cpu2 0 0 0 100 0 0 0 0 0 0
intr 12345 0 0
ctxt 999
btime 1700000000
"""

PROC_STAT_AFTER = """cpu  200 0 200 1600 0 0 0 0 0 0
cpu0 40 0 20 140 0 0 0 0 0 0
cpu1 10 0 10 180 0 0 0 0 0 0
cpu2 0 0 0 150 50 0 0 0 0 0
"""


class ProcStatParserTest(unittest.TestCase):
    def test_parse_proc_stat_skips_aggregate_and_non_cpu_lines(self) -> None:
        parsed = parse_proc_stat(PROC_STAT_BEFORE)
        self.assertEqual(sorted(parsed), [0, 1, 2])
        self.assertEqual(parsed[0], (10, 0, 10, 80, 0, 0, 0, 0, 0, 0))

    def test_busy_pct(self) -> None:
        before = parse_proc_stat(PROC_STAT_BEFORE)
        after = parse_proc_stat(PROC_STAT_AFTER)
        # cpu0: total 100→200 (+100), idle 80→140 (+60) → busy 40%
        self.assertAlmostEqual(busy_pct(before, after, [0]), 40.0)
        # cpu1: +100 total, +100 idle → 0%
        self.assertAlmostEqual(busy_pct(before, after, [1]), 0.0)
        # cpu2: iowait 도 idle 로 센다 → +100 total, idle+iowait +100 → 0%
        self.assertAlmostEqual(busy_pct(before, after, [2]), 0.0)
        # 평균
        self.assertAlmostEqual(busy_pct(before, after, [0, 1]), 20.0)
        # 없는 CPU 는 무시, 전부 없으면 0.0
        self.assertAlmostEqual(busy_pct(before, after, [0, 99]), 40.0)
        self.assertEqual(busy_pct(before, after, [99]), 0.0)
        # 변화가 없으면(동일 샘플) 0.0
        self.assertEqual(busy_pct(before, before, [0, 1, 2]), 0.0)

    def test_parse_cpulist(self) -> None:
        self.assertEqual(parse_cpulist("0,2,4,6"), [0, 2, 4, 6])
        self.assertEqual(parse_cpulist("0-3,8"), [0, 1, 2, 3, 8])
        self.assertEqual(parse_cpulist("2"), [2])
        with self.assertRaises(ValueError):
            parse_cpulist("0, 2")
        with self.assertRaises(ValueError):
            parse_cpulist("a")


class FlagsTest(unittest.TestCase):
    def test_informational_flags_do_not_break_ok(self) -> None:
        self.assertTrue(is_informational("turbo=on"))
        self.assertTrue(is_informational("env_overrides_dropped"))
        self.assertFalse(is_informational("governor=powersave"))
        self.assertTrue(flags_ok([]))
        self.assertTrue(flags_ok(["turbo=on", "env_overrides_dropped"]))
        self.assertFalse(flags_ok(["turbo=on", "governor=powersave"]))
        self.assertTrue(flags_ok(["load_1m=2.4>2.0"]))  # 2026-11-08: 자기 부하라 정보용
        self.assertFalse(flags_ok(["pinned_busy=23.1%>10%"]))

    def test_blas_flag_label(self) -> None:
        self.assertEqual(blas_flag_label(""), "unknown")
        self.assertEqual(blas_flag_label("unknown"), "unknown")
        self.assertEqual(blas_flag_label("blas"), "reference")
        self.assertEqual(blas_flag_label("MKL"), "mkl")

    def test_dropped_env_overrides(self) -> None:
        env = {"PATH": "/bin", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "2", "KMP_AFFINITY": "x", "GOMP_CPU_AFFINITY": "0", "MKL_DEBUG": "1", "HOME": "/h"}
        self.assertEqual(dropped_env_overrides(env), ["GOMP_CPU_AFFINITY", "KMP_AFFINITY", "MKL_DEBUG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"])
        self.assertEqual(dropped_env_overrides({"PATH": "/bin"}), [])


class LaunchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    @unittest.skipIf(shutil.which("taskset") is None, "taskset 필요")
    def test_launch_prefix(self) -> None:
        self.assertEqual(launch_prefix(self.cfg, 1), ["taskset", "-c", self.cfg.measurement.single_core])
        self.assertEqual(launch_prefix(self.cfg, 4), ["taskset", "-c", self.cfg.measurement.cores])
        self.assertEqual(launch_prefix(self.cfg, self.cfg.measurement.threads), ["taskset", "-c", "0,2,4,6"])

    def test_launch_prefix_without_taskset(self) -> None:
        with mock.patch("harness.protocol.shutil.which", return_value=None):
            self.assertEqual(launch_prefix(self.cfg, 4), [])

    def test_launch_env_whitelist_and_threads(self) -> None:
        user_env = {
            "PATH": "/usr/bin:/bin", "HOME": "/home/x", "LANG": "ko_KR.UTF-8", "TMPDIR": "/tmp/x", "VIRTUAL_ENV": "/v",
            "LC_ALL": "ko_KR.UTF-8", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "16", "KMP_BLOCKTIME": "0",
            "ANTHROPIC_API_KEY": "secret", "PYTHONHASHSEED": "123", "SOME_RANDOM_VAR": "1",
        }
        with mock.patch.dict(os.environ, user_env, clear=True), mock.patch("harness.protocol._warned_env_drop", set()):
            with mock.patch("sys.stderr") as err:
                env = launch_env(self.cfg, 4, openmp=False)
            self.assertTrue(any("OMP_NUM_THREADS" in str(c) for c in err.write.call_args_list), "제거된 변수 경고가 있어야 한다")
        # 상속 화이트리스트
        for k in ("PATH", "HOME", "LANG", "TMPDIR", "VIRTUAL_ENV"):
            self.assertEqual(env[k], user_env[k])
        # 고정값
        self.assertEqual(env["LC_ALL"], "C")
        self.assertEqual(env["PYTHONHASHSEED"], "0")
        # 스레드 변수는 사용자 값이 아니라 threads
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "4")
        self.assertEqual(env["MKL_NUM_THREADS"], "4")
        # 비밀·잡동사니·사용자 스레드 변수는 상속하지 않는다
        for k in ("ANTHROPIC_API_KEY", "SOME_RANDOM_VAR", "KMP_BLOCKTIME"):
            self.assertNotIn(k, env)
        # openmp=False 면 OMP_PROC_BIND 등 없음
        self.assertNotIn("OMP_PROC_BIND", env)

    def test_launch_env_openmp_extra(self) -> None:
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            env = launch_env(self.cfg, 1, openmp=True)
        self.assertEqual(env["OMP_NUM_THREADS"], "1")
        self.assertEqual(env["OMP_PROC_BIND"], "true")
        self.assertEqual(env["OMP_PLACES"], "cores")
        self.assertEqual(env["OMP_DYNAMIC"], "false")
        self.assertNotIn("HOME", env)  # 사용자 환경에 없던 것은 생기지 않는다

    def test_launch_env_extra_env_is_applied_last(self) -> None:
        """언어의 env 설정(설계 v3 §3)은 스레드 변수·omp_env_extra 뒤에 덮어쓴다. 값은 문자열로 들어간다."""
        extra = {"NUMBA_NUM_THREADS": "4", "NUMBA_THREADING_LAYER": "omp", "OMP_PLACES": "threads", "HL_NUM_THREADS": 4}
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            plain = launch_env(self.cfg, 4, openmp=True)
            env = launch_env(self.cfg, 4, openmp=True, extra_env=extra)
            self.assertEqual(launch_env(self.cfg, 4, openmp=True, extra_env=None), plain)
            self.assertEqual(launch_env(self.cfg, 4, openmp=True, extra_env={}), plain)
        self.assertEqual(env["NUMBA_NUM_THREADS"], "4")
        self.assertEqual(env["NUMBA_THREADING_LAYER"], "omp")
        self.assertEqual(env["HL_NUM_THREADS"], "4")
        self.assertEqual(env["OMP_PLACES"], "threads", "extra_env 가 omp_env_extra 를 덮어쓴다")
        self.assertEqual(plain["OMP_PLACES"], "cores")
        # 나머지는 그대로다
        self.assertEqual({k: v for k, v in env.items() if k not in extra}, {k: v for k, v in plain.items() if k not in extra})
        self.assertNotIn("NUMBA_NUM_THREADS", plain)

    def test_launch_env_python_io_encoding_only_off_posix(self) -> None:
        """비-POSIX(개발용 Windows)에서만 워커의 표준 스트림을 UTF-8 로 고정한다. POSIX 의 환경은 그대로다."""
        with mock.patch.dict(os.environ, {"PATH": "/usr/bin"}, clear=True):
            here = launch_env(self.cfg, 4, openmp=False)
            with mock.patch.object(protocol, "IS_POSIX", True):
                posix = launch_env(self.cfg, 4, openmp=False)
            with mock.patch.object(protocol, "IS_POSIX", False):
                other = launch_env(self.cfg, 4, openmp=False)
        self.assertNotIn("PYTHONIOENCODING", posix)
        self.assertEqual(other["PYTHONIOENCODING"], "utf-8")
        self.assertEqual({k: v for k, v in other.items() if k != "PYTHONIOENCODING"}, posix)
        self.assertEqual(here, posix if os.name == "posix" else other)  # 이 머신의 실제 값


class PlatformFlagTest(unittest.TestCase):
    """비-Linux 는 ``platform=<sys.platform>`` 플래그(정보용 아님)로 표시된다 (설계 v3 §0)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_flag_by_platform_name(self) -> None:
        self.assertIsNone(platform_flag("linux"))
        self.assertEqual(platform_flag("win32"), "platform=win32")
        self.assertEqual(platform_flag("darwin"), "platform=darwin")
        self.assertFalse(is_informational("platform=win32"))
        self.assertFalse(flags_ok(["turbo=on", "platform=win32"]))

    def test_this_machine(self) -> None:
        import sys

        expected = None if sys.platform.startswith("linux") else f"platform={sys.platform}"
        self.assertEqual(platform_flag(), expected)

    def test_check_static_carries_the_flag(self) -> None:
        """이 머신에서 실제로 돈다(BLAS 정체 서브프로세스 포함): 플래그는 플랫폼이 정하고, 있으면 ok 가 아니다."""
        with contextlib.redirect_stderr(io.StringIO()) as err:
            st = check_static(self.cfg, {})
        self.assertIsInstance(st, StaticStatus)
        mark = platform_flag()
        self.assertEqual([f for f in st.flags if f.startswith("platform=")], [mark] if mark else [])
        self.assertEqual(st.ok, flags_ok(st.flags))
        if mark:
            self.assertFalse(st.ok)
            self.assertIn(mark, err.getvalue())

    def test_check_static_and_dynamic_with_a_foreign_platform(self) -> None:
        with mock.patch.object(protocol, "platform_flag", return_value="platform=testos"), \
                mock.patch.object(protocol, "blas_identity_subprocess", return_value={"name": "scipy-openblas", "version": "0.3.30"}), \
                contextlib.redirect_stderr(io.StringIO()):
            static = check_static(self.cfg, {})
            dynamic = check_dynamic(self.cfg, sample_s=0.0)
        self.assertIn("platform=testos", static.flags)
        self.assertFalse(static.ok)
        self.assertTrue(static.blas_ok)
        self.assertIn("platform=testos", dynamic.flags)
        self.assertFalse(dynamic.ok)

    def test_check_dynamic_on_linux_has_no_platform_flag(self) -> None:
        with mock.patch.object(protocol, "platform_flag", return_value=None), contextlib.redirect_stderr(io.StringIO()):
            dynamic = check_dynamic(self.cfg, sample_s=0.0)
        self.assertFalse([f for f in dynamic.flags if f.startswith("platform=")])

    def test_load_average_is_zero_where_the_os_has_none(self) -> None:
        with mock.patch.object(os, "getloadavg", None, create=True):
            self.assertEqual(protocol.read_load_1m(), 0.0)
            with mock.patch.object(protocol, "read_proc_stat", return_value={}), contextlib.redirect_stderr(io.StringIO()):
                st = check_dynamic(self.cfg, sample_s=0.0)  # /proc/stat 도 getloadavg 도 없는 플랫폼: 죽지 않고 0 을 준다
        self.assertEqual((st.load_1m, st.pinned_busy_pct, st.sibling_busy_pct), (0.0, 0.0, 0.0))
        with mock.patch.object(os, "getloadavg", lambda: (1.5, 1.0, 0.5), create=True):
            self.assertEqual(protocol.read_load_1m(), 1.5)


class ConfigOptionTest(unittest.TestCase):
    """선택 설정(설계 v3 §3)은 속성이 있으면 그 값, 없으면 ``OPTION_DEFAULTS`` — 그 기본값은 설정 로더의 기본값과 같아야 한다."""

    def test_present_attribute_wins(self) -> None:
        cfg = SimpleNamespace(lint=SimpleNamespace(enabled=False), halide=SimpleNamespace(autoscheduler="Adams2019"),
                              build=SimpleNamespace(rust=SimpleNamespace(cargo="/opt/cargo")))
        self.assertIs(config_option(cfg, "lint.enabled"), False)
        self.assertEqual(config_option(cfg, "halide.autoscheduler"), "Adams2019")
        self.assertEqual(config_option(cfg, "build.rust").cargo, "/opt/cargo")
        lang = SimpleNamespace(env={"HL_NUM_THREADS": "4"}, halide_mode="auto")
        self.assertEqual(config_option(lang, "env"), {"HL_NUM_THREADS": "4"})
        self.assertEqual(config_option(lang, "halide_mode"), "auto")

    def test_missing_attribute_anywhere_on_the_path_gives_the_default(self) -> None:
        for obj in (None, SimpleNamespace(), SimpleNamespace(halide=SimpleNamespace(), build=SimpleNamespace(cc="gcc"))):
            self.assertIs(config_option(obj, "lint.enabled"), True)
            self.assertIs(config_option(obj, "safety.enabled"), False)
            self.assertIs(config_option(obj, "perf_summary.enabled"), True)
            self.assertEqual(config_option(obj, "halide.autoscheduler"), "Li2018")  # 2026-11-12 결정
            self.assertEqual(config_option(obj, "halide.autoscheduler_params"), {})
            self.assertEqual(config_option(obj, "build.rust").cargo, "cargo")
            self.assertEqual(config_option(obj, "env"), {})
            self.assertIsNone(config_option(obj, "halide_mode"))
        with self.assertRaises(KeyError):
            config_option(SimpleNamespace(), "no.such.option")

    def test_defaults_match_the_config_loader(self) -> None:
        self.assertEqual(config_option(None, "lint.enabled"), config_module.LintCfg().enabled)
        self.assertEqual(config_option(None, "safety.enabled"), config_module.SafetyCfg().enabled)
        self.assertEqual(config_option(None, "perf_summary.enabled"), config_module.PerfSummaryCfg().enabled)
        halide = config_module.HalideCfg()
        self.assertEqual(config_option(None, "halide.autoscheduler"), halide.autoscheduler)
        self.assertEqual(config_option(None, "halide.autoscheduler_params"), halide.autoscheduler_params)
        self.assertEqual(vars(config_option(None, "build.rust")), vars(config_module.RustCfg()))
        fields = config_module.LanguageCfg.__dataclass_fields__
        self.assertEqual(config_option(None, "env"), fields["env"].default_factory())
        self.assertEqual(config_option(None, "halide_mode"), fields["halide_mode"].default)

    def test_there_is_no_settle_wait(self) -> None:
        """docs/decisions.md 2026-11-08: load_1m 은 정보용 플래그다 — 동적 검사 전에 부하가 내려가길 기다리는 설정은 없다."""
        self.assertFalse([k for k in protocol.OPTION_DEFAULTS if "settle" in k])
        self.assertNotIn("max_settle_s", config_module.MeasurementCfg.__dataclass_fields__)
        self.assertNotIn("settle_wait_s", DynamicStatus.__dataclass_fields__)
        self.assertFalse(hasattr(protocol, "settle_load"))

    def test_loaded_config_carries_every_option(self) -> None:
        cfg = load_config()
        for key in protocol.OPTION_DEFAULTS:
            obj = next(iter(cfg.languages.values())) if key in ("env", "halide_mode") else cfg
            value = obj
            for name in key.split("."):
                self.assertTrue(hasattr(value, name), f"{key}: 설정 객체에 {name} 이 없다")
                value = getattr(value, name)
            self.assertIs(config_option(obj, key), value)


def _proc(stdout: str = "", *, rc: int = 0, stderr: str = "") -> ProcResult:
    return ProcResult(rc=rc, timed_out=False, elapsed_s=0.01, stdout_tail=stdout, stderr_tail=stderr, signal_name=None)


PYTHON_TOOLCHAIN_DOC = {"numba_version": "0.68.1", "halide_version": "22.1.0", "halide_autoscheduler_loadable": True,
                        "lint_tools": {"ruff": "0.17.6", "pyflakes": "4.1.0"}, "errors": {}}
VERSION_OUTPUTS = {
    "gcc": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0\nCopyright (C) 2025 Free Software Foundation, Inc.\n",
    "/opt/rust/bin/cargo": "cargo 1.92.0 (344c4567c 2025-10-21)\n",
    "rustc": "rustc 1.92.0 (ded5c06cf 2025-12-08)\n",
    "cppcheck": "Cppcheck 2.13.0\n",
    "clang-tidy": "Ubuntu LLVM version 18.1.3\n  Optimized build.\n",
}


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def fake_which(present: set[str]):
    """``shutil.which`` 의 대역: 맨 이름은 ``present`` 에 있으면 ``/bin/<이름>``, 경로는 그 경로나 마지막 이름이 ``present`` 에
    있으면 그 경로 그대로(``rust_tool`` 이 찾은 경로를 ``compiler_version`` 이 다시 확인한다)."""
    def which(name: str, *args: object, **kwargs: object) -> str | None:
        if "/" in name.replace("\\", "/"):
            return name if name in present or _basename(name) in present else None
        return f"/bin/{name}" if name in present else None
    return which


class ToolchainIdentityTest(unittest.TestCase):
    """``toolchain_identity`` (설계 v3 §9.5)를 가짜 ``run_isolated`` 로: 어떤 명령을 띄우고 출력을 어떻게 읽는가."""

    def setUp(self) -> None:
        self.cfg = SimpleNamespace(build=SimpleNamespace(cc="gcc", rust=SimpleNamespace(cargo="/opt/rust/bin/cargo")),
                                   halide=SimpleNamespace(autoscheduler="Adams2019"))
        self.calls: list[list[str]] = []
        self.present = set(VERSION_OUTPUTS)
        self.outputs = dict(VERSION_OUTPUTS)
        self.python_result = _proc(json.dumps(PYTHON_TOOLCHAIN_DOC) + "\n")

    def identity(self, cfg: object) -> dict:
        def run(cmd: list[str], **kwargs: object) -> ProcResult:
            self.calls.append(list(cmd))
            if "--python-toolchain" in cmd:
                return self.python_result
            self.assertEqual(cmd[1:], ["--version"])
            return _proc(self.outputs[cmd[0]] if cmd[0] in self.outputs else self.outputs[_basename(cmd[0])])

        with mock.patch.object(protocol, "run_isolated", side_effect=run), \
                mock.patch.object(protocol.shutil, "which", side_effect=fake_which(self.present)), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            identity = toolchain_identity(cfg)
        self.stderr = err.getvalue()
        return identity

    def test_all_tools_present(self) -> None:
        identity = self.identity(self.cfg)
        self.assertEqual(identity, {
            "cc": "gcc", "cc_version": "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0",
            "cargo": "cargo 1.92.0 (344c4567c 2025-10-21)", "rustc": "rustc 1.92.0 (ded5c06cf 2025-12-08)",
            "numba_version": "0.68.1", "halide_version": "22.1.0",
            "halide_autoscheduler": "Adams2019", "halide_autoscheduler_loadable": True,
            "lint_tools": {"ruff": "0.17.6", "pyflakes": "4.1.0", "cppcheck": "2.13.0", "clang-tidy": "18.1.3"},
            "errors": {},
        })
        # numba·halide·ruff·pyflakes 는 커널을 돌리는 인터프리터(PYTHON_EXE)에서, 설정의 오토스케줄러 이름으로 확인한다
        python_calls = [c for c in self.calls if "--python-toolchain" in c]
        self.assertEqual(python_calls, [[str(PYTHON_EXE), "-m", "harness.protocol", "--python-toolchain", "Adams2019"]])
        # cargo 는 설정의 명령(build.rust.cargo; 경로면 그대로), rustc 는 rust_tool 이 PATH 에서 찾은 실행 파일을 본다
        self.assertIn(["/opt/rust/bin/cargo", "--version"], self.calls)
        self.assertIn(["/bin/rustc", "--version"], self.calls)

    def test_cargo_and_rustc_are_probed_in_the_build_environment(self) -> None:
        """cargo·rustc 의 버전은 Rust 빌드와 같은 환경(``toolchain_env``: RUSTUP_TOOLCHAIN·RUSTUP_HOME·CARGO_HOME 포함)으로 본다 —
        아니면 지문의 rustc 가 빌드한 rustc 와 다르거나(고정한 툴체인), rustup 이 툴체인을 못 찾아 cargo 가 없다고 멈춘다."""
        from harness.build import rust as rust_build

        envs: dict[str, dict[str, str]] = {}

        def run(cmd: list[str], **kwargs: object) -> ProcResult:
            if "--python-toolchain" in cmd:
                return self.python_result
            envs[cmd[0]] = dict(kwargs["env"])  # type: ignore[arg-type]
            return _proc(self.outputs[cmd[0]] if cmd[0] in self.outputs else self.outputs[_basename(cmd[0])])

        user = {"RUSTUP_TOOLCHAIN": "1.92.0", "RUSTUP_HOME": "/opt/rustup", "CARGO_HOME": "/opt/cargo", "OMP_NUM_THREADS": "64"}
        with mock.patch.dict(os.environ, user), mock.patch.object(protocol, "IS_POSIX", True), \
                mock.patch.object(protocol, "run_isolated", side_effect=run), \
                mock.patch.object(protocol.shutil, "which", side_effect=fake_which(set(VERSION_OUTPUTS))), \
                contextlib.redirect_stderr(io.StringIO()):
            toolchain_identity(self.cfg)
            with mock.patch.object(rust_build, "IS_POSIX", True):
                self.assertEqual(rust_build.toolchain_env(), protocol.toolchain_env(), "빌드와 도구 점검이 같은 환경")
                self.assertEqual(rust_build._tool_env("/opt/rust/bin/cargo"), protocol.rust_tool_env("/opt/rust/bin/cargo"))
        rust_keys = {"RUSTUP_TOOLCHAIN": "1.92.0", "RUSTUP_HOME": "/opt/rustup", "CARGO_HOME": "/opt/cargo"}
        for tool in ("/opt/rust/bin/cargo", "/bin/rustc"):
            self.assertEqual({k: envs[tool].get(k) for k in protocol.TOOLCHAIN_ENV_KEYS}, rust_keys, tool)
            self.assertEqual(envs[tool]["OMP_NUM_THREADS"], "1", "사용자의 스레드 변수는 여전히 상속하지 않는다")
            # 빌드처럼 PATH 앞에 설정한 cargo 의 디렉터리가 붙는다
            self.assertEqual(envs[tool]["PATH"].split(os.pathsep)[0], os.path.dirname("/opt/rust/bin/cargo"), tool)
        self.assertEqual(envs["gcc"], protocol.launch_env_minimal(), "C 컴파일러는 최소 환경 그대로")
        self.assertFalse(set(protocol.TOOLCHAIN_ENV_KEYS) & set(envs["gcc"]))

    def test_absent_tools_are_none_and_are_not_launched(self) -> None:
        self.present = {"gcc"}
        identity = self.identity(self.cfg)
        self.assertEqual(identity["cc_version"], "gcc (Ubuntu 15.2.0-4ubuntu4) 15.2.0")
        self.assertIsNone(identity["cargo"])
        self.assertIsNone(identity["rustc"])
        self.assertEqual(identity["lint_tools"], {"ruff": "0.17.6", "pyflakes": "4.1.0", "cppcheck": None, "clang-tidy": None})
        self.assertEqual([c[0] for c in self.calls if c[1:] == ["--version"]], ["gcc"])

    def test_python_side_results_are_passed_through(self) -> None:
        doc = {"numba_version": None, "halide_version": "22.1.0", "halide_autoscheduler_loadable": False,
               "lint_tools": {"ruff": None, "pyflakes": "4.1.0"},
               "errors": {"numba": "ModuleNotFoundError: No module named 'numba'", "halide_autoscheduler": "no plugin file"}}
        self.python_result = _proc("some banner line\n" + json.dumps(doc) + "\n")
        identity = self.identity(self.cfg)
        self.assertIsNone(identity["numba_version"])
        self.assertIs(identity["halide_autoscheduler_loadable"], False)
        self.assertEqual(identity["errors"], doc["errors"])
        self.assertIsNone(identity["lint_tools"]["ruff"])
        self.assertEqual(identity["lint_tools"]["cppcheck"], "2.13.0")

    def test_failed_python_probe_means_nothing_python_side_is_usable(self) -> None:
        for result in (_proc("", rc=1, stderr="Traceback ...\nImportError: boom"), _proc("not json\n"), _proc("{broken\n")):
            self.python_result = result
            identity = self.identity(self.cfg)
            self.assertIsNone(identity["numba_version"])
            self.assertIsNone(identity["halide_version"])
            self.assertIsNone(identity["halide_autoscheduler_loadable"])
            self.assertEqual(identity["lint_tools"], {"ruff": None, "pyflakes": None, "cppcheck": "2.13.0", "clang-tidy": "18.1.3"})
            self.assertIn("python", identity["errors"])
            self.assertIn("python-toolchain", self.stderr)

    def test_launch_failure_of_the_interpreter_is_reported_not_raised(self) -> None:
        def run(cmd: list[str], **kwargs: object) -> ProcResult:
            raise FileNotFoundError("no such interpreter")

        with mock.patch.object(protocol, "run_isolated", side_effect=run), contextlib.redirect_stderr(io.StringIO()):
            doc = protocol.python_toolchain_subprocess("Mullapudi2016")
        self.assertIsNone(doc["numba_version"])
        self.assertIn("no such interpreter", doc["errors"]["python"])

    def test_version_output_in_an_unknown_format_keeps_the_first_line(self) -> None:
        self.outputs["cppcheck"] = "cppcheck development build\nmore\n"
        self.outputs["clang-tidy"] = "LLVM (http://llvm.org/):\n  LLVM version 17.0.6\n"
        identity = self.identity(self.cfg)
        self.assertEqual(identity["lint_tools"]["cppcheck"], "cppcheck development build")
        self.assertEqual(identity["lint_tools"]["clang-tidy"], "17.0.6")

    def test_failing_version_command_counts_as_absent(self) -> None:
        def run(cmd: list[str], **kwargs: object) -> ProcResult:
            if "--python-toolchain" in cmd:
                return self.python_result
            if _basename(cmd[0]) == "rustc":
                raise OSError("exec format error")
            return _proc("", rc=127) if cmd[0] == "gcc" else _proc(VERSION_OUTPUTS[cmd[0]])

        with mock.patch.object(protocol, "run_isolated", side_effect=run), \
                mock.patch.object(protocol.shutil, "which", side_effect=fake_which(set(VERSION_OUTPUTS))), \
                contextlib.redirect_stderr(io.StringIO()):
            identity = toolchain_identity(self.cfg)
        self.assertIsNone(identity["cc_version"])
        self.assertIsNone(identity["rustc"])
        self.assertEqual(identity["cargo"], "cargo 1.92.0 (344c4567c 2025-10-21)")

    def test_without_a_config_the_defaults_are_probed(self) -> None:
        self.present = {"gcc", "cargo"}
        self.outputs["cargo"] = "cargo 1.80.0\n"
        identity = self.identity(None)
        self.assertEqual((identity["cc"], identity["cargo"], identity["halide_autoscheduler"]), ("gcc", "cargo 1.80.0", "Li2018"))
        self.assertIn([str(PYTHON_EXE), "-m", "harness.protocol", "--python-toolchain", "Li2018"], self.calls)

    def test_old_style_config_without_the_optional_sections(self) -> None:
        identity = self.identity(SimpleNamespace(build=SimpleNamespace(cc="gcc")))
        self.assertEqual(identity["halide_autoscheduler"], "Li2018")
        self.assertIsNone(identity["cargo"], "기본 명령 cargo 는 이 테스트의 PATH 에 없다")


class RustToolTest(unittest.TestCase):
    """``rust_tool``: 설정의 경로는 그대로, 이름은 PATH → ``$CARGO_HOME/bin`` → ``~/.cargo/bin`` (decisions.md 2026-11-12: rustup 을
    PATH 를 바꾸지 않고 설치했다). ``rust_tool_env``: PATH 앞에 찾은 cargo 의 디렉터리."""

    EXE = ".exe" if os.name == "nt" else ""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="rust_tool_test_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.cargo_home = self.tmp / "cargo_home"
        self.home = self.tmp / "home"
        for d in (self.cargo_home / "bin", self.home / ".cargo" / "bin"):
            d.mkdir(parents=True)
        self.which: dict[str, str] = {}
        for patcher in (mock.patch.object(protocol.shutil, "which", side_effect=lambda name, *a, **k: self.which.get(name)),
                        mock.patch.object(protocol.Path, "home", return_value=self.home),
                        mock.patch.dict(os.environ, {"CARGO_HOME": str(self.cargo_home)})):
            patcher.start()
            self.addCleanup(patcher.stop)

    def tool(self, base: Path, name: str) -> str:
        path = base / "bin" / (name + self.EXE)
        path.write_bytes(b"")
        return str(path)

    def test_path_hit_comes_first(self) -> None:
        self.tool(self.cargo_home, "cargo")
        self.tool(self.home / ".cargo", "cargo")
        self.which["cargo"] = "/usr/local/bin/cargo"
        self.assertEqual(protocol.rust_tool("cargo"), "/usr/local/bin/cargo")
        self.assertEqual(protocol.rust_tool("cargo", "cargo"), "/usr/local/bin/cargo")

    def test_cargo_home_fallback(self) -> None:
        found = self.tool(self.cargo_home, "rustc")
        self.tool(self.home / ".cargo", "rustc")
        self.assertEqual(protocol.rust_tool("rustc"), found)

    def test_home_fallback(self) -> None:
        found = self.tool(self.home / ".cargo", "cargo")
        self.assertEqual(protocol.rust_tool("cargo"), found)          # $CARGO_HOME/bin 에는 없다
        os.environ.pop("CARGO_HOME")
        self.assertEqual(protocol.rust_tool("cargo"), found)          # $CARGO_HOME 이 없다

    def test_configured_path_is_kept(self) -> None:
        self.which["cargo"] = "/usr/local/bin/cargo"
        configured = str(self.tmp / "toolchains" / "cargo")             # 없는 파일이어도 그대로(실행이 실패해 알린다)
        self.assertEqual(protocol.rust_tool("cargo", configured), configured)
        self.assertEqual(protocol.rust_tool("cargo", "/opt/rust/bin/cargo"), "/opt/rust/bin/cargo")

    def test_configured_name_is_searched_like_the_default(self) -> None:
        self.which["cargo-nightly"] = "/usr/bin/cargo-nightly"
        self.assertEqual(protocol.rust_tool("cargo", "cargo-nightly"), "/usr/bin/cargo-nightly")

    def test_not_found_returns_the_name(self) -> None:
        self.assertEqual(protocol.rust_tool("cargo"), "cargo")
        self.assertEqual(protocol.rust_tool("cargo", "cargo-nightly"), "cargo-nightly")

    def test_tool_env_prepends_the_cargo_directory(self) -> None:
        cargo = self.tool(self.cargo_home, "cargo")
        with mock.patch.object(protocol, "IS_POSIX", False):
            env = protocol.rust_tool_env(cargo)
            self.assertEqual(env["PATH"].split(os.pathsep), [str(self.cargo_home / "bin"), *os.environ["PATH"].split(os.pathsep)])
            # 맨 이름(찾지 못한 경우)이면 PATH 를 바꾸지 않는다
            self.assertEqual(protocol.rust_tool_env("cargo"), protocol.toolchain_env())
            # 인자가 없으면 rust_tool("cargo") 의 결과
            self.assertEqual(protocol.rust_tool_env()["PATH"].split(os.pathsep)[0], str(self.cargo_home / "bin"))


class RustFlagsTest(unittest.TestCase):
    """``rust_flags``: 설정의 ``rustflags`` + Linux 에서만 ``rustflags_linux`` (decisions.md 2026-11-12)."""

    CFG = SimpleNamespace(rustflags=["-C", "target-cpu=native"], rustflags_linux=["-C", "link-arg=-Wl,--no-undefined"])

    def test_linux_true_appends_the_linux_flags(self) -> None:
        self.assertEqual(protocol.rust_flags(self.CFG, linux=True), ["-C", "target-cpu=native", "-C", "link-arg=-Wl,--no-undefined"])

    def test_linux_false_does_not(self) -> None:
        self.assertEqual(protocol.rust_flags(self.CFG, linux=False), ["-C", "target-cpu=native"])

    def test_default_is_this_platform(self) -> None:
        with mock.patch.object(protocol.sys, "platform", "linux"):
            self.assertEqual(protocol.rust_flags(self.CFG), protocol.rust_flags(self.CFG, linux=True))
        for name in ("win32", "darwin"):
            with mock.patch.object(protocol.sys, "platform", name):
                self.assertEqual(protocol.rust_flags(self.CFG), ["-C", "target-cpu=native"])

    def test_config_objects(self) -> None:
        self.assertEqual(protocol.rust_flags(config_option(None, "build.rust"), linux=True),
                         ["-C", "target-cpu=native", "-C", "link-arg=-Wl,--no-undefined"])
        self.assertEqual(protocol.rust_flags(config_module.RustCfg(), linux=False), ["-C", "target-cpu=native"])
        self.assertEqual(protocol.rust_flags(SimpleNamespace(), linux=True), [])


@requires_modules("numba", "halide")
class PythonToolchainLiveTest(unittest.TestCase):
    """실제 ``PYTHON_EXE`` 서브프로세스: numba·halide 임포트와 오토스케줄러 플러그인 로드(워커와 같은 로더)."""

    def test_default_autoscheduler_is_loadable(self) -> None:
        default = config_option(None, "halide.autoscheduler")
        self.assertEqual(default, "Li2018")
        doc = protocol.python_toolchain_subprocess(default)
        self.assertEqual(doc["errors"], {})
        self.assertRegex(doc["numba_version"], r"^\d+\.\d+")
        self.assertRegex(doc["halide_version"], r"^\d+\.\d+")
        self.assertIs(doc["halide_autoscheduler_loadable"], True)
        self.assertEqual(set(doc["lint_tools"]), {"ruff", "pyflakes"})

    def test_every_selectable_autoscheduler_is_loadable(self) -> None:
        """기본 Li2018 과 민감도 분석용 Adams2019·Mullapudi2016 (2026-11-12 결정) 모두 halide 22.1 휠의 플러그인으로 로드된다
        (이 Windows 머신에서 셋 다 확인했다)."""
        for name in ("Li2018", "Adams2019", "Mullapudi2016"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                self.assertIsNone(protocol.halide_autoscheduler_error(name))

    def test_unknown_autoscheduler_is_not_loadable_and_says_why(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            reason = protocol.halide_autoscheduler_error("NoSuchScheduler2099")
        self.assertIn("NoSuchScheduler2099", reason)
        self.assertEqual(err.getvalue(), "", "로더가 stderr 에 적는 사유는 가로채 돌려준다")
        self.assertIsNone(protocol.halide_autoscheduler_error("Li2018"))


class RuntimeProbeTest(unittest.TestCase):
    """``runtime_probe_subprocess`` (run.py 의 도구 점검): 명령의 꼴과 결과 읽기(가짜 run_isolated), 그리고 실제 서브프로세스."""

    def probe(self, result: ProcResult, kind: str = "halide") -> tuple[str | None, list[str], dict]:
        calls: list[tuple[list[str], dict]] = []

        def run(cmd: list[str], **kwargs: object) -> ProcResult:
            calls.append((list(cmd), kwargs))
            return result

        with mock.patch.object(protocol, "run_isolated", side_effect=run):
            error = protocol.runtime_probe_subprocess(kind, env={"HL_NUM_THREADS": "4"}, prefix=["taskset", "-c", "0-3"],
                                                      autoscheduler="Mullapudi2016", params={"last_level_cache_size": 1048576},
                                                      threads=4)
        ((cmd, kwargs),) = calls
        return error, cmd, kwargs

    def test_command_and_environment(self) -> None:
        error, cmd, kwargs = self.probe(_proc('{"error": null}\n'))
        self.assertIsNone(error)
        self.assertEqual(cmd, ["taskset", "-c", "0-3", str(PYTHON_EXE), "-m", "harness.protocol", "--probe-runtime", "halide",
                               "--halide-autoscheduler", "Mullapudi2016", "--halide-params", '{"last_level_cache_size": 1048576}',
                               "--threads", "4"])
        self.assertEqual(kwargs["env"], {"HL_NUM_THREADS": "4"})
        _, cmd, _ = self.probe(_proc('{"error": null}\n'), kind="numba")
        self.assertEqual(cmd[-2:], ["--probe-runtime", "numba"])

    def test_reported_error_and_a_process_that_ends_without_a_result(self) -> None:
        self.assertEqual(self.probe(_proc('warning line\n{"error": "ValueError: bad"}\n'))[0], "ValueError: bad")
        aborted = ProcResult(rc=-6, timed_out=False, elapsed_s=0.1, stdout_tail="", stderr_tail="Unknown key: beam_sizee\n",
                             signal_name="SIGABRT")
        error = self.probe(aborted)[0]
        self.assertIn("SIGABRT", error)
        self.assertIn("Unknown key: beam_sizee", error)
        self.assertIn("rc=1", self.probe(_proc("", rc=1, stderr="Traceback"))[0])


@requires_modules("halide")
class HalideRuntimeProbeLiveTest(unittest.TestCase):
    """실제 서브프로세스: 워커와 같은 준비(make_loader)로 오토스케줄러를 설정의 매개변수로 건다. 모르는 키는 Halide 22.1 에서
    프로세스 중단이다 — 그것도 사유가 된다(도구 점검의 exit 7)."""

    def test_good_and_bad_parameters(self) -> None:
        env = protocol.launch_env_minimal()
        for name in ("Li2018", "Mullapudi2016"):  # 기본값과 설정으로 고르는 것 하나
            with self.subTest(name=name):
                self.assertIsNone(protocol.runtime_probe_subprocess("halide", env=env, autoscheduler=name, params={}, threads=2))
        error = protocol.runtime_probe_subprocess("halide", env=env, autoscheduler="Mullapudi2016",
                                                  params={"no_such_parameter_2099": "1"}, threads=2)
        self.assertIsNotNone(error)


@requires_modules("numba")
class NumbaRuntimeProbeLiveTest(unittest.TestCase):
    def test_threading_layer(self) -> None:
        env = protocol.launch_env_minimal()
        self.assertIsNone(protocol.runtime_probe_subprocess("numba", env={**env, "NUMBA_THREADING_LAYER": "workqueue"}))
        error = protocol.runtime_probe_subprocess("numba", env={**env, "NUMBA_THREADING_LAYER": "no_such_layer"})
        self.assertIsNotNone(error)


class EntryPointStreamsTest(unittest.TestCase):
    """회귀: ``python -m harness.protocol --help`` 가 파이프에서 로캘 코드페이지로 찍혀(Windows) 한글이 깨졌다."""

    def test_help_is_utf8_on_a_pipe(self) -> None:
        r = subprocess.run([str(PYTHON_EXE), "-m", "harness.protocol", "--help"], capture_output=True, timeout=120,
                           cwd=str(protocol.ROOT), env={**os.environ, "PYTHONIOENCODING": ""})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("측정 규약", r.stdout.decode("utf-8"))


class ProbeMachineTest(unittest.TestCase):
    """``probe_machine`` 은 툴체인 정체를 machine.yaml 키로 옮긴다(서브프로세스를 띄우는 헬퍼는 대역으로 바꾼다)."""

    def probe(self, cfg: object, toolchain: dict) -> tuple[dict, mock.Mock]:
        identity = {"python_version": "3.14.4", "numpy_version": "2.3.5", "numpy_blas": {"name": "scipy-openblas"},
                    "sdk_versions": {"anthropic": None, "openai": None}}
        with mock.patch.object(protocol, "toolchain_identity", return_value=toolchain) as probe, \
                mock.patch.object(protocol, "env_identity_subprocess", return_value=identity), \
                mock.patch.object(protocol, "compiler_version", return_value=None), \
                mock.patch.object(protocol, "resolve_march", return_value="-march=x"), \
                mock.patch.object(protocol, "_openmp_works", return_value=True), \
                mock.patch.object(protocol, "_sudo_nopasswd", return_value=False):
            return protocol.probe_machine(cfg), probe

    def test_toolchain_keys(self) -> None:
        cfg = load_config()
        machine, probe = self.probe(cfg, TOOLCHAIN)
        probe.assert_called_once_with(cfg)
        self.assertEqual(machine["compilers"]["gcc"], TOOLCHAIN["cc_version"])
        self.assertTrue(machine["compilers"]["openmp_gcc"])
        self.assertIsNone(machine["compilers"]["clang"])
        for key in ("cargo", "rustc", "numba_version", "halide_version", "halide_autoscheduler", "halide_autoscheduler_loadable",
                    "lint_tools"):
            self.assertEqual(machine[key], TOOLCHAIN[key], key)
        self.assertEqual(machine["march_resolved"], "-march=x")
        self.assertNotIn("errors", machine)

    def test_other_compiler_and_no_config(self) -> None:
        machine, probe = self.probe(None, {**TOOLCHAIN, "cc": "tcc", "cc_version": "tcc 0.9.27"})
        probe.assert_called_once_with(None)
        self.assertEqual(machine["compilers"]["tcc"], "tcc 0.9.27")
        self.assertIsNone(machine["compilers"]["gcc"])
        self.assertFalse(machine["compilers"]["openmp_gcc"])


class CallCpusTest(unittest.TestCase):
    """bench 에서 코드를 호출하는 동안의 주 스레드 CPU 집합(2026-11-11)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_openmp_c_is_left_to_libgomp(self) -> None:
        from harness.protocol import call_cpus
        self.assertIsNone(call_cpus(self.cfg, backend="c", openmp=True, threads=self.cfg.measurement.threads))

    def test_python_gets_all_pinned_cores(self) -> None:
        from harness.protocol import call_cpus
        self.assertEqual(call_cpus(self.cfg, backend="python", openmp=False, threads=self.cfg.measurement.threads),
                         self.cfg.measurement.cores)

    def test_sequential_c_gets_first_pinned_cpu(self) -> None:
        """순차 C 는 고정 코어의 첫 CPU(= OpenMP 주 스레드의 place) 하나에서 돈다: libgomp 작업 스레드의 코어와 겹치지 않는다."""
        from harness.protocol import call_cpus
        first = str(parse_cpulist(self.cfg.measurement.cores)[0])
        self.assertEqual(call_cpus(self.cfg, backend="c", openmp=False, threads=1), first)
        self.assertEqual(call_cpus(self.cfg, backend="python", openmp=False, threads=1), first)

    def test_call_cpus_stay_inside_the_bench_taskset(self) -> None:
        """measurement.threads == 1 이면 bench 워커는 single_core 에 고정된다: 호출 CPU 도 그 안에서 골라야 한다."""
        import dataclasses

        from harness.protocol import bench_cpus, call_cpus, launch_prefix
        cfg1 = dataclasses.replace(self.cfg, measurement=dataclasses.replace(self.cfg.measurement, threads=1))
        self.assertEqual(bench_cpus(cfg1), cfg1.measurement.single_core)
        self.assertEqual(bench_cpus(self.cfg), self.cfg.measurement.cores)
        for cfg in (self.cfg, cfg1):
            allowed = set(parse_cpulist(bench_cpus(cfg)))
            prefix = launch_prefix(cfg, cfg.measurement.threads)
            if prefix:
                self.assertEqual(set(parse_cpulist(prefix[-1])), allowed)
            for backend, openmp, threads in (("python", False, cfg.measurement.threads), ("c", False, 1), ("python", False, 1)):
                got = call_cpus(cfg, backend=backend, openmp=openmp, threads=threads)
                self.assertIsNotNone(got)
                self.assertLessEqual(set(parse_cpulist(got)), allowed, (backend, threads, got))

    def test_openmp_runtime_is_left_alone_whatever_the_backend(self) -> None:
        """2026-11-23 일반화(docs/decisions.md 2026-11-23; 호출 친화도 규칙은 2026-11-11·2026-11-12): 판정은 ``openmp`` (= ``LanguageCfg.omp_runtime``)만 본다 — numba(스레딩 계층 omp)도
        OpenMP C 처럼 libgomp 의 고정을 그대로 둔다. ``backend`` 는 판정에 쓰지 않는다."""
        from harness.protocol import call_cpus
        for backend in ("numba", "c", "python", "halide", "rust"):
            for threads in (1, self.cfg.measurement.threads):
                with self.subTest(backend=backend, threads=threads):
                    self.assertIsNone(call_cpus(self.cfg, backend=backend, openmp=True, threads=threads))

    def test_halide_and_rust_follow_the_non_openmp_rule(self) -> None:
        """Halide 스레드 풀·rayon 은 OpenMP 가 아니다: threads > 1 이면 고정 코어 전체, threads == 1 이면 첫 CPU."""
        import dataclasses

        from harness.protocol import bench_cpus, call_cpus
        cfg1 = dataclasses.replace(self.cfg, measurement=dataclasses.replace(self.cfg.measurement, threads=1))
        for cfg in (self.cfg, cfg1):
            pinned = bench_cpus(cfg)
            first = str(parse_cpulist(pinned)[0])
            for backend in ("halide", "rust"):
                with self.subTest(backend=backend, measurement_threads=cfg.measurement.threads):
                    self.assertEqual(call_cpus(cfg, backend=backend, openmp=False, threads=4), pinned)
                    self.assertEqual(call_cpus(cfg, backend=backend, openmp=False, threads=1), first)

    def test_policy_version_is_unchanged(self) -> None:
        """규칙의 일반화는 통합 전 하네스가 만들 수 있던 조합의 결과를 바꾸지 않으므로 판(版)을 올리지 않았다(env_hash 유지)."""
        self.assertEqual(protocol.CALL_AFFINITY_POLICY, "2026-11-11")

    def test_worker_resolves_cpu_arguments(self) -> None:
        from harness.worker import resolve_call_cpus
        self.assertEqual(resolve_call_cpus("auto", "python", [0, 2, 4, 6]), (0, 2, 4, 6))
        self.assertIsNone(resolve_call_cpus("auto", "c", [0, 2, 4, 6]))
        self.assertIsNone(resolve_call_cpus("none", "python", [0, 2]))
        self.assertEqual(resolve_call_cpus("2", "c", [0, 2, 4, 6]), (2,))
        self.assertEqual(resolve_call_cpus("0,2,4-6", "python", []), (0, 2, 4, 5, 6))
        with self.assertRaises(ValueError):
            resolve_call_cpus(",", "python", [0])


class LiveChecksTest(unittest.TestCase):
    """실제 sysfs/proc 를 읽는 검사(이 머신에서 1초)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config()

    def test_sibling_cpus_excludes_pinned(self) -> None:
        pinned = parse_cpulist(self.cfg.measurement.cores)
        sibs = sibling_cpus(pinned)
        self.assertFalse(set(sibs) & set(pinned))

    def test_check_dynamic_shape(self) -> None:
        st = check_dynamic(self.cfg, sample_s=0.3)
        self.assertIsInstance(st, DynamicStatus)
        self.assertGreaterEqual(st.load_1m, 0.0)
        self.assertTrue(0.0 <= st.pinned_busy_pct <= 100.0)
        self.assertTrue(0.0 <= st.sibling_busy_pct <= 100.0)
        self.assertEqual(st.ok, flags_ok(st.flags))
        self.assertEqual(st.busy_ok, st.pinned_busy_pct <= self.cfg.measurement.max_pinned_busy_pct
                         and st.sibling_busy_pct <= self.cfg.measurement.max_pinned_busy_pct)
        self.assertRegex(st.checked_at, r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}$")
        if not st.governor_ok and self.cfg.measurement.require_performance_governor:
            self.assertIn(f"governor={st.governor}", st.flags)

    def test_environment_fingerprint_is_deterministic(self) -> None:
        static = StaticStatus(
            governor="powersave", governor_ok=False, epp="performance", epp_ok=True, no_turbo=0, taskset_available=True,
            cores="0,2,4,6", single_core="2", threads=4,
            blas={"name": "scipy-openblas", "version": "0.3.30", "config": "OpenBLAS 0.3.30", "loaded_lib": "/x.so"},
            blas_ok=True, flags=["governor=powersave", "turbo=on"], ok=False, checked_at="2026-11-04T10:00:00+09:00",
        )
        machine = {
            "cpu_model": "cpu", "kernel": "k", "glibc": "2.43", "compilers": {"gcc": "gcc 15"}, "march_resolved": "-march=x",
            "python_version": "3.14.4", "numpy_version": "2.3.5", "sdk_versions": {"anthropic": "2.4.0", "openai": "3.44.0"},
        }
        fp1, h1 = environment_fingerprint(self.cfg, machine, static)
        fp2, h2 = environment_fingerprint(self.cfg, machine, static)
        self.assertEqual(h1, h2)
        self.assertRegex(h1, r"^[0-9a-f]{16}$")
        self.assertEqual(fp1["cores"], "0,2,4,6")
        self.assertIn("-O3", fp1["flags"])
        self.assertIn("-fopenmp", fp1["flags"])
        self.assertEqual(fp1["numpy_blas"]["name"], "scipy-openblas")
        self.assertEqual(fp1["governor"], "powersave")
        # 거버너가 바뀌면 지문도 바뀐다
        static2 = StaticStatus(**{**static.__dict__, "governor": "performance"})
        _, h3 = environment_fingerprint(self.cfg, machine, static2)
        self.assertNotEqual(h1, h3)

    def test_environment_fingerprint_keeps_exactly_up_keys(self) -> None:
        """통합(docs/decisions.md 2026-11-23): 지문은 통합 전(6f02d16)의 사전 그대로다 — 툴체인 정체(rustc·numba·halide·린트 도구)는 넣지 않고
        도구를 띄우지도 않는다. machine.yaml 에 새 프로브 키가 있어도 지문은 같다."""
        static = StaticStatus(
            governor="performance", governor_ok=True, epp="performance", epp_ok=True, no_turbo=1, taskset_available=True,
            cores="0,2,4,6", single_core="2", threads=4, blas={"name": "scipy-openblas", "version": "0.3.30", "config": ""},
            blas_ok=True, flags=[], ok=True, checked_at="2026-11-21T10:00:00+09:00",
        )
        machine = {"cpu_model": "cpu", "kernel": "k", "glibc": "2.43", "compilers": {"gcc": "gcc 15"}, "march_resolved": "-march=x",
                   "python_version": "3.14.4", "numpy_version": "2.3.5", "sdk_versions": {"anthropic": None, "openai": None}}
        with mock.patch.object(protocol, "toolchain_identity", side_effect=AssertionError("지문은 툴체인을 보지 않는다")):
            fp, digest = environment_fingerprint(self.cfg, machine, static)
            richer = {**machine, **{k: TOOLCHAIN[k] for k in ("cargo", "rustc", "numba_version", "halide_version", "lint_tools")}}
            _, digest2 = environment_fingerprint(self.cfg, richer, static)
        self.assertEqual(set(fp), UP_FINGERPRINT_KEYS)
        self.assertEqual(digest2, digest)
        self.assertEqual(fp["call_affinity_policy"], protocol.CALL_AFFINITY_POLICY)
        self.assertIs(fp["bench_loop_fsync"], bool(self.cfg.measurement.bench_loop_fsync))


if __name__ == "__main__":
    unittest.main()
