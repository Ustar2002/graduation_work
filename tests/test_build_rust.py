"""build/rust.py 테스트 (설계 v3 §6).

- 순수 함수: cargo 로그 → ``error_detail``·경고 수, cargo 를 띄우는 환경.
- 템플릿: ``Cargo.toml`` 이 계약대로이고 ``Cargo.lock`` 이 고정한 rayon 버전을 담는다.
- ``FakeCargoTest``: cargo·rustc 를 흉내 내는 ``run_isolated`` 대역으로 ``build_rust`` 의 흐름(파일, 명령, 환경, 실패의 분류, 심벌
  검사)을 본다. cargo 가 없어도, 어느 플랫폼에서나 돈다. ``TargetLockTest``·``ProbeToolchainTest``: target 디렉터리의 프로세스 간
  잠금과 run 시작 때의 툴체인 점검(``probe_toolchain``).
- ``RealCargoTest``: 진짜 cargo 로 끝에서 끝까지 — 오프라인 빌드 → ctypes 로드 → 내적 계산, 컴파일 오류의 rustc 메시지,
  패닉 = abort, 샘플끼리 나눠 쓰는 target 디렉터리. cargo 와 의존성 캐시(``cargo fetch --locked --manifest-path
  harness/build/rust_template/Cargo.toml`` 1회)가 있어야 한다. 테스트용 커널은 서브프로세스에서만 로드한다.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

import harness.build.rust as rust_mod
from harness import sigs
from harness.build.c import ToolchainMissing
from harness.build.rust import (
    ARTIFACT_NAME,
    CARGO_CDYLIB,
    FETCH_HINT,
    LOCK_NAME,
    PROBE_DIR_NAME,
    TEMPLATE_DIR,
    build_rust,
    cargo_error_detail,
    count_rust_warnings,
    lock_wait_s,
    probe_toolchain,
    target_lock,
    toolchain_env,
)
from harness.extract import extract
from harness.paths import ROOT
from harness.procutil import ProcResult
from harness.protocol import launch_env_minimal
from harness.taskspec import load_taskspec
from tests._support import requires_tool

DOT_SPEC = load_taskspec(ROOT / "tasks" / "dot" / "signatures.yaml")

# rayon 으로 조각마다 f64 로 누적하는 내적. 음수 n 은 하네스가 주지 않는 값이고, 주면 패닉한다(패닉 = abort 확인에 쓴다).
RS_DOT = '''use rayon::prelude::*;
use std::slice;

const CHUNK: usize = 4096;

/// Dot product: every chunk is accumulated in f64 and the partial sums are added in f64.
#[no_mangle]
pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
    let len = usize::try_from(n).expect("n must not be negative");
    let xs = slice::from_raw_parts(x, len);
    let ys = slice::from_raw_parts(y, len);
    let total: f64 = xs
        .par_chunks(CHUNK)
        .zip(ys.par_chunks(CHUNK))
        .map(|(a, b)| a.iter().zip(b).map(|(p, q)| f64::from(*p) * f64::from(*q)).sum::<f64>())
        .sum();
    *out = total as f32;
}
'''

# keyword 모드: 하네스가 주는 src/kernel_args.rs 의 구조체를 쓴다
RS_DOT_KEYWORD = '''mod kernel_args;
use kernel_args::DotArgs;
use std::slice;

#[no_mangle]
pub unsafe extern "C" fn dot(args: *const DotArgs) {
    let a = &*args;
    let len = a.n as usize;
    let xs = slice::from_raw_parts(a.x, len);
    let ys = slice::from_raw_parts(a.y, len);
    let mut total = 0.0f64;
    for (p, q) in xs.iter().zip(ys) {
        total += f64::from(*p) * f64::from(*q);
    }
    *a.out = total as f32;
}
'''

RS_HEAD = '#[no_mangle]\npub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) '
RS_BROKEN = RS_HEAD + "{\n    *out = undefined_name;\n}\n"
RS_CONSTANT = RS_HEAD + "{\n    let _ = (x, y, n);\n    *out = 42.0;\n}\n"
RS_NOT_EXPORTED = RS_CONSTANT.replace("#[no_mangle]\n", "")


# ----------------------------------------------------------------------------- 순수 함수

COMPILE_ERROR_LOG = """   Compiling kernel v0.1.0 (/work/results/built/run/dot/rust/tag/sample_0)
error[E0425]: cannot find value `undefined_name` in this scope
 --> src/lib.rs:3:12
  |
3 |     *out = undefined_name;
  |            ^^^^^^^^^^^^^^ not found in this scope

For more information about this error, try `rustc --explain E0425`.
error: could not compile `kernel` (lib) due to 1 previous error
"""

MANY_ERRORS_LOG = """   Compiling kernel v0.1.0 (C:\\work\\sample_0)
warning: unused variable: `n`
 --> src\\lib.rs:2:76
  |
2 | pub unsafe extern "C" fn dot(x: *const f32, y: *const f32, out: *mut f32, n: i32) {
  |                                                                            ^ help: if this is intentional, prefix it with an underscore: `_n`
  |
  = note: `#[warn(unused_variables)]` on by default

error[E0433]: failed to resolve: use of unresolved module or unlinked crate `ndarray`
 --> src\\lib.rs:1:5
  |
1 | use ndarray::Array1;
  |     ^^^^^^^ use of unresolved module or unlinked crate `ndarray`

error: expected one of `!` or `::`, found `fn`
 --> src\\lib.rs:9:5
  |
9 | pub fn helper() {}
  |     ^^ expected one of `!` or `::`

error[E0308]: mismatched types
  --> src\\lib.rs:12:12
   |
12 |     *out = 1.0f64;
   |            ^^^^^^ expected `f32`, found `f64`

error[E0599]: no method named `par_iter` found for raw pointer `*const f32` in the current scope
  --> src\\lib.rs:13:7

warning: `kernel` (lib) generated 1 warning
error: aborting due to 4 previous errors; 1 warning emitted
error: could not compile `kernel` (lib) due to 4 previous errors; 1 warning emitted
"""

RUSTC_KILLED_LOG = """   Compiling kernel v0.1.0 (/work/sample_0)
error: could not compile `kernel` (lib)

Caused by:
  process didn't exit successfully: `rustc --crate-name kernel --edition=2021 src/lib.rs` (signal: 9, SIGKILL: kill)
"""

WARNINGS_LOG = """   Compiling kernel v0.1.0 (/work/sample_0)
warning: unused variable: `n`
 --> src/lib.rs:2:76

warning[E0133]: call to unsafe function is unsafe and requires unsafe block
 --> src/lib.rs:4:14

warning: `kernel` (lib) generated 2 warnings
    Finished `release` profile [optimized] target(s) in 3.21s
"""


class LogParsingTest(unittest.TestCase):
    def test_first_error_with_its_location(self) -> None:
        self.assertEqual(cargo_error_detail(COMPILE_ERROR_LOG),
                         "error[E0425]: cannot find value `undefined_name` in this scope --> src/lib.rs:3:12")

    def test_first_three_errors_without_the_summary_lines(self) -> None:
        self.assertEqual(cargo_error_detail(MANY_ERRORS_LOG), " | ".join([
            "error[E0433]: failed to resolve: use of unresolved module or unlinked crate `ndarray` --> src/lib.rs:1:5",
            "error: expected one of `!` or `::`, found `fn` --> src/lib.rs:9:5",
            "error[E0308]: mismatched types --> src/lib.rs:12:12",
        ]))
        self.assertEqual(cargo_error_detail(MANY_ERRORS_LOG, limit=1),
                         "error[E0433]: failed to resolve: use of unresolved module or unlinked crate `ndarray` --> src/lib.rs:1:5")

    def test_error_without_a_location_line(self) -> None:
        log = "error: linking with `cc` failed: exit status: 1\n  |\n  = note: /usr/bin/ld: cannot find -lfoo\n"
        self.assertEqual(cargo_error_detail(log), "error: linking with `cc` failed: exit status: 1")

    def test_tail_of_the_log_when_rustc_reported_no_error(self) -> None:
        self.assertEqual(cargo_error_detail(RUSTC_KILLED_LOG), (
            "error: could not compile `kernel` (lib) | Caused by: | process didn't exit successfully: "
            "`rustc --crate-name kernel --edition=2021 src/lib.rs` (signal: 9, SIGKILL: kill)"))
        self.assertEqual(cargo_error_detail(""), "")

    def test_warning_count_ignores_the_cargo_summary(self) -> None:
        self.assertEqual(count_rust_warnings(WARNINGS_LOG), 2)
        self.assertEqual(count_rust_warnings(MANY_ERRORS_LOG), 1)
        self.assertEqual(count_rust_warnings(COMPILE_ERROR_LOG), 0)


class ToolchainEnvTest(unittest.TestCase):
    """POSIX: 화이트리스트 환경 + 툴체인 위치 변수만. 그 밖(개발용 Windows): 현재 환경 그대로."""

    VARS = {"CARGO_HOME": "/opt/cargo-home", "RUSTUP_TOOLCHAIN": "1.92.0", "RUSTFLAGS": "-C opt-level=0",
            "CARGO_BUILD_TARGET": "wasm32-unknown-unknown", "OMP_NUM_THREADS": "13"}

    def test_posix_starts_from_the_minimal_environment(self) -> None:
        with mock.patch.dict(os.environ, self.VARS), mock.patch.object(rust_mod, "IS_POSIX", True):
            os.environ.pop("RUSTUP_HOME", None)
            env = toolchain_env()
            self.assertEqual(env, {**launch_env_minimal(), "CARGO_HOME": "/opt/cargo-home", "RUSTUP_TOOLCHAIN": "1.92.0"})
            self.assertNotIn("RUSTUP_HOME", env)
            os.environ["RUSTUP_HOME"] = "/opt/rustup-home"
            self.assertEqual(toolchain_env()["RUSTUP_HOME"], "/opt/rustup-home")

    def test_other_platforms_inherit_the_current_environment(self) -> None:
        with mock.patch.dict(os.environ, self.VARS), mock.patch.object(rust_mod, "IS_POSIX", False):
            self.assertEqual(toolchain_env(), dict(os.environ))


class TemplateTest(unittest.TestCase):
    def test_manifest_is_the_contract(self) -> None:
        manifest = tomllib.loads((TEMPLATE_DIR / "Cargo.toml").read_text(encoding="utf-8"))
        self.assertEqual(manifest["package"], {"name": "kernel", "version": "0.1.0", "edition": "2021"})
        self.assertEqual(manifest["package"]["name"], rust_mod.PACKAGE)
        self.assertEqual(manifest["lib"], {"crate-type": ["cdylib"]})
        self.assertEqual(list(manifest["dependencies"]), ["rayon"])
        self.assertRegex(manifest["dependencies"]["rayon"], r"^=1\.\d+\.\d+$")     # 정확한 버전 고정
        self.assertEqual(manifest["profile"], {"release": {"opt-level": 3, "codegen-units": 1, "lto": "fat", "panic": "abort",
                                                           "debug": False}})
        self.assertEqual(set(manifest), {"package", "lib", "dependencies", "profile"})

    def test_lock_file_pins_the_same_rayon(self) -> None:
        manifest = tomllib.loads((TEMPLATE_DIR / "Cargo.toml").read_text(encoding="utf-8"))
        lock = tomllib.loads((TEMPLATE_DIR / "Cargo.lock").read_text(encoding="utf-8"))
        packages = {p["name"]: p for p in lock["package"]}
        self.assertEqual("=" + packages["rayon"]["version"], manifest["dependencies"]["rayon"])
        self.assertEqual(packages["kernel"], {"name": "kernel", "version": "0.1.0", "dependencies": ["rayon"]})
        for name, package in packages.items():
            if name != "kernel":
                with self.subTest(package=name):
                    self.assertEqual(package["source"], "registry+https://github.com/rust-lang/crates.io-index")
                    self.assertRegex(package["checksum"], r"^[0-9a-f]{64}$")

    def test_template_files(self) -> None:
        self.assertEqual(rust_mod.TEMPLATE_FILES, ("Cargo.toml", "Cargo.lock"))
        # src/lib.rs 는 `cargo fetch --manifest-path` 가 매니페스트를 읽게 하는 자리표시다: 샘플 디렉터리로 복사되지 않고
        # probe_toolchain 만 빌드한다(알려진 정상 코드: 주석뿐이다)
        self.assertTrue((TEMPLATE_DIR / "src" / "lib.rs").is_file())
        lines = (TEMPLATE_DIR / "src" / "lib.rs").read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(ln.startswith("//") or not ln.strip() for ln in lines), lines)
        self.assertFalse((TEMPLATE_DIR / "target").exists(), "템플릿 디렉터리에서 빌드하지 않는다")


# ----------------------------------------------------------------------------- cargo 대역

CARGO = sys.executable  # shutil.which 가 찾는 실행 파일이면 된다: 실행은 아래 대역이 가로챈다
RUSTC_VERSION = "rustc 1.92.0 (ded5c06cf 2025-12-08)"
ELF_LIBRARY = b"\x7fELF\x02\x01\x01 fake cdylib"
NM_DEFINED = "0000000000012340 T dot\n0000000000012400 T rust_eh_personality\n"
KERNEL_ARGS_RS = sigs.signature_bundle(DOT_SPEC, "rust", "rust", "keyword")["args_struct"]

FINISHED = "    Finished `release` profile [optimized] target(s) in 3.21s\n"
COMPILING_KERNEL = "   Compiling kernel v0.1.0 (/work/sample_0)\n"
# 소스에 든 표식 → cargo build 가 한 일: (build.log 내용, rc, 시간 초과, 라이브러리를 만들었는가)
SCENARIOS: dict[str, tuple[str, int, bool, bool]] = {
    "undefined_name": (COMPILE_ERROR_LOG, 101, False, False),      # RS_BROKEN
    "WARNINGS": (WARNINGS_LOG, 0, False, True),
    "DEPENDENCY_CACHE_MISSING": ("error: no matching package named `rayon` found\nlocation searched: crates.io index\n"
                                 "required by package `kernel v0.1.0 (/work/sample_0)`\n", 101, False, False),
    "LOCK_FILE_OUTDATED": ("error: the lock file /work/sample_0/Cargo.lock needs to be updated but --locked was passed to prevent this\n",
                           101, False, False),
    "DEPENDENCY_FAILS": ("   Compiling rayon-core v1.13.0\nerror[E0463]: can't find crate for `std`\n"
                         "error: could not compile `rayon-core` (lib) due to 1 previous error\n", 101, False, False),
    "LINKER_MISSING": (COMPILING_KERNEL + "error: linker `cc` not found\n  |\n  = note: No such file or directory (os error 2)\n\n"
                       "error: could not compile `kernel` (lib) due to 1 previous error\n", 101, False, False),
    "CARGO_KILLED": (COMPILING_KERNEL, -9, False, False),
    "SLOW_KERNEL": ("   Compiling rayon v1.12.0\n" + COMPILING_KERNEL, -9, True, False),
    "SLOW_DEPENDENCIES": ("   Compiling crossbeam-utils v0.8.23\n   Compiling rayon-core v1.13.0\n", -9, True, False),
    "NO_ARTIFACT": (COMPILING_KERNEL + FINISHED, 0, False, False),
    # 링커가 있지만 링크가 실패한다(환경 문제): build_rust 는 커널 크레이트의 실패로 보지만 probe_toolchain 은 환경으로 본다
    "LINK_FAILS": (COMPILING_KERNEL + "error: linking with `cc` failed: exit status: 1\n  |\n  = note: /usr/bin/ld: cannot find -lgcc_s\n\n"
                   "error: could not compile `kernel` (lib) due to 1 previous error\n", 101, False, False),
}


def proc_result(rc: int = 0, *, timed_out: bool = False, stdout: str = "", stderr: str = "") -> ProcResult:
    return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.01, stdout_tail=stdout, stderr_tail=stderr,
                      signal_name="SIGKILL" if rc == -9 else None)


class FakeCargo:
    """``harness.build.rust.run_isolated`` 의 대역. ``rustc --version``, ``cargo clean``, ``cargo build`` 를 흉내 낸다:
    build 는 ``src/lib.rs`` 에 든 ``SCENARIOS`` 의 표식대로 ``build.log`` 를 쓰고 ``CARGO_TARGET_DIR`` 에 라이브러리를 만든다."""

    def __init__(self) -> None:
        self.calls: list[SimpleNamespace] = []
        self.program = CARGO                 # 명령에서 cargo 실행 파일의 이름(그 다음 인자가 하위 명령이다)
        self.library = ELF_LIBRARY
        self.clean = proc_result(stderr="     Removed 14 files, 1.1MiB total\n")
        self.unstartable: set[str] = set()   # 이 실행 파일로 시작하는 명령은 띄울 수 없다(FileNotFoundError)
        self.force: str | None = None        # 있으면 소스의 표식과 무관하게 이 SCENARIOS 항목대로 빌드한다(probe_toolchain 용)
        self.on_call: Callable[[SimpleNamespace], None] | None = None   # clean·build 를 흉내 내기 전에 부른다

    def __call__(self, cmd: list[str], *, timeout_s: float, cwd: Path, env: dict[str, str] | None = None,
                 stdout_path: Path | None = None, stderr_path: Path | None = None, tail_chars: int = 2000) -> ProcResult:
        if cmd[0] in self.unstartable:
            raise FileNotFoundError(2, "No such file or directory", cmd[0])
        if cmd[0] == rust_mod.RUSTC:
            return proc_result(stdout=RUSTC_VERSION + "\n")
        call = SimpleNamespace(cmd=list(cmd), subcommand=cmd[cmd.index(self.program) + 1], timeout_s=timeout_s, cwd=Path(cwd),
                               env=dict(env or {}), stderr_path=stderr_path)
        self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)
        if call.subcommand == "clean":
            return self.clean
        source = (call.cwd / "src" / "lib.rs").read_text(encoding="utf-8")
        log, rc, timed_out, built = SCENARIOS[self.force] if self.force else next(
            (s for marker, s in SCENARIOS.items() if marker in source), (COMPILING_KERNEL + FINISHED, 0, False, True))
        assert stderr_path is not None
        Path(stderr_path).write_text(log, encoding="utf-8")
        if built:
            library = Path(call.env["CARGO_TARGET_DIR"]) / "release" / CARGO_CDYLIB
            library.parent.mkdir(parents=True, exist_ok=True)
            library.write_bytes(self.library(source) if callable(self.library) else self.library)
        return proc_result(rc, timed_out=timed_out)


class FakeCargoCase(unittest.TestCase):
    """대역 cargo 로 ``build_rust`` 를 부르는 테스트의 공통 부분. ``nm`` 은 PATH 에 없는 것으로 둔다."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="build_rust_test_")).resolve()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.out = self.tmp / "sample_0"
        self.rcfg = SimpleNamespace(cargo=CARGO)
        self.cargo = FakeCargo()
        rust_mod.rustc_version.cache_clear()
        self.addCleanup(rust_mod.rustc_version.cache_clear)   # 대역의 버전 문자열이 다른 테스트로 새지 않게
        for name, value in (("run_isolated", self.cargo), ("NM", "no-such-nm-for-build-rust-test")):
            patcher = mock.patch.object(rust_mod, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def build(self, code: str = RS_DOT, **kwargs: Any) -> Any:
        kwargs.setdefault("launch_prefix", [])
        return build_rust(code, self.out, kwargs.pop("rcfg", self.rcfg), symbol="dot", **kwargs)

    def files(self) -> set[str]:
        return {p.relative_to(self.out).as_posix() for p in self.out.rglob("*") if p.is_file()}


class FakeCargoTest(FakeCargoCase):
    def test_files_commands_and_environment(self) -> None:
        res = self.build()
        self.assertEqual((res.ok, res.error_kind, res.error_detail), (True, "", ""))
        clean, build = self.cargo.calls
        self.assertEqual(clean.cmd, [CARGO, "clean", "--release", "-p", "kernel", "--offline", "--locked"])
        self.assertEqual(build.cmd, [CARGO, "build", "--release", "--offline", "--locked"])
        self.assertEqual(res.command, build.cmd)
        target = self.out / "target"
        for call in (clean, build):
            with self.subTest(call=call.subcommand):
                self.assertEqual(call.cwd, self.out)
                self.assertEqual(call.env, {**toolchain_env(), "RUSTFLAGS": "-C target-cpu=native", "CARGO_TARGET_DIR": str(target),
                                            "CARGO_TERM_COLOR": "never", "CARGO_TERM_QUIET": "false"})
        self.assertEqual((build.timeout_s, build.stderr_path), (600.0, self.out / "build.log"))

        self.assertEqual(self.files(), {"Cargo.toml", "Cargo.lock", "src/lib.rs", "build.log", "build.json", ARTIFACT_NAME,
                                        f"target/release/{CARGO_CDYLIB}", f"target/{LOCK_NAME}"})
        for name in ("Cargo.toml", "Cargo.lock"):
            self.assertEqual((self.out / name).read_bytes(), (TEMPLATE_DIR / name).read_bytes())
        self.assertEqual((self.out / "src" / "lib.rs").read_bytes().decode("utf-8"), RS_DOT)
        self.assertEqual(res.artifact, self.out / ARTIFACT_NAME)
        self.assertEqual((self.out / ARTIFACT_NAME).read_bytes(), ELF_LIBRARY)

        self.assertEqual((res.compiler, res.compiler_version, res.n_warnings, res.log), ("rustc", RUSTC_VERSION, 0, COMPILING_KERNEL + FINISHED))
        self.assertEqual(res.extra, {"rustflags": ["-C", "target-cpu=native"], "symbol_check": "skipped:nm_not_found"})
        self.assertEqual((res.undefined_symbols, res.forbidden_symbols, res.thread_symbols, res.march_resolved), ([], [], [], ""))
        self.assertEqual(json.loads((self.out / "build.json").read_text(encoding="utf-8")), json.loads(json.dumps(res.to_dict())))

    def test_missing_config_values_fall_back_to_the_defaults(self) -> None:
        # 설정 객체가 없거나 속성이 없으면 설계 v3 §8 의 기본값: cargo, -C target-cpu=native, 600 s, 오프라인
        self.assertEqual((rust_mod.DEFAULT_CARGO, rust_mod.DEFAULT_RUSTFLAGS, rust_mod.DEFAULT_BUILD_TIMEOUT_S, rust_mod.DEFAULT_OFFLINE),
                         ("cargo", ("-C", "target-cpu=native"), 600.0, True))
        self.cargo.program = "cargo"
        cargo_on_path = mock.patch.object(rust_mod.shutil, "which", side_effect=lambda name: "/usr/bin/cargo" if name == "cargo" else None)
        for rcfg in (None, SimpleNamespace(), object()):
            with self.subTest(rcfg=rcfg), cargo_on_path:
                self.cargo.calls.clear()
                res = self.build(rcfg=rcfg)
                clean, build = self.cargo.calls
                self.assertEqual(clean.cmd, ["cargo", "clean", "--release", "-p", "kernel", "--offline", "--locked"])
                self.assertEqual(res.command, ["cargo", "build", "--release", "--offline", "--locked"])
                self.assertEqual((build.timeout_s, build.env["RUSTFLAGS"]), (600.0, "-C target-cpu=native"))

    def test_configured_values(self) -> None:
        rcfg = SimpleNamespace(cargo=CARGO, rustflags=["-C", "target-cpu=haswell", "-C", "opt-level=2"], build_timeout_s=42, offline=False)
        res = self.build(rcfg=rcfg)
        clean, build = self.cargo.calls
        self.assertEqual(clean.cmd, [CARGO, "clean", "--release", "-p", "kernel", "--locked"])
        self.assertEqual(build.cmd, [CARGO, "build", "--release", "--locked"])
        self.assertEqual((build.env["RUSTFLAGS"], build.timeout_s), ("-C target-cpu=haswell -C opt-level=2", 42.0))
        self.assertEqual(res.extra["rustflags"], ["-C", "target-cpu=haswell", "-C", "opt-level=2"])

    def test_args_module_is_written_only_when_given(self) -> None:
        self.build()
        self.assertFalse((self.out / "src" / "kernel_args.rs").exists())
        self.out = self.tmp / "sample_keyword"
        res = self.build(RS_DOT_KEYWORD, args_module=KERNEL_ARGS_RS)
        self.assertTrue(res.ok)
        self.assertEqual((self.out / "src" / "kernel_args.rs").read_bytes().decode("utf-8"), KERNEL_ARGS_RS)
        self.assertIn("pub struct DotArgs {", KERNEL_ARGS_RS)
        self.assertEqual((self.out / "src" / "lib.rs").read_bytes().decode("utf-8"), RS_DOT_KEYWORD)

    def test_target_dir_is_shared_when_given(self) -> None:
        shared = self.tmp / "run" / "_cargo_target"
        res = self.build(target_dir=shared)
        self.assertTrue(res.ok)
        self.assertEqual([call.env["CARGO_TARGET_DIR"] for call in self.cargo.calls], [str(shared), str(shared)])
        self.assertFalse((self.out / "target").exists())
        self.assertEqual((self.out / ARTIFACT_NAME).read_bytes(), (shared / "release" / CARGO_CDYLIB).read_bytes())

    def test_launch_prefix_is_in_front_of_the_build_command(self) -> None:
        prefix = ["taskset", "-c", "0-3"]
        res = self.build(launch_prefix=prefix)
        clean, build = self.cargo.calls
        self.assertEqual(build.cmd, [*prefix, CARGO, "build", "--release", "--offline", "--locked"])
        self.assertEqual(res.command, build.cmd)
        self.assertEqual(clean.cmd[0], CARGO, "정리는 컴파일이 아니다: 코어를 고정하지 않는다")

    def test_kernel_package_is_cleaned_before_every_build(self) -> None:
        # target 디렉터리를 나눠 쓰는 다른 샘플의 산출물이 남아 있으면 cargo 가 그것을 새것으로 볼 수 있다(모듈 docstring)
        self.build()
        self.build(RS_BROKEN)
        self.assertEqual([call.subcommand for call in self.cargo.calls], ["clean", "build", "clean", "build"])

    def test_source_gets_a_final_newline(self) -> None:
        self.build(RS_DOT.rstrip("\n"))
        self.assertEqual((self.out / "src" / "lib.rs").read_bytes().decode("utf-8"), RS_DOT)

    def test_compile_error(self) -> None:
        self.out.mkdir()
        (self.out / ARTIFACT_NAME).write_bytes(b"library of an earlier build")
        res = self.build(RS_BROKEN)
        self.assertEqual((res.ok, res.artifact, res.error_kind), (False, None, "compile_error"))
        self.assertEqual(res.error_detail, "error[E0425]: cannot find value `undefined_name` in this scope --> src/lib.rs:3:12")
        self.assertEqual(res.log, COMPILE_ERROR_LOG)
        self.assertFalse((self.out / ARTIFACT_NAME).exists(), "이전 빌드의 산출물이 남으면 안 된다")
        saved = json.loads((self.out / "build.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["ok"], saved["artifact"], saved["error_kind"], saved["extra"]),
                         (False, None, "compile_error", {"rustflags": ["-C", "target-cpu=native"]}))

    def test_warnings_are_counted(self) -> None:
        res = self.build(RS_DOT + "// WARNINGS\n")
        self.assertEqual((res.ok, res.n_warnings), (True, 2))

    def test_timeout_while_compiling_the_kernel(self) -> None:
        res = self.build(RS_DOT + "// SLOW_KERNEL\n", rcfg=SimpleNamespace(cargo=CARGO, build_timeout_s=1.5))
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "timeout", "compile timeout after 1.5s"))
        self.assertEqual(json.loads((self.out / "build.json").read_text(encoding="utf-8"))["error_kind"], "timeout")

    def test_failures_before_the_kernel_is_compiled_are_toolchain_problems(self) -> None:
        # 이 샘플의 코드와 무관한 실패: compile_error 결과도 build.json 도 남기지 않는다
        for marker in ("DEPENDENCY_CACHE_MISSING", "LOCK_FILE_OUTDATED", "DEPENDENCY_FAILS", "LINKER_MISSING", "CARGO_KILLED",
                       "SLOW_DEPENDENCIES", "NO_ARTIFACT"):
            with self.subTest(marker=marker):
                self.out = self.tmp / marker.lower()
                with self.assertRaises(ToolchainMissing) as cm:
                    self.build(RS_DOT + f"// {marker}\n")
                self.assertFalse((self.out / "build.json").exists())
                self.assertFalse((self.out / ARTIFACT_NAME).exists())
                if SCENARIOS[marker][1] == 101:
                    self.assertIn(FETCH_HINT, str(cm.exception))
                    self.assertIn(SCENARIOS[marker][0].splitlines()[0].strip(), str(cm.exception))

    def test_clean_failure_is_a_toolchain_problem(self) -> None:
        failures = {
            "rc=101": proc_result(101, stderr="error: no matching package named `rayon` found\nlocation searched: crates.io index\n"),
            "120s 안에 끝나지 않았다": proc_result(-9, timed_out=True),
        }
        for why, result in failures.items():
            with self.subTest(why=why):
                self.cargo.calls.clear()
                self.cargo.clean = result
                with self.assertRaises(ToolchainMissing) as cm:
                    self.build()
                self.assertIn(why, str(cm.exception))
                self.assertIn(FETCH_HINT, str(cm.exception))
                self.assertEqual([call.subcommand for call in self.cargo.calls], ["clean"], "정리가 실패하면 빌드하지 않는다")
                self.assertFalse((self.out / "build.json").exists())

    def test_missing_cargo(self) -> None:
        with self.assertRaises(ToolchainMissing) as cm:
            self.build(rcfg=SimpleNamespace(cargo="no-such-cargo-for-build-rust-test"))
        self.assertIn("no-such-cargo-for-build-rust-test", str(cm.exception))
        self.assertFalse(self.out.exists(), "도구가 없으면 아무것도 쓰지 않는다")
        self.assertEqual(self.cargo.calls, [])

    def test_commands_that_cannot_be_started(self) -> None:
        self.cargo.unstartable = {"no-such-launcher-for-build-rust-test"}
        with self.assertRaises(ToolchainMissing) as cm:
            self.build(launch_prefix=["no-such-launcher-for-build-rust-test", "-c", "0"])
        self.assertIn("no-such-launcher-for-build-rust-test", str(cm.exception))
        self.assertIsInstance(cm.exception.__cause__, OSError)
        self.assertFalse((self.out / "build.json").exists())
        self.cargo.unstartable = {CARGO}     # PATH 검사와 실행 사이에 사라진 경우
        with self.assertRaises(ToolchainMissing) as cm:
            self.build()
        self.assertIsInstance(cm.exception.__cause__, OSError)

    def test_rustc_version_is_empty_when_rustc_cannot_run(self) -> None:
        self.cargo.unstartable = {rust_mod.RUSTC}
        res = self.build()
        self.assertEqual((res.ok, res.compiler, res.compiler_version), (True, "rustc", ""))

    def test_link_failure_inside_the_kernel_crate_is_still_a_compile_error(self) -> None:
        # 샘플 하나만 보면 모델 코드의 링크 실패와 구별할 수 없다: 환경 쪽은 run 시작 때의 probe_toolchain 이 잡는다
        res = self.build(RS_DOT + "// LINK_FAILS\n")
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))


def lock_is_held(target: Path) -> bool:
    """다른 열린 파일로 ``target`` 의 잠금을 시도해 본다(잡혀 있으면 True; 잡지 않았으면 곧바로 놓는다)."""
    fd = os.open(target / LOCK_NAME, os.O_RDWR | os.O_CREAT)
    try:
        try:
            rust_mod._try_lock(fd)
        except OSError:
            return True
        rust_mod._unlock(fd)
        return False
    finally:
        os.close(fd)


class TargetLockTest(FakeCargoCase):
    """같은 target 디렉터리를 쓰는 빌드끼리는 정리 → 빌드 → 산출물 읽기가 겹치지 않는다(프로세스 간 잠금)."""

    def test_clean_and_build_run_under_the_lock(self) -> None:
        held: list[tuple[str, bool]] = []
        self.cargo.on_call = lambda call: held.append((call.subcommand, lock_is_held(Path(call.env["CARGO_TARGET_DIR"]))))
        res = self.build()
        self.assertTrue(res.ok, res.error_detail)
        self.assertEqual(held, [("clean", True), ("build", True)])
        self.assertFalse(lock_is_held(self.out / "target"), "빌드가 끝나면 잠금을 놓는다")
        # 실패한 빌드(예외)도 잠금을 놓는다
        self.out = self.tmp / "no_artifact"
        with self.assertRaises(ToolchainMissing):
            self.build(RS_DOT + "// NO_ARTIFACT\n")
        self.assertFalse(lock_is_held(self.out / "target"))

    def test_concurrent_builds_in_one_target_do_not_overlap(self) -> None:
        shared = self.tmp / "_cargo_target"
        spans: dict[str, list[float]] = {}
        barrier = threading.Barrier(2)

        def on_call(call: SimpleNamespace) -> None:
            name = call.cwd.name
            if call.subcommand == "clean":
                spans[name] = [time.monotonic()]
            else:
                time.sleep(0.3)   # 빌드가 도는 동안: 잠금이 없으면 다른 쪽의 정리·빌드가 이 사이에 끼어든다
                spans[name].append(time.monotonic())

        self.cargo.on_call = on_call
        self.cargo.library = lambda source: ELF_LIBRARY + source.encode("utf-8")[-12:]
        results: dict[str, Any] = {}

        def worker(name: str) -> None:
            barrier.wait()
            results[name] = build_rust(f"{RS_DOT}// kernel {name}\n", self.tmp / name, self.rcfg, launch_prefix=[], symbol="dot",
                                       target_dir=shared)

        threads = [threading.Thread(target=worker, args=(name,)) for name in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(30)
        self.assertEqual(sorted(results), ["a", "b"])
        (a0, a1), (b0, b1) = spans["a"], spans["b"]
        self.assertTrue(a1 <= b0 or b1 <= a0, f"정리~빌드 구간이 겹쳤다: a={spans['a']} b={spans['b']}")
        for name in ("a", "b"):
            with self.subTest(name=name):
                self.assertTrue(results[name].ok)
                self.assertTrue((self.tmp / name / ARTIFACT_NAME).read_bytes().endswith(f"kernel {name}\n".encode()))

    def test_lock_wait_timeout_is_a_toolchain_problem(self) -> None:
        self.assertEqual(lock_wait_s(600.0), 2.0 * (120.0 + 600.0))
        target = self.out / "target"
        with mock.patch.object(rust_mod, "CLEAN_TIMEOUT_S", 0.0), mock.patch.object(rust_mod, "LOCK_POLL_S", 0.01), \
                target_lock(target, 1.0):
            with self.assertRaises(ToolchainMissing) as cm:
                self.build(rcfg=SimpleNamespace(cargo=CARGO, build_timeout_s=0.1))
        self.assertIn(LOCK_NAME, str(cm.exception))
        self.assertEqual(self.cargo.calls, [], "잠금을 얻지 못하면 cargo 를 부르지 않는다")
        self.assertFalse((self.out / "build.json").exists())
        self.assertTrue(self.build().ok, "잠금이 풀리면 다시 빌드할 수 있다")


class ProbeToolchainTest(FakeCargoCase):
    """``probe_toolchain``: 템플릿의 자리표시 크레이트를 ``build_rust`` 와 같은 명령·환경으로 빌드하고, 어떤 실패든 ToolchainMissing."""

    def setUp(self) -> None:
        super().setUp()
        self.target = self.tmp / "run" / "_cargo_target"

    def probe(self, **kwargs: Any) -> None:
        kwargs.setdefault("launch_prefix", [])
        probe_toolchain(kwargs.pop("rcfg", self.rcfg), self.target, **kwargs)

    def test_success_uses_the_build_command_and_environment(self) -> None:
        self.probe()
        probe_dir = self.target / PROBE_DIR_NAME
        self.assertEqual([call.subcommand for call in self.cargo.calls], ["clean", "build"])
        probe_build = self.cargo.calls[1]
        self.assertEqual((probe_build.cwd, probe_build.stderr_path), (probe_dir, probe_dir / "probe.log"))
        self.assertEqual((probe_dir / "src" / "lib.rs").read_bytes(), (TEMPLATE_DIR / "src" / "lib.rs").read_bytes())
        for name in ("Cargo.toml", "Cargo.lock"):
            self.assertEqual((probe_dir / name).read_bytes(), (TEMPLATE_DIR / name).read_bytes())
        # 샘플 빌드와 명령·환경·시간 초과가 같다(cwd 만 다르다)
        self.build(target_dir=self.target)
        sample_build = self.cargo.calls[3]
        self.assertEqual((probe_build.cmd, probe_build.env, probe_build.timeout_s),
                         (sample_build.cmd, sample_build.env, sample_build.timeout_s))
        self.assertEqual(probe_build.env["CARGO_TARGET_DIR"], str(self.target.resolve()))

    def test_launch_prefix_and_configured_values_are_used(self) -> None:
        rcfg = SimpleNamespace(cargo=CARGO, rustflags=["-C", "target-cpu=haswell"], build_timeout_s=42, offline=False)
        self.probe(rcfg=rcfg, launch_prefix=["taskset", "-c", "2"])
        probe_build = self.cargo.calls[1]
        self.assertEqual(probe_build.cmd, ["taskset", "-c", "2", CARGO, "build", "--release", "--locked"])
        self.assertEqual((probe_build.env["RUSTFLAGS"], probe_build.timeout_s), ("-C target-cpu=haswell", 42.0))

    def test_every_failure_is_a_toolchain_problem(self) -> None:
        decisive = {
            "LINK_FAILS": "error: linking with `cc` failed: exit status: 1",
            "DEPENDENCY_CACHE_MISSING": "error: no matching package named `rayon` found",
            "LINKER_MISSING": "error: linker `cc` not found",
            "undefined_name": "error[E0425]: cannot find value `undefined_name` in this scope --> src/lib.rs:3:12",
            "SLOW_DEPENDENCIES": "600s 안에 끝나지 않았다",
            "SLOW_KERNEL": "600s 안에 끝나지 않았다",
            "CARGO_KILLED": "rc=-9",
            "NO_ARTIFACT": "rc=0 인데 산출물이 없다",
        }
        for scenario, line in decisive.items():
            with self.subTest(scenario=scenario):
                self.cargo.force = scenario
                with self.assertRaises(ToolchainMissing) as cm:
                    self.probe()
                self.assertIn(line, str(cm.exception))
                self.assertIn(FETCH_HINT, str(cm.exception))
                self.assertFalse(lock_is_held(self.target))

    def test_missing_cargo_and_unstartable_commands(self) -> None:
        with self.assertRaises(ToolchainMissing):
            self.probe(rcfg=SimpleNamespace(cargo="no-such-cargo-for-build-rust-test"))
        self.assertEqual(self.cargo.calls, [])
        self.cargo.unstartable = {"no-such-launcher-for-build-rust-test"}
        with self.assertRaises(ToolchainMissing) as cm:
            self.probe(launch_prefix=["no-such-launcher-for-build-rust-test"])
        self.assertIsInstance(cm.exception.__cause__, OSError)


class CliTest(FakeCargoCase):
    """``python -m harness.build.rust`` (진단용): 0 빌드 성공 / 1 커널이 빌드되지 않음 / 2 설정·도구 문제."""

    def run_cli(self, code: str, **load_config: Any) -> tuple[int, str]:
        src = self.tmp / "kernel.rs"
        src.write_text(code, encoding="utf-8")
        load_config.setdefault("return_value", SimpleNamespace(build=SimpleNamespace(rust=self.rcfg)))
        stdout = io.StringIO()
        with mock.patch.object(rust_mod, "load_config", **load_config), contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(io.StringIO()):
            rc = rust_mod.main(["--src", str(src), "--out-dir", str(self.out), "--symbol", "dot", "--no-taskset"])
        return rc, stdout.getvalue()

    def test_exit_codes(self) -> None:
        rc, out = self.run_cli(RS_DOT)
        self.assertEqual((rc, json.loads(out)["ok"], json.loads(out)["command"]), (0, True, [CARGO, "build", "--release", "--offline", "--locked"]))
        rc, out = self.run_cli(RS_BROKEN)
        self.assertEqual((rc, json.loads(out)["error_kind"]), (1, "compile_error"))
        self.assertEqual(self.run_cli(RS_DOT + "// DEPENDENCY_CACHE_MISSING\n"), (2, ""))
        self.assertEqual(self.run_cli(RS_DOT, side_effect=rust_mod.ConfigError("build.rust: unknown key")), (2, ""))


class SymbolCheckTest(FakeCargoCase):
    """빌드가 성공한 뒤의 심벌 검사: ELF 이고 ``nm`` 이 있을 때만 C 와 같은 ``nm -D --defined-only`` 검사를 한다."""

    def with_nm(self, defined: str) -> mock.MagicMock:
        """``nm`` 이 PATH 에 있고 ``defined`` 를 출력하는 상태로 만든다. ``nm_dynamic`` 의 대역을 돌려준다."""
        nm = mock.MagicMock(return_value=defined)
        for name, value in (("NM", sys.executable), ("nm_dynamic", nm)):
            patcher = mock.patch.object(rust_mod, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        return nm

    def test_exported_symbol(self) -> None:
        nm = self.with_nm(NM_DEFINED)
        res = self.build()
        self.assertEqual((res.ok, res.artifact, res.extra["symbol_check"]), (True, self.out / ARTIFACT_NAME, "nm"))
        nm.assert_called_once_with(self.out, "--defined-only", ARTIFACT_NAME)

    def test_symbol_not_exported(self) -> None:
        self.with_nm("0000000000012400 T rust_eh_personality\n0000000000012340 t dot\n")
        res = self.build()
        self.assertEqual((res.ok, res.artifact, res.error_kind, res.error_detail), (False, None, "signature_mismatch", "symbol dot not exported"))
        self.assertEqual(res.extra["symbol_check"], "nm")
        self.assertEqual(json.loads((self.out / "build.json").read_text(encoding="utf-8"))["error_detail"], "symbol dot not exported")

    def test_nm_failure_is_not_a_missing_symbol(self) -> None:
        nm = self.with_nm("")
        nm.side_effect = ToolchainMissing("`nm -D --defined-only kernel.so` 실패(rc=1)")
        with self.assertRaises(ToolchainMissing):
            self.build()
        self.assertFalse((self.out / "build.json").exists())

    def test_skipped_without_nm(self) -> None:
        res = self.build()      # setUp: nm 이 PATH 에 없다
        self.assertEqual((res.ok, res.extra["symbol_check"]), (True, "skipped:nm_not_found"))

    def test_skipped_for_a_library_that_is_not_elf(self) -> None:
        nm = self.with_nm(NM_DEFINED)
        self.cargo.library = b"MZ\x90\x00 windows dll"      # 개발용 Windows 의 kernel.dll: nm -D 의 대상이 아니다
        res = self.build()
        self.assertEqual((res.ok, res.extra["symbol_check"]), (True, "skipped:not_elf"))
        nm.assert_not_called()


# ----------------------------------------------------------------------------- 진짜 cargo

# 라이브러리를 ctypes 로 열어 dot 을 한 번 부른다: argv = 라이브러리 경로, n, positional|struct
CALL_DOT = r'''
import ctypes
import os
import sys

if os.name == "nt":  # 프로세스가 죽을 때 오류 대화 상자를 띄우지 않는다
    ctypes.windll.kernel32.SetErrorMode(0x8003)
lib = ctypes.CDLL(sys.argv[1])
n, mode = int(sys.argv[2]), sys.argv[3]
if not hasattr(lib, "dot"):
    print("symbol dot not found")
    sys.exit(0)
pointer = ctypes.POINTER(ctypes.c_float)
size = max(n, 0)
x = (ctypes.c_float * size)(*[0.5 * i for i in range(size)])
y = (ctypes.c_float * size)(*[float(i % 7 + 1) for i in range(size)])
out = (ctypes.c_float * 1)(-1.0)
lib.dot.restype = None
print("calling", flush=True)
if mode == "struct":
    class DotArgs(ctypes.Structure):
        _fields_ = [("x", pointer), ("y", pointer), ("out", pointer), ("n", ctypes.c_int)]
    lib.dot.argtypes = [ctypes.POINTER(DotArgs)]
    lib.dot(ctypes.byref(DotArgs(x, y, out, n)))
else:
    lib.dot.argtypes = [pointer, pointer, pointer, ctypes.c_int]
    lib.dot(x, y, out, n)
print("result", repr(out[0]), flush=True)
'''

N = 10000   # 4096 원소 조각 세 개


def expected_dot(n: int) -> float:
    """``CALL_DOT`` 의 입력에 대한 내적을 float32 로 내린 값(모든 부분 합이 f64 로 정확하다 → 누적 순서와 무관)."""
    exact = sum(0.5 * i * float(i % 7 + 1) for i in range(n))
    return struct.unpack("f", struct.pack("f", exact))[0]


def call_dot(library: Path, n: int = N, mode: str = "positional") -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", CALL_DOT, str(library), str(n), mode], stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, encoding="utf-8", errors="replace", timeout=300, env={**os.environ, "RAYON_NUM_THREADS": "2"})


def backdated(write: Callable[..., None]) -> Callable[..., None]:
    """쓴 파일의 수정 시각을 2000-01-01 로 돌려놓는 쓰기 함수(시계가 뒤로 간 머신의 흉내)."""
    def wrapper(path: Any, *args: Any, **kwargs: Any) -> None:
        write(path, *args, **kwargs)
        os.utime(path, (946684800, 946684800))
    return wrapper


@requires_tool("cargo")
class RealCargoTest(unittest.TestCase):
    """샘플 디렉터리마다 다른 커널을, run.py 처럼 target 디렉터리 하나를 나눠 쓰며 빌드한다(의존성은 한 번만 컴파일된다)."""

    tmp: Path
    target: Path
    dot: Any

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = Path(tempfile.mkdtemp(prefix="build_rust_test_")).resolve()
        cls.addClassCleanup(shutil.rmtree, cls.tmp, ignore_errors=True)
        cls.target = cls.tmp / "_cargo_target"
        rust_mod.rustc_version.cache_clear()
        cls.dot = cls.build_kernel("dot", RS_DOT)

    @classmethod
    def build_kernel(cls, name: str, code: str, **kwargs: Any) -> Any:
        return build_rust(code, cls.tmp / name, SimpleNamespace(), launch_prefix=[], symbol="dot", target_dir=cls.target, **kwargs)

    def test_correct_kernel_builds_loads_and_computes(self) -> None:
        res, out = self.dot, self.tmp / "dot"
        self.assertEqual((res.ok, res.error_kind, res.error_detail), (True, "", ""))
        self.assertEqual(res.artifact, out / ARTIFACT_NAME)
        self.assertEqual({p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()},
                         {"Cargo.toml", "Cargo.lock", "src/lib.rs", "build.log", "build.json", ARTIFACT_NAME})
        self.assertEqual(res.command, ["cargo", "build", "--release", "--offline", "--locked"])
        self.assertEqual(res.compiler, "rustc")
        self.assertRegex(res.compiler_version, r"^rustc \d+\.\d+\.\d+")
        self.assertRegex(res.log, r"(?m)^\s*Compiling rayon v\d")           # 고정한 의존성이 캐시에서(오프라인) 컴파일됐다
        self.assertRegex(res.log, r"(?m)^\s*Compiling kernel v0\.1\.0 ")
        self.assertEqual(res.extra["rustflags"], ["-C", "target-cpu=native"])
        elf = (out / ARTIFACT_NAME).read_bytes().startswith(b"\x7fELF")
        expected_check = "skipped:not_elf" if not elf else "nm" if shutil.which("nm") else "skipped:nm_not_found"
        self.assertEqual(res.extra["symbol_check"], expected_check)
        self.assertEqual(json.loads((out / "build.json").read_text(encoding="utf-8")), json.loads(json.dumps(res.to_dict())))

        proc = call_dot(res.artifact)
        self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["calling", "result", repr(expected_dot(N))]), proc.stderr[-2000:])

    def test_panic_aborts_the_process(self) -> None:
        proc = call_dot(self.dot.artifact, n=-1)
        self.assertEqual(proc.stdout.split(), ["calling"], "패닉 뒤에는 호출자로 돌아오지 않는다(panic = abort)")
        self.assertIn("n must not be negative", proc.stderr)
        if os.name == "posix":
            self.assertEqual(proc.returncode, -signal.SIGABRT)
        else:
            self.assertNotEqual(proc.returncode, 0)

    def test_keyword_kernel_gets_the_args_module(self) -> None:
        res = self.build_kernel("keyword", RS_DOT_KEYWORD, args_module=KERNEL_ARGS_RS)
        self.assertEqual((res.ok, res.error_detail), (True, ""))
        self.assertEqual((self.tmp / "keyword" / "src" / "kernel_args.rs").read_bytes().decode("utf-8"), KERNEL_ARGS_RS)
        proc = call_dot(res.artifact, mode="struct")
        self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["calling", "result", repr(expected_dot(N))]), proc.stderr[-2000:])
        # 구조체 파일을 주지 않으면(positional 조건) 그 모듈을 쓰는 커널은 컴파일 오류다
        res = self.build_kernel("keyword_without_module", RS_DOT_KEYWORD)
        self.assertEqual((res.ok, res.error_kind), (False, "compile_error"))
        self.assertRegex(res.error_detail, r"^error\[E0583\]: file not found for module `kernel_args` --> src/lib\.rs:1:1")

    def test_compile_error_has_the_rustc_message(self) -> None:
        res = self.build_kernel("broken", RS_BROKEN)
        self.assertEqual((res.ok, res.artifact, res.error_kind), (False, None, "compile_error"))
        self.assertEqual(res.error_detail, "error[E0425]: cannot find value `undefined_name` in this scope --> src/lib.rs:3:12")
        self.assertIn("could not compile `kernel`", res.log)
        self.assertFalse((self.tmp / "broken" / ARTIFACT_NAME).exists())
        saved = json.loads((self.tmp / "broken" / "build.json").read_text(encoding="utf-8"))
        self.assertEqual((saved["ok"], saved["error_kind"], saved["error_detail"]), (False, "compile_error", res.error_detail))

    def test_shared_target_dir_never_serves_another_samples_library(self) -> None:
        # cargo 는 소스의 수정 시각으로만 신선도를 판단하고, 샘플 디렉터리가 달라도 패키지가 같아 산출물이 겹친다: 소스가
        # 이미 있는 산출물(setUpClass 의 dot)보다 오래돼 보이면 다시 컴파일하지 않는다. 빌드 전의 정리가 이것을 막는다.
        with mock.patch.object(rust_mod, "atomic_write_text", backdated(rust_mod.atomic_write_text)), \
                mock.patch.object(rust_mod, "atomic_write_bytes", backdated(rust_mod.atomic_write_bytes)):
            broken = self.build_kernel("stale_broken", RS_BROKEN)
            constant = self.build_kernel("stale_constant", RS_CONSTANT)
        self.assertLess((self.tmp / "stale_broken" / "src" / "lib.rs").stat().st_mtime, 1e9)
        self.assertEqual((broken.ok, broken.error_kind), (False, "compile_error"), "다른 샘플의 라이브러리로 통과하면 안 된다")
        self.assertTrue(constant.ok, constant.error_detail)
        proc = call_dot(constant.artifact)
        self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["calling", "result", "42.0"]), proc.stderr[-2000:])

    def test_missing_no_mangle_is_caught_by_extract_and_by_the_symbol_check(self) -> None:
        bundle = sigs.signature_bundle(DOT_SPEC, "rust", "rust", "positional")
        ext = extract(f"```rust\n{RS_NOT_EXPORTED}```\n", backend="rust", signature_spec={}, bundle=bundle)
        self.assertEqual((ext.ok, ext.signature_ok, ext.detail), (False, False, "no_mangle_missing"))
        self.assertTrue(extract(f"```rust\n{RS_CONSTANT}```\n", backend="rust", signature_spec={}, bundle=bundle).ok)
        # 추출을 거치지 않고 빌드하면: 심벌 검사를 할 수 있는 곳(ELF + nm)에서는 빌드 단계가, 아니면 로드한 쪽이 잡는다
        res = self.build_kernel("not_exported", RS_NOT_EXPORTED)
        if res.extra["symbol_check"] == "nm":
            self.assertEqual((res.ok, res.error_kind, res.error_detail), (False, "signature_mismatch", "symbol dot not exported"))
        else:
            self.assertTrue(res.ok, res.error_detail)
            proc = call_dot(res.artifact)
            self.assertEqual((proc.returncode, proc.stdout.strip()), (0, "symbol dot not found"), proc.stderr[-2000:])

    def test_concurrent_builds_of_different_kernels_get_their_own_library(self) -> None:
        # 같은 target 을 쓰는 두 빌드를 동시에(병렬 devcheck, 같은 run id 의 프로세스 둘): 잠금이 없으면 한쪽이 다른 쪽의
        # 라이브러리를 복사하거나 산출물을 못 찾았다. 커널마다 결과가 다르므로 바뀌어 들어가면 계산 값이 틀린다.
        kernels = {f"concurrent_{value:g}": RS_CONSTANT.replace("42.0", f"{value:.1f}") for value in (7.0, 8.0, 9.0)}
        results: dict[str, Any] = {}
        errors: list[BaseException] = []
        barrier = threading.Barrier(len(kernels))

        def worker(name: str, code: str) -> None:
            try:
                barrier.wait()
                results[name] = self.build_kernel(name, code)
            except BaseException as e:   # 스레드의 예외를 테스트로 넘긴다
                errors.append(e)

        threads = [threading.Thread(target=worker, args=item) for item in kernels.items()]
        for t in threads:
            t.start()
        for t in threads:
            t.join(600)
        self.assertEqual(errors, [])
        for name in kernels:
            with self.subTest(name=name):
                res = results[name]
                self.assertTrue(res.ok, res.error_detail)
                proc = call_dot(res.artifact)
                expected = repr(float(name.rsplit("_", 1)[1]))
                self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["calling", "result", expected]), proc.stderr[-2000:])

    def test_probe_builds_the_placeholder_and_leaves_the_dependencies(self) -> None:
        target = self.tmp / "_probe_target"
        probe_toolchain(SimpleNamespace(), target, launch_prefix=[])
        probe_log = (target / PROBE_DIR_NAME / "probe.log").read_text(encoding="utf-8")
        self.assertRegex(probe_log, r"(?m)^\s*Compiling rayon v\d")
        self.assertTrue((target / "release" / CARGO_CDYLIB).is_file())
        res = build_rust(RS_CONSTANT, self.tmp / "after_probe", SimpleNamespace(), launch_prefix=[], symbol="dot", target_dir=target)
        self.assertTrue(res.ok, res.error_detail)
        self.assertNotRegex(res.log, r"(?m)^\s*Compiling rayon", "샘플은 점검이 남긴 의존성 빌드를 재사용한다")
        self.assertRegex(res.log, r"(?m)^\s*Compiling kernel v0\.1\.0 ")
        proc = call_dot(res.artifact)
        self.assertEqual((proc.returncode, proc.stdout.split()), (0, ["calling", "result", "42.0"]), proc.stderr[-2000:])

    def test_probe_reports_a_missing_dependency_cache(self) -> None:
        empty_home = self.tmp / "empty_cargo_home_probe"
        empty_home.mkdir()
        with mock.patch.dict(os.environ, {"CARGO_HOME": str(empty_home)}), self.assertRaises(ToolchainMissing) as cm:
            probe_toolchain(SimpleNamespace(), self.tmp / "_probe_no_cache", launch_prefix=[])
        self.assertIn(FETCH_HINT, str(cm.exception))
        self.assertIn("rayon", str(cm.exception))

    def test_missing_dependency_cache_is_a_toolchain_problem(self) -> None:
        empty_home = self.tmp / "empty_cargo_home"
        empty_home.mkdir()
        with mock.patch.dict(os.environ, {"CARGO_HOME": str(empty_home)}), self.assertRaises(ToolchainMissing) as cm:
            self.build_kernel("no_cache", RS_DOT)
        self.assertIn(FETCH_HINT, str(cm.exception))
        self.assertFalse((self.tmp / "no_cache" / "build.json").exists())


if __name__ == "__main__":
    unittest.main()
