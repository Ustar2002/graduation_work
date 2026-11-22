"""harness/run.py 테스트: 종료 규칙(verdict), 건너뜀 사전 평가, raw 탐색 순서, 백엔드별 단계 순서와 행, 선택 단계
(lint·perf_summary·sanitizer), 수정 턴, 도구 점검, 재사용 API(setup_run / process_sample / finish_run), main 의 종료 코드.

빌드 함수·추출·워커·린트·sanitizer·요약은 대역(fake)이다. 여기서 보는 것은 오케스트레이션 — 무엇을 어떤 순서와 인자로 부르고
그 결과를 행에 어떻게 옮기는가 — 이고, 실제 도구가 도는 경로는 각 모듈의 테스트와 Linux 통합 실행이 본다. 설정과 과제는
임시 디렉터리에 만든 고정본을 쓴다(저장소의 experiment.yaml·tasks/ 가 바뀌어도 같은 테스트다).
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import yaml

from harness import run, sigs
from harness import worker as worker_module  # 상수만 쓴다(워커가 쓰는 문서의 꼴)
from harness.build.c import BuildResult, ToolchainMissing
from harness.config import Config, load_config
from harness.paths import raw_turn_path
from harness.procutil import ProcResult
from harness.protocol import DynamicStatus, StaticStatus
from harness.run import (
    EXIT_BASELINE,
    EXIT_ERROR,
    EXIT_HARNESS,
    EXIT_OK,
    EXIT_PROTOCOL,
    EXIT_TOOLCHAIN,
    BaselineError,
    HarnessError,
    RunContext,
    RunSetupError,
    TaskContext,
    WorkerOutcome,
    build_flags_text,
    find_raw_files,
    finish_run,
    inspect_args,
    missing_tools,
    missing_stages,
    needs_processing,
    performance_dims,
    process_sample,
    result_json_path,
    run_worker,
    setup_run,
    sha256_file,
    worker_backend_args,
)
from harness.safety import LINT_CATEGORIES, LintResult, SanitizeResult
from harness.schema import read_rows
from harness.taskspec import load_taskspec

# 미리 임포트한다: run.py 가 halide 행의 cflags 를 만들며 쓰는 자리에서 임포트하는데, 그것이 Fakes 의 sys.modules 패치 안에서
# 처음이면 패치가 끝날 때 지워져 다음 임포트가 numpy 를 다시 올린다(UserWarning: The NumPy module was reloaded).
importlib.import_module("harness.kernels")

RUN_ID = "r1"
TAG = "args-positional_hint-off_fb-compiler"
CODE = "void gemm(void) {}\n"
PREFIX = ["taskset", "-c", "9"]
BUILD_LOG_TEXT = "kernel.c:3:5: optimized: loop vectorized using 32 byte vectors\n" * 400  # 꼬리만 읽으면 잘리는 길이
CARGO_TOML = (
    '[package]\nname = "kernel"\n\n[profile.release]\nopt-level = 3\ncodegen-units = 1\nlto = "fat"\npanic = "abort"\ndebug = false\n'
)
RUST_CFLAGS = 'RUSTFLAGS=-C target-cpu=native; profile.release: opt-level=3 codegen-units=1 lto="fat" panic="abort" debug=false'

TASK_YAML = {
    "task": "gemm", "function": "gemm", "params": ["A", "B", "C", "M", "N", "K"],
    "args": [
        {"name": "A", "kind": "in", "dtype": "float32", "shape": ["M", "K"]},
        {"name": "B", "kind": "in", "dtype": "float32", "shape": ["K", "N"]},
        {"name": "C", "kind": "out", "dtype": "float32", "shape": ["M", "N"]},
        {"name": "M", "kind": "dim"}, {"name": "N", "kind": "dim"}, {"name": "K", "kind": "dim"},
    ],
    "baseline": {"backend": "python", "file": "baseline/baseline.py", "gate_blas": True},
}
MANIFEST = {
    "seed": 20261103, "test_hash": "2077183b4edc9ec8",
    "files": {"tiny": {"M": 7, "N": 5, "K": 3, "sha256_inputs": "aa"}, "large": {"M": 1024, "N": 512, "K": 256, "sha256_inputs": "bb"}},
}
STATIC = StaticStatus(
    governor="performance", governor_ok=True, epp="performance", epp_ok=True, no_turbo=1, taskset_available=True,
    cores="0,2,4,6", single_core="2", threads=4, blas={"name": "scipy-openblas", "version": "0.3.30"}, blas_ok=True,
    flags=[], ok=True, checked_at="2026-11-21T10:00:00+09:00",
)
DYNAMIC = DynamicStatus(load_1m=0.5, load_ok=True, pinned_busy_pct=1.0, sibling_busy_pct=2.0, busy_ok=True,
                        governor="performance", governor_ok=True, flags=[], ok=True, checked_at="2026-11-21T10:05:00+09:00")
TOOLCHAIN = {
    "cc": "gcc", "cc_version": "gcc 15.2.0", "cargo": "cargo 1.92.0", "rustc": "rustc 1.92.0", "numba_version": "0.68.1",
    "halide_version": "22.1.0", "halide_autoscheduler": "Mullapudi2016", "halide_autoscheduler_loadable": True,
    "lint_tools": {"ruff": "0.17.6", "pyflakes": "4.1.0", "cppcheck": None, "clang-tidy": None}, "errors": {},
}

BLAS = {"name": "scipy-openblas", "version": "0.3.30"}
IMPORT_OK = {"mode": "import", "complete": True, "load_ok": True, "load_error": None, "classification": "pass", "blas": BLAS,
             "inspect": {"signatures": ["(Array(float32, 2, 'C'),)"], "parallel_diagnostics": ""}}
CHECK_PASS = {
    "mode": "check", "complete": True, "classification": "pass", "error_detail": "", "inputs_mutated": False, "oob_write": False,
    "max_abs_err": 1e-6, "max_tol_ratio": 0.25,
    "tests": [{"name": "tiny", "classification": "pass", "pattern": "", "max_tol_ratio": 0.25, "nondeterministic": False,
               "runs": [{"elapsed_ms": 1.0}, {"elapsed_ms": 3.0}]}],
}
CHECK_WRONG = {**CHECK_PASS, "classification": "wrong_output", "error_detail": "mismatch"}
BENCH_PASS = {
    "mode": "bench", "complete": True, "classification": "pass", "error_detail": "", "final_check": {"classification": "pass"},
    "blas": BLAS, "warmup": 3, "repeats": 10, "repeats_reduced": False, "warmup_ms": [5.0], "timed_ms": [4.0, 4.2],
    "median_ms": 4.0, "mean_ms": 4.1, "min_ms": 4.0, "max_ms": 4.2, "cv": 0.03, "gflops": 100.0,
    "nondeterministic_runs": 0, "nondeterministic_bitwise_runs": 0, "inputs_mutated": False, "oob_write": False,
}
PAIRED_OK = {**BENCH_PASS, "median_ms": 8.0, "cv": 0.01, "warmup_ms": [9.0], "timed_ms": [8.0, 8.0]}


def write_config(root: Path) -> Path:
    """테스트용 설정 파일(경로는 전부 ``root`` 아래). 언어 일곱 개와 v3 의 선택 섹션을 다 갖는다."""
    doc = {
        "run": {"tasks": ["gemm", "nohalide", "cbase"], "languages": ["numpy", "c_seq", "c_openmp"], "n_samples": 3,
                "max_repair_turns": 3, "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"}},
        "model": {"provider": "dummy", "name": "dummy", "max_tokens": 1000, "temperature": None, "seed": None,
                  "request_timeout_s": 60, "max_retries": 0,
                  "anthropic": {"api_key_env": "ANTHROPIC_API_KEY", "effort": "medium", "thinking": "adaptive"},
                  "openai_compat": {"api_key_env": "OPENAI_API_KEY", "base_url_env": "LLM_BASE_URL", "max_tokens_param": "max_tokens"}},
        "languages": {
            "numpy": {"backend": "python", "prompt": "prompts/lang/numpy.md", "threads": "measurement"},
            "c_seq": {"backend": "c", "prompt": "prompts/lang/c_seq.md", "openmp": False, "threads": 1},
            "c_openmp": {"backend": "c", "prompt": "prompts/lang/c_openmp.md", "openmp": True, "threads": "measurement"},
            "numba": {"backend": "numba", "prompt": "prompts/lang/numba.md", "threads": "measurement",
                      "env": {"NUMBA_NUM_THREADS": "${threads}", "NUMBA_THREADING_LAYER": "omp"}},
            "halide_auto": {"backend": "halide", "prompt": "prompts/lang/halide_auto.md", "threads": "measurement",
                            "halide_mode": "auto", "env": {"HL_NUM_THREADS": "${threads}"}},
            "halide_manual": {"backend": "halide", "prompt": "prompts/lang/halide_manual.md", "threads": "measurement",
                              "halide_mode": "manual", "env": {"HL_NUM_THREADS": "${threads}"}},
            "rust": {"backend": "rust", "prompt": "prompts/lang/rust.md", "threads": "measurement",
                     "env": {"RAYON_NUM_THREADS": "${threads}"}},
        },
        "build": {"cc": "gcc", "cflags": ["-O3", "-march=native"], "base_flags": ["-std=gnu11", "-fPIC", "-shared"],
                  "openmp_flag": "-fopenmp", "ldflags": ["-lm"], "diag_flags": [], "compile_timeout_s": 60,
                  "forbidden_symbols": ["system"], "thread_symbols_prefix": ["GOMP_"],
                  "rust": {"cargo": "cargo", "rustflags": ["-C", "target-cpu=native"], "build_timeout_s": 600, "offline": True}},
        "halide": {"autoscheduler": "Mullapudi2016", "autoscheduler_params": {"last_level_cache_size": "1048576"}},
        "safety": {"enabled": False, "timeout_s": 300},
        "lint": {"enabled": True, "plugin": "plugins/lint/run_lint.py", "timeout_s": 60},
        "perf_summary": {"enabled": True},
        "measurement": {
            "cores": "0,2,4,6", "single_core": "2", "threads": 4, "omp_env_extra": {"OMP_PROC_BIND": "true"}, "warmup": 3,
            "repeats": 10, "min_repeats": 3, "check_repeats": 2, "check_timeout_s": 180, "bench_timeout_s": 300,
            "max_worker_mem_gb": 8, "max_load_1m": 2.0, "max_pinned_busy_pct": 10.0, "require_performance_governor": True,
            "baseline_min_gflops": 30, "baseline_drift_pct_max": 10, "correctness_tests": ["tiny", "rect", "small"],
            "performance_test": "large",
        },
        "paths": {"results_csv": str(root / "runs.csv"), "results_csv_dummy": str(root / "runs_dummy.csv"),
                  "raw_dir": str(root / "raw"), "built_dir": str(root / "built")},
    }
    path = root / "experiment.yaml"
    path.write_text(yaml.safe_dump(doc), encoding="utf-8")
    return path


def write_tasks(root: Path) -> Path:
    """테스트용 과제 디렉터리: ``gemm``, halide_auto 를 지원하지 않는 ``nohalide``, C 기준선을 쓰는 ``cbase``."""
    tasks = root / "tasks"
    variants = {
        "gemm": {},
        "nohalide": {"unsupported": {"halide_auto": "not expressible as a Halide pipeline"}},
        "cbase": {"baseline": {"backend": "c", "file": "baseline/baseline.c", "openmp": True}},
    }
    for name, over in variants.items():
        doc = {**TASK_YAML, "task": name, **over}
        (tasks / name / "baseline").mkdir(parents=True)
        (tasks / name / "signatures.yaml").write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        (tasks / name / "tolerance.yaml").write_text("task: x\ndtype: float32\nrtol: 1.0e-3\natol: 1.0e-3\nsentinel: -2048.0\n",
                                                     encoding="utf-8")
        (tasks / name / "baseline" / Path(str(doc["baseline"]["file"])).name).write_text("# baseline\n", encoding="utf-8")
    return tasks


def raw_doc(language: str, sample_idx: int = 0, turn: int = 0, *, task: str = "gemm", **over: Any) -> dict[str, Any]:
    """generate·repair 가 쓰는 raw 문서의 꼴(설계 §5, v3 §9.3: 턴 ≥ 1 도 같은 키)."""
    doc: dict[str, Any] = {
        "run_id": RUN_ID, "task": task, "language": language, "sample_idx": sample_idx, "turn": turn,
        "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"},
        "provider": "dummy", "model_requested": "dummy-model", "model_served": "dummy", "max_tokens": 1000,
        "sampling": {"temperature_sent": None, "effort": None, "seed": None}, "prompt_hash": "0123456789abcdef",
        "response_text": "```c\n" + CODE + "```\n", "stop_reason": "end_turn", "stop_details_category": None,
        "tokens_in": 10, "tokens_out": 20, "tokens_thinking": None, "tokens_text": None, "n_thinking_blocks": 0,
        "latency_ms": 0.0, "sdk_version": None, "error": None,
    }
    doc.update(over)
    return doc


def write_raw(cfg: Config, doc: dict[str, Any]) -> Path:
    cond = doc["conditions"]
    tag = f"args-{cond['args']}_hint-{cond['grammar_hint']}_fb-{cond['feedback']}"
    path = raw_turn_path(cfg.paths.raw_dir, RUN_ID, doc["task"], doc["language"], tag, doc["sample_idx"], doc["turn"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def result_doc(raw_path: Path, row: dict[str, Any] | None = None, **over: Any) -> dict[str, Any]:
    """그 raw 로 쓴 result.json 의 꼴(``finalize_row``): 기본은 모든 선택 단계(bench·safety·lint)를 요청한 처리의 결과."""
    return {"raw_sha256": sha256_file(raw_path), "row": dict(row or {}), "stages": {},
            "requested": {"bench": True, "safety": True, "lint": True}, **over}


def _proc(rc: int, *, timed_out: bool = False, signal_name: str | None = None) -> ProcResult:
    return ProcResult(rc=rc, timed_out=timed_out, elapsed_s=0.1, stdout_tail="", stderr_tail="stderr tail", signal_name=signal_name)


def _outcome(rc: int, doc: dict | None, **kw) -> WorkerOutcome:
    return WorkerOutcome(stage="check", proc=_proc(rc, **kw), doc=doc, out_path=Path("/nonexistent/check.json"), stdout_bytes=0, timeout_s=180.0)


@dataclass
class Exit:
    """가짜 워커의 끝: 결과 문서와 프로세스 상태."""

    doc: dict[str, Any] | None
    rc: int = 0
    timed_out: bool = False
    signal_name: str | None = None
    stdout_bytes: int = 0


def lint_result(*findings: tuple[str, int], tool: str = "baseline:ruff+pyflakes", ok: bool = True, error: str = "") -> LintResult:
    """(범주, 줄) 목록으로 만든 린트 결과."""
    items = [{"rule": f"ruff/R{i}", "severity": "warning", "line": line, "message": f"finding {i}", "category": category}
             for i, (category, line) in enumerate(findings)]
    summary = {category: sum(1 for f in items if f["category"] == category) for category in LINT_CATEGORIES}
    return LintResult(tool=tool, findings=items, summary=summary, ok=ok, error=error)


def sanitize_result(asan: int | None = 0, ubsan: int | None = 0, kinds: tuple[str, ...] = (), detail: str = "") -> SanitizeResult:
    return SanitizeResult(ran=True, build_ok=True, asan_findings=asan, ubsan_findings=ubsan, kinds=list(kinds), detail=detail,
                          summary={"log_files": []})


class Fakes:
    """``process_sample`` 이 부르는 바깥 것들의 대역. 부른 순서는 ``events``, 인자는 ``*_calls`` 에 남는다.

    저장소에 있는 함수는 ``autospec`` 으로 바꿔 실제 시그니처와 맞지 않는 호출이 테스트를 실패시키게 한다. numba·halide·rust
    빌드 모듈은 ``sys.modules`` 에 대역 모듈을 넣는다(설계 v3 §6 의 시그니처 그대로; run.py 는 쓰는 자리에서 임포트한다).
    """

    def __init__(self, case: unittest.TestCase) -> None:
        self.events: list[str] = []
        self.extract_calls: list[dict[str, Any]] = []
        self.build_calls: list[dict[str, Any]] = []
        self.worker_calls: list[dict[str, Any]] = []
        self.lint_calls: list[dict[str, Any]] = []
        self.summary_calls: list[dict[str, Any]] = []
        self.sanitize_calls: list[dict[str, Any]] = []
        self.extraction = SimpleNamespace(ok=True, code=CODE, signature_ok=True, signature_exact=True, detail="", n_blocks=1)
        self.builds: dict[str, BuildResult | Exception] = {}  # 백엔드 → 정해 둔 결과(없으면 성공)
        self.exits: dict[str, Exit] = {"import": Exit(IMPORT_OK), "check": Exit(CHECK_PASS), "bench": Exit(BENCH_PASS),
                                       "paired_baseline": Exit(PAIRED_OK), "safety": Exit(CHECK_PASS)}
        self.lint = lint_result()
        self.summary: dict[str, Any] = {"language": "x", "loops": [], "notes": "3 loops found"}
        self.sanitize = sanitize_result()
        self.sanitize_runs_worker = False
        self.rust_probes: list[dict[str, Any]] = []
        self.rust_probe_error: Exception | None = None

        modules = {"harness.build.numba": SimpleNamespace(build_numba=self.build_numba),
                   "harness.build.halide": SimpleNamespace(build_halide=self.build_halide),
                   "harness.build.rust": SimpleNamespace(build_rust=self.build_rust, probe_toolchain=self.probe_toolchain)}
        for patcher in (
            mock.patch("harness.extract.extract", autospec=True, side_effect=self.extract),
            mock.patch.object(run, "build_c", autospec=True, side_effect=self.build_c),
            mock.patch.object(run, "build_python", autospec=True, side_effect=self.build_python),
            mock.patch.dict(sys.modules, modules),
            mock.patch.object(run, "run_worker", autospec=True, side_effect=self.run_worker),
            mock.patch.object(run.safety, "run_lint", autospec=True, side_effect=self.run_lint),
            mock.patch.object(run.safety, "run_sanitizers", autospec=True, side_effect=self.run_sanitizers),
            mock.patch.object(run.perf_summary, "summarize", autospec=True, side_effect=self.summarize),
            mock.patch.object(run.protocol, "check_dynamic", autospec=True, side_effect=self.check_dynamic),
            mock.patch.object(run.protocol, "launch_prefix", autospec=True, return_value=list(PREFIX)),
        ):
            patcher.start()
            case.addCleanup(patcher.stop)

    # ------------------------------------------------------------------ 추출·빌드

    def extract(self, response_text: str, **kwargs: Any) -> SimpleNamespace:
        self.events.append("extract")
        self.extract_calls.append({"response_text": response_text, **kwargs})
        return self.extraction

    def _build(self, backend: str, source: str, code: str, out_dir: Path, compiler: tuple[str, str], kwargs: dict[str, Any],
               extra: dict[str, Any] | None = None, files: dict[str, str] | None = None) -> BuildResult:
        self.events.append(f"build:{backend}")
        self.build_calls.append({"backend": backend, "code": code, "out_dir": Path(out_dir), **kwargs})
        planned = self.builds.get(backend)
        if isinstance(planned, Exception):
            raise planned
        for name, text in {source: code, **(files or {})}.items():  # 실제 빌드 함수처럼 소스를 먼저 쓴다(실패해도 남는다)
            (Path(out_dir) / name).parent.mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / name).write_text(text, encoding="utf-8")
        if planned is not None:
            return planned
        return BuildResult(ok=True, artifact=Path(out_dir) / "artifact.bin", error_kind="", error_detail="", log="tail only",
                           build_time_ms=12.5, command=["build"], compiler=compiler[0], compiler_version=compiler[1],
                           n_warnings=1, march_resolved="-march=x" if backend == "c" else "", extra=dict(extra or {}))

    def build_c(self, code: str, out_dir: Path, bcfg: Any, **kwargs: Any) -> BuildResult:
        return self._build("c", "kernel.c", code, out_dir, ("gcc", "gcc 15.2.0"), {"bcfg": bcfg, **kwargs},
                           files={"build.log": BUILD_LOG_TEXT})

    def build_python(self, code: str, out_dir: Path) -> BuildResult:
        return self._build("python", "kernel.py", code, out_dir, ("python", "3.14.0"), {})

    def build_numba(self, code: str, out_dir: Path, *, function: str, expected_signature: str) -> BuildResult:
        return self._build("numba", "kernel.py", code, out_dir, ("numba", "0.68.1"),
                           {"function": function, "expected_signature": expected_signature},
                           extra={"decorator": "njit", "options": {"parallel": True}, "uses_prange": True})

    def build_halide(self, code: str, out_dir: Path, *, function: str, halide_mode: str) -> BuildResult:
        return self._build("halide", "kernel.py", code, out_dir, ("halide", "22.1.0"),
                           {"function": function, "halide_mode": halide_mode}, extra={"schedule_calls": [], "imports_halide": True})

    def build_rust(self, code: str, out_dir: Path, rcfg: Any, *, launch_prefix: list[str], symbol: str,
                   args_module: str | None = None, target_dir: Path | None = None) -> BuildResult:
        return self._build("rust", "src/lib.rs", code, out_dir, ("cargo", "cargo 1.92.0"),
                           {"rcfg": rcfg, "launch_prefix": launch_prefix, "symbol": symbol, "args_module": args_module,
                            "target_dir": target_dir},
                           files={"Cargo.toml": CARGO_TOML, "build.log": "   Compiling kernel v0.1.0\n"})

    def probe_toolchain(self, rcfg: Any, target_dir: Path, *, launch_prefix: list[str]) -> None:
        self.events.append("probe:rust")
        self.rust_probes.append({"rcfg": rcfg, "target_dir": Path(target_dir), "launch_prefix": launch_prefix})
        if self.rust_probe_error is not None:
            raise self.rust_probe_error

    # ------------------------------------------------------------------ 워커·선택 단계

    def run_worker(self, cfg: Any, **kwargs: Any) -> WorkerOutcome:
        stage = kwargs["stage"]
        self.events.append(f"worker:{stage}")
        self.worker_calls.append(kwargs)
        exit_ = self.exits[stage]
        proc = ProcResult(rc=exit_.rc, timed_out=exit_.timed_out, elapsed_s=0.2, stdout_tail="", stderr_tail="worker stderr",
                          signal_name=exit_.signal_name)
        return WorkerOutcome(stage=stage, proc=proc, doc=exit_.doc, out_path=kwargs["out_path"],
                             stdout_bytes=exit_.stdout_bytes, timeout_s=kwargs["timeout_s"])

    def run_lint(self, cfg: Any, **kwargs: Any) -> LintResult:
        self.events.append("lint")
        self.lint_calls.append({**kwargs, "file_existed": Path(kwargs["file"]).is_file()})
        return self.lint

    def summarize(self, **kwargs: Any) -> dict[str, Any]:
        self.events.append("perf_summary")
        self.summary_calls.append(kwargs)
        return dict(self.summary)

    def run_sanitizers(self, cfg: Any, **kwargs: Any) -> SanitizeResult:
        self.events.append("sanitize")
        self.sanitize_calls.append(kwargs)
        if self.sanitize_runs_worker:  # 실제 run_sanitizers 가 넘겨받은 worker 를 부르는 방식 그대로(설계 v3 §12)
            kwargs["worker"]("safety", "check", kwargs["tests"], 300.0, ["--check-repeats", "1"],
                             artifact=Path(kwargs["sample_dir"]) / "san" / "kernel_san.so",
                             env_extra={"LD_PRELOAD": "/usr/lib/libasan.so", "ASAN_OPTIONS": "detect_leaks=0"}, max_mem_gb=0)
        return self.sanitize

    def check_dynamic(self, cfg: Any, **kwargs: Any) -> DynamicStatus:
        self.events.append("dynamic")
        return DYNAMIC

    def calls(self, stage: str) -> list[dict[str, Any]]:
        return [c for c in self.worker_calls if c["stage"] == stage]


class PipelineCase(unittest.TestCase):
    """임시 디렉터리의 설정·과제와 대역 위에서 ``process_sample`` 을 돌리는 공통 준비."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="run_test_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.config_path = write_config(self.root)
        self.cfg = load_config(self.config_path)
        self.tasks_dir = write_tasks(self.root)
        patcher = mock.patch.object(run, "TASKS_DIR", self.tasks_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.fakes = Fakes(self)
        self.spec = load_taskspec(self.tasks_dir / "gemm" / "signatures.yaml")
        self.tctx = TaskContext(task="gemm", test_dir=self.root / "tests", manifest=MANIFEST,
                                tolerance_path=self.tasks_dir / "gemm" / "tolerance.yaml",
                                signatures=yaml.safe_load((self.tasks_dir / "gemm" / "signatures.yaml").read_text(encoding="utf-8")),
                                spec=self.spec, baseline_run={"median_ms": 10.0}, baseline_1t={"median_ms": 30.0})
        self.log = ""

    def context(self, *, cfg: Config | None = None, skip_bench: bool = True, force: bool = False) -> RunContext:
        return RunContext(cfg=cfg or self.cfg, machine={"hostname": "testhost", "cpu_model": "Test CPU"}, static=STATIC,
                          fingerprint={"python_version": "3.14.0", "numpy_version": "2.3.5"}, env_hash="envhash", run_id=RUN_ID,
                          skip_bench=skip_bench, force=force, version="v-test")

    def process(self, raw: dict[str, Any] | Path, **ctx_kwargs: Any) -> tuple[Any, str]:
        """raw 하나를 처리한다(문서를 주면 먼저 파일로 쓴다). stderr 는 ``self.log`` 에 모은다."""
        raw_path = raw if isinstance(raw, Path) else write_raw(self.cfg, raw)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                return process_sample(self.context(**ctx_kwargs), self.tctx, raw_path)
            finally:
                self.log += err.getvalue()

    def sample_dir(self, language: str, sample_idx: int = 0, tag: str = TAG) -> Path:
        return self.cfg.paths.built_dir / RUN_ID / "gemm" / language / tag / f"sample_{sample_idx}"

    def csv_rows(self) -> list[dict[str, str]]:
        path = self.cfg.paths.results_csv_dummy
        return read_rows(path) if path.exists() else []

    def stages(self, sample_dir: Path) -> dict[str, Any]:
        return json.loads((sample_dir / "result.json").read_text(encoding="utf-8"))["stages"]


# ----------------------------------------------------------------------------- 종료 규칙·건너뜀·raw 탐색


class VerdictTest(unittest.TestCase):
    """설계 §8-6: harness_error 는 worker_error 가 있거나 결과 문서를 쓰지 못하고 끝났을 때만."""

    def test_worker_error_in_doc_is_harness_error(self):
        with self.assertRaises(HarnessError):
            _outcome(3, {"worker_error": "Traceback ...", "complete": False}).verdict()

    def test_rc3_without_any_doc_is_harness_error(self):
        with self.assertRaises(HarnessError):
            _outcome(3, None).verdict()
        with self.assertRaises(HarnessError):
            _outcome(3, {}).verdict()

    def test_any_exit_without_a_result_document_is_harness_error(self):
        """워커는 커널을 로드하기 전에 첫 문서를 쓴다. 문서가 없으면 LLM 코드는 돌지 않았다 — 인자 오류(rc=2), 임포트 실패(rc=1),
        시작 전의 시그널·시간 초과 어느 것도 커널의 runtime_crash 가 아니다."""
        for outcome in (_outcome(2, None), _outcome(1, None), _outcome(0, None), _outcome(-9, None, signal_name="SIGKILL"),
                        _outcome(-9, None, timed_out=True, signal_name="SIGKILL")):
            with self.assertRaises(HarnessError) as caught:
                outcome.verdict()
            self.assertIn("stderr tail", str(caught.exception))

    def test_rc3_with_partial_doc_is_kernel_exit_not_harness_error(self):
        # LLM 커널이 exit(3) 을 부른 경우: 워커는 in_progress 까지 썼지만 worker_error 는 없다 → runtime_crash
        doc = {"complete": False, "in_progress": {"test": "tiny", "call_idx": 0}}
        kind, detail = _outcome(3, doc).verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("rc=3", detail)

    def test_signal_is_runtime_crash_with_location(self):
        doc = {"complete": False, "in_progress": {"test": "rect", "call_idx": 1}}
        kind, detail = _outcome(-11, doc, signal_name="SIGSEGV").verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("SIGSEGV", detail)
        self.assertIn("rect call 1", detail)

    def test_timeout(self):
        kind, _ = _outcome(-9, {"complete": False}, timed_out=True, signal_name="SIGKILL").verdict()
        self.assertEqual(kind, "timeout")

    def test_rc0_incomplete_is_runtime_crash(self):
        kind, detail = _outcome(0, {"complete": False}).verdict()
        self.assertEqual(kind, "runtime_crash")
        self.assertIn("complete", detail)

    def test_complete_is_ok(self):
        self.assertIsNone(_outcome(0, {"complete": True, "classification": "pass"}).verdict())


class ApplyCheckDocTest(unittest.TestCase):
    def test_partial_check_document_leaves_unchecked_flags_na(self) -> None:
        """호출을 하나도 검사하기 전에 끝난 check 문서(in_progress 만 있고 runs 없음; inputs_mutated·oob_write 는 None)는
        행의 inputs_mutated·oob_write 를 None 으로 둔다 — CSV 에서 NA (검사하지 않은 것을 '아니다'로 적지 않는다)."""
        row = run.RunRow(run_id=RUN_ID, task="gemm", language="c_seq", sample_idx=0, turn=0)
        partial = {"mode": "check", "complete": False, "in_progress": {"test": "tiny", "call_idx": 0}, "tests": [],
                   "inputs_mutated": None, "oob_write": None, "classification": None}
        run.apply_check_doc(row, partial)
        self.assertEqual((row.inputs_mutated, row.oob_write, row.check_tests), (None, None, []))
        csv_row = row.to_csv_dict()
        self.assertEqual((csv_row["inputs_mutated"], csv_row["oob_write"]), ("", ""))
        run.apply_check_doc(row, {**partial, "inputs_mutated": False, "oob_write": True})
        self.assertEqual((row.inputs_mutated, row.oob_write), (False, True))


class NeedsProcessingTest(unittest.TestCase):
    """기준선 재측정 여부를 정하는 사전 평가가 process_sample 의 건너뜀 조건과 같은 답을 내야 한다."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        cfg = load_config()
        self.cfg = replace(cfg, paths=replace(cfg.paths, built_dir=root / "built", raw_dir=root / "raw"))
        self.raw = root / "raw" / "r1" / "gemm" / "numpy" / "args-positional_hint-off_fb-compiler" / "sample_0.json"
        self.raw.parent.mkdir(parents=True)
        self.raw.write_text(json.dumps({"language": "numpy", "sample_idx": 0, "task": "gemm",
                                        "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"}}))

    def _ctx(self, force: bool = False, *, skip_bench: bool = False, lint: bool = True, safety: bool = False) -> RunContext:
        cfg = replace(self.cfg, lint=replace(self.cfg.lint, enabled=lint), safety=replace(self.cfg.safety, enabled=safety))
        return RunContext(cfg=cfg, machine={}, static=None, fingerprint={}, env_hash="x", run_id="r1",
                          skip_bench=skip_bench, force=force, version="test")

    def _write_result(self, row: dict[str, Any], **over: Any) -> None:
        rp = self._result_path()
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(json.dumps(result_doc(self.raw, row, **over)))

    def test_result_written_without_a_requested_stage_is_stale(self):
        """회귀: ``--skip-bench`` 로 쓴 통과 행이 bench 를 요청한 run 에서도 최신으로 보여 bench 없이 끝났다. 저장된 결과에 없던
        bench 를 이 run 이 요청하고 그 행에서 bench 가 돌았을 것이면 다시 처리한다. safety·lint 만 빠진 결과는 다시 처리하지
        않는다(다시 처리하면 이미 잰 시간까지 다시 잰다): ``missing_stages`` 가 알려 주고 ``--force`` 에 맡긴다."""
        passed = {"passed": True, "bench_ran": False, "backend": "python", "error_stage": ""}
        self._write_result(passed, requested={"bench": False, "safety": False, "lint": True})
        self.assertFalse(needs_processing(self._ctx(skip_bench=True), "gemm", self.raw), "--skip-bench 끼리는 최신")
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw), "bench 를 요청하면 다시 처리")
        # 실패한 행은 bench 가 돌지 않으므로 --skip-bench 로 썼어도 최신이다
        self._write_result({**passed, "passed": False, "error_stage": "check", "error_type": "wrong_output"},
                           requested={"bench": False, "safety": False, "lint": True})
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))
        # lint 를 켜기 전에 쓴 행: 빠진 단계로 알려지지만 다시 처리하지는 않는다(extract 에서 실패한 행은 린트할 것이 없다)
        self._write_result({**passed, "bench_ran": True}, requested={"bench": True, "safety": False, "lint": False})
        stored = json.loads(self._result_path().read_text())
        self.assertEqual(missing_stages(self._ctx(), stored), ["lint"])
        self.assertEqual(missing_stages(self._ctx(lint=False), stored), [])
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw), "lint 만 빠진 결과는 다시 측정하지 않는다")
        self._write_result({"passed": False, "error_stage": "extract", "backend": "python"},
                           requested={"bench": True, "safety": False, "lint": False})
        self.assertEqual(missing_stages(self._ctx(), json.loads(self._result_path().read_text())), [])
        # safety 는 check 까지 간 C 행에만 돈다
        c_row = {"passed": False, "bench_ran": False, "backend": "c", "error_stage": "check", "error_type": "wrong_output"}
        self._write_result(c_row, requested={"bench": True, "safety": False, "lint": True})
        stored = json.loads(self._result_path().read_text())
        self.assertEqual(missing_stages(self._ctx(safety=True), stored), ["safety"])
        self.assertEqual(missing_stages(self._ctx(), stored), [])
        self.assertFalse(needs_processing(self._ctx(safety=True), "gemm", self.raw), "safety 만 빠진 결과는 다시 측정하지 않는다")
        for not_reached in ({**c_row, "error_stage": "build", "error_type": "compile_error"}, {**c_row, "error_type": "timeout"},
                            {**c_row, "backend": "python"}):
            self._write_result(not_reached, requested={"bench": True, "safety": False, "lint": True})
            self.assertEqual(missing_stages(self._ctx(safety=True), json.loads(self._result_path().read_text())), [], not_reached)

    def test_old_result_without_requested_stages_is_judged_by_its_row(self):
        """``requested`` 가 없는 예전 result.json: 행과 단계 기록이 보여 주는 것으로 추정한다 — 완전한 행은 다시 처리하지 않는다."""
        complete = {"passed": True, "bench_ran": True, "backend": "c", "error_stage": "", "lint_findings_total": 0,
                    "asan_findings": 0, "ubsan_findings": 0}
        self._write_result(complete, requested=None)
        self.assertFalse(needs_processing(self._ctx(safety=True), "gemm", self.raw))
        # 린트가 돌았지만 열이 NA 인 행(도구 없음)도 단계 기록(stages.lint)이 있으면 돈 것이다
        self._write_result({**complete, "lint_findings_total": None}, requested=None, stages={"lint": {"ok": True}})
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))
        self._write_result({**complete, "bench_ran": False}, requested=None)
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw), "통과했는데 bench 가 없던 행")
        self.assertFalse(needs_processing(self._ctx(skip_bench=True), "gemm", self.raw))
        self._write_result({**complete, "asan_findings": None, "ubsan_findings": None}, requested=None)
        stored = json.loads(self._result_path().read_text())
        self.assertEqual(missing_stages(self._ctx(safety=True), stored), ["safety"], "sanitizer 가 돌지 않은 C 행")
        self.assertFalse(needs_processing(self._ctx(safety=True), "gemm", self.raw), "알리기만 하고 다시 측정하지 않는다")
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))

    def _result_path(self) -> Path:
        return self.cfg.paths.built_dir / "r1" / "gemm" / "numpy" / "args-positional_hint-off_fb-compiler" / "sample_0" / "result.json"

    def test_no_result_json_needs_processing(self):
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertFalse(self._result_path().parent.exists(), "사전 평가는 디렉터리를 만들지 않는다")

    def test_matching_result_json_skips(self):
        rp = self._result_path(); rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps(result_doc(self.raw)))
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertTrue(needs_processing(self._ctx(force=True), "gemm", self.raw), "--force 는 항상 처리")

    def test_changed_raw_needs_processing(self):
        rp = self._result_path(); rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps(result_doc(self.raw)))
        self.raw.write_text(self.raw.read_text() + " ")
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw))

    def test_unknown_language_is_not_pending(self):
        self.raw.write_text(json.dumps({"language": "fortran", "sample_idx": 0, "conditions": {}}))
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))

    def test_each_turn_has_its_own_result(self):
        """수정 턴의 raw 는 ``sample_<i>/turn_<t>/result.json`` 을 본다: 턴 0 이 최신이어도 턴 1 은 따로 판단한다."""
        turn1 = self.raw.with_name("sample_0.turn1.json")
        turn1.write_text(json.dumps({"language": "numpy", "sample_idx": 0, "turn": 1, "task": "gemm",
                                     "conditions": {"args": "positional", "grammar_hint": "off", "feedback": "compiler"}}))
        rp = self._result_path(); rp.parent.mkdir(parents=True)
        rp.write_text(json.dumps(result_doc(self.raw)))
        self.assertFalse(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertTrue(needs_processing(self._ctx(), "gemm", turn1))
        turn_result = rp.parent / "turn_1" / "result.json"
        self.assertFalse(turn_result.parent.exists(), "사전 평가는 디렉터리를 만들지 않는다")
        turn_result.parent.mkdir()
        turn_result.write_text(json.dumps(result_doc(turn1)))
        self.assertFalse(needs_processing(self._ctx(), "gemm", turn1))
        self.raw.write_text(self.raw.read_text() + " ")  # 턴 0 의 raw 만 바뀜
        self.assertTrue(needs_processing(self._ctx(), "gemm", self.raw))
        self.assertFalse(needs_processing(self._ctx(), "gemm", turn1))


class FindRawFilesTest(unittest.TestCase):
    """raw 탐색: 턴 파일 포함, 보관본 제외, 과제·언어·조건 태그·샘플·턴 순."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="run_find_")
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.cfg = load_config(write_config(self.root))
        self.base = self.cfg.paths.raw_dir / RUN_ID
        other_tag = "args-keyword_hint-off_fb-compiler"
        names = [
            ("gemm", "numpy", TAG, "sample_0.json"), ("gemm", "numpy", TAG, "sample_0.turn1.json"),
            ("gemm", "numpy", TAG, "sample_0.turn2.json"), ("gemm", "numpy", TAG, "sample_0.turn10.json"),
            ("gemm", "numpy", TAG, "sample_2.json"), ("gemm", "numpy", TAG, "sample_10.json"),
            ("gemm", "numpy", TAG, "sample_10.turn1.json"), ("gemm", "numpy", other_tag, "sample_1.json"),
            ("gemm", "c_seq", TAG, "sample_1.json"), ("cbase", "numpy", TAG, "sample_0.json"),
            # 처리 대상이 아닌 것들
            ("gemm", "numpy", TAG, "sample_0.prev1.json"), ("gemm", "numpy", TAG, "sample_0.turn1.prev1.json"),
            ("gemm", "numpy", TAG, "sample_x.json"), ("gemm", "numpy", TAG, "notes.json"),
            ("stranger", "numpy", TAG, "sample_0.json"),
        ]
        for task, language, tag, name in names:
            path = self.base / task / language / tag / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{}", encoding="utf-8")

    def found(self, *args: Any, **kwargs: Any) -> list[str]:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            files = find_raw_files(self.cfg, RUN_ID, *args, **kwargs)
        self.log = err.getvalue()
        return [p.relative_to(self.base).as_posix() for p in files]

    def test_order_is_task_language_tag_sample_turn(self) -> None:
        keyword = "args-keyword_hint-off_fb-compiler"
        self.assertEqual(self.found(), [
            f"cbase/numpy/{TAG}/sample_0.json",
            f"gemm/c_seq/{TAG}/sample_1.json",
            f"gemm/numpy/{keyword}/sample_1.json",
            f"gemm/numpy/{TAG}/sample_0.json",
            f"gemm/numpy/{TAG}/sample_0.turn1.json",
            f"gemm/numpy/{TAG}/sample_0.turn2.json",
            f"gemm/numpy/{TAG}/sample_0.turn10.json",  # 턴은 수로 정렬한다(문자열이면 turn10 < turn2)
            f"gemm/numpy/{TAG}/sample_2.json",
            f"gemm/numpy/{TAG}/sample_10.json",  # 샘플 번호도 수로 정렬한다
            f"gemm/numpy/{TAG}/sample_10.turn1.json",
        ])
        self.assertIn("stranger", self.log, "설정 run.tasks 에 없는 과제는 경고하고 뺀다")

    def test_filters(self) -> None:
        self.assertEqual(self.found({"c_seq"}), [f"gemm/c_seq/{TAG}/sample_1.json"])
        self.assertEqual(self.found(None, {10}), [f"gemm/numpy/{TAG}/sample_10.json", f"gemm/numpy/{TAG}/sample_10.turn1.json"])
        self.assertEqual(self.found(tasks=["cbase"]), [f"cbase/numpy/{TAG}/sample_0.json"])
        self.assertEqual(self.found({"numpy"}, {0}, tasks=["gemm"]),
                         [f"gemm/numpy/{TAG}/sample_0.json", f"gemm/numpy/{TAG}/sample_0.turn1.json",
                          f"gemm/numpy/{TAG}/sample_0.turn2.json", f"gemm/numpy/{TAG}/sample_0.turn10.json"])
        self.assertEqual(self.found({"rust"}), [])

    def test_missing_run_directory(self) -> None:
        self.assertEqual(find_raw_files(self.cfg, "no-such-run"), [])


# ----------------------------------------------------------------------------- 워커 명령


class RunWorkerTest(PipelineCase):
    """``run_worker`` 가 만드는 명령과 환경(설계 v3 §10.3). 대역을 걷어 내고 ``run_isolated`` 만 가로챈다."""

    def launch(self, **over: Any) -> tuple[list[str], dict[str, str], WorkerOutcome]:
        seen: dict[str, Any] = {}

        def isolated(cmd: list[str], **kwargs: Any) -> ProcResult:
            seen.update(cmd=list(cmd), **kwargs)
            Path(kwargs["stdout_path"]).write_text("kernel said hi", encoding="utf-8")
            return _proc(0)

        kwargs: dict[str, Any] = dict(stage="check", threads=4, openmp=False, backend="python", artifact=self.root / "kernel.py",
                                      mode="check", tests=[self.root / "tiny.npz", self.root / "rect.npz"],
                                      tolerance=self.root / "tolerance.yaml", signatures=self.root / "signatures.yaml",
                                      out_path=self.root / "work" / "check.json", cwd=self.root / "work", timeout_s=180.0)
        kwargs.update(over)
        with mock.patch.object(run, "run_isolated", side_effect=isolated), mock.patch.dict("os.environ", {"PATH": "/usr/bin"}, clear=True):
            outcome = run_worker(self.cfg, **kwargs)  # 이 모듈이 임포트해 둔 실제 함수(대역은 run.run_worker 만 바꾼다)
        return seen["cmd"], seen["env"], outcome

    def test_default_command(self) -> None:
        cmd, env, outcome = self.launch()
        self.assertEqual(cmd[:3], PREFIX)
        self.assertEqual(cmd[4:6], ["-m", "harness.worker"])
        tail = cmd[6:]
        self.assertEqual(tail[:4], ["--backend", "python", "--artifact", str(self.root / "kernel.py")])
        self.assertEqual(tail[tail.index("--max-mem-gb") + 1], "8.0")
        self.assertEqual(tail[tail.index("--tests") + 1], f"{self.root / 'tiny.npz'},{self.root / 'rect.npz'}")
        self.assertEqual(tail[tail.index("--args-mode") + 1], "positional")
        self.assertEqual(env["OMP_NUM_THREADS"], "4")
        self.assertNotIn("OMP_PROC_BIND", env)
        self.assertEqual(outcome.stdout_bytes, len("kernel said hi"))
        self.assertIsNone(outcome.doc)

    def test_backend_args_env_and_memory_limit(self) -> None:
        cmd, env, _ = self.launch(
            backend="halide", mode="import", tests=[], openmp=True, args_mode="keyword",
            backend_args=["--halide-mode", "auto", "--threads", "4"], extra=["--inspect"],
            env_extra={"HL_NUM_THREADS": "4", "OMP_PROC_BIND": "close"}, max_mem_gb=0)
        self.assertNotIn("--tests", cmd)
        self.assertEqual(cmd[-5:], ["--halide-mode", "auto", "--threads", "4", "--inspect"])
        self.assertEqual(cmd[cmd.index("--max-mem-gb") + 1], "0")
        self.assertEqual(cmd[cmd.index("--args-mode") + 1], "keyword")
        self.assertEqual(env["HL_NUM_THREADS"], "4")
        self.assertEqual(env["OMP_PROC_BIND"], "close", "env_extra 는 launch_env 가 만든 값 위에 덮어쓴다")

    def test_stale_result_is_removed_before_launch(self) -> None:
        out = self.root / "work" / "check.json"
        out.parent.mkdir()
        out.write_text(json.dumps({"complete": True, "classification": "pass"}), encoding="utf-8")
        _, _, outcome = self.launch()
        self.assertIsNone(outcome.doc, "이전 실행의 결과 문서를 새 결과로 읽지 않는다")

    def test_backend_args_per_language(self) -> None:
        langs = self.cfg.languages
        for name in ("numpy", "c_seq", "c_openmp", "numba", "rust"):
            self.assertEqual(worker_backend_args(self.cfg, langs[name]), [], name)
        self.assertEqual(worker_backend_args(self.cfg, langs["halide_manual"]), ["--halide-mode", "manual", "--threads", "4"])
        self.assertEqual(worker_backend_args(self.cfg, langs["halide_auto"]),
                         ["--halide-mode", "auto", "--threads", "4", "--halide-autoscheduler", "Mullapudi2016",
                          "--halide-params", '{"last_level_cache_size": "1048576"}'])
        self.assertEqual(run.WORKER_BACKEND, {"c": "c", "rust": "c", "python": "python", "numba": "numba", "halide": "halide"})

    def test_inspect_dims_come_from_the_performance_shape(self) -> None:
        self.assertEqual(performance_dims(self.cfg, self.tctx), {"M": 1024, "N": 512, "K": 256})
        self.assertEqual(inspect_args(self.cfg, self.tctx), ["--inspect", "--inspect-dims", '{"M": 1024, "N": 512, "K": 256}'])
        for manifest in ({}, {"files": {}}, {"files": {"large": {"M": 1, "N": 2}}}, {"files": {"tiny": MANIFEST["files"]["tiny"]}}):
            tctx = replace(self.tctx, manifest=manifest)
            self.assertIsNone(performance_dims(self.cfg, tctx), manifest)
            self.assertEqual(inspect_args(self.cfg, tctx), ["--inspect"])


# ----------------------------------------------------------------------------- 백엔드별 단계 순서와 행


class BackendPipelineTest(PipelineCase):
    """언어마다: 단계 순서, 빌드 함수·워커에 넘기는 인자, 행의 백엔드 관련 열 (설계 v3 §10.2)."""

    FULL = ["extract", "build:{}", "lint", "worker:import", "perf_summary", "worker:check", "dynamic",
            "worker:paired_baseline", "worker:bench"]

    def run_language(self, language: str, **raw_over: Any) -> Any:
        row, status = self.process(raw_doc(language, **raw_over), skip_bench=False)
        self.assertEqual(status, "processed")
        return row

    def assert_sequence(self, backend: str) -> None:
        self.assertEqual(self.fakes.events, [e.format(backend) for e in self.FULL])

    def assert_passing_row(self, row: Any, language: str, backend: str) -> None:
        self.assertEqual((row.run_id, row.task, row.language, row.backend), (RUN_ID, "gemm", language, backend))
        self.assertEqual((row.args, row.grammar_hint, row.feedback, row.sample_idx, row.turn),
                         ("positional", "off", "compiler", 0, 0))
        self.assertTrue(row.passed)
        self.assertEqual(row.turns_to_pass, 0)
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("", "", ""))
        self.assertTrue(row.valid_sample and row.extract_ok and row.build_ok and row.bench_ran)
        self.assertEqual((row.time_median_ms, row.baseline_time_ms, row.speedup_vs_baseline), (4.0, 8.0, 2.0))
        self.assertEqual((row.baseline_run_time_ms, row.baseline_time_1t_ms, row.speedup_vs_baseline_1t), (10.0, 30.0, 7.5))
        self.assertEqual(row.lint_findings_total, 0)
        self.assertEqual(row.perf_summary, self.fakes.summary)
        self.assertIsNone(row.asan_findings)
        self.assertEqual(self.csv_rows()[-1]["pass"], "1")

    def test_numpy(self) -> None:
        row = self.run_language("numpy")
        self.assert_sequence("python")
        self.assert_passing_row(row, "numpy", "python")
        sample_dir = self.sample_dir("numpy")
        # extract: 시그니처 묶음(sigs.signature_bundle)을 넘긴다
        (call,) = self.fakes.extract_calls
        bundle = sigs.signature_bundle(self.spec, "numpy", "python", "positional")
        self.assertEqual(call["bundle"], bundle)
        self.assertEqual((call["backend"], call["args_mode"], call["work_dir"]), ("python", "positional", sample_dir))
        self.assertEqual(call["signature_spec"]["function"], "gemm")
        self.assertEqual(self.fakes.build_calls, [{"backend": "python", "code": CODE, "out_dir": sample_dir}])
        # 워커: 기존 언어의 인자는 그대로이고 import 에 관찰 인자만 붙는다
        for call in self.fakes.worker_calls:
            if call["stage"] == "paired_baseline":
                continue
            self.assertEqual((call["backend"], call["backend_args"], call["threads"], call["openmp"], call["args_mode"]),
                             ("python", [], 4, False, "positional"), call["stage"])
            self.assertEqual((call["env_extra"], call["max_mem_gb"], call["artifact"]), ({}, None, sample_dir / "artifact.bin"))
            self.assertEqual((call["cwd"], call["out_path"]), (sample_dir, sample_dir / f"{call['stage']}.json"))
        imp, chk, bench = (self.fakes.calls(s)[0] for s in ("import", "check", "bench"))
        self.assertEqual((imp["mode"], imp["tests"], imp["timeout_s"]), ("import", [], 180.0))
        self.assertEqual(imp["extra"], ["--inspect", "--inspect-dims", '{"M": 1024, "N": 512, "K": 256}'])
        self.assertEqual(chk["tests"], [self.root / "tests" / f"{n}.npz" for n in ("tiny", "rect", "small")])
        self.assertEqual((chk["extra"], chk["timeout_s"]), (["--check-repeats", "2"], 180.0))
        self.assertEqual((bench["tests"], bench["timeout_s"]), ([self.root / "tests" / "large.npz"], 300.0))
        self.assertEqual(bench["extra"], ["--warmup", "3", "--repeats", "10", "--min-repeats", "3", "--budget-s", "300.0"])
        # 대응 기준선은 그 샘플의 디렉터리에 남는다(조건 태그·턴이 다른 샘플과 겹치지 않는다)
        (paired,) = self.fakes.calls("paired_baseline")
        self.assertEqual((paired["out_path"], paired["cwd"], paired["mode"]), (sample_dir / "paired_baseline.json", sample_dir, "bench"))
        self.assertEqual((paired["backend"], paired["artifact"]), ("python", self.tasks_dir / "gemm" / "baseline" / "baseline.py"))
        # 행: python 계열은 cflags 가 없다
        self.assertEqual((row.compiler, row.compiler_version, row.cflags, row.march_resolved), ("python", "3.14.0", None, None))
        self.assertEqual(row.omp_env, {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        self.assertEqual((row.threads, row.cores, row.taskset_used), (4, "0,2,4,6", True))
        self.assertEqual(row.built_path, str(sample_dir.resolve()))
        self.assertEqual(result_json_path(row), sample_dir.resolve() / "result.json")
        self.assertEqual(sorted(self.stages(sample_dir)),
                         ["bench", "build", "check", "dynamic", "extract", "import", "lint", "paired_baseline", "perf_summary"])
        self.assertNotIn("log", self.stages(sample_dir)["build"])

    def test_c_openmp(self) -> None:
        row = self.run_language("c_openmp")
        self.assert_sequence("c")
        self.assert_passing_row(row, "c_openmp", "c")
        (build,) = self.fakes.build_calls
        self.assertEqual((build["openmp"], build["launch_prefix"], build["symbol"], build["args_header"]), (True, PREFIX, "gemm", None))
        self.assertIs(build["bcfg"], self.cfg.build)
        self.assertEqual({(c["backend"], c["openmp"], c["threads"]) for c in self.fakes.worker_calls if c["stage"] != "paired_baseline"},
                         {("c", True, 4)})
        self.assertEqual((row.compiler, row.compiler_version, row.march_resolved), ("gcc", "gcc 15.2.0", "-march=x"))
        self.assertEqual(row.cflags, "-std=gnu11 -fPIC -shared -O3 -march=native -fopenmp -lm")
        self.assertEqual(row.omp_env["OMP_PROC_BIND"], "true")
        self.assertEqual(self.fakes.lint_calls[0]["file"], self.sample_dir("c_openmp") / "kernel.c")
        self.assertEqual(self.fakes.sanitize_calls, [], "safety.enabled 가 꺼져 있으면 sanitizer 는 돌지 않는다")

    def test_extract_preprocesses_with_the_configured_compiler(self) -> None:
        """C 응답의 전처리(extract 의 ``cc``)는 빌드와 같은 설정의 컴파일러(build.cc)로 한다."""
        self.run_language("c_seq")
        self.assertEqual(self.fakes.extract_calls[0]["cc"], "gcc")
        cfg = replace(self.cfg, build=replace(self.cfg.build, cc="clang"))
        self.process(raw_doc("c_seq", 1), cfg=cfg)
        self.assertEqual(self.fakes.extract_calls[1]["cc"], "clang")

    def test_c_seq_runs_on_the_single_core(self) -> None:
        row = self.run_language("c_seq")
        self.assert_passing_row(row, "c_seq", "c")
        self.assertEqual((row.threads, row.cores), (1, "2"))
        self.assertEqual(row.cflags, "-std=gnu11 -fPIC -shared -O3 -march=native -lm")
        self.assertEqual(row.omp_env, {"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        self.assertFalse(self.fakes.build_calls[0]["openmp"])
        self.assertEqual({(c["openmp"], c["threads"]) for c in self.fakes.worker_calls if c["stage"] != "paired_baseline"}, {(False, 1)})

    def test_c_keyword_gets_the_args_header(self) -> None:
        conditions = {"args": "keyword", "grammar_hint": "off", "feedback": "compiler"}
        row, _ = self.process(raw_doc("c_seq", conditions=conditions))
        bundle = sigs.signature_bundle(self.spec, "c_seq", "c", "keyword")
        self.assertIn("gemm_args", bundle["args_struct"])
        self.assertEqual(self.fakes.build_calls[0]["args_header"], bundle["args_struct"])
        self.assertEqual(self.fakes.extract_calls[0]["bundle"], bundle)
        self.assertEqual(self.fakes.extract_calls[0]["args_mode"], "keyword")
        self.assertEqual({c["args_mode"] for c in self.fakes.worker_calls}, {"keyword"})
        self.assertEqual(row.args, "keyword")
        self.assertTrue((self.sample_dir("c_seq", tag="args-keyword_hint-off_fb-compiler") / "result.json").is_file())

    def test_numba(self) -> None:
        row = self.run_language("numba")
        self.assert_sequence("numba")
        self.assert_passing_row(row, "numba", "numba")
        (build,) = self.fakes.build_calls
        self.assertEqual((build["function"], build["expected_signature"]), ("gemm", sigs.numba_signature(self.spec)))
        self.assertEqual(self.fakes.extract_calls[0]["bundle"]["numba_signature"], sigs.numba_signature(self.spec))
        env = {"NUMBA_NUM_THREADS": "4", "NUMBA_THREADING_LAYER": "omp"}
        for call in self.fakes.worker_calls:
            if call["stage"] != "paired_baseline":
                self.assertEqual((call["backend"], call["backend_args"], call["env_extra"], call["openmp"]), ("numba", [], env, False))
        self.assertEqual(self.fakes.calls("import")[0]["tests"], [])
        self.assertEqual((row.compiler, row.compiler_version, row.cflags), ("numba", "0.68.1", None))
        self.assertEqual(row.omp_env, {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", **env})
        self.assertIsNone(self.fakes.calls("paired_baseline")[0].get("env_extra"), "기준선에는 언어의 env 가 가지 않는다")
        summary = self.fakes.summary_calls[0]
        self.assertEqual((summary["backend"], summary["language"]), ("numba", "numba"))
        self.assertEqual(summary["build_extra"], {"decorator": "njit", "options": {"parallel": True}, "uses_prange": True})
        self.assertEqual(summary["inspect"], IMPORT_OK["inspect"])

    def test_halide_auto(self) -> None:
        row = self.run_language("halide_auto")
        self.assert_sequence("halide")
        self.assert_passing_row(row, "halide_auto", "halide")
        (build,) = self.fakes.build_calls
        self.assertEqual((build["function"], build["halide_mode"]), ("gemm", "auto"))
        self.assertEqual(self.fakes.extract_calls[0]["bundle"]["params"], ["A", "B", "M", "N", "K"], "halide 는 출력 인자가 없다")
        backend_args = ["--halide-mode", "auto", "--threads", "4", "--halide-autoscheduler", "Mullapudi2016",
                        "--halide-params", '{"last_level_cache_size": "1048576"}']
        for call in self.fakes.worker_calls:
            if call["stage"] != "paired_baseline":
                self.assertEqual((call["backend"], call["backend_args"], call["env_extra"]),
                                 ("halide", backend_args, {"HL_NUM_THREADS": "4"}), call["stage"])
        # import 는 첫 정확성 테스트 크기로 파이프라인을 만들어 보고, 요약은 성능 형태의 스케줄을 기술한다
        imp = self.fakes.calls("import")[0]
        self.assertEqual(imp["tests"], [self.root / "tests" / "tiny.npz"])
        self.assertEqual(imp["extra"], ["--inspect", "--inspect-dims", '{"M": 1024, "N": 512, "K": 256}'])
        # cflags: 오토스케줄러와 워커가 실제로 넘기는 매개변수(하네스가 더하는 parallelism = 스레드 수 포함, 값은 문자열)
        self.assertEqual((row.compiler, row.compiler_version, row.cflags),
                         ("halide", "22.1.0", 'autoscheduler=Mullapudi2016 params={"last_level_cache_size":"1048576","parallelism":"4"}'))
        self.assertEqual(row.omp_env["HL_NUM_THREADS"], "4")
        self.assertEqual(row.contract_violation, [])

    def test_halide_manual(self) -> None:
        row = self.run_language("halide_manual")
        self.assert_sequence("halide")
        self.assert_passing_row(row, "halide_manual", "halide")
        self.assertEqual(self.fakes.build_calls[0]["halide_mode"], "manual")
        self.assertEqual({tuple(c["backend_args"]) for c in self.fakes.worker_calls if c["stage"] != "paired_baseline"},
                         {("--halide-mode", "manual", "--threads", "4")})
        self.assertEqual(row.cflags, "autoscheduler=none")
        self.assertEqual(self.fakes.calls("import")[0]["tests"], [self.root / "tests" / "tiny.npz"])

    def test_rust(self) -> None:
        row = self.run_language("rust")
        self.assert_sequence("rust")
        self.assert_passing_row(row, "rust", "rust")
        (build,) = self.fakes.build_calls
        self.assertIs(build["rcfg"], self.cfg.build.rust)
        self.assertEqual((build["launch_prefix"], build["symbol"], build["args_module"]), (PREFIX, "gemm", None))
        self.assertEqual(build["target_dir"], self.cfg.paths.built_dir / RUN_ID / "_cargo_target", "cargo target 은 run 당 하나")
        # 산출물은 C ABI 공유 라이브러리: 워커는 c 백엔드로 돌리고 OpenMP 환경은 주지 않는다
        for call in self.fakes.worker_calls:
            if call["stage"] != "paired_baseline":
                self.assertEqual((call["backend"], call["backend_args"], call["openmp"], call["env_extra"]),
                                 ("c", [], False, {"RAYON_NUM_THREADS": "4"}))
        self.assertEqual((row.compiler, row.compiler_version, row.cflags), ("cargo", "cargo 1.92.0", RUST_CFLAGS))
        self.assertEqual(row.omp_env["RAYON_NUM_THREADS"], "4")
        self.assertNotIn("OMP_PROC_BIND", row.omp_env)
        self.assertEqual(self.fakes.lint_calls[0]["file"], self.sample_dir("rust") / "src" / "lib.rs")
        self.assertEqual(self.fakes.summary_calls[0]["backend"], "rust")
        self.assertEqual(self.fakes.summary_calls[0]["build_log"], "   Compiling kernel v0.1.0\n")

    def test_rust_keyword_gets_the_args_module(self) -> None:
        conditions = {"args": "keyword", "grammar_hint": "off", "feedback": "compiler"}
        self.process(raw_doc("rust", conditions=conditions))
        bundle = sigs.signature_bundle(self.spec, "rust", "rust", "keyword")
        self.assertIn("pub struct GemmArgs", bundle["args_struct"])
        self.assertEqual(self.fakes.build_calls[0]["args_module"], bundle["args_struct"])

    def test_rust_flags_without_a_readable_manifest(self) -> None:
        lang = self.cfg.languages["rust"]
        with contextlib.redirect_stderr(io.StringIO()) as err:
            self.assertEqual(build_flags_text(self.cfg, lang, self.root / "nowhere"), "RUSTFLAGS=-C target-cpu=native")
        self.assertIn("Cargo.toml", err.getvalue())
        (self.root / "Cargo.toml").write_text('[package]\nname = "kernel"\n', encoding="utf-8")  # 프로파일이 없는 manifest
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(build_flags_text(self.cfg, lang, self.root), "RUSTFLAGS=-C target-cpu=native")

    def test_skip_bench_stops_after_check(self) -> None:
        row, _ = self.process(raw_doc("numpy"))
        self.assertEqual(self.fakes.events, ["extract", "build:python", "lint", "worker:import", "perf_summary", "worker:check"])
        self.assertTrue(row.passed)
        self.assertFalse(row.bench_ran)
        self.assertIsNone(row.time_median_ms)

    def test_language_environment_reaches_the_row_and_every_worker(self) -> None:
        """언어의 env 설정은 워커 환경과 행의 omp_env 에 같이 들어간다(NUMBA_·RAYON_·HL_ 접두)."""
        for language, key in (("numba", "NUMBA_NUM_THREADS"), ("halide_auto", "HL_NUM_THREADS"), ("rust", "RAYON_NUM_THREADS")):
            row, _ = self.process(raw_doc(language))
            self.assertEqual(row.omp_env[key], "4", language)
            self.assertEqual({c["env_extra"][key] for c in self.fakes.worker_calls if c["cwd"] == self.sample_dir(language)}, {"4"})


# ----------------------------------------------------------------------------- 실패 경로


class FailurePathTest(PipelineCase):
    def failed_build(self, kind: str, detail: str, **extra: Any) -> BuildResult:
        return BuildResult(ok=False, artifact=None, error_kind=kind, error_detail=detail, log="", build_time_ms=1.0, command=["x"],
                           compiler="numba", compiler_version="0.68.1", n_warnings=0, extra=extra)

    def test_generation_gate(self) -> None:
        cases = [
            ({"error": "transient:RateLimitError:429:slow down"}, "api_error", False),
            ({"stop_reason": "max_tokens"}, "truncated", False),
            ({"stop_reason": "refusal", "stop_details_category": "cyber"}, "refusal:cyber", True),
        ]
        for i, (over, detail, valid) in enumerate(cases):
            row, status = self.process(raw_doc("numpy", sample_idx=i, **over))
            self.assertEqual(status, "processed")
            self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("generate", "", detail))
            self.assertIs(row.valid_sample, valid, detail)
            self.assertFalse(row.passed)
            self.assertIsNone(row.turns_to_pass)
        self.assertEqual(self.fakes.events, [], "생성 게이트에 걸린 응답은 추출·빌드·실행을 하지 않는다")
        self.assertEqual([r["valid_sample"] for r in self.csv_rows()], ["0", "0", "1"])

    def test_extract_failure(self) -> None:
        self.fakes.extraction = SimpleNamespace(ok=False, code=None, signature_ok=None, signature_exact=None,
                                                detail="no_code_block", n_blocks=0)
        row, _ = self.process(raw_doc("numpy"))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("extract", "signature_mismatch", "no_code_block"))
        self.assertEqual(self.fakes.events, ["extract"], "소스 파일이 없으면 린트도 돌지 않는다")
        self.assertIsNone(row.lint_findings_total)
        self.assertIs(row.extract_ok, False)

    def test_numba_signature_missing_is_a_build_stage_signature_mismatch(self) -> None:
        self.fakes.builds["numba"] = self.failed_build("signature_mismatch", "numba_signature_missing", decorator="njit")
        row, _ = self.process(raw_doc("numba"))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("build", "signature_mismatch", "numba_signature_missing"))
        self.assertEqual(self.fakes.events, ["extract", "build:numba", "lint"], "빌드가 실패해도 린트는 돈다; 워커는 돌지 않는다")
        self.assertIs(row.build_ok, False)
        self.assertEqual(self.stages(self.sample_dir("numba"))["build"]["extra"], {"decorator": "njit"})

    def test_numba_signature_missing_found_by_the_worker(self) -> None:
        self.fakes.exits["import"] = Exit({"complete": True, "load_ok": False, "load_error": "numba_signature_missing",
                                           "classification": "signature_mismatch", "error_detail": "numba_signature_missing"})
        row, _ = self.process(raw_doc("numba"))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("build", "signature_mismatch", "numba_signature_missing"))
        self.assertIsNone(row.perf_summary, "로드에 실패한 커널은 요약하지 않는다")
        self.assertNotIn("perf_summary", self.fakes.events)

    def test_compile_error_and_compile_timeout(self) -> None:
        self.fakes.builds["c"] = self.failed_build("compile_error", "kernel.c:3:1: error: expected ';'")
        row, _ = self.process(raw_doc("c_seq", sample_idx=0))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("build", "compile_error", "kernel.c:3:1: error: expected ';'"))
        self.fakes.builds["c"] = self.failed_build("timeout", "compile timeout after 60s")
        row, _ = self.process(raw_doc("c_seq", sample_idx=1))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("build", "compile_error", "compile_timeout"))

    def test_import_stage_failures_are_build_stage_errors(self) -> None:
        self.fakes.exits["import"] = Exit({"complete": False, "in_progress": {"test": "tiny", "call_idx": None}}, rc=-11,
                                          signal_name="SIGSEGV")
        row, _ = self.process(raw_doc("halide_manual", sample_idx=0))
        self.assertEqual((row.error_stage, row.error_type), ("build", "runtime_crash"))
        self.assertIn("SIGSEGV", row.error_detail)
        self.fakes.exits["import"] = Exit({"complete": True, "load_ok": False, "load_error": "NameError: name 'hl' is not defined",
                                           "classification": "compile_error"})
        row, _ = self.process(raw_doc("halide_manual", sample_idx=1))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail),
                         ("build", "compile_error", "NameError: name 'hl' is not defined"))
        # 로드 실패를 적은 뒤 워커가 비정상 종료했으면 종료 규칙의 판정이 그 자리에 남는다(로드 전의 종료와 같다)
        self.fakes.exits["import"] = Exit({"complete": False, "load_ok": False, "load_error": "x", "classification": "compile_error"},
                                          rc=-9, timed_out=True)
        row, _ = self.process(raw_doc("halide_manual", sample_idx=2))
        self.assertEqual((row.error_stage, row.error_type), ("build", "timeout"))

    def test_worker_that_ends_while_inspecting_does_not_fail_the_sample(self) -> None:
        """커널이 로드된 뒤(load_ok) import 워커가 하는 일은 관찰값 수집뿐이다. 거기서 죽거나 시간을 다 써도 빌드 오류가 아니다:
        요약은 그 사실(워커가 수집 전에 써 둔 ``inspect_error``)만 가진 관찰값으로 만들고 판정은 check 가 내린다."""
        # harness.worker.collect_inspect 가 수집 직전에 써 두는 문서 그대로(그 뒤 프로세스가 끝났다)
        loaded = {"mode": "import", "backend": "halide", "complete": False, "load_ok": True, "load_error": None,
                  "classification": "pass", "error_detail": "", "blas": BLAS, "build_ms": 12.0,
                  "inspect": {"inspect_error": worker_module.INSPECT_UNFINISHED},
                  "in_progress": {"test": worker_module.INSPECT_STEP, "call_idx": None}}
        endings = (Exit(loaded, rc=-6, signal_name="SIGABRT"), Exit(loaded, rc=-9, timed_out=True, signal_name="SIGKILL"),
                   Exit(loaded, rc=1))
        for i, ending in enumerate(endings):
            self.fakes.exits["import"] = ending
            row, _ = self.process(raw_doc("halide_auto", sample_idx=i))
            self.assertEqual((row.passed, row.error_stage, row.error_type), (True, "", ""), ending)
            self.assertEqual(self.fakes.summary_calls[i]["inspect"], {"inspect_error": worker_module.INSPECT_UNFINISHED},
                             "요약이 '수집이 끝나지 않았다'를 적을 수 있게 워커의 inspect 를 그대로 넘긴다")
            self.assertEqual(row.perf_summary, self.fakes.summary)
            stage = self.stages(self.sample_dir("halide_auto", i))["import"]
            self.assertEqual((stage["complete"], stage["rc"], stage["timed_out"]), (False, ending.rc, ending.timed_out))
        self.assertEqual(self.fakes.events[:6], ["extract", "build:halide", "lint", "worker:import", "perf_summary", "worker:check"])
        self.assertIn("signal SIGABRT during inspect call ?", self.log)
        self.assertEqual(self.log.count("관찰값(--inspect) 없이 계속한다"), 3)
        # 그 뒤의 판정은 평소대로다: check 가 실패하면 check 의 실패다
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        row, _ = self.process(raw_doc("halide_auto", sample_idx=3))
        self.assertEqual((row.error_stage, row.error_type), ("check", "wrong_output"))

    def test_check_failures(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        row, _ = self.process(raw_doc("numpy", sample_idx=0))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("check", "wrong_output", "mismatch"))
        self.assertFalse(row.passed)
        self.assertIsNone(row.turns_to_pass)
        self.assertEqual(row.check_tests[0]["elapsed_ms"], 2.0)
        self.fakes.exits["check"] = Exit({"complete": False, "in_progress": {"test": "small", "call_idx": 0}}, rc=-9, timed_out=True,
                                         signal_name="SIGKILL")
        row, _ = self.process(raw_doc("numpy", sample_idx=1))
        self.assertEqual((row.error_stage, row.error_type), ("check", "timeout"))
        self.fakes.exits["check"] = Exit({**CHECK_PASS, "oob_write": True})
        row, _ = self.process(raw_doc("numpy", sample_idx=2))
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("check", "wrong_output", "out_of_bounds_write"))

    def test_bench_outcomes(self) -> None:
        self.fakes.exits["bench"] = Exit({**BENCH_PASS, "classification": "timeout", "error_detail": "first_call_90000ms_projected_360s",
                                          "median_ms": None})
        row, _ = self.process(raw_doc("numpy", sample_idx=0), skip_bench=False)
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("bench", "timeout", "first_call_90000ms_projected_360s"))
        self.assertTrue(row.passed, "bench timeout 은 성능 실패이지 정확성 실패가 아니다")
        self.assertEqual(row.turns_to_pass, 0)
        self.assertIsNone(row.time_median_ms)
        self.fakes.exits["bench"] = Exit({"complete": False, "in_progress": {"test": "large", "call_idx": 2}}, rc=-9, timed_out=True)
        row, _ = self.process(raw_doc("numpy", sample_idx=1), skip_bench=False)
        self.assertEqual((row.error_stage, row.error_type, row.passed), ("bench", "timeout", True))
        self.fakes.exits["bench"] = Exit({**BENCH_PASS, "classification": "race_condition", "error_detail": "nondeterministic_runs=2/10",
                                          "nondeterministic_runs": 2})
        row, _ = self.process(raw_doc("numpy", sample_idx=2), skip_bench=False)
        self.assertEqual((row.error_stage, row.error_type, row.passed), ("bench", "race_condition", False))
        self.assertEqual(row.time_median_ms, 4.0, "실패한 bench 도 잰 시간은 남긴다")
        self.assertIsNone(row.speedup_vs_baseline)
        self.fakes.exits["bench"] = Exit({"complete": False, "in_progress": {"test": "large", "call_idx": 0}}, rc=-11,
                                         signal_name="SIGSEGV")
        row, _ = self.process(raw_doc("numpy", sample_idx=3), skip_bench=False)
        self.assertEqual((row.error_stage, row.error_type, row.passed), ("bench", "runtime_crash", False))

    def test_failed_paired_baseline_is_flagged_not_fatal(self) -> None:
        self.fakes.exits["paired_baseline"] = Exit({**PAIRED_OK, "classification": "wrong_output", "final_check": {"classification": "wrong_output"}})
        row, _ = self.process(raw_doc("numpy"), skip_bench=False)
        self.assertTrue(row.passed)
        self.assertIn("paired_baseline_failed", row.protocol_flags)
        self.assertFalse(row.protocol_ok)
        self.assertIsNone(row.baseline_time_ms)
        self.assertIsNone(row.speedup_vs_baseline)
        self.assertEqual(row.speedup_vs_baseline_1t, 7.5)

    def test_worker_that_never_reported_stops_the_run_without_a_row(self) -> None:
        self.fakes.exits["check"] = Exit(None, rc=2)  # 워커가 인자 오류로 시작도 못 했다
        with self.assertRaises(HarnessError):
            self.process(raw_doc("numpy"))
        self.assertEqual(self.csv_rows(), [])
        self.assertFalse((self.sample_dir("numpy") / "result.json").exists())

    def test_worker_error_stops_the_run_without_a_row(self) -> None:
        self.fakes.exits["import"] = Exit({"worker_error": "Traceback ...", "complete": False}, rc=3)
        with self.assertRaises(HarnessError):
            self.process(raw_doc("numba"))
        self.assertEqual(self.csv_rows(), [])

    def test_missing_compiler_during_a_build_is_not_a_row(self) -> None:
        self.fakes.builds["c"] = ToolchainMissing("C 컴파일러 'gcc' 를 찾을 수 없다(PATH)")
        with self.assertRaises(ToolchainMissing):
            self.process(raw_doc("c_seq"))
        self.assertEqual(self.csv_rows(), [])
        self.assertFalse((self.sample_dir("c_seq") / "result.json").exists())

    def test_unavailable_autoscheduler_reported_by_the_worker_is_not_a_compile_error_row(self) -> None:
        self.fakes.exits["import"] = Exit({"complete": True, "load_ok": False, "classification": "compile_error",
                                           "load_error": "halide_autoscheduler_unavailable:Mullapudi2016"})
        with self.assertRaises(ToolchainMissing) as caught:
            self.process(raw_doc("halide_auto"))
        self.assertIn("halide_autoscheduler_unavailable:Mullapudi2016", str(caught.exception))
        self.assertEqual(self.csv_rows(), [])

    def test_unreadable_unknown_and_unsupported_raws_are_ignored(self) -> None:
        broken = self.cfg.paths.raw_dir / RUN_ID / "gemm" / "numpy" / TAG / "sample_7.json"
        broken.parent.mkdir(parents=True)
        broken.write_text("{not json", encoding="utf-8")
        self.assertEqual(self.process(broken), (None, "ignored"))
        self.assertEqual(self.process(raw_doc("fortran")), (None, "ignored"))
        self.tctx = replace(self.tctx, spec=load_taskspec(self.tasks_dir / "nohalide" / "signatures.yaml"), task="nohalide")
        self.assertEqual(self.process(raw_doc("halide_auto", task="nohalide")), (None, "ignored"))
        self.assertIn("not expressible as a Halide pipeline", self.log)
        self.assertEqual(self.fakes.events, [])
        self.assertEqual(self.csv_rows(), [])
        row, status = self.process(raw_doc("halide_manual", task="nohalide"))  # 같은 과제의 다른 언어는 돈다
        self.assertEqual((status, row.task, row.passed), ("processed", "nohalide", True))

    def test_up_to_date_result_is_skipped_and_force_reprocesses(self) -> None:
        raw_path = write_raw(self.cfg, raw_doc("numpy"))
        first, status = self.process(raw_path)
        self.assertEqual(status, "processed")
        n_events = len(self.fakes.events)
        again, status = self.process(raw_path)
        self.assertEqual(status, "skipped")
        self.assertEqual(again.to_csv_dict(), first.to_csv_dict())
        self.assertEqual(len(self.fakes.events), n_events, "건너뛴 샘플은 아무것도 다시 돌리지 않는다")
        _, status = self.process(raw_path, force=True)
        self.assertEqual(status, "processed")
        self.assertEqual(len(self.fakes.events), 2 * n_events)
        self.assertEqual(len(self.csv_rows()), 1, "같은 키의 행은 교체된다")


class ContractViolationTest(PipelineCase):
    def halide_build(self, out_dir: Path, calls: list[str]) -> BuildResult:
        return BuildResult(ok=True, artifact=out_dir / "kernel.py", error_kind="", error_detail="", log="", build_time_ms=1.0,
                           command=["python", "compile"], compiler="halide", compiler_version="22.1.0", n_warnings=0,
                           extra={"schedule_calls": calls, "imports_halide": True})

    def test_schedule_in_halide_auto_is_observed_not_failed(self) -> None:
        self.fakes.builds["halide"] = self.halide_build(self.sample_dir("halide_auto"), ["parallel", "vectorize"])
        row, _ = self.process(raw_doc("halide_auto"))
        self.assertEqual(row.contract_violation, ["schedule_in_halide_auto"])
        self.assertTrue(row.passed, "관찰만 한다: 판정은 바뀌지 않는다")
        self.assertTrue(row.build_ok)

    def test_schedule_calls_are_expected_in_halide_manual(self) -> None:
        self.fakes.builds["halide"] = self.halide_build(self.sample_dir("halide_manual"), ["parallel", "vectorize"])
        row, _ = self.process(raw_doc("halide_manual"))
        self.assertEqual(row.contract_violation, [])

    def test_existing_violations(self) -> None:
        self.fakes.builds["c"] = BuildResult(
            ok=True, artifact=self.sample_dir("c_seq") / "kernel.so", error_kind="", error_detail="", log="", build_time_ms=1.0,
            command=["gcc"], compiler="gcc", compiler_version="gcc 15.2.0", n_warnings=0, undefined_symbols=["GOMP_parallel", "system"],
            forbidden_symbols=["system"], thread_symbols=["GOMP_parallel"])
        self.fakes.exits["check"] = Exit(CHECK_PASS, stdout_bytes=12)
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual(row.contract_violation, ["forbidden_symbol:system", "threads_in_c_seq", "stdout_output"])
        self.assertEqual(row.undefined_symbols, ["GOMP_parallel", "system"])
        row, _ = self.process(raw_doc("c_openmp"))
        self.assertNotIn("threads_in_c_seq", row.contract_violation)


# ----------------------------------------------------------------------------- 선택 단계: lint · perf_summary · sanitizer


class LintStageTest(PipelineCase):
    COLUMNS = ("lint_findings_total", *(f"lint_{c}" for c in LINT_CATEGORIES))

    def columns(self, row: Any) -> dict[str, Any]:
        return {name: getattr(row, name) for name in self.COLUMNS}

    def test_findings_fill_the_columns(self) -> None:
        self.fakes.lint = lint_result(("numeric", 12), ("performance", 3), ("performance", 9))
        row, _ = self.process(raw_doc("numpy"))
        self.assertEqual(self.columns(row), {"lint_findings_total": 3, "lint_memory": 0, "lint_numeric": 1, "lint_concurrency": 0,
                                             "lint_shape_dtype": 0, "lint_api_misuse": 0, "lint_performance": 2})
        stage = self.stages(self.sample_dir("numpy"))["lint"]
        self.assertEqual((stage["tool"], stage["ok"], stage["error"]), ("baseline:ruff+pyflakes", True, ""))
        self.assertEqual([f["line"] for f in stage["findings"]], [12, 3, 9])
        self.assertEqual(stage["summary"]["performance"], 2)
        (call,) = self.fakes.lint_calls
        sample_dir = self.sample_dir("numpy")
        self.assertEqual((call["language"], call["backend"], call["task"]), ("numpy", "python", "gemm"))
        self.assertEqual((call["file"], call["out"], call["cwd"]), (sample_dir / "kernel.py", sample_dir / "lint.json", sample_dir))
        self.assertTrue(call["file_existed"], "린트는 소스 파일이 생긴 뒤에 돈다")
        self.assertEqual(call["signature"], {"function": "gemm", "params": ["A", "B", "C", "M", "N", "K"],
                                             "signature": "def gemm(A, B, C, M, N, K):", "args_mode": "positional"})
        json.dumps(call["signature"])  # 플러그인에 JSON 으로 넘어간다
        self.assertEqual(self.csv_rows()[0]["lint_performance"], "2")

    def test_clean_run_is_zero_not_na(self) -> None:
        row, _ = self.process(raw_doc("numpy"))
        self.assertEqual(set(self.columns(row).values()), {0})
        self.assertEqual(self.csv_rows()[0]["lint_findings_total"], "0")

    def test_no_linter_ran_is_na_not_zero(self) -> None:
        for i, result in enumerate((lint_result(tool="none"), lint_result(tool="none", ok=False, error="plugin_failed: rc=2"))):
            self.fakes.lint = result
            row, _ = self.process(raw_doc("rust", sample_idx=i))
            self.assertEqual(set(self.columns(row).values()), {None}, result.error)
            self.assertTrue(row.passed, "린트가 돌지 않아도 실험은 계속된다")
        self.assertEqual([r["lint_findings_total"] for r in self.csv_rows()], ["", ""])
        self.assertEqual(self.stages(self.sample_dir("rust", 1))["lint"]["error"], "plugin_failed: rc=2")

    def test_lint_runs_even_when_the_build_fails(self) -> None:
        self.fakes.builds["c"] = BuildResult(ok=False, artifact=None, error_kind="compile_error", error_detail="error: x", log="",
                                             build_time_ms=1.0, command=["gcc"], compiler="gcc", compiler_version="gcc 15.2.0",
                                             n_warnings=0)
        self.fakes.lint = lint_result(("memory", 4))
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual((row.error_stage, row.lint_findings_total, row.lint_memory), ("build", 1, 1))
        self.assertEqual(self.fakes.events, ["extract", "build:c", "lint"])
        self.assertTrue(self.fakes.lint_calls[0]["file_existed"])
        self.assertIn("lint", self.stages(self.sample_dir("c_seq")))

    def test_disabled_lint_is_not_run(self) -> None:
        cfg = replace(self.cfg, lint=replace(self.cfg.lint, enabled=False))
        row, _ = self.process(raw_doc("numpy"), cfg=cfg)
        self.assertNotIn("lint", self.fakes.events)
        self.assertEqual(set(self.columns(row).values()), {None})
        self.assertNotIn("lint", self.stages(self.sample_dir("numpy")))


class PerfSummaryStageTest(PipelineCase):
    def test_summary_inputs_and_row(self) -> None:
        row, _ = self.process(raw_doc("c_openmp"))
        (call,) = self.fakes.summary_calls
        self.assertEqual((call["language"], call["backend"], call["task"]), ("c_openmp", "c", "gemm"))
        self.assertEqual(call["code"], CODE)
        self.assertEqual(call["build_log"], BUILD_LOG_TEXT, "build.log 전문을 넘긴다(BuildResult.log 나 꼬리가 아니다)")
        self.assertIs(call["spec"], self.spec)
        self.assertEqual(call["inspect"], IMPORT_OK["inspect"])
        self.assertEqual(call["build_extra"], {})
        self.assertEqual(row.perf_summary, self.fakes.summary)
        self.assertEqual(self.stages(self.sample_dir("c_openmp"))["perf_summary"], {"ok": True, "notes": "3 loops found"})
        self.assertEqual(json.loads(self.csv_rows()[0]["perf_summary"]), self.fakes.summary)

    def test_python_family_has_no_build_log(self) -> None:
        self.process(raw_doc("numpy"))
        self.assertEqual(self.fakes.summary_calls[0]["build_log"], "")

    def test_failure_note_marks_the_stage_but_keeps_the_row(self) -> None:
        note = f"{run.perf_summary.ERROR_NOTE_PREFIX} ValueError: boom"
        self.fakes.summary = {"language": "numpy", "loops": [], "notes": note}
        row, _ = self.process(raw_doc("numpy"))
        self.assertTrue(row.passed, "요약 실패는 판정을 바꾸지 않는다")
        self.assertEqual(row.perf_summary["notes"], note)
        self.assertEqual(self.stages(self.sample_dir("numpy"))["perf_summary"], {"ok": False, "notes": note})

    def test_summary_survives_a_later_check_failure(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        row, _ = self.process(raw_doc("numpy"))
        self.assertEqual(row.error_type, "wrong_output")
        self.assertEqual(row.perf_summary, self.fakes.summary)

    def test_disabled_summary_is_not_computed(self) -> None:
        cfg = replace(self.cfg, perf_summary=replace(self.cfg.perf_summary, enabled=False))
        row, _ = self.process(raw_doc("numpy"), cfg=cfg)
        self.assertEqual(self.fakes.summary_calls, [])
        self.assertIsNone(row.perf_summary)
        self.assertNotIn("perf_summary", self.stages(self.sample_dir("numpy")))


class SanitizerStageTest(PipelineCase):
    def setUp(self) -> None:
        super().setUp()
        self.cfg = replace(self.cfg, safety=replace(self.cfg.safety, enabled=True))

    def test_no_findings(self) -> None:
        row, _ = self.process(raw_doc("c_openmp"), skip_bench=False)
        self.assertEqual(self.fakes.events, ["extract", "build:c", "lint", "worker:import", "perf_summary", "worker:check", "sanitize",
                                             "dynamic", "worker:paired_baseline", "worker:bench"])
        self.assertEqual((row.asan_findings, row.ubsan_findings), (0, 0))
        self.assertTrue(row.passed)
        (call,) = self.fakes.sanitize_calls
        sample_dir = self.sample_dir("c_openmp")
        self.assertEqual((call["code"], call["sample_dir"], call["openmp"], call["launch_prefix"], call["symbol"], call["args_header"]),
                         (CODE, sample_dir, True, PREFIX, "gemm", None))
        self.assertEqual(call["tests"], [self.root / "tests" / f"{n}.npz" for n in ("tiny", "rect", "small")])
        stage = self.stages(sample_dir)["sanitize"]
        self.assertEqual((stage["ran"], stage["build_ok"], stage["asan_findings"], stage["kinds"]), (True, True, 0, []))
        self.assertEqual((self.csv_rows()[0]["asan_findings"], self.csv_rows()[0]["ubsan_findings"]), ("0", "0"))

    def test_finding_on_a_failing_sample_is_memory_error(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        self.fakes.sanitize = sanitize_result(2, 1, ("heap-buffer-overflow", "signed-integer-overflow"))
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual((row.error_stage, row.error_type), ("check", "memory_error"))
        self.assertEqual(row.error_detail, "asan:heap-buffer-overflow; was wrong_output/mismatch")
        self.assertEqual((row.asan_findings, row.ubsan_findings, row.passed), (2, 1, False))

    def test_crash_with_a_finding_is_memory_error_and_keeps_the_crash_detail(self) -> None:
        self.fakes.exits["check"] = Exit({"complete": False, "in_progress": {"test": "rect", "call_idx": 0}}, rc=-11, signal_name="SIGSEGV")
        self.fakes.sanitize = sanitize_result(1, None, ("stack-buffer-overflow",), detail="incomplete:signal SIGABRT")
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual(row.error_type, "memory_error")
        self.assertEqual(row.error_detail, "asan:stack-buffer-overflow; was runtime_crash/signal SIGSEGV during rect call 0")
        self.assertEqual((row.asan_findings, row.ubsan_findings), (1, None))

    def test_finding_on_a_passing_sample_does_not_change_pass(self) -> None:
        self.fakes.sanitize = sanitize_result(1, 0, ("heap-use-after-free",))
        row, _ = self.process(raw_doc("c_seq"), skip_bench=False)
        self.assertTrue(row.passed, "pass 는 정확성 정의다: 잠재 결함은 asan_findings 열로만 남는다")
        self.assertEqual((row.error_stage, row.error_type, row.error_detail), ("", "", ""))
        self.assertEqual((row.asan_findings, row.turns_to_pass, row.time_median_ms), (1, 0, 4.0))

    def test_only_crashes_and_wrong_output_become_memory_error(self) -> None:
        self.fakes.sanitize = sanitize_result(1, 0, ("heap-buffer-overflow",))
        self.fakes.exits["check"] = Exit({**CHECK_WRONG, "classification": "race_condition", "error_detail": "nondeterministic: rect"})
        row, _ = self.process(raw_doc("c_openmp", sample_idx=0))
        self.assertEqual((row.error_type, row.error_detail, row.asan_findings), ("race_condition", "nondeterministic: rect", 1))
        self.fakes.exits["check"] = Exit({**CHECK_WRONG, "classification": "numeric_instability", "error_detail": "imprecise"})
        row, _ = self.process(raw_doc("c_openmp", sample_idx=1))
        self.assertEqual(row.error_type, "numeric_instability")

    def test_without_an_asan_report_the_verdict_stays(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        for i, result in enumerate((sanitize_result(0, 3, ("signed-integer-overflow",)), sanitize_result(None, None, detail="incomplete:rc=1"))):
            self.fakes.sanitize = result
            row, _ = self.process(raw_doc("c_seq", sample_idx=i))
            self.assertEqual((row.error_type, row.error_detail), ("wrong_output", "mismatch"))
            self.assertEqual((row.asan_findings, row.ubsan_findings), (result.asan_findings, result.ubsan_findings))

    def test_skipped_after_a_check_timeout(self) -> None:
        self.fakes.exits["check"] = Exit({"complete": False, "in_progress": {"test": "small", "call_idx": 0}}, rc=-9, timed_out=True)
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual(row.error_type, "timeout")
        self.assertEqual(self.fakes.sanitize_calls, [], "계측 빌드는 더 느려 같은 시간 상한에 또 걸린다")
        self.assertEqual((row.asan_findings, row.ubsan_findings), (None, None))
        self.assertNotIn("sanitize", self.stages(self.sample_dir("c_seq")))

    def test_only_the_c_backend_is_sanitized(self) -> None:
        for language in ("numpy", "numba", "halide_manual", "rust"):
            row, _ = self.process(raw_doc(language))
            self.assertTrue(row.passed, language)
            self.assertEqual((row.asan_findings, row.ubsan_findings), (None, None))
        self.assertEqual(self.fakes.sanitize_calls, [])

    def test_not_run_when_the_build_or_load_failed(self) -> None:
        self.fakes.exits["import"] = Exit({"complete": True, "load_ok": False, "load_error": "undefined symbol: foo",
                                           "classification": "compile_error"})
        row, _ = self.process(raw_doc("c_seq"))
        self.assertEqual(row.error_stage, "build")
        self.assertEqual(self.fakes.sanitize_calls, [])

    def test_keyword_sample_passes_its_args_header(self) -> None:
        self.process(raw_doc("c_seq", conditions={"args": "keyword", "grammar_hint": "off", "feedback": "compiler"}))
        self.assertEqual(self.fakes.sanitize_calls[0]["args_header"], sigs.signature_bundle(self.spec, "c_seq", "c", "keyword")["args_struct"])

    def test_sanitizer_worker_goes_through_the_sample_worker(self) -> None:
        """``run_sanitizers`` 가 넘겨받은 worker 로 계측 빌드를 돌린다: 산출물·환경·메모리 한도만 바뀌고 나머지는 그 샘플의 것이다."""
        self.fakes.sanitize_runs_worker = True
        self.process(raw_doc("c_openmp"))
        (call,) = self.fakes.calls("safety")
        sample_dir = self.sample_dir("c_openmp")
        self.assertEqual((call["mode"], call["backend"], call["threads"], call["openmp"]), ("check", "c", 4, True))
        self.assertEqual(call["artifact"], sample_dir / "san" / "kernel_san.so")
        self.assertEqual(call["env_extra"], {"LD_PRELOAD": "/usr/lib/libasan.so", "ASAN_OPTIONS": "detect_leaks=0"})
        self.assertEqual((call["max_mem_gb"], call["extra"], call["timeout_s"]), (0, ["--check-repeats", "1"], 300.0))
        self.assertEqual(call["out_path"], sample_dir / "safety.json")
        self.assertIn("safety", self.stages(sample_dir), "워커 요약은 stages.safety, SanitizeResult 는 stages.sanitize")
        self.assertEqual(self.fakes.calls("check")[0]["artifact"], sample_dir / "artifact.bin")

    def test_latent_finding_explains_a_later_bench_crash(self) -> None:
        """check 는 통과했지만 ASan 보고가 있던 커널이 성능 형태에서 죽으면 그 턴의 판정(runtime_crash)이 memory_error 가 된다."""
        self.fakes.sanitize = sanitize_result(1, 0, ("heap-buffer-overflow",))
        self.fakes.exits["bench"] = Exit({"complete": False, "in_progress": {"test": "large", "call_idx": 0}}, rc=-11, signal_name="SIGSEGV")
        row, _ = self.process(raw_doc("c_seq"), skip_bench=False)
        self.assertEqual((row.error_stage, row.error_type, row.passed), ("bench", "memory_error", False))
        self.assertEqual(row.error_detail, "asan:heap-buffer-overflow; was runtime_crash/signal SIGSEGV during large call 0")

    def test_missing_compiler_for_the_sanitizer_build_is_not_a_row(self) -> None:
        with mock.patch.object(run.safety, "run_sanitizers", side_effect=ToolchainMissing("gcc 없음")):
            with self.assertRaises(ToolchainMissing):
                self.process(raw_doc("c_seq"))
        self.assertEqual(self.csv_rows(), [])

    def test_disabled_by_default(self) -> None:
        self.assertFalse(load_config(self.config_path).safety.enabled)


# ----------------------------------------------------------------------------- 수정 턴


class RepairTurnTest(PipelineCase):
    def test_turn_directories_and_turns_to_pass(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        first, _ = self.process(raw_doc("c_seq", sample_idx=1))
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        second, _ = self.process(raw_doc("c_seq", sample_idx=1, turn=1, repair={"parent_raw": "sample_1.json"}))
        self.fakes.exits["check"] = Exit(CHECK_PASS)
        third, _ = self.process(raw_doc("c_seq", sample_idx=1, turn=2), skip_bench=False)

        base = self.sample_dir("c_seq", 1)
        self.assertEqual([c["out_dir"] for c in self.fakes.build_calls], [base, base / "turn_1", base / "turn_2"])
        for turn, row in enumerate((first, second, third)):
            turn_dir = base if turn == 0 else base / f"turn_{turn}"
            self.assertEqual(row.turn, turn)
            self.assertEqual(row.built_path, str(turn_dir.resolve()))
            self.assertEqual(result_json_path(row), turn_dir.resolve() / "result.json")
            self.assertTrue((turn_dir / "result.json").is_file())
            self.assertEqual(row.raw_path.replace("\\", "/").rsplit("/", 1)[-1], "sample_1.json" if turn == 0 else f"sample_1.turn{turn}.json")
        # 통과한 턴만 turns_to_pass 를 갖는다(그 턴 번호)
        self.assertEqual([(r.passed, r.turns_to_pass) for r in (first, second, third)], [(False, None), (False, None), (True, 2)])
        # 수정 턴의 행은 턴 0 행과 같은 run·과제·언어·조건·샘플 번호를 갖는다(분석이 그렇게 잇는다)
        key = lambda r: (r.run_id, r.task, r.language, r.args, r.grammar_hint, r.feedback, r.sample_idx)  # noqa: E731
        self.assertEqual({key(r) for r in (first, second, third)}, {(RUN_ID, "gemm", "c_seq", "positional", "off", "compiler", 1)})
        self.assertEqual([(r["turn"], r["pass"], r["turns_to_pass"]) for r in self.csv_rows()], [("0", "0", ""), ("1", "0", ""), ("2", "1", "2")])
        # 턴마다 워커 결과·대응 기준선이 그 턴 디렉터리에 남는다
        self.assertEqual([c["cwd"] for c in self.fakes.calls("check")], [base, base / "turn_1", base / "turn_2"])
        self.assertEqual(self.fakes.calls("paired_baseline")[0]["out_path"], base / "turn_2" / "paired_baseline.json")
        # 로그의 샘플 이름은 산출물 디렉터리의 경로(조건 태그·턴 포함)다
        self.assertIn(f"gemm/c_seq/{TAG}/sample_1/turn_2: pass=1", self.log)
        self.assertIn(f"gemm/c_seq/{TAG}/sample_1: pass=0", self.log)

    def test_turns_are_skipped_independently(self) -> None:
        turn0 = write_raw(self.cfg, raw_doc("numpy"))
        turn1 = write_raw(self.cfg, raw_doc("numpy", turn=1))
        self.assertEqual(self.process(turn0)[1], "processed")
        self.assertEqual(self.process(turn1)[1], "processed")
        self.assertEqual(self.process(turn0)[1], "skipped")
        self.assertEqual(self.process(turn1)[1], "skipped")
        turn1.write_text(json.dumps(raw_doc("numpy", turn=1, response_text="changed")), encoding="utf-8")
        self.assertEqual(self.process(turn0)[1], "skipped")
        self.assertEqual(self.process(turn1)[1], "processed")
        self.assertEqual(len(self.csv_rows()), 2)

    def test_bench_timeout_in_a_repair_turn_counts_as_passed_at_that_turn(self) -> None:
        self.fakes.exits["bench"] = Exit({**BENCH_PASS, "classification": "timeout", "error_detail": "first_call_90000ms"})
        row, _ = self.process(raw_doc("numpy", turn=3), skip_bench=False)
        self.assertEqual((row.passed, row.turns_to_pass, row.error_type), (True, 3, "timeout"))

    def test_generation_gate_in_a_repair_turn(self) -> None:
        row, _ = self.process(raw_doc("numpy", turn=1, stop_reason="refusal", stop_details_category=None))
        self.assertEqual((row.turn, row.error_stage, row.error_detail, row.valid_sample), (1, "generate", "refusal:", True))
        self.assertTrue((self.sample_dir("numpy") / "turn_1" / "result.json").is_file())


# ----------------------------------------------------------------------------- 도구 점검 · setup_run · finish_run · main


class MissingToolsTest(unittest.TestCase):
    """도구 점검(설계 v3 §10.6): 처리할 언어의 도구만 요구한다."""

    @classmethod
    def setUpClass(cls) -> None:
        tmp = tempfile.TemporaryDirectory(prefix="run_tools_")
        cls.addClassCleanup(tmp.cleanup)
        cls.cfg = load_config(write_config(Path(tmp.name)))
        cls.all = sorted(cls.cfg.languages)

    def problems(self, languages: Any, *, baseline_c: bool = False, **over: Any) -> list[str]:
        return missing_tools(self.cfg, {**TOOLCHAIN, **over}, languages, baseline_c=baseline_c)

    def test_everything_present(self) -> None:
        self.assertEqual(self.problems(self.all, baseline_c=True), [])

    def test_each_backend_needs_its_own_tool(self) -> None:
        cases = [
            ({"cc_version": None}, {"c_seq", "c_openmp"}, "gcc"),
            ({"cargo": None}, {"rust"}, "cargo"),
            ({"rustc": None}, {"rust"}, "rustc"),
            ({"numba_version": None, "errors": {"numba": "ModuleNotFoundError: No module named 'numba'"}}, {"numba"}, "No module named 'numba'"),
            ({"halide_version": None, "errors": {"python": "rc=1: boom"}}, {"halide_auto", "halide_manual"}, "rc=1: boom"),
            ({"halide_autoscheduler_loadable": False, "errors": {"halide_autoscheduler": "no plugin file"}}, {"halide_auto"}, "no plugin file"),
        ]
        for over, needing, text in cases:
            for language in self.all:
                found = self.problems([language], **over)
                self.assertEqual(bool(found), language in needing, (over, language))
            (line,) = self.problems(self.all, **over)
            self.assertIn(text, line)
            for language in needing:
                self.assertIn(language, line)

    def test_halide_manual_does_not_need_the_autoscheduler(self) -> None:
        self.assertEqual(self.problems(["halide_manual"], halide_autoscheduler_loadable=False), [])
        found = self.problems(["halide_auto"], halide_version=None, halide_autoscheduler_loadable=None)
        self.assertEqual(len(found), 1, "halide 가 없으면 그 한 가지만 보고한다(오토스케줄러는 따로 세지 않는다)")

    def test_c_baseline_needs_the_compiler_even_for_python_only_runs(self) -> None:
        self.assertEqual(self.problems(["numpy"], cc_version=None), [])
        (line,) = self.problems(["numpy"], baseline_c=True, cc_version=None)
        self.assertIn("C 기준선", line)
        (line,) = self.problems(["numpy", "c_seq"], baseline_c=True, cc_version=None)
        self.assertIn("c_seq", line)
        self.assertIn("C 기준선", line)

    def test_several_missing_tools_are_all_listed(self) -> None:
        found = self.problems(self.all, cc_version=None, cargo=None, numba_version=None)
        self.assertEqual(len(found), 3)

    def test_languages_not_in_the_config_are_ignored(self) -> None:
        self.assertEqual(self.problems(["fortran"], cc_version=None), [])


class SetupCase(PipelineCase):
    """``setup_run`` 과 ``main`` 테스트의 공통 준비: 머신을 건드리는 것(규약 검사·툴체인·테스트 생성·기준선)을 대역으로 바꾼다."""

    def setUp(self) -> None:
        super().setUp()
        self.static = STATIC
        self.toolchain: dict[str, Any] = dict(TOOLCHAIN)
        self.baseline_calls: list[tuple[str, bool]] = []
        self.baseline_error: Exception | None = None
        self.prefix_probes: list[list[str]] = []  # probe_launch_prefixes 가 run_isolated 로 띄운 명령
        self.prefix_result = _proc(0)
        self.runtime_probes: list[dict[str, Any]] = []  # protocol.runtime_probe_subprocess 호출
        self.runtime_errors: dict[str, str] = {}  # 런타임 종류 → 돌려줄 사유
        for patcher in (
            mock.patch.object(run, "run_isolated", side_effect=self.fake_prefix_probe),
            mock.patch.object(run.protocol, "runtime_probe_subprocess", side_effect=self.fake_runtime_probe),
            mock.patch.object(run, "load_machine", return_value={"hostname": "testhost"}),
            mock.patch.object(run.protocol, "check_static", side_effect=lambda cfg, machine: self.static),
            mock.patch.object(run.protocol, "toolchain_identity", side_effect=lambda cfg: self.toolchain),
            mock.patch.object(run.protocol, "environment_fingerprint", autospec=True, side_effect=self.fingerprint),
            mock.patch.object(run, "harness_version", return_value="v-test"),
            mock.patch.object(run, "ensure_tests", side_effect=lambda cfg, task: (self.root / "tests", MANIFEST)),
            mock.patch.object(run, "prepare_baselines", autospec=True, side_effect=self.prepare_baselines),
            mock.patch.object(run, "install_terminate_handlers"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def fake_prefix_probe(self, cmd: list[str], **kwargs: Any) -> ProcResult:
        self.prefix_probes.append(list(cmd))
        return self.prefix_result

    def fake_runtime_probe(self, kind: str, *, env: dict[str, str], prefix: list[str] | None = None, **kwargs: Any) -> str | None:
        self.runtime_probes.append({"kind": kind, "env": env, "prefix": prefix, **kwargs})
        return self.runtime_errors.get(kind)

    def fingerprint(self, cfg: Any, machine: Any, static: Any, *, toolchain: Any = None) -> tuple[dict[str, Any], str]:
        self.fingerprint_toolchain = toolchain
        return {"python_version": "3.14.0", "numpy_version": "2.3.5"}, "envhash"

    def prepare_baselines(self, ctx: RunContext, tctx: TaskContext, *, reuse: bool) -> None:
        self.baseline_calls.append((tctx.task, reuse))
        if self.baseline_error is not None:
            raise self.baseline_error
        tctx.baseline_run, tctx.baseline_1t = {"median_ms": 10.0}, {"median_ms": 30.0}

    def setup(self, **kwargs: Any) -> RunContext:
        with contextlib.redirect_stderr(io.StringIO()) as err:
            try:
                return setup_run(self.cfg, RUN_ID, **kwargs)
            finally:
                self.log += err.getvalue()

    def main(self, *args: str) -> int:
        with contextlib.redirect_stderr(io.StringIO()) as err, contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                return run.main(["--run-id", RUN_ID, "--config", str(self.config_path), *args])
            finally:
                self.log += err.getvalue()
                self.out = out.getvalue()

    def no_rows_written(self) -> bool:
        return not self.csv_rows() and not list(self.cfg.paths.built_dir.rglob("result.json"))


class SetupRunTest(SetupCase):
    def test_context_for_the_raws_of_the_run(self) -> None:
        for doc in (raw_doc("numpy", 1), raw_doc("numpy", 0), raw_doc("c_seq", 0), raw_doc("numpy", 0, turn=1), raw_doc("numpy", 0, task="cbase")):
            write_raw(self.cfg, doc)
        ctx = self.setup(skip_bench=True)
        base = self.cfg.paths.raw_dir / RUN_ID
        self.assertEqual([p.relative_to(base).as_posix() for p in ctx.raw_files], [
            f"cbase/numpy/{TAG}/sample_0.json", f"gemm/c_seq/{TAG}/sample_0.json", f"gemm/numpy/{TAG}/sample_0.json",
            f"gemm/numpy/{TAG}/sample_0.turn1.json", f"gemm/numpy/{TAG}/sample_1.json"])
        self.assertEqual(sorted(ctx.tasks), ["cbase", "gemm"])
        tctx = ctx.tasks["gemm"]
        self.assertEqual((tctx.task, tctx.manifest, tctx.spec.function), ("gemm", MANIFEST, "gemm"))
        self.assertEqual(tctx.tolerance_path, self.tasks_dir / "gemm" / "tolerance.yaml")
        self.assertEqual((ctx.run_id, ctx.env_hash, ctx.version, ctx.skip_bench, ctx.force), (RUN_ID, "envhash", "v-test", True, False))
        self.assertIs(self.fingerprint_toolchain, self.toolchain, "도구 점검에 쓴 툴체인 정체가 그대로 지문에 들어간다")
        self.assertEqual(self.baseline_calls, [], "--skip-bench 면 기준선을 재지 않는다")
        self.assertTrue(self.no_rows_written())

    def test_filters_select_the_raws(self) -> None:
        for doc in (raw_doc("numpy", 0), raw_doc("numpy", 1), raw_doc("c_seq", 0), raw_doc("numpy", 0, task="cbase")):
            write_raw(self.cfg, doc)
        ctx = self.setup(skip_bench=True, tasks=["gemm"], languages={"numpy"}, samples={1})
        self.assertEqual([p.name for p in ctx.raw_files], ["sample_1.json"])
        self.assertEqual(sorted(ctx.tasks), ["gemm"])

    def test_no_raw_files(self) -> None:
        with self.assertRaises(RunSetupError) as caught:
            self.setup()
        self.assertEqual(caught.exception.exit_code, EXIT_ERROR)
        self.assertIn("raw 파일이 없다", str(caught.exception))

    def test_protocol_failures(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        self.static = replace(STATIC, blas={"name": "blas"}, blas_ok=False, flags=["blas=reference"], ok=False)
        with self.assertRaises(RunSetupError) as caught:
            self.setup(skip_bench=True)
        self.assertEqual(caught.exception.exit_code, EXIT_PROTOCOL)
        self.static = replace(STATIC, governor="powersave", governor_ok=False, flags=["governor=powersave", "turbo=on"], ok=False)
        ctx = self.setup(skip_bench=True)  # 플래그는 경고와 함께 계속
        self.assertIn("governor=powersave", self.log)
        self.assertEqual(ctx.static.flags, ["governor=powersave", "turbo=on"])
        with self.assertRaises(RunSetupError) as caught:
            self.setup(skip_bench=True, strict_protocol=True)
        self.assertEqual(caught.exception.exit_code, EXIT_PROTOCOL)
        self.static = replace(STATIC, flags=["turbo=on"], ok=True)
        self.setup(skip_bench=True, strict_protocol=True)  # 정보용 플래그만 있으면 strict 도 통과

    def test_missing_tool_stops_before_anything_is_written(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        write_raw(self.cfg, raw_doc("c_seq"))
        write_raw(self.cfg, raw_doc("rust"))
        self.toolchain = {**TOOLCHAIN, "cc_version": None, "cargo": None}
        with self.assertRaises(RunSetupError) as caught:
            self.setup()
        self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN)
        message = str(caught.exception)
        self.assertIn("gcc", message)
        self.assertIn("c_seq", message)
        self.assertIn("cargo", message)
        self.assertEqual(self.baseline_calls, [], "도구 점검은 기준선 측정보다 먼저다")
        self.assertTrue(self.no_rows_written())
        # 그 도구가 필요한 언어를 빼면 돈다
        ctx = self.setup(languages={"numpy"})
        self.assertEqual([p.parts[-3] for p in ctx.raw_files], ["numpy"])

    def test_every_new_backend_is_checked(self) -> None:
        cases = {"numba": {"numba_version": None}, "halide_manual": {"halide_version": None},
                 "halide_auto": {"halide_autoscheduler_loadable": False}, "rust": {"rustc": None}}
        for i, (language, over) in enumerate(cases.items()):
            write_raw(self.cfg, raw_doc(language, i))
            self.toolchain = {**TOOLCHAIN, **over}
            with self.assertRaises(RunSetupError) as caught:
                self.setup(skip_bench=True, languages={language})
            self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN, language)
            self.assertIn(language, str(caught.exception))
        self.assertTrue(self.no_rows_written())

    def test_unsupported_pair_is_skipped_with_a_warning_and_needs_no_tool(self) -> None:
        write_raw(self.cfg, raw_doc("halide_auto", task="nohalide"))
        write_raw(self.cfg, raw_doc("numpy", task="nohalide"))
        self.toolchain = {**TOOLCHAIN, "halide_version": None, "halide_autoscheduler_loadable": None}
        ctx = self.setup(skip_bench=True)
        self.assertEqual([p.parts[-3] for p in ctx.raw_files], ["numpy"])
        self.assertIn("nohalide 는 halide_auto 를 지원하지 않는다", self.log)
        self.assertIn("not expressible as a Halide pipeline", self.log)

    def test_only_unsupported_raws_leave_nothing_to_do(self) -> None:
        write_raw(self.cfg, raw_doc("halide_auto", task="nohalide"))
        ctx = self.setup()
        self.assertEqual((ctx.raw_files, ctx.tasks), ([], {}))
        self.assertEqual(self.baseline_calls, [])

    def test_baselines_are_measured_only_for_tasks_with_pending_samples(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        ctx = self.setup()
        self.assertEqual(self.baseline_calls, [("gemm", False)])
        self.assertEqual(ctx.tasks["gemm"].baseline_run, {"median_ms": 10.0})
        self.setup(no_baseline=True)
        self.assertEqual(self.baseline_calls[-1], ("gemm", True), "--no-baseline 은 기존 파일을 재사용한다")

    def test_nothing_pending_keeps_the_baselines_on_disk(self) -> None:
        """모두 최신이면 다시 재지 않고 디스크의 기준선을 읽어 둔다 — 준비 뒤에 생기는 수정 턴의 행이 그 값을 쓴다."""
        raw_path = write_raw(self.cfg, raw_doc("numpy"))
        sample_dir = self.sample_dir("numpy")
        sample_dir.mkdir(parents=True)
        (sample_dir / "result.json").write_text(json.dumps(result_doc(raw_path)), encoding="utf-8")
        ctx = self.setup()
        self.assertEqual(self.baseline_calls, [])
        self.assertIsNone(ctx.tasks["gemm"].baseline_run, "디스크에 기준선이 없으면 NA 로 남는다")
        bdir = self.cfg.paths.built_dir / RUN_ID / "gemm" / "baseline"
        bdir.mkdir(parents=True)
        (bdir / "bench_run.json").write_text(json.dumps({"median_ms": 11.0}), encoding="utf-8")
        (bdir / "bench_1t.json").write_text(json.dumps({"median_ms": 33.0}), encoding="utf-8")
        ctx = self.setup()
        self.assertEqual(self.baseline_calls, [])
        self.assertEqual((ctx.tasks["gemm"].baseline_run, ctx.tasks["gemm"].baseline_1t), ({"median_ms": 11.0}, {"median_ms": 33.0}))
        self.assertIn("기준선 측정을 건너뜀", self.log)
        ctx = self.setup(force=True)  # --force 는 전부 다시 처리하므로 기준선도 다시 잰다
        self.assertEqual(self.baseline_calls, [("gemm", False)])

    def test_c_baseline_needs_the_compiler_unless_bench_is_skipped(self) -> None:
        write_raw(self.cfg, raw_doc("numpy", task="cbase"))
        self.toolchain = {**TOOLCHAIN, "cc_version": None}
        with self.assertRaises(RunSetupError) as caught:
            self.setup()
        self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN)
        self.assertIn("C 기준선", str(caught.exception))
        self.setup(skip_bench=True)  # 기준선을 빌드하지 않으면 컴파일러도 필요 없다

    def test_baseline_failures_keep_their_exit_codes(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        cases = [(BaselineError("gflops too low"), EXIT_BASELINE), (HarnessError("worker_error"), EXIT_HARNESS),
                 (ToolchainMissing("gcc 없음"), EXIT_TOOLCHAIN), (RuntimeError("gen_tests.py 실패"), EXIT_ERROR),
                 (FileNotFoundError("기준선이 없다"), EXIT_ERROR)]
        for error, code in cases:
            self.baseline_error = error
            with self.assertRaises(RunSetupError) as caught:
                self.setup()
            self.assertEqual(caught.exception.exit_code, code, error)
            self.assertIn(str(error), str(caught.exception))
        self.assertTrue(self.no_rows_written())

    def test_every_launch_prefix_is_tried_once_before_anything_runs(self) -> None:
        """run 이 쓸 taskset 접두(언어의 스레드 수, bench 면 기준선의 measurement.threads·1스레드)마다 한 번 돌려 본다.
        설정의 코어가 이 머신에 없으면 모든 명령이 rc 1 로 끝나 compile_error/runtime_crash 행이 되므로 행을 쓰기 전에 exit 7."""
        prefixes = {1: ["taskset", "-c", "2"], 4: ["taskset", "-c", "0,2,4,6"]}
        write_raw(self.cfg, raw_doc("numpy"))
        with mock.patch.object(run.protocol, "launch_prefix", side_effect=lambda cfg, threads: list(prefixes[threads])):
            self.setup(skip_bench=True)
            self.assertEqual(self.prefix_probes, [[*prefixes[4], str(run.PYTHON_EXE), "-c", "pass"]], "numpy 만: 4스레드 접두 하나")
            self.prefix_probes.clear()
            self.setup()
            self.assertEqual([c[:3] for c in self.prefix_probes], [prefixes[1], prefixes[4]], "bench 면 1스레드 기준선의 접두도")
            self.prefix_probes.clear()
            self.prefix_result = ProcResult(rc=1, timed_out=False, elapsed_s=0.0, stdout_tail="",
                                            stderr_tail="taskset: failed to set pid 0's affinity: Invalid argument", signal_name=None)
            with self.assertRaises(RunSetupError) as caught:
                self.setup()
        self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN)
        message = str(caught.exception)
        self.assertIn("measurement.single_core=2", message)
        self.assertIn("measurement.cores=0,2,4,6", message)
        self.assertIn("Invalid argument", message)
        self.assertEqual(self.baseline_calls, [("gemm", False)], "실패한 준비에서는 기준선을 재지 않는다(앞의 성공한 준비 한 번만)")
        self.assertTrue(self.no_rows_written())
        # 접두가 비면(taskset 없음·비 Linux) 돌려 볼 것이 없다
        self.prefix_probes.clear()
        with mock.patch.object(run.protocol, "launch_prefix", return_value=[]):
            self.setup(skip_bench=True)
        self.assertEqual(self.prefix_probes, [])

    def test_language_runtime_settings_are_tried_in_the_language_environment(self) -> None:
        for language in ("numpy", "numba", "halide_auto", "halide_manual"):
            write_raw(self.cfg, raw_doc(language))
        self.setup(skip_bench=True)
        by_kind = {p["kind"]: p for p in self.runtime_probes}
        self.assertEqual(sorted(by_kind), ["halide", "numba"], "halide_manual 은 오토스케줄러를 쓰지 않는다")
        halide = by_kind["halide"]
        self.assertEqual((halide["autoscheduler"], halide["params"], halide["threads"], halide["prefix"]),
                         ("Mullapudi2016", {"last_level_cache_size": "1048576"}, 4, PREFIX))
        self.assertEqual(halide["env"]["HL_NUM_THREADS"], "4")
        self.assertEqual((by_kind["numba"]["env"]["NUMBA_THREADING_LAYER"], by_kind["numba"]["env"]["NUMBA_NUM_THREADS"]), ("omp", "4"))
        self.runtime_errors = {"halide": "RuntimeError: unknown key beam_sizee", "numba": "ValueError: Unknown threading layer"}
        with self.assertRaises(RunSetupError) as caught:
            self.setup(skip_bench=True)
        self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN)
        message = str(caught.exception)
        self.assertIn("halide.autoscheduler_params", message)
        self.assertIn("unknown key beam_sizee", message)
        self.assertIn("NUMBA_THREADING_LAYER=omp", message)
        self.assertTrue(self.no_rows_written())

    def test_rust_toolchain_is_built_once_in_the_run_target_dir(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        self.setup(skip_bench=True)
        self.assertEqual(self.fakes.rust_probes, [], "rust 샘플이 없으면 점검하지 않는다")
        write_raw(self.cfg, raw_doc("rust"))
        self.setup(skip_bench=True)
        (probe,) = self.fakes.rust_probes
        self.assertIs(probe["rcfg"], self.cfg.build.rust)
        self.assertEqual(probe["target_dir"], self.cfg.paths.built_dir / RUN_ID / run.CARGO_TARGET_DIR, "샘플 빌드와 같은 target")
        self.assertEqual(probe["launch_prefix"], PREFIX)
        self.fakes.rust_probe_error = ToolchainMissing("자리표시 크레이트 빌드가 rc=101: no matching package named `rayon`")
        with self.assertRaises(RunSetupError) as caught:
            self.setup(skip_bench=True)
        self.assertEqual(caught.exception.exit_code, EXIT_TOOLCHAIN)
        self.assertIn("rayon", str(caught.exception))
        self.assertTrue(self.no_rows_written())

    def test_broken_task_spec(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        for text in ("task: gemm\nargs: nope\n", "task: [gemm\n"):  # 명세가 틀린 것, YAML 자체가 깨진 것
            (self.tasks_dir / "gemm" / "signatures.yaml").write_text(text, encoding="utf-8")
            with self.assertRaises(RunSetupError) as caught:
                self.setup(skip_bench=True)
            self.assertEqual(caught.exception.exit_code, EXIT_ERROR)
            self.assertIn("과제 명세를 읽을 수 없다", str(caught.exception))


class ReuseApiTest(SetupCase):
    """수정 루프(harness.repair)가 쓰는 방식: 준비 한 번 → 나중에 생긴 턴 raw 를 처리 → 끝에 보정."""

    def test_setup_then_process_turns_then_finish(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        turn0 = write_raw(self.cfg, raw_doc("c_seq"))
        ctx = self.setup(tasks=["gemm"], languages={"c_seq"})
        self.assertEqual(ctx.raw_files, [turn0])
        with contextlib.redirect_stderr(io.StringIO()):
            row0, _ = process_sample(ctx, ctx.tasks["gemm"], turn0)
            self.assertEqual((row0.passed, row0.turn), (False, 0))
            # 준비 뒤에 생긴 수정 턴의 raw 도 같은 컨텍스트로 처리한다
            self.fakes.exits["check"] = Exit(CHECK_PASS)
            self.fakes.exits["paired_baseline"] = Exit({**PAIRED_OK, "median_ms": 20.0})
            turn1 = write_raw(self.cfg, raw_doc("c_seq", turn=1, model_served="other-model"))
            row1, status = process_sample(ctx, ctx.tasks["gemm"], turn1)
            self.assertEqual((status, row1.passed, row1.turn, row1.turns_to_pass), ("processed", True, 1, 1))
            self.assertEqual((row1.baseline_run_time_ms, row1.baseline_time_ms), (10.0, 20.0))
            self.assertIsNone(row1.baseline_drift_pct)
            finish_run(ctx, [(row0, result_json_path(row0)), (row1, result_json_path(row1))])
        # finish_run: drift 는 과제의 대응 기준선 중앙값 기준, model_served 가 섞였으면 플래그
        self.assertEqual(row1.baseline_drift_pct, 0.0)
        self.assertEqual((row0.protocol_flags, row1.protocol_flags), (["model_served_mixed"], ["model_served_mixed"]))
        self.assertFalse(row1.protocol_ok)
        stored = json.loads(result_json_path(row1).read_text(encoding="utf-8"))
        self.assertEqual((stored["row"]["baseline_drift_pct"], stored["row"]["protocol_flags"]), (0.0, ["model_served_mixed"]))
        self.assertEqual([(r["turn"], r["protocol_ok"]) for r in self.csv_rows()], [("0", "0"), ("1", "0")])

    def test_finish_run_flags_baseline_drift(self) -> None:
        rows = []
        for i, paired in enumerate((8.0, 8.0, 12.0)):
            self.fakes.exits["paired_baseline"] = Exit({**PAIRED_OK, "median_ms": paired})
            row, _ = self.process(raw_doc("numpy", i), skip_bench=False)
            rows.append(row)
        with contextlib.redirect_stderr(io.StringIO()):
            finish_run(self.context(), [(row, result_json_path(row)) for row in rows])
        self.assertEqual([row.baseline_drift_pct for row in rows], [0.0, 0.0, 50.0])
        self.assertEqual([row.protocol_flags for row in rows], [[], [], ["baseline_drift=+50.0%"]])
        self.assertEqual([r["protocol_ok"] for r in self.csv_rows()], ["1", "1", "0"])


class MainTest(SetupCase):
    def test_processes_every_raw_in_order_and_summarizes(self) -> None:
        self.fakes.exits["check"] = Exit(CHECK_WRONG)
        for doc in (raw_doc("numpy", 1), raw_doc("numpy", 0), raw_doc("numpy", 0, turn=1), raw_doc("rust", 0)):
            write_raw(self.cfg, doc)
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertEqual([(r["language"], r["sample_idx"], r["turn"]) for r in self.csv_rows()],
                         [("numpy", "0", "0"), ("numpy", "0", "1"), ("numpy", "1", "0"), ("rust", "0", "0")])
        self.assertIn("run 요약: 행 4개 (건너뜀 0개)", self.out)
        self.assertIn("gemm/numpy turn 1", self.out)
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertIn("run 요약: 행 4개 (건너뜀 4개)", self.out)
        self.assertEqual(self.main("--skip-bench", "--languages", "rust", "--samples", "0", "--force"), EXIT_OK)
        self.assertIn("run 요약: 행 1개 (건너뜀 0개)", self.out)

    def test_conditions_of_one_run_are_told_apart(self) -> None:
        """한 run 에 조건이 둘이면(같은 언어·같은 샘플 번호) 로그와 요약이 조건 태그로 구분한다."""
        keyword = {"args": "keyword", "grammar_hint": "off", "feedback": "compiler"}
        write_raw(self.cfg, raw_doc("numpy", 0))
        write_raw(self.cfg, raw_doc("numpy", 0, conditions=keyword))
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertEqual([(r["args"], r["sample_idx"]) for r in self.csv_rows()], [("keyword", "0"), ("positional", "0")])
        for tag in ("args-keyword_hint-off_fb-compiler", TAG):
            self.assertIn(f"gemm/numpy/{tag}/sample_0: pass=1", self.log)
            self.assertIn(f"  gemm/numpy {tag} ", self.out)
        self.assertEqual(self.out.count("n=1 pass=1"), 2)

    def stored(self, language: str, sample_idx: int) -> tuple[Any, ...]:
        row = json.loads((self.sample_dir(language, sample_idx) / "result.json").read_text(encoding="utf-8"))["row"]
        return row["baseline_drift_pct"], row["protocol_flags"], row["protocol_ok"]

    def test_filtered_invocation_keeps_the_run_wide_drift_reference(self) -> None:
        """회귀: 필터(--languages)로 돈 호출이 그 부분집합의 대응 기준선 중앙값으로 drift 를 다시 정해, 아무것도 다시 재지 않았는데
        건너뛴 행의 baseline_drift 플래그와 protocol_ok 가 바뀌었다. 보정은 run 의 모든 result.json 으로 한다."""
        paired = [10.0, 12.0, 12.1, 10.0, 10.0, 10.0]  # 처리 순서: c_seq 0,1,2 → numpy 0,1,2

        class PairedSequence(dict):
            def __getitem__(self, stage: str) -> Exit:
                if stage == "paired_baseline":
                    return Exit({**PAIRED_OK, "median_ms": paired.pop(0)})
                return super().__getitem__(stage)

        self.fakes.exits = PairedSequence(self.fakes.exits)
        for language in ("numpy", "c_seq"):
            for i in range(3):
                write_raw(self.cfg, raw_doc(language, i))
        self.assertEqual(self.main(), EXIT_OK)
        before = {(lang, i): self.stored(lang, i) for lang in ("numpy", "c_seq") for i in range(3)}
        self.assertEqual(before[("c_seq", 1)], (20.0, ["baseline_drift=+20.0%"], False))
        self.assertEqual(before[("c_seq", 0)], (0.0, [], True))
        csv_before = self.csv_rows()
        self.assertEqual(self.main("--languages", "c_seq"), EXIT_OK)
        self.assertIn("run 요약: 행 3개 (건너뜀 3개)", self.out)
        self.assertEqual({key: self.stored(*key) for key in before}, before)
        self.assertEqual(self.csv_rows(), csv_before)
        self.assertEqual(self.out.count("flagged=2"), 1, "요약도 run 전체 기준으로 보정한 값을 센다")

    def test_model_served_mix_is_seen_across_invocations(self) -> None:
        write_raw(self.cfg, raw_doc("numpy", 0, model_served="model-a"))
        write_raw(self.cfg, raw_doc("c_seq", 0, model_served="model-b"))
        self.assertEqual(self.main("--skip-bench", "--languages", "numpy"), EXIT_OK)
        self.assertEqual(self.main("--skip-bench", "--languages", "c_seq"), EXIT_OK)
        self.assertEqual({(r["language"], r["protocol_flags"]) for r in self.csv_rows()},
                         {("numpy", '["model_served_mixed"]'), ("c_seq", '["model_served_mixed"]')})

    def test_rows_written_with_skip_bench_are_benched_by_a_later_full_run(self) -> None:
        """회귀: ``--skip-bench`` 로 쓴 통과 행이 최신으로 보여, 뒤의 측정 run 이 bench 도 기준선도 없이 exit 0 으로 끝났다."""
        write_raw(self.cfg, raw_doc("numpy", 0))
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertEqual([(r["pass"], r["bench_ran"]) for r in self.csv_rows()], [("1", "0")])
        self.assertEqual(self.main(), EXIT_OK)
        self.assertIn("bench 없이 쓰였다 — 다시 처리한다", self.log)
        self.assertEqual(self.baseline_calls, [("gemm", False)], "다시 처리할 샘플이 있으므로 기준선도 잰다")
        self.assertIn("worker:bench", self.fakes.events)
        self.assertEqual([(r["pass"], r["bench_ran"], r["time_median_ms"]) for r in self.csv_rows()], [("1", "1", "4.0")])
        stored = json.loads((self.sample_dir("numpy") / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(stored["requested"], {"bench": True, "safety": False, "lint": True})
        self.assertEqual(self.main(), EXIT_OK)
        self.assertIn("run 요약: 행 1개 (건너뜀 1개)", self.out)
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertIn("run 요약: 행 1개 (건너뜀 1개)", self.out, "bench 까지 한 결과는 --skip-bench 에도 최신이다")

    def test_row_run_id_is_the_run_being_processed(self) -> None:
        """회귀: 다른 run 의 raw 를 복사해 다시 재면 행의 run_id 는 raw 안의 값(원래 run)이고 산출물은 새 run 아래라, 원래 run 의
        CSV 행이 새 측정으로 덮였다. 행의 run_id 는 처리 중인 run 이고 raw 의 값은 result.json 의 raw_run_id 에만 남는다."""
        original = run.RunRow(run_id="orig", created_at="2026-11-01T00:00:00+09:00", task="gemm", language="numpy", backend="python",
                              args="positional", grammar_hint="off", feedback="compiler", sample_idx=0, turn=0, provider="dummy",
                              model="dummy", passed=True, valid_sample=True, bench_ran=True, time_median_ms=15.17)
        run.upsert_row(self.cfg.paths.results_csv_dummy, original)
        before = [r for r in self.csv_rows() if r["run_id"] == "orig"]
        for i in range(2):
            write_raw(self.cfg, raw_doc("numpy", i, run_id="orig"))
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertEqual([r for r in self.csv_rows() if r["run_id"] == "orig"], before, "원래 run 의 행은 그대로다")
        self.assertEqual(sorted((r["run_id"], r["sample_idx"]) for r in self.csv_rows() if r["run_id"] != "orig"),
                         [(RUN_ID, "0"), (RUN_ID, "1")])
        stored = json.loads((self.sample_dir("numpy") / "result.json").read_text(encoding="utf-8"))
        self.assertEqual((stored["row"]["run_id"], stored["raw_run_id"]), (RUN_ID, "orig"))
        self.assertEqual(self.log.count("raw 의 run_id='orig'"), 1, "run 당 한 번만 경고한다")
        write_raw(self.cfg, raw_doc("numpy", 2))  # raw 의 run_id 가 같으면 raw_run_id 를 남기지 않는다
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertNotIn("raw_run_id", json.loads((self.sample_dir("numpy", 2) / "result.json").read_text(encoding="utf-8")))

    def test_missing_tool_is_exit_7_and_no_row(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        write_raw(self.cfg, raw_doc("numba"))
        self.toolchain = {**TOOLCHAIN, "numba_version": None, "errors": {"numba": "ModuleNotFoundError: No module named 'numba'"}}
        self.assertEqual(self.main("--skip-bench"), EXIT_TOOLCHAIN)
        self.assertEqual(EXIT_TOOLCHAIN, 7)
        self.assertIn("No module named 'numba'", self.log)
        self.assertEqual(self.fakes.events, [], "아무 샘플도 처리하지 않는다(도구가 있는 언어의 것도)")
        self.assertTrue(self.no_rows_written())

    def test_toolchain_missing_during_a_build_is_exit_7(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        write_raw(self.cfg, raw_doc("rust"))
        self.fakes.builds["rust"] = ToolchainMissing("cargo 를 시작할 수 없다")
        self.assertEqual(self.main("--skip-bench"), EXIT_TOOLCHAIN)
        self.assertIn("cargo 를 시작할 수 없다", self.log)
        self.assertEqual([r["language"] for r in self.csv_rows()], ["numpy"], "앞서 끝난 샘플의 행은 남고 그 샘플의 행은 없다")
        self.assertFalse((self.sample_dir("rust") / "result.json").exists())

    def test_harness_error_is_exit_5(self) -> None:
        write_raw(self.cfg, raw_doc("numpy"))
        self.fakes.exits["import"] = Exit(None, rc=2)
        self.assertEqual(self.main("--skip-bench"), EXIT_HARNESS)
        self.assertTrue(self.no_rows_written())

    def test_setup_errors_become_exit_codes(self) -> None:
        self.assertEqual(self.main(), EXIT_ERROR)  # raw 없음
        write_raw(self.cfg, raw_doc("numpy"))
        self.baseline_error = BaselineError("gflops too low")
        self.assertEqual(self.main(), EXIT_BASELINE)
        self.assertIn("gflops too low", self.log)
        self.assertEqual(self.main("--languages", "cobol"), run.EXIT_CONFIG)

    def test_unsupported_pair_leaves_no_row(self) -> None:
        write_raw(self.cfg, raw_doc("halide_auto", task="nohalide"))
        write_raw(self.cfg, raw_doc("halide_manual", task="nohalide"))
        self.assertEqual(self.main("--skip-bench"), EXIT_OK)
        self.assertEqual([(r["task"], r["language"]) for r in self.csv_rows()], [("nohalide", "halide_manual")])
        self.assertIn("지원하지 않는다", self.log)


class EntryPointStreamsTest(unittest.TestCase):
    """회귀: ``python -m harness.run --help`` 가 파이프에서 로캘 코드페이지로 찍혀(Windows) 한글이 깨졌다(설계 v3 §16)."""

    def test_help_is_utf8_on_a_pipe(self) -> None:
        import os
        import subprocess

        r = subprocess.run([str(run.PYTHON_EXE), "-m", "harness.run", "--help"], capture_output=True, timeout=120, cwd=str(run.ROOT),
                           env={**os.environ, "PYTHONIOENCODING": ""})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("오케스트레이션", r.stdout.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
