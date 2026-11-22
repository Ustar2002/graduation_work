"""오케스트레이션 (설계 §8 run.py, v3 §10): raw 응답 하나를 행 하나로 만든다.

    PYTHON_EXE -m harness.run --run-id ID [--config] [--languages a,b] [--samples 0,1] [--skip-bench]
                              [--no-baseline] [--force] [--strict-protocol] [--rebuild-csv]

    생성 게이트 → extract → build → [lint] → import → [perf_summary] → check → [safety] → bench → 행(§11) → CSV

재사용 API(설계 v3 §10.1; ``main`` 과 수정 루프 ``harness.repair`` 가 같이 쓴다):

    ctx = setup_run(cfg, run_id, …)                              # 실패는 RunSetupError(exit_code)
    row, status = process_sample(ctx, ctx.tasks[task], raw_path)   # raw 하나(턴 하나) → 행 하나
    finish_run(ctx, [(row, result_json_path(row)), …])             # drift·model_served 보정

1. ``setup_run``: ``protocol.check_static()`` 1회(플래그는 경고와 함께 계속; ``--strict-protocol`` 이면 exit 4; BLAS 가 OpenBLAS 가
   아니면 항상 exit 4) → 처리할 raw 찾기(과제가 지원하지 않는 언어는 경고하고 뺀다) → 도구 점검: 처리할 언어의 컴파일러·cargo·
   numba·halide·오토스케줄러가 없으면 **행을 쓰기 전에** exit 7 (없는 도구는 모델 오류가 아니다) → 쓸 수 있는지 한 번씩 돌려
   보기(실행 접두 taskset, halide_auto 오토스케줄러 매개변수·numba 스레딩 계층, Rust 자리표시 크레이트; 실패는 exit 7).
2. 테스트 npz 가 없으면 ``gen_tests.py`` 를 ``run_isolated`` 로 실행. manifest 에서 ``test_seed``, ``test_hash``.
3. run 시작 기준선(bench, measurement.threads) → gflops/BLAS 게이트(실패 시 exit 6) → 1스레드 기준선.
   ``results/built/<run_id>/<task>/baseline/{bench_run.json,bench_1t.json}``. ``--no-baseline`` 이면 기존 파일 재사용.
   처리할 샘플이 없는 과제는 다시 재지 않고 디스크의 기준선을 읽어 둔다.
4. raw 순회(과제·언어·조건·샘플·턴 순). 턴 0 은 ``sample_<i>.json`` → ``…/sample_<i>/``, 수정 턴 t 는 ``sample_<i>.turn<t>.json``
   → ``…/sample_<i>/turn_<t>/`` (``paths.built_turn_dir``). ``result.json`` 의 ``raw_sha256`` 이 같고 ``--force`` 가 아니며 이 run 이
   요청한 bench 가 빠지지 않았으면 건너뜀(``is_up_to_date``; safety·lint 만 빠진 결과는 알리고 건너뛴다 — 다시 처리하면 측정도
   다시 하게 된다). 행의 ``run_id`` 는 처리 중인 run 이다.
5. 생성 게이트(API 오류 / max_tokens 절단 / refusal) → extract 없이 행만 기록.
6. extract(시그니처 묶음) → build(백엔드별: c·python·numba·halide·rust) → lint(소스 파일이 생긴 뒤; 빌드가 실패해도) →
   import(워커 ``--inspect``) → perf_summary → check → sanitizer(C, 켜져 있을 때). 워커 종료 규칙: timed_out → timeout;
   rc<0 → runtime_crash(signal); worker_error 또는 결과 문서를 한 번도 쓰지 못한 종료 → harness_error(행 없음, exit 5);
   결과 문서가 있는 rc==3 은 커널의 exit(3) 로 보아 runtime_crash; rc==0인데 미완료 → runtime_crash; 그 밖 rc>0 → runtime_crash.
   import 워커가 커널을 로드한 뒤(load_ok) 관찰값을 모으다 끝난 것은 판정이 아니다(경고하고 관찰값 없이 계속).
   판정이 runtime_crash·wrong_output 인데 ASan 보고가 있으면 memory_error (통과한 샘플의 pass 는 바꾸지 않는다).
7. bench 직전: ``protocol.check_dynamic()`` → 대응 기준선(샘플 디렉터리의 ``paired_baseline.json``) → 샘플 bench.
8. pass = check 전부 pass ∧ ¬inputs_mutated ∧ ¬oob_write ∧ (bench 미실행 ∨ (final_check pass ∧ 크래시 없음 ∧ nondeterministic_runs==0)).
   bench timeout 만 pass=1·시간 NA·error_stage=bench/error_type=timeout. ``turns_to_pass`` 는 통과한 행의 턴 번호.
9. 행 → ``result.json`` (row + 단계별 상세 + raw_sha256) → ``schema.upsert_row`` (provider==dummy 면 runs_dummy.csv).
10. ``finish_run`` (이 run 의 모든 result.json 행으로; 필터와 무관): 과제별 대응 기준선 중앙값 기준 drift 플래그, ``model_served``
    가 단일값이 아니면 경고 + 플래그 ``model_served_mixed``. 이어 run 요약. ``--rebuild-csv`` 는 result.json 에서 CSV 재생성
    (result.json 이 없는 기존 CSV 행은 남긴다).

Ctrl-C/SIGTERM/SIGHUP: ``procutil.install_terminate_handlers()`` 가 SIGTERM·SIGHUP 을 KeyboardInterrupt 로 바꾸고,
``run_isolated`` 가 실행 중이던 워커 프로세스 그룹을 SIGKILL 한 뒤 exit 130 으로 끝난다(처리 중이던 샘플의 행은 쓰지 않음).

LLM 코드는 이 프로세스에서 절대 실행하지 않는다(extract/compile()/컴파일러/nm 만; 실행은 harness.worker 서브프로세스).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
import sys
import tomllib
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

from harness import perf_summary, protocol, safety
from harness.build.c import BUILD_LOG, BuildResult, ToolchainMissing, build_c
from harness.build.python import build_python
from harness.config import Config, ConfigError, LanguageCfg, load_config, load_machine
from harness.paths import (
    PYTHON_EXE, ROOT, TASKS_DIR, atomic_write_json, baseline_dir, built_turn_dir, cond_tag, ensure_dir, parse_raw_turn_name,
    rel_to_root,
)
from harness.procutil import ProcResult, install_terminate_handlers, run_isolated
from harness.protocol import StaticStatus, config_option, now_iso
from harness.schema import RunRow, harness_version, read_result_rows, rebuild_csvs, upsert_row
from harness.sigs import signature_bundle
from harness.taskspec import TaskSpec, load_taskspec

_LOG = "[run]"

EXIT_OK = 0
EXIT_ERROR = 1  # 입력 없음 등 일반 오류
EXIT_CONFIG = 2  # 설정 오류
EXIT_PROTOCOL = 4  # --strict-protocol 위반 또는 BLAS 가 OpenBLAS 가 아님
EXIT_HARNESS = 5  # 워커 harness_error (worker_error / 결과 문서 없는 종료)
EXIT_BASELINE = 6  # 기준선 게이트 실패
EXIT_TOOLCHAIN = 7  # 처리할 언어의 도구(컴파일러·cargo·numba·halide·오토스케줄러)가 없음 — 행을 쓰지 않는다
EXIT_INTERRUPTED = 130  # Ctrl-C / SIGTERM / SIGHUP 으로 중단 (실행 중이던 서브프로세스 그룹은 정리됨)

GEN_TESTS_TIMEOUT_S = 600.0
PREFIX_PROBE_TIMEOUT_S = 60.0  # probe_launch_prefixes 의 명령 하나의 시간 상한
RESULT_JSON = "result.json"
PAIRED_BASELINE_JSON = "paired_baseline.json"  # 샘플(턴) 디렉터리 안의 대응 기준선 결과 문서
CARGO_TARGET_DIR = "_cargo_target"  # run 당 하나: <built_dir>/<run_id>/_cargo_target (의존성 빌드 재사용)
# 행의 omp_env 열에 남기는 스레드 관련 변수의 접두(언어의 env 설정이 넣는 NUMBA_/RAYON_/HL_ 포함)
OMP_ENV_PREFIXES = ("OMP_", "OPENBLAS_", "MKL_", "GOMP_", "KMP_", "NUMBA_", "RAYON_", "HL_")
# 언어 백엔드 → 워커의 --backend (설계 v3 §5.1). rust 산출물은 C ABI 공유 라이브러리라 c 로 실행한다.
WORKER_BACKEND = {"c": "c", "rust": "c", "python": "python", "numba": "numba", "halide": "halide"}
# 언어 백엔드 → 빌드 함수가 샘플 디렉터리에 쓰는 소스 파일 (설계 v3 §10.2a)
SOURCE_FILE = {"c": "kernel.c", "python": "kernel.py", "numba": "kernel.py", "halide": "kernel.py", "rust": "src/lib.rs"}
# 워커가 오토스케줄러 플러그인을 로드하지 못했을 때의 load_error 접두 (설계 v3 §5.3)
AUTOSCHEDULER_UNAVAILABLE = "halide_autoscheduler_unavailable:"
# ASan 보고가 있으면 memory_error 로 바뀌는 판정 (설계 v3 §10.2)
MEMORY_ERROR_FROM = ("runtime_crash", "wrong_output")
# 하네스 소스 상태를 CSV 에 남기는 플래그 이름
FLAG_MODEL_MIXED = "model_served_mixed"
FLAG_PAIRED_FAILED = "paired_baseline_failed"
DRIFT_FLAG_PREFIX = "baseline_drift="


class HarnessError(RuntimeError):
    """워커가 하네스 문제로 끝났다(worker_error, 또는 결과 문서를 쓰지 못한 종료). run 을 중단한다(exit 5)."""


class BaselineError(RuntimeError):
    """기준선 게이트 실패(참조 BLAS 의심 등). run 을 중단한다(exit 6)."""


class RunSetupError(RuntimeError):
    """``setup_run`` 이 run 을 시작할 수 없다. ``exit_code`` 는 프로세스 종료 코드(``EXIT_*``)다."""

    def __init__(self, message: str, exit_code: int) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ----------------------------------------------------------------------------- 컨텍스트


@dataclass
class TaskContext:
    """과제 하나의 고정 입력(테스트·허용오차·시그니처·기준선)."""

    task: str
    test_dir: Path
    manifest: dict[str, Any]
    tolerance_path: Path
    signatures: dict[str, Any]
    spec: TaskSpec
    baseline_artifact: Path | None = None  # C 기준선은 run 당 한 번 빌드한 .so
    baseline_run: dict[str, Any] | None = None
    baseline_1t: dict[str, Any] | None = None

    def test_path(self, name: str) -> Path:
        return self.test_dir / f"{name}.npz"


@dataclass
class RunContext:
    """run 전체에 공통인 상태. ``raw_files`` 는 ``setup_run`` 이 찾은 처리 대상(처리 순서)이다."""

    cfg: Config
    machine: dict[str, Any]
    static: StaticStatus
    fingerprint: dict[str, Any]
    env_hash: str
    run_id: str
    skip_bench: bool
    force: bool
    version: str
    tasks: dict[str, TaskContext] = field(default_factory=dict)
    raw_files: list[Path] = field(default_factory=list)
    warned: set[str] = field(default_factory=set)  # run 당 한 번만 낼 경고의 키

    def csv_for(self, provider: str | None) -> Path:
        return self.cfg.paths.results_csv_dummy if provider == "dummy" else self.cfg.paths.results_csv


@dataclass(frozen=True)
class SampleRef:
    """raw 하나가 가리키는 샘플의 한 턴: 산출물 디렉터리(``paths.built_turn_dir``)와 로그용 이름(``sample_label``)."""

    sample_dir: Path
    label: str


@dataclass
class WorkerOutcome:
    """워커 실행 결과(프로세스 + 결과 문서)."""

    stage: str
    proc: ProcResult
    doc: dict[str, Any] | None
    out_path: Path
    stdout_bytes: int
    timeout_s: float

    def in_progress_desc(self) -> str:
        ip = (self.doc or {}).get("in_progress") or {}
        test = ip.get("test") or "?"
        idx = ip.get("call_idx")
        return f"{test} call {idx if idx is not None else '?'}"

    def verdict(self) -> tuple[str, str] | None:
        """설계 §8-6 종료 규칙. 정상 종료면 None. harness_error 면 ``HarnessError``.

        harness_error 는 (a) 워커가 스스로 남긴 worker_error, (b) 결과 문서를 한 번도 쓰지 못하고 끝난 경우다. 워커는 커널을
        로드하기 전에 첫 문서를 쓰므로 문서가 없으면 LLM 코드는 실행된 적이 없다(인터프리터·하네스 임포트·인자 오류 등) —
        그것을 커널의 runtime_crash 로 적으면 하네스 문제가 모델 오류가 된다. 문서(in_progress 등)가 있는데 worker_error 가
        없으면 rc 가 3 이어도 LLM 커널이 exit(3) 을 부른 것이므로 runtime_crash 다(rc>0 규칙).
        """
        doc = self.doc or {}
        if doc.get("worker_error") or not doc:
            reason = str(doc.get("worker_error") or f"결과 문서 없이 끝났다 (timed_out={self.proc.timed_out}): {self.proc.stderr_tail}")
            raise HarnessError(f"{self.stage}: worker_error (rc={self.proc.rc}) — 결과 {self.out_path}\n{reason}")
        if self.proc.timed_out:
            return "timeout", f"worker timeout after {self.timeout_s:g}s during {self.in_progress_desc()}"
        if self.proc.rc < 0:
            return "runtime_crash", f"signal {self.proc.signal_name} during {self.in_progress_desc()}"
        if self.proc.rc == 0 and not doc.get("complete"):
            return "runtime_crash", "worker exited without complete result"
        if self.proc.rc > 0:
            return "runtime_crash", f"rc={self.proc.rc}: {self.proc.stderr_tail.strip()[-300:]}"
        return None

    def summary(self) -> dict[str, Any]:
        """result.json 에 넣는 단계 요약(결과 문서 전체는 <stage>.json 에 있다)."""
        return {
            "rc": self.proc.rc, "timed_out": self.proc.timed_out, "signal": self.proc.signal_name,
            "elapsed_s": round(self.proc.elapsed_s, 3), "timeout_s": self.timeout_s, "stdout_bytes": self.stdout_bytes,
            "stdout_tail": self.proc.stdout_tail, "stderr_tail": self.proc.stderr_tail, "result_file": rel_to_root(self.out_path),
            "complete": bool((self.doc or {}).get("complete")), "classification": (self.doc or {}).get("classification"),
            "error_detail": (self.doc or {}).get("error_detail"),
        }


# ----------------------------------------------------------------------------- 작은 헬퍼


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any] | None:
    """JSON 파일을 읽는다. 없거나 깨졌으면 None(+경고)."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        print(f"{_LOG} 경고: {path} 읽기 실패: {e}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def read_text(path: Path) -> str:
    """텍스트 파일 전문. 없으면 빈 문자열(깨진 바이트는 대체 문자)."""
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def blas_string(blas: dict[str, Any] | None) -> str | None:
    if not blas:
        return None
    name = str(blas.get("name") or "").strip()
    version = str(blas.get("version") or "").strip()
    return f"{name} {version}".strip() or None


def bench_args(cfg: Config) -> list[str]:
    m = cfg.measurement
    return ["--warmup", str(m.warmup), "--repeats", str(m.repeats), "--min-repeats", str(m.min_repeats),
            "--budget-s", str(m.bench_timeout_s)]


def omp_env_of(env: dict[str, str]) -> dict[str, str]:
    return {k: v for k, v in env.items() if k.startswith(OMP_ENV_PREFIXES)}


def sample_label(task: str | None, language: str | None, tag: str, sample_idx: int | None, turn: int | None) -> str:
    """로그용 샘플 이름 = ``<built_dir>/<run_id>`` 아래 그 샘플(턴) 디렉터리의 경로:
    ``gemm/c_seq/<cond_tag>/sample_1`` (턴 0) | ``gemm/c_seq/<cond_tag>/sample_1/turn_2`` (수정 턴).

    조건 태그까지 적는다: 한 run 에 조건이 둘 이상 있으면(수정 루프의 피드백 조건 대응 비교 등) 언어와 샘플 번호만으로는
    어느 행인지 알 수 없다.
    """
    base = f"{task}/{language}/{tag}/sample_{sample_idx}"
    return f"{base}/turn_{turn}" if turn else base


def row_tag(row: RunRow) -> str:
    """행의 조건 태그(``paths.cond_tag``)."""
    return cond_tag(str(row.args), str(row.grammar_hint), str(row.feedback))


def row_label(row: RunRow) -> str:
    """행의 ``sample_label``."""
    return sample_label(row.task, row.language, row_tag(row), row.sample_idx, row.turn)


def result_json_path(row: RunRow) -> Path:
    """행의 ``built_path`` (샘플·턴 디렉터리)에 있는 ``result.json``."""
    built = Path(row.built_path or "")
    return (built if built.is_absolute() else ROOT / built) / RESULT_JSON


def raw_task(raw_path: Path) -> str:
    """raw 경로 ``…/<task>/<language>/<cond_tag>/sample_*.json`` 의 과제 이름."""
    return raw_path.parts[-4]


def raw_language(raw_path: Path) -> str:
    """raw 경로 ``…/<task>/<language>/<cond_tag>/sample_*.json`` 의 언어 이름."""
    return raw_path.parts[-3]


def warn(msg: str) -> None:
    print(f"{_LOG} 경고: {msg}", file=sys.stderr)


def info(msg: str) -> None:
    print(f"{_LOG} {msg}", file=sys.stderr)


# ----------------------------------------------------------------------------- 워커 실행


def run_worker(
    cfg: Config,
    *,
    stage: str,
    threads: int,
    openmp: bool,
    backend: str,
    artifact: Path,
    mode: str,
    tests: list[Path],
    tolerance: Path,
    signatures: Path,
    out_path: Path,
    cwd: Path,
    timeout_s: float,
    args_mode: str = "positional",
    extra: list[str] | None = None,
    env_extra: dict[str, str] | None = None,
    backend_args: list[str] | None = None,
    max_mem_gb: float | None = None,
) -> WorkerOutcome:
    """``taskset`` 접두 + 화이트리스트 환경으로 ``harness.worker`` 를 띄운다. stdout/stderr 는 ``<cwd>/<stage>.*.log``.

    - ``backend``: 워커의 ``--backend`` 값. ``backend_args``: 그 백엔드의 추가 인자(halide 모드·오토스케줄러 등).
    - ``env_extra``: ``protocol.launch_env`` 가 만든 환경 위에 덮어쓰는 변수(언어의 env 설정, sanitizer 런타임 변수).
    - ``max_mem_gb``: ``--max-mem-gb`` (기본은 설정값; 0 이하면 워커가 RLIMIT_AS 를 걸지 않는다 — ASan 실행용).
    """
    mem_gb = cfg.measurement.max_worker_mem_gb if max_mem_gb is None else max_mem_gb
    cmd = [*protocol.launch_prefix(cfg, threads), str(PYTHON_EXE), "-m", "harness.worker",
           "--backend", backend, "--artifact", str(artifact), "--mode", mode,
           "--tolerance", str(tolerance), "--signatures", str(signatures), "--out", str(out_path), "--args-mode", args_mode,
           "--max-mem-gb", str(mem_gb)]
    if tests:
        cmd += ["--tests", ",".join(str(t) for t in tests)]
    cmd += [*(backend_args or []), *(extra or [])]
    env = protocol.launch_env(cfg, threads, openmp=openmp, extra_env=env_extra)
    ensure_dir(cwd)
    if out_path.exists():
        out_path.unlink()  # 이전 실행의 결과를 새 결과로 오인하지 않게
    stdout_log = cwd / f"{stage}.stdout.log"
    stderr_log = cwd / f"{stage}.stderr.log"
    proc = run_isolated(cmd, timeout_s=timeout_s, cwd=cwd, env=env, stdout_path=stdout_log, stderr_path=stderr_log)
    try:
        stdout_bytes = stdout_log.stat().st_size
    except OSError:
        stdout_bytes = 0
    doc = read_json(out_path)
    return WorkerOutcome(stage=stage, proc=proc, doc=doc, out_path=out_path, stdout_bytes=stdout_bytes, timeout_s=timeout_s)


def worker_backend_args(cfg: Config, lang: LanguageCfg) -> list[str]:
    """백엔드별 워커 인자(설계 v3 §5.1). halide: 모드와 스레드 수, auto 모드면 설정의 오토스케줄러 이름과 인자."""
    if lang.backend != "halide":
        return []
    mode = config_option(lang, "halide_mode")
    args = ["--halide-mode", str(mode), "--threads", str(lang.threads)]
    if mode == "auto":
        args += ["--halide-autoscheduler", str(config_option(cfg, "halide.autoscheduler")),
                 "--halide-params", json.dumps(dict(config_option(cfg, "halide.autoscheduler_params")), sort_keys=True)]
    return args


def performance_dims(cfg: Config, tctx: TaskContext) -> dict[str, int] | None:
    """성능 형태의 dim 값(테스트 manifest 의 ``files.<performance_test>``). manifest 에 전부 있지 않으면 None."""
    entry = (tctx.manifest.get("files") or {}).get(cfg.measurement.performance_test) or {}
    names = tctx.spec.dim_names
    return {name: int(entry[name]) for name in names} if all(name in entry for name in names) else None


def inspect_args(cfg: Config, tctx: TaskContext) -> list[str]:
    """import 단계의 관찰 인자: ``--inspect`` 와, 성능 형태의 dim 값을 알면 ``--inspect-dims``.

    halide 의 스케줄은 크기에 따라 달라지므로 요약은 측정되는 크기의 파이프라인을 기술해야 한다(설계 v3 §5.1).
    """
    dims = performance_dims(cfg, tctx)
    return ["--inspect", *(["--inspect-dims", json.dumps(dims)] if dims is not None else [])]


# ----------------------------------------------------------------------------- 준비 단계 (1–3)


def ensure_tests(cfg: Config, task: str) -> tuple[Path, dict[str, Any]]:
    """테스트 npz/manifest 가 없으면 gen_tests.py 를 격리 실행한다. (test_dir, manifest)."""
    test_dir = TASKS_DIR / task / "tests"
    m = cfg.measurement
    needed = [test_dir / f"{n}.npz" for n in [*m.correctness_tests, m.performance_test]] + [test_dir / "manifest.json"]
    if not all(p.exists() for p in needed):
        gen = TASKS_DIR / task / "gen_tests.py"
        if not gen.exists():
            raise FileNotFoundError(f"테스트 생성기가 없다: {gen}")
        info(f"테스트 입력이 없어 생성한다: {gen}")
        r = run_isolated([str(PYTHON_EXE), str(gen), "--out", str(test_dir)], timeout_s=GEN_TESTS_TIMEOUT_S, cwd=ROOT,
                         env=protocol.launch_env_minimal())
        if r.rc != 0 or r.timed_out:
            raise RuntimeError(f"gen_tests.py 실패(rc={r.rc}, timed_out={r.timed_out}): {r.stderr_tail.strip()[-500:]}")
        missing = [str(p) for p in needed if not p.exists()]
        if missing:
            raise RuntimeError(f"gen_tests.py 뒤에도 없는 파일: {missing}")
    manifest = read_json(test_dir / "manifest.json") or {}
    if "test_hash" not in manifest:
        warn(f"{test_dir / 'manifest.json'} 에 test_hash 가 없다")
    return test_dir, manifest


def load_signatures(task: str) -> dict[str, Any]:
    path = TASKS_DIR / task / "signatures.yaml"
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 최상위가 매핑이 아니다")
    return data


def signature_spec_for(signatures: dict[str, Any], language: str) -> dict[str, Any]:
    """extract 에 넘기는 시그니처 사양: 최상위(function, params, languages) + 그 언어 항목(positional, keyword, …)."""
    langs = signatures.get("languages") or {}
    entry = langs.get(language) if isinstance(langs, dict) else None
    spec: dict[str, Any] = dict(signatures)
    if isinstance(entry, dict):
        spec.update(entry)
    spec["language"] = language
    return spec


def baseline_gate(doc: dict[str, Any] | None, min_gflops: float | None, *, require_openblas: bool = True) -> tuple[bool, str]:
    """기준선 bench 결과가 쓸 만한가: 완료 ∧ final_check pass ∧ (gflops 하한) ∧ (python 기준선이면 OpenBLAS)."""
    if not doc:
        return False, "결과 문서 없음"
    if not doc.get("complete"):
        return False, "incomplete"
    fc = doc.get("final_check") or {}
    if doc.get("classification") != "pass" or fc.get("classification") != "pass":
        return False, f"classification={doc.get('classification')} ({doc.get('error_detail') or fc.get('pattern')})"
    gflops = doc.get("gflops")
    if min_gflops is not None and (gflops is None or gflops < min_gflops):
        return False, f"gflops={gflops} < baseline_min_gflops={min_gflops:g}"
    if require_openblas:
        name = str((doc.get("blas") or {}).get("name") or "")
        if "openblas" not in name.lower():
            return False, f"worker BLAS={name or 'unknown'} (OpenBLAS 아님)"
    return True, ""


def baseline_artifact(ctx: RunContext, tctx: TaskContext) -> Path:
    """과제 명세의 기준선 파일. python 은 그대로, c 는 run 당 한 번 기준선 디렉터리에 빌드한다(샘플과 같은 플래그)."""
    b = tctx.spec.baseline
    src = TASKS_DIR / tctx.task / str(b["file"])
    if not src.exists():
        raise FileNotFoundError(f"기준선이 없다: {src}")
    if b["backend"] == "python":
        return src
    if tctx.baseline_artifact is not None and tctx.baseline_artifact.exists():
        return tctx.baseline_artifact
    cfg = ctx.cfg
    build_dir = ensure_dir(baseline_dir(cfg.paths.built_dir, ctx.run_id, tctx.task) / "build")
    br = build_c(src.read_text(encoding="utf-8"), build_dir, cfg.build, openmp=bool(b.get("openmp")),
                 launch_prefix=protocol.launch_prefix(cfg, cfg.measurement.threads), symbol=tctx.spec.function)
    if not br.ok or br.artifact is None:
        raise BaselineError(f"C 기준선 빌드 실패({src}): {br.error_kind} {br.error_detail}")
    tctx.baseline_artifact = br.artifact
    return br.artifact


def run_baseline_bench(ctx: RunContext, tctx: TaskContext, *, threads: int, out_path: Path) -> tuple[dict[str, Any] | None, WorkerOutcome]:
    """과제의 기준선(python 또는 빌드한 C)을 bench 한다. 결과 문서는 ``out_path``, 로그는 그 옆(``<이름>.*.log``).
    (결과 문서, 실행 결과)를 돌려준다. harness_error 는 예외."""
    cfg = ctx.cfg
    b = tctx.spec.baseline
    oc = run_worker(
        cfg, stage=out_path.stem, threads=threads, openmp=b["backend"] == "c" and bool(b.get("openmp")), backend=str(b["backend"]),
        artifact=baseline_artifact(ctx, tctx), mode="bench",
        tests=[tctx.test_path(cfg.measurement.performance_test)], tolerance=tctx.tolerance_path, signatures=tctx.spec.path,
        out_path=out_path, cwd=out_path.parent, timeout_s=cfg.measurement.bench_timeout_s, extra=bench_args(cfg),
    )
    verdict = oc.verdict()
    if verdict is not None:
        warn(f"기준선 {out_path.name}: {verdict[0]} — {verdict[1]}")
    return oc.doc, oc


def _gate_opts(tctx: TaskContext) -> bool:
    b = tctx.spec.baseline
    return b["backend"] == "python" and bool(b.get("gate_blas", True))


def _rate(doc: dict[str, Any]) -> str:
    return f", {doc['gflops']:.1f} GFLOPS" if doc.get("gflops") else ""


def prepare_baselines(ctx: RunContext, tctx: TaskContext, *, reuse: bool) -> None:
    """§8-3: run 시작 기준선(+게이트)과 1스레드 기준선."""
    cfg = ctx.cfg
    m = cfg.measurement
    bdir = ensure_dir(baseline_dir(cfg.paths.built_dir, ctx.run_id, tctx.task))
    run_path, one_path = bdir / "bench_run.json", bdir / "bench_1t.json"
    if reuse:
        run_doc, one_doc = read_json(run_path), read_json(one_path)
        if run_doc is None or one_doc is None:
            raise BaselineError(f"--no-baseline: 기존 기준선이 없다 ({run_path}, {one_path})")
        info(f"기준선 재사용: {run_path}, {one_path}")
    else:
        info(f"{tctx.task}: run 시작 기준선 측정 (threads={m.threads}, cores={m.cores}) …")
        run_doc, _ = run_baseline_bench(ctx, tctx, threads=m.threads, out_path=run_path)
        min_gflops = m.baseline_min_gflops if tctx.spec.baseline.get("gate_min_gflops") else None
        ok, why = baseline_gate(run_doc, min_gflops, require_openblas=_gate_opts(tctx))
        if not ok:
            raise BaselineError(
                f"run 시작 기준선 게이트 실패: {why}. 원인 후보: NumPy 가 참조 BLAS(netlib)에 링크됨(시스템 python3-numpy), "
                f"격리 venv 가 아님, 다른 부하. 확인: `{PYTHON_EXE} -m harness.protocol --probe` 의 numpy_blas, "
                f"`{PYTHON_EXE} -m harness.protocol --check`."
            )
        info(f"{tctx.task}: run 시작 기준선 median {run_doc['median_ms']:.2f} ms{_rate(run_doc)}")
        info(f"{tctx.task}: 1스레드 기준선 측정 (threads=1, core={m.single_core}) …")
        one_doc, _ = run_baseline_bench(ctx, tctx, threads=1, out_path=one_path)
        ok1, why1 = baseline_gate(one_doc, None, require_openblas=_gate_opts(tctx))
        if not ok1:
            raise BaselineError(f"1스레드 기준선 실패: {why1}")
        info(f"{tctx.task}: 1스레드 기준선 median {one_doc['median_ms']:.2f} ms{_rate(one_doc)}")
    tctx.baseline_run = run_doc
    tctx.baseline_1t = one_doc


def adopt_existing_baselines(ctx: RunContext, tctx: TaskContext) -> None:
    """처리할 샘플이 없어 기준선을 다시 재지 않을 때: 디스크의 run 시작·1스레드 기준선이 둘 다 있으면 그대로 읽어 둔다.

    다시 재면 디스크의 기준선 파일이 이미 쓴 행의 증빙과 어긋난다. 읽어 둔 값은 준비 뒤에 생기는 raw(수정 턴)를
    ``process_sample`` 이 처리할 때 그 행의 ``baseline_run_time_ms``·``baseline_time_1t_ms`` 가 된다(없으면 NA).
    """
    bdir = baseline_dir(ctx.cfg.paths.built_dir, ctx.run_id, tctx.task)
    run_doc, one_doc = read_json(bdir / "bench_run.json"), read_json(bdir / "bench_1t.json")
    if run_doc is not None and one_doc is not None:
        tctx.baseline_run, tctx.baseline_1t = run_doc, one_doc


def missing_tools(cfg: Config, toolchain: dict[str, Any], languages: Iterable[str], *, baseline_c: bool) -> list[str]:
    """처리할 언어(와 빌드할 C 기준선)에 필요한데 쓸 수 없는 도구(설계 v3 §10.6). 문제마다 한 줄, 없으면 빈 리스트.

    없는 도구로 생긴 실패가 ``compile_error`` 행이 되면 pass 율이 오염되므로 행을 쓰기 전에 가려낸다.
    ``toolchain`` 은 ``protocol.toolchain_identity`` 의 결과다.
    """
    errors = toolchain.get("errors") or {}
    needed_by: dict[str, list[str]] = {}  # 문제 → 그 도구가 필요한 언어

    def require(available: bool, problem: str, who: str) -> None:
        if not available:
            needed_by.setdefault(problem, []).append(who)

    def why(key: str) -> str:
        return str(errors.get(key) or errors.get("python") or "사유 없음")

    no_cc = f"C 컴파일러 `{toolchain['cc']}` 를 실행할 수 없다 (build.cc)"
    for name in sorted(languages):
        lang = cfg.languages.get(name)
        if lang is None:
            continue
        if lang.backend == "c":
            require(toolchain["cc_version"] is not None, no_cc, name)
        elif lang.backend == "rust":
            cargo = config_option(cfg, "build.rust").cargo
            require(toolchain["cargo"] is not None, f"`{cargo}` 를 실행할 수 없다 (build.rust.cargo)", name)
            require(toolchain["rustc"] is not None, "`rustc` 를 실행할 수 없다", name)
        elif lang.backend == "numba":
            require(toolchain["numba_version"] is not None, f"{PYTHON_EXE} 에서 numba 를 임포트할 수 없다 ({why('numba')})", name)
        elif lang.backend == "halide":
            has_halide = toolchain["halide_version"] is not None
            require(has_halide, f"{PYTHON_EXE} 에서 halide 를 임포트할 수 없다 ({why('halide')})", name)
            if has_halide and config_option(lang, "halide_mode") == "auto":
                require(toolchain["halide_autoscheduler_loadable"] is True,
                        f"Halide 오토스케줄러 {toolchain['halide_autoscheduler']} 의 플러그인을 로드할 수 없다 "
                        f"({why('halide_autoscheduler')})", name)
    if baseline_c:
        require(toolchain["cc_version"] is not None, no_cc, "C 기준선")
    return [f"{problem} — 필요한 곳: {', '.join(who)}" for problem, who in needed_by.items()]


def probe_launch_prefixes(cfg: Config, thread_counts: Iterable[int]) -> list[str]:
    """run 이 쓸 실행 접두(``protocol.launch_prefix``; taskset)마다 ``[*접두, PYTHON_EXE, -c, pass]`` 를 한 번 돌려 본다. 문제마다
    한 줄(없으면 빈 리스트). 접두가 비면(taskset 없음, 비 Linux) 건너뛴다.

    ``measurement.cores``·``single_core`` 가 이 머신에 없는 코어를 가리키면 taskset 이 모든 명령을 rc 1 로 끝낸다 — 빌드는
    ``compile_error``, 워커는 ``runtime_crash`` 행이 되므로 행을 쓰기 전에 가려낸다(기준선 측정이 없는 ``--skip-bench`` 도)."""
    problems: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for threads in sorted(set(thread_counts)):
        prefix = protocol.launch_prefix(cfg, threads)
        if not prefix or tuple(prefix) in seen:
            continue
        seen.add(tuple(prefix))
        m = cfg.measurement
        setting = f"measurement.single_core={m.single_core}" if threads == 1 else f"measurement.cores={m.cores}"
        cmd = [*prefix, str(PYTHON_EXE), "-c", "pass"]
        try:
            r = run_isolated(cmd, timeout_s=PREFIX_PROBE_TIMEOUT_S, cwd=ROOT, env=protocol.launch_env_minimal())
        except OSError as e:
            problems.append(f"실행 접두 `{' '.join(prefix)}` 를 실행할 수 없다 ({setting}): {e}")
            continue
        if r.rc != 0 or r.timed_out:
            problems.append(f"`{' '.join(cmd)}` 가 실패했다 (rc={r.rc}, timed_out={r.timed_out}) — {setting} 가 이 머신의 CPU 와 "
                            f"맞는지 확인하라: {r.stderr_tail.strip()[-300:] or '(stderr 없음)'}")
    return problems


def probe_language_runtimes(cfg: Config, languages: Iterable[str]) -> list[str]:
    """halide_auto 의 오토스케줄러 매개변수와 numba 의 스레딩 계층을 언어의 실행 접두·환경으로 띄운 서브프로세스에서 워커와 같은
    코드로 써 본다(``protocol.runtime_probe_subprocess``). 문제마다 그 설정을 적은 한 줄(없으면 빈 리스트).

    틀린 매개변수(Halide 22.1 은 모르는 키에서 프로세스를 중단한다)나 띄울 수 없는 ``NUMBA_THREADING_LAYER`` 는 첫 샘플의
    워커에서 ``worker_error`` (exit 5)로 드러나는데, 둘 다 설정 문제이므로 행을 쓰기 전에 도구 문제(exit 7)로 멈춘다."""
    problems: list[str] = []
    seen: set[tuple[Any, ...]] = set()
    for name in sorted(languages):
        lang = cfg.languages.get(name)
        if lang is None:
            continue
        lang_env = dict(config_option(lang, "env"))
        kwargs: dict[str, Any] = {}
        if lang.backend == "halide" and config_option(lang, "halide_mode") == "auto":
            kind = "halide"
            kwargs = {"autoscheduler": str(config_option(cfg, "halide.autoscheduler")),
                      "params": dict(config_option(cfg, "halide.autoscheduler_params")), "threads": lang.threads}
            setting = (f"halide.autoscheduler={kwargs['autoscheduler']} "
                       f"halide.autoscheduler_params={json.dumps(kwargs['params'], sort_keys=True)} (threads={lang.threads})")
        elif lang.backend == "numba":
            kind = "numba"
            setting = f"languages.{name}.env NUMBA_THREADING_LAYER={lang_env.get('NUMBA_THREADING_LAYER', '(없음: numba 기본값)')}"
        else:
            continue
        env = protocol.launch_env(cfg, lang.threads, openmp=False, extra_env=lang_env)
        prefix = protocol.launch_prefix(cfg, lang.threads)
        key = (kind, json.dumps(kwargs, sort_keys=True), tuple(sorted(env.items())), tuple(prefix))
        if key in seen:
            continue
        seen.add(key)
        error = protocol.runtime_probe_subprocess(kind, env=env, prefix=prefix, **kwargs)
        if error is not None:
            problems.append(f"{setting} 를 쓸 수 없다: {error} — 필요한 곳: {name}")
    return problems


def probe_rust_toolchain(cfg: Config, run_id: str, languages: Iterable[str]) -> None:
    """rust 언어가 있으면 ``harness.build.rust.probe_toolchain`` 으로 자리표시 크레이트를 run 의 공유 cargo target 디렉터리에서
    샘플 빌드와 같은 설정·접두로 한 번 빌드한다(의존성 빌드는 샘플이 재사용한다). 실패는 ``ToolchainMissing``."""
    seen: set[tuple[str, ...]] = set()
    for name in sorted(languages):
        lang = cfg.languages.get(name)
        if lang is None or lang.backend != "rust":
            continue
        prefix = protocol.launch_prefix(cfg, lang.threads)
        if tuple(prefix) in seen:
            continue
        seen.add(tuple(prefix))
        from harness.build.rust import probe_toolchain  # 쓰는 자리에서 임포트(build_kernel 과 같다)

        probe_toolchain(config_option(cfg, "build.rust"), cfg.paths.built_dir / run_id / CARGO_TARGET_DIR, launch_prefix=prefix)


def setup_run(
    cfg: Config,
    run_id: str,
    *,
    skip_bench: bool = False,
    force: bool = False,
    strict_protocol: bool = False,
    no_baseline: bool = False,
    tasks: list[str] | None = None,
    languages: set[str] | None = None,
    samples: set[int] | None = None,
) -> RunContext:
    """run 하나의 준비(설계 v3 §10.1): 정적 규약 → 처리 대상 raw → 도구 점검(§10.6) → 환경 지문 → 과제별 테스트·기준선.

    ``tasks``·``languages``·``samples`` 는 처리 대상 raw 의 필터다(None = 전부). 찾은 raw 는 ``ctx.raw_files`` 에 처리 순서대로
    들어 있고, 과제가 지원하지 않는 언어의 raw 는 경고와 함께 빠진다. 준비 뒤에 생긴 raw(수정 턴)도 같은 ``ctx`` 로
    ``process_sample`` 에 넘길 수 있다. 실패는 전부 ``RunSetupError`` (``exit_code`` = 종료 코드)이고 행은 하나도 쓰지 않는다.
    """
    machine = load_machine()
    static = protocol.check_static(cfg, machine)
    if not static.blas_ok:
        raise RunSetupError(f"중단: NumPy BLAS 가 OpenBLAS 가 아니다 ({static.blas}). 격리 venv 에 pip 휠 numpy 를 설치하라.",
                            EXIT_PROTOCOL)
    real_flags = [f for f in static.flags if not protocol.is_informational(f)]
    if real_flags:
        warn(f"정적 규약 플래그 {real_flags} — 행의 protocol_flags 에 기록하고 계속한다")
        if strict_protocol:
            raise RunSetupError(f"--strict-protocol: 플래그가 있어 중단한다 (exit {EXIT_PROTOCOL})", EXIT_PROTOCOL)

    found = find_raw_files(cfg, run_id, languages, samples, tasks=tasks)
    if not found:
        raise RunSetupError(f"raw 파일이 없다: {cfg.paths.raw_dir / run_id} — 먼저 `python -m harness.generate --run-id {run_id}` "
                            "를 실행하라.", EXIT_ERROR)
    try:
        specs = {task: load_taskspec(TASKS_DIR / task / "signatures.yaml") for task in sorted({raw_task(p) for p in found})}
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise RunSetupError(f"중단: 과제 명세를 읽을 수 없다: {e}", EXIT_ERROR) from None
    unsupported = {(raw_task(p), raw_language(p)) for p in found if not specs[raw_task(p)].supports(raw_language(p))}
    for task, language in sorted(unsupported):
        warn(f"{task} 는 {language} 를 지원하지 않는다({specs[task].unsupported[language]}) — 그 raw 는 처리하지 않는다")
    raw_files = [p for p in found if (raw_task(p), raw_language(p)) not in unsupported]

    toolchain = protocol.toolchain_identity(cfg)
    fingerprint, env_hash = protocol.environment_fingerprint(cfg, machine, static, toolchain=toolchain)
    ctx = RunContext(cfg=cfg, machine=machine, static=static, fingerprint=fingerprint, env_hash=env_hash, run_id=run_id,
                     skip_bench=skip_bench, force=force, version=harness_version(), raw_files=raw_files)
    if ctx.version.startswith("nogit-"):
        info(f"harness_version={ctx.version} (git 저장소가 아님)")

    tasks_present = sorted({raw_task(p) for p in raw_files})
    # 처리할 샘플이 하나도 없는 과제는 기준선을 다시 재지 않는다(디스크의 기준선 파일이 행의 증빙과 어긋나지 않게).
    pending = {task: any(needs_processing(ctx, task, p) for p in raw_files if raw_task(p) == task) for task in tasks_present}
    builds_c_baseline = not skip_bench and any(pending[t] and specs[t].baseline["backend"] == "c" for t in tasks_present)
    problems = missing_tools(cfg, toolchain, {raw_language(p) for p in raw_files}, baseline_c=builds_c_baseline)
    if problems:
        raise RunSetupError("중단: 필요한 도구가 없다. 없는 도구는 모델의 오류가 아니므로 행을 쓰지 않는다 "
                            f"(exit {EXIT_TOOLCHAIN}):\n" + "\n".join(f"  - {p}" for p in problems), EXIT_TOOLCHAIN)
    # 도구가 있어도 쓸 수 있는지 한 번씩 돌려 본다(행을 쓰기 전): 실행 접두(taskset 코어) → 언어 런타임 설정 → Rust 툴체인
    if raw_files:
        run_languages = {raw_language(p) for p in raw_files}
        thread_counts = {cfg.languages[n].threads for n in run_languages if n in cfg.languages}
        if not skip_bench:  # 기준선은 measurement.threads 와 1스레드로 잰다
            thread_counts |= {cfg.measurement.threads, 1}
        for probe in (lambda: probe_launch_prefixes(cfg, thread_counts), lambda: probe_language_runtimes(cfg, run_languages)):
            problems = probe()
            if problems:
                raise RunSetupError("중단: 도구·설정을 쓸 수 없다. 모델의 오류가 아니므로 행을 쓰지 않는다 "
                                    f"(exit {EXIT_TOOLCHAIN}):\n" + "\n".join(f"  - {p}" for p in problems), EXIT_TOOLCHAIN)
        try:
            probe_rust_toolchain(cfg, run_id, run_languages)
        except ToolchainMissing as e:
            raise RunSetupError(f"중단: 도구 없음 (Rust 툴체인 점검): {e}", EXIT_TOOLCHAIN) from None

    try:
        for task in tasks_present:
            test_dir, manifest = ensure_tests(cfg, task)
            tctx = TaskContext(task=task, test_dir=test_dir, manifest=manifest,
                               tolerance_path=TASKS_DIR / task / "tolerance.yaml", signatures=load_signatures(task),
                               spec=specs[task])
            if not tctx.tolerance_path.exists():
                raise FileNotFoundError(f"허용오차 파일이 없다: {tctx.tolerance_path}")
            if not skip_bench:
                if pending[task]:
                    prepare_baselines(ctx, tctx, reuse=no_baseline)
                else:
                    info(f"{task}: 처리할 샘플이 없어(모두 최신) 기준선 측정을 건너뜀 — 기존 "
                         f"{rel_to_root(baseline_dir(cfg.paths.built_dir, run_id, task))} 유지")
                    adopt_existing_baselines(ctx, tctx)
            ctx.tasks[task] = tctx
    except HarnessError as e:
        raise RunSetupError(f"harness_error (기준선): {e}", EXIT_HARNESS) from None
    except BaselineError as e:
        raise RunSetupError(f"중단: {e}", EXIT_BASELINE) from None
    except ToolchainMissing as e:
        raise RunSetupError(f"중단: 도구 없음 (기준선 빌드): {e}", EXIT_TOOLCHAIN) from None
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        raise RunSetupError(f"중단: {e}", EXIT_ERROR) from None
    return ctx


# ----------------------------------------------------------------------------- 샘플 처리 (4–9)


def sample_ref(ctx: RunContext, task: str, raw: dict[str, Any]) -> SampleRef:
    """raw 의 내용(언어·조건·샘플 번호·턴)이 가리키는 산출물 위치. 디렉터리를 만들지 않는다."""
    cfg = ctx.cfg
    cond = raw.get("conditions") or {}
    defaults = cfg.run.conditions
    language = str(raw.get("language") or "")
    tag = cond_tag(str(cond.get("args") or defaults.args), str(cond.get("grammar_hint") or defaults.grammar_hint),
                   str(cond.get("feedback") or defaults.feedback))
    sample_idx, turn = int(raw["sample_idx"]), int(raw.get("turn") or 0)
    return SampleRef(sample_dir=built_turn_dir(cfg.paths.built_dir, ctx.run_id, task, language, tag, sample_idx, turn),
                     label=sample_label(task, language, tag, sample_idx, turn))


def base_row(ctx: RunContext, tctx: TaskContext, raw: dict[str, Any], raw_path: Path, sample_dir: Path,
             lang: LanguageCfg, env: dict[str, str]) -> RunRow:
    """raw + 정적 컨텍스트로 채울 수 있는 열을 전부 채운 행."""
    cfg = ctx.cfg
    cond = raw.get("conditions") or {}
    sampling = raw.get("sampling") or {}
    static = ctx.static
    machine = ctx.machine
    fp = ctx.fingerprint
    return RunRow(
        # run_id 는 처리 중인 run(raw 를 찾은 디렉터리)이다 — 산출물 위치(sample_ref)와 같은 출처. raw 안의 run_id 가 다르면
        # (다른 run 의 raw 를 복사해 다시 잰 경우) process_sample 이 경고하고 result.json 의 raw_run_id 에 남긴다.
        run_id=ctx.run_id, created_at=now_iso(), task=tctx.task, language=lang.name,
        backend=lang.backend, args=str(cond.get("args") or cfg.run.conditions.args),
        grammar_hint=str(cond.get("grammar_hint") or cfg.run.conditions.grammar_hint),
        feedback=str(cond.get("feedback") or cfg.run.conditions.feedback),
        sample_idx=int(raw["sample_idx"]), turn=int(raw.get("turn") or 0),
        provider=raw.get("provider"), model=raw.get("model_requested"), model_served=raw.get("model_served"),
        sampling=dict(sampling) if sampling else None, effort=sampling.get("effort"), max_tokens=raw.get("max_tokens"),
        temperature=sampling.get("temperature_sent"), seed=sampling.get("seed"), prompt_hash=raw.get("prompt_hash"),
        stop_reason=raw.get("stop_reason"), stop_details_category=raw.get("stop_details_category"),
        tokens_in=raw.get("tokens_in"), tokens_out=raw.get("tokens_out"), tokens_thinking=raw.get("tokens_thinking"),
        tokens_text=raw.get("tokens_text"), n_thinking_blocks=raw.get("n_thinking_blocks"), latency_ms=raw.get("latency_ms"),
        api_error=raw.get("error"), valid_sample=True,
        contract_violation=[], passed=False, turns_to_pass=None, error_type="", error_stage="", error_detail="",
        bench_ran=False, threads=lang.threads,
        cores=cfg.measurement.single_core if lang.threads == 1 else cfg.measurement.cores,
        omp_env=omp_env_of(env), taskset_used=bool(protocol.launch_prefix(cfg, lang.threads)),
        governor=static.governor, governor_ok=static.governor_ok, epp=static.epp, no_turbo=static.no_turbo,
        protocol_flags=list(static.flags), protocol_ok=static.ok, protocol_checked_at=static.checked_at,
        hostname=machine.get("hostname") or socket.gethostname(), cpu_model=machine.get("cpu_model") or fp.get("cpu_model"),
        python_version=fp.get("python_version"), numpy_version=fp.get("numpy_version"), numpy_blas=blas_string(static.blas),
        sdk_version=raw.get("sdk_version"), test_seed=tctx.manifest.get("seed"), test_hash=tctx.manifest.get("test_hash"),
        env_hash=ctx.env_hash, raw_path=rel_to_root(raw_path), built_path=rel_to_root(sample_dir), harness_version=ctx.version,
    )


def add_flag(row: RunRow, flag: str) -> None:
    flags = list(row.protocol_flags or [])
    if flag not in flags:
        flags.append(flag)
    row.protocol_flags = flags


def add_violation(row: RunRow, item: str) -> None:
    items = list(row.contract_violation or [])
    if item not in items:
        items.append(item)
    row.contract_violation = items


def finalize_row(ctx: RunContext, row: RunRow, sample_dir: Path, raw_sha256: str, stages: dict[str, Any],
                 meta: dict[str, Any] | None = None) -> Path:
    """pass 파생 열을 정리하고 result.json 저장 + CSV upsert. result.json 경로를 돌려준다.

    result.json 에는 이 처리가 요청받은 선택 단계(``requested``: ``requested_stages``)도 남는다 — 그 단계 없이 쓴 결과를 다음
    run 이 최신으로 보지 않게(``is_up_to_date``). ``meta`` 는 그 밖의 최상위 키(예: ``raw_run_id``)다."""
    if row.error_type:
        row.passed = bool(row.passed) and row.error_type == "timeout" and row.error_stage == "bench"
    row.turns_to_pass = row.turn if row.passed else None
    row.protocol_ok = protocol.flags_ok(row.protocol_flags or [])
    result_path = sample_dir / RESULT_JSON
    atomic_write_json(result_path, {"row": row.to_dict(), "raw_sha256": raw_sha256, "stages": stages,
                                    "requested": requested_stages(ctx), **(meta or {}), "written_at": now_iso()})
    csv_path = ctx.csv_for(row.provider)
    replaced = upsert_row(csv_path, row)
    info(f"{row_label(row)}: pass={int(bool(row.passed))} "
         f"error={row.error_stage}/{row.error_type}{(' ' + row.error_detail) if row.error_detail else ''} "
         f"→ {rel_to_root(csv_path)} ({'교체' if replaced else '추가'})")
    return result_path


def generation_gate(raw: dict[str, Any]) -> tuple[str, bool] | None:
    """§8-5 생성 게이트: 응답을 쓸 수 없으면 (error_detail, valid_sample), 쓸 수 있으면 None.

    API 오류와 max_tokens 절단은 유효 샘플이 아니고(pass@k 분모에서 빠진다), 거절은 유효한 실패다.
    """
    if raw.get("error"):
        return "api_error", False
    if raw.get("stop_reason") == "max_tokens":
        return "truncated", False
    if raw.get("stop_reason") == "refusal":
        return f"refusal:{raw.get('stop_details_category') or ''}", True
    return None


def build_kernel(ctx: RunContext, tctx: TaskContext, lang: LanguageCfg, code: str, sample_dir: Path,
                 bundle: dict[str, Any]) -> BuildResult:
    """백엔드에 맞는 빌드 함수로 ``code`` 를 ``sample_dir`` 에 빌드한다(설계 v3 §6). 컴파일러·cargo 가 없으면 ``ToolchainMissing``.

    키워드 인자 조건이면 ``bundle["args_struct"]`` (하네스가 주는 매개변수 구조체)가 C 는 ``kernel_args.h``, Rust 는
    ``src/kernel_args.rs`` 로 같이 놓인다.
    """
    cfg = ctx.cfg
    function = tctx.spec.function
    prefix = protocol.launch_prefix(cfg, lang.threads)
    args_struct = bundle["args_struct"] or None
    if lang.backend == "c":
        return build_c(code, sample_dir, cfg.build, openmp=lang.openmp, launch_prefix=prefix, symbol=function,
                       args_header=args_struct)
    if lang.backend == "python":
        return build_python(code, sample_dir)
    # 새 백엔드의 빌드 모듈은 쓰는 자리에서 임포트한다(그 언어를 돌리지 않는 run 은 모듈이 없어도 된다)
    if lang.backend == "numba":
        from harness.build.numba import build_numba

        return build_numba(code, sample_dir, function=function, expected_signature=bundle["numba_signature"])
    if lang.backend == "halide":
        from harness.build.halide import build_halide

        return build_halide(code, sample_dir, function=function, halide_mode=str(config_option(lang, "halide_mode")))
    if lang.backend == "rust":
        from harness.build.rust import build_rust

        return build_rust(code, sample_dir, config_option(cfg, "build.rust"), launch_prefix=prefix, symbol=function,
                          args_module=args_struct, target_dir=cfg.paths.built_dir / ctx.run_id / CARGO_TARGET_DIR)
    raise ValueError(f"빌드 방법을 모르는 backend: {lang.backend!r} (언어 {lang.name})")


def build_flags_text(cfg: Config, lang: LanguageCfg, sample_dir: Path) -> str | None:
    """행의 ``cflags``: 그 백엔드의 성능을 정하는 빌드 설정. python·numba 는 없다(None).

    - c: base_flags + cflags (+ openmp_flag) + ldflags.
    - rust: ``RUSTFLAGS`` 와, 그 빌드가 쓴 ``Cargo.toml`` 의 ``[profile.release]`` (TOML 표기 그대로 ``키=값``).
    - halide: auto 는 ``autoscheduler=<이름> params=<워커가 오토스케줄러에 넘기는 매개변수의 정규 JSON>`` (하네스가 더하는
      ``parallelism`` 포함; ``harness.kernels.autoscheduler_params``), manual 은 ``autoscheduler=none``.
    """
    if lang.backend == "c":
        b = cfg.build
        return " ".join([*b.base_flags, *b.cflags, *([b.openmp_flag] if lang.openmp else []), *b.ldflags])
    if lang.backend == "halide":
        if config_option(lang, "halide_mode") != "auto":
            return "autoscheduler=none"
        from harness.kernels import KernelOptions, autoscheduler_params  # 하네스 코드(LLM 코드 아님): 워커와 같은 매개변수

        name = str(config_option(cfg, "halide.autoscheduler"))
        params = autoscheduler_params(KernelOptions(threads=lang.threads, halide_mode="auto", halide_autoscheduler=name,
                                                    halide_params=dict(config_option(cfg, "halide.autoscheduler_params"))))
        return f"autoscheduler={name} params={json.dumps(params, sort_keys=True, separators=(',', ':'))}"
    if lang.backend == "rust":
        text = "RUSTFLAGS=" + " ".join(config_option(cfg, "build.rust").rustflags)
        manifest = sample_dir / "Cargo.toml"
        try:
            profile = tomllib.loads(manifest.read_text(encoding="utf-8"))["profile"]["release"]
        except (OSError, tomllib.TOMLDecodeError, KeyError) as e:
            warn(f"{manifest} 에서 [profile.release] 를 읽을 수 없다({type(e).__name__}: {e}); cflags 에는 RUSTFLAGS 만 남는다")
            return text
        return text + "; profile.release: " + " ".join(f"{key}={json.dumps(value)}" for key, value in profile.items())
    return None


def check_tests_summary(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """check 결과의 테스트별 요약(§11 check_tests)."""
    out: list[dict[str, Any]] = []
    for t in doc.get("tests") or []:
        runs = t.get("runs") or []
        elapsed = [r.get("elapsed_ms") for r in runs if r.get("elapsed_ms") is not None]
        out.append({
            "name": t.get("name"), "classification": t.get("classification"), "pattern": t.get("pattern") or "",
            "max_tol_ratio": t.get("max_tol_ratio"), "elapsed_ms": (sum(elapsed) / len(elapsed)) if elapsed else None,
            "nondeterministic": bool(t.get("nondeterministic")),
        })
    return out


def apply_check_doc(row: RunRow, doc: dict[str, Any]) -> None:
    """check 결과 문서(완전/부분)에서 행의 관찰 열을 채운다."""
    row.check_tests = check_tests_summary(doc)
    row.check_max_abs_err = doc.get("max_abs_err")
    row.check_max_tol_ratio = doc.get("max_tol_ratio")
    if doc.get("inputs_mutated") is not None:
        row.inputs_mutated = bool(doc.get("inputs_mutated"))
    if doc.get("oob_write") is not None:
        row.oob_write = bool(doc.get("oob_write"))
    if doc.get("blas"):
        row.numpy_blas = blas_string(doc.get("blas"))


def check_failure(doc: dict[str, Any]) -> tuple[str, str] | None:
    """끝까지 돈 check 문서의 판정: 통과면 None, 아니면 (error_type, error_detail)."""
    if doc.get("classification") == "pass" and not doc.get("inputs_mutated") and not doc.get("oob_write"):
        return None
    error_type = str(doc.get("classification") or "wrong_output")
    if error_type == "pass":  # classification 은 pass 인데 입력 변조/OOB 만 있는 경우(방어)
        return "wrong_output", "inputs_mutated" if doc.get("inputs_mutated") else "out_of_bounds_write"
    return error_type, str(doc.get("error_detail") or "")


def apply_bench_doc(row: RunRow, doc: dict[str, Any]) -> None:
    """bench 결과 문서(완전/부분)에서 시간·주파수·온도 열을 채운다(timeout 행도 부분값 기록)."""
    row.time_samples_ms = {"warmup": list(doc.get("warmup_ms") or []), "timed": list(doc.get("timed_ms") or [])}
    row.warmup = doc.get("warmup")
    row.repeats = doc.get("repeats")
    row.repeats_reduced = bool(doc.get("repeats_reduced")) if doc.get("repeats_reduced") is not None else None
    row.cpu_mhz_median = doc.get("cpu_mhz_median")
    row.cpu_mhz_min = doc.get("cpu_mhz_min")
    row.throttle_events = doc.get("throttle_events")
    row.temp_c_start = doc.get("temp_c_start")
    row.temp_c_end = doc.get("temp_c_end")
    row.nondeterministic_runs = doc.get("nondeterministic_runs")
    row.nondeterministic_bitwise_runs = doc.get("nondeterministic_bitwise_runs")
    if doc.get("inputs_mutated"):
        row.inputs_mutated = True
    if doc.get("oob_write"):
        row.oob_write = True


def apply_bench_times(row: RunRow, doc: dict[str, Any]) -> None:
    row.time_median_ms = doc.get("median_ms")
    row.time_mean_ms = doc.get("mean_ms")
    row.time_min_ms = doc.get("min_ms")
    row.time_max_ms = doc.get("max_ms")
    row.time_cv = doc.get("cv")
    row.gflops = doc.get("gflops")


STAGE_REQUESTS = ("bench", "safety", "lint")  # result.json 의 ``requested`` 에 남는 선택 단계
# 빠졌으면 그 샘플을 자동으로 다시 처리하는 단계. safety·lint 는 넣지 않는다: 다시 처리하면 빌드·check·bench 를 전부 다시 해
# 이미 측정한 행의 시간이 조용히 바뀐다(설정을 나중에 켰다는 이유만으로). 그 경우는 알리기만 하고 ``--force`` 에 맡긴다.
REPROCESS_STAGES = ("bench",)


def requested_stages(ctx: RunContext) -> dict[str, bool]:
    """이 처리가 요청한 선택 단계: bench(``--skip-bench`` 가 아님), safety·lint(설정)."""
    return {"bench": not ctx.skip_bench, "safety": bool(config_option(ctx.cfg, "safety.enabled")),
            "lint": bool(config_option(ctx.cfg, "lint.enabled"))}


def stored_stages(existing: dict[str, Any]) -> dict[str, bool]:
    """result.json 을 쓸 때 요청됐던 선택 단계. ``requested`` 가 없는 예전 result.json 은 행과 단계 기록이 보여 주는 것으로
    추정한다: bench 는 실행됐거나(bench_ran) 필요 없었고(실패한 행), sanitizer·lint 는 단계 기록이나 NA 가 아닌 열이 있으면
    돌았다."""
    req = existing.get("requested")
    if isinstance(req, dict):
        return {stage: bool(req.get(stage)) for stage in STAGE_REQUESTS}
    row = existing.get("row") if isinstance(existing.get("row"), dict) else {}
    stages = existing.get("stages") if isinstance(existing.get("stages"), dict) else {}
    return {"bench": bool(row.get("bench_ran")) or not row.get("passed"),
            "safety": "sanitize" in stages or row.get("asan_findings") is not None or row.get("ubsan_findings") is not None,
            "lint": "lint" in stages or any(row.get(k) is not None for k in row if str(k).startswith("lint_"))}


def stage_applies(stage: str, row: dict[str, Any]) -> bool:
    """그 단계가 이 행에서 돌았을 것인가: bench 는 check 를 통과한 행, sanitizer 는 check 까지 간 C 행(check 시간 초과 제외),
    lint 는 소스 파일이 생긴 행(extract 뒤). 돌 수 없는 단계가 빠진 결과는 다시 처리해도 같으므로 최신이다."""
    error_stage = str(row.get("error_stage") or "")
    if stage == "bench":
        return bool(row.get("passed"))
    if stage == "safety":
        return (row.get("backend") == "c" and error_stage not in ("generate", "extract", "build")
                and not (error_stage == "check" and row.get("error_type") == "timeout"))
    return error_stage not in ("generate", "extract")


def missing_stages(ctx: RunContext, existing: dict[str, Any]) -> list[str]:
    """이 run 이 요청했는데 저장된 결과가 그 단계 없이 쓰였고, 그 행에서 돌았을 단계들."""
    row = existing.get("row") if isinstance(existing.get("row"), dict) else {}
    want, had = requested_stages(ctx), stored_stages(existing)
    return [stage for stage in STAGE_REQUESTS if want[stage] and not had[stage] and stage_applies(stage, row)]


def is_up_to_date(ctx: RunContext, existing: dict[str, Any] | None, raw_sha256: str) -> bool:
    """그 턴의 ``result.json`` (``existing``)을 다시 처리하지 않아도 되는가(``needs_processing`` 과 ``process_sample`` 의 같은 규칙):
    ``--force`` 가 아니고, 같은 raw(``raw_sha256``)로 썼고, 이 run 이 요청한 bench 가 빠지지 않았다(``REPROCESS_STAGES``) —
    ``--skip-bench`` 로 쓴 통과 행은 bench 를 요청한 run 에서 다시 처리한다. safety·lint 만 빠진 결과(설정을 나중에 켠 경우)는
    최신으로 본다: 다시 처리하면 측정까지 다시 하므로 ``process_sample`` 이 알리기만 하고 ``--force`` 에 맡긴다."""
    if ctx.force or not existing or existing.get("raw_sha256") != raw_sha256:
        return False
    return not any(stage in REPROCESS_STAGES for stage in missing_stages(ctx, existing))


def needs_processing(ctx: RunContext, task: str, raw_path: Path) -> bool:
    """process_sample 의 건너뜀 조건(``is_up_to_date``)을 미리 평가한다. 디렉터리를 만들지 않는다.
    --force 면 항상 True. 읽을 수 없는 raw·설정에 없는 언어는 process_sample 이 무시하므로 False."""
    if ctx.force:
        return True
    raw = read_json(raw_path)
    if raw is None or str(raw.get("language") or "") not in ctx.cfg.languages:
        return False
    existing = read_json(sample_ref(ctx, task, raw).sample_dir / RESULT_JSON)
    return not is_up_to_date(ctx, existing, sha256_file(raw_path))


def process_sample(ctx: RunContext, tctx: TaskContext, raw_path: Path) -> tuple[RunRow | None, str]:
    """raw 하나(샘플의 한 턴)를 끝까지 처리한다 → (행, 상태). 상태 ∈ processed | skipped | ignored.

    행을 쓰지 않고 올리는 예외: ``HarnessError`` (워커가 하네스 문제로 끝남), ``ToolchainMissing`` (빌드 도구나 오토스케줄러가
    없음) — 둘 다 생성된 코드의 잘못이 아니므로 행으로 남기지 않는다.
    """
    from harness.extract import extract  # 지연 임포트: --help/--rebuild-csv 가 extract 없이도 돌게

    cfg = ctx.cfg
    raw = read_json(raw_path)
    if raw is None:
        warn(f"raw 를 읽을 수 없어 건너뜀: {raw_path}")
        return None, "ignored"
    language = str(raw.get("language") or "")
    lang = cfg.languages.get(language)
    if lang is None:
        warn(f"설정에 없는 언어 {language!r}: {raw_path} 건너뜀")
        return None, "ignored"
    if not tctx.spec.supports(language):
        warn(f"{tctx.task} 는 {language} 를 지원하지 않는다({tctx.spec.unsupported[language]}): {raw_path} 건너뜀")
        return None, "ignored"
    ref = sample_ref(ctx, tctx.task, raw)
    label = ref.label
    sample_dir = ensure_dir(ref.sample_dir)
    raw_sha256 = sha256_file(raw_path)

    existing = read_json(sample_dir / RESULT_JSON)
    if is_up_to_date(ctx, existing, raw_sha256):
        not_run = missing_stages(ctx, existing or {})
        if not_run:  # safety·lint 를 나중에 켠 경우: 그 열은 NA 로 남는다(자동으로 다시 측정하지 않는다)
            info(f"{label}: result.json 이 최신이라 건너뜀 — {', '.join(not_run)} 없이 쓰인 결과다(그 열은 NA; 채우려면 --force, "
                 "측정도 다시 한다)")
        else:
            info(f"{label}: result.json 이 최신이라 건너뜀 (--force 로 다시)")
        prev = (existing or {}).get("row")
        return (RunRow.from_dict(prev) if isinstance(prev, dict) else None), "skipped"
    if existing and existing.get("raw_sha256") == raw_sha256 and not ctx.force:
        info(f"{label}: result.json 이 {', '.join(missing_stages(ctx, existing))} 없이 쓰였다 — 다시 처리한다")

    if raw.get("backend") and raw.get("backend") != lang.backend:
        warn(f"{raw_path}: raw backend={raw.get('backend')!r} 와 설정 backend={lang.backend!r} 가 다르다; 설정을 쓴다")
    meta: dict[str, Any] = {}
    raw_run_id = raw.get("run_id")
    if raw_run_id is not None and str(raw_run_id) != ctx.run_id:
        # 행의 run_id 는 처리 중인 run 이다(base_row). raw 의 값은 result.json 에만 남긴다(CSV 열은 없다).
        meta["raw_run_id"] = str(raw_run_id)
        key = f"raw_run_id:{raw_run_id}"
        if key not in ctx.warned:
            ctx.warned.add(key)
            warn(f"raw 의 run_id={raw_run_id!r} 가 처리 중인 run {ctx.run_id!r} 와 다르다(복사한 raw?) — 행은 {ctx.run_id!r} 로 "
                 "기록하고 원래 값은 result.json 의 raw_run_id 에 남긴다")

    openmp = lang.backend == "c" and lang.openmp
    lang_env = dict(config_option(lang, "env"))
    row = base_row(ctx, tctx, raw, raw_path, sample_dir, lang,
                   protocol.launch_env(cfg, lang.threads, openmp=openmp, extra_env=lang_env))
    stages: dict[str, Any] = {}
    m = cfg.measurement
    asan_kind = ""  # sanitizer 가 낸 첫 ASan 보고의 종류(memory_error 규칙의 detail)

    def done() -> tuple[RunRow, str]:
        finalize_row(ctx, row, sample_dir, raw_sha256, stages, meta)
        return row, "processed"

    def fail(stage: str, error_type: str, detail: str) -> tuple[RunRow, str]:
        if error_type in MEMORY_ERROR_FROM and (row.asan_findings or 0) > 0:
            error_type, detail = "memory_error", f"asan:{asan_kind}; was {error_type}/{detail}"
        row.error_stage, row.error_type, row.error_detail, row.passed = stage, error_type, detail, False
        return done()

    # 5. 생성 게이트
    gate = generation_gate(raw)
    if gate is not None:
        row.valid_sample = gate[1]
        return fail("generate", "", gate[0])

    # 6-a. extract
    args_mode = row.args or "positional"
    bundle = signature_bundle(tctx.spec, language, lang.backend, args_mode)
    ext = extract(str(raw.get("response_text") or ""), backend=lang.backend,
                  signature_spec=signature_spec_for(tctx.signatures, language), args_mode=args_mode, work_dir=sample_dir,
                  bundle=bundle, cc=cfg.build.cc)
    row.extract_ok, row.signature_ok, row.signature_exact = ext.ok, ext.signature_ok, ext.signature_exact
    stages["extract"] = {"ok": ext.ok, "signature_ok": ext.signature_ok, "signature_exact": ext.signature_exact,
                         "detail": ext.detail, "n_blocks": ext.n_blocks}
    if not ext.ok or ext.code is None:
        return fail("extract", "signature_mismatch", ext.detail or "no_code")
    code = ext.code

    # 6-b. build (+ lint: 소스 파일이 생겼으므로 빌드가 실패해도 돌린다)
    br = build_kernel(ctx, tctx, lang, code, sample_dir, bundle)
    row.build_ok, row.build_time_ms, row.n_warnings = br.ok, br.build_time_ms, br.n_warnings
    row.compiler, row.compiler_version = br.compiler, br.compiler_version or None
    row.cflags = build_flags_text(cfg, lang, sample_dir)
    row.march_resolved = br.march_resolved or None
    row.undefined_symbols = list(br.undefined_symbols)
    for s in br.forbidden_symbols:
        add_violation(row, f"forbidden_symbol:{s}")
    if br.thread_symbols and lang.backend == "c" and not lang.openmp:
        add_violation(row, "threads_in_c_seq")
    if lang.backend == "halide" and config_option(lang, "halide_mode") == "auto" and br.extra.get("schedule_calls"):
        add_violation(row, "schedule_in_halide_auto")
    bd = br.to_dict()
    bd.pop("log", None)
    stages["build"] = bd
    if config_option(cfg, "lint.enabled"):
        lint = safety.run_lint(
            cfg, language=language, backend=lang.backend, file=sample_dir / SOURCE_FILE[lang.backend], task=tctx.task,
            signature={"function": bundle["function"], "params": bundle["params"], "signature": bundle["signature"],
                       "args_mode": args_mode},
            out=sample_dir / "lint.json", cwd=sample_dir)
        for column, value in safety.lint_columns(lint).items():
            setattr(row, column, value)
        stages["lint"] = asdict(lint)
    if not br.ok:
        return fail("build", "signature_mismatch" if br.error_kind == "signature_mismatch" else "compile_error",
                    "compile_timeout" if br.error_kind == "timeout" else (br.error_detail or br.error_kind))
    assert br.artifact is not None
    backend_args = worker_backend_args(cfg, lang)

    def worker(stage: str, mode: str, tests: list[Path], timeout_s: float, extra: list[str] | None = None, *,
               artifact: Path | None = None, env_extra: dict[str, str] | None = None,
               max_mem_gb: float | None = None) -> WorkerOutcome:
        """이 샘플의 워커 실행(``safety.run_sanitizers`` 도 이것으로 sanitizer 빌드를 돌린다; 설계 v3 §10.3)."""
        oc = run_worker(cfg, stage=stage, threads=lang.threads, openmp=openmp, backend=WORKER_BACKEND[lang.backend],
                        artifact=artifact or br.artifact, mode=mode, tests=tests, tolerance=tctx.tolerance_path,
                        signatures=tctx.spec.path, out_path=sample_dir / f"{stage}.json", cwd=sample_dir, timeout_s=timeout_s,
                        args_mode=args_mode, extra=extra, env_extra={**lang_env, **(env_extra or {})},
                        backend_args=backend_args, max_mem_gb=max_mem_gb)
        stages[stage] = oc.summary()
        if oc.stdout_bytes > 0:
            add_violation(row, "stdout_output")
        if oc.doc and oc.doc.get("blas"):
            row.numpy_blas = blas_string(oc.doc.get("blas"))
        return oc

    # 6-c. import (로드 실패는 build 단계 오류로 기록한다). halide 는 첫 정확성 테스트 크기로 파이프라인을 만들어 본다.
    import_tests = [tctx.test_path(m.correctness_tests[0])] if lang.backend == "halide" else []
    oc = worker("import", "import", import_tests, m.check_timeout_s, inspect_args(cfg, tctx))
    verdict = oc.verdict()
    idoc = oc.doc or {}
    if not idoc.get("load_ok"):
        if verdict is not None:
            return fail("build", *verdict)
        detail = str(idoc.get("load_error") or idoc.get("error_detail") or "load failed")
        if detail.startswith(AUTOSCHEDULER_UNAVAILABLE):
            raise ToolchainMissing(f"{label}: 워커가 Halide 오토스케줄러 플러그인을 로드하지 못했다 ({detail})")
        return fail("build", str(idoc.get("classification") or "compile_error"), detail)
    if verdict is not None:
        # 커널은 이미 로드됐다(load_ok). import 모드가 그 뒤에 하는 일은 관찰값 수집(--inspect; halide 는 성능 형태로 파이프라인을
        # 다시 만든다)뿐이고 관찰은 판정이 아니다 — 여기서 실패로 적으면 하네스가 보지 못한 것이 커널의 빌드 오류가 된다.
        # 요약만 관찰값 없이 만들고 판정은 check·bench 에 맡긴다(성능 형태에서 정말 죽는 커널은 bench 가 같은 자리에서 잡는다).
        warn(f"{label}: import 워커가 커널을 로드한 뒤 끝까지 가지 못했다 ({verdict[0]}: {verdict[1]}) — "
             "관찰값(--inspect) 없이 계속한다")

    if config_option(cfg, "perf_summary.enabled"):
        summary = perf_summary.summarize(language=language, backend=lang.backend, code=code,
                                         build_log=read_text(sample_dir / BUILD_LOG), task=tctx.task, spec=tctx.spec,
                                         inspect=idoc.get("inspect"), build_extra=br.extra)
        row.perf_summary = summary
        notes = str(summary.get("notes") or "")
        stages["perf_summary"] = {"ok": not notes.startswith(perf_summary.ERROR_NOTE_PREFIX), "notes": notes}

    # 6-d. check (+ sanitizer: C 커널만, check 가 시간 초과로 끝났으면 계측 빌드는 같은 상한에 또 걸리므로 건너뛴다)
    tests = [tctx.test_path(n) for n in m.correctness_tests]
    oc = worker("check", "check", tests, m.check_timeout_s, ["--check-repeats", str(m.check_repeats)])
    if oc.doc:
        apply_check_doc(row, oc.doc)
    failure = oc.verdict() or check_failure(oc.doc or {})
    if config_option(cfg, "safety.enabled") and lang.backend == "c" and not (failure and failure[0] == "timeout"):
        san = safety.run_sanitizers(cfg, code=code, sample_dir=sample_dir, openmp=openmp,
                                    launch_prefix=protocol.launch_prefix(cfg, lang.threads), symbol=tctx.spec.function,
                                    args_header=bundle["args_struct"] or None, tests=tests, worker=worker)
        row.asan_findings, row.ubsan_findings = san.asan_findings, san.ubsan_findings
        stages["sanitize"] = asdict(san)
        if (san.asan_findings or 0) > 0:
            asan_kind = san.kinds[0]
    if failure is not None:
        return fail("check", *failure)
    row.passed = True
    if ctx.skip_bench:
        return done()

    # 7. 동적 규약 검사 → 대응 기준선 → 샘플 bench
    dyn = protocol.check_dynamic(cfg)
    stages["dynamic"] = asdict(dyn)
    row.load_1m, row.load_ok = dyn.load_1m, dyn.load_ok
    row.pinned_busy_pct, row.sibling_busy_pct = dyn.pinned_busy_pct, dyn.sibling_busy_pct
    row.governor, row.governor_ok = dyn.governor, dyn.governor_ok
    row.protocol_checked_at = dyn.checked_at
    for f in dyn.flags:
        add_flag(row, f)

    row.baseline_run_time_ms = (tctx.baseline_run or {}).get("median_ms")
    row.baseline_time_1t_ms = (tctx.baseline_1t or {}).get("median_ms")
    paired_doc, poc = run_baseline_bench(ctx, tctx, threads=m.threads, out_path=sample_dir / PAIRED_BASELINE_JSON)
    stages["paired_baseline"] = poc.summary()
    ok, why = baseline_gate(paired_doc, None, require_openblas=_gate_opts(tctx))
    if ok and paired_doc is not None:
        row.baseline_time_ms = paired_doc.get("median_ms")
        row.baseline_time_cv = paired_doc.get("cv")
        row.baseline_samples_ms = {"warmup": list(paired_doc.get("warmup_ms") or []), "timed": list(paired_doc.get("timed_ms") or [])}
        # baseline_drift_pct 와 drift 플래그는 run 끝에서 정한다(patch_baseline_drift: 과제별 대응 기준선 중앙값 기준).
    else:
        warn(f"{label}: 대응 기준선 실패({why}); speedup_vs_baseline 은 NA")
        add_flag(row, FLAG_PAIRED_FAILED)

    oc = worker("bench", "bench", [tctx.test_path(m.performance_test)], m.bench_timeout_s, bench_args(cfg))
    row.bench_ran = True
    if oc.doc:
        apply_bench_doc(row, oc.doc)
    bdoc = oc.doc or {}
    cls = str(bdoc.get("classification") or "")
    # 워커가 시간 상한에 걸렸거나, 워커 스스로 첫 호출로 예산 초과를 내다보고 그만둔 경우(classification=timeout)
    verdict = oc.verdict() or (("timeout", str(bdoc.get("error_detail") or "")) if cls == "timeout" else None)
    if verdict is not None and verdict[0] == "timeout":
        # bench timeout 만 pass 유지(성능 실패이지 정확성 실패가 아님)
        row.error_stage, (row.error_type, row.error_detail) = "bench", verdict
        return done()
    if verdict is not None:
        return fail("bench", *verdict)
    apply_bench_times(row, bdoc)
    if cls != "pass" or bdoc.get("inputs_mutated") or bdoc.get("oob_write") or (bdoc.get("nondeterministic_runs") or 0) > 0:
        return fail("bench", cls if cls and cls != "pass" else "wrong_output", str(bdoc.get("error_detail") or ""))
    if row.time_median_ms:
        if row.baseline_time_ms is not None:
            row.speedup_vs_baseline = row.baseline_time_ms / row.time_median_ms
        if row.baseline_time_1t_ms is not None:
            row.speedup_vs_baseline_1t = row.baseline_time_1t_ms / row.time_median_ms
    return done()


# ----------------------------------------------------------------------------- raw 탐색·요약 (4, 10)


def find_raw_files(cfg: Config, run_id: str, languages: set[str] | None = None, samples: set[int] | None = None, *,
                   tasks: list[str] | None = None) -> list[Path]:
    """``raw_dir/<run_id>/<task>/<language>/<cond_tag>/`` 의 ``sample_<i>.json`` (턴 0)과 ``sample_<i>.turn<t>.json`` (수정 턴).

    ``.prevN.json`` 은 뺀다. 필터(None = 전부)를 적용하고 과제·언어·조건 태그·샘플 번호·턴 순으로 정렬한다 — 한 샘플의 턴은
    순서대로 이어서 처리된다.
    """
    base = cfg.paths.raw_dir / run_id
    if not base.exists():
        return []
    keyed: list[tuple[tuple[str, str, str, int, int], Path]] = []
    for p in base.glob("*/*/*/sample_*.json"):
        parsed = parse_raw_turn_name(p.name)
        if parsed is None:
            continue
        sample_idx, turn = parsed
        task, language, tag = raw_task(p), raw_language(p), p.parent.name
        if task not in cfg.run.tasks:
            warn(f"설정 run.tasks 에 없는 과제 {task!r}: {p} 건너뜀")
            continue
        if tasks is not None and task not in tasks:
            continue
        if languages is not None and language not in languages:
            continue
        if samples is not None and sample_idx not in samples:
            continue
        keyed.append(((task, language, tag, sample_idx, turn), p))
    return [p for _, p in sorted(keyed)]


def print_summary(rows: list[RunRow], n_skipped: int) -> None:
    """과제·언어·조건(수정 턴은 턴별로 따로)마다 pass/오류 유형/생성 실패/플래그 수를 stdout 에 출력한다.
    조건 태그는 행들의 조건이 둘 이상일 때만 이름에 적는다(조건이 하나인 run 의 출력은 과제/언어 그대로다)."""
    several_tags = len({row_tag(r) for r in rows}) > 1
    groups: dict[str, list[RunRow]] = {}
    for r in rows:
        name = f"{r.task}/{r.language}" + (f" {row_tag(r)}" if several_tags else "") + (f" turn {r.turn}" if r.turn else "")
        groups.setdefault(name, []).append(r)
    print(f"run 요약: 행 {len(rows)}개 (건너뜀 {n_skipped}개)")
    for name in sorted(groups):
        rs = groups[name]
        n_pass = sum(1 for r in rs if r.passed)
        n_gen = sum(1 for r in rs if r.error_stage == "generate")
        n_flag = sum(1 for r in rs if not protocol.flags_ok(r.protocol_flags or []))
        errors = Counter(f"{r.error_stage}/{r.error_type or r.error_detail}" for r in rs if r.error_stage)
        err_desc = ", ".join(f"{k}:{v}" for k, v in sorted(errors.items())) or "-"
        times = [r.time_median_ms for r in rs if r.time_median_ms is not None]
        speed = [r.speedup_vs_baseline for r in rs if r.speedup_vs_baseline is not None]
        t_desc = f" median_ms={sorted(times)[len(times) // 2]:.2f}" if times else ""
        s_desc = f" speedup={sorted(speed)[len(speed) // 2]:.2f}" if speed else ""
        print(f"  {name:20s} n={len(rs)} pass={n_pass} generate_failed={n_gen} flagged={n_flag} errors=[{err_desc}]{t_desc}{s_desc}")


def patch_baseline_drift(ctx: RunContext, entries: list[tuple[RunRow, Path]]) -> None:
    """drift 를 **과제별 대응 기준선들의 중앙값** 기준으로 다시 계산한다(2026-11-05 사용자 승인으로 규약 변경).

    예전 기준(run 시작 1회 측정)은 그 한 번이 느린 상태에서 재지면 거의 모든 행에 플래그가 붙었다. 넘겨받은 모든 행
    (이번에 건너뛴 행 포함)을 대상으로 ``baseline_drift_pct`` 와 ``baseline_drift=…%`` 플래그를 갱신하고, 값이 바뀐 행만
    result.json/CSV 를 다시 쓴다."""
    limit = ctx.cfg.measurement.baseline_drift_pct_max
    by_task: dict[str, list[float]] = {}
    for row, _ in entries:
        if row.baseline_time_ms is not None:
            by_task.setdefault(str(row.task), []).append(float(row.baseline_time_ms))
    for row, result_path in entries:
        vals = sorted(by_task.get(str(row.task), []))
        if row.baseline_time_ms is None or not vals:
            continue
        mid = len(vals) // 2
        median = vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0
        drift = (float(row.baseline_time_ms) - median) / median * 100.0 if median > 0 else 0.0
        flags = [f for f in (row.protocol_flags or []) if not str(f).startswith(DRIFT_FLAG_PREFIX)]
        if abs(drift) > limit:
            flags.append(f"{DRIFT_FLAG_PREFIX}{drift:+.1f}%")
            warn(f"{row_label(row)}: 대응 기준선 drift {drift:+.1f}% > {limit:g}% (과제 중앙값 {median:.2f} ms)")
        old_drift = row.baseline_drift_pct
        if flags == list(row.protocol_flags or []) and old_drift is not None and abs(old_drift - drift) < 1e-9:
            continue
        row.baseline_drift_pct = drift
        row.protocol_flags = flags
        row.protocol_ok = protocol.flags_ok(flags)
        data = read_json(result_path) or {}
        data["row"] = row.to_dict()
        atomic_write_json(result_path, data)
        upsert_row(ctx.csv_for(row.provider), row)


def patch_model_served_mixed(ctx: RunContext, processed: list[tuple[RunRow, Path]]) -> None:
    """model_served 가 단일값이 아니면 경고하고 넘겨받은 행들에 플래그를 붙여 result.json/CSV 를 갱신한다."""
    served = {str(r.model_served) for r, _ in processed if r.model_served}
    if len(served) <= 1:
        return
    warn(f"model_served 가 단일값이 아니다: {sorted(served)} → 플래그 {FLAG_MODEL_MIXED}")
    for row, result_path in processed:
        if FLAG_MODEL_MIXED in (row.protocol_flags or []):
            continue
        add_flag(row, FLAG_MODEL_MIXED)
        row.protocol_ok = protocol.flags_ok(row.protocol_flags or [])
        data = read_json(result_path) or {}
        data["row"] = row.to_dict()
        atomic_write_json(result_path, data)
        upsert_row(ctx.csv_for(row.provider), row)


def finish_run(ctx: RunContext, processed: list[tuple[RunRow, Path]]) -> None:
    """샘플 처리가 끝난 뒤의 행 보정(설계 v3 §10.1). ``processed`` 는 (행, 그 ``result.json`` 경로 — ``result_json_path``)이고
    건너뛴 행도 넣는다: 과제별 대응 기준선 중앙값 기준의 drift 플래그, ``model_served`` 혼재 플래그."""
    patch_baseline_drift(ctx, processed)
    patch_model_served_mixed(ctx, processed)


# ----------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m harness.run", description="추출·빌드·실행·측정 오케스트레이션 (설계 §8)")
    parser.add_argument("--run-id", default=None, help="results/raw/<run_id> 의 run (--rebuild-csv 외에는 필수)")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--languages", default=None, help="쉼표 구분 언어 필터")
    parser.add_argument("--samples", default=None, help="쉼표 구분 sample_idx 필터 (예: 0,1)")
    parser.add_argument("--skip-bench", action="store_true", help="check 까지만 (bench·대응 기준선 생략)")
    parser.add_argument("--no-baseline", action="store_true", help="기존 기준선 파일 재사용 (없으면 오류)")
    parser.add_argument("--force", action="store_true", help="result.json 이 최신이어도 다시 처리")
    parser.add_argument("--strict-protocol", action="store_true", help="정적 규약 플래그가 있으면 중단 (exit 4)")
    parser.add_argument("--rebuild-csv", action="store_true", help="built_dir 의 모든 result.json 에서 CSV 재생성 후 종료")
    return parser


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    # SIGTERM/SIGHUP 도 KeyboardInterrupt 로 받아 run_isolated 가 워커 그룹을 SIGKILL 하고 풀리게 한다(설계 §7).
    # 기본 처리(즉시 종료)로 죽으면 새 세션의 워커가 고정 코어를 계속 점유해 다음 run 의 측정을 오염시킨다.
    install_terminate_handlers()
    try:
        return _main(ns)
    except KeyboardInterrupt as e:
        print(f"{_LOG} 중단됨 ({e or 'SIGINT'}); 실행 중이던 서브프로세스 그룹은 정리했다 (exit {EXIT_INTERRUPTED})", file=sys.stderr)
        return EXIT_INTERRUPTED


def _main(ns: argparse.Namespace) -> int:
    """``main`` 의 본체(시그널 처리 바깥)."""
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return EXIT_CONFIG

    if ns.rebuild_csv:
        counts = rebuild_csvs(cfg.paths.built_dir, cfg.paths.results_csv, cfg.paths.results_csv_dummy)
        for path, n in counts.items():
            print(f"{path}: {n}행")
        return EXIT_OK
    if not ns.run_id:
        print(f"{_LOG} --run-id 가 필요하다 (--rebuild-csv 제외)", file=sys.stderr)
        return EXIT_CONFIG

    languages = {s.strip() for s in ns.languages.split(",") if s.strip()} if ns.languages else None
    samples = {int(s) for s in ns.samples.split(",") if s.strip()} if ns.samples else None
    if languages:
        unknown = sorted(languages - set(cfg.languages))
        if unknown:
            print(f"{_LOG} 설정에 없는 언어: {unknown}", file=sys.stderr)
            return EXIT_CONFIG

    # 1–3. 정적 규약·도구 점검·과제별 테스트·기준선
    try:
        ctx = setup_run(cfg, ns.run_id, skip_bench=ns.skip_bench, force=ns.force, strict_protocol=ns.strict_protocol,
                        no_baseline=ns.no_baseline, languages=languages, samples=samples)
    except RunSetupError as e:
        print(f"{_LOG} {e}", file=sys.stderr)
        return e.exit_code

    # 4–9. 샘플 처리
    processed: list[tuple[RunRow, Path]] = []
    n_skipped = 0
    for raw_path in ctx.raw_files:
        try:
            row, status = process_sample(ctx, ctx.tasks[raw_task(raw_path)], raw_path)
        except ToolchainMissing as e:
            print(f"{_LOG} 도구 없음 — 행을 쓰지 않고 중단한다 (raw {rel_to_root(raw_path)}): {e}", file=sys.stderr)
            return EXIT_TOOLCHAIN
        except HarnessError as e:
            print(f"{_LOG} harness_error — 행을 쓰지 않고 중단한다 (raw {rel_to_root(raw_path)}):\n{e}", file=sys.stderr)
            return EXIT_HARNESS
        except Exception:
            print(f"{_LOG} 처리 중 예외 (raw {rel_to_root(raw_path)}):\n{traceback.format_exc()}", file=sys.stderr)
            return EXIT_HARNESS
        if row is None:
            continue
        if status == "skipped":
            n_skipped += 1
        processed.append((row, result_json_path(row)))

    # 10. 보정·요약. 보정은 이 run 의 모든 행(디스크의 result.json, 수정 턴 포함)으로 한다 — 필터(--languages/--samples)로 돈
    # 호출이 그 부분집합만으로 과제별 대응 기준선 중앙값과 model_served 를 다시 정하면 측정하지 않은 행의 플래그가 바뀐다
    # (harness.repair 와 같은 기준). 요약은 이번 호출의 행만, 보정된 값으로 낸다.
    entries = [(row, result_json_path(row)) for row in read_result_rows(cfg.paths.built_dir / ns.run_id)]
    finish_run(ctx, entries)
    patched = {row.key(): row for row, _ in entries}
    print_summary([patched.get(row.key(), row) for row, _ in processed], n_skipped)
    return EXIT_OK


if __name__ == "__main__":
    # 진단은 한국어다: 파이프·파일·로캘 코드페이지 콘솔로 나갈 때 깨지지 않게 스트림을 정리한다(진입점에서만; 설계 v3 §16).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
