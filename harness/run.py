"""오케스트레이션 (설계 §8 run.py 1–10): raw 응답 → extract → build → import → check → bench → 행(§11) → CSV.

    PYTHON_EXE -m harness.run --run-id ID [--config] [--languages a,b] [--samples 0,1] [--skip-bench]
                              [--no-baseline] [--force] [--strict-protocol] [--rebuild-csv]

1. 설정·machine.yaml 로드, ``protocol.check_static()`` 1회. 플래그는 경고와 함께 계속; ``--strict-protocol`` 이면 중단(exit 4).
   BLAS 이름에 ``openblas`` 가 없으면 항상 중단(exit 4).
2. 테스트 npz 가 없으면 ``gen_tests.py`` 를 ``run_isolated`` 로 실행. manifest 에서 ``test_seed``, ``test_hash``.
3. run 시작 기준선(bench, measurement.threads) → gflops/BLAS 게이트(실패 시 exit 6) → 1스레드 기준선.
   ``results/built/<run_id>/<task>/baseline/{bench_run.json,bench_1t.json}``. ``--no-baseline`` 이면 기존 파일 재사용.
4. raw 순회(필터). ``result.json`` 의 ``raw_sha256`` 이 같고 ``--force`` 가 아니면 건너뜀.
5. 생성 게이트(API 오류 / max_tokens 절단 / refusal) → extract 없이 행만 기록.
6. extract → build → import(워커) → check → bench. 워커 종료 규칙: timed_out → timeout; rc<0 → runtime_crash(signal);
   worker_error(또는 결과 문서 없이 rc==3) → harness_error(행 없음, exit 5); 결과 문서가 있는 rc==3 은 커널의 exit(3) 로 보아 runtime_crash;
   rc==0인데 미완료 → runtime_crash; 그 밖 rc>0 → runtime_crash.
7. bench 직전: ``protocol.check_dynamic()`` → 대응 기준선(``baseline/paired_sample_<lang>_<i>.json``) → drift 플래그 → 샘플 bench.
8. pass = check 전부 pass ∧ ¬inputs_mutated ∧ ¬oob_write ∧ (bench 미실행 ∨ (final_check pass ∧ 크래시 없음 ∧ nondeterministic_runs==0)).
   bench timeout 만 pass=1·시간 NA·error_stage=bench/error_type=timeout.
9. 행 → ``result.json`` (row + 단계별 상세 + raw_sha256) → ``schema.upsert_row`` (provider==dummy 면 runs_dummy.csv).
10. run 요약. ``model_served`` 가 단일값이 아니면 경고 + 플래그 ``model_served_mixed``. ``--rebuild-csv`` 는 result.json 에서 CSV 재생성.

Ctrl-C/SIGTERM/SIGHUP: ``procutil.install_terminate_handlers()`` 가 SIGTERM·SIGHUP 을 KeyboardInterrupt 로 바꾸고,
``run_isolated`` 가 실행 중이던 워커 프로세스 그룹을 SIGKILL 한 뒤 exit 130 으로 끝난다(처리 중이던 샘플의 행은 쓰지 않음).

LLM 코드는 이 프로세스에서 절대 실행하지 않는다(extract/compile()/gcc/nm 만; 실행은 harness.worker 서브프로세스).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import socket
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness import protocol
from harness.build.c import build_c
from harness.build.python import build_python
from harness.config import Config, ConfigError, LanguageCfg, load_config, load_machine
from harness.paths import (
    PYTHON_EXE, ROOT, TASKS_DIR, atomic_write_json, baseline_dir, built_sample_dir, cond_tag, ensure_dir, rel_to_root,
)
from harness.procutil import ProcResult, install_terminate_handlers, run_isolated
from harness.protocol import StaticStatus, now_iso
from harness.schema import RunRow, harness_version, rebuild_csvs, upsert_row
from harness.taskspec import TaskSpec, load_taskspec

_LOG = "[run]"

EXIT_OK = 0
EXIT_ERROR = 1  # 입력 없음 등 일반 오류
EXIT_CONFIG = 2  # 설정 오류
EXIT_PROTOCOL = 4  # --strict-protocol 위반 또는 BLAS 가 OpenBLAS 가 아님
EXIT_HARNESS = 5  # 워커 harness_error (rc=3 / worker_error)
EXIT_BASELINE = 6  # 기준선 게이트 실패
EXIT_INTERRUPTED = 130  # Ctrl-C / SIGTERM / SIGHUP 으로 중단 (실행 중이던 서브프로세스 그룹은 정리됨)

WORKER_ERROR_RC = 3
GEN_TESTS_TIMEOUT_S = 600.0
BASELINE_FILE = "baseline.py"
RESULT_JSON = "result.json"
RAW_SAMPLE_RE = re.compile(r"^sample_(\d+)\.json$")
OMP_ENV_PREFIXES = ("OMP_", "OPENBLAS_", "MKL_", "GOMP_", "KMP_")
# 하네스 소스 상태를 CSV 에 남기는 플래그 이름
FLAG_MODEL_MIXED = "model_served_mixed"
FLAG_PAIRED_FAILED = "paired_baseline_failed"


class HarnessError(RuntimeError):
    """워커가 하네스 버그로 끝났다(rc=3 또는 worker_error). run 을 중단한다(exit 5)."""


class BaselineError(RuntimeError):
    """기준선 게이트 실패(참조 BLAS 의심 등). run 을 중단한다(exit 6)."""


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
    """run 전체에 공통인 상태."""

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

    def csv_for(self, provider: str | None) -> Path:
        return self.cfg.paths.results_csv_dummy if provider == "dummy" else self.cfg.paths.results_csv


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
        """설계 §8-6 종료 규칙. 정상 종료면 None. harness_error 면 ``HarnessError``."""
        doc = self.doc or {}
        # harness_error 는 (a) 워커가 스스로 남긴 worker_error, 또는 (b) 결과 문서를 하나도 쓰기 전에 rc=3 으로 죽은 경우에만.
        # 결과 문서(in_progress 등)가 있는데 worker_error 가 없으면 LLM 커널이 exit(3) 을 부른 것이므로 runtime_crash(rc>0 규칙).
        if doc.get("worker_error") or (self.proc.rc == WORKER_ERROR_RC and not self.doc):
            tb = str(doc.get("worker_error") or self.proc.stderr_tail)
            raise HarnessError(f"{self.stage}: worker_error (rc={self.proc.rc}) — 결과 {self.out_path}\n{tb}")
        if self.proc.timed_out:
            return "timeout", f"worker timeout after {self.timeout_s:g}s during {self.in_progress_desc()}"
        if self.proc.rc < 0:
            if doc.get("complete"):
                # 결과까지 다 쓴 뒤 인터프리터 종료 중 시그널: 가드 밖 힙 손상(음수·원거리 인덱스 쓰기)이 유력
                return "runtime_crash", f"signal {self.proc.signal_name} after complete result (heap corruption suspected)"
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
) -> WorkerOutcome:
    """``taskset`` 접두 + 화이트리스트 환경으로 ``harness.worker`` 를 띄운다. stdout/stderr 는 ``<cwd>/<stage>.*.log``."""
    cmd = [*protocol.launch_prefix(cfg, threads), str(PYTHON_EXE), "-m", "harness.worker",
           "--backend", backend, "--artifact", str(artifact), "--mode", mode,
           "--tolerance", str(tolerance), "--signatures", str(signatures), "--out", str(out_path), "--args-mode", args_mode,
           "--max-mem-gb", str(cfg.measurement.max_worker_mem_gb)]
    if tests:
        cmd += ["--tests", ",".join(str(t) for t in tests)]
    cmd += list(extra or [])
    env = protocol.launch_env(cfg, threads, openmp=openmp)
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


def baseline_artifact(ctx: RunContext, tctx: TaskContext, bdir: Path) -> Path:
    """과제 명세의 기준선 파일. python 은 그대로, c 는 run 당 한 번 ``bdir`` 에 빌드한다(샘플과 같은 플래그)."""
    b = tctx.spec.baseline
    src = TASKS_DIR / tctx.task / str(b["file"])
    if not src.exists():
        raise FileNotFoundError(f"기준선이 없다: {src}")
    if b["backend"] == "python":
        return src
    if tctx.baseline_artifact is not None and tctx.baseline_artifact.exists():
        return tctx.baseline_artifact
    cfg = ctx.cfg
    br = build_c(src.read_text(encoding="utf-8"), ensure_dir(bdir / "build"), cfg.build, openmp=bool(b.get("openmp")),
                 launch_prefix=protocol.launch_prefix(cfg, cfg.measurement.threads), symbol=tctx.spec.function)
    if not br.ok or br.artifact is None:
        raise BaselineError(f"C 기준선 빌드 실패({src}): {br.error_kind} {br.error_detail}")
    tctx.baseline_artifact = br.artifact
    return br.artifact


def run_baseline_bench(ctx: RunContext, tctx: TaskContext, *, threads: int, out_name: str) -> tuple[dict[str, Any] | None, WorkerOutcome]:
    """과제의 기준선(python 또는 빌드한 C)을 bench 한다. (결과 문서, 실행 결과). harness_error 는 예외."""
    cfg = ctx.cfg
    bdir = ensure_dir(baseline_dir(cfg.paths.built_dir, ctx.run_id, tctx.task))
    b = tctx.spec.baseline
    artifact = baseline_artifact(ctx, tctx, bdir)
    stage = Path(out_name).stem
    oc = run_worker(
        cfg, stage=stage, threads=threads, openmp=b["backend"] == "c" and bool(b.get("openmp")), backend=str(b["backend"]),
        artifact=artifact, mode="bench",
        tests=[tctx.test_path(cfg.measurement.performance_test)], tolerance=tctx.tolerance_path, signatures=tctx.spec.path,
        out_path=bdir / out_name, cwd=bdir, timeout_s=cfg.measurement.bench_timeout_s, extra=bench_args(cfg),
    )
    verdict = oc.verdict()
    if verdict is not None:
        warn(f"기준선 {out_name}: {verdict[0]} — {verdict[1]}")
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
        run_doc, _ = run_baseline_bench(ctx, tctx, threads=m.threads, out_name="bench_run.json")
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
        one_doc, _ = run_baseline_bench(ctx, tctx, threads=1, out_name="bench_1t.json")
        ok1, why1 = baseline_gate(one_doc, None, require_openblas=_gate_opts(tctx))
        if not ok1:
            raise BaselineError(f"1스레드 기준선 실패: {why1}")
        info(f"{tctx.task}: 1스레드 기준선 median {one_doc['median_ms']:.2f} ms{_rate(one_doc)}")
    tctx.baseline_run = run_doc
    tctx.baseline_1t = one_doc


# ----------------------------------------------------------------------------- 샘플 처리 (4–9)


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
        run_id=str(raw.get("run_id") or ctx.run_id), created_at=now_iso(), task=tctx.task, language=lang.name,
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


def finalize_row(ctx: RunContext, row: RunRow, sample_dir: Path, raw_sha256: str, stages: dict[str, Any]) -> Path:
    """pass 파생 열을 정리하고 result.json 저장 + CSV upsert. result.json 경로를 돌려준다."""
    row.turns_to_pass = 0 if row.passed else None
    row.protocol_ok = protocol.flags_ok(row.protocol_flags or [])
    if row.error_type:
        row.passed = bool(row.passed) and row.error_type == "timeout" and row.error_stage == "bench"
    result_path = sample_dir / RESULT_JSON
    atomic_write_json(result_path, {"row": row.to_dict(), "raw_sha256": raw_sha256, "stages": stages, "written_at": now_iso()})
    csv_path = ctx.csv_for(row.provider)
    replaced = upsert_row(csv_path, row)
    info(f"{row.task}/{row.language}/sample_{row.sample_idx}: pass={int(bool(row.passed))} error={row.error_stage}/{row.error_type}"
         f"{(' ' + row.error_detail) if row.error_detail else ''} → {rel_to_root(csv_path)} ({'교체' if replaced else '추가'})")
    return result_path


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


def needs_processing(ctx: RunContext, task: str, raw_path: Path) -> bool:
    """process_sample 의 건너뜀 조건(result.json 의 raw_sha256 일치)을 미리 평가한다. 디렉터리를 만들지 않는다.
    --force 면 항상 True. 읽을 수 없는 raw·설정에 없는 언어는 process_sample 이 무시하므로 False."""
    if ctx.force:
        return True
    raw = read_json(raw_path)
    if raw is None:
        return False
    cfg = ctx.cfg
    language = str(raw.get("language") or "")
    if language not in cfg.languages:
        return False
    cond = raw.get("conditions") or {}
    tag = cond_tag(str(cond.get("args") or cfg.run.conditions.args), str(cond.get("grammar_hint") or cfg.run.conditions.grammar_hint),
                   str(cond.get("feedback") or cfg.run.conditions.feedback))
    sample_dir = built_sample_dir(cfg.paths.built_dir, ctx.run_id, task, language, tag, int(raw["sample_idx"]))
    existing = read_json(sample_dir / RESULT_JSON)
    return not (existing and existing.get("raw_sha256") == sha256_file(raw_path))


def process_sample(ctx: RunContext, tctx: TaskContext, raw_path: Path) -> tuple[RunRow | None, str]:
    """raw 하나를 끝까지 처리한다 → (행, 상태). 상태 ∈ processed | skipped | ignored."""
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
    cond = raw.get("conditions") or {}
    tag = cond_tag(str(cond.get("args") or cfg.run.conditions.args), str(cond.get("grammar_hint") or cfg.run.conditions.grammar_hint),
                   str(cond.get("feedback") or cfg.run.conditions.feedback))
    sample_idx = int(raw["sample_idx"])
    sample_dir = ensure_dir(built_sample_dir(cfg.paths.built_dir, ctx.run_id, tctx.task, language, tag, sample_idx))
    raw_sha256 = sha256_file(raw_path)

    existing = read_json(sample_dir / RESULT_JSON)
    if existing and existing.get("raw_sha256") == raw_sha256 and not ctx.force:
        info(f"{language}/sample_{sample_idx}: result.json 이 최신이라 건너뜀 (--force 로 다시)")
        prev = existing.get("row")
        return (RunRow.from_dict(prev) if isinstance(prev, dict) else None), "skipped"

    if raw.get("backend") and raw.get("backend") != lang.backend:
        warn(f"{raw_path}: raw backend={raw.get('backend')!r} 와 설정 backend={lang.backend!r} 가 다르다; 설정을 쓴다")

    openmp = lang.backend == "c" and lang.openmp
    env = protocol.launch_env(cfg, lang.threads, openmp=openmp)
    row = base_row(ctx, tctx, raw, raw_path, sample_dir, lang, env)
    stages: dict[str, Any] = {}
    m = cfg.measurement

    # 5. 생성 게이트
    if raw.get("error"):
        row.error_stage, row.error_type, row.error_detail = "generate", "", "api_error"
        row.passed, row.valid_sample = False, False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    if raw.get("stop_reason") == "max_tokens":
        row.error_stage, row.error_type, row.error_detail = "generate", "", "truncated"
        row.passed, row.valid_sample = False, False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    if raw.get("stop_reason") == "refusal":
        row.error_stage, row.error_type = "generate", ""
        row.error_detail = f"refusal:{raw.get('stop_details_category') or ''}"
        row.passed, row.valid_sample = False, True
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"

    # 6-a. extract
    ext = extract(str(raw.get("response_text") or ""), backend=lang.backend,
                  signature_spec=signature_spec_for(tctx.signatures, language), args_mode=row.args or "positional",
                  work_dir=sample_dir)
    row.extract_ok, row.signature_ok, row.signature_exact = ext.ok, ext.signature_ok, ext.signature_exact
    stages["extract"] = {"ok": ext.ok, "signature_ok": ext.signature_ok, "signature_exact": ext.signature_exact,
                         "detail": ext.detail, "n_blocks": ext.n_blocks}
    if not ext.ok or ext.code is None:
        row.error_stage, row.error_type, row.error_detail = "extract", "signature_mismatch", ext.detail or "no_code"
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"

    # 6-b. build
    if lang.backend == "c":
        br = build_c(ext.code, sample_dir, cfg.build, openmp=openmp, launch_prefix=protocol.launch_prefix(cfg, lang.threads),
                     symbol=tctx.spec.function)
        flags = [*cfg.build.base_flags, *cfg.build.cflags, *([cfg.build.openmp_flag] if openmp else []), *cfg.build.ldflags]
        row.cflags = " ".join(flags)
        row.march_resolved = br.march_resolved or None
    else:
        br = build_python(ext.code, sample_dir)
    row.build_ok, row.build_time_ms, row.n_warnings = br.ok, br.build_time_ms, br.n_warnings
    row.compiler, row.compiler_version = br.compiler, br.compiler_version or None
    row.undefined_symbols = list(br.undefined_symbols)
    for s in br.forbidden_symbols:
        add_violation(row, f"forbidden_symbol:{s}")
    if br.thread_symbols and lang.backend == "c" and not lang.openmp:
        add_violation(row, "threads_in_c_seq")
    bd = br.to_dict()
    bd.pop("log", None)
    stages["build"] = bd
    if not br.ok:
        row.error_stage = "build"
        row.error_type = "signature_mismatch" if br.error_kind == "signature_mismatch" else "compile_error"
        row.error_detail = "compile_timeout" if br.error_kind == "timeout" else (br.error_detail or br.error_kind)
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    assert br.artifact is not None

    def worker(stage: str, mode: str, tests: list[Path], timeout_s: float, extra: list[str] | None = None) -> WorkerOutcome:
        oc = run_worker(cfg, stage=stage, threads=lang.threads, openmp=openmp, backend=lang.backend, artifact=br.artifact,
                        mode=mode, tests=tests, tolerance=tctx.tolerance_path, signatures=tctx.spec.path,
                        out_path=sample_dir / f"{stage}.json",
                        cwd=sample_dir, timeout_s=timeout_s, args_mode=row.args or "positional", extra=extra)
        stages[stage] = oc.summary()
        if oc.stdout_bytes > 0:
            add_violation(row, "stdout_output")
        if oc.doc and oc.doc.get("blas"):
            row.numpy_blas = blas_string(oc.doc.get("blas"))
        return oc

    # 6-c. import (로드 실패는 build 단계 오류로 기록한다)
    oc = worker("import", "import", [], m.check_timeout_s)
    verdict = oc.verdict()
    if verdict is not None:
        row.error_stage, (row.error_type, row.error_detail) = "build", verdict
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    idoc = oc.doc or {}
    if not idoc.get("load_ok"):
        row.error_stage = "build"
        row.error_type = str(idoc.get("classification") or "compile_error")
        row.error_detail = str(idoc.get("load_error") or idoc.get("error_detail") or "load failed")
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"

    # 6-d. check
    tests = [tctx.test_path(n) for n in m.correctness_tests]
    oc = worker("check", "check", tests, m.check_timeout_s, ["--check-repeats", str(m.check_repeats)])
    if oc.doc:
        apply_check_doc(row, oc.doc)
    verdict = oc.verdict()
    if verdict is not None:
        row.error_stage, (row.error_type, row.error_detail) = "check", verdict
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    cdoc = oc.doc or {}
    if cdoc.get("classification") != "pass" or cdoc.get("inputs_mutated") or cdoc.get("oob_write"):
        row.error_stage = "check"
        row.error_type = str(cdoc.get("classification") or "wrong_output")
        row.error_detail = str(cdoc.get("error_detail") or "")
        if row.error_type == "pass":  # classification 은 pass 인데 입력 변조/OOB 만 있는 경우(방어)
            row.error_type = "wrong_output"
            row.error_detail = "inputs_mutated" if cdoc.get("inputs_mutated") else "out_of_bounds_write"
        row.passed = False
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    row.passed = True
    if ctx.skip_bench:
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"

    # 7. 동적 규약 검사 → 대응 기준선 → 샘플 bench
    dyn = protocol.check_dynamic(cfg)
    stages["dynamic"] = asdict(dyn)
    row.load_1m, row.load_ok = dyn.load_1m, dyn.load_ok
    row.pinned_busy_pct, row.sibling_busy_pct = dyn.pinned_busy_pct, dyn.sibling_busy_pct
    row.governor, row.governor_ok = dyn.governor, dyn.governor_ok
    row.protocol_checked_at = dyn.checked_at
    for f in dyn.flags:
        add_flag(row, f)

    base_run = tctx.baseline_run or {}
    row.baseline_run_time_ms = base_run.get("median_ms")
    row.baseline_time_1t_ms = (tctx.baseline_1t or {}).get("median_ms")
    paired_doc, poc = run_baseline_bench(ctx, tctx, threads=m.threads, out_name=f"paired_sample_{language}_{sample_idx}.json")
    stages["paired_baseline"] = poc.summary()
    ok, why = baseline_gate(paired_doc, None, require_openblas=_gate_opts(tctx))
    if ok and paired_doc is not None:
        row.baseline_time_ms = paired_doc.get("median_ms")
        row.baseline_time_cv = paired_doc.get("cv")
        row.baseline_samples_ms = {"warmup": list(paired_doc.get("warmup_ms") or []), "timed": list(paired_doc.get("timed_ms") or [])}
        # baseline_drift_pct 와 drift 플래그는 run 끝에서 정한다(patch_baseline_drift: 과제별 대응 기준선 중앙값 기준).
    else:
        warn(f"{language}/sample_{sample_idx}: 대응 기준선 실패({why}); speedup_vs_baseline 은 NA")
        add_flag(row, FLAG_PAIRED_FAILED)

    oc = worker("bench", "bench", [tctx.test_path(m.performance_test)], m.bench_timeout_s, bench_args(cfg))
    row.bench_ran = True
    if oc.doc:
        apply_bench_doc(row, oc.doc)
    verdict = oc.verdict()
    if verdict is not None:
        row.error_stage, (row.error_type, row.error_detail) = "bench", verdict
        row.passed = verdict[0] == "timeout"  # bench timeout 만 pass 유지(성능 실패이지 정확성 실패가 아님)
        finalize_row(ctx, row, sample_dir, raw_sha256, stages)
        return row, "processed"
    bdoc = oc.doc or {}
    cls = str(bdoc.get("classification") or "")
    if cls == "timeout":
        row.error_stage, row.error_type, row.error_detail = "bench", "timeout", str(bdoc.get("error_detail") or "")
        row.passed = True
    elif cls != "pass" or bdoc.get("inputs_mutated") or bdoc.get("oob_write") or (bdoc.get("nondeterministic_runs") or 0) > 0:
        apply_bench_times(row, bdoc)
        row.error_stage = "bench"
        row.error_type = cls if cls and cls != "pass" else "wrong_output"
        row.error_detail = str(bdoc.get("error_detail") or "")
        row.passed = False
    else:
        apply_bench_times(row, bdoc)
        if row.time_median_ms:
            if row.baseline_time_ms is not None:
                row.speedup_vs_baseline = row.baseline_time_ms / row.time_median_ms
            if row.baseline_time_1t_ms is not None:
                row.speedup_vs_baseline_1t = row.baseline_time_1t_ms / row.time_median_ms
    finalize_row(ctx, row, sample_dir, raw_sha256, stages)
    return row, "processed"


# ----------------------------------------------------------------------------- raw 탐색·요약 (4, 10)


def find_raw_files(cfg: Config, run_id: str, languages: set[str] | None, samples: set[int] | None) -> list[Path]:
    """``raw_dir/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json`` (``.prevN.json`` 제외), 필터 적용, 정렬."""
    base = cfg.paths.raw_dir / run_id
    if not base.exists():
        return []
    out: list[Path] = []
    for p in sorted(base.glob("*/*/*/sample_*.json")):
        mm = RAW_SAMPLE_RE.match(p.name)
        if not mm:
            continue
        task, language = p.parts[-4], p.parts[-3]
        if task not in cfg.run.tasks:
            warn(f"설정 run.tasks 에 없는 과제 {task!r}: {p} 건너뜀")
            continue
        if languages is not None and language not in languages:
            continue
        if samples is not None and int(mm.group(1)) not in samples:
            continue
        out.append(p)
    return out


def print_summary(rows: list[RunRow], n_skipped: int) -> None:
    """언어별 pass/오류 유형/생성 실패/플래그 수를 stdout 에 출력한다."""
    by_lang: dict[str, list[RunRow]] = {}
    for r in rows:
        by_lang.setdefault(f"{r.task}/{r.language}", []).append(r)
    print(f"run 요약: 행 {len(rows)}개 (건너뜀 {n_skipped}개)")
    for lang in sorted(by_lang):
        rs = by_lang[lang]
        n_pass = sum(1 for r in rs if r.passed)
        n_gen = sum(1 for r in rs if r.error_stage == "generate")
        n_flag = sum(1 for r in rs if not protocol.flags_ok(r.protocol_flags or []))
        errors = Counter(f"{r.error_stage}/{r.error_type or r.error_detail}" for r in rs if r.error_stage)
        err_desc = ", ".join(f"{k}:{v}" for k, v in sorted(errors.items())) or "-"
        times = [r.time_median_ms for r in rs if r.time_median_ms is not None]
        speed = [r.speedup_vs_baseline for r in rs if r.speedup_vs_baseline is not None]
        t_desc = f" median_ms={sorted(times)[len(times) // 2]:.2f}" if times else ""
        s_desc = f" speedup={sorted(speed)[len(speed) // 2]:.2f}" if speed else ""
        print(f"  {lang:20s} n={len(rs)} pass={n_pass} generate_failed={n_gen} flagged={n_flag} errors=[{err_desc}]{t_desc}{s_desc}")


DRIFT_FLAG_PREFIX = "baseline_drift="


def patch_baseline_drift(ctx: RunContext, entries: list[tuple[RunRow, Path]]) -> None:
    """drift 를 **과제별 대응 기준선들의 중앙값** 기준으로 다시 계산한다(2026-11-05 사용자 승인으로 규약 변경).

    예전 기준(run 시작 1회 측정)은 그 한 번이 느린 상태에서 재지면 거의 모든 행에 플래그가 붙었다. 이 run 의 모든 행
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
            warn(f"{row.task}/{row.language}/sample_{row.sample_idx}: 대응 기준선 drift {drift:+.1f}% > {limit:g}% (과제 중앙값 {median:.2f} ms)")
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
    """model_served 가 단일값이 아니면 경고하고 이 run 의 행들에 플래그를 붙여 result.json/CSV 를 갱신한다."""
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

    # 1. 정적 규약
    machine = load_machine()
    static = protocol.check_static(cfg, machine)
    if not static.blas_ok:
        print(f"{_LOG} 중단: NumPy BLAS 가 OpenBLAS 가 아니다 ({static.blas}). 격리 venv 에 pip 휠 numpy 를 설치하라.", file=sys.stderr)
        return EXIT_PROTOCOL
    real_flags = [f for f in static.flags if not protocol.is_informational(f)]
    if real_flags:
        warn(f"정적 규약 플래그 {real_flags} — 행의 protocol_flags 에 기록하고 계속한다")
        if ns.strict_protocol:
            print(f"{_LOG} --strict-protocol: 플래그가 있어 중단한다 (exit {EXIT_PROTOCOL})", file=sys.stderr)
            return EXIT_PROTOCOL
    fingerprint, env_hash = protocol.environment_fingerprint(cfg, machine, static)
    ctx = RunContext(cfg=cfg, machine=machine, static=static, fingerprint=fingerprint, env_hash=env_hash, run_id=ns.run_id,
                     skip_bench=ns.skip_bench, force=ns.force, version=harness_version())
    if ctx.version.startswith("nogit-"):
        info(f"harness_version={ctx.version} (git 저장소가 아님)")

    raw_files = find_raw_files(cfg, ns.run_id, languages, samples)
    if not raw_files:
        print(f"{_LOG} raw 파일이 없다: {cfg.paths.raw_dir / ns.run_id} — 먼저 `python -m harness.generate --run-id {ns.run_id}` 를 실행하라.",
              file=sys.stderr)
        return EXIT_ERROR
    tasks_present = sorted({p.parts[-4] for p in raw_files})
    # 처리할 샘플이 하나도 없는 과제는 기준선을 다시 재지 않는다(디스크의 기준선 파일이 행의 증빙과 어긋나지 않게).
    pending = {task: any(needs_processing(ctx, task, p) for p in raw_files if p.parts[-4] == task) for task in tasks_present}

    # 2–3. 과제별 테스트·기준선
    try:
        for task in tasks_present:
            test_dir, manifest = ensure_tests(cfg, task)
            sig_path = TASKS_DIR / task / "signatures.yaml"
            tctx = TaskContext(task=task, test_dir=test_dir, manifest=manifest,
                               tolerance_path=TASKS_DIR / task / "tolerance.yaml", signatures=load_signatures(task),
                               spec=load_taskspec(sig_path))
            if not tctx.tolerance_path.exists():
                raise FileNotFoundError(f"허용오차 파일이 없다: {tctx.tolerance_path}")
            if not ns.skip_bench:
                if pending[task]:
                    prepare_baselines(ctx, tctx, reuse=ns.no_baseline)
                else:
                    info(f"{task}: 처리할 샘플이 없어(모두 최신) 기준선 측정을 건너뜀 — 기존 "
                         f"{rel_to_root(baseline_dir(cfg.paths.built_dir, ns.run_id, task))} 유지")
            ctx.tasks[task] = tctx
    except HarnessError as e:
        print(f"{_LOG} harness_error (기준선): {e}", file=sys.stderr)
        return EXIT_HARNESS
    except BaselineError as e:
        print(f"{_LOG} 중단: {e}", file=sys.stderr)
        return EXIT_BASELINE
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        print(f"{_LOG} 중단: {e}", file=sys.stderr)
        return EXIT_ERROR

    # 4–9. 샘플 처리
    processed: list[tuple[RunRow, Path]] = []
    rows: list[RunRow] = []
    n_skipped = 0
    for raw_path in raw_files:
        tctx = ctx.tasks[raw_path.parts[-4]]
        try:
            row, status = process_sample(ctx, tctx, raw_path)
        except HarnessError as e:
            print(f"{_LOG} harness_error — 행을 쓰지 않고 중단한다 (raw {rel_to_root(raw_path)}):\n{e}", file=sys.stderr)
            return EXIT_HARNESS
        except Exception:
            print(f"{_LOG} 처리 중 예외 (raw {rel_to_root(raw_path)}):\n{traceback.format_exc()}", file=sys.stderr)
            return EXIT_HARNESS
        if row is None:
            continue
        rows.append(row)
        if status == "skipped":
            n_skipped += 1
        built = Path(row.built_path or "")
        processed.append((row, (built if built.is_absolute() else ROOT / built) / RESULT_JSON))

    # 10. 요약
    patch_baseline_drift(ctx, processed)
    patch_model_served_mixed(ctx, processed)
    print_summary(rows, n_skipped)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
