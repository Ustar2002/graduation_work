"""개발용 커널 점검 도구 (설계 v3 §14). **측정 도구가 아니다** — 더미 응답·기준선·손으로 쓴 커널이 논리적으로 맞는지를
하네스의 실제 워커(``harness.worker --mode check``)로 빠르게 확인한다. 결과 행(CSV)은 쓰지 않는다.

    python tools/devcheck.py kernel --task T --language L --file PATH [--args-mode positional|keyword]
                                    [--tests tiny,rect,small] [--bench] [--keep DIR]
    python tools/devcheck.py baseline --task T [--tests …] [--bench] [--keep DIR]
    python tools/devcheck.py dummies [--tasks a,b] [--languages x,y] [--keep DIR]   # 모든 0.md → 표
    python tools/devcheck.py gen --task T                # tasks/T/gen_tests.py 를 그대로 실행(네 형태 전부)

종료 코드: **0** 전부 통과 · **1** 커널이 틀렸다(추출·빌드·로드·정확성 실패 중 하나라도) · **2** 사용법·도구 문제
(과제·파일·테스트 npz 없음, 컴파일러 없음, 아직 없는 백엔드, 워커 자체의 오류).

출력: 테스트 형태마다 한 줄(``<과제>/<언어> <형태> PASS|FAIL|NOT-RUN …``)과 마지막에 워커의 판정 한 줄
(``<과제>/<언어> => PASS|FAIL|PROBLEM stage=… classification=… [toolchain=zig-serial] detail=…``)을 stdout 에, 진단은 stderr 에
``[devcheck] …`` 로 쓴다. ``toolchain=zig-serial`` 은 그 판정이 zig 의 **직렬** 빌드로 나왔다는 표시다(아래 "C"): 그 줄의 PASS 는
논리 검증일 뿐 OpenMP 병렬 실행을 확인한 것이 아니다. ``dummies`` 의 표와 요약 줄에도 같은 표시가 나온다.

단계: (``.md`` 면) ``harness.extract`` 로 코드 추출 → 백엔드별 빌드 → 워커 check (→ ``--bench`` 면 large 1회, 참고값).

- 언어 → 백엔드: ``config/experiment.yaml`` 의 ``languages`` 에 있으면 그것을, 없으면 대응표
  ``{numpy: python, numba: numba, halide_auto: halide, halide_manual: halide, c_seq: c, c_openmp: c, rust: rust}`` 를 쓴다.
- C: Linux 에 설정의 컴파일러(gcc)가 있으면 ``harness.build.c.build_c`` 로 **설정의 실제 플래그**를 쓴다. 없으면
  ``python -m ziglang cc`` 로 빌드한다 — **직렬**이다: ``-fopenmp`` 가 없어 OpenMP pragma 는 무시되고 ``<omp.h>`` 는
  ``tools/devcheck_shim/omp.h`` 스텁이며, Windows 에 없는 ``aligned_alloc``/``posix_memalign`` 은 ``shim.h`` 가 메운다.
  즉 **논리 검증만** 한다(병렬 버그·gcc 전용 동작·POSIX 전용 API 는 Linux 에서 확인). zig 의 첫 빌드는 libc 캐시를 만드느라
  수십 초 걸린다.
- C, 실제 스레드(선택; ``--cc zig-omp`` 를 직접 줄 때만): OpenMP 로 빌드하는 대상(c_openmp 언어, ``openmp: true`` 인 C 기준선)을
  ``python -m ziglang cc -fopenmp -c`` 로 컴파일하고 이 머신의 LLVM OpenMP 런타임(Windows: Visual C++ 재배포 패키지의
  ``libomp140.x86_64.dll``; 환경변수 ``DEVCHECK_LIBOMP`` 로 다른 경로)에 링크한다. ``<omp.h>`` 는 선언만 있는
  ``tools/devcheck_shim/omp_real/omp.h`` 다. 판정 줄의 표시는 ``toolchain=zig-omp``. 스레드 수는 그 언어의 설정값이고
  ``DEVCHECK_OMP_THREADS`` 를 주면 그 값이다. 경쟁 상태·잘못된 pragma 를 일찍 드러내려는 것이고, 컴파일러(clang)도
  런타임(libomp)도 측정 경로의 gcc + libgomp 가 아니므로 Linux 확인을 대신하지 않는다. OpenMP 로 빌드하지 않는 대상(c_seq)은
  이 옵션에서도 직렬 zig 빌드다.
- Rust: ``harness.build.rust.build_rust`` (없으면 exit 2). numba·halide: 워커의 ``--backend numba`` /
  ``--backend halide --halide-mode … --threads N`` (설계 v3 §5.1).
- 워커는 ``procutil.run_isolated`` 로 띄운다(cwd = 빌드 디렉터리, ``PYTHONPATH`` = 저장소 루트). 이 프로세스는 LLM 코드를
  임포트·실행하지 않는다. 코어 고정(taskset)·측정 규약 검사는 하지 않는다: ``--bench`` 의 시간은 참고값일 뿐이다.
- 더미 점검(``dummies``)에서 지원한다고 돼 있는데(``signatures.yaml`` 의 ``unsupported`` 에 없음) ``0.md`` 가 없는 조합은
  ``missing`` 으로 실패 처리한다(설계 v3 §4: 검증된 더미가 있어야 실험 대상이다). ``--tasks``/``--languages`` 를 주지 않으면
  과제는 **과제 디렉터리(``*/signatures.yaml``) ∪ 더미 디렉터리**, 언어는 **``0.md`` 가 있는 언어 ∪ 설정의 ``languages``** 다 —
  더미 디렉터리가 통째로 없는 과제, 더미가 하나도 없는 언어도 표에 ``missing`` 으로 나온다.
- 입력 파일은 UTF-8 이어야 한다(아니면 그 파일을 이름으로 알리고 exit 2). 맨 앞의 UTF-8 BOM 은 소스 파일·raw JSON 에서는
  떼고(컴파일러와 같다), 응답 문서(``.md``)에서는 그대로 둔다(더미 provider 가 파일 내용을 그대로 응답으로 쓰기 때문; ``read_source``).
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # `python tools/devcheck.py` 로 직접 실행해도 harness 패키지를 찾게 한다
    sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from harness import sigs  # noqa: E402
from harness.paths import DEFAULT_EXPERIMENT_YAML, PYTHON_EXE, TASKS_DIR, atomic_write_text  # noqa: E402
from harness.procutil import ProcResult, run_isolated, setup_stdio  # noqa: E402
from harness.taskspec import TaskSpec, load_taskspec  # noqa: E402

_LOG = "[devcheck]"

EXIT_OK = 0
EXIT_FAIL = 1  # 커널이 틀렸다
EXIT_PROBLEM = 2  # 사용법·도구 문제

SHIM_DIR = ROOT / "tools" / "devcheck_shim"
DUMMY_DIR = ROOT / "harness" / "dummy_responses"

# config/experiment.yaml 에 없는 언어의 백엔드(설계 v3 §14)
FALLBACK_BACKENDS: dict[str, str] = {
    "numpy": "python", "numba": "numba", "halide_auto": "halide", "halide_manual": "halide",
    "c_seq": "c", "c_openmp": "c", "rust": "rust",
}
FALLBACK_HALIDE_MODE: dict[str, str] = {"halide_auto": "auto", "halide_manual": "manual"}
FALLBACK_OPENMP: dict[str, bool] = {"c_openmp": True}
FALLBACK_SINGLE_THREAD: frozenset[str] = frozenset({"c_seq"})
# 표의 언어 순서(성능 결정 부담 순; 그 밖의 언어는 이름순으로 뒤에)
LANGUAGE_ORDER: tuple[str, ...] = ("numpy", "numba", "halide_auto", "halide_manual", "c_openmp", "c_seq", "rust")

ZIG_CFLAGS: tuple[str, ...] = ("-O3", "-march=native")  # 인트린식(AVX2/FMA)을 쓰는 커널이 빌드되도록 실제 플래그와 맞춘다
ZIG_TIMEOUT_S = 600.0  # 첫 빌드는 libc 캐시를 만든다
# 판정 줄·표에 싣는 빌드 방식 표시: zig 의 직렬 빌드로 얻은 판정은 논리 검증일 뿐이다(OpenMP pragma 무시, 설정의 컴파일러 아님)
ZIG_TOOLCHAIN = "zig-serial"
# 선택 기능 ``--cc zig-omp``: clang 의 OpenMP 코드 생성 + 이 머신의 LLVM OpenMP 런타임으로 실제 스레드를 쓴다(gcc + libgomp 아님)
ZIG_OMP_TOOLCHAIN = "zig-omp"
OMP_REAL_DIR = SHIM_DIR / "omp_real"  # 런타임 함수의 선언만 있는 omp.h (스텁 SHIM_DIR/omp.h 보다 먼저 찾게 한다)
C_TOOLCHAIN_CHOICES: tuple[str, ...] = ("auto", "gcc", "zig", ZIG_OMP_TOOLCHAIN)
UTF8_BOM = "﻿"
GEN_TIMEOUT_S = 3600.0
STDERR_TAIL_LINES = 12
# 워커가 오토스케줄러 플러그인을 로드하지 못했을 때의 load_error 접두(harness.run.AUTOSCHEDULER_UNAVAILABLE 와 같다; 설계 v3 §5.3)
AUTOSCHEDULER_UNAVAILABLE = "halide_autoscheduler_unavailable:"

# Windows 에서 커널이 프로세스를 죽였을 때의 종료 코드(NTSTATUS) 이름
NTSTATUS_NAMES: dict[int, str] = {
    0xC0000005: "access violation", 0xC00000FD: "stack overflow", 0xC0000094: "integer divide by zero",
    0xC000001D: "illegal instruction", 0xC0000409: "fast-fail/abort", 0xC0000374: "heap corruption",
    0x80000003: "breakpoint",
}


class DevcheckProblem(Exception):
    """사용법·도구 문제(exit 2): 커널의 잘잘못과 무관하게 점검을 진행할 수 없다."""


def diag(msg: str) -> None:
    print(f"{_LOG} {msg}", file=sys.stderr, flush=True)


def emit(line: str) -> None:
    print(line, flush=True)


def child_env() -> dict[str, str]:
    """devcheck 가 띄우는 파이썬 자식의 공통 환경: 현재 환경 + ``PYTHONPATH`` = 저장소 루트, 표준 스트림은 UTF-8
    (자식의 로그를 UTF-8 로 읽는다)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


# ----------------------------------------------------------------------------- 설정·언어


@dataclass
class Settings:
    """devcheck 에 필요한 설정 값. ``config/experiment.yaml`` 을 느슨하게 읽는다(없는 키는 기본값) — 다른 그룹이
    설정 스키마를 고치는 중이어도 점검 도구는 돌아야 한다. gcc 경로만 ``harness.config.load_config`` 의 엄격한 값을 쓴다."""

    path: Path
    languages: dict[str, Any] = field(default_factory=dict)
    threads: int = 4
    check_repeats: int = 2
    check_timeout_s: float = 180.0
    bench_timeout_s: float = 300.0
    correctness_tests: list[str] = field(default_factory=lambda: ["tiny", "rect", "small"])
    performance_test: str = "large"
    cc: str = "gcc"
    halide_autoscheduler: str = "Mullapudi2016"
    halide_autoscheduler_params: dict[str, Any] = field(default_factory=dict)


def _section(raw: Any, key: str) -> dict[str, Any]:
    value = raw.get(key) if isinstance(raw, dict) else None
    return value if isinstance(value, dict) else {}


def load_settings(path: Path) -> Settings:
    """``experiment.yaml`` 에서 devcheck 가 쓰는 값만 읽는다. 파일이 없거나 깨졌으면 경고하고 기본값으로 진행한다."""
    path = Path(path).resolve()
    raw: Any = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        diag(f"참고: 설정 파일이 없다({path}); 기본값과 언어 대응표로 진행한다")
    except (OSError, yaml.YAMLError) as e:
        diag(f"경고: 설정 파일을 읽을 수 없다({path}: {e}); 기본값과 언어 대응표로 진행한다")
    s = Settings(path=path)
    s.languages = _section(raw, "languages")
    m = _section(raw, "measurement")
    b = _section(raw, "build")
    h = _section(raw, "halide")
    if isinstance(m.get("threads"), int) and not isinstance(m.get("threads"), bool) and m["threads"] >= 1:
        s.threads = m["threads"]
    if isinstance(m.get("check_repeats"), int) and not isinstance(m.get("check_repeats"), bool) and m["check_repeats"] >= 1:
        s.check_repeats = m["check_repeats"]
    for key in ("check_timeout_s", "bench_timeout_s"):
        value = m.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            setattr(s, key, float(value))
    tests = m.get("correctness_tests")
    if isinstance(tests, list) and tests and all(isinstance(t, str) for t in tests):
        s.correctness_tests = list(tests)
    if isinstance(m.get("performance_test"), str) and m["performance_test"]:
        s.performance_test = m["performance_test"]
    if isinstance(b.get("cc"), str) and b["cc"]:
        s.cc = b["cc"]
    if isinstance(h.get("autoscheduler"), str) and h["autoscheduler"]:
        s.halide_autoscheduler = h["autoscheduler"]
    if isinstance(h.get("autoscheduler_params"), dict):
        s.halide_autoscheduler_params = dict(h["autoscheduler_params"])
    return s


@dataclass
class LanguageInfo:
    """언어 하나의 실행 방식: 백엔드, OpenMP 여부, 스레드 수, halide 모드, 워커에 더할 환경변수."""

    name: str
    backend: str
    openmp: bool = False
    threads: int = 1
    halide_mode: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    source: str = "fallback"  # config | fallback | baseline


def _default_env(backend: str, threads: int) -> dict[str, str]:
    n = str(threads)
    return {"numba": {"NUMBA_NUM_THREADS": n}, "halide": {"HL_NUM_THREADS": n}, "rust": {"RAYON_NUM_THREADS": n}}.get(backend, {})


def resolve_language(settings: Settings, language: str) -> LanguageInfo:
    """설정에 정의된 언어면 그 항목을, 아니면 대응표를 쓴다. 어느 쪽에도 없으면 ``DevcheckProblem``."""
    entry = settings.languages.get(language)
    cpu_cap = os.cpu_count() or 1
    if isinstance(entry, dict) and isinstance(entry.get("backend"), str):
        backend = entry["backend"]
        threads_raw = entry.get("threads", "measurement")
        if threads_raw == "measurement":
            threads = settings.threads
        elif isinstance(threads_raw, int) and not isinstance(threads_raw, bool) and threads_raw >= 1:
            threads = threads_raw
        else:
            raise DevcheckProblem(f"설정 languages.{language}.threads 가 잘못됐다: {threads_raw!r}")
        threads = max(1, min(threads, cpu_cap))
        halide_mode = entry.get("halide_mode") if isinstance(entry.get("halide_mode"), str) else FALLBACK_HALIDE_MODE.get(language)
        env_raw = entry.get("env")
        env = ({str(k): str(v).replace("${threads}", str(threads)) for k, v in env_raw.items()}
               if isinstance(env_raw, dict) else _default_env(backend, threads))
        info = LanguageInfo(name=language, backend=backend, openmp=bool(entry.get("openmp", False)), threads=threads,
                            halide_mode=halide_mode, env=env, source="config")
    elif language in FALLBACK_BACKENDS:
        backend = FALLBACK_BACKENDS[language]
        threads = 1 if language in FALLBACK_SINGLE_THREAD else max(1, min(settings.threads, cpu_cap))
        info = LanguageInfo(name=language, backend=backend, openmp=FALLBACK_OPENMP.get(language, False), threads=threads,
                            halide_mode=FALLBACK_HALIDE_MODE.get(language), env=_default_env(backend, threads), source="fallback")
    else:
        known = sorted(set(settings.languages) | set(FALLBACK_BACKENDS))
        raise DevcheckProblem(f"알 수 없는 언어 {language!r} (설정·대응표에 있는 언어: {', '.join(known)})")
    if info.backend not in sigs.BACKENDS:
        raise DevcheckProblem(f"언어 {language!r} 의 백엔드 {info.backend!r} 를 모른다 (허용: {', '.join(sigs.BACKENDS)})")
    if info.backend == "halide" and info.halide_mode not in ("auto", "manual"):
        raise DevcheckProblem(f"halide 언어 {language!r} 에 halide_mode(auto|manual)가 없다")
    return info


def worker_backend_args(lang: LanguageInfo, settings: Settings) -> list[str]:
    """워커의 백엔드 인자(설계 v3 §5.1). rust 산출물은 C ABI 공유 라이브러리라 ``--backend c`` 로 실행한다."""
    if lang.backend in ("c", "rust"):
        return ["--backend", "c"]
    if lang.backend == "python":
        return ["--backend", "python"]
    if lang.backend == "numba":
        return ["--backend", "numba", "--threads", str(lang.threads)]
    args = ["--backend", "halide", "--halide-mode", str(lang.halide_mode), "--threads", str(lang.threads)]
    if lang.halide_mode == "auto":
        args += ["--halide-autoscheduler", settings.halide_autoscheduler]
        if settings.halide_autoscheduler_params:
            args += ["--halide-params", json.dumps(settings.halide_autoscheduler_params, sort_keys=True)]
    return args


def worker_env(lang: LanguageInfo) -> dict[str, str]:
    """워커 환경: 현재 환경 + ``PYTHONPATH`` = 저장소 루트 + 스레드 변수 + 언어별 변수(설정의 ``env``)."""
    env = child_env()
    n = str(lang.threads)
    env.update({"OMP_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n, "MKL_NUM_THREADS": n})
    env.update(lang.env)
    return env


# ----------------------------------------------------------------------------- 과제


@dataclass
class TaskCtx:
    task: str
    dir: Path
    spec: TaskSpec
    signatures: Path
    tolerance: Path
    tests_dir: Path

    def test_paths(self, names: list[str]) -> list[Path]:
        paths = [self.tests_dir / f"{n}.npz" for n in names]
        missing = [p.name for p in paths if not p.is_file()]
        if missing:
            raise DevcheckProblem(f"{self.task}: 테스트 입력이 없다({', '.join(missing)}; {self.tests_dir}). "
                                  f"먼저 `python tools/devcheck.py gen --task {self.task}` 를 실행하라")
        return paths


def load_task(task: str, tasks_dir: Path) -> TaskCtx:
    """과제 디렉터리의 명세·허용오차·테스트 위치. 경로는 절대 경로로 만든다(워커의 cwd 는 빌드 디렉터리다)."""
    tdir = (Path(tasks_dir) / task).resolve()
    sig = tdir / "signatures.yaml"
    tol = tdir / "tolerance.yaml"
    if not sig.is_file():
        known = sorted(p.parent.name for p in Path(tasks_dir).glob("*/signatures.yaml"))
        raise DevcheckProblem(f"과제 {task!r} 가 없다({sig}); 있는 과제: {', '.join(known) or '없음'}")
    if not tol.is_file():
        raise DevcheckProblem(f"{task}: 허용오차 파일이 없다({tol})")
    try:
        spec = load_taskspec(sig)
    except (OSError, ValueError, yaml.YAMLError) as e:
        raise DevcheckProblem(f"{task}: 과제 명세를 읽을 수 없다: {e}") from None
    return TaskCtx(task=task, dir=tdir, spec=spec, signatures=sig, tolerance=tol, tests_dir=tdir / "tests")


def split_names(text: str | None) -> list[str] | None:
    if text is None:
        return None
    names = [x.strip() for x in text.split(",") if x.strip()]
    if not names:
        raise DevcheckProblem(f"빈 목록: {text!r}")
    return names


# ----------------------------------------------------------------------------- 결과


@dataclass
class ShapeVerdict:
    name: str
    verdict: str  # PASS | FAIL | NOT-RUN
    info: str = ""


@dataclass
class Outcome:
    """점검 하나의 결과. ``status``: pass | fail(커널이 틀림) | problem(도구·사용법 문제)."""

    label: str
    status: str = "problem"
    stage: str = ""
    classification: str = ""
    detail: str = ""
    shapes: list[ShapeVerdict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    bench_ms: float | None = None
    build_dir: Path | None = None
    toolchain: str = ""  # 판정을 낸 빌드 방식 표시: zig 직렬 빌드면 ``ZIG_TOOLCHAIN``, 그 밖(설정의 gcc·cargo·python 계열)은 ""

    def exit_code(self) -> int:
        return {"pass": EXIT_OK, "fail": EXIT_FAIL}.get(self.status, EXIT_PROBLEM)

    def final_line(self) -> str:
        """최종 판정 한 줄. ``toolchain=`` 은 ``detail=``(공백을 포함하는 자유 문장, 줄 끝까지) 앞에 둔다."""
        word = {"pass": "PASS", "fail": "FAIL"}.get(self.status, "PROBLEM")
        parts = [f"{self.label} => {word}"]
        if self.stage:
            parts.append(f"stage={self.stage}")
        parts.append(f"classification={self.classification or '-'}")
        if self.toolchain:
            parts.append(f"toolchain={self.toolchain}")
        if self.detail:
            parts.append(f"detail={self.detail}")
        if self.bench_ms is not None:
            parts.append(f"bench_large_ms={self.bench_ms:.3f}")
        return " ".join(parts)


def combine_exit_codes(outcomes: list[Outcome]) -> int:
    """하나라도 틀리면 1, 틀린 것은 없지만 도구 문제가 있으면 2, 전부 통과면 0."""
    if any(o.status == "fail" for o in outcomes):
        return EXIT_FAIL
    if any(o.status != "pass" for o in outcomes):
        return EXIT_PROBLEM
    return EXIT_OK


def _one_line(text: Any, limit: int = 300) -> str:
    return " ".join(str(text).split())[:limit]


def _fmt_num(value: Any) -> str:
    return "-" if value is None else f"{float(value):.3g}"


def _dims_text(dims: Any) -> str:
    return " ".join(f"{k}={v}" for k, v in dims.items()) if isinstance(dims, dict) else ""


def exit_description(proc: ProcResult) -> str:
    """워커 프로세스의 종료를 사람이 읽을 말로: 시그널 이름(POSIX) 또는 NTSTATUS 이름(Windows)."""
    if proc.signal_name:
        return f"signal {proc.signal_name}"
    code = proc.rc & 0xFFFFFFFF
    if code in NTSTATUS_NAMES:
        return f"exit code 0x{code:08X} ({NTSTATUS_NAMES[code]})"
    return f"rc={proc.rc}"


def _stderr_excerpt(proc: ProcResult) -> str:
    """워커 stderr 꼬리에서 원인을 말해 주는 부분: ``error:`` 줄(argparse 등)이 있으면 마지막 것, 없으면 끝의 몇 줄."""
    lines = [ln.strip() for ln in proc.stderr_tail.splitlines() if ln.strip()]
    errors = [ln for ln in lines if "error:" in ln]
    if errors:
        return errors[-1]
    return " | ".join(lines[-STDERR_TAIL_LINES:])


def interpret_worker(proc: ProcResult, doc: dict[str, Any] | None, timeout_s: float) -> tuple[str, str, str] | None:
    """워커 프로세스 수준의 종료 규칙(run.py ``WorkerOutcome.verdict`` 와 같은 구분) → ``(status, classification, detail)``.
    정상 종료(완전한 결과 문서)면 None."""
    doc = doc or {}
    in_progress = doc.get("in_progress") or {}
    where = f"{in_progress.get('test') or '?'} call {in_progress.get('call_idx') if in_progress.get('call_idx') is not None else '?'}"
    if doc.get("worker_error"):
        return "problem", "harness_error", "worker_error: " + _one_line(str(doc["worker_error"]).strip().splitlines()[-1])
    if proc.timed_out:
        return "fail", "timeout", f"worker timeout after {timeout_s:g}s during {where}"
    if not doc:
        # 결과 문서를 하나도 쓰지 못했다: 인자 오류(아직 없는 백엔드)·하네스 임포트 오류 — 커널 탓이 아니다
        return "problem", "worker_failed", f"worker wrote no result ({exit_description(proc)}): {_one_line(_stderr_excerpt(proc), 600)}"
    if not doc.get("complete"):
        return "fail", "runtime_crash", f"worker died ({exit_description(proc)}) during {where}"
    if proc.rc != 0:
        return "fail", "runtime_crash", f"worker {exit_description(proc)} after a complete result"
    return None


def shape_verdicts_from_check(doc: dict[str, Any], names: list[str]) -> list[ShapeVerdict]:
    """check 결과 문서의 테스트별 판정 → 형태마다 한 줄. 문서에 없는 형태(앞에서 중단)는 NOT-RUN."""
    by_name = {str(t.get("name")): t for t in (doc.get("tests") or []) if isinstance(t, dict)}
    out: list[ShapeVerdict] = []
    for name in names:
        t = by_name.get(name)
        if t is None:
            out.append(ShapeVerdict(name, "NOT-RUN"))
            continue
        runs = [r for r in (t.get("runs") or []) if isinstance(r, dict)]
        cls = t.get("classification")
        ok = cls == "pass" and not t.get("nondeterministic")
        bits = [_dims_text(t.get("dims"))]
        if not ok:
            label = str(cls or "incomplete") + (f"/{t.get('pattern')}" if t.get("pattern") else "")
            if t.get("nondeterministic"):
                label += " nondeterministic"
            bits.append(label)
        bits.append(f"max_abs_err={_fmt_num(t.get('max_abs_err'))}")
        if t.get("max_tol_ratio") is not None:
            bits.append(f"tol_ratio={_fmt_num(t.get('max_tol_ratio'))}")
        unwritten = [r.get("n_unwritten") for r in runs if r.get("n_unwritten")]
        if unwritten:
            bits.append(f"n_unwritten={max(unwritten)}")
        elapsed = [r.get("elapsed_ms") for r in runs if r.get("elapsed_ms") is not None]
        if elapsed:
            bits.append(f"call_ms={min(elapsed):.3g}")
        if t.get("build_ms") is not None:
            bits.append(f"build_ms={_fmt_num(t.get('build_ms'))}")
        out.append(ShapeVerdict(name, "PASS" if ok else "FAIL", " ".join(b for b in bits if b)))
    return out


# ----------------------------------------------------------------------------- 빌드


@dataclass
class Built:
    ok: bool
    artifact: Path | None = None
    error_kind: str = ""
    error_detail: str = ""
    note: str = ""
    toolchain: str = ""  # ``Outcome.toolchain`` 으로 넘어간다(zig 직렬 빌드만 ``ZIG_TOOLCHAIN``)


def _from_build_result(res: Any, note: str = "") -> Built:
    return Built(ok=bool(res.ok), artifact=res.artifact, error_kind=str(res.error_kind or ""),
                 error_detail=_one_line(res.error_detail or res.error_kind or ""), note=note)


def ziglang_available() -> bool:
    return importlib.util.find_spec("ziglang") is not None


def choose_c_toolchain(settings: Settings, forced: str = "auto") -> str:
    """``gcc``(하네스의 실제 빌드) 또는 ``zig``(직렬 논리 검증). auto: Linux 에 설정의 컴파일러가 있으면 gcc, 아니면 zig.

    하네스의 C 빌드(ELF ``kernel.so``, ``nm -D``)는 Linux 전용이라 다른 플랫폼에서는 gcc 가 있어도 zig 를 쓴다."""
    has_cc = shutil.which(settings.cc) is not None
    if forced == "gcc":
        if not has_cc:
            raise DevcheckProblem(f"--cc gcc: 설정의 컴파일러 {settings.cc!r} 가 PATH 에 없다")
        return "gcc"
    if forced in ("zig", ZIG_OMP_TOOLCHAIN):
        if not ziglang_available():
            raise DevcheckProblem(f"--cc {forced}: ziglang 패키지가 이 인터프리터에 없다")
        if forced == ZIG_OMP_TOOLCHAIN:
            libomp_runtime()  # 없으면 여기서 DevcheckProblem (빌드를 시작하기 전에 알린다)
        return forced
    if has_cc and sys.platform.startswith("linux"):
        return "gcc"
    if ziglang_available():
        return "zig"
    raise DevcheckProblem(f"C 컴파일러가 없다: {settings.cc!r} 도 ziglang 패키지도 찾지 못했다 "
                          "(Linux: gcc 설치, 개발 머신: 개발용 venv 에 ziglang)")


def zig_artifact_name() -> str:
    return "kernel.dll" if os.name == "nt" else "kernel.so"


def stub_omp_threads() -> int:
    """환경변수 ``DEVCHECK_OMP_THREADS`` (기본 1): 스텁 ``omp_get_max_threads()`` 가 돌려줄 값.

    실행은 여전히 직렬이다. ``omp_get_max_threads()`` 로 조각 수를 정하는 커널(2-pass 누적 합, 조각 정렬 뒤 병합 등)의
    여러 조각 경로를 순서대로 돌려 **논리**를 확인하려는 것이다(경쟁 상태는 여전히 못 잡는다)."""
    raw = os.environ.get("DEVCHECK_OMP_THREADS", "").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
    except ValueError:
        raise DevcheckProblem(f"DEVCHECK_OMP_THREADS 는 양의 정수여야 한다: {raw!r}") from None
    if value < 1:
        raise DevcheckProblem(f"DEVCHECK_OMP_THREADS 는 양의 정수여야 한다: {raw!r}")
    return value


def zig_command(out_name: str) -> list[str]:
    """``python -m ziglang cc`` 명령(cwd = 빌드 디렉터리). ``-fopenmp`` 가 없으므로 OpenMP pragma 는 무시된다."""
    cmd = [str(PYTHON_EXE), "-m", "ziglang", "cc", "-std=gnu11", "-shared", "-Wall", "-Wno-unknown-pragmas", *ZIG_CFLAGS,
           f"-DDEVCHECK_OMP_THREADS={stub_omp_threads()}",
           "-I", str(SHIM_DIR), "-include", "shim.h", "-o", out_name, "kernel.c", "-lm"]
    if os.name != "nt":
        cmd.insert(cmd.index("-shared"), "-fPIC")
    return cmd


def libomp_runtime() -> Path:
    """``--cc zig-omp`` 가 링크할 LLVM OpenMP 런타임(공유 라이브러리). 환경변수 ``DEVCHECK_LIBOMP`` 가 우선이고, 없으면
    Windows 의 ``%SystemRoot%\\System32\\libomp140.x86_64.dll`` (Visual C++ 재배포 패키지에 들어 있다)."""
    raw = os.environ.get("DEVCHECK_LIBOMP", "").strip()
    if raw:
        path = Path(raw)
    elif os.name == "nt":
        path = Path(os.environ.get("SystemRoot") or r"C:\Windows") / "System32" / "libomp140.x86_64.dll"
    else:
        raise DevcheckProblem(f"--cc {ZIG_OMP_TOOLCHAIN}: 이 플랫폼에는 기본 OpenMP 런타임 경로가 없다 "
                              "(DEVCHECK_LIBOMP 에 LLVM libomp 공유 라이브러리의 경로를 준다; Linux 에서는 gcc 를 쓴다)")
    if not path.is_file():
        raise DevcheckProblem(f"--cc {ZIG_OMP_TOOLCHAIN}: OpenMP 런타임이 없다: {path} (DEVCHECK_LIBOMP 로 경로를 준다)")
    return path


def zig_omp_commands(out_name: str, runtime: Path) -> tuple[list[str], list[str]]:
    """실제 스레드 빌드의 두 명령(cwd = 빌드 디렉터리): ``-fopenmp -c`` 컴파일과, 런타임 파일을 입력으로 직접 주는 링크.

    링크 단계에 ``-fopenmp`` 를 주면 드라이버가 ``-lomp`` 를 찾는데 zig 에는 그 라이브러리가 없다. 그래서 목적 파일을 먼저
    만들고 런타임을 링크 입력으로 준다. ``-I omp_real`` 이 스텁 디렉터리보다 앞이라 ``<omp.h>`` 는 선언만 있는 헤더다
    (``shim.h`` 는 그대로 쓴다: Windows 에 없는 ``aligned_alloc`` 등)."""
    zig = [str(PYTHON_EXE), "-m", "ziglang", "cc"]
    compile_cmd = [*zig, "-std=gnu11", "-c", "-Wall", *ZIG_CFLAGS, "-fopenmp",
                   "-I", str(OMP_REAL_DIR), "-I", str(SHIM_DIR), "-include", "shim.h", "-o", "kernel.o", "kernel.c"]
    if os.name != "nt":
        compile_cmd.insert(compile_cmd.index("-c"), "-fPIC")
    link_cmd = [*zig, "-shared", "-o", out_name, "kernel.o", str(runtime), "-lm"]
    return compile_cmd, link_cmd


def compiler_error_lines(log: str, limit: int = 3) -> list[str]:
    """컴파일러 로그에서 ``error:`` 를 포함한 첫 ``limit`` 줄. 없으면 마지막 비어 있지 않은 ``limit`` 줄."""
    lines = [ln.strip() for ln in log.splitlines() if ln.strip()]
    errors = [ln for ln in lines if "error:" in ln]
    return errors[:limit] if errors else lines[-limit:]


def build_c_zig(code: str, build_dir: Path, args_header: str | None) -> Built:
    """``kernel.c`` (+ 키워드 모드의 ``kernel_args.h``)를 쓰고 zig 로 공유 라이브러리를 만든다(직렬 논리 검증용)."""
    atomic_write_text(build_dir / "kernel.c", code if code.endswith("\n") else code + "\n")
    if args_header:
        atomic_write_text(build_dir / sigs.C_ARGS_HEADER_NAME, args_header)
    out_name = zig_artifact_name()
    artifact = build_dir / out_name
    if artifact.exists():
        artifact.unlink()
    log_path = build_dir / "build.log"
    try:
        proc = run_isolated(zig_command(out_name), timeout_s=ZIG_TIMEOUT_S, cwd=build_dir,
                            stdout_path=build_dir / "build.stdout.log", stderr_path=log_path)
    except OSError as e:
        raise DevcheckProblem(f"zig 실행 실패: {e}") from None
    log = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    note = "zig cc, serial (OpenMP pragmas ignored)"
    if proc.timed_out:
        raise DevcheckProblem(f"zig 빌드가 {ZIG_TIMEOUT_S:g}s 안에 끝나지 않았다")
    if "ziglang" in log and "No module named" in log:
        raise DevcheckProblem(f"ziglang 패키지가 {PYTHON_EXE} 에 없다: {_one_line(log)}")
    if proc.rc != 0 or not artifact.exists():
        return Built(ok=False, error_kind="compile_error",
                     error_detail=_one_line(" | ".join(compiler_error_lines(log)) or f"rc={proc.rc}"), note=note,
                     toolchain=ZIG_TOOLCHAIN)
    return Built(ok=True, artifact=artifact, note=note, toolchain=ZIG_TOOLCHAIN)


def build_c_zig_omp(code: str, build_dir: Path, args_header: str | None) -> Built:
    """``kernel.c`` 를 zig(clang)의 ``-fopenmp`` 로 컴파일해 LLVM OpenMP 런타임에 링크한다 — 실제 스레드로 도는 공유 라이브러리.
    컴파일·링크의 진단은 둘 다 ``build.log`` 에 이어 쓴다."""
    runtime = libomp_runtime()
    atomic_write_text(build_dir / "kernel.c", code if code.endswith("\n") else code + "\n")
    if args_header:
        atomic_write_text(build_dir / sigs.C_ARGS_HEADER_NAME, args_header)
    out_name = zig_artifact_name()
    artifact = build_dir / out_name
    for stale in (artifact, build_dir / "kernel.o"):
        if stale.exists():
            stale.unlink()
    log_path = build_dir / "build.log"
    note = f"zig cc -fopenmp + LLVM OpenMP runtime {runtime.name} (real threads; not gcc/libgomp)"
    log = ""
    for step, cmd in zip(("compile", "link"), zig_omp_commands(out_name, runtime)):
        step_log = build_dir / f"build.{step}.log"
        try:
            proc = run_isolated(cmd, timeout_s=ZIG_TIMEOUT_S, cwd=build_dir,
                                stdout_path=build_dir / "build.stdout.log", stderr_path=step_log)
        except OSError as e:
            raise DevcheckProblem(f"zig 실행 실패: {e}") from None
        text = step_log.read_text(encoding="utf-8", errors="replace") if step_log.exists() else ""
        log += text
        atomic_write_text(log_path, log)
        if proc.timed_out:
            raise DevcheckProblem(f"zig {step} 가 {ZIG_TIMEOUT_S:g}s 안에 끝나지 않았다")
        if "ziglang" in text and "No module named" in text:
            raise DevcheckProblem(f"ziglang 패키지가 {PYTHON_EXE} 에 없다: {_one_line(text)}")
        if proc.rc != 0:
            return Built(ok=False, error_kind="compile_error",
                         error_detail=_one_line(" | ".join(compiler_error_lines(text)) or f"{step} rc={proc.rc}"), note=note,
                         toolchain=ZIG_OMP_TOOLCHAIN)
    if not artifact.exists():
        return Built(ok=False, error_kind="compile_error", error_detail="link produced no library", note=note,
                     toolchain=ZIG_OMP_TOOLCHAIN)
    return Built(ok=True, artifact=artifact, note=note, toolchain=ZIG_OMP_TOOLCHAIN)


def build_c_gcc(code: str, build_dir: Path, settings: Settings, lang: LanguageInfo, symbol: str, args_header: str | None) -> Built:
    """하네스의 실제 C 빌드(``harness.build.c.build_c``; 설정의 실제 플래그). 코어 고정(taskset 접두)은 하지 않는다 —
    측정이 아니고, 설정의 ``measurement.cores`` 가 이 머신과 맞지 않아도 점검은 돌아야 한다."""
    import harness.build.c as build_c_mod
    from harness.config import ConfigError, load_config

    try:
        cfg = load_config(settings.path)
    except ConfigError as e:
        raise DevcheckProblem(f"gcc 빌드에는 유효한 설정이 필요하다: {e}") from None
    kwargs: dict[str, Any] = {"openmp": lang.openmp, "launch_prefix": [], "symbol": symbol}
    if args_header:
        if "args_header" in inspect.signature(build_c_mod.build_c).parameters:
            kwargs["args_header"] = args_header
        else:  # 설계 v3 §6 의 args_header 가 아직 없는 build_c: 헤더를 직접 써 둔다
            atomic_write_text(build_dir / sigs.C_ARGS_HEADER_NAME, args_header)
    missing = getattr(build_c_mod, "ToolchainMissing", None)
    try:
        res = build_c_mod.build_c(code, build_dir, cfg.build, **kwargs)
    except OSError as e:
        raise DevcheckProblem(f"C 컴파일러 실행 실패: {e}") from None
    except Exception as e:
        if missing is not None and isinstance(e, missing):
            raise DevcheckProblem(f"C 툴체인이 없다: {e}") from None
        raise
    flags = " ".join([*cfg.build.cflags, *([cfg.build.openmp_flag] if lang.openmp else [])])
    return _from_build_result(res, note=f"{cfg.build.cc} {flags}")


def save_python_kernel(code: str, build_dir: Path, note: str = "") -> Built:
    """python 계열의 "그대로" 빌드: ``kernel.py`` 저장 + ``compile()`` 구문 검사(코드는 실행하지 않는다;
    ``harness.build.python.build_python`` 과 같은 판정). 임포트 검사는 워커가 한다."""
    artifact = build_dir / "kernel.py"
    atomic_write_text(artifact, code if code.endswith("\n") else code + "\n")
    try:
        compile(code, "kernel.py", "exec")
    except (SyntaxError, ValueError, RecursionError, MemoryError, OverflowError) as e:
        if isinstance(e, SyntaxError):
            loc = f" (line {e.lineno})" if e.lineno is not None else ""
            detail = f"{type(e).__name__}: {e.msg}{loc}"
        else:
            detail = f"{type(e).__name__}: {e}"
        return Built(ok=False, error_kind="compile_error", error_detail=_one_line(detail), note=note)
    return Built(ok=True, artifact=artifact, note=note)


def build_python_family(code: str, build_dir: Path, lang: LanguageInfo, spec: TaskSpec) -> Built:
    """python 계열: ``kernel.py`` 저장 + 구문 검사. numba·halide 는 전용 빌드 모듈(설계 v3 §6: 데코레이터 서명·스케줄 호출
    검사)이 있으면 그것을 쓰고, 아직 없으면 구문 검사만 하고 그 사실을 알린다."""
    if lang.backend == "numba":
        try:
            mod = importlib.import_module("harness.build.numba")
        except ImportError:
            return save_python_kernel(code, build_dir, note="harness.build.numba 없음: 구문 검사만(데코레이터 서명 검사 생략)")
        return _from_build_result(mod.build_numba(code, build_dir, function=spec.function,
                                                  expected_signature=sigs.numba_signature(spec)))
    if lang.backend == "halide":
        try:
            mod = importlib.import_module("harness.build.halide")
        except ImportError:
            return save_python_kernel(code, build_dir, note="harness.build.halide 없음: 구문 검사만")
        return _from_build_result(mod.build_halide(code, build_dir, function=spec.function, halide_mode=str(lang.halide_mode)))
    return save_python_kernel(code, build_dir)


def build_rust(code: str, build_dir: Path, settings: Settings, spec: TaskSpec, args_module: str | None) -> Built:
    """``harness.build.rust.build_rust`` (설계 v3 §6). 모듈·설정이 아직 없으면 ``DevcheckProblem``."""
    try:
        mod = importlib.import_module("harness.build.rust")
    except ImportError as e:
        raise DevcheckProblem(f"rust backend not available yet (harness.build.rust: {e})") from None
    from harness.config import ConfigError, load_config

    try:
        cfg = load_config(settings.path)
    except ConfigError as e:
        raise DevcheckProblem(f"rust 빌드에는 유효한 설정이 필요하다: {e}") from None
    rcfg = getattr(cfg.build, "rust", None)
    if rcfg is None:
        raise DevcheckProblem("rust backend not available yet (설정에 build.rust 가 없다)")
    import harness.build.c as build_c_mod

    missing = getattr(build_c_mod, "ToolchainMissing", None)
    target_dir = Path(tempfile.gettempdir()) / "devcheck_cargo_target"  # 의존성(rayon) 빌드를 점검 사이에 재사용한다
    try:
        res = mod.build_rust(code, build_dir, rcfg, launch_prefix=[], symbol=spec.function, args_module=args_module,
                             target_dir=target_dir)
    except OSError as e:
        raise DevcheckProblem(f"cargo 실행 실패: {e}") from None
    except Exception as e:
        if missing is not None and isinstance(e, missing):
            raise DevcheckProblem(f"Rust 툴체인이 없다: {e}") from None
        raise
    return _from_build_result(res, note="cargo --release")


# ----------------------------------------------------------------------------- 추출·워커·점검


def read_source(path: Path) -> tuple[str, bool]:
    """파일 내용과 "응답 문서인가"(추출이 필요한가). ``.md`` 는 응답, ``.json`` 은 raw(``response_text``), 그 밖은 소스 그대로.

    파일은 UTF-8 이어야 한다. 아니면 그 파일을 이름으로 알리는 ``DevcheckProblem`` 이다(``dummies`` 에서는 그 행만
    ``problem`` 이 되고 나머지 조합은 계속 점검한다).

    맨 앞의 UTF-8 BOM(Windows 편집기·PowerShell 5.1 의 ``-Encoding utf8`` 이 붙인다):

    - 소스 파일(``.py``/``.c``/``.rs`` …)과 raw JSON: 내용이 아니므로 뗀다(``utf-8-sig``). 파이썬·C·Rust 컴파일러와 워커의
      모듈 로더는 BOM 을 건너뛰는데 문자열을 받는 ``compile()`` 만 ``U+FEFF`` 를 구문 오류로 본다 — 맞는 커널을
      틀렸다고(exit 1) 판정하면 안 된다.
    - 응답 문서(``.md``): 떼지 않는다. 더미 provider(``harness.llm_client``)는 파일 내용을 그대로 응답 텍스트로 쓰므로
      devcheck 도 같은 텍스트로 추출해야 여기서 통과한 더미가 실제 파이프라인에서 같은 판정을 받는다(BOM 바로 뒤가
      펜스면 ``harness.extract`` 가 그 펜스를 인식하지 못한다). BOM 이 있다는 사실은 ``check`` 가 알린다.
    """
    suffix = Path(path).suffix.lower()
    try:
        text = Path(path).read_text(encoding="utf-8" if suffix == ".md" else "utf-8-sig")
    except OSError as e:
        raise DevcheckProblem(f"파일을 읽을 수 없다: {path} ({e})") from None
    except UnicodeDecodeError as e:
        raise DevcheckProblem(f"파일이 UTF-8 이 아니다: {path} ({e})") from None
    if suffix == ".json":
        try:
            return str(json.loads(text).get("response_text") or ""), True
        except (ValueError, AttributeError) as e:
            raise DevcheckProblem(f"raw JSON 이 아니다: {path} ({e})") from None
    return text, suffix == ".md"


def extract_code(text: str, tctx: TaskCtx, lang: LanguageInfo, args_mode: str, work_dir: Path | None,
                 cc: str | None = None) -> Any:
    """``harness.extract.extract`` 호출. ``bundle`` 인자(설계 v3 §7)와 전처리 컴파일러 ``cc`` 는 그 매개변수가 있을 때만 넘긴다."""
    from harness.extract import extract

    spec = tctx.spec
    bundle = sigs.signature_bundle(spec, lang.name, lang.backend, args_mode)
    positional = sigs.signature_bundle(spec, lang.name, lang.backend, "positional")["signature"]
    signature_spec: dict[str, Any] = {"function": spec.function, "params": list(bundle["params"]), "positional": positional,
                                      "language": lang.name}
    if args_mode == "keyword":
        signature_spec["keyword"] = bundle["signature"]
    kwargs: dict[str, Any] = {"backend": lang.backend, "signature_spec": signature_spec, "args_mode": args_mode, "work_dir": work_dir}
    parameters = inspect.signature(extract).parameters
    if "bundle" in parameters:
        kwargs["bundle"] = bundle
    if cc is not None and "cc" in parameters:  # C 응답의 전처리 컴파일러: 빌드와 같은 설정의 build.cc (harness.run 과 같다)
        kwargs["cc"] = cc
    try:
        return extract(text, **kwargs)
    except ValueError as e:  # 예: 아직 그 백엔드를 모르는 extract (설계 v3 §7 이전) — 커널의 잘못이 아니다
        raise DevcheckProblem(f"harness.extract 가 이 요청을 처리하지 못했다(backend={lang.backend!r}; 아직 지원하지 않는 "
                              f"백엔드일 수 있다): {e}") from None


def run_worker(build_dir: Path, *, lang: LanguageInfo, settings: Settings, tctx: TaskCtx, artifact: Path, mode: str,
               tests: list[Path], args_mode: str, timeout_s: float, extra: list[str]) -> tuple[ProcResult, dict[str, Any] | None]:
    """``PYTHON_EXE -m harness.worker`` 를 격리 실행한다(cwd = 빌드 디렉터리). 결과 문서는 ``<mode>.json``."""
    out_path = build_dir / f"{mode}.json"
    if out_path.exists():
        out_path.unlink()
    cmd = [str(PYTHON_EXE), "-m", "harness.worker", *worker_backend_args(lang, settings), "--artifact", str(artifact),
           "--mode", mode, "--tests", ",".join(str(t) for t in tests), "--tolerance", str(tctx.tolerance),
           "--signatures", str(tctx.signatures), "--out", str(out_path), "--args-mode", args_mode, *extra]
    proc = run_isolated(cmd, timeout_s=timeout_s, cwd=build_dir, env=worker_env(lang),
                        stdout_path=build_dir / f"{mode}.stdout.log", stderr_path=build_dir / f"{mode}.stderr.log")
    doc: dict[str, Any] | None = None
    if out_path.exists():
        try:
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
            doc = loaded if isinstance(loaded, dict) else None
        except ValueError:
            doc = None
    return proc, doc


@dataclass
class CheckRequest:
    """점검 하나의 입력."""

    label: str
    tctx: TaskCtx
    lang: LanguageInfo
    args_mode: str = "positional"
    tests: list[str] = field(default_factory=list)
    bench: bool = False
    code: str | None = None  # 소스 코드(추출이 끝났거나 필요 없는 경우)
    response: str | None = None  # 응답 문서(.md): 추출이 필요하다
    artifact: Path | None = None  # 빌드 없이 그대로 쓰는 산출물(python 기준선)
    c_toolchain: str = "auto"
    timeout_s: float | None = None


def check(req: CheckRequest, settings: Settings, build_dir: Path) -> Outcome:
    """추출 → 빌드 → (halide: 워커 import) → 워커 check (→ bench). 형태별 판정과 최종 판정을 담아 돌려준다(``DevcheckProblem`` 은
    호출자가 받는다). 단계 이름은 harness.run 의 행과 같다: 커널 로드 실패는 ``stage=build``."""
    out = Outcome(label=req.label, build_dir=build_dir)
    names = list(req.tests)
    tctx, lang, spec = req.tctx, req.lang, req.tctx.spec
    not_run = [ShapeVerdict(n, "NOT-RUN") for n in names]
    test_paths = tctx.test_paths(names)  # 없으면 DevcheckProblem
    bench_path = tctx.test_paths([settings.performance_test])[0] if req.bench else None
    if not spec.supports(lang.name):
        out.notes.append(f"unsupported in signatures.yaml: {spec.unsupported[lang.name]}")
    bundle = sigs.signature_bundle(spec, lang.name, lang.backend, req.args_mode)

    # 1. 추출
    code = req.code
    if req.response is not None:
        has_bom = req.response.startswith(UTF8_BOM)
        if has_bom:  # 떼지 않는다(read_source): 더미 provider 가 주는 응답 텍스트와 같은 것으로 추출한다
            out.notes.append("response starts with a UTF-8 BOM (U+FEFF); it is passed to extraction unchanged, as the dummy provider does")
        c_gcc = lang.backend == "c" and choose_c_toolchain(settings, req.c_toolchain) == "gcc"
        ext = extract_code(req.response, tctx, lang, req.args_mode, build_dir if c_gcc else None, settings.cc)
        if not ext.ok or ext.code is None:
            out.status, out.stage, out.classification = "fail", "extract", "signature_mismatch"
            out.detail, out.shapes = _one_line(ext.detail or "no_code"), not_run
            if has_bom:
                # BOM 이 원인인지 확인해 알린다: BOM 을 뗀 텍스트는 추출되는데 그대로는 안 되면(BOM 바로 뒤가 펜스) BOM 탓이다
                retry = extract_code(req.response[len(UTF8_BOM):], tctx, lang, req.args_mode, build_dir if c_gcc else None,
                                     settings.cc)
                if retry.ok and retry.code is not None:
                    out.detail += " (caused by the UTF-8 BOM at the start of the response file: remove it)"
            return out
        if ext.detail:
            out.notes.append(f"extract: {ext.detail}")
        code = ext.code

    # 2. 빌드
    artifact = req.artifact
    if artifact is None:
        assert code is not None
        if lang.backend == "c":
            header = bundle["args_struct"] or None
            toolchain = choose_c_toolchain(settings, req.c_toolchain)
            if toolchain == "gcc":
                built = build_c_gcc(code, build_dir, settings, lang, spec.function, header)
            elif toolchain == ZIG_OMP_TOOLCHAIN and lang.openmp:
                built = build_c_zig_omp(code, build_dir, header)
                if os.environ.get("DEVCHECK_OMP_THREADS", "").strip():  # 실제 팀 크기를 이 값으로(워커의 OMP_NUM_THREADS)
                    lang = replace(lang, threads=stub_omp_threads())
                out.notes.append(f"OMP_NUM_THREADS={lang.threads}")
            else:  # zig 직렬: OpenMP 로 빌드하지 않는 대상(c_seq)은 zig-omp 에서도 여기로 온다(하네스의 c_seq 빌드와 같다)
                built = build_c_zig(code, build_dir, header)
        elif lang.backend == "rust":
            built = build_rust(code, build_dir, settings, spec, bundle["args_struct"] or None)
        else:
            built = build_python_family(code, build_dir, lang, spec)
        out.toolchain = built.toolchain
        if built.note:
            out.notes.append(f"build: {built.note}")
        if not built.ok or built.artifact is None:
            out.status, out.stage = "fail", "build"
            out.classification = "signature_mismatch" if built.error_kind == "signature_mismatch" else "compile_error"
            out.detail = "compile_timeout" if built.error_kind == "timeout" else (built.error_detail or built.error_kind)
            out.shapes = not_run
            return out
        artifact = built.artifact

    timeout_s = req.timeout_s if req.timeout_s is not None else settings.check_timeout_s
    # 3a. import (halide 만): harness.run 처럼 첫 정확성 형태로 파이프라인을 만들어 본다. 정의·스케줄·JIT 의 실패와 Func 가 아닌
    # 반환값은 여기서 로드 실패(build 단계: compile_error / signature_mismatch)가 된다 — check 의 runtime_crash 가 아니다.
    if lang.backend == "halide":
        proc, idoc = run_worker(build_dir, lang=lang, settings=settings, tctx=tctx, artifact=Path(artifact).resolve(), mode="import",
                                tests=test_paths[:1], args_mode=req.args_mode, timeout_s=timeout_s, extra=[])
        verdict = interpret_worker(proc, idoc, timeout_s)
        idoc = idoc or {}
        if not idoc.get("load_ok"):
            out.stage, out.shapes = "build", not_run
            if verdict is not None:
                out.status, out.classification, out.detail = verdict
                return out
            detail = str(idoc.get("load_error") or idoc.get("error_detail") or "load failed")
            if detail.startswith(AUTOSCHEDULER_UNAVAILABLE):  # 도구 문제(harness.run 의 ToolchainMissing, exit 7)
                raise DevcheckProblem(f"Halide 오토스케줄러 플러그인을 로드할 수 없다 ({detail})")
            out.status, out.classification = "fail", str(idoc.get("classification") or "compile_error")
            out.detail = _one_line(detail)
            return out
        if verdict is not None:  # 로드는 됐다: harness.run 처럼 판정은 check 에 맡긴다
            out.notes.append(f"import: 커널을 로드한 뒤 워커가 끝까지 가지 못했다 ({verdict[1]}); check 로 판정한다")

    # 3. 워커 check
    proc, doc = run_worker(build_dir, lang=lang, settings=settings, tctx=tctx, artifact=Path(artifact).resolve(), mode="check",
                           tests=test_paths, args_mode=req.args_mode, timeout_s=timeout_s,
                           extra=["--check-repeats", str(settings.check_repeats)])
    out.stage = "check"
    out.shapes = shape_verdicts_from_check(doc or {}, names)
    verdict = interpret_worker(proc, doc, timeout_s)
    if verdict is not None:
        out.status, out.classification, out.detail = verdict
        return out
    assert doc is not None
    if doc.get("load_ok") is False:  # 로드 실패는 harness.run 처럼 build 단계의 오류다
        out.status, out.stage = "fail", "build"
        out.classification = str(doc.get("classification") or "compile_error")
        out.detail = _one_line(doc.get("load_error") or doc.get("error_detail") or "load failed")
        return out
    out.classification = str(doc.get("classification") or "")
    out.detail = _one_line(doc.get("error_detail") or "")
    passed = out.classification == "pass" and not doc.get("inputs_mutated") and not doc.get("oob_write")
    if not passed:
        out.status = "fail"
        if out.classification == "pass":  # 방어: 판정은 pass 인데 플래그만 있는 문서
            out.classification = "wrong_output"
            out.detail = "inputs_mutated" if doc.get("inputs_mutated") else "out_of_bounds_write"
        return out
    out.status = "pass"

    # 4. bench (참고값: large 1회, 워밍업 없음)
    if bench_path is not None:
        btimeout = settings.bench_timeout_s
        proc, bdoc = run_worker(build_dir, lang=lang, settings=settings, tctx=tctx, artifact=Path(artifact).resolve(), mode="bench",
                                tests=[bench_path], args_mode=req.args_mode, timeout_s=btimeout,
                                extra=["--warmup", "0", "--repeats", "1", "--min-repeats", "1", "--budget-s", str(btimeout)])
        verdict = interpret_worker(proc, bdoc, btimeout)
        if verdict is not None:
            out.status, out.classification, out.detail = verdict
            out.stage = "bench"
            return out
        assert bdoc is not None
        timed = bdoc.get("timed_ms") or bdoc.get("warmup_ms") or []
        out.bench_ms = float(timed[0]) if timed else None
        bcls = str(bdoc.get("classification") or "")
        if bcls == "timeout":
            out.notes.append(f"bench: {_one_line(bdoc.get('error_detail') or 'timeout')} (참고값; 정확성 판정에는 영향 없음)")
        elif bcls != "pass":
            out.status, out.stage, out.classification = "fail", "bench", bcls or "wrong_output"
            out.detail = _one_line(bdoc.get("error_detail") or "")
    return out


def report(out: Outcome) -> None:
    """형태마다 한 줄 + 최종 판정 한 줄(stdout), 참고 사항(stderr)."""
    width = max([len(s.name) for s in out.shapes] + [5])
    for s in out.shapes:
        emit(f"{out.label} {s.name:<{width}} {s.verdict:<7} {s.info}".rstrip())
    for note in out.notes:
        diag(f"{out.label}: {note}")
    emit(out.final_line())


def run_check(req: CheckRequest, settings: Settings, keep: Path | None) -> Outcome:
    """빌드 디렉터리를 마련해 ``check`` 를 돌리고 보고한다. 도구 문제는 ``status="problem"`` 인 결과로 바꾼다."""
    if keep is not None:
        build_dir = Path(keep).resolve()
        build_dir.mkdir(parents=True, exist_ok=True)
    else:
        build_dir = Path(tempfile.mkdtemp(prefix="devcheck_")).resolve()
    try:
        not_run = [ShapeVerdict(n, "NOT-RUN") for n in req.tests]
        try:
            out = check(req, settings, build_dir)
        except DevcheckProblem as e:
            out = Outcome(label=req.label, status="problem", stage="setup", classification="devcheck_problem",
                          detail=_one_line(e, 600), shapes=not_run, build_dir=build_dir)
        except Exception as e:  # devcheck·하네스의 버그: 커널 실패(exit 1)로 보이지 않게 문제(exit 2)로 보고한다
            traceback.print_exc(file=sys.stderr)
            out = Outcome(label=req.label, status="problem", stage="setup", classification="devcheck_internal_error",
                          detail=_one_line(f"{type(e).__name__}: {e}", 600), shapes=not_run, build_dir=build_dir)
        report(out)
        if keep is not None:
            diag(f"{req.label}: 빌드 디렉터리를 남겼다: {build_dir}")
        return out
    finally:
        if keep is None:
            shutil.rmtree(build_dir, ignore_errors=True)


# ----------------------------------------------------------------------------- 하위 명령


def cmd_kernel(ns: argparse.Namespace, settings: Settings) -> int:
    tctx = load_task(ns.task, ns.tasks_dir)
    lang = resolve_language(settings, ns.language)
    text, is_response = read_source(ns.file)
    req = CheckRequest(label=f"{ns.task}/{ns.language}", tctx=tctx, lang=lang, args_mode=ns.args_mode,
                       tests=split_names(ns.tests) or list(settings.correctness_tests), bench=ns.bench,
                       code=None if is_response else text, response=text if is_response else None,
                       c_toolchain=ns.cc, timeout_s=ns.timeout)
    return run_check(req, settings, ns.keep).exit_code()


def baseline_request(tctx: TaskCtx, settings: Settings, tests: list[str], *, bench: bool, c_toolchain: str,
                     timeout_s: float | None) -> CheckRequest:
    """과제 명세의 기준선(``baseline``)을 점검 요청으로. python 기준선은 파일 그대로, C 기준선은 빌드한다."""
    b = tctx.spec.baseline
    src = tctx.dir / str(b.get("file"))
    if not src.is_file():
        raise DevcheckProblem(f"{tctx.task}: 기준선 파일이 없다({src})")
    threads = max(1, min(settings.threads, os.cpu_count() or 1))
    label = f"{tctx.task}/baseline"
    if b.get("backend") == "python":
        lang = LanguageInfo(name="baseline", backend="python", threads=threads, source="baseline")
        return CheckRequest(label=label, tctx=tctx, lang=lang, tests=tests, bench=bench, artifact=src, timeout_s=timeout_s)
    lang = LanguageInfo(name="baseline", backend="c", openmp=bool(b.get("openmp")), threads=threads, source="baseline")
    return CheckRequest(label=label, tctx=tctx, lang=lang, tests=tests, bench=bench, code=read_source(src)[0],
                        c_toolchain=c_toolchain, timeout_s=timeout_s)


def cmd_baseline(ns: argparse.Namespace, settings: Settings) -> int:
    tctx = load_task(ns.task, ns.tasks_dir)
    req = baseline_request(tctx, settings, split_names(ns.tests) or list(settings.correctness_tests), bench=ns.bench,
                           c_toolchain=ns.cc, timeout_s=ns.timeout)
    return run_check(req, settings, ns.keep).exit_code()


def _ordered_languages(names: set[str]) -> list[str]:
    known = [n for n in LANGUAGE_ORDER if n in names]
    return known + sorted(names - set(known))


def default_dummy_tasks(tasks_dir: Path, dummy_dir: Path) -> list[str]:
    """``dummies`` 의 기본 과제 목록: 과제 디렉터리(``*/signatures.yaml``)와 더미 디렉터리의 하위 디렉터리의 **합집합**(이름순).

    더미가 있는 과제만 고르면 더미 디렉터리를 통째로 빠뜨린 과제가 표에서 사라져 exit 0 이 된다(설계 v3 §4 의 "검증된 더미가
    있어야 실험 대상" 규칙이 과제 단위로 빠진다). 과제 쪽에만 있으면 그 과제의 모든 언어가 ``missing``, 더미 쪽에만 있으면
    (과제 이름 오타·지운 과제의 더미) ``problem`` 행으로 드러난다. ``.``·``_`` 로 시작하는 디렉터리는 과제가 아니다."""
    names = {p.parent.name for p in Path(tasks_dir).glob("*/signatures.yaml")}
    if Path(dummy_dir).is_dir():
        names |= {p.name for p in Path(dummy_dir).iterdir() if p.is_dir() and not p.name.startswith((".", "_"))}
    return sorted(names)


def default_dummy_languages(settings: Settings, dummy_dir: Path, tasks: list[str]) -> list[str]:
    """``dummies`` 의 기본 언어 목록: 고른 과제에 ``0.md`` 가 있는 언어와 설정(``languages``)에 정의된 언어의 **합집합**
    (성능 결정 부담 순). 설정에 있는 언어는 실험 대상이므로 더미가 하나도 없어도 표에 나와야 한다(``missing``)."""
    found = {p.parent.name for t in tasks for p in (Path(dummy_dir) / t).glob("*/0.md")}
    return _ordered_languages(found | {str(name) for name in settings.languages})


def cmd_dummies(ns: argparse.Namespace, settings: Settings) -> int:
    dummy_dir = Path(ns.dummy_dir)
    tasks = split_names(ns.tasks)
    if tasks is None:
        tasks = default_dummy_tasks(Path(ns.tasks_dir), dummy_dir)
    if not tasks:
        raise DevcheckProblem(f"점검할 과제가 없다(과제 디렉터리 {ns.tasks_dir}, 더미 디렉터리 {dummy_dir})")
    languages = split_names(ns.languages)
    if languages is None:
        languages = default_dummy_languages(settings, dummy_dir, tasks)
    if not languages:
        raise DevcheckProblem(f"점검할 언어가 없다({dummy_dir} 에 0.md 가 없고 설정 {settings.path} 에 languages 도 없음)")
    if ns.tasks is None or ns.languages is None:
        diag(f"dummies: 기본 목록(과제 = 과제 디렉터리 ∪ 더미 디렉터리, 언어 = 0.md 가 있는 언어 ∪ 설정의 languages) → "
             f"과제 {len(tasks)}개 × 언어 {len(languages)}개({', '.join(languages)})")
    tests = split_names(ns.tests) or list(settings.correctness_tests)

    rows: list[tuple[str, str, Outcome | None, str, str]] = []  # (task, language, outcome, result, detail)
    outcomes: list[Outcome] = []
    counts = {"pass": 0, "fail": 0, "problem": 0, "unsupported": 0, "missing": 0}
    for task in tasks:
        try:
            tctx: TaskCtx | None = load_task(task, ns.tasks_dir)
            task_problem = ""
        except DevcheckProblem as e:
            tctx, task_problem = None, _one_line(e)
        for language in languages:
            label = f"{task}/{language}"
            md = dummy_dir / task / language / "0.md"
            if tctx is None:
                outcome = Outcome(label=label, status="problem", stage="setup", classification="devcheck_problem", detail=task_problem)
            elif not tctx.spec.supports(language):
                rows.append((task, language, None, "unsupported", tctx.spec.unsupported[language]))
                counts["unsupported"] += 1
                continue
            elif not md.is_file():
                # 지원한다고 돼 있는데 검증된 더미가 없다(설계 v3 §4: 더미 0.md 가 있어야 실험 대상이다)
                outcome = Outcome(label=label, status="fail", stage="setup", classification="missing_dummy",
                                  detail=f"no {md.relative_to(dummy_dir).as_posix()} (and not listed in unsupported)")
                rows.append((task, language, outcome, "missing", outcome.detail))
                outcomes.append(outcome)
                counts["missing"] += 1
                emit(outcome.final_line())
                continue
            else:
                try:
                    lang = resolve_language(settings, language)
                    text, _ = read_source(md)
                    req = CheckRequest(label=label, tctx=tctx, lang=lang, args_mode=ns.args_mode, tests=tests, response=text,
                                       c_toolchain=ns.cc, timeout_s=ns.timeout)
                    keep = (Path(ns.keep) / task / language) if ns.keep is not None else None
                    outcome = run_check(req, settings, keep)
                except DevcheckProblem as e:
                    outcome = Outcome(label=label, status="problem", stage="setup", classification="devcheck_problem", detail=_one_line(e, 600))
                    emit(outcome.final_line())
            if tctx is None:
                emit(outcome.final_line())
            outcomes.append(outcome)
            counts[outcome.status if outcome.status in counts else "problem"] += 1
            rows.append((task, language, outcome, outcome.status, outcome.detail if outcome.status != "pass" else ""))

    # 표
    emit("")
    tw = max(len("task"), *(len(r[0]) for r in rows))
    lw = max(len("language"), *(len(r[1]) for r in rows))
    sw = max(7, *(len(n) for n in tests))
    emit(f"{'task':<{tw}}  {'language':<{lw}}  " + "  ".join(f"{n:<{sw}}" for n in tests) + "  result       detail")
    for task, language, outcome, result, detail in rows:
        by_name = {s.name: s.verdict for s in (outcome.shapes if outcome is not None else [])}
        cells = "  ".join(f"{by_name.get(n, '-'):<{sw}}" for n in tests)
        stage = f"{outcome.stage}/{outcome.classification}: " if (outcome is not None and outcome.status not in ("pass",) and outcome.stage) else ""
        text = f"{stage}{detail}"
        if outcome is not None and outcome.toolchain:  # zig 직렬 빌드로 나온 판정임을 표에도 남긴다(pass 행의 detail 은 이 표시뿐이다)
            marker = f"toolchain={outcome.toolchain}"
            text = f"{text} [{marker}]" if text else marker
        emit(f"{task:<{tw}}  {language:<{lw}}  {cells}  {result:<11}  {text}".rstrip())
    emit("")
    summary = "summary: " + ", ".join(f"{counts[k]} {k}" for k in ("pass", "fail", "missing", "problem", "unsupported"))
    serial_passes = sum(1 for o in outcomes if o.status == "pass" and o.toolchain == ZIG_TOOLCHAIN)
    if serial_passes:
        summary += f"; {serial_passes} of the pass rows used toolchain={ZIG_TOOLCHAIN} (serial build: logic only, OpenMP pragmas ignored)"
    omp_passes = sum(1 for o in outcomes if o.status == "pass" and o.toolchain == ZIG_OMP_TOOLCHAIN)
    if omp_passes:
        summary += (f"; {omp_passes} of the pass rows used toolchain={ZIG_OMP_TOOLCHAIN} "
                    "(real threads with clang + LLVM OpenMP runtime: not gcc/libgomp)")
    emit(summary)
    return combine_exit_codes(outcomes)


def cmd_gen(ns: argparse.Namespace, settings: Settings) -> int:
    tdir = (Path(ns.tasks_dir) / ns.task).resolve()
    gen = tdir / "gen_tests.py"
    if not gen.is_file():
        raise DevcheckProblem(f"테스트 생성기가 없다: {gen}")
    with tempfile.TemporaryDirectory(prefix="devcheck_gen_") as tmp:
        log = Path(tmp) / "gen.stderr.log"
        t0 = time.perf_counter()
        proc = run_isolated([str(PYTHON_EXE), str(gen)], timeout_s=GEN_TIMEOUT_S, cwd=ROOT, env=child_env(),
                            stdout_path=Path(tmp) / "gen.stdout.log", stderr_path=log)
        elapsed = time.perf_counter() - t0
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    for line in text.splitlines():
        print(line, file=sys.stderr)
    if proc.timed_out or proc.rc != 0:
        emit(f"{ns.task}/gen => FAIL {'timeout' if proc.timed_out else f'rc={proc.rc}'} after {elapsed:.1f}s")
        return EXIT_FAIL
    tests_dir = tdir / "tests"
    names = [*settings.correctness_tests, settings.performance_test]
    total = 0
    missing: list[str] = []
    for name in names:
        path = tests_dir / f"{name}.npz"
        if not path.is_file():
            missing.append(name)
            continue
        size = path.stat().st_size
        total += size
        emit(f"{ns.task}/gen {name:<6} {size:>12d} B  {path}")
    manifest: dict[str, Any] = {}
    try:
        manifest = json.loads((tests_dir / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        missing.append("manifest.json")
    if missing:
        emit(f"{ns.task}/gen => FAIL missing after generation: {', '.join(missing)}")
        return EXIT_FAIL
    emit(f"{ns.task}/gen => PASS test_hash={manifest.get('test_hash')} seed={manifest.get('seed')} "
         f"total_mb={total / 1e6:.1f} elapsed_s={elapsed:.1f}")
    return EXIT_OK


# ----------------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python tools/devcheck.py",
        description="개발용 커널 점검(측정 아님): 추출 → 빌드 → harness.worker check. 종료 코드 0 통과 / 1 실패 / 2 사용법·도구 문제.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=DEFAULT_EXPERIMENT_YAML, help="experiment.yaml (기본 config/experiment.yaml)")
    common.add_argument("--tasks-dir", type=Path, default=TASKS_DIR, help="과제 디렉터리(기본 tasks/)")
    checks = argparse.ArgumentParser(add_help=False)
    checks.add_argument("--tests", default=None, help="정확성 형태(쉼표 구분; 기본 설정의 correctness_tests = tiny,rect,small)")
    checks.add_argument("--keep", type=Path, default=None, metavar="DIR", help="빌드 디렉터리를 DIR 에 남긴다(기본: 임시 디렉터리, 끝나면 삭제)")
    checks.add_argument("--cc", choices=C_TOOLCHAIN_CHOICES, default="auto",
                        help="C 툴체인: auto(Linux 에 설정의 컴파일러가 있으면 gcc, 아니면 zig) | gcc | zig(직렬) | "
                             "zig-omp(OpenMP 대상만 clang -fopenmp + LLVM OpenMP 런타임으로 실제 스레드; 개발 머신용)")
    checks.add_argument("--timeout", type=float, default=None, help="check 워커 제한 시간(초; 기본 설정의 check_timeout_s)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("kernel", parents=[common, checks], help="커널 파일 하나를 점검한다(.md 응답이면 코드 추출부터)")
    p.add_argument("--task", required=True)
    p.add_argument("--language", required=True)
    p.add_argument("--file", type=Path, required=True, help="응답(.md)·raw(.json) 또는 소스 파일(.py/.c/.rs)")
    p.add_argument("--args-mode", choices=sigs.ARGS_MODES, default="positional")
    p.add_argument("--bench", action="store_true", help="통과하면 large 로 1회 실행 시간을 잰다(참고값)")
    p.set_defaults(func=cmd_kernel)

    p = sub.add_parser("baseline", parents=[common, checks], help="과제의 성능 기준선(baseline)을 점검한다")
    p.add_argument("--task", required=True)
    p.add_argument("--bench", action="store_true", help="통과하면 large 로 1회 실행 시간을 잰다(참고값)")
    p.set_defaults(func=cmd_baseline)

    p = sub.add_parser("dummies", parents=[common, checks], help="더미 응답의 모든 0.md 를 점검해 표로 낸다")
    p.add_argument("--tasks", default=None, help="쉼표 구분 과제(기본: 과제 디렉터리의 과제 ∪ 더미 디렉터리가 있는 과제)")
    p.add_argument("--languages", default=None, help="쉼표 구분 언어(기본: 0.md 가 있는 언어 ∪ 설정의 languages)")
    p.add_argument("--args-mode", choices=sigs.ARGS_MODES, default="positional")
    p.add_argument("--dummy-dir", type=Path, default=DUMMY_DIR, help="더미 응답 디렉터리(기본 harness/dummy_responses)")
    p.set_defaults(func=cmd_dummies)

    p = sub.add_parser("gen", parents=[common], help="tasks/<task>/gen_tests.py 를 그대로 실행한다(네 형태 전부)")
    p.add_argument("--task", required=True)
    p.set_defaults(func=cmd_gen)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    ns = parser.parse_args(argv)  # 사용법 오류는 argparse 가 exit 2 로 끝낸다
    try:
        settings = load_settings(ns.config)
        return int(ns.func(ns, settings))
    except DevcheckProblem as e:
        emit(f"devcheck => PROBLEM {_one_line(e, 600)}")
        return EXIT_PROBLEM
    except KeyboardInterrupt:
        diag("중단됨")
        return 130
    except Exception as e:  # 버그는 커널 실패(1)가 아니라 도구 문제(2)로 끝낸다
        traceback.print_exc(file=sys.stderr)
        emit(f"devcheck => PROBLEM internal error: {_one_line(f'{type(e).__name__}: {e}', 600)}")
        return EXIT_PROBLEM


if __name__ == "__main__":
    # 스트림 정리는 진입점에서만 한다(``harness.procutil.setup_stdio``): main() 안에서 하면 같은 프로세스에서 main() 을 부르는
    # 쪽(단위 테스트)의 실제 sys.stdout 이 끝까지 바뀌어, 뒤에 도는 테스트의 출력 인코딩이 실행 순서에 따라 달라진다.
    setup_stdio()
    sys.exit(main())
