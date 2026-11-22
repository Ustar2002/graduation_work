"""측정 규약 강제 (설계 §9, v3 §0·§3·§9.5): 거버너·EPP·turbo·taskset·BLAS 정체(정적), 부하·코어 사용률(동적),
워커 실행 접두(``taskset``)와 화이트리스트 환경, 머신 프로브(``config/machine.yaml``), 툴체인 정체, 환경 지문.

하네스는 Linux 전용 측정 도구다. 다른 플랫폼(개발용 Windows)에서도 임포트되고 검사 함수가 죽지 않지만, 그때 만든 상태에는
플래그 ``platform=<sys.platform>`` 이 붙어 그 행은 ``protocol_ok=0`` 이다(설계 v3 §0).

CLI:
    python -m harness.protocol --check                  # static + dynamic 결과 JSON (플래그 있으면 exit 1)
    python -m harness.protocol --probe [--write [PATH]] # 머신 프로브 JSON; --write 면 machine.yaml 저장
    python -m harness.protocol --fingerprint            # 환경 지문 (machine.yaml 필요)
    python -m harness.protocol --env-identity           # (내부용) 현재 인터프리터의 python/numpy/BLAS/SDK 정체 JSON
    python -m harness.protocol --python-toolchain NAME  # (내부용) 현재 인터프리터의 numba·halide·린트 도구와 오토스케줄러 NAME
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import re
import shutil
import socket
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

from harness.config import Config, ConfigError, load_config, load_machine, parse_cpulist
from harness.paths import PYTHON_EXE, ROOT, atomic_write_text, machine_yaml_for_host
from harness.procutil import IS_POSIX, run_isolated

_LOG = "[protocol]"

SYS_CPU = Path("/sys/devices/system/cpu")
PROC_STAT = Path("/proc/stat")

FIX_GOVERNOR = "sudo cpupower frequency-set -g performance"
FIX_EPP = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"

# 정보용 플래그(ok 판정에서 제외): turbo 상태, 사용자 환경의 스레드 변수 제거 알림
# load_1m 은 2026-11-08 부터 정보용: 직전 샘플의 4스레드 bench 가 올린 1분 평균을 다음 검사가 읽는 자기 부하라 판정에서 뺀다(decisions.md).
# multiple_libgomp 은 2026-11-23 부터: numba(스레딩 계층 omp)와 C+OpenMP 기준선이 서로 다른 libgomp 를 올린 bench 를 표시만 한다
# (2026-11-12 결정은 둘이 하나를 공유한다고 가정한다 — Linux 에서 확인할 것; 규약 위반으로 세지는 않는다).
INFO_FLAG_PREFIXES: tuple[str, ...] = ("turbo=", "env_overrides_dropped", "load_1m=", "multiple_libgomp")

# 워커 환경 화이트리스트(상속 허용)와 고정값, 제거 대상 접두
ENV_INHERIT: tuple[str, ...] = ("PATH", "HOME", "LANG", "TMPDIR", "VIRTUAL_ENV")
ENV_DROP_PREFIXES: tuple[str, ...] = ("OMP_", "GOMP_", "OPENBLAS_", "MKL_", "KMP_")

DEFAULT_CC = "gcc"  # 설정 없이 도는 프로브가 쓰는 컴파일러
# cargo·rustc 를 띄우는 환경에 더하는 사용자 변수(rustup 의 위치와 툴체인 고정). 빌드(build/rust.py)와 도구 점검이 같은 환경을 쓴다.
TOOLCHAIN_ENV_KEYS: tuple[str, ...] = ("CARGO_HOME", "RUSTUP_HOME", "RUSTUP_TOOLCHAIN")
RUNTIME_PROBE_TIMEOUT_S = 120.0  # --probe-runtime 서브프로세스의 시간 상한

# 설계 v3 §3 의 선택 설정과 그 기본값(YAML 에 키를 쓰지 않았을 때와 같은 값). 앞의 것들은 Config 기준, 마지막 둘은
# LanguageCfg 기준의 속성 경로다. 이 키들을 모르던 때의 Config(해당 속성이 없다)와 설정 없이 도는 프로브(cfg=None)도
# ``config_option`` 으로 읽으면 같은 값이 나온다. 값은 읽기 전용으로 쓴다(공유 객체).
OPTION_DEFAULTS: dict[str, Any] = {
    "lint.enabled": True,
    "safety.enabled": False,
    "perf_summary.enabled": True,
    "halide.autoscheduler": "Li2018",
    "halide.autoscheduler_params": {},
    "build.rust": SimpleNamespace(cargo="cargo", rustflags=["-C", "target-cpu=native"],
                                  rustflags_linux=["-C", "link-arg=-Wl,--no-undefined"], build_timeout_s=600.0, offline=True),
    "env": {},
    "halide_mode": None,
}
_MISSING = object()

# PATH 에서 찾는 린트 도구와, ``--version`` 출력에서 버전을 읽는 정규식(plugins/lint/run_lint.py 가 남기는 버전과 같은 문자열)
PATH_LINT_TOOLS: dict[str, re.Pattern[str]] = {
    "cppcheck": re.compile(r"Cppcheck (\S+)"),
    "clang-tidy": re.compile(r"version (\S+)"),
}
# PYTHON_EXE 의 패키지로 도는 린트 도구(배포 이름)
PYTHON_LINT_TOOLS: tuple[str, ...] = ("ruff", "pyflakes")

# /proc/cpuinfo flags 중 기록할 ISA 비트(순서 고정)
ISA_FLAGS_OF_INTEREST: tuple[str, ...] = (
    "sse2", "sse3", "ssse3", "sse4_1", "sse4_2", "avx", "avx2", "fma", "f16c", "bmi1", "bmi2",
    "avx_vnni", "avx512f", "avx512bw", "avx512dq", "avx512vl", "avx512_vnni", "amx_tile", "amx_bf16",
)

_warned_env_drop: set[str] = set()
_warned_taskset = False


@dataclass
class StaticStatus:
    """run당 1회 검사하는 정적 규약 상태."""

    governor: str
    governor_ok: bool
    epp: str
    epp_ok: bool
    no_turbo: int | None
    taskset_available: bool
    cores: str
    single_core: str
    threads: int
    blas: dict[str, Any]
    blas_ok: bool
    flags: list[str]
    ok: bool
    checked_at: str


@dataclass
class DynamicStatus:
    """bench 직전마다 검사하는 동적 상태(1초 샘플)."""

    load_1m: float
    load_ok: bool
    pinned_busy_pct: float
    sibling_busy_pct: float
    busy_ok: bool
    governor: str
    governor_ok: bool
    flags: list[str]
    ok: bool
    checked_at: str


# ----------------------------------------------------------------------------- 작은 헬퍼


def now_iso() -> str:
    """시간대 포함 ISO 8601(초 단위)."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def config_option(obj: Any, path: str) -> Any:
    """선택 설정 ``obj.<path>`` (점으로 이은 속성 경로; ``OPTION_DEFAULTS`` 의 키). 경로의 속성이 하나라도 없으면 그 기본값."""
    value = obj
    for name in path.split("."):
        value = getattr(value, name, _MISSING)
        if value is _MISSING:
            return OPTION_DEFAULTS[path]
    return value


def platform_flag(platform_name: str | None = None) -> str | None:
    """Linux 가 아니면 플래그 ``platform=<sys.platform>`` (정보용이 아니다), Linux 면 None.

    Linux 밖에서는 거버너·코어 고정·부하를 확인할 수 없어 측정 규약이 성립하지 않는다. 인자는 테스트용이다(기본: 현재 플랫폼).
    """
    name = sys.platform if platform_name is None else platform_name
    return None if name.startswith("linux") else f"platform={name}"


def read_load_1m() -> float:
    """1분 load average. ``os.getloadavg`` 가 없는 플랫폼(Windows)은 0.0 이다 — 잰 값이 아님은 ``platform=`` 플래그가 알린다."""
    getloadavg = getattr(os, "getloadavg", None)
    return float(getloadavg()[0]) if getloadavg is not None else 0.0


def _read_text(path: Path) -> str | None:
    """sysfs/proc 파일을 읽는다. 없거나 못 읽으면 None."""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def read_cpufreq(cpu: int, name: str) -> str | None:
    """``/sys/devices/system/cpu/cpu<n>/cpufreq/<name>`` 값."""
    return _read_text(SYS_CPU / f"cpu{cpu}" / "cpufreq" / name)


def thread_siblings(cpu: int) -> list[int]:
    """``topology/thread_siblings_list`` 의 CPU 목록(자기 자신 포함). 없으면 ``[cpu]``."""
    text = _read_text(SYS_CPU / f"cpu{cpu}" / "topology" / "thread_siblings_list")
    if not text:
        return [cpu]
    try:
        return parse_cpulist(text)
    except ValueError:
        return [cpu]


def sibling_cpus(pinned: list[int]) -> list[int]:
    """고정 코어들의 SMT 형제 중 고정 집합에 없는 CPU(정렬)."""
    pinned_set = set(pinned)
    sibs: set[int] = set()
    for cpu in pinned:
        sibs.update(thread_siblings(cpu))
    return sorted(sibs - pinned_set)


def pinned_cpus(cfg: Config) -> list[int]:
    """측정에 쓰는 CPU 집합: ``cores`` ∪ ``single_core``."""
    return sorted(set(parse_cpulist(cfg.measurement.cores)) | set(parse_cpulist(cfg.measurement.single_core)))


def online_cpus() -> list[int]:
    """온라인 CPU 목록(``/sys/devices/system/cpu/online``; 없으면 0..cpu_count-1)."""
    text = _read_text(SYS_CPU / "online")
    if text:
        try:
            return parse_cpulist(text)
        except ValueError:
            pass
    return list(range(os.cpu_count() or 1))


def parse_proc_stat(text: str) -> dict[int, tuple[int, ...]]:
    """``/proc/stat`` 의 ``cpu<n>`` 줄을 {cpu: (user, nice, system, idle, iowait, irq, softirq, steal, ...)} 로."""
    out: dict[int, tuple[int, ...]] = {}
    for line in text.splitlines():
        if not line.startswith("cpu"):
            continue
        parts = line.split()
        name = parts[0]
        if name == "cpu" or not name[3:].isdigit():
            continue
        try:
            out[int(name[3:])] = tuple(int(x) for x in parts[1:])
        except ValueError:
            continue
    return out


def read_proc_stat() -> dict[int, tuple[int, ...]]:
    """현재 ``/proc/stat`` 샘플."""
    return parse_proc_stat(_read_text(PROC_STAT) or "")


def busy_pct(before: dict[int, tuple[int, ...]], after: dict[int, tuple[int, ...]], cpus: list[int]) -> float:
    """두 샘플 사이 CPU들의 평균 사용률(%). idle = idle + iowait, total = 앞 8개 필드 합."""
    vals: list[float] = []
    for cpu in cpus:
        b = before.get(cpu)
        a = after.get(cpu)
        if b is None or a is None or len(b) < 5 or len(a) < 5:
            continue
        total_b = sum(b[:8])
        total_a = sum(a[:8])
        d_total = total_a - total_b
        if d_total <= 0:
            continue
        d_idle = (a[3] + a[4]) - (b[3] + b[4])
        vals.append(100.0 * max(0.0, 1.0 - d_idle / d_total))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)


def _common_value(values: dict[int, str | None]) -> str:
    """CPU별 값이 모두 같으면 그 값, 다르면 ``mixed(cpu:값,…)``, 전부 못 읽으면 ``unknown``."""
    readable = {cpu: v for cpu, v in values.items() if v is not None}
    if not readable:
        return "unknown"
    distinct = sorted(set(readable.values()))
    if len(distinct) == 1 and len(readable) == len(values):
        return distinct[0]
    return "mixed(" + ",".join(f"{cpu}:{v if v is not None else 'unknown'}" for cpu, v in sorted(values.items())) + ")"


def governor_status(cpus: list[int]) -> tuple[str, bool]:
    """(거버너 문자열, 전부 performance 인가)."""
    values = {cpu: read_cpufreq(cpu, "scaling_governor") for cpu in cpus}
    ok = bool(values) and all(v == "performance" for v in values.values())
    return _common_value(values), ok


def epp_status(cpus: list[int]) -> tuple[str, bool]:
    """(EPP 문자열, 전부 performance 인가). EPP 파일이 아예 없으면 ``n/a``·ok."""
    values = {cpu: read_cpufreq(cpu, "energy_performance_preference") for cpu in cpus}
    if all(v is None for v in values.values()):
        return "n/a", True
    ok = all(v == "performance" for v in values.values())
    return _common_value(values), ok


def read_no_turbo() -> int | None:
    """``intel_pstate/no_turbo`` (0=터보 켜짐). 없으면 ``cpufreq/boost`` 를 뒤집어 쓰고, 그것도 없으면 None."""
    text = _read_text(SYS_CPU / "intel_pstate" / "no_turbo")
    if text is not None and text.isdigit():
        return int(text)
    boost = _read_text(SYS_CPU / "cpufreq" / "boost")
    if boost is not None and boost.isdigit():
        return 0 if int(boost) else 1
    return None


def is_informational(flag: str) -> bool:
    """ok 판정에서 제외되는 정보용 플래그인가."""
    return flag.startswith(INFO_FLAG_PREFIXES)


def flags_ok(flags: list[str]) -> bool:
    """정보용을 제외한 플래그가 없으면 True."""
    return not [f for f in flags if not is_informational(f)]


def blas_flag_label(name: str) -> str:
    """OpenBLAS 가 아닐 때의 플래그 라벨: 이름 없음/unknown → ``unknown``, 일반 "blas"(netlib) → ``reference``, 그 외 소문자 이름."""
    lowered = (name or "").strip().lower()
    if not lowered or lowered == "unknown":
        return "unknown"
    if lowered in ("blas", "netlib", "reference"):
        return "reference"
    return lowered


def dropped_env_overrides(environ: dict[str, str] | None = None) -> list[str]:
    """사용자 환경에 있지만 워커에 상속하지 않는 스레드 관련 변수 이름(정렬)."""
    env = os.environ if environ is None else environ
    return sorted(k for k in env if k.startswith(ENV_DROP_PREFIXES))


def warn_power_settings(governor: str, governor_ok: bool, epp: str, epp_ok: bool, cpus: list[int]) -> None:
    """거버너·EPP가 performance 가 아니면 해결 명령과 함께 stderr 경고."""
    cpu_desc = ",".join(str(c) for c in cpus)
    if not governor_ok:
        print(
            f"{_LOG} 경고: scaling_governor={governor} (cpu {cpu_desc}); 'performance' 가 필요하다. "
            f"해결: {FIX_GOVERNOR}",
            file=sys.stderr,
        )
    if not epp_ok:
        print(
            f"{_LOG} 경고: energy_performance_preference={epp} (cpu {cpu_desc}); 'performance' 가 필요하다. "
            f"해결: {FIX_EPP}",
            file=sys.stderr,
        )


# ----------------------------------------------------------------------------- 정적·동적 검사


def check_static(cfg: Config, machine: dict[str, Any]) -> StaticStatus:
    """run당 1회: 플랫폼·거버너·EPP·turbo·taskset·BLAS 정체. 플래그는 경고(해결 명령 포함)와 함께 돌려준다."""
    cpus = pinned_cpus(cfg)
    require = cfg.measurement.require_performance_governor
    governor, governor_ok = governor_status(cpus)
    epp, epp_ok = epp_status(cpus)
    no_turbo = read_no_turbo()
    taskset_available = shutil.which("taskset") is not None
    blas = blas_identity_subprocess()
    blas_name = str(blas.get("name") or "")
    blas_ok = "openblas" in blas_name.lower()

    flags: list[str] = []
    platform_mark = platform_flag()
    if platform_mark:
        flags.append(platform_mark)
        print(f"{_LOG} 경고: Linux 가 아니라 측정 규약을 강제할 수 없다 (플래그 {platform_mark}; 이 run 의 행은 protocol_ok=0)",
              file=sys.stderr)
    if require:
        if not governor_ok:
            flags.append(f"governor={governor}")
        if not epp_ok:
            flags.append(f"epp={epp}")
        warn_power_settings(governor, governor_ok, epp, epp_ok, cpus)
    elif not governor_ok or not epp_ok:
        print(f"{_LOG} 참고: governor={governor}, epp={epp} (require_performance_governor=false 라 플래그 없음)", file=sys.stderr)
    if no_turbo == 0:
        flags.append("turbo=on")
    if not taskset_available:
        flags.append("taskset_missing")
        print(f"{_LOG} 경고: taskset 이 없어 코어 고정을 못 한다 (util-linux 설치 필요)", file=sys.stderr)
    if not blas_ok:
        flags.append(f"blas={blas_flag_label(blas_name)}")
        print(
            f"{_LOG} 경고: NumPy BLAS 가 OpenBLAS 가 아니다 ({blas}). 격리 venv(.venv)에 pip 휠 numpy==2.3.5 를 설치하라.",
            file=sys.stderr,
        )
    dropped = dropped_env_overrides()
    if dropped:
        flags.append("env_overrides_dropped")
        print(f"{_LOG} 참고: 사용자 환경의 {dropped} 는 워커에 상속하지 않는다 (플래그 env_overrides_dropped)", file=sys.stderr)

    probed_host = machine.get("hostname") if isinstance(machine, dict) else None
    if probed_host and probed_host != socket.gethostname():
        print(
            f"{_LOG} 경고: machine.yaml 의 hostname={probed_host!r} 가 현재 {socket.gethostname()!r} 와 다르다. "
            "`python -m harness.protocol --probe --write` 로 갱신하라.",
            file=sys.stderr,
        )

    return StaticStatus(
        governor=governor,
        governor_ok=governor_ok,
        epp=epp,
        epp_ok=epp_ok,
        no_turbo=no_turbo,
        taskset_available=taskset_available,
        cores=cfg.measurement.cores,
        single_core=cfg.measurement.single_core,
        threads=cfg.measurement.threads,
        blas=blas,
        blas_ok=blas_ok,
        flags=flags,
        ok=flags_ok(flags),
        checked_at=now_iso(),
    )


def check_dynamic(cfg: Config, *, sample_s: float = 1.0) -> DynamicStatus:
    """bench 직전마다: 1초 ``/proc/stat`` 샘플로 고정 코어·형제 코어 사용률, load average, 거버너 재확인.

    ``load_1m`` 은 정보용 플래그다(``INFO_FLAG_PREFIXES``; 2026-11-08): 넘어도 ``ok`` 판정에는 들어가지 않는다.
    ``/proc/stat``·``os.getloadavg`` 가 없는 플랫폼에서는 사용률과 load 가 0 으로 남고 ``platform=`` 플래그가 붙는다.
    """
    cpus = pinned_cpus(cfg)
    sibs = sibling_cpus(cpus)
    before = read_proc_stat()
    time.sleep(sample_s)
    after = read_proc_stat()
    pinned_busy = busy_pct(before, after, cpus)
    sibling_busy = busy_pct(before, after, sibs) if sibs else 0.0
    load_1m = read_load_1m()
    m = cfg.measurement
    load_ok = load_1m <= m.max_load_1m
    busy_ok = pinned_busy <= m.max_pinned_busy_pct and sibling_busy <= m.max_pinned_busy_pct
    governor, governor_ok = governor_status(cpus)

    flags: list[str] = []
    if not load_ok:
        flags.append(f"load_1m={load_1m:.1f}>{m.max_load_1m:.1f}")
    if pinned_busy > m.max_pinned_busy_pct:
        flags.append(f"pinned_busy={pinned_busy:.1f}%>{m.max_pinned_busy_pct:g}%")
    if sibling_busy > m.max_pinned_busy_pct:
        flags.append(f"sibling_busy={sibling_busy:.1f}%>{m.max_pinned_busy_pct:g}%")
    if m.require_performance_governor and not governor_ok:
        flags.append(f"governor={governor}")
        print(f"{_LOG} 경고: scaling_governor={governor}; 해결: {FIX_GOVERNOR}", file=sys.stderr)
    for f in flags:
        if not f.startswith("governor="):
            print(f"{_LOG} 경고: 동적 규약 위반 {f}", file=sys.stderr)
    platform_mark = platform_flag()
    if platform_mark:  # check_static 이 이미 경고했다(샘플마다 되풀이하지 않는다)
        flags.append(platform_mark)

    return DynamicStatus(
        load_1m=load_1m,
        load_ok=load_ok,
        pinned_busy_pct=round(pinned_busy, 2),
        sibling_busy_pct=round(sibling_busy, 2),
        busy_ok=busy_ok,
        governor=governor,
        governor_ok=governor_ok,
        flags=flags,
        ok=flags_ok(flags),
        checked_at=now_iso(),
    )


# ----------------------------------------------------------------------------- 워커 실행 환경


def launch_prefix(cfg: Config, threads: int) -> list[str]:
    """``taskset`` 접두: threads==1 이면 ``single_core``, 아니면 ``cores``. taskset 이 없으면 ``[]`` + 경고."""
    global _warned_taskset
    if shutil.which("taskset") is None:
        if not _warned_taskset:
            print(f"{_LOG} 경고: taskset 이 없어 코어 고정 없이 실행한다 (플래그 taskset_missing)", file=sys.stderr)
            _warned_taskset = True
        return []
    spec = cfg.measurement.single_core if threads == 1 else cfg.measurement.cores
    return ["taskset", "-c", spec]


# 호출 친화도 규칙의 판(版). 규칙을 바꾸면 올린다 — 환경 지문(env_hash)에 들어가 어느 규칙으로 잰 행인지 구분된다.
CALL_AFFINITY_POLICY = "2026-11-11"


def bench_cpus(cfg: Config) -> str:
    """bench 워커가 taskset 으로 고정되는 CPU 목록(``launch_prefix(cfg, measurement.threads)`` 와 같은 집합)."""
    return cfg.measurement.single_core if int(cfg.measurement.threads) == 1 else cfg.measurement.cores


def call_cpus(cfg: Config, *, backend: str, openmp: bool, threads: int) -> str | None:
    """bench(교차 측정)에서 그 코드를 **호출하는 동안** 워커의 주 스레드가 가져야 할 CPU 집합(taskset 목록 문자열).

    - OpenMP 런타임 위에서 도는 코드(``openmp``; OpenMP C, 그리고 스레딩 계층이 omp 인 numba — ``LanguageCfg.omp_runtime``):
      ``None`` — libgomp 가 ``OMP_PROC_BIND``·``OMP_PLACES`` 로 정한 고정(주 스레드 = place 0)을 그대로 둔다.
    - 스레드를 쓸 수 있는 그 밖의 코드(python, halide, rayon 을 쓰는 rust): bench 워커의 고정 코어 전체. 프롬프트에 적은
      환경(고정 코어 전체, 스레드 수)과 같다. 그 코드가 처음 만드는 스레드 풀이 이 마스크를 물려받는다.
    - 순차 코드(``threads == 1``; c_seq): 고정 코어의 첫 CPU 하나. 프롬프트의 "코어 1개에 고정"과 같다. 첫 CPU 는 OpenMP 주
      스레드의 place 이자 워커의 주 스레드가 이미 돌던 CPU 라서, 호출마다 식은 코어로 옮겨 가지 않고(검토 실측: 다른 코어로
      옮기면 +36 %) libgomp 작업 스레드의 코어(나머지 place)와도 겹치지 않는다. check 단계의 ``single_core`` 는 프로세스
      전체를 taskset 으로 고정하는 별도 규약이다.

    왜 필요한가: 교차 측정에서는 한 프로세스에 OpenMP 공유 객체가 함께 로드되면 libgomp 가 주 스레드를 CPU 하나에 묶는다.
    OpenMP 가 아닌 코드를 부를 때 이 집합을 적용해야 그 코드가 만드는 스레드가 1-CPU 마스크를 물려받지 않는다
    (2026-11-11, docs/decisions.md). 반환값은 언제나 ``bench_cpus(cfg)`` 의 부분집합이다.

    ``backend`` 는 판정에 쓰지 않는다(호출부의 기록용). 2026-11-11 판의 조건은 ``backend == "c" and openmp`` 였는데, 그때
    가능하던 조합(C·python 에서 ``openmp`` 는 C 일 때만 참)에서는 지금 규칙과 결과가 같다 — 새 언어의 규칙만 더한 것이라
    ``CALL_AFFINITY_POLICY`` 는 올리지 않았다(2026-11-23)."""
    if openmp:
        return None
    pinned = bench_cpus(cfg)
    if int(threads) == 1:
        return str(parse_cpulist(pinned)[0])
    return pinned


def launch_env(cfg: Config, threads: int, *, openmp: bool, extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """워커 환경: 화이트리스트(PATH, HOME, LANG, TMPDIR, VIRTUAL_ENV 상속; LC_ALL=C, PYTHONHASHSEED=0, PYTHONPATH=ROOT)
    + 스레드 변수(OMP/OPENBLAS/MKL_NUM_THREADS=threads) + (openmp 이면) ``measurement.omp_env_extra``
    + ``extra_env`` (언어의 ``env`` 설정 — NUMBA_/RAYON_/HL_ 스레드 변수 등; 마지막에 덮어쓴다).

    사용자 환경의 OMP_*/GOMP_*/OPENBLAS_*/MKL_*/KMP_* 는 상속하지 않고 한 번 경고한다(플래그 env_overrides_dropped).
    """
    env: dict[str, str] = {}
    for key in ENV_INHERIT:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env["LC_ALL"] = "C"
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = str(ROOT)  # 어떤 cwd 에서도 `python -m harness.worker` 가 되게
    if not IS_POSIX:
        # 개발용 Windows: 워커의 진단(로그 파일로 가는 stderr)이 로캘 코드페이지가 아니라 UTF-8 로 남게 한다
        env["PYTHONIOENCODING"] = "utf-8"

    dropped = dropped_env_overrides()
    new_drops = [k for k in dropped if k not in _warned_env_drop]
    if new_drops:
        _warned_env_drop.update(new_drops)
        print(f"{_LOG} 경고: 사용자 환경의 {new_drops} 를 워커에 상속하지 않는다 (플래그 env_overrides_dropped)", file=sys.stderr)

    n = str(int(threads))
    env["OMP_NUM_THREADS"] = n
    env["OPENBLAS_NUM_THREADS"] = n
    env["MKL_NUM_THREADS"] = n
    # 교차 측정(커널·기준선 번갈아 호출)에서 OpenBLAS 스레드의 긴 스핀 대기가 커널 스레드와 코어를 다투지 않게 빨리 재운다
    env["OPENBLAS_THREAD_TIMEOUT"] = str(int(getattr(cfg.measurement, "openblas_thread_timeout", 16)))
    if openmp:
        env.update({k: str(v) for k, v in cfg.measurement.omp_env_extra.items()})
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})
    return env


# ----------------------------------------------------------------------------- BLAS·환경 정체


def blas_identity() -> dict[str, Any]:
    """현재 프로세스의 NumPy BLAS 정체: show_config 의 blas name/version/openblas configuration + 로드된 *blas* .so 경로."""
    import numpy as np  # 지연 임포트: 오케스트레이터는 서브프로세스 헬퍼를 쓴다

    name = ""
    version = ""
    config = ""
    try:
        info = np.show_config(mode="dicts")
        blas = (info.get("Build Dependencies") or {}).get("blas") or {}
        name = str(blas.get("name") or "")
        version = str(blas.get("version") or "")
        config = str(blas.get("openblas configuration") or "")
    except Exception as e:  # show_config 형식이 바뀐 경우에도 정체 기록은 계속한다
        print(f"{_LOG} 경고: numpy.show_config 읽기 실패: {e}", file=sys.stderr)
    # BLAS 라이브러리가 실제로 매핑되도록 작은 sgemm 을 한 번 호출한다
    a = np.ones((8, 8), dtype=np.float32)
    np.matmul(a, a)
    loaded: list[str] = []
    try:
        with open("/proc/self/maps", "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                path = parts[-1] if len(parts) >= 6 else ""
                if "blas" in path.lower() and path not in loaded:
                    loaded.append(path)
    except OSError:
        pass
    return {"name": name, "version": version, "config": config, "loaded_lib": ";".join(loaded)}


def _dist_version(dist: str) -> str | None:
    """설치된 배포판의 버전(임포트 없이 메타데이터만). 없으면 None."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sdk_versions() -> dict[str, str | None]:
    """설치된 LLM SDK 버전."""
    return {dist: _dist_version(dist) for dist in ("anthropic", "openai")}


def env_identity() -> dict[str, Any]:
    """현재 인터프리터의 python/numpy/BLAS/SDK 정체(``--env-identity`` 가 출력하는 것)."""
    import numpy as np

    return {
        "python_exe": sys.executable,
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "numpy_blas": blas_identity(),
        "sdk_versions": _sdk_versions(),
    }


def _unknown_blas(error: str) -> dict[str, Any]:
    return {"name": "unknown", "version": "", "config": "", "loaded_lib": "", "error": error}


def env_identity_subprocess(python_exe: Path = PYTHON_EXE) -> dict[str, Any]:
    """``PYTHON_EXE -m harness.protocol --env-identity`` 를 격리 실행해 결과 JSON을 돌려준다. 실패해도 예외 없이 unknown."""
    env = launch_env_minimal()
    try:
        r = run_isolated(
            [str(python_exe), "-m", "harness.protocol", "--env-identity"],
            timeout_s=120,
            cwd=ROOT,
            env=env,
            tail_chars=200_000,
        )
    except OSError as e:
        print(f"{_LOG} 경고: {python_exe} 실행 실패: {e}", file=sys.stderr)
        return {"python_exe": str(python_exe), "python_version": None, "numpy_version": None,
                "numpy_blas": _unknown_blas(str(e)), "sdk_versions": {"anthropic": None, "openai": None}}
    line = ""
    for candidate in reversed(r.stdout_tail.strip().splitlines()):
        if candidate.startswith("{"):
            line = candidate
            break
    if r.rc != 0 or not line:
        err = r.stderr_tail.strip()[-500:]
        print(f"{_LOG} 경고: env-identity 서브프로세스 실패(rc={r.rc}): {err}", file=sys.stderr)
        return {"python_exe": str(python_exe), "python_version": None, "numpy_version": None,
                "numpy_blas": _unknown_blas(err), "sdk_versions": {"anthropic": None, "openai": None}}
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"{_LOG} 경고: env-identity 출력 파싱 실패: {e}", file=sys.stderr)
        return {"python_exe": str(python_exe), "python_version": None, "numpy_version": None,
                "numpy_blas": _unknown_blas(str(e)), "sdk_versions": {"anthropic": None, "openai": None}}
    return data


def blas_identity_subprocess(python_exe: Path = PYTHON_EXE) -> dict[str, Any]:
    """``PYTHON_EXE`` 서브프로세스에서 잰 NumPy BLAS 정체(§2: machine.yaml 의 numpy_blas 는 반드시 이것)."""
    data = env_identity_subprocess(python_exe)
    blas = data.get("numpy_blas")
    if not isinstance(blas, dict):
        return _unknown_blas("missing numpy_blas in env identity")
    return blas


def launch_env_minimal() -> dict[str, str]:
    """프로브용 최소 환경(스레드 1; 사용자 OMP_* 등 미상속)."""
    env: dict[str, str] = {}
    for key in ENV_INHERIT:
        value = os.environ.get(key)
        if value is not None:
            env[key] = value
    env.update({"LC_ALL": "C", "PYTHONHASHSEED": "0", "PYTHONPATH": str(ROOT),
                "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    return env


def toolchain_env(*, posix: bool | None = None) -> dict[str, str]:
    """cargo·rustc 를 띄우는 환경: POSIX 는 ``launch_env_minimal`` + 사용자의 ``TOOLCHAIN_ENV_KEYS`` (rustup 이 툴체인을 찾는 데
    필요하다), 그 밖(개발용 Windows)은 현재 환경 전체. Rust 빌드(``harness.build.rust``)와 도구 점검(``toolchain_identity``)이
    같은 함수(``rust_tool_env``)를 써서 점검한 rustc 와 빌드한 rustc 가 같다. ``posix`` 는 테스트용이다(기본: ``procutil.IS_POSIX``)."""
    if not (IS_POSIX if posix is None else posix):
        return dict(os.environ)
    env = launch_env_minimal()
    env.update({key: os.environ[key] for key in TOOLCHAIN_ENV_KEYS if key in os.environ})
    return env


def rust_tool(name: str, configured: str | None = None) -> str:
    """``cargo``·``rustc`` 의 실행 파일(빌드·프롬프트·도구 점검이 같이 쓴다).

    ``configured``(설정의 ``build.rust.cargo`` 등)가 경로(구분자 포함)면 그대로 쓴다. 이름뿐이면 PATH → ``$CARGO_HOME/bin`` →
    ``~/.cargo/bin`` 순으로 찾는다. rustup 은 셸 설정을 바꾸지 않고 설치할 수 있어(decisions.md 2026-11-12: PATH 를
    건드리지 않았다) PATH 에 없는 것이 정상일 수 있다. 어디에도 없으면 이름을 그대로 돌려준다 — 호출부가 실행에 실패해
    ``ToolchainMissing`` 으로 알린다."""
    want = configured or name
    if os.sep in want or (os.altsep is not None and os.altsep in want):
        return want
    found = shutil.which(want)
    if found:
        return found
    exe = want + (".exe" if os.name == "nt" and not want.lower().endswith(".exe") else "")
    for home in (os.environ.get("CARGO_HOME"), str(Path.home() / ".cargo")):
        if home:
            candidate = Path(home) / "bin" / exe
            if candidate.is_file():
                return str(candidate)
    return want


def rust_tool_env(cargo: str | None = None, *, posix: bool | None = None) -> dict[str, str]:
    """``rust_tool`` 로 찾은 cargo·rustc 를 띄우는 환경: ``toolchain_env`` 의 PATH 앞에 ``cargo`` 의 디렉터리를 붙인다(경로일
    때만). rustup 을 PATH 를 바꾸지 않고 설치한 머신(decisions.md 2026-11-12)에서도 cargo 가 같은 디렉터리의 rustc 를 찾는다.
    ``cargo`` 가 None 이면 ``rust_tool("cargo")``."""
    env = toolchain_env(posix=posix)
    tool = rust_tool("cargo") if cargo is None else cargo
    parent = os.path.dirname(tool)
    if parent:
        env["PATH"] = parent + os.pathsep + env["PATH"] if env.get("PATH") else parent
    return env


def rust_flags(rcfg: Any, *, linux: bool | None = None) -> list[str]:
    """Rust 빌드의 ``RUSTFLAGS`` 원소: 설정의 ``rustflags`` + (Linux 에서만) ``rustflags_linux``.

    빌드(``harness.build.rust``), 프롬프트의 빌드 명령(``harness.generate``), 행의 ``cflags`` 열이 모두 이 함수의 값을 써서
    공개한 명령과 실행한 명령이 같다. ``rcfg`` 는 ``RustCfg`` 또는 같은 속성을 가진 객체(``config_option`` 의 기본값),
    ``linux`` 는 테스트용이다(기본: ``sys.platform`` 이 linux 로 시작하는가)."""
    flags = [str(f) for f in (getattr(rcfg, "rustflags", None) or [])]
    if sys.platform.startswith("linux") if linux is None else linux:
        flags += [str(f) for f in (getattr(rcfg, "rustflags_linux", None) or [])]
    return flags


# ----------------------------------------------------------------------------- 컴파일러·머신 프로브


def _version_output(command: str, env: dict[str, str] | None = None) -> str | None:
    """``command --version`` 의 stdout(앞뒤 공백 제거). 실행 파일이 없거나 실행이 실패했거나 출력이 없으면 None.
    ``env`` 가 없으면 ``launch_env_minimal()`` 로 띄운다."""
    if shutil.which(command) is None:
        return None
    try:
        r = run_isolated([command, "--version"], timeout_s=30, cwd=ROOT, env=launch_env_minimal() if env is None else env)
    except OSError:
        return None
    if r.rc != 0:
        return None
    return r.stdout_tail.strip() or None


def compiler_version(cc: str, env: dict[str, str] | None = None) -> str | None:
    """``cc --version`` 첫 줄(gcc·clang·cargo·rustc 공통). 없으면 None. ``env``: ``_version_output`` 과 같다."""
    text = _version_output(cc, env)
    return text.splitlines()[0].strip() if text else None


def path_tool_version(tool: str, pattern: re.Pattern[str]) -> str | None:
    """PATH 의 ``tool --version`` 출력에서 ``pattern`` 의 첫 그룹(버전 문자열). 도구가 없으면 None.

    출력은 있는데 패턴이 맞지 않으면(모르는 형식) 첫 줄을 그대로 돌려준다 — 있는 도구를 없다고 적지 않는다.
    """
    text = _version_output(tool)
    if text is None:
        return None
    m = pattern.search(text)
    return m.group(1) if m else text.splitlines()[0].strip()


_MARCH_RE = re.compile(r"^\s*-(march|mtune)=\s*(\S+)\s*$")


def resolve_march(cc: str = "gcc") -> str | None:
    """``cc -march=native -Q --help=target`` 에서 ``-march=``/``-mtune=`` 해석값 (예 ``-march=alderlake -mtune=alderlake``)."""
    if shutil.which(cc) is None:
        return None
    try:
        r = run_isolated([cc, "-march=native", "-Q", "--help=target"], timeout_s=30, cwd=ROOT,
                         env=launch_env_minimal(), tail_chars=200_000)
    except OSError:
        return None
    if r.rc != 0:
        return None
    found: dict[str, str] = {}
    for line in r.stdout_tail.splitlines():
        m = _MARCH_RE.match(line)
        if m:
            found[m.group(1)] = m.group(2)
    if "march" not in found:
        return None
    parts = [f"-march={found['march']}"]
    if "mtune" in found:
        parts.append(f"-mtune={found['mtune']}")
    return " ".join(parts)


def _openmp_works(cc: str) -> bool:
    """``cc -fopenmp`` 로 omp.h 를 포함한 작은 파일이 컴파일되는가(실행하지 않음)."""
    if shutil.which(cc) is None:
        return False
    src = "#include <omp.h>\nint probe(void) { return omp_get_max_threads(); }\n"
    with tempfile.TemporaryDirectory(prefix="omp_probe_") as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "omp_probe.c").write_text(src, encoding="utf-8")
        try:
            r = run_isolated([cc, "-fopenmp", "-c", "omp_probe.c", "-o", "omp_probe.o"], timeout_s=60,
                             cwd=tmp_path, env=launch_env_minimal())
        except OSError:
            return False
        return r.rc == 0


def _cpuinfo_field(key: str) -> str | None:
    text = _read_text(Path("/proc/cpuinfo")) or ""
    for line in text.splitlines():
        if line.startswith(key):
            _, _, value = line.partition(":")
            return value.strip()
    return None


def _cache_summary(cpu: int) -> dict[str, Any]:
    """``cpu<n>/cache/index*`` 의 레벨·종류별 크기와 공유 CPU 목록."""
    out: dict[str, Any] = {}
    base = SYS_CPU / f"cpu{cpu}" / "cache"
    if not base.exists():
        return out
    for idx in sorted(base.glob("index*")):
        level = _read_text(idx / "level")
        ctype = _read_text(idx / "type")
        size = _read_text(idx / "size")
        shared = _read_text(idx / "shared_cpu_list")
        if not level or not ctype:
            continue
        suffix = {"Data": "d", "Instruction": "i"}.get(ctype, "")
        out[f"L{level}{suffix}"] = {"size": size, "shared_cpu_list": shared}
    out["measured_on"] = f"cpu{cpu}"
    return out


def _memory_gb() -> float | None:
    text = _read_text(Path("/proc/meminfo")) or ""
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            kb = int(line.split()[1])
            return round(kb / (1024 * 1024), 1)
    return None


def _os_name() -> str:
    text = _read_text(Path("/etc/os-release")) or ""
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip().strip('"')
    return platform.platform()


def _sudo_nopasswd() -> bool:
    """``sudo -n true`` 가 비밀번호 없이 성공하는가(아무것도 바꾸지 않는 확인)."""
    if shutil.which("sudo") is None:
        return False
    try:
        r = run_isolated(["sudo", "-n", "true"], timeout_s=10, cwd=ROOT, env=launch_env_minimal())
    except OSError:
        return False
    return r.rc == 0


# ----------------------------------------------------------------------------- 툴체인 정체 (설계 v3 §9.5)


def halide_autoscheduler_error(name: str) -> str | None:
    """현재 인터프리터에서 Halide 오토스케줄러 ``name`` 의 플러그인을 로드해 본다. 되면 None, 안 되면 사유 한 줄.

    워커가 쓰는 로더(``harness.kernels.halide_pipe.load_autoscheduler``; 설계 v3 §8.3)를 그대로 부른다 — 도구 점검과 워커가
    다른 답을 내면 없는 플러그인이 ``compile_error`` 행이 된다. 로더가 stderr 에 적는 사유를 가로채 돌려준다.
    """
    import halide as hl

    from harness.kernels import KernelError
    from harness.kernels.halide_pipe import load_autoscheduler

    said = io.StringIO()
    try:
        with contextlib.redirect_stderr(said):
            load_autoscheduler(hl, name)
    except KernelError as e:
        return " ".join(said.getvalue().split()) or e.detail
    return None


def python_toolchain(autoscheduler: str) -> dict[str, Any]:
    """현재 인터프리터에서 쓸 수 있는 numba·halide·린트 도구(``--python-toolchain`` 이 출력하는 것).

    - ``numba_version`` / ``halide_version``: 임포트가 되면 버전, 안 되면 None. 메타데이터가 아니라 실제 임포트로 확인한다 —
      설치돼 있어도 임포트가 깨져 있으면 워커가 커널을 돌릴 수 없다.
    - ``halide_autoscheduler_loadable``: 오토스케줄러 ``autoscheduler`` 의 플러그인이 로드되는가(halide 가 없으면 None).
    - ``lint_tools``: ruff·pyflakes 의 버전(설치돼 있지 않으면 None).
    - ``errors``: 쓸 수 없는 것의 사유(``numba``, ``halide``, ``halide_autoscheduler``).
    """
    out: dict[str, Any] = {"numba_version": None, "halide_version": None, "halide_autoscheduler_loadable": None,
                           "lint_tools": {tool: _dist_version(tool) for tool in PYTHON_LINT_TOOLS}, "errors": {}}
    for package in ("numba", "halide"):
        try:
            module = __import__(package)
        except Exception as e:  # noqa: BLE001 - ImportError 뿐 아니라 임포트 중의 어떤 예외든 "쓸 수 없다"
            out["errors"][package] = f"{type(e).__name__}: {e}"
            continue
        # halide 패키지에는 __version__ 이 없다(22.1): 메타데이터에서 읽는다
        out[f"{package}_version"] = str(getattr(module, "__version__", None) or _dist_version(package) or "unknown")
    if out["halide_version"] is not None:
        error = halide_autoscheduler_error(autoscheduler)
        out["halide_autoscheduler_loadable"] = error is None
        if error is not None:
            out["errors"]["halide_autoscheduler"] = error
    return out


def python_toolchain_subprocess(autoscheduler: str, python_exe: Path = PYTHON_EXE) -> dict[str, Any]:
    """``PYTHON_EXE -m harness.protocol --python-toolchain <autoscheduler>`` 를 격리 실행한 결과(``python_toolchain`` 의 형태).

    커널을 돌리는 것은 오케스트레이터가 아니라 ``PYTHON_EXE`` 이므로 그 인터프리터에서 확인한다. 실행이 실패해도 예외 없이
    전부 None 인 결과를 주고 사유를 ``errors["python"]`` 에 적는다.
    """
    try:
        r = run_isolated([str(python_exe), "-m", "harness.protocol", "--python-toolchain", autoscheduler],
                         timeout_s=120, cwd=ROOT, env=launch_env_minimal(), tail_chars=200_000)
    except OSError as e:
        reason = f"{python_exe} 실행 실패: {e}"
    else:
        line = next((ln for ln in reversed(r.stdout_tail.strip().splitlines()) if ln.startswith("{")), "")
        try:
            data = json.loads(line) if r.rc == 0 else None
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
        reason = f"rc={r.rc}, timed_out={r.timed_out}: {r.stderr_tail.strip()[-500:]}"
    print(f"{_LOG} 경고: python-toolchain 서브프로세스 실패: {reason}", file=sys.stderr)
    return {"numba_version": None, "halide_version": None, "halide_autoscheduler_loadable": None,
            "lint_tools": dict.fromkeys(PYTHON_LINT_TOOLS), "errors": {"python": reason}}


def runtime_probe(kind: str, *, autoscheduler: str | None = None, params: dict[str, Any] | None = None,
                  threads: int | None = None) -> str | None:
    """현재 인터프리터에서 언어의 런타임 설정을 워커와 같은 코드로 써 본다(``--probe-runtime``). 되면 None, 안 되면 사유 한 줄.
    LLM 코드는 실행하지 않는다.

    - ``halide``: ``harness.kernels.halide_pipe.make_loader`` (워커의 준비 그대로): 오토스케줄러 플러그인 로드 + 하네스의 한 줄짜리
      파이프라인에 설정의 매개변수(+ 하네스가 더하는 ``parallelism`` = ``threads``)로 오토스케줄러 적용. 모르는 매개변수 키는
      Halide 22.1 에서 예외가 아니라 프로세스 중단이다 — 부모(``runtime_probe_subprocess``)가 종료 코드로 알아챈다.
    - ``numba``: ``harness.kernels.pyfunc.start_threading_layer`` — 환경의 ``NUMBA_THREADING_LAYER`` 를 띄운다.
    """
    try:
        if kind == "halide":
            from harness.kernels import KernelOptions
            from harness.kernels.halide_pipe import make_loader

            make_loader(KernelOptions(threads=threads, halide_mode="auto", halide_autoscheduler=str(autoscheduler),
                                      halide_params=dict(params or {})))
        elif kind == "numba":
            from harness.kernels.pyfunc import start_threading_layer

            start_threading_layer()
        else:
            raise ValueError(f"모르는 런타임 점검: {kind!r}")
    except Exception as e:  # noqa: BLE001 - 어떤 실패든 "그 설정을 쓸 수 없다"
        return f"{type(e).__name__}: {' '.join(str(getattr(e, 'detail', None) or e).split())}"
    return None


def runtime_probe_subprocess(kind: str, *, env: dict[str, str], prefix: list[str] | None = None, autoscheduler: str | None = None,
                             params: dict[str, Any] | None = None, threads: int | None = None,
                             python_exe: Path = PYTHON_EXE) -> str | None:
    """``[prefix…] PYTHON_EXE -m harness.protocol --probe-runtime <kind> …`` 를 언어의 환경 ``env`` 로 격리 실행한다(run.py 의
    도구 점검). 되면 None, 안 되면 사유 한 줄 — 결과 JSON 없이 끝난 프로세스(Halide 의 중단 등)도 사유가 된다."""
    cmd = [*(prefix or []), str(python_exe), "-m", "harness.protocol", "--probe-runtime", kind]
    if kind == "halide":
        cmd += ["--halide-autoscheduler", str(autoscheduler), "--halide-params", json.dumps(dict(params or {}), sort_keys=True),
                "--threads", str(threads)]
    try:
        r = run_isolated(cmd, timeout_s=RUNTIME_PROBE_TIMEOUT_S, cwd=ROOT, env=env, tail_chars=200_000)
    except OSError as e:
        return f"{python_exe} 실행 실패: {e}"
    line = next((ln for ln in reversed(r.stdout_tail.strip().splitlines()) if ln.startswith("{")), "")
    try:
        data = json.loads(line) if r.rc == 0 else None
    except json.JSONDecodeError:
        data = None
    if isinstance(data, dict) and "error" in data:
        return None if data["error"] is None else str(data["error"])
    stderr = " | ".join(ln.strip() for ln in r.stderr_tail.strip().splitlines()[-5:] if ln.strip())
    return f"점검 프로세스가 결과 없이 끝났다 (rc={r.rc}, timed_out={r.timed_out}, signal={r.signal_name}): {stderr[-500:]}"


def toolchain_identity(cfg: Config | None) -> dict[str, Any]:
    """빌드·실행·린트 도구의 정체. run.py 의 도구 점검(설계 v3 §9.5)과 머신 프로브가 쓴다(환경 지문에는 넣지 않는다 —
    지문의 키는 2026-11-12 측정 규약 그대로다; docs/decisions.md 2026-11-23).

    - ``cc`` (설정의 컴파일러 이름), ``cc_version`` / ``cargo`` / ``rustc`` (``--version`` 첫 줄).
    - ``numba_version`` / ``halide_version``, ``halide_autoscheduler`` (설정의 이름) / ``halide_autoscheduler_loadable``.
    - ``lint_tools``: ruff·pyflakes(``PYTHON_EXE`` 의 패키지), cppcheck·clang-tidy(PATH) 의 버전.
    - ``errors``: ``python_toolchain`` 이 남긴 사유.

    쓸 수 없는 도구의 값은 None 이다. ``cfg`` 가 None 이면(설정 없이 도는 프로브) cc=gcc 와 ``OPTION_DEFAULTS`` 로 본다.
    """
    cc = cfg.build.cc if cfg is not None else DEFAULT_CC
    autoscheduler = str(config_option(cfg, "halide.autoscheduler"))
    py = python_toolchain_subprocess(autoscheduler)
    lint_tools = dict(py["lint_tools"])
    for tool, pattern in PATH_LINT_TOOLS.items():
        lint_tools[tool] = path_tool_version(tool, pattern)
    cargo = rust_tool("cargo", str(config_option(cfg, "build.rust").cargo))
    return {
        "cc": cc,
        "cc_version": compiler_version(cc),
        # cargo·rustc 는 빌드와 같은 실행 파일(rust_tool)·환경(rust_tool_env: RUSTUP_TOOLCHAIN·RUSTUP_HOME·CARGO_HOME + cargo 의
        # 디렉터리를 앞에 붙인 PATH)으로 본다
        "cargo": compiler_version(cargo, env=rust_tool_env(cargo)),
        "rustc": compiler_version(rust_tool("rustc"), env=rust_tool_env(cargo)),
        "numba_version": py["numba_version"],
        "halide_version": py["halide_version"],
        "halide_autoscheduler": autoscheduler,
        "halide_autoscheduler_loadable": py["halide_autoscheduler_loadable"],
        "lint_tools": lint_tools,
        "errors": dict(py["errors"]),
    }


def probe_machine(cfg: Config | None = None) -> dict[str, Any]:
    """§2 ``machine.yaml`` 키 전부를 실측한다(BLAS·SDK 버전은 PYTHON_EXE 서브프로세스로). 컴파일러·cargo·오토스케줄러는
    설정의 것을 본다(``toolchain_identity``; ``cfg`` 가 None 이면 기본값)."""
    cpus = online_cpus()
    packages: set[str] = set()
    cores: set[tuple[str, str]] = set()
    for cpu in cpus:
        pkg = _read_text(SYS_CPU / f"cpu{cpu}" / "topology" / "physical_package_id") or "0"
        core = _read_text(SYS_CPU / f"cpu{cpu}" / "topology" / "core_id") or str(cpu)
        packages.add(pkg)
        cores.add((pkg, core))
    siblings = {cpu: thread_siblings(cpu) for cpu in cpus}
    smt_active = _read_text(SYS_CPU / "smt" / "active")
    smt = (smt_active == "1") if smt_active is not None else any(len(s) > 1 for s in siblings.values())

    def _cpulist(path: Path) -> list[int]:
        text = _read_text(path)
        if not text:
            return []
        try:
            return parse_cpulist(text)
        except ValueError:
            return []

    p_cores = _cpulist(Path("/sys/devices/cpu_core/cpus"))
    e_cores = _cpulist(Path("/sys/devices/cpu_atom/cpus"))
    flags_text = _cpuinfo_field("flags") or ""
    flag_set = set(flags_text.split())
    isa = [f for f in ISA_FLAGS_OF_INTEREST if f in flag_set]

    governor, _ = governor_status(cpus)
    epp, _ = epp_status(cpus)
    gov_avail_text = read_cpufreq(cpus[0], "scaling_available_governors") if cpus else None
    identity = env_identity_subprocess()
    core_pattern = _read_text(Path("/proc/sys/kernel/core_pattern")) or ""

    toolchain = toolchain_identity(cfg)
    cc = toolchain["cc"]
    gcc_ver = toolchain["cc_version"] if cc == "gcc" else compiler_version("gcc")
    clang_ver = toolchain["cc_version"] if cc == "clang" else compiler_version("clang")
    compilers: dict[str, Any] = {
        "gcc": gcc_ver,
        "clang": clang_ver,
        "openmp_gcc": _openmp_works("gcc") if gcc_ver else False,
        "openmp_clang": _openmp_works("clang") if clang_ver else False,
    }
    if cc not in ("gcc", "clang"):
        compilers[cc] = toolchain["cc_version"]

    notes: list[str] = []
    if p_cores and e_cores:
        notes.append(f"hybrid CPU: P-cores {p_cores[0]}-{p_cores[-1]} (SMT pairs), E-cores {e_cores[0]}-{e_cores[-1]}; measure on P-cores only")
    if governor != "performance":
        notes.append(f"governor={governor}: fix with `{FIX_GOVERNOR}`")
    if epp not in ("performance", "n/a"):
        notes.append(f"epp={epp}: fix with `{FIX_EPP}`")
    if core_pattern.startswith("|") and "apport" in core_pattern:
        notes.append("apport core_pattern active: worker sets PR_SET_DUMPABLE=0")
    blas_name = str((identity.get("numpy_blas") or {}).get("name") or "")
    if "openblas" not in blas_name.lower():
        notes.append(f"numpy BLAS is not OpenBLAS ({blas_name or 'unknown'}); install pip wheel numpy in .venv")

    return {
        "probed_at": now_iso(),
        "hostname": socket.gethostname(),
        "os": _os_name(),
        "kernel": platform.release(),
        "glibc": platform.libc_ver()[1] or None,
        "cpu_model": _cpuinfo_field("model name"),
        "sockets": len(packages),
        "physical_cores": len(cores),
        "logical_cpus": len(cpus),
        "smt": smt,
        "p_core_cpus": p_cores,
        "e_core_cpus": e_cores,
        "thread_siblings": siblings,
        "isa": isa,
        "caches": _cache_summary(p_cores[0] if p_cores else (cpus[0] if cpus else 0)),
        "memory_gb": _memory_gb(),
        "governor": governor,
        "governor_available": gov_avail_text.split() if gov_avail_text else [],
        "epp": epp,
        "no_turbo": read_no_turbo(),
        "scaling_driver": read_cpufreq(cpus[0], "scaling_driver") if cpus else None,
        "taskset_available": shutil.which("taskset") is not None,
        "cpupower_available": shutil.which("cpupower") is not None,
        "sudo_nopasswd": _sudo_nopasswd(),
        "apport_core_pattern": core_pattern.startswith("|") and "apport" in core_pattern,
        "compilers": compilers,
        "march_resolved": resolve_march(cc),
        "python_exe": str(PYTHON_EXE),
        "python_version": identity.get("python_version"),
        "numpy_version": identity.get("numpy_version"),
        "numpy_blas": identity.get("numpy_blas"),
        "sdk_versions": identity.get("sdk_versions"),
        "cargo": toolchain["cargo"],
        "rustc": toolchain["rustc"],
        "numba_version": toolchain["numba_version"],
        "halide_version": toolchain["halide_version"],
        "halide_autoscheduler": toolchain["halide_autoscheduler"],
        "halide_autoscheduler_loadable": toolchain["halide_autoscheduler_loadable"],
        "lint_tools": toolchain["lint_tools"],
        "bandwidth_gbs": None,
        "peak_gflops": None,
        "notes": "; ".join(notes),
    }


def write_machine_yaml(machine: dict[str, Any], path: Path | None = None) -> Path:
    """머신 프로브 결과를 YAML로 원자적으로 쓴다."""
    path = path if path is not None else machine_yaml_for_host(str(machine.get("hostname") or "") or None)
    header = "# harness.protocol --probe --write 가 생성한 파일. 손으로 고치지 말고 다시 프로브하라.\n"
    body = yaml.safe_dump(machine, sort_keys=False, allow_unicode=True, default_flow_style=False, width=120)
    atomic_write_text(path, header + body)
    return path


# ----------------------------------------------------------------------------- 환경 지문


def environment_fingerprint(cfg: Config, machine: dict[str, Any], static: StaticStatus) -> tuple[dict[str, Any], str]:
    """재현성 지문: 하드웨어·툴체인·플래그·고정 코어·전원 상태를 모아 (dict, sha256[:16]).

    키와 값은 측정 규약의 일부라 고정이다(docs/decisions.md 2026-11-12 측정 규약, 2026-11-23 통합): 같은 머신·설정이면 이전 행과 ``env_hash`` 가 같아야
    한다. rustc·numba·halide·린트 도구의 버전(``toolchain_identity``)은 지문에 넣지 않는다 — 그 정보는 행 수준 열
    (``compiler``·``compiler_version``·``cflags``)과 result.json 에 남긴다.
    """
    cc = cfg.build.cc
    compilers = machine.get("compilers") if isinstance(machine.get("compilers"), dict) else {}
    cc_version = compilers.get(cc) if compilers else None
    if not cc_version:
        cc_version = compiler_version(cc)
    march = machine.get("march_resolved") or resolve_march(cc)
    python_version = machine.get("python_version")
    numpy_version = machine.get("numpy_version")
    sdk_versions = machine.get("sdk_versions")
    if not python_version or not numpy_version or not isinstance(sdk_versions, dict):
        identity = env_identity_subprocess()
        python_version = python_version or identity.get("python_version")
        numpy_version = numpy_version or identity.get("numpy_version")
        sdk_versions = sdk_versions if isinstance(sdk_versions, dict) else identity.get("sdk_versions")
    blas = static.blas or {}
    fp: dict[str, Any] = {
        "cpu_model": machine.get("cpu_model") or _cpuinfo_field("model name"),
        "kernel": machine.get("kernel") or platform.release(),
        "glibc": machine.get("glibc") or platform.libc_ver()[1],
        "cc": cc,
        "cc_version": cc_version,
        "march_resolved": march,
        "python_version": python_version,
        "numpy_version": numpy_version,
        "numpy_blas": {"name": blas.get("name"), "version": blas.get("version"), "config": blas.get("config")},
        "sdk_versions": sdk_versions,
        "flags": " ".join([*cfg.build.base_flags, *cfg.build.cflags, cfg.build.openmp_flag, *cfg.build.ldflags]),
        "cores": cfg.measurement.cores,
        "single_core": cfg.measurement.single_core,
        "threads": cfg.measurement.threads,
        "omp_env": dict(cfg.measurement.omp_env_extra),
        "call_affinity_policy": CALL_AFFINITY_POLICY,
        "bench_loop_fsync": bool(cfg.measurement.bench_loop_fsync),
        "governor": static.governor,
        "epp": static.epp,
        "no_turbo": static.no_turbo,
    }
    canonical = json.dumps(fp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return fp, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- CLI


def suggest_cores(machine: dict[str, Any], n: int) -> list[int]:
    """측정 코어 후보: 성능 코어(P-core; 하이브리드가 아니면 모든 코어) 가운데 **물리 코어마다 CPU 하나**(SMT 형제 중 번호가
    가장 작은 것)를 골라 앞에서부터 ``n`` 개. machine.yaml(``probe_machine``)의 ``p_core_cpus``·``thread_siblings`` 를 쓴다.
    새 장비로 옮길 때 ``measurement.cores`` 를 정하는 근거(docs/desktop_setup.md)."""
    cpus = [int(c) for c in (machine.get("p_core_cpus") or [])]
    if not cpus:
        siblings = machine.get("thread_siblings") or {}
        cpus = sorted(int(c) for c in siblings) if siblings else list(range(int(machine.get("logical_cpus") or 0)))
    siblings = {int(k): sorted(int(x) for x in v) for k, v in (machine.get("thread_siblings") or {}).items()}
    seen: set[tuple[int, ...]] = set()
    picked: list[int] = []
    for c in sorted(cpus):
        group = tuple(siblings.get(c, [c]))
        if group in seen:
            continue
        seen.add(group)
        picked.append(min(group))
    return picked[: max(0, int(n))]


def main(argv: list[str] | None = None) -> int:
    """``--check`` / ``--probe [--write [PATH]]`` / ``--fingerprint`` / ``--env-identity`` / ``--suggest-cores N`` /
    ``--python-toolchain NAME`` / ``--probe-runtime {halide,numba}``."""
    parser = argparse.ArgumentParser(prog="python -m harness.protocol", description="측정 규약 검사·머신 프로브")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--check", action="store_true", help="정적+동적 규약 검사 결과를 JSON으로 출력 (플래그 있으면 exit 1)")
    parser.add_argument("--probe", action="store_true", help="머신 프로브 JSON 출력")
    parser.add_argument("--write", nargs="?", const="", default=None, metavar="PATH",
                        help="--probe 결과를 YAML로 저장 (기본 config/machines/<hostname>.yaml)")
    parser.add_argument("--fingerprint", action="store_true", help="환경 지문(dict, hash) 출력")
    parser.add_argument("--env-identity", action="store_true", help="(내부용) 현재 인터프리터의 python/numpy/BLAS/SDK 정체 JSON")
    parser.add_argument("--suggest-cores", type=int, default=None, metavar="N",
                        help="config/machine.yaml 에서 측정 코어 N 개를 고른다(물리 성능 코어마다 CPU 하나) → measurement.cores 후보")
    parser.add_argument("--python-toolchain", default=None, metavar="NAME",
                        help="(내부용) 현재 인터프리터의 numba·halide·린트 도구 정체와 오토스케줄러 NAME 의 로드 가능 여부 JSON")
    parser.add_argument("--probe-runtime", choices=("halide", "numba"), default=None,
                        help="(내부용) 언어의 런타임 설정을 워커와 같은 코드로 써 본다: {\"error\": null | 사유} JSON")
    parser.add_argument("--halide-autoscheduler", default=None, help="--probe-runtime halide: 오토스케줄러 이름")
    parser.add_argument("--halide-params", type=json.loads, default=None, help="--probe-runtime halide: 매개변수 JSON 객체")
    parser.add_argument("--threads", type=int, default=None, help="--probe-runtime halide: 스레드 수(parallelism)")
    ns = parser.parse_args(argv)

    if ns.suggest_cores is not None:
        picked = suggest_cores(load_machine(), ns.suggest_cores)
        total = len(suggest_cores(load_machine(), 10_000))
        print(",".join(str(c) for c in picked))
        print(f"{_LOG} 물리 성능 코어 {total}개 중 {len(picked)}개 (SMT 형제·효율 코어 제외). "
              f"measurement.cores 에 넣고 threads 를 {len(picked)} 로 맞춘다.", file=sys.stderr)
        return 0 if len(picked) == ns.suggest_cores else 1

    if ns.env_identity:
        print(json.dumps(env_identity(), ensure_ascii=False))
        return 0
    if ns.python_toolchain is not None:
        # ASCII 로만 찍는다: 부모는 파이프를 UTF-8 로 읽는데 로캘이 UTF-8 이 아닌 플랫폼에서는 사유의 한글이 깨진다
        print(json.dumps(python_toolchain(ns.python_toolchain)))
        return 0
    if ns.probe_runtime is not None:
        error = runtime_probe(ns.probe_runtime, autoscheduler=ns.halide_autoscheduler, params=ns.halide_params, threads=ns.threads)
        print(json.dumps({"error": error}))  # ASCII (위와 같은 이유)
        return 0

    if not (ns.check or ns.probe or ns.fingerprint):
        parser.print_help()
        return 1

    cfg: Config | None = None
    if ns.check or ns.fingerprint or ns.probe:
        try:
            cfg = load_config(ns.config)
        except ConfigError as e:
            if ns.probe and not (ns.check or ns.fingerprint):
                print(f"{_LOG} 참고: 설정 로드 실패({e}); 프로브는 기본값(cc={DEFAULT_CC})으로 진행", file=sys.stderr)
            else:
                print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
                return 2

    rc = 0
    if ns.probe:
        machine = probe_machine(cfg)
        print(json.dumps(machine, ensure_ascii=False, indent=2))
        if ns.write is not None:
            out = write_machine_yaml(machine, Path(ns.write) if ns.write else None)
            print(f"{_LOG} machine.yaml 저장: {out}", file=sys.stderr)
            blas = machine.get("numpy_blas") or {}
            if "openblas" not in str(blas.get("name") or "").lower():
                print(f"{_LOG} 경고: numpy_blas={blas} — OpenBLAS 가 아니다", file=sys.stderr)
                rc = 1

    if ns.check or ns.fingerprint:
        assert cfg is not None
        machine = load_machine()
        static = check_static(cfg, machine)
        out: dict[str, Any] = {"static": asdict(static)}
        if ns.check:
            dynamic = check_dynamic(cfg)
            out["dynamic"] = asdict(dynamic)
            if not (static.ok and dynamic.ok):
                rc = 1
        if ns.fingerprint:
            fp, digest = environment_fingerprint(cfg, machine, static)
            out["fingerprint"] = fp
            out["env_hash"] = digest
        print(json.dumps(out, ensure_ascii=False, indent=2))
    return rc


if __name__ == "__main__":
    # 진단은 한국어다: 파이프·파일·로캘 코드페이지 콘솔로 나갈 때 깨지지 않게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
