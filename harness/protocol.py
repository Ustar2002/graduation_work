"""측정 규약 강제 (설계 §9): 거버너·EPP·turbo·taskset·BLAS 정체(정적), 부하·코어 사용률(동적),
워커 실행 접두(``taskset``)와 화이트리스트 환경, 머신 프로브(``config/machine.yaml``), 환경 지문.

CLI:
    python -m harness.protocol --check                  # static + dynamic 결과 JSON (플래그 있으면 exit 1)
    python -m harness.protocol --probe [--write [PATH]] # 머신 프로브 JSON; --write 면 machine.yaml 저장
    python -m harness.protocol --fingerprint            # 환경 지문 (machine.yaml 필요)
    python -m harness.protocol --env-identity           # (내부용) 현재 인터프리터의 python/numpy/BLAS/SDK 정체 JSON
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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
from typing import Any

import yaml

from harness.config import Config, ConfigError, load_config, load_machine, parse_cpulist
from harness.paths import DEFAULT_MACHINE_YAML, PYTHON_EXE, ROOT, atomic_write_text
from harness.procutil import run_isolated

_LOG = "[protocol]"

SYS_CPU = Path("/sys/devices/system/cpu")
PROC_STAT = Path("/proc/stat")

FIX_GOVERNOR = "sudo cpupower frequency-set -g performance"
FIX_EPP = "echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference"

# 정보용 플래그(ok 판정에서 제외): turbo 상태, 사용자 환경의 스레드 변수 제거 알림
INFO_FLAG_PREFIXES: tuple[str, ...] = ("turbo=", "env_overrides_dropped")

# 워커 환경 화이트리스트(상속 허용)와 고정값, 제거 대상 접두
ENV_INHERIT: tuple[str, ...] = ("PATH", "HOME", "LANG", "TMPDIR", "VIRTUAL_ENV")
ENV_DROP_PREFIXES: tuple[str, ...] = ("OMP_", "GOMP_", "OPENBLAS_", "MKL_", "KMP_")

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
    """run당 1회: 거버너·EPP·turbo·taskset·BLAS 정체. 플래그는 경고(해결 명령 포함)와 함께 돌려준다."""
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
    """bench 직전마다: 1초 ``/proc/stat`` 샘플로 고정 코어·형제 코어 사용률, load average, 거버너 재확인."""
    cpus = pinned_cpus(cfg)
    sibs = sibling_cpus(cpus)
    before = read_proc_stat()
    time.sleep(sample_s)
    after = read_proc_stat()
    pinned_busy = busy_pct(before, after, cpus)
    sibling_busy = busy_pct(before, after, sibs) if sibs else 0.0
    load_1m = float(os.getloadavg()[0])
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


def launch_env(cfg: Config, threads: int, *, openmp: bool) -> dict[str, str]:
    """워커 환경: 화이트리스트(PATH, HOME, LANG, TMPDIR, VIRTUAL_ENV 상속; LC_ALL=C, PYTHONHASHSEED=0, PYTHONPATH=ROOT)
    + 스레드 변수(OMP/OPENBLAS/MKL_NUM_THREADS=threads) + (openmp 이면) ``measurement.omp_env_extra``.

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

    dropped = dropped_env_overrides()
    new_drops = [k for k in dropped if k not in _warned_env_drop]
    if new_drops:
        _warned_env_drop.update(new_drops)
        print(f"{_LOG} 경고: 사용자 환경의 {new_drops} 를 워커에 상속하지 않는다 (플래그 env_overrides_dropped)", file=sys.stderr)

    n = str(int(threads))
    env["OMP_NUM_THREADS"] = n
    env["OPENBLAS_NUM_THREADS"] = n
    env["MKL_NUM_THREADS"] = n
    if openmp:
        env.update({k: str(v) for k, v in cfg.measurement.omp_env_extra.items()})
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


def _sdk_versions() -> dict[str, str | None]:
    """설치된 LLM SDK 버전(임포트 없이 메타데이터만)."""
    out: dict[str, str | None] = {}
    for dist in ("anthropic", "openai"):
        try:
            out[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            out[dist] = None
    return out


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


# ----------------------------------------------------------------------------- 컴파일러·머신 프로브


def compiler_version(cc: str) -> str | None:
    """``cc --version`` 첫 줄. 없으면 None."""
    if shutil.which(cc) is None:
        return None
    try:
        r = run_isolated([cc, "--version"], timeout_s=30, cwd=ROOT, env=launch_env_minimal())
    except OSError:
        return None
    if r.rc != 0:
        return None
    first = r.stdout_tail.strip().splitlines()
    return first[0].strip() if first else None


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


def probe_machine(cc: str = "gcc") -> dict[str, Any]:
    """§2 ``machine.yaml`` 키 전부를 실측한다(BLAS·SDK 버전은 PYTHON_EXE 서브프로세스로)."""
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

    gcc_ver = compiler_version("gcc")
    clang_ver = compiler_version("clang")
    cc_ver = gcc_ver if cc == "gcc" else compiler_version(cc)
    compilers: dict[str, Any] = {
        "gcc": gcc_ver,
        "clang": clang_ver,
        "openmp_gcc": _openmp_works("gcc") if gcc_ver else False,
        "openmp_clang": _openmp_works("clang") if clang_ver else False,
    }
    if cc not in ("gcc", "clang"):
        compilers[cc] = cc_ver

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
        "bandwidth_gbs": None,
        "peak_gflops": None,
        "notes": "; ".join(notes),
    }


def write_machine_yaml(machine: dict[str, Any], path: Path = DEFAULT_MACHINE_YAML) -> Path:
    """머신 프로브 결과를 YAML로 원자적으로 쓴다."""
    header = "# harness.protocol --probe --write 가 생성한 파일. 손으로 고치지 말고 다시 프로브하라.\n"
    body = yaml.safe_dump(machine, sort_keys=False, allow_unicode=True, default_flow_style=False, width=120)
    atomic_write_text(path, header + body)
    return path


# ----------------------------------------------------------------------------- 환경 지문


def environment_fingerprint(cfg: Config, machine: dict[str, Any], static: StaticStatus) -> tuple[dict[str, Any], str]:
    """재현성 지문: 하드웨어·툴체인·플래그·고정 코어·전원 상태를 모아 (dict, sha256[:16])."""
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
        "governor": static.governor,
        "epp": static.epp,
        "no_turbo": static.no_turbo,
    }
    canonical = json.dumps(fp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return fp, hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    """``--check`` / ``--probe [--write [PATH]]`` / ``--fingerprint`` / ``--env-identity``."""
    parser = argparse.ArgumentParser(prog="python -m harness.protocol", description="측정 규약 검사·머신 프로브")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--check", action="store_true", help="정적+동적 규약 검사 결과를 JSON으로 출력 (플래그 있으면 exit 1)")
    parser.add_argument("--probe", action="store_true", help="머신 프로브 JSON 출력")
    parser.add_argument("--write", nargs="?", const=DEFAULT_MACHINE_YAML, default=None, type=Path, metavar="PATH",
                        help="--probe 결과를 YAML로 저장 (기본 config/machine.yaml)")
    parser.add_argument("--fingerprint", action="store_true", help="환경 지문(dict, hash) 출력")
    parser.add_argument("--env-identity", action="store_true", help="(내부용) 현재 인터프리터의 python/numpy/BLAS/SDK 정체 JSON")
    ns = parser.parse_args(argv)

    if ns.env_identity:
        print(json.dumps(env_identity(), ensure_ascii=False))
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
                print(f"{_LOG} 참고: 설정 로드 실패({e}); 프로브는 cc=gcc 로 진행", file=sys.stderr)
            else:
                print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
                return 2

    rc = 0
    if ns.probe:
        machine = probe_machine(cc=cfg.build.cc if cfg is not None else "gcc")
        print(json.dumps(machine, ensure_ascii=False, indent=2))
        if ns.write is not None:
            out = write_machine_yaml(machine, Path(ns.write))
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
    sys.exit(main())
