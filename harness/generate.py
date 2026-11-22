"""생성 단계 (설계 §5, v3 §5): 과제 × 언어 × 조건 × n 샘플만큼 LLM 을 호출하고 원문을 ``results/raw/`` 에 저장한다.

    PYTHON_EXE -m harness.generate [--config] [--run-id ID] [--provider P] [--model M] [--languages a,b] [--n N]
                                   [--tasks t] [--args positional|keyword] [--grammar-hint on|off]
                                   [--feedback compiler|compiler+lint] [--yes] [--force] [--no-retry-failed]

- 프롬프트: system = ``prompts/system.md``; user = 언어 파일(``prompts/lang/<lang>.md``)을 ``string.Template`` 로 렌더링
  (``${task_spec}`` = ``tasks/<task>/spec.md``). ``prompt_hash = sha256(system + "\\n\\x00\\n" + user)[:16]``.
- 시그니처·매개변수 설명은 ``harness.sigs.signature_bundle`` (YAML ``languages`` 항목이 있으면 그것이 우선한다).
- 템플릿 변수(어느 언어 템플릿에서나 쓸 수 있다; ``render_prompt`` 의 ``providers``): ``signature``, ``indexing``, ``param_docs``,
  ``function``, ``python_version``, ``numpy_version``, ``cc``, ``cc_version``, ``build_command``, ``threads``, ``cores_desc``,
  ``cpu_summary``, ``task_spec``, ``numba_signature``, ``numba_version``, ``numba_threading_layer``, ``halide_version``,
  ``halide_autoscheduler``, ``halide_coord_example``, ``rustc_version``, ``cargo_command``, ``rust_profile``, ``rayon_version``,
  ``args_note``, ``grammar_hint``. 버전은 머신 프로필(``config/machines/<호스트>.yaml``, 없으면 ``config/machine.yaml``)에 있으면 그 값, 없으면 ``PYTHON_EXE`` 서브프로세스나 도구에
  묻고, 그래도 모르면 경고와 함께 ``unknown`` 이다. release 프로파일과 rayon 버전은 Rust 빌드 템플릿의 ``Cargo.toml`` 에서 읽는다.
- 조건: ``--args``·``--grammar-hint``·``--feedback`` 이 설정의 ``run.conditions`` 를 덮어쓴다(raw 의 ``conditions`` 와 경로의
  ``cond_tag`` 에 반영된다). ``args=keyword`` 면 ``${args_note}`` 가, ``grammar_hint=on`` 이면 ``${grammar_hint}``
  (``prompts/hints/<언어>.md``)가 채워진다. positional·hint off 에서는 둘 다 빈 문자열이라 기존 프롬프트와 바이트 단위로 같다.
- 건너뛰는 조합: 과제가 그 언어를 지원하지 않으면(``signatures.yaml`` 의 ``unsupported``) ``unsupported``,
  ``grammar_hint=on`` 인데 그 언어의 힌트 파일이 없으면 ``skip`` (on 과 off 의 프롬프트가 같아지는 것을 막는다). 끝에
  과제 × 언어 표(``call``/``skip``/``unsupported``)를 낸다.
- 저장: ``results/raw/<run_id>/<task>/<language>/<cond_tag>/sample_<i>.json`` (키는 §5 raw JSON 목록 그대로).
- 건너뛰기: 기존 파일이 ``error is None and stop_reason in {end_turn, stop_sequence}`` 면 건너뜀; 그 외(오류·절단·refusal)는
  재호출(``--no-retry-failed`` 면 건너뜀); ``--force`` 면 모두 재호출. 기존 파일은 ``sample_<i>.prev<N>.json`` 으로 보관.
- 비-dummy: ``preflight()`` → 계획 출력 → ``--yes`` 없으면 ``Proceed? [y/N]``; 비대화형이고 ``--yes`` 없으면 exit 2.
- ``error`` 가 ``fatal:`` 이면 그 raw 를 저장한 뒤 즉시 중단(exit 3). transient 오류는 계속 진행한다.
- 종료 코드: 0 정상, 1 설정/입력 오류, 2 확인 거부·비대화형, 3 치명 API 오류, 130 중단(Ctrl-C).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any, Callable

from harness import sigs
from harness.build.rust import cargo_command_text
from harness.config import (
    ARGS_MODES, FEEDBACKS, GRAMMAR_HINTS, BuildCfg, Config, ConfigError, LanguageCfg, RustCfg, config_snapshot,
    load_config, load_machine, parse_cpulist,
)
from harness.llm_client import LLMClient, LLMClientError, LLMResponse
from harness.paths import PROMPTS_DIR, ROOT, TASKS_DIR, atomic_write_json, cond_tag, raw_sample_path, rel_to_root
from harness.protocol import (
    compiler_version, env_identity_subprocess, now_iso, python_toolchain_subprocess, rust_tool, toolchain_env,
)
from harness.schema import harness_version
from harness.taskspec import TaskSpec, TaskSpecError, load_taskspec

_LOG = "[generate]"

OK_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
PROMPT_HASH_SEPARATOR = "\n\x00\n"

HINTS_DIR: Path = PROMPTS_DIR / "hints"
GRAMMAR_HINT_HEADING = "## Language notes"
# Rust 빌드 템플릿(harness.build.rust 가 샘플마다 복사해 쓰는 것): 프롬프트가 서술하는 release 프로파일과 rayon 버전의 출처
RUST_MANIFEST: Path = ROOT / "harness" / "build" / "rust_template" / "Cargo.toml"
NUMBA_THREADING_LAYER_ENV = "NUMBA_THREADING_LAYER"
UNKNOWN = "unknown"

# (과제, 언어) 조합의 계획 상태(끝에 내는 표의 값)
PAIR_CALL = "call"
PAIR_SKIP = "skip"
PAIR_UNSUPPORTED = "unsupported"
# 샘플의 계획: decide_action 의 call/skip_ok/skip_failed 밖에, 시드 복사본이라 다시 묻지 않는 raw
SKIP_SEEDED = "skip_seeded"

# march 해석값 → (P-core, E-core) 마이크로아키텍처 이름 (알려진 것만; 모르면 생략)
_UARCH_BY_MARCH: dict[str, tuple[str, str]] = {
    "alderlake": ("Golden Cove", "Gracemont"),
    "raptorlake": ("Raptor Cove", "Gracemont"),
}


# ----------------------------------------------------------------------------- 머신 서술

def _format_cache_size(size: Any) -> str | None:
    """sysfs 캐시 크기 문자열("48K", "1280K", "18432K", "1M")을 "48 KiB" / "1.25 MiB" 로."""
    if size is None:
        return None
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMG]?)\s*", str(size))
    if not m:
        return str(size)
    value = float(m.group(1))
    unit = m.group(2)
    bytes_ = value * {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}[unit]
    if bytes_ >= 1024 ** 2:
        mib = bytes_ / 1024 ** 2
        text = f"{mib:.2f}".rstrip("0").rstrip(".")
        return f"{text} MiB"
    kib = bytes_ / 1024
    text = f"{kib:.2f}".rstrip("0").rstrip(".")
    return f"{text} KiB"


def _clean_cpu_model(model: Any) -> str:
    text = re.sub(r"\((?:R|TM|r|tm)\)", "", str(model or "")).strip()
    text = re.sub(r"\s+", " ", text)
    return text or "unknown CPU"


def _uarch_names(machine: dict[str, Any]) -> tuple[str | None, str | None]:
    march = str(machine.get("march_resolved") or "")
    m = re.search(r"-march=([A-Za-z0-9_-]+)", march)
    if not m:
        return None, None
    return _UARCH_BY_MARCH.get(m.group(1).lower(), (None, None))


def _core_kind(machine: dict[str, Any], pinned: list[int]) -> tuple[str, str | None]:
    """(코어 종류 문구, 마이크로아키텍처 이름)."""
    p_cores = set(int(c) for c in (machine.get("p_core_cpus") or []))
    e_cores = set(int(c) for c in (machine.get("e_core_cpus") or []))
    p_name, e_name = _uarch_names(machine)
    if p_cores and set(pinned) <= p_cores:
        return "performance core", p_name
    if e_cores and set(pinned) <= e_cores:
        return "efficiency core", e_name
    return "core", None


def _smt_free(machine: dict[str, Any], pinned: list[int]) -> bool:
    """고정 CPU 집합에 SMT 형제 쌍이 없으면 True(= 코어당 1스레드)."""
    siblings = machine.get("thread_siblings") or {}
    pinned_set = set(pinned)
    for cpu in pinned:
        sibs = siblings.get(cpu, siblings.get(str(cpu), []))
        for s in sibs or []:
            if int(s) != cpu and int(s) in pinned_set:
                return False
    return True


def pinned_cpus_for(cfg: Config, threads: int) -> list[int]:
    """threads==1 이면 single_core, 아니면 cores (protocol.launch_prefix 와 같은 규칙)."""
    return parse_cpulist(cfg.measurement.single_core if threads == 1 else cfg.measurement.cores)


def cores_desc(cfg: Config, machine: dict[str, Any], threads: int) -> str:
    """``${cores_desc}``: "1 physical performance core, 1 thread" / "4 physical performance cores, 1 thread per core"."""
    pinned = pinned_cpus_for(cfg, threads)
    kind, _ = _core_kind(machine, pinned)
    n = len(pinned)
    if threads == 1:
        return f"1 physical {kind}, 1 thread"
    plural = kind + "s"
    if threads == n:
        return f"{n} physical {plural}, 1 thread per core"
    return f"{n} physical {plural}, {threads} threads"


def cpu_summary(cfg: Config, machine: dict[str, Any], threads: int) -> str:
    """``${cpu_summary}``: 머신 프로필에서 조립하되 고정 코어 집합만 서술한다(설계 §4)."""
    pinned = pinned_cpus_for(cfg, threads)
    n = len(pinned)
    kind, uarch = _core_kind(machine, pinned)
    model = _clean_cpu_model(machine.get("cpu_model"))
    core_text = f"{n} physical {kind}{'' if n == 1 else 's'}"
    if uarch:
        core_text += f" ({uarch})"
    if threads == 1:
        thread_text = "one thread"
    elif threads == n:
        thread_text = "one thread per core"
    else:
        thread_text = f"{threads} threads"
    if machine.get("smt") and _smt_free(machine, pinned):
        thread_text += ", no SMT"
    parts = [model, f"the kernel runs pinned to {core_text}, {thread_text}"]

    isa = [str(f).lower() for f in (machine.get("isa") or [])]
    if isa:
        vec: list[str] = []
        if "avx512f" in isa:
            vec.append("AVX-512")
        elif "avx2" in isa:
            vec.append("AVX2")
        elif "avx" in isa:
            vec.append("AVX")
        if "fma" in isa:
            vec.append("FMA")
        text = " + ".join(vec) if vec else "no AVX"
        if "avx512f" not in isa:
            text += " (no AVX-512)"
        parts.append(text)

    caches = machine.get("caches") or {}
    if isinstance(caches, dict):
        l1 = _format_cache_size((caches.get("L1d") or {}).get("size"))
        l2 = _format_cache_size((caches.get("L2") or {}).get("size"))
        l3 = _format_cache_size((caches.get("L3") or {}).get("size"))
        per_core = [s for s in (f"L1d {l1}" if l1 else None, f"L2 {l2}" if l2 else None) if s]
        cache_parts: list[str] = []
        if per_core:
            cache_parts.append(" and ".join(per_core) + " per core")
        if l3:
            cache_parts.append(f"L3 {l3} shared")
        if cache_parts:
            parts.append(", ".join(cache_parts))
    return "; ".join(parts)


def build_command(bcfg: BuildCfg, *, openmp: bool) -> str:
    """프롬프트에 공개하는 전체 컴파일 명령(diag_flags 는 코드 생성 무관이라 제외; build/c.py 가 따로 붙인다)."""
    parts = [bcfg.cc, *bcfg.base_flags, *bcfg.cflags]
    if openmp:
        parts.append(bcfg.openmp_flag)
    parts += ["-o", "kernel.so", "kernel.c", *bcfg.ldflags]
    return " ".join(parts)


def cargo_command(rcfg: RustCfg, *, linux: bool | None = None) -> str:
    """프롬프트에 공개하는 Rust 빌드 명령: ``RUSTFLAGS="…" cargo build --release [--offline] --locked``
    (build/rust.py 가 실행하는 것; 산출물 위치만 정하는 ``CARGO_TARGET_DIR`` 은 코드 생성과 무관해 뺀다).

    문자열은 빌드 모듈의 ``cargo_command_text`` 가 만든다: ``RUSTFLAGS`` 는 빌드·``cflags`` 열과 같은 ``protocol.rust_flags`` 의
    값이고(Linux 에서만 ``rustflags_linux`` 가 붙는다), 도구는 경로가 아니라 맨 이름 ``cargo`` 로 적는다 — 설정이나
    ``protocol.rust_tool`` 이 전체 경로를 찾아도 프롬프트가 머신마다 달라지지 않는다(decisions.md 2026-11-23).
    ``linux`` 는 테스트용이다(기본: 이 머신)."""
    return cargo_command_text(rcfg, linux=linux)


# ----------------------------------------------------------------------------- 환경 값

def _toml_value(value: Any) -> str:
    """TOML 값 하나를 원문 꼴로(``3``, ``"fat"``, ``false``)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    return str(value)


class _EnvInfo:
    """프롬프트가 서술하는 환경 값(파이썬·NumPy·Numba·Halide·컴파일러 버전, Rust 빌드 템플릿)을 한 번만 조사한다.

    버전은 머신 프로필(``config/machines/<호스트>.yaml``, 없으면 ``config/machine.yaml``)에 있으면 그 값을 쓰고, 없으면 ``PYTHON_EXE`` 서브프로세스나 도구에 묻는다(프로브가
    머신 프로필에 쓰는 것과 같은 함수). 어느 쪽으로도 알 수 없으면 경고하고 ``"unknown"`` 으로 둔다. release 프로파일과
    rayon 버전은 Rust 빌드 템플릿의 ``Cargo.toml`` 에서 읽는다.
    """

    def __init__(self, cfg: Config, machine: dict[str, Any]) -> None:
        self.cfg = cfg
        self.machine = machine
        self._identity: dict[str, Any] | None = None
        self._cc_version: str | None = None
        self._python_toolchain: dict[str, Any] | None = None
        self._rust_manifest: dict[str, Any] | None = None
        self._resolved: dict[str, str] = {}

    def _identity_value(self, key: str) -> str:
        value = self.machine.get(key)
        if value:
            return str(value)
        if self._identity is None:
            print(f"{_LOG} 참고: machine.yaml 에 {key} 가 없어 서브프로세스로 조사한다", file=sys.stderr)
            self._identity = env_identity_subprocess()
        return str(self._identity.get(key) or "unknown")

    def python_version(self) -> str:
        return self._identity_value("python_version")

    def numpy_version(self) -> str:
        return self._identity_value("numpy_version")

    def cc_version(self) -> str:
        if self._cc_version is None:
            cc = self.cfg.build.cc
            compilers = self.machine.get("compilers") if isinstance(self.machine.get("compilers"), dict) else {}
            value = compilers.get(cc) if compilers else None
            if not value:
                value = compiler_version(cc)
            if not value:
                print(f"{_LOG} 경고: {cc} 버전을 알 수 없다", file=sys.stderr)
                value = "unknown"
            self._cc_version = str(value)
        return self._cc_version

    def _once(self, key: str, find: Callable[[], Any], missing: str) -> str:
        """``find()`` 의 값을 한 번만 구한다. 값이 없으면 ``missing`` 을 경고로 한 번 남기고 ``"unknown"``."""
        if key not in self._resolved:
            value = find()
            if not value:
                print(f"{_LOG} 경고: {missing} — 프롬프트에 {UNKNOWN!r} 으로 들어간다", file=sys.stderr)
                value = UNKNOWN
            self._resolved[key] = str(value)
        return self._resolved[key]

    def _package_version(self, key: str) -> Any:
        """머신 프로필의 ``numba_version``/``halide_version``, 없으면 ``PYTHON_EXE`` 에서 임포트해 본 버전."""
        value = self.machine.get(key)
        if value:
            return value
        if self._python_toolchain is None:
            print(f"{_LOG} 참고: machine.yaml 에 {key} 가 없어 서브프로세스로 조사한다", file=sys.stderr)
            self._python_toolchain = python_toolchain_subprocess(self.cfg.halide.autoscheduler)
        return self._python_toolchain.get(key)

    def numba_version(self) -> str:
        return self._once("numba_version", lambda: self._package_version("numba_version"), "numba 버전을 알 수 없다")

    def halide_version(self) -> str:
        return self._once("halide_version", lambda: self._package_version("halide_version"), "halide 버전을 알 수 없다")

    def rustc_version(self) -> str:
        # rustup 이 PATH 밖(~/.cargo/bin)에 있어도 찾는다(protocol.rust_tool; decisions.md 2026-11-12 설치)
        return self._once("rustc_version",
                          lambda: self.machine.get("rustc") or compiler_version(rust_tool("rustc", getattr(self.cfg.build.rust, "rustc", None)), env=toolchain_env()),
                          "rustc 버전을 알 수 없다")

    def _manifest(self) -> dict[str, Any]:
        """Rust 빌드 템플릿의 ``Cargo.toml`` (없거나 읽을 수 없으면 경고하고 빈 dict)."""
        if self._rust_manifest is None:
            try:
                with RUST_MANIFEST.open("rb") as fh:
                    self._rust_manifest = tomllib.load(fh)
            except (OSError, tomllib.TOMLDecodeError) as e:
                print(f"{_LOG} 경고: Rust 빌드 템플릿을 읽을 수 없다: {RUST_MANIFEST} ({e})", file=sys.stderr)
                self._rust_manifest = {}
        return self._rust_manifest

    def rust_profile(self) -> str:
        """``[profile.release]`` 의 설정을 파일 순서대로: `` `opt-level = 3`, `lto = "fat"`, … ``."""
        def find() -> str:
            profile = (self._manifest().get("profile") or {}).get("release")
            if not isinstance(profile, dict):
                return ""
            return ", ".join(f"`{key} = {_toml_value(value)}`" for key, value in profile.items())
        return self._once("rust_profile", find, f"{RUST_MANIFEST.name} 에 [profile.release] 가 없다")

    def rayon_version(self) -> str:
        """``[dependencies] rayon`` 의 고정 버전(``"=1.2.3"`` → ``1.2.3``)."""
        def find() -> str:
            dep = (self._manifest().get("dependencies") or {}).get("rayon")
            requirement = dep.get("version") if isinstance(dep, dict) else dep
            return requirement.strip().lstrip("=").strip() if isinstance(requirement, str) else ""
        return self._once("rayon_version", find, f"{RUST_MANIFEST.name} 에 rayon 의존성 버전이 없다")


# ----------------------------------------------------------------------------- 프롬프트

@dataclass
class RenderedPrompt:
    """한 (task, language) 의 렌더링 결과."""

    system: str
    user: str
    prompt_hash: str
    signature: str


def prompt_hash(system: str, user: str) -> str:
    """``sha256(system + "\\n\\x00\\n" + user)[:16]``."""
    return hashlib.sha256((system + PROMPT_HASH_SEPARATOR + user).encode("utf-8")).hexdigest()[:16]


def load_spec(task: str) -> TaskSpec:
    """``tasks/<task>/signatures.yaml`` 의 과제 명세. 없거나 잘못됐으면 ConfigError."""
    path = TASKS_DIR / task / "signatures.yaml"
    try:
        return load_taskspec(path)
    except FileNotFoundError:
        raise ConfigError(f"signatures.yaml 이 없다: {path}") from None
    except TaskSpecError as e:
        raise ConfigError(f"과제 명세가 잘못됐다: {e}") from None


def hint_path(language: str) -> Path:
    """``grammar_hint=on`` 일 때 붙는 문법 안내 파일: ``prompts/hints/<language>.md``."""
    return HINTS_DIR / f"{language}.md"


def render_grammar_hint(language: str) -> str:
    """``${grammar_hint}`` (on 일 때): ``"## Language notes\\n\\n" + 힌트 파일 내용 + "\\n\\n"``. 파일이 없으면 ConfigError
    (``main`` 은 그런 언어를 렌더링 전에 건너뛴다)."""
    path = hint_path(language)
    if not path.is_file():
        raise ConfigError(f"grammar_hint=on 인데 힌트 파일이 없다: {path}")
    hint = path.read_text(encoding="utf-8").strip("\n")
    return f"{GRAMMAR_HINT_HEADING}\n\n{hint}\n\n"


def halide_coordinate_example(spec: TaskSpec) -> str:
    """``${halide_coord_example}``: Halide 좌표 규칙(설계 v3 §4: 마지막 배열 축이 첫 좌표)의 구체적인 예 한 문장.
    그 과제의 입력 배열 가운데 차원이 가장 많은 첫 배열로 든다(매개변수 설명에 나오는 이름·형태 그대로)."""
    arg = max(spec.inputs, key=lambda a: len(a.shape))
    dims = [str(d).strip() for d in arg.shape]
    shape = "(" + ", ".join(dims) + ("," if len(dims) == 1 else "") + ")"
    index = [f"i{k}" for k in range(len(dims))]
    ranges = " and ".join(f"0 <= {i} < {d}" for i, d in zip(index, dims))
    return (f"For example, the input `{arg.name}` has shape {shape}: its element at array index ({', '.join(index)}), with {ranges}, "
            f"is `{arg.name}[{', '.join(reversed(index))}]`.")


def render_args_note(bundle: dict[str, Any], backend: str, args_mode: str) -> str:
    """``${args_note}``: keyword 모드에서만 매개변수 설명 바로 뒤에 붙는 문단(개행 두 개로 시작해 빈 줄 하나를 사이에 둔다).
    positional 이면 빈 문자열.

    - python 계열(python·numba·halide): 하네스가 모든 인자를 이름으로 넘긴다는 사실.
    - c: 하네스가 쓰는 ``kernel_args.h`` 전문과 ``#include`` 안내. rust: ``src/kernel_args.rs`` 전문과 ``mod`` 안내.
    """
    if args_mode != "keyword":
        return ""
    function = bundle["function"]
    if backend == "c":
        note = ("The harness writes the following file as `kernel_args.h` into the directory of `kernel.c` before compiling:\n\n"
                f"```c\n{bundle['args_struct']}```\n\n"
                'Include it with `#include "kernel_args.h"` and use the structure it declares; do not declare the structure '
                "yourself. The harness fills one such structure and passes its address as `args`.")
    elif backend == "rust":
        note = ("The harness writes the following file as `src/kernel_args.rs` next to `src/lib.rs` before building:\n\n"
                f"```rust\n{bundle['args_struct']}```\n\n"
                f"Declare it as a module with `mod {sigs.RUST_ARGS_MODULE};` and use the structure it defines; do not define the "
                "structure yourself. The harness fills one such structure and passes its address as `args`.")
    else:
        call = ", ".join(f"{name}=..." for name in bundle["params"])
        note = (f"Every parameter of `{function}` is keyword-only (they follow the bare `*` in the signature). The harness passes "
                f"all arguments by name, as `{function}({call})`, never by position.")
    return "\n\n" + note


def _numba_threading_layer(lang: LanguageCfg) -> str:
    layer = lang.env.get(NUMBA_THREADING_LAYER_ENV)
    if not layer:
        raise ConfigError(f"languages.{lang.name}.env: 템플릿이 서술하는 {NUMBA_THREADING_LAYER_ENV} 가 설정에 없다")
    return layer


def render_prompt(cfg: Config, machine: dict[str, Any], task: str, lang: LanguageCfg, *, args_mode: str,
                  grammar_hint: str = "off", env: _EnvInfo | None = None) -> RenderedPrompt:
    """system.md + 언어 파일(${task_spec} = spec.md) 렌더링(설계 §4, v3 §5 변수).

    변수는 어느 언어 템플릿에서나 전부 쓸 수 있고, 값은 템플릿이 실제로 쓰는 것만 구한다(numpy 프롬프트를 만들면서
    rustc 를 찾지 않는다).
    """
    env = env or _EnvInfo(cfg, machine)
    system_path = PROMPTS_DIR / "system.md"
    spec_path = TASKS_DIR / task / "spec.md"
    for p in (system_path, lang.prompt, spec_path):
        if not p.is_file():
            raise ConfigError(f"프롬프트/명세 파일이 없다: {p}")
    system = system_path.read_text(encoding="utf-8").rstrip("\n") + "\n"
    template = Template(lang.prompt.read_text(encoding="utf-8"))
    spec = load_spec(task)
    bundle = sigs.signature_bundle(spec, lang.name, lang.backend, args_mode)
    providers: dict[str, Callable[[], str]] = {
        "signature": lambda: bundle["signature"],
        "indexing": lambda: str((spec.languages_raw.get(lang.name) or {}).get("indexing") or ""),
        "param_docs": lambda: bundle["param_docs"],
        "function": lambda: bundle["function"],
        "python_version": env.python_version,
        "numpy_version": env.numpy_version,
        "cc": lambda: cfg.build.cc,
        "cc_version": env.cc_version,
        "build_command": lambda: build_command(cfg.build, openmp=lang.openmp) if lang.backend == "c" else "",
        "threads": lambda: str(lang.threads),
        "cores_desc": lambda: cores_desc(cfg, machine, lang.threads),
        "cpu_summary": lambda: cpu_summary(cfg, machine, lang.threads),
        "task_spec": lambda: spec_path.read_text(encoding="utf-8").rstrip("\n"),
        "numba_signature": lambda: sigs.numba_signature(spec),
        "numba_version": env.numba_version,
        "numba_threading_layer": lambda: _numba_threading_layer(lang),
        "halide_version": env.halide_version,
        "halide_autoscheduler": lambda: cfg.halide.autoscheduler,
        "halide_coord_example": lambda: halide_coordinate_example(spec),
        "rustc_version": env.rustc_version,
        "cargo_command": lambda: cargo_command(cfg.build.rust),
        "rust_profile": env.rust_profile,
        "rayon_version": env.rayon_version,
        "args_note": lambda: render_args_note(bundle, lang.backend, args_mode),
        "grammar_hint": lambda: render_grammar_hint(lang.name) if grammar_hint == "on" else "",
    }
    names = template.get_identifiers()
    unknown = [name for name in names if name not in providers]
    if unknown:
        raise ConfigError(f"{lang.prompt}: 알 수 없는 템플릿 변수 {unknown}")
    try:
        user = template.substitute({name: providers[name]() for name in names})
    except ValueError as e:
        raise ConfigError(f"{lang.prompt}: 템플릿 변수 오류 {e}") from None
    user = user.rstrip("\n") + "\n"
    return RenderedPrompt(system=system, user=user, prompt_hash=prompt_hash(system, user), signature=bundle["signature"])


# ----------------------------------------------------------------------------- 작업 계획

@dataclass
class WorkItem:
    """샘플 하나의 계획."""

    task: str
    language: str
    sample_idx: int
    path: Path
    action: str  # call | skip_ok | skip_failed
    previous: dict[str, Any] | None


@dataclass
class PairPlan:
    """(과제, 언어) 조합 하나의 계획과 진행(끝에 내는 표의 한 칸)."""

    status: str  # PAIR_CALL | PAIR_SKIP | PAIR_UNSUPPORTED
    planned: int = 0  # 호출하기로 한 샘플 수
    called: int = 0  # 실제로 호출한 샘플 수

    def cell(self) -> str:
        return f"{PAIR_CALL} {self.called}/{self.planned}" if self.status == PAIR_CALL else self.status


def coverage_table(tasks: list[str], languages: list[str], pairs: dict[tuple[str, str], PairPlan]) -> str:
    """과제 × 언어 표. 칸: ``call <호출한 수>/<호출하기로 한 수>`` | ``skip`` (호출할 샘플이 없다: 기존 raw 를 그대로 두거나
    그 언어의 힌트 파일이 없다) | ``unsupported`` (과제가 그 언어를 지원하지 않는다)."""
    rows = [["task", *languages]] + [[task, *(pairs[(task, language)].cell() for language in languages)] for task in tasks]
    widths = [max(len(row[col]) for row in rows) for col in range(len(rows[0]))]
    return "\n".join("  ".join(text.ljust(width) for text, width in zip(row, widths)).rstrip() for row in rows)


def _read_previous(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(f"{_LOG} 경고: 기존 raw 를 읽을 수 없어 재호출한다: {path} ({e})", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def decide_action(path: Path, *, force: bool, no_retry_failed: bool) -> tuple[str, dict[str, Any] | None]:
    """건너뛰기 규칙(설계 §5)."""
    if not path.exists():
        return "call", None
    prev = _read_previous(path)
    if force or prev is None:
        return "call", prev
    if prev.get("error") is None and prev.get("stop_reason") in OK_STOP_REASONS:
        return "skip_ok", prev
    return ("skip_failed" if no_retry_failed else "call"), prev


def archive_previous(path: Path) -> Path | None:
    """기존 ``sample_<i>.json`` 을 ``sample_<i>.prev<N>.json`` (가장 작은 빈 N) 으로 옮긴다."""
    if not path.exists():
        return None
    n = 1
    while True:
        target = path.with_name(f"{path.stem}.prev{n}.json")
        if not target.exists():
            os.replace(path, target)
            return target
        n += 1


def default_run_id(provider: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{provider}"


def build_raw(cfg: Config, *, run_id: str, task: str, lang: LanguageCfg, sample_idx: int, turn: int,
              conditions: dict[str, str], prompt: dict[str, Any], prompt_hash: str, resp: LLMResponse,
              snapshot: dict[str, Any], version: str) -> dict[str, Any]:
    """raw JSON (설계 §5 키 순서 그대로). 턴 0 은 ``prompt = {system, user}``; 수정 턴(harness.repair)은 ``messages`` 가 더 있다."""
    m = cfg.model
    provider = m.provider
    if provider == "anthropic":
        temperature_sent: float | None = m.temperature
        thinking: str | None = m.anthropic.thinking
        effort: str | None = m.anthropic.effort
        seed: int | None = None
    elif provider == "openai_compat":
        temperature_sent = m.temperature
        thinking = None
        effort = None
        seed = m.seed
    else:  # dummy: 아무것도 보내지 않는다
        temperature_sent = None
        thinking = None
        effort = None
        seed = None
    return {
        "run_id": run_id,
        "task": task,
        "language": lang.name,
        "backend": lang.backend,
        "sample_idx": sample_idx,
        "turn": turn,
        "conditions": {"args": conditions["args"], "grammar_hint": conditions["grammar_hint"], "feedback": conditions["feedback"]},
        "provider": provider,
        "model_requested": resp.model_requested,
        "model_served": resp.model_served,
        "sampling": {
            "temperature_sent": temperature_sent,
            "temperature_effective": "provider_default" if temperature_sent is None else temperature_sent,
            "thinking": thinking,
            "effort": effort,
            "seed": seed,
            "top_p": None,
        },
        "max_tokens": m.max_tokens,
        "prompt": prompt,
        "prompt_hash": prompt_hash,
        "response_text": resp.text,
        "stop_reason": resp.stop_reason,
        "raw_stop_reason": resp.raw_stop_reason,
        "stop_details_category": resp.stop_details_category,
        "n_thinking_blocks": resp.n_thinking_blocks,
        "n_text_blocks": resp.n_text_blocks,
        "tokens_in": resp.tokens_in,
        "tokens_out": resp.tokens_out,
        "tokens_thinking": resp.tokens_thinking,
        "tokens_text": resp.tokens_text,
        "latency_ms": resp.latency_ms,
        "request_id": resp.request_id,
        "base_url": resp.base_url,
        "api_key_present": resp.api_key_present,
        "sdk_version": resp.sdk_version,
        "error": resp.error,
        "created_at": now_iso(),
        "harness_version": version,
        "config_snapshot": snapshot,
    }


# ----------------------------------------------------------------------------- 실행

@dataclass
class Summary:
    calls: int = 0
    skipped_ok: int = 0
    skipped_failed: int = 0
    failed: int = 0
    truncated: int = 0
    refusals: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_thinking: int = 0

    def line(self, run_id: str, raw_root: Path) -> str:
        return (
            f"{_LOG} run_id={run_id} calls={self.calls} skipped_ok={self.skipped_ok} skipped_failed={self.skipped_failed} "
            f"failed={self.failed} truncated={self.truncated} refusals={self.refusals} "
            f"tokens_in={self.tokens_in} tokens_out={self.tokens_out} (thinking {self.tokens_thinking}) raw={rel_to_root(raw_root)}"
        )


def confirm_calls(n_calls: int, cfg: Config, *, yes: bool, log: str = _LOG) -> bool:
    """실제 API 호출 전의 확인 게이트(generate 와 repair 가 같이 쓴다): 계획을 출력하고 확인을 받는다.
    비대화형이고 --yes 가 없으면 False. ``log`` 는 진단 줄의 머리말(부르는 모듈의 ``[module]``)이다."""
    m = cfg.model
    effort = m.anthropic.effort if m.provider == "anthropic" else None
    print(f"{log} {n_calls} calls to {m.name} via {m.provider}, max_tokens={m.max_tokens}, effort={effort}, "
          f"temperature={'provider_default' if m.temperature is None else m.temperature}", file=sys.stderr)
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{log} 비대화형 실행이다. 실제 API 호출에는 --yes 가 필요하다 (exit 2)", file=sys.stderr)
        return False
    try:
        answer = input("Proceed? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness.generate", description="LLM 생성 단계: 프롬프트 렌더링·호출·원문 저장")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--run-id", default=None, help="run 식별자 (기본 YYYYmmdd-HHMMSS-<provider>)")
    parser.add_argument("--provider", default=None, help="model.provider 덮어쓰기 (anthropic|openai_compat|dummy; env LLM_PROVIDER 와 동일)")
    parser.add_argument("--model", default=None, help="model.name 덮어쓰기 (env LLM_MODEL 과 동일)")
    parser.add_argument("--languages", default=None, help="쉼표 구분 언어 목록 (기본 run.languages)")
    parser.add_argument("--n", type=int, default=None, help="샘플 수 (기본 run.n_samples)")
    parser.add_argument("--tasks", default=None, help="쉼표 구분 과제 목록 (기본 run.tasks)")
    parser.add_argument("--args", choices=ARGS_MODES, default=None, help="조건 args 덮어쓰기 (기본 run.conditions.args)")
    parser.add_argument("--grammar-hint", choices=GRAMMAR_HINTS, default=None, help="조건 grammar_hint 덮어쓰기 (기본 run.conditions.grammar_hint)")
    parser.add_argument("--feedback", choices=FEEDBACKS, default=None, help="조건 feedback 덮어쓰기 (기본 run.conditions.feedback)")
    parser.add_argument("--yes", action="store_true", help="실제 API 호출 확인 프롬프트 생략")
    parser.add_argument("--force", action="store_true", help="기존 raw 가 있어도 모두 재호출")
    parser.add_argument("--no-retry-failed", action="store_true", help="오류·절단 샘플도 재호출하지 않음")
    ns = parser.parse_args(argv)

    # CLI 덮어쓰기는 로더의 env 경로를 그대로 탄다(검증 규칙이 같이 적용된다)
    if ns.provider:
        os.environ["LLM_PROVIDER"] = ns.provider
    if ns.model:
        os.environ["LLM_MODEL"] = ns.model
    try:
        cfg = load_config(ns.config, overrides={"args": ns.args, "grammar_hint": ns.grammar_hint, "feedback": ns.feedback})
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 1
    machine = load_machine()
    provider = cfg.model.provider

    tasks = [t.strip() for t in ns.tasks.split(",") if t.strip()] if ns.tasks else list(cfg.run.tasks)
    languages = [l.strip() for l in ns.languages.split(",") if l.strip()] if ns.languages else list(cfg.run.languages)
    n = ns.n if ns.n is not None else cfg.run.n_samples
    if n < 1:
        print(f"{_LOG} --n 은 1 이상이어야 한다", file=sys.stderr)
        return 1
    unknown = [l for l in languages if l not in cfg.languages]
    if unknown:
        print(f"{_LOG} 알 수 없는 언어 {unknown} (정의: {list(cfg.languages)})", file=sys.stderr)
        return 1
    # run.tasks 밖의 과제는 harness.run·repair 가 건너뛴다: 그 응답은 행이 되지 못하므로 호출하지 않는다
    outside = [t for t in tasks if t not in cfg.run.tasks]
    if outside:
        print(f"{_LOG} 설정 run.tasks 에 없는 과제 {outside} (run/repair 가 건너뛴다; run.tasks: {list(cfg.run.tasks)})", file=sys.stderr)
        return 1
    run_id = ns.run_id or default_run_id(provider)
    cond = cfg.run.conditions
    conditions = {"args": cond.args, "grammar_hint": cond.grammar_hint, "feedback": cond.feedback}
    tag = cond_tag(cond.args, cond.grammar_hint, cond.feedback)
    raw_root = cfg.paths.raw_dir / run_id

    # 계획: 지원하지 않는 조합과 힌트가 없는 언어를 먼저 거른다
    hint_missing: set[str] = set()
    if cond.grammar_hint == "on":
        for language in languages:
            if not hint_path(language).is_file():
                hint_missing.add(language)
                print(f"{_LOG} 경고: grammar_hint=on 인데 {rel_to_root(hint_path(language))} 가 없어 {language} 를 건너뛴다 "
                      "(on 과 off 의 프롬프트가 같아지면 조건 비교가 무의미하다)", file=sys.stderr)
    pairs: dict[tuple[str, str], PairPlan] = {}
    items: list[WorkItem] = []
    try:
        for task in tasks:
            spec = load_spec(task)
            for language in languages:
                if not spec.supports(language):
                    pairs[(task, language)] = PairPlan(PAIR_UNSUPPORTED)
                    print(f"{_LOG} {task}/{language}: 지원하지 않는 조합이라 건너뛴다 — {spec.unsupported[language]}", file=sys.stderr)
                    continue
                if language in hint_missing:
                    pairs[(task, language)] = PairPlan(PAIR_SKIP)
                    continue
                planned = 0
                for i in range(n):
                    path = raw_sample_path(cfg.paths.raw_dir, run_id, task, language, tag, i)
                    action, prev = decide_action(path, force=ns.force, no_retry_failed=ns.no_retry_failed)
                    if action == "call" and prev is not None and prev.get("seeded_from"):
                        # 다른 피드백 조건의 턴 0 을 복사한 raw(harness.repair --seed-from-feedback): 여기서 새로 물으면 대응
                        # 비교의 짝이 다른 응답에서 시작한다 — 원본을 채운 뒤 다시 시드해야 한다
                        print(f"{_LOG} 경고: {rel_to_root(path)} 는 {prev['seeded_from']} 의 시드 복사본이라 다시 호출하지 않는다 — "
                              "원본을 채운 뒤 `python -m harness.repair --seed-from-feedback …` 로 다시 시드하라", file=sys.stderr)
                        action = SKIP_SEEDED
                    items.append(WorkItem(task, language, i, path, action, prev))
                    planned += action == "call"
                pairs[(task, language)] = PairPlan(PAIR_CALL if planned else PAIR_SKIP, planned=planned)
    except ConfigError as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return 1
    n_calls = sum(1 for it in items if it.action == "call")
    summary = Summary(
        skipped_ok=sum(1 for it in items if it.action == "skip_ok"),
        skipped_failed=sum(1 for it in items if it.action == "skip_failed"),
    )

    # 프롬프트는 먼저 전부 렌더링한다(템플릿 오류를 호출 전에 잡는다)
    env = _EnvInfo(cfg, machine)
    if not machine:
        print(f"{_LOG} 경고: machine.yaml 이 없어 하드웨어 서술이 빈약하다. `.venv/bin/python -m harness.protocol --probe --write` 를 먼저 실행하라.", file=sys.stderr)
    prompts: dict[tuple[str, str], RenderedPrompt] = {}
    try:
        for task, language in dict.fromkeys((it.task, it.language) for it in items):
            prompts[(task, language)] = render_prompt(cfg, machine, task, cfg.languages[language], args_mode=cond.args,
                                                      grammar_hint=cond.grammar_hint, env=env)
        client = LLMClient(cfg)
    except (ConfigError, LLMClientError) as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return 1

    def report() -> None:
        print(coverage_table(tasks, languages, pairs))
        print(summary.line(run_id, raw_root))

    version = harness_version()
    if provider != "dummy":
        if version.startswith("nogit-"):
            print(f"{_LOG} 경고: git 저장소가 아니라 harness_version={version} 이다 (원칙 4: 재현성을 위해 git 으로 관리하라)", file=sys.stderr)
        try:
            client.preflight()
        except Exception as e:  # noqa: BLE001 — 인증/모델명 오류를 호출 전에 드러낸다
            print(f"{_LOG} preflight 실패: {type(e).__name__}: {e}", file=sys.stderr)
            return 3
        if n_calls == 0:
            print(f"{_LOG} 호출할 샘플이 없다 (모두 건너뜀)", file=sys.stderr)
            report()
            return 0
        if not confirm_calls(n_calls, cfg, yes=ns.yes):
            return 2
    else:
        client.preflight()
        print(f"{_LOG} {n_calls} dummy calls (run_id={run_id})", file=sys.stderr)

    snapshot = config_snapshot(cfg)
    rc = 0
    try:
        for it in items:
            if it.action != "call":
                continue
            lang = cfg.languages[it.language]
            prompt = prompts[(it.task, it.language)]
            resp = client.complete(prompt.system, prompt.user, dummy_key=(it.task, it.language, it.sample_idx))
            raw = build_raw(cfg, run_id=run_id, task=it.task, lang=lang, sample_idx=it.sample_idx, turn=0, conditions=conditions,
                            prompt={"system": prompt.system, "user": prompt.user}, prompt_hash=prompt.prompt_hash, resp=resp,
                            snapshot=snapshot, version=version)
            archived = archive_previous(it.path)
            atomic_write_json(it.path, raw)
            summary.calls += 1
            pairs[(it.task, it.language)].called += 1
            summary.tokens_in += resp.tokens_in
            summary.tokens_out += resp.tokens_out
            summary.tokens_thinking += resp.tokens_thinking or 0
            label = f"{it.task}/{it.language}/sample_{it.sample_idx}"
            if resp.error is not None:
                summary.failed += 1
                print(f"{_LOG} {label}: error={resp.error}" + (f" (이전 파일 → {archived.name})" if archived else ""), file=sys.stderr)
                if resp.error.startswith("fatal:"):
                    print(f"{_LOG} 치명 오류 — 중단한다 (exit 3). raw 저장: {rel_to_root(it.path)}", file=sys.stderr)
                    rc = 3
                    break
                continue
            if resp.stop_reason == "max_tokens":
                summary.truncated += 1
            elif resp.stop_reason == "refusal":
                summary.refusals += 1
            print(
                f"{_LOG} {label}: stop={resp.stop_reason} model={resp.model_served} tokens_in={resp.tokens_in} "
                f"tokens_out={resp.tokens_out} thinking={resp.tokens_thinking} latency={resp.latency_ms:.0f}ms"
                + (f" (이전 파일 → {archived.name})" if archived else ""),
                file=sys.stderr,
            )
    except KeyboardInterrupt:
        print(f"{_LOG} 사용자 중단 (Ctrl-C)", file=sys.stderr)
        rc = 130
    report()
    return rc


if __name__ == "__main__":
    # 진단은 한국어다: 파이프·파일로 나갈 때 로캘 코드페이지가 아니라 UTF-8 로 남게 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
