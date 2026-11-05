"""생성 단계 (설계 §5): 과제 × 언어 × 조건 × n 샘플만큼 LLM 을 호출하고 원문을 ``results/raw/`` 에 저장한다.

    PYTHON_EXE -m harness.generate [--config] [--run-id ID] [--provider P] [--model M] [--languages a,b] [--n N]
                                   [--tasks t] [--yes] [--force] [--no-retry-failed]

- 프롬프트: system = ``prompts/system.md``; user = 언어 파일(``prompts/lang/<lang>.md``)을 ``string.Template`` 로 렌더링
  (``${task_spec}`` = ``tasks/<task>/spec.md``). ``prompt_hash = sha256(system + "\\n\\x00\\n" + user)[:16]``.
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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any

import yaml

from harness.config import BuildCfg, Config, ConfigError, LanguageCfg, config_snapshot, load_config, load_machine, parse_cpulist
from harness.llm_client import LLMClient, LLMClientError, LLMResponse
from harness.paths import PROMPTS_DIR, TASKS_DIR, atomic_write_json, cond_tag, raw_sample_path, rel_to_root
from harness.protocol import compiler_version, env_identity_subprocess, now_iso
from harness.schema import harness_version

_LOG = "[generate]"

OK_STOP_REASONS = frozenset({"end_turn", "stop_sequence"})
PROMPT_HASH_SEPARATOR = "\n\x00\n"

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
    """``${cpu_summary}``: machine.yaml 에서 조립하되 고정 코어 집합만 서술한다(설계 §4)."""
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


class _EnvInfo:
    """machine.yaml 에 없는 값(파이썬/NumPy 버전, 컴파일러 버전)을 한 번만 조사한다."""

    def __init__(self, cfg: Config, machine: dict[str, Any]) -> None:
        self.cfg = cfg
        self.machine = machine
        self._identity: dict[str, Any] | None = None
        self._cc_version: str | None = None

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


def load_signature(task: str, language: str, args_mode: str) -> tuple[str, str]:
    """``tasks/<task>/signatures.yaml`` 에서 (시그니처, 인덱싱 문구). 없으면 ConfigError."""
    path = TASKS_DIR / task / "signatures.yaml"
    try:
        with path.open("r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raise ConfigError(f"signatures.yaml 이 없다: {path}") from None
    entry = (spec.get("languages") or {}).get(language)
    if not isinstance(entry, dict):
        raise ConfigError(f"{path}: languages.{language} 항목이 없다")
    signature = entry.get(args_mode)
    if not isinstance(signature, str) or not signature:
        raise ConfigError(f"{path}: languages.{language}.{args_mode} 시그니처가 없다 (M4 전까지 positional 만 지원)")
    return signature, str(entry.get("indexing") or "")


def render_prompt(cfg: Config, machine: dict[str, Any], task: str, lang: LanguageCfg, *, args_mode: str, env: _EnvInfo | None = None) -> RenderedPrompt:
    """system.md + 언어 파일(${task_spec} = spec.md) 렌더링(설계 §4 변수)."""
    env = env or _EnvInfo(cfg, machine)
    system_path = PROMPTS_DIR / "system.md"
    spec_path = TASKS_DIR / task / "spec.md"
    for p in (system_path, lang.prompt, spec_path):
        if not p.is_file():
            raise ConfigError(f"프롬프트/명세 파일이 없다: {p}")
    system = system_path.read_text(encoding="utf-8").rstrip("\n") + "\n"
    template = lang.prompt.read_text(encoding="utf-8")
    task_spec = spec_path.read_text(encoding="utf-8").rstrip("\n")
    signature, indexing = load_signature(task, lang.name, args_mode)
    variables = {
        "signature": signature,
        "indexing": indexing,
        "python_version": env.python_version(),
        "numpy_version": env.numpy_version(),
        "cc": cfg.build.cc,
        "cc_version": env.cc_version(),
        "build_command": build_command(cfg.build, openmp=lang.openmp) if lang.backend == "c" else "",
        "threads": str(lang.threads),
        "cores_desc": cores_desc(cfg, machine, lang.threads),
        "cpu_summary": cpu_summary(cfg, machine, lang.threads),
        "task_spec": task_spec,
    }
    try:
        user = Template(template).substitute(variables)
    except (KeyError, ValueError) as e:
        raise ConfigError(f"{lang.prompt}: 템플릿 변수 오류 {e}") from None
    user = user.rstrip("\n") + "\n"
    return RenderedPrompt(system=system, user=user, prompt_hash=prompt_hash(system, user), signature=signature)


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


def build_raw(cfg: Config, run_id: str, task: str, lang: LanguageCfg, sample_idx: int, prompt: RenderedPrompt, resp: LLMResponse,
              snapshot: dict[str, Any], version: str) -> dict[str, Any]:
    """raw JSON (설계 §5 키 순서 그대로)."""
    m = cfg.model
    cond = cfg.run.conditions
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
        "turn": 0,
        "conditions": {"args": cond.args, "grammar_hint": cond.grammar_hint, "feedback": cond.feedback},
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
        "prompt": {"system": prompt.system, "user": prompt.user},
        "prompt_hash": prompt.prompt_hash,
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


def _confirm(n_calls: int, cfg: Config, *, yes: bool) -> bool:
    """계획을 출력하고 확인을 받는다. 비대화형이고 --yes 가 없으면 False."""
    m = cfg.model
    effort = m.anthropic.effort if m.provider == "anthropic" else None
    print(f"{_LOG} {n_calls} calls to {m.name} via {m.provider}, max_tokens={m.max_tokens}, effort={effort}, "
          f"temperature={'provider_default' if m.temperature is None else m.temperature}", file=sys.stderr)
    if yes:
        return True
    if not sys.stdin.isatty():
        print(f"{_LOG} 비대화형 실행이다. 실제 API 호출에는 --yes 가 필요하다 (exit 2)", file=sys.stderr)
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
        cfg = load_config(ns.config)
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
    run_id = ns.run_id or default_run_id(provider)
    cond = cfg.run.conditions
    tag = cond_tag(cond.args, cond.grammar_hint, cond.feedback)
    raw_root = cfg.paths.raw_dir / run_id

    # 계획
    items: list[WorkItem] = []
    for task in tasks:
        for language in languages:
            for i in range(n):
                path = raw_sample_path(cfg.paths.raw_dir, run_id, task, language, tag, i)
                action, prev = decide_action(path, force=ns.force, no_retry_failed=ns.no_retry_failed)
                items.append(WorkItem(task, language, i, path, action, prev))
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
        for task in tasks:
            for language in languages:
                prompts[(task, language)] = render_prompt(cfg, machine, task, cfg.languages[language], args_mode=cond.args, env=env)
        client = LLMClient(cfg)
    except (ConfigError, LLMClientError) as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return 1

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
            print(summary.line(run_id, raw_root))
            return 0
        if not _confirm(n_calls, cfg, yes=ns.yes):
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
            raw = build_raw(cfg, run_id, it.task, lang, it.sample_idx, prompt, resp, snapshot, version)
            archived = archive_previous(it.path)
            atomic_write_json(it.path, raw)
            summary.calls += 1
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
    print(summary.line(run_id, raw_root))
    return rc


if __name__ == "__main__":
    sys.exit(main())
