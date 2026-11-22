"""실험 설정 로더·검증 (설계 §2, v3 §8).

``config/experiment.yaml`` 을 중첩 frozen dataclass(``Config``)로 읽는다.
순서: 1) YAML → 2) 덮어쓰기: 환경변수(``LLM_PROVIDER``→model.provider, ``LLM_MODEL``→model.name)와 호출자가 준 조건
(``overrides``: ``run.conditions`` 의 args·grammar_hint·feedback) → 3) 검증(실패 시 ``ConfigError`` 에 키 경로)
→ 4) 상대 경로를 ROOT 기준으로 절대화.

v3(2026-11-20)에서 더한 것은 모두 선택 키다(없으면 dataclass 의 기본값 = 설계 v3 §8 의 값; 예전 설정 파일도 그대로 읽힌다):
``languages.<언어>.env``·``halide_mode``, ``build.rust``·``build.sanitize_flags``, 최상위 ``halide``·``safety``·``lint``·
``perf_summary``. 새 백엔드의 언어에만 필수인 것이 둘 있다: halide 는 ``halide_mode``, numba·halide·rust 는 ``env`` 의 스레드
변수(``THREAD_ENV``)가 그 언어의 스레드 수와 같아야 한다.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from harness.paths import DEFAULT_EXPERIMENT_YAML, DEFAULT_MACHINE_YAML, resolve

_LOG = "[config]"

PROVIDERS = ("anthropic", "openai_compat", "dummy")
BACKENDS = ("python", "c", "numba", "halide", "rust")
HALIDE_MODES = ("auto", "manual")
ARGS_MODES = ("positional", "keyword")
GRAMMAR_HINTS = ("on", "off")
FEEDBACKS = ("compiler", "compiler+lint")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens")

# 수정 루프의 설계 상한(CLAUDE.md §1: 최대 3턴; ``turns_to_pass`` 는 0~3)
MAX_REPAIR_TURNS = 3
# ``load_config(overrides=…)`` 로 덮어쓸 수 있는 ``run.conditions`` 의 키
CONDITION_KEYS = ("args", "grammar_hint", "feedback")
# ``languages.<언어>.env`` 값에서 그 언어의 스레드 수로 치환되는 자리표시
THREADS_PLACEHOLDER = "${threads}"
# 백엔드의 런타임이 스레드 수를 읽는 환경변수(``OMP_NUM_THREADS`` 가 아니다). 언어의 ``env`` 가 이 변수를 그 언어의 스레드 수로
# 주지 않으면 측정 스레드 수가 설정대로 고정되지 않고, 프롬프트의 "Thread count" 와 행의 ``threads`` 열이 사실이 아니게 된다
# → 로더가 거부한다.
THREAD_ENV: dict[str, str] = {"numba": "NUMBA_NUM_THREADS", "halide": "HL_NUM_THREADS", "rust": "RAYON_NUM_THREADS"}
# 하네스가 스레드 수로 넣는 오토스케줄러 매개변수(설정에서 따로 주면 프롬프트의 서술과 어긋난다)
HALIDE_RESERVED_PARAMS = ("parallelism",)

# taskset -c 형식: "0,2,4,6", "0-3", "0-3,8" (공백 없음)
_CPULIST_RE = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")


class ConfigError(ValueError):
    """설정 검증 실패. 메시지는 ``<키.경로>: <이유>`` 형식."""


# ----------------------------------------------------------------------------- dataclasses


@dataclass(frozen=True)
class ConditionsCfg:
    args: str
    grammar_hint: str
    feedback: str


@dataclass(frozen=True)
class RunCfg:
    tasks: list[str]
    languages: list[str]
    n_samples: int
    max_repair_turns: int
    conditions: ConditionsCfg


@dataclass(frozen=True)
class AnthropicCfg:
    api_key_env: str
    effort: str | None
    thinking: str


@dataclass(frozen=True)
class OpenAICompatCfg:
    api_key_env: str
    base_url_env: str
    max_tokens_param: str


@dataclass(frozen=True)
class ModelCfg:
    provider: str
    name: str
    max_tokens: int
    temperature: float | None
    seed: int | None
    request_timeout_s: float
    max_retries: int
    anthropic: AnthropicCfg
    openai_compat: OpenAICompatCfg


@dataclass(frozen=True)
class LanguageCfg:
    name: str
    backend: str
    prompt: Path
    threads: int  # "measurement" 는 로더가 measurement.threads 정수로 치환한다
    openmp: bool = False
    env: dict[str, str] = field(default_factory=dict)  # 워커에 더하는 환경변수(값의 ``${threads}`` 는 로더가 치환했다)
    halide_mode: str | None = None  # halide 백엔드만: auto | manual


@dataclass(frozen=True)
class RustCfg:
    cargo: str = "cargo"
    rustflags: list[str] = field(default_factory=lambda: ["-C", "target-cpu=native"])
    build_timeout_s: float = 600.0
    offline: bool = True


@dataclass(frozen=True)
class BuildCfg:
    cc: str
    cflags: list[str]
    base_flags: list[str]
    openmp_flag: str
    ldflags: list[str]
    diag_flags: list[str]
    compile_timeout_s: float
    forbidden_symbols: list[str]
    thread_symbols_prefix: list[str]
    rust: RustCfg = field(default_factory=RustCfg)
    sanitize_flags: list[str] = field(default_factory=lambda: ["-fsanitize=address,undefined", "-fno-omit-frame-pointer", "-g"])


@dataclass(frozen=True)
class MeasurementCfg:
    cores: str
    single_core: str
    threads: int
    omp_env_extra: dict[str, str]
    warmup: int
    repeats: int
    min_repeats: int
    check_repeats: int
    check_timeout_s: float
    bench_timeout_s: float
    max_worker_mem_gb: float
    max_load_1m: float
    max_pinned_busy_pct: float
    require_performance_governor: bool
    baseline_min_gflops: float
    baseline_drift_pct_max: float
    correctness_tests: list[str]
    performance_test: str
    # 선택: 동적 규약 검사 전에 1분 부하가 max_load_1m 아래로 내려가길 기다리는 상한(초). 0 = 기다리지 않는다(기존 절차).
    # 측정 절차를 바꾸므로 사용자 확인 전에는 0 이다(기본값은 protocol.OPTION_DEFAULTS 와 같다).
    max_settle_s: float = 0.0


@dataclass(frozen=True)
class PathsCfg:
    results_csv: Path
    results_csv_dummy: Path
    raw_dir: Path
    built_dir: Path


@dataclass(frozen=True)
class HalideCfg:
    autoscheduler: str = "Mullapudi2016"
    autoscheduler_params: dict[str, str] = field(default_factory=dict)  # ``parallelism`` 은 하네스가 스레드 수로 넣는다


@dataclass(frozen=True)
class SafetyCfg:
    """ASan/UBSan 재빌드·실행(C 백엔드만). 실험 때 켠다."""

    enabled: bool = False
    timeout_s: float = 300.0


@dataclass(frozen=True)
class LintCfg:
    enabled: bool = True
    plugin: Path = field(default_factory=lambda: resolve("plugins/lint/run_lint.py"))
    timeout_s: float = 60.0


@dataclass(frozen=True)
class PerfSummaryCfg:
    enabled: bool = True


@dataclass(frozen=True)
class Config:
    run: RunCfg
    model: ModelCfg
    languages: dict[str, LanguageCfg]
    build: BuildCfg
    measurement: MeasurementCfg
    paths: PathsCfg
    source_path: Path = field(default=DEFAULT_EXPERIMENT_YAML)
    env_overrides: dict[str, str] = field(default_factory=dict)  # 적용된 환경변수 덮어쓰기 기록
    halide: HalideCfg = field(default_factory=HalideCfg)
    safety: SafetyCfg = field(default_factory=SafetyCfg)
    lint: LintCfg = field(default_factory=LintCfg)
    perf_summary: PerfSummaryCfg = field(default_factory=PerfSummaryCfg)


# ----------------------------------------------------------------------------- helpers


def parse_cpulist(spec: str) -> list[int]:
    """``taskset -c`` 형식("0,2,4,6", "0-3,8")을 정렬된 CPU 번호 리스트로 바꾼다."""
    spec = str(spec).strip()
    if not _CPULIST_RE.match(spec):
        raise ValueError(f"CPU 목록 형식이 아니다: {spec!r} (예: '0,2,4,6' 또는 '0-3')")
    cpus: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-"))
            if hi < lo:
                raise ValueError(f"CPU 범위가 거꾸로다: {part!r}")
            cpus.update(range(lo, hi + 1))
        else:
            cpus.add(int(part))
    return sorted(cpus)


def _require_mapping(obj: Any, path: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ConfigError(f"{path}: 매핑이어야 한다 (현재 {type(obj).__name__})")
    return obj


def _check_keys(obj: dict[str, Any], path: str, required: tuple[str, ...], optional: tuple[str, ...] = ()) -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ConfigError(f"{path}: 필수 키 누락 {missing}")
    unknown = [k for k in obj if k not in required and k not in optional]
    if unknown:
        raise ConfigError(f"{path}: 알 수 없는 키 {unknown}")


def _str(obj: dict[str, Any], path: str, key: str, *, allowed: tuple[str, ...] | None = None, nonempty: bool = True) -> str:
    v = obj[key]
    if not isinstance(v, str):
        raise ConfigError(f"{path}.{key}: 문자열이어야 한다 (현재 {v!r})")
    if nonempty and not v:
        raise ConfigError(f"{path}.{key}: 빈 문자열은 허용되지 않는다")
    if allowed is not None and v not in allowed:
        raise ConfigError(f"{path}.{key}: {v!r} 는 허용값 {list(allowed)} 에 없다")
    return v


def _int(obj: dict[str, Any], path: str, key: str, *, min_value: int | None = None, max_value: int | None = None) -> int:
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{path}.{key}: 정수여야 한다 (현재 {v!r})")
    if min_value is not None and v < min_value:
        raise ConfigError(f"{path}.{key}: {v} < 최소값 {min_value}")
    if max_value is not None and v > max_value:
        raise ConfigError(f"{path}.{key}: {v} > 최대값 {max_value}")
    return v


def _num(obj: dict[str, Any], path: str, key: str, *, min_value: float | None = None, exclusive: bool = False) -> float:
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError(f"{path}.{key}: 숫자여야 한다 (현재 {v!r})")
    if min_value is not None and (v <= min_value if exclusive else v < min_value):
        op = "<=" if exclusive else "<"
        raise ConfigError(f"{path}.{key}: {v} {op} {min_value}")
    return float(v)


def _bool(obj: dict[str, Any], path: str, key: str) -> bool:
    v = obj[key]
    if not isinstance(v, bool):
        raise ConfigError(f"{path}.{key}: 불리언이어야 한다 (현재 {v!r})")
    return v


def _str_list(obj: dict[str, Any], path: str, key: str, *, nonempty: bool = False) -> list[str]:
    v = obj[key]
    if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
        raise ConfigError(f"{path}.{key}: 문자열 리스트여야 한다 (현재 {v!r})")
    if nonempty and not v:
        raise ConfigError(f"{path}.{key}: 비어 있으면 안 된다")
    return list(v)


def _str_map(raw: Any, path: str) -> dict[str, str]:
    """문자열 → 스칼라 매핑을 문자열 → 문자열로 읽는다(환경변수·오토스케줄러 매개변수).

    YAML 에서 따옴표를 빼먹어 불리언이 된 값은 ``"true"``/``"false"`` 로 정규화한다(``omp_env_extra`` 와 같은 규칙).
    """
    obj = _require_mapping(raw, path)
    out: dict[str, str] = {}
    for k, v in obj.items():
        if not isinstance(k, str) or not k:
            raise ConfigError(f"{path}: 키는 비어 있지 않은 문자열이어야 한다 ({k!r})")
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif not isinstance(v, (str, int, float)):
            raise ConfigError(f"{path}.{k}: 문자열 또는 숫자여야 한다 (현재 {v!r})")
        out[k] = str(v)
    return out


# ----------------------------------------------------------------------------- section parsers


def _parse_run(raw: Any, overrides: dict[str, str | None] | None) -> RunCfg:
    p = "run"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, ("tasks", "languages", "n_samples", "max_repair_turns", "conditions"))
    cond = dict(_require_mapping(obj["conditions"], f"{p}.conditions"))
    _check_keys(cond, f"{p}.conditions", CONDITION_KEYS)
    for key, value in (overrides or {}).items():
        if key not in CONDITION_KEYS:
            raise ConfigError(f"overrides: 알 수 없는 조건 {key!r} (허용: {list(CONDITION_KEYS)})")
        if value is not None:
            cond[key] = value  # 아래의 허용값 검사를 그대로 거친다
    return RunCfg(
        tasks=_str_list(obj, p, "tasks", nonempty=True),
        languages=_str_list(obj, p, "languages", nonempty=True),
        n_samples=_int(obj, p, "n_samples", min_value=1),
        max_repair_turns=_int(obj, p, "max_repair_turns", min_value=0, max_value=MAX_REPAIR_TURNS),
        conditions=ConditionsCfg(
            args=_str(cond, f"{p}.conditions", "args", allowed=ARGS_MODES),
            grammar_hint=_str(cond, f"{p}.conditions", "grammar_hint", allowed=GRAMMAR_HINTS),
            feedback=_str(cond, f"{p}.conditions", "feedback", allowed=FEEDBACKS),
        ),
    )


def _parse_model(raw: Any, env_overrides: dict[str, str]) -> ModelCfg:
    p = "model"
    obj = dict(_require_mapping(raw, p))
    _check_keys(
        obj,
        p,
        ("provider", "name", "max_tokens", "temperature", "seed", "request_timeout_s", "max_retries", "anthropic", "openai_compat"),
    )
    # 환경변수 덮어쓰기 (빈 문자열은 무시)
    env_provider = os.environ.get("LLM_PROVIDER", "").strip()
    env_model = os.environ.get("LLM_MODEL", "").strip()
    if env_provider:
        obj["provider"] = env_provider
        env_overrides["LLM_PROVIDER"] = env_provider
    if env_model:
        obj["name"] = env_model
        env_overrides["LLM_MODEL"] = env_model

    provider = _str(obj, p, "provider", allowed=PROVIDERS)
    name = _str(obj, p, "name")
    max_tokens = _int(obj, p, "max_tokens", min_value=1)

    temperature: float | None
    if obj["temperature"] is None:
        temperature = None
    else:
        temperature = _num(obj, p, "temperature", min_value=0.0)
        print(f"{_LOG} 경고: model.temperature={temperature} 가 설정됐다. Claude 4.7+/5.x는 temperature를 400으로 거부한다.", file=sys.stderr)

    seed: int | None
    if obj["seed"] is None:
        seed = None
    else:
        seed = _int(obj, p, "seed")

    a = _require_mapping(obj["anthropic"], f"{p}.anthropic")
    _check_keys(a, f"{p}.anthropic", ("api_key_env", "effort", "thinking"))
    effort = None if a["effort"] is None else _str(a, f"{p}.anthropic", "effort", allowed=EFFORTS)
    anthropic_cfg = AnthropicCfg(
        api_key_env=_str(a, f"{p}.anthropic", "api_key_env"),
        effort=effort,
        thinking=_str(a, f"{p}.anthropic", "thinking"),
    )

    o = _require_mapping(obj["openai_compat"], f"{p}.openai_compat")
    _check_keys(o, f"{p}.openai_compat", ("api_key_env", "base_url_env", "max_tokens_param"))
    openai_cfg = OpenAICompatCfg(
        api_key_env=_str(o, f"{p}.openai_compat", "api_key_env"),
        base_url_env=_str(o, f"{p}.openai_compat", "base_url_env"),
        max_tokens_param=_str(o, f"{p}.openai_compat", "max_tokens_param", allowed=MAX_TOKENS_PARAMS),
    )

    if provider == "anthropic":
        if seed is not None:
            raise ConfigError(f"{p}.seed: anthropic 제공자에는 seed가 없다 (null이어야 함)")
        if anthropic_cfg.effort is None:
            raise ConfigError(f"{p}.anthropic.effort: anthropic 제공자에서는 null을 허용하지 않는다 ({list(EFFORTS)} 중 하나)")
    if provider == "openai_compat" and openai_cfg.api_key_env.startswith("ANTHROPIC_"):
        raise ConfigError(f"{p}.openai_compat.api_key_env: ANTHROPIC_ 접두 변수는 거부한다 (키 유출 방지): {openai_cfg.api_key_env!r}")

    return ModelCfg(
        provider=provider,
        name=name,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
        request_timeout_s=_num(obj, p, "request_timeout_s", min_value=0.0, exclusive=True),
        max_retries=_int(obj, p, "max_retries", min_value=0),
        anthropic=anthropic_cfg,
        openai_compat=openai_cfg,
    )


def _parse_language_env(raw: Any, path: str, threads: int) -> dict[str, str]:
    """``languages.<언어>.env``: 값의 ``${threads}`` 를 그 언어의 스레드 수로 치환한다. 그 밖의 ``${…}`` 는 오타로 보고 거부한다
    (치환되지 않은 자리표시가 워커 환경에 그대로 들어가면 런타임이 조용히 기본 스레드 수로 돈다)."""
    out: dict[str, str] = {}
    for key, value in _str_map(raw, path).items():
        value = value.replace(THREADS_PLACEHOLDER, str(threads))
        if "${" in value:
            raise ConfigError(f"{path}.{key}: 알 수 없는 자리표시 {value!r} (쓸 수 있는 것: {THREADS_PLACEHOLDER})")
        out[key] = value
    return out


def _parse_languages(raw: Any, measurement_threads: int) -> dict[str, LanguageCfg]:
    p = "languages"
    obj = _require_mapping(raw, p)
    if not obj:
        raise ConfigError(f"{p}: 비어 있으면 안 된다")
    out: dict[str, LanguageCfg] = {}
    for name, body in obj.items():
        lp = f"{p}.{name}"
        b = _require_mapping(body, lp)
        _check_keys(b, lp, ("backend", "prompt", "threads"), optional=("openmp", "env", "halide_mode"))
        threads_raw = b["threads"]
        if threads_raw == "measurement":
            threads = measurement_threads
        elif isinstance(threads_raw, int) and not isinstance(threads_raw, bool) and threads_raw >= 1:
            threads = threads_raw
        else:
            raise ConfigError(f"{lp}.threads: 'measurement' 또는 양의 정수여야 한다 (현재 {threads_raw!r})")
        openmp = _bool(b, lp, "openmp") if "openmp" in b else False
        backend = _str(b, lp, "backend", allowed=BACKENDS)
        halide_mode: str | None = None
        if backend == "halide":
            if "halide_mode" not in b:
                raise ConfigError(f"{lp}.halide_mode: halide 백엔드에는 필수다 ({list(HALIDE_MODES)} 중 하나)")
            halide_mode = _str(b, lp, "halide_mode", allowed=HALIDE_MODES)
        elif "halide_mode" in b:
            raise ConfigError(f"{lp}.halide_mode: halide 백엔드에서만 쓴다 (backend={backend!r})")
        env = _parse_language_env(b["env"], f"{lp}.env", threads) if "env" in b else {}
        thread_env = THREAD_ENV.get(backend)
        if thread_env is not None and env.get(thread_env) != str(threads):
            raise ConfigError(f'{lp}.env.{thread_env}: {backend} 백엔드는 이 변수가 그 언어의 스레드 수({threads})여야 한다 '
                              f'(현재 {env.get(thread_env)!r}; "{THREADS_PLACEHOLDER}" 로 쓴다)')
        out[name] = LanguageCfg(
            name=str(name),
            backend=backend,
            prompt=resolve(_str(b, lp, "prompt")),
            threads=threads,
            openmp=openmp,
            env=env,
            halide_mode=halide_mode,
        )
    return out


def _parse_rust(raw: Any) -> RustCfg:
    p = "build.rust"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, (), optional=("cargo", "rustflags", "build_timeout_s", "offline"))
    kwargs: dict[str, Any] = {}
    if "cargo" in obj:
        kwargs["cargo"] = _str(obj, p, "cargo")
    if "rustflags" in obj:
        kwargs["rustflags"] = _str_list(obj, p, "rustflags")
    if "build_timeout_s" in obj:
        kwargs["build_timeout_s"] = _num(obj, p, "build_timeout_s", min_value=0.0, exclusive=True)
    if "offline" in obj:
        kwargs["offline"] = _bool(obj, p, "offline")
    return RustCfg(**kwargs)


def _parse_build(raw: Any) -> BuildCfg:
    p = "build"
    obj = _require_mapping(raw, p)
    _check_keys(
        obj,
        p,
        ("cc", "cflags", "base_flags", "openmp_flag", "ldflags", "diag_flags", "compile_timeout_s", "forbidden_symbols", "thread_symbols_prefix"),
        optional=("rust", "sanitize_flags"),
    )
    kwargs: dict[str, Any] = {}
    if "rust" in obj:
        kwargs["rust"] = _parse_rust(obj["rust"])
    if "sanitize_flags" in obj:
        kwargs["sanitize_flags"] = _str_list(obj, p, "sanitize_flags", nonempty=True)
    return BuildCfg(
        cc=_str(obj, p, "cc"),
        cflags=_str_list(obj, p, "cflags"),
        base_flags=_str_list(obj, p, "base_flags"),
        openmp_flag=_str(obj, p, "openmp_flag"),
        ldflags=_str_list(obj, p, "ldflags"),
        diag_flags=_str_list(obj, p, "diag_flags"),
        compile_timeout_s=_num(obj, p, "compile_timeout_s", min_value=0.0, exclusive=True),
        forbidden_symbols=_str_list(obj, p, "forbidden_symbols"),
        thread_symbols_prefix=_str_list(obj, p, "thread_symbols_prefix"),
        **kwargs,
    )


def _parse_measurement(raw: Any) -> MeasurementCfg:
    p = "measurement"
    obj = _require_mapping(raw, p)
    _check_keys(
        obj,
        p,
        (
            "cores", "single_core", "threads", "omp_env_extra", "warmup", "repeats", "min_repeats", "check_repeats",
            "check_timeout_s", "bench_timeout_s", "max_worker_mem_gb", "max_load_1m", "max_pinned_busy_pct",
            "require_performance_governor", "baseline_min_gflops", "baseline_drift_pct_max", "correctness_tests",
            "performance_test",
        ),
        optional=("max_settle_s",),
    )
    kwargs: dict[str, Any] = {}
    if "max_settle_s" in obj:
        kwargs["max_settle_s"] = _num(obj, p, "max_settle_s", min_value=0.0)
    cores = str(obj["cores"]).strip()
    single_core = str(obj["single_core"]).strip()
    try:
        core_list = parse_cpulist(cores)
    except ValueError as e:
        raise ConfigError(f"{p}.cores: {e}") from None
    try:
        single_list = parse_cpulist(single_core)
    except ValueError as e:
        raise ConfigError(f"{p}.single_core: {e}") from None
    if len(single_list) != 1:
        raise ConfigError(f"{p}.single_core: CPU 하나여야 한다 (현재 {single_core!r})")
    threads = _int(obj, p, "threads", min_value=1)
    if threads > len(core_list):
        raise ConfigError(f"{p}.threads: {threads} 가 고정 코어 수 {len(core_list)} 보다 크다")
    omp_raw = _require_mapping(obj["omp_env_extra"], f"{p}.omp_env_extra")
    omp_env_extra: dict[str, str] = {}
    for k, v in omp_raw.items():
        if not isinstance(k, str) or not k:
            raise ConfigError(f"{p}.omp_env_extra: 키는 비어 있지 않은 문자열이어야 한다 ({k!r})")
        if isinstance(v, bool):
            # YAML에서 따옴표를 빼먹어 불리언이 된 경우 — 환경변수 문자열로 정규화한다
            v = "true" if v else "false"
        omp_env_extra[k] = str(v)
    repeats = _int(obj, p, "repeats", min_value=1)
    min_repeats = _int(obj, p, "min_repeats", min_value=1)
    if min_repeats > repeats:
        raise ConfigError(f"{p}.min_repeats: {min_repeats} > repeats {repeats}")
    return MeasurementCfg(
        cores=cores,
        single_core=single_core,
        threads=threads,
        omp_env_extra=omp_env_extra,
        warmup=_int(obj, p, "warmup", min_value=0),
        repeats=repeats,
        min_repeats=min_repeats,
        check_repeats=_int(obj, p, "check_repeats", min_value=1),
        check_timeout_s=_num(obj, p, "check_timeout_s", min_value=0.0, exclusive=True),
        bench_timeout_s=_num(obj, p, "bench_timeout_s", min_value=0.0, exclusive=True),
        max_worker_mem_gb=_num(obj, p, "max_worker_mem_gb", min_value=0.0, exclusive=True),
        max_load_1m=_num(obj, p, "max_load_1m", min_value=0.0),
        max_pinned_busy_pct=_num(obj, p, "max_pinned_busy_pct", min_value=0.0),
        require_performance_governor=_bool(obj, p, "require_performance_governor"),
        baseline_min_gflops=_num(obj, p, "baseline_min_gflops", min_value=0.0),
        baseline_drift_pct_max=_num(obj, p, "baseline_drift_pct_max", min_value=0.0),
        correctness_tests=_str_list(obj, p, "correctness_tests", nonempty=True),
        performance_test=_str(obj, p, "performance_test"),
        **kwargs,
    )


def _parse_paths(raw: Any) -> PathsCfg:
    p = "paths"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, ("results_csv", "results_csv_dummy", "raw_dir", "built_dir"))
    return PathsCfg(
        results_csv=resolve(_str(obj, p, "results_csv")),
        results_csv_dummy=resolve(_str(obj, p, "results_csv_dummy")),
        raw_dir=resolve(_str(obj, p, "raw_dir")),
        built_dir=resolve(_str(obj, p, "built_dir")),
    )


def _parse_halide(raw: Any) -> HalideCfg:
    p = "halide"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, (), optional=("autoscheduler", "autoscheduler_params"))
    kwargs: dict[str, Any] = {}
    if "autoscheduler" in obj:
        kwargs["autoscheduler"] = _str(obj, p, "autoscheduler")
    if "autoscheduler_params" in obj:
        params = _str_map(obj["autoscheduler_params"], f"{p}.autoscheduler_params")
        reserved = [k for k in params if k in HALIDE_RESERVED_PARAMS]
        if reserved:
            raise ConfigError(f"{p}.autoscheduler_params: {reserved} 는 하네스가 그 언어의 스레드 수로 넣는다 (설정에서 줄 수 없다)")
        kwargs["autoscheduler_params"] = params
    return HalideCfg(**kwargs)


def _parse_safety(raw: Any) -> SafetyCfg:
    p = "safety"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, (), optional=("enabled", "timeout_s"))
    kwargs: dict[str, Any] = {}
    if "enabled" in obj:
        kwargs["enabled"] = _bool(obj, p, "enabled")
    if "timeout_s" in obj:
        kwargs["timeout_s"] = _num(obj, p, "timeout_s", min_value=0.0, exclusive=True)
    return SafetyCfg(**kwargs)


def _parse_lint(raw: Any) -> LintCfg:
    p = "lint"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, (), optional=("enabled", "plugin", "timeout_s"))
    kwargs: dict[str, Any] = {}
    if "enabled" in obj:
        kwargs["enabled"] = _bool(obj, p, "enabled")
    if "plugin" in obj:
        kwargs["plugin"] = resolve(_str(obj, p, "plugin"))
    if "timeout_s" in obj:
        kwargs["timeout_s"] = _num(obj, p, "timeout_s", min_value=0.0, exclusive=True)
    return LintCfg(**kwargs)


def _parse_perf_summary(raw: Any) -> PerfSummaryCfg:
    p = "perf_summary"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, (), optional=("enabled",))
    return PerfSummaryCfg(enabled=_bool(obj, p, "enabled")) if "enabled" in obj else PerfSummaryCfg()


# ----------------------------------------------------------------------------- public API


def load_config(path: Path | None = None, *, overrides: dict[str, str | None] | None = None) -> Config:
    """``experiment.yaml`` 을 읽어 검증된 ``Config`` 를 돌려준다. 실패하면 ``ConfigError``.

    ``overrides`` 는 ``run.conditions`` 의 조건 값을 덮어쓴다(CLI 의 ``--args``·``--grammar-hint``·``--feedback``):
    ``{"args": …, "grammar_hint": …, "feedback": …}`` 중 주어진 키만, 값이 None 인 키는 설정 파일의 값을 그대로 둔다.
    덮어쓴 값도 설정 파일의 값과 같은 허용값 검사를 거친다.
    """
    src = Path(path) if path is not None else DEFAULT_EXPERIMENT_YAML
    src = resolve(src)
    try:
        with src.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except FileNotFoundError:
        raise ConfigError(f"(file): 설정 파일이 없다: {src}") from None
    except yaml.YAMLError as e:
        raise ConfigError(f"(file): YAML 구문 오류 {src}: {e}") from None
    top = _require_mapping(raw, "(top)")
    _check_keys(top, "(top)", ("run", "model", "languages", "build", "measurement", "paths"),
                optional=("halide", "safety", "lint", "perf_summary"))

    env_overrides: dict[str, str] = {}
    run = _parse_run(top["run"], overrides)
    model = _parse_model(top["model"], env_overrides)
    measurement = _parse_measurement(top["measurement"])
    languages = _parse_languages(top["languages"], measurement.threads)
    build = _parse_build(top["build"])
    paths = _parse_paths(top["paths"])

    for lang in run.languages:
        if lang not in languages:
            raise ConfigError(f"run.languages: {lang!r} 가 languages 에 정의돼 있지 않다 (정의: {list(languages)})")
    if measurement.performance_test in measurement.correctness_tests:
        raise ConfigError(f"measurement.performance_test: {measurement.performance_test!r} 가 correctness_tests 에도 있다")

    return Config(
        run=run,
        model=model,
        languages=languages,
        build=build,
        measurement=measurement,
        paths=paths,
        source_path=src,
        env_overrides=env_overrides,
        halide=_parse_halide(top["halide"]) if "halide" in top else HalideCfg(),
        safety=_parse_safety(top["safety"]) if "safety" in top else SafetyCfg(),
        lint=_parse_lint(top["lint"]) if "lint" in top else LintCfg(),
        perf_summary=_parse_perf_summary(top["perf_summary"]) if "perf_summary" in top else PerfSummaryCfg(),
    )


def load_machine(path: Path | None = None) -> dict[str, Any]:
    """``config/machine.yaml`` 을 dict로 읽는다. 없으면 ``{}`` 와 함께 stderr 경고."""
    src = resolve(path) if path is not None else DEFAULT_MACHINE_YAML
    if not src.exists():
        print(f"{_LOG} 경고: {src} 가 없다. 먼저 `.venv/bin/python -m harness.protocol --probe --write` 를 실행하라.", file=sys.stderr)
        return {}
    with src.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError(f"(machine): {src} 의 최상위가 매핑이 아니다")
    return data


def _jsonable(obj: Any) -> Any:
    """dataclass/Path를 JSON 가능한 값으로 재귀 변환한다."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def config_snapshot(cfg: Config) -> dict[str, Any]:
    """raw JSON에 넣을 설정 스냅샷(Path는 문자열, 경로는 절대)."""
    snap = _jsonable(cfg)
    assert isinstance(snap, dict)
    return snap


def main(argv: list[str] | None = None) -> int:
    """설정을 읽어 검증하고 스냅샷을 JSON으로 출력한다."""
    parser = argparse.ArgumentParser(prog="python -m harness.config", description="실험 설정 로드·검증")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로 (기본 config/experiment.yaml)")
    parser.add_argument("--machine", action="store_true", help="machine.yaml도 읽어 함께 출력")
    ns = parser.parse_args(argv)
    try:
        cfg = load_config(ns.config)
    except ConfigError as e:
        print(f"{_LOG} 설정 오류: {e}", file=sys.stderr)
        return 2
    out: dict[str, Any] = {"config": config_snapshot(cfg)}
    if ns.machine:
        out["machine"] = load_machine()
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # 스냅샷의 경로·설정 원문에 로캘 코드페이지로 찍을 수 없는 글자가 있을 수 있다 → 스트림을 정리한다(진입점에서만).
    from harness.procutil import setup_stdio

    setup_stdio()
    sys.exit(main())
