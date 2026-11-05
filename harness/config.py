"""실험 설정 로더·검증 (설계 §2).

``config/experiment.yaml`` 을 중첩 frozen dataclass(``Config``)로 읽는다.
순서: 1) YAML → 2) 환경변수 덮어쓰기(``LLM_PROVIDER``→model.provider, ``LLM_MODEL``→model.name)
→ 3) 검증(실패 시 ``ConfigError`` 에 키 경로) → 4) 상대 경로를 ROOT 기준으로 절대화.
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
BACKENDS = ("python", "c")
ARGS_MODES = ("positional", "keyword")
GRAMMAR_HINTS = ("on", "off")
FEEDBACKS = ("compiler", "compiler+lint")
EFFORTS = ("low", "medium", "high", "xhigh", "max")
MAX_TOKENS_PARAMS = ("max_tokens", "max_completion_tokens")

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


@dataclass(frozen=True)
class PathsCfg:
    results_csv: Path
    results_csv_dummy: Path
    raw_dir: Path
    built_dir: Path


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


def _int(obj: dict[str, Any], path: str, key: str, *, min_value: int | None = None) -> int:
    v = obj[key]
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{path}.{key}: 정수여야 한다 (현재 {v!r})")
    if min_value is not None and v < min_value:
        raise ConfigError(f"{path}.{key}: {v} < 최소값 {min_value}")
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


# ----------------------------------------------------------------------------- section parsers


def _parse_run(raw: Any) -> RunCfg:
    p = "run"
    obj = _require_mapping(raw, p)
    _check_keys(obj, p, ("tasks", "languages", "n_samples", "max_repair_turns", "conditions"))
    cond = _require_mapping(obj["conditions"], f"{p}.conditions")
    _check_keys(cond, f"{p}.conditions", ("args", "grammar_hint", "feedback"))
    return RunCfg(
        tasks=_str_list(obj, p, "tasks", nonempty=True),
        languages=_str_list(obj, p, "languages", nonempty=True),
        n_samples=_int(obj, p, "n_samples", min_value=1),
        max_repair_turns=_int(obj, p, "max_repair_turns", min_value=0),
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


def _parse_languages(raw: Any, measurement_threads: int) -> dict[str, LanguageCfg]:
    p = "languages"
    obj = _require_mapping(raw, p)
    if not obj:
        raise ConfigError(f"{p}: 비어 있으면 안 된다")
    out: dict[str, LanguageCfg] = {}
    for name, body in obj.items():
        lp = f"{p}.{name}"
        b = _require_mapping(body, lp)
        _check_keys(b, lp, ("backend", "prompt", "threads"), optional=("openmp",))
        threads_raw = b["threads"]
        if threads_raw == "measurement":
            threads = measurement_threads
        elif isinstance(threads_raw, int) and not isinstance(threads_raw, bool) and threads_raw >= 1:
            threads = threads_raw
        else:
            raise ConfigError(f"{lp}.threads: 'measurement' 또는 양의 정수여야 한다 (현재 {threads_raw!r})")
        openmp = _bool(b, lp, "openmp") if "openmp" in b else False
        out[name] = LanguageCfg(
            name=str(name),
            backend=_str(b, lp, "backend", allowed=BACKENDS),
            prompt=resolve(_str(b, lp, "prompt")),
            threads=threads,
            openmp=openmp,
        )
    return out


def _parse_build(raw: Any) -> BuildCfg:
    p = "build"
    obj = _require_mapping(raw, p)
    _check_keys(
        obj,
        p,
        ("cc", "cflags", "base_flags", "openmp_flag", "ldflags", "diag_flags", "compile_timeout_s", "forbidden_symbols", "thread_symbols_prefix"),
    )
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
    )
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


# ----------------------------------------------------------------------------- public API


def load_config(path: Path | None = None) -> Config:
    """``experiment.yaml`` 을 읽어 검증된 ``Config`` 를 돌려준다. 실패하면 ``ConfigError``."""
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
    _check_keys(top, "(top)", ("run", "model", "languages", "build", "measurement", "paths"))

    env_overrides: dict[str, str] = {}
    run = _parse_run(top["run"])
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
    sys.exit(main())
