"""LLM 호출 클라이언트 (설계 §5). 제공자: anthropic(스트리밍) · openai_compat · dummy.

- ``LLMResponse`` 는 한 번의 호출 결과(텍스트·토큰·지연·중단 사유·오류)를 담는다.
- ``complete()`` 는 어떤 경우에도 예외를 던지지 않고 ``error`` 필드로 보고한다
  (``fatal:<Type>:<status>:<msg>`` 는 generate 가 즉시 중단, ``transient:...`` 는 다음 실행에서 재호출).
  ``preflight()`` 만 예외를 던진다.
- 스트림 도중 SSE ``event: error`` 로 오는 오류(응답은 이미 HTTP 200 으로 시작됨; SDK ``max_retries`` 미적용)는 SDK 가
  평범한 ``APIStatusError(status_code=200)`` 로 던지므로 HTTP 상태가 아니라 body 의 ``error.type`` 으로 분류한다
  (``overloaded_error``/``api_error``/``rate_limit_error``/``timeout_error`` → transient, 인증·권한·요청 오류 → fatal).
- SDK 는 지연 임포트한다(anthropic 2.x: ``messages.stream`` 에 ``temperature`` 키워드가 없어 ``extra_body`` 로만 보낸다;
  ``request_id`` 는 ``with client.messages.stream(...) as stream`` 의 stream 객체(MessageStream)에 있고,
  ``usage.output_tokens_details.thinking_tokens`` · ``msg.stop_details.category`` 는 설치된 2.4.0 소스로 확인했다).
- 개발 중 네트워크 호출 금지: ``--info`` 는 클라이언트만 만들고 요청을 보내지 않는다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from harness.config import Config, ConfigError, load_config
from harness.paths import ROOT

_LOG = "[llm_client]"

DUMMY_ROOT: Path = ROOT / "harness" / "dummy_responses"
ANTHROPIC_DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"
_ERROR_MSG_CHARS = 500

# openai_compat finish_reason 정규화(설계 §5). 그 외 값은 원문 유지.
_OPENAI_STOP_MAP = {"stop": "end_turn", "length": "max_tokens", "content_filter": "refusal"}

# 오류 body 의 ``error.type`` 기준 분류(anthropic ``types/shared/error_type.py`` 의 ErrorType). HTTP 상태가 4xx/5xx 가 아닌
# APIStatusError — 스트림 도중 SSE ``event: error`` (응답 상태 200) — 에만 쓴다. 아래 타입은 fatal, 그 외(``overloaded_error``,
# ``api_error``, ``rate_limit_error``, ``timeout_error``, 모르는 타입)는 서버 쪽 중단으로 보고 transient.
_FATAL_ERROR_TYPES = frozenset({"invalid_request_error", "authentication_error", "permission_error", "not_found_error", "billing_error"})


class LLMClientError(RuntimeError):
    """클라이언트 구성 오류(키 환경변수 누락 등). 호출 오류는 ``LLMResponse.error`` 로 보고한다."""


@dataclass
class LLMResponse:
    """한 번의 LLM 호출 결과."""

    text: str
    model_requested: str
    model_served: str
    provider: str
    tokens_in: int
    tokens_out: int
    tokens_thinking: int | None
    tokens_text: int | None
    latency_ms: float
    stop_reason: str
    raw_stop_reason: str
    stop_details_category: str | None
    n_thinking_blocks: int
    n_text_blocks: int
    request_id: str | None
    raw_usage: dict[str, Any]
    base_url: str | None
    sdk_version: str | None
    api_key_present: bool
    error: str | None  # "fatal:<Type>:<status>:<msg>" | "transient:<Type>:<status>:<msg>" | None


def _one_line(text: str, limit: int = _ERROR_MSG_CHARS) -> str:
    """오류 메시지를 한 줄로 줄인다."""
    return " ".join(str(text).split())[:limit]


def _error_body_type(e: BaseException) -> str | None:
    """오류 body 의 ``error.type`` (anthropic ``APIStatusError.type``; 없으면 body dict 에서 직접 찾는다). 모르면 None."""
    etype = getattr(e, "type", None)
    if isinstance(etype, str) and etype:
        return etype
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict) and isinstance(err.get("type"), str) and err["type"]:
            return str(err["type"])
    return None


def _model_dump(obj: Any) -> dict[str, Any]:
    """pydantic 객체를 JSON 가능한 dict 로(실패하면 {})."""
    if obj is None:
        return {}
    try:
        dump = getattr(obj, "model_dump", None)
        if callable(dump):
            data = dump(mode="json")
            return data if isinstance(data, dict) else {}
    except Exception as e:  # 기록용 데이터라 실패해도 호출 결과는 살린다
        print(f"{_LOG} 경고: usage 직렬화 실패: {e}", file=sys.stderr)
    return {}


class LLMClient:
    """설정의 provider 에 따라 anthropic / openai_compat / dummy 로 동작한다."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.provider: str = cfg.model.provider
        self.model: str = cfg.model.name
        self.base_url: str | None = None
        self.sdk_version: str | None = None
        self.api_key_present: bool = False
        self._client: Any = None
        self._errors: tuple[tuple[type[BaseException], ...], tuple[type[BaseException], ...], type[BaseException], type[BaseException]] | None = None
        if self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai_compat":
            self._init_openai_compat()
        elif self.provider == "dummy":
            self.base_url = None
        else:
            raise LLMClientError(f"알 수 없는 provider: {self.provider!r}")

    # ------------------------------------------------------------------ 초기화

    def _init_anthropic(self) -> None:
        import anthropic

        m = self.cfg.model
        kwargs: dict[str, Any] = {"timeout": float(m.request_timeout_s), "max_retries": int(m.max_retries)}
        key_env = m.anthropic.api_key_env
        if key_env != ANTHROPIC_DEFAULT_KEY_ENV:
            # SDK 는 ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN/프로필만 찾으므로 다른 이름이면 명시한다
            key = os.environ.get(key_env)
            if not key:
                raise LLMClientError(f"환경변수 {key_env} 가 비어 있다 (model.anthropic.api_key_env)")
            kwargs["api_key"] = key
        client = anthropic.Anthropic(**kwargs)
        self._client = client
        self.api_key_present = bool(
            getattr(client, "api_key", None) or getattr(client, "auth_token", None) or getattr(client, "credentials", None)
        )
        self.base_url = str(client.base_url)
        self.sdk_version = anthropic.__version__
        self._errors = (
            (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError,
             anthropic.BadRequestError, anthropic.UnprocessableEntityError),
            (anthropic.RateLimitError, anthropic.InternalServerError),
            anthropic.APIStatusError,
            anthropic.APIConnectionError,  # APITimeoutError 포함
        )
        if not self.api_key_present:
            print(f"{_LOG} 경고: Anthropic 자격증명을 찾지 못했다 ({ANTHROPIC_DEFAULT_KEY_ENV}/ANTHROPIC_AUTH_TOKEN/프로필). preflight 가 실패할 것이다.", file=sys.stderr)

    def _init_openai_compat(self) -> None:
        import openai

        m = self.cfg.model
        base_url = os.environ.get(m.openai_compat.base_url_env)
        if not base_url:
            raise LLMClientError(f"환경변수 {m.openai_compat.base_url_env} 가 비어 있다 (model.openai_compat.base_url_env)")
        key = os.environ.get(m.openai_compat.api_key_env)
        self.api_key_present = bool(key)
        client = openai.OpenAI(
            base_url=base_url,
            api_key=key or "EMPTY",
            timeout=float(m.request_timeout_s),
            max_retries=int(m.max_retries),
        )
        self._client = client
        self.base_url = str(client.base_url)
        self.sdk_version = openai.__version__
        self._errors = (
            (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
             openai.BadRequestError, openai.UnprocessableEntityError),
            (openai.RateLimitError, openai.InternalServerError),
            openai.APIStatusError,
            openai.APIConnectionError,
        )

    # ------------------------------------------------------------------ 공개 API

    def preflight(self) -> None:
        """호출 전 점검. anthropic/openai_compat: ``models.retrieve(name)`` (과금 없음) — 인증·모델명 오류면 예외를 던진다.
        dummy: 더미 응답 디렉터리가 있는지 확인한다."""
        if self.provider == "dummy":
            if not DUMMY_ROOT.is_dir():
                raise LLMClientError(f"더미 응답 디렉터리가 없다: {DUMMY_ROOT}")
            return
        info = self._client.models.retrieve(self.model)
        served = getattr(info, "id", None)
        print(f"{_LOG} preflight ok: {self.provider} model={self.model} (id={served}) base_url={self.base_url}", file=sys.stderr)

    def complete(self, system: str, user: str, *, dummy_key: tuple[str, str, int] | None = None) -> LLMResponse:
        """시스템·사용자 프롬프트로 한 번 호출한다. 예외는 던지지 않고 ``error`` 로 보고한다."""
        if self.provider == "dummy":
            return self._complete_dummy(system, user, dummy_key)
        t0 = time.perf_counter()
        try:
            if self.provider == "anthropic":
                return self._complete_anthropic(system, user, t0)
            return self._complete_openai_compat(system, user, t0)
        except Exception as e:  # noqa: BLE001 — 설계: 호출자에게 던지지 않는다
            latency_ms = (time.perf_counter() - t0) * 1000.0
            error = self._classify_error(e)
            print(f"{_LOG} 호출 실패 ({latency_ms:.0f} ms): {error}", file=sys.stderr)
            return self._error_response(error, latency_ms)

    # ------------------------------------------------------------------ 제공자별 구현

    def _complete_anthropic(self, system: str, user: str, t0: float) -> LLMResponse:
        m = self.cfg.model
        extra: dict[str, Any] = {}
        if m.temperature is not None:
            # SDK 2.x 에는 temperature 키워드가 없다(TypeError). Claude 4.7+/5.x 는 400 으로 거부한다.
            extra["extra_body"] = {"temperature": m.temperature}
        request_id: str | None = None
        with self._client.messages.stream(
            model=self.model,
            max_tokens=int(m.max_tokens),
            system=system,
            messages=[{"role": "user", "content": user}],
            thinking={"type": m.anthropic.thinking},
            output_config={"effort": m.anthropic.effort},
            **extra,
        ) as stream:
            msg = stream.get_final_message()
            try:
                request_id = getattr(stream, "request_id", None)
            except Exception:  # 헤더 접근 실패는 기록 누락으로만 취급
                request_id = None
        latency_ms = (time.perf_counter() - t0) * 1000.0
        if not request_id:
            request_id = getattr(msg, "_request_id", None)

        text_parts: list[str] = []
        n_text = 0
        n_thinking = 0
        for block in msg.content:
            btype = getattr(block, "type", "")
            if btype == "text":
                text_parts.append(str(getattr(block, "text", "")))
                n_text += 1
            elif btype in ("thinking", "redacted_thinking"):
                n_thinking += 1
        usage = msg.usage
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "output_tokens_details", None)
        thinking_tokens = getattr(details, "thinking_tokens", None) if details is not None else None
        tokens_thinking = int(thinking_tokens) if thinking_tokens is not None else None
        tokens_text = tokens_out - tokens_thinking if tokens_thinking is not None else None
        stop_reason = str(msg.stop_reason or "")
        stop_details = getattr(msg, "stop_details", None)
        category = getattr(stop_details, "category", None) if stop_details is not None else None
        return LLMResponse(
            text="".join(text_parts),
            model_requested=self.model,
            model_served=str(msg.model),
            provider=self.provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_thinking=tokens_thinking,
            tokens_text=tokens_text,
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            raw_stop_reason=stop_reason,
            stop_details_category=str(category) if category is not None else None,
            n_thinking_blocks=n_thinking,
            n_text_blocks=n_text,
            request_id=str(request_id) if request_id else None,
            raw_usage=_model_dump(usage),
            base_url=self.base_url,
            sdk_version=self.sdk_version,
            api_key_present=self.api_key_present,
            error=None,
        )

    def _complete_openai_compat(self, system: str, user: str, t0: float) -> LLMResponse:
        m = self.cfg.model
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            m.openai_compat.max_tokens_param: int(m.max_tokens),
        }
        if m.temperature is not None:
            kwargs["temperature"] = m.temperature
        if m.seed is not None:
            kwargs["seed"] = m.seed
        resp = self._client.chat.completions.create(**kwargs)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        choice = resp.choices[0] if getattr(resp, "choices", None) else None
        message = getattr(choice, "message", None)
        text = str(getattr(message, "content", None) or "") if message is not None else ""
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None) if message is not None else None
        raw_stop = str(getattr(choice, "finish_reason", None) or "") if choice is not None else ""
        usage = getattr(resp, "usage", None)
        tokens_in = int(getattr(usage, "prompt_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "completion_tokens", 0) or 0)
        details = getattr(usage, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details is not None else None
        tokens_thinking = int(reasoning_tokens) if reasoning_tokens is not None else None
        tokens_text = tokens_out - tokens_thinking if tokens_thinking is not None else None
        request_id = getattr(resp, "_request_id", None)
        return LLMResponse(
            text=text,
            model_requested=self.model,
            model_served=str(getattr(resp, "model", "") or ""),
            provider=self.provider,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_thinking=tokens_thinking,
            tokens_text=tokens_text,
            latency_ms=latency_ms,
            stop_reason=_OPENAI_STOP_MAP.get(raw_stop, raw_stop),
            raw_stop_reason=raw_stop,
            stop_details_category=None,
            n_thinking_blocks=1 if reasoning else 0,
            n_text_blocks=1 if text else 0,
            request_id=str(request_id) if request_id else None,
            raw_usage=_model_dump(usage),
            base_url=self.base_url,
            sdk_version=self.sdk_version,
            api_key_present=self.api_key_present,
            error=None,
        )

    def _complete_dummy(self, system: str, user: str, dummy_key: tuple[str, str, int] | None) -> LLMResponse:
        """``harness/dummy_responses/<task>/<language>/<k>.md`` (k = sample_idx mod 파일 수)를 응답으로 돌려준다."""
        if dummy_key is None:
            return self._error_response("fatal:DummyKeyMissing::dummy provider 는 dummy_key=(task, language, sample_idx) 가 필요하다", 0.0)
        task, language, sample_idx = dummy_key
        directory = DUMMY_ROOT / str(task) / str(language)
        files = sorted((p for p in directory.glob("*.md") if p.stem.isdigit()), key=lambda p: int(p.stem))
        if not files:
            return self._error_response(f"fatal:DummyResponseMissing::{directory} 에 <k>.md 더미 응답이 없다", 0.0)
        chosen = files[int(sample_idx) % len(files)]
        try:
            text = chosen.read_text(encoding="utf-8")
        except OSError as e:
            return self._error_response(f"fatal:{type(e).__name__}::{_one_line(str(e))}", 0.0)
        tokens_in = len((system + " " + user).split())
        tokens_out = len(text.split())
        return LLMResponse(
            text=text,
            model_requested=self.model,
            model_served="dummy",
            provider="dummy",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            tokens_thinking=None,
            tokens_text=None,
            latency_ms=0.0,
            stop_reason="end_turn",
            raw_stop_reason="end_turn",
            stop_details_category=None,
            n_thinking_blocks=0,
            n_text_blocks=1,
            request_id=None,
            raw_usage={"source": str(chosen.relative_to(ROOT)), "fake_tokens": "word_count"},
            base_url=None,
            sdk_version=None,
            api_key_present=False,
            error=None,
        )

    # ------------------------------------------------------------------ 오류

    def _classify_error(self, e: BaseException) -> str:
        """예외 → ``fatal:...`` / ``transient:...`` (구체 → 일반 순서)."""
        kind = "fatal"
        status = getattr(e, "status_code", None)
        note = ""
        if self._errors is not None:
            fatal_types, transient_types, status_error, connection_error = self._errors
            if isinstance(e, fatal_types):
                kind = "fatal"
            elif isinstance(e, transient_types):
                kind = "transient"
            elif isinstance(e, status_error):
                if isinstance(status, int) and 400 <= status < 600:
                    kind = "transient" if status >= 500 else "fatal"
                else:
                    # 스트림 도중 SSE error 이벤트: HTTP 응답은 이미 200 으로 시작됐고 SDK 는 상태 200 의 평범한 APIStatusError 를
                    # 던진다(anthropic ``_streaming.py`` → ``_make_status_error``). ``max_retries`` 는 스트림 시작 후 오류에 적용되지
                    # 않으므로 body 의 error.type 으로 가른다(모르는 타입은 transient: 다음 실행에서 재호출).
                    etype = _error_body_type(e)
                    kind = "fatal" if etype in _FATAL_ERROR_TYPES else "transient"
                    note = f"in-stream error event ({etype or 'unknown type'}): "
            elif isinstance(e, connection_error):
                kind = "transient"
            else:
                kind = "fatal"  # 알 수 없는 예외(SDK 불일치 등)는 중단해서 사람이 보게 한다
        return f"{kind}:{type(e).__name__}:{status if status is not None else ''}:{_one_line(note + str(e))}"

    def _error_response(self, error: str, latency_ms: float) -> LLMResponse:
        return LLMResponse(
            text="",
            model_requested=self.model,
            model_served="",
            provider=self.provider,
            tokens_in=0,
            tokens_out=0,
            tokens_thinking=None,
            tokens_text=None,
            latency_ms=latency_ms,
            stop_reason="",
            raw_stop_reason="",
            stop_details_category=None,
            n_thinking_blocks=0,
            n_text_blocks=0,
            request_id=None,
            raw_usage={},
            base_url=self.base_url,
            sdk_version=self.sdk_version,
            api_key_present=self.api_key_present,
            error=error,
        )


# ---------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> int:
    """``--info``: 클라이언트 구성 출력(요청 없음). ``--preflight``: models.retrieve (네트워크). ``--dummy TASK LANG IDX``: 더미 응답."""
    parser = argparse.ArgumentParser(prog="python -m harness.llm_client", description="LLM 클라이언트 진단")
    parser.add_argument("--config", type=Path, default=None, help="experiment.yaml 경로")
    parser.add_argument("--provider", default=None, help="model.provider 덮어쓰기 (LLM_PROVIDER)")
    parser.add_argument("--info", action="store_true", help="provider/model/base_url/sdk/api_key_present 출력 (요청 없음)")
    parser.add_argument("--preflight", action="store_true", help="models.retrieve 로 인증·모델명 확인 (네트워크 호출)")
    parser.add_argument("--dummy", nargs=3, metavar=("TASK", "LANGUAGE", "IDX"), default=None, help="더미 응답을 LLMResponse JSON 으로 출력")
    ns = parser.parse_args(argv)
    if ns.provider:
        os.environ["LLM_PROVIDER"] = ns.provider
    try:
        cfg = load_config(ns.config)
        client = LLMClient(cfg)
    except (ConfigError, LLMClientError) as e:
        print(f"{_LOG} 오류: {e}", file=sys.stderr)
        return 1
    did = False
    if ns.info:
        print(json.dumps({
            "provider": client.provider, "model": client.model, "base_url": client.base_url,
            "sdk_version": client.sdk_version, "api_key_present": client.api_key_present,
            "max_tokens": cfg.model.max_tokens, "temperature": cfg.model.temperature, "seed": cfg.model.seed,
            "effort": cfg.model.anthropic.effort, "thinking": cfg.model.anthropic.thinking,
        }, ensure_ascii=False, indent=2))
        did = True
    if ns.preflight:
        client.preflight()
        print("preflight ok")
        did = True
    if ns.dummy is not None:
        task, language, idx = ns.dummy
        resp = client.complete("system", "user", dummy_key=(task, language, int(idx)))
        print(json.dumps(asdict(resp), ensure_ascii=False, indent=2))
        did = True
    if not did:
        parser.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
