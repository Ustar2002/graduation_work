"""llm_client 테스트: 예외 분류(``_classify_error``)와 anthropic 스트리밍 경로를 ``httpx2.MockTransport`` 로 네트워크 없이 통과한다.

- 스트림 도중 SSE ``event: error`` (응답은 HTTP 200 으로 시작됨)는 SDK 가 ``APIStatusError(status_code=200)`` 로 던진다
  → HTTP 상태가 아니라 body ``error.type`` 으로 분류해야 한다(``overloaded_error`` 가 fatal 이면 generate 배치 전체가 중단된다).
- 정상 스트림: 텍스트 결합·thinking 토큰·stop_details·request_id·요청 JSON(thinking/output_config/extra_body) 확인.

실제 API 호출·네트워크 없음: ``MockTransport`` 가 모든 요청을 가로채고 SDK 재시도는 ``max_retries=0`` 으로 끈다.
"""

from __future__ import annotations

import dataclasses
import json
import os
import unittest
from typing import Any, Callable
from unittest import mock

import anthropic
import httpx2

from harness.config import Config, load_config
from harness.llm_client import LLMClient

FAKE_KEY = "test-key-not-real"
URL = "https://api.anthropic.com/v1/messages"
REQUEST_ID = "req_mock_123"

Event = tuple[str, dict[str, Any]]


# ----------------------------------------------------------------------------- SSE 스트림 조립

def _sse(events: list[Event]) -> bytes:
    return "".join(f"event: {ev}\ndata: {json.dumps(data)}\n\n" for ev, data in events).encode()


def _message_start(model: str = "claude-mock-served") -> list[Event]:
    return [("message_start", {"type": "message_start", "message": {
        "id": "msg_mock", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 11, "output_tokens": 1}}})]


def _text_block(index: int, *chunks: str) -> list[Event]:
    events: list[Event] = [("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "text", "text": ""}})]
    events += [("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "text_delta", "text": t}}) for t in chunks]
    events.append(("content_block_stop", {"type": "content_block_stop", "index": index}))
    return events


def _thinking_block(index: int) -> list[Event]:
    return [
        ("content_block_start", {"type": "content_block_start", "index": index, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "thinking_delta", "thinking": "let me think"}}),
        ("content_block_delta", {"type": "content_block_delta", "index": index, "delta": {"type": "signature_delta", "signature": "sig"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": index}),
    ]


def _message_end(*, stop_reason: str = "end_turn", output_tokens: int = 40, thinking_tokens: int | None = 15,
                 stop_details: dict[str, Any] | None = None) -> list[Event]:
    delta: dict[str, Any] = {"stop_reason": stop_reason, "stop_sequence": None}
    if stop_details is not None:
        delta["stop_details"] = stop_details
    usage: dict[str, Any] = {"output_tokens": output_tokens}
    if thinking_tokens is not None:
        usage["output_tokens_details"] = {"thinking_tokens": thinking_tokens}
    return [("message_delta", {"type": "message_delta", "delta": delta, "usage": usage}), ("message_stop", {"type": "message_stop"})]


def _error_event(etype: str, message: str = "boom") -> list[Event]:
    return [("error", {"type": "error", "error": {"type": etype, "message": message}})]


def _error_body(etype: str, message: str = "boom") -> dict[str, Any]:
    return {"type": "error", "error": {"type": etype, "message": message}}


# ----------------------------------------------------------------------------- 클라이언트 조립

def _anthropic_cfg(**model_overrides: Any) -> Config:
    """실제 experiment.yaml 을 읽되 provider=anthropic, 재시도 0, 짧은 타임아웃으로 바꾼다(환경변수 덮어쓰기는 무시)."""
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": FAKE_KEY}):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)
        cfg = load_config()
    model = dataclasses.replace(cfg.model, provider="anthropic", name="claude-mock", max_retries=0, request_timeout_s=5.0, **model_overrides)
    return dataclasses.replace(cfg, model=model)


def _llm(cfg: Config, handler: Callable[[httpx2.Request], httpx2.Response] | None = None) -> LLMClient:
    """``LLMClient`` 를 만들고(네트워크 없음) SDK 클라이언트를 MockTransport 기반으로 바꿔 끼운다."""
    with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": FAKE_KEY}):
        llm = LLMClient(cfg)
    if handler is not None:
        llm._client = anthropic.Anthropic(
            api_key=FAKE_KEY, max_retries=0, timeout=5.0,
            http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        )
    return llm


class _Recorder:
    """요청을 기록하고 미리 정한 응답을 돌려주는 MockTransport 핸들러."""

    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status
        self.requests: list[dict[str, Any]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append({"url": str(request.url), "headers": dict(request.headers), "json": json.loads(request.content)})
        content_type = "text/event-stream" if self.status == 200 else "application/json"
        return httpx2.Response(self.status, headers={"content-type": content_type, "request-id": REQUEST_ID}, content=self.body)


def _status_error(cls: type[anthropic.APIStatusError], status: int, body: Any) -> anthropic.APIStatusError:
    response = httpx2.Response(status, request=httpx2.Request("POST", URL))
    return cls(f"{body}", response=response, body=body)


# ----------------------------------------------------------------------------- 테스트

class ClassifyErrorTest(unittest.TestCase):
    """``_classify_error`` 의 분류 표(설계 §5 예외 체인 + in-stream error 이벤트)."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.llm = _llm(_anthropic_cfg())

    def test_in_stream_error_event_is_classified_by_body_type(self) -> None:
        # anthropic _streaming.py: event: error → _make_status_error(..., response=self.response) → 상태 200 의 APIStatusError
        for etype in ("overloaded_error", "api_error", "rate_limit_error", "timeout_error"):
            with self.subTest(etype=etype):
                err = self.llm._classify_error(_status_error(anthropic.APIStatusError, 200, _error_body(etype)))
                self.assertTrue(err.startswith("transient:APIStatusError:200:"), err)
                self.assertIn(f"in-stream error event ({etype})", err)
        for etype in ("invalid_request_error", "authentication_error", "permission_error", "not_found_error", "billing_error"):
            with self.subTest(etype=etype):
                err = self.llm._classify_error(_status_error(anthropic.APIStatusError, 200, _error_body(etype)))
                self.assertTrue(err.startswith("fatal:APIStatusError:200:"), err)
                self.assertIn(f"in-stream error event ({etype})", err)

    def test_in_stream_unknown_or_malformed_body_is_transient(self) -> None:
        err = self.llm._classify_error(_status_error(anthropic.APIStatusError, 200, _error_body("brand_new_error")))
        self.assertTrue(err.startswith("transient:APIStatusError:200:in-stream error event (brand_new_error)"), err)
        err = self.llm._classify_error(_status_error(anthropic.APIStatusError, 200, "not json at all"))
        self.assertTrue(err.startswith("transient:APIStatusError:200:in-stream error event (unknown type)"), err)

    def test_http_status_chain_is_unchanged(self) -> None:
        cases: list[tuple[anthropic.APIStatusError, str]] = [
            (_status_error(anthropic.BadRequestError, 400, _error_body("invalid_request_error")), "fatal:BadRequestError:400:"),
            (_status_error(anthropic.AuthenticationError, 401, _error_body("authentication_error")), "fatal:AuthenticationError:401:"),
            (_status_error(anthropic.NotFoundError, 404, _error_body("not_found_error")), "fatal:NotFoundError:404:"),
            (_status_error(anthropic.APIStatusError, 418, _error_body("api_error")), "fatal:APIStatusError:418:"),  # 그 외 4xx
            (_status_error(anthropic.RateLimitError, 429, _error_body("rate_limit_error")), "transient:RateLimitError:429:"),
            (_status_error(anthropic.InternalServerError, 500, _error_body("api_error")), "transient:InternalServerError:500:"),
            (_status_error(anthropic.OverloadedError, 529, _error_body("overloaded_error")), "transient:OverloadedError:529:"),
            (_status_error(anthropic.APIStatusError, 599, _error_body("api_error")), "transient:APIStatusError:599:"),  # 그 외 5xx
        ]
        for exc, prefix in cases:
            with self.subTest(prefix=prefix):
                err = self.llm._classify_error(exc)
                self.assertTrue(err.startswith(prefix), err)
                self.assertNotIn("in-stream", err)
        request = httpx2.Request("POST", URL)
        self.assertTrue(self.llm._classify_error(anthropic.APIConnectionError(request=request)).startswith("transient:APIConnectionError::"))
        self.assertTrue(self.llm._classify_error(anthropic.APITimeoutError(request=request)).startswith("transient:APITimeoutError::"))
        self.assertTrue(self.llm._classify_error(RuntimeError("sdk mismatch")).startswith("fatal:RuntimeError::sdk mismatch"))

    def test_error_string_is_one_line_and_bounded(self) -> None:
        err = self.llm._classify_error(_status_error(anthropic.APIStatusError, 200, _error_body("overloaded_error", "x\n" * 2000)))
        self.assertNotIn("\n", err)
        self.assertLess(len(err), 600)


class AnthropicStreamMockTest(unittest.TestCase):
    """``complete()`` 를 SDK 의 실제 스트리밍 경로(MockTransport)로 통과시킨다."""

    def setUp(self) -> None:
        self.cfg = _anthropic_cfg()

    def _complete(self, body: bytes, status: int = 200, cfg: Config | None = None) -> tuple[Any, _Recorder]:
        recorder = _Recorder(body, status)
        resp = _llm(cfg or self.cfg, recorder).complete("SYSTEM PROMPT", "USER PROMPT")
        return resp, recorder

    def test_in_stream_overloaded_is_transient(self) -> None:
        # 회귀: 예전에는 fatal:APIStatusError:200 으로 분류돼 generate 가 나머지 샘플 호출을 모두 중단했다(exit 3)
        stream = _sse(_message_start() + _text_block(0, "partial") + _error_event("overloaded_error", "Overloaded"))
        resp, recorder = self._complete(stream)
        self.assertIsNotNone(resp.error)
        self.assertTrue(resp.error.startswith("transient:APIStatusError:200:in-stream error event (overloaded_error)"), resp.error)
        self.assertFalse(resp.error.startswith("fatal:"))
        self.assertEqual((resp.text, resp.tokens_in, resp.tokens_out, resp.stop_reason, resp.model_served), ("", 0, 0, "", ""))
        self.assertEqual(len(recorder.requests), 1)  # max_retries=0: 재시도 없음
        self.assertTrue(recorder.requests[0]["url"].endswith("/v1/messages"))

    def test_in_stream_invalid_request_is_fatal(self) -> None:
        stream = _sse(_message_start() + _error_event("invalid_request_error", "bad thinking config"))
        resp, _ = self._complete(stream)
        self.assertTrue(resp.error.startswith("fatal:APIStatusError:200:in-stream error event (invalid_request_error)"), resp.error)
        self.assertIn("bad thinking config", resp.error)

    def test_http_level_errors_keep_sdk_classes(self) -> None:
        resp, _ = self._complete(json.dumps(_error_body("overloaded_error", "Overloaded")).encode(), status=529)
        self.assertTrue(resp.error.startswith("transient:OverloadedError:529:"), resp.error)
        resp, _ = self._complete(json.dumps(_error_body("authentication_error", "invalid x-api-key")).encode(), status=401)
        self.assertTrue(resp.error.startswith("fatal:AuthenticationError:401:"), resp.error)

    def test_request_body_matches_design(self) -> None:
        stream = _sse(_message_start() + _text_block(0, "ok") + _message_end())
        _, recorder = self._complete(stream)
        body = recorder.requests[0]["json"]
        self.assertEqual(body["model"], "claude-mock")
        self.assertEqual(body["max_tokens"], self.cfg.model.max_tokens)
        self.assertEqual(body["system"], "SYSTEM PROMPT")
        self.assertEqual(body["messages"], [{"role": "user", "content": "USER PROMPT"}])
        self.assertEqual(body["thinking"], {"type": self.cfg.model.anthropic.thinking})
        self.assertEqual(body["output_config"], {"effort": self.cfg.model.anthropic.effort})
        self.assertIs(body["stream"], True)
        self.assertNotIn("temperature", body)  # temperature: null → 보내지 않음
        self.assertNotIn("fallbacks", body)
        self.assertNotIn("anthropic-beta", {k.lower() for k in recorder.requests[0]["headers"]})

    def test_temperature_is_sent_via_extra_body(self) -> None:
        cfg = _anthropic_cfg(temperature=0.2)
        stream = _sse(_message_start() + _text_block(0, "ok") + _message_end())
        _, recorder = self._complete(stream, cfg=cfg)
        self.assertEqual(recorder.requests[0]["json"]["temperature"], 0.2)

    def test_success_stream_fields(self) -> None:
        stream = _sse(
            _message_start("claude-mock-served")
            + _thinking_block(0)
            + _text_block(1, "```python\n", "def gemm(A, B, C, M, N, K):\n")
            + _text_block(2, "    C[:] = A @ B\n```\n")
            + _message_end(output_tokens=40, thinking_tokens=15)
        )
        resp, _ = self._complete(stream)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.text, "```python\ndef gemm(A, B, C, M, N, K):\n    C[:] = A @ B\n```\n")
        self.assertEqual((resp.provider, resp.model_requested, resp.model_served), ("anthropic", "claude-mock", "claude-mock-served"))
        self.assertEqual((resp.tokens_in, resp.tokens_out, resp.tokens_thinking, resp.tokens_text), (11, 40, 15, 25))
        self.assertEqual((resp.stop_reason, resp.raw_stop_reason, resp.stop_details_category), ("end_turn", "end_turn", None))
        self.assertEqual((resp.n_thinking_blocks, resp.n_text_blocks), (1, 2))
        self.assertEqual(resp.request_id, REQUEST_ID)
        self.assertEqual(resp.raw_usage["output_tokens_details"], {"thinking_tokens": 15})
        self.assertEqual(resp.raw_usage["input_tokens"], 11)
        self.assertEqual(resp.sdk_version, anthropic.__version__)
        self.assertTrue(resp.api_key_present)
        self.assertGreaterEqual(resp.latency_ms, 0.0)

    def test_usage_without_thinking_details(self) -> None:
        stream = _sse(_message_start() + _text_block(0, "ok") + _message_end(output_tokens=7, thinking_tokens=None))
        resp, _ = self._complete(stream)
        self.assertIsNone(resp.error)
        self.assertEqual((resp.tokens_out, resp.tokens_thinking, resp.tokens_text), (7, None, None))

    def test_refusal_records_stop_details_category(self) -> None:
        stream = _sse(
            _message_start()
            + _text_block(0, "I can't help with that.")
            + _message_end(stop_reason="refusal", stop_details={"type": "refusal", "category": "cyber", "explanation": "policy"})
        )
        resp, _ = self._complete(stream)
        self.assertIsNone(resp.error)
        self.assertEqual((resp.stop_reason, resp.stop_details_category), ("refusal", "cyber"))

    def test_max_tokens_stop_reason_passes_through(self) -> None:
        stream = _sse(_message_start() + _text_block(0, "trunc") + _message_end(stop_reason="max_tokens"))
        resp, _ = self._complete(stream)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.stop_reason, "max_tokens")


if __name__ == "__main__":
    unittest.main()
