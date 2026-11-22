"""llm_client 테스트: 예외 분류(``_classify_error``)와 anthropic 스트리밍 경로를 ``httpx2.MockTransport`` 로 네트워크 없이 통과한다.

- 스트림 도중 SSE ``event: error`` (응답은 HTTP 200 으로 시작됨)는 SDK 가 ``APIStatusError(status_code=200)`` 로 던진다
  → HTTP 상태가 아니라 body ``error.type`` 으로 분류해야 한다(``overloaded_error`` 가 fatal 이면 generate 배치 전체가 중단된다).
- 정상 스트림: 텍스트 결합·thinking 토큰·stop_details·request_id·요청 JSON(thinking/output_config/extra_body) 확인.
- 수정 루프(설계 v3 §9.2): ``messages`` 로 준 대화가 그대로 나가는지(anthropic·openai_compat), dummy 의 턴 파일 찾기.
  SDK 가 없는 머신에서도 도는 것: 대화 검사(``conversation``), dummy 제공자, SDK 자리에 가짜 모듈을 끼워 본 요청 조립.

실제 API 호출·네트워크 없음: ``MockTransport`` 가 모든 요청을 가로채고 SDK 재시도는 ``max_retries=0`` 으로 끈다.
"""

from __future__ import annotations

import contextlib
import dataclasses
import importlib
import io
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any, Callable
from unittest import mock

try:
    import anthropic
    import httpx2
except ModuleNotFoundError:  # LLM SDK 가 없는 개발용 venv: SDK 를 쓰는 클래스만 requires_modules 표시로 건너뛴다
    anthropic = httpx2 = None  # type: ignore[assignment]
try:
    import openai
except ModuleNotFoundError:
    openai = None  # type: ignore[assignment]

from harness import llm_client
from harness.config import Config, load_config
from harness.llm_client import LLMClient, conversation
from harness.paths import ROOT
from tests._support import requires_modules

# SDK 가 필요한 클래스에만 붙인다(모듈 전체를 건너뛰지 않는다 — SDK 가 필요 없는 테스트는 개발 머신에서도 돌아야 한다).
# Linux 에서는 건너뛰지 않고 ModuleNotFoundError 로 실패한다(tests/_support.py).
requires_llm_sdk = requires_modules("anthropic", "httpx2")

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

@requires_llm_sdk
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


@requires_llm_sdk
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

    def test_messages_are_sent_as_given_and_user_is_not_used(self) -> None:
        recorder = _Recorder(_sse(_message_start() + _text_block(0, "fixed") + _message_end()))
        resp = _llm(self.cfg, recorder).complete("SYSTEM PROMPT", "NOT SENT", messages=REPAIR_MESSAGES)
        self.assertIsNone(resp.error)
        self.assertEqual(resp.text, "fixed")
        body = recorder.requests[0]["json"]
        self.assertEqual(body["messages"], REPAIR_MESSAGES)  # 이전 응답은 텍스트만: thinking 블록을 다시 보내지 않는다
        self.assertEqual(body["system"], "SYSTEM PROMPT")
        self.assertNotIn("NOT SENT", json.dumps(body))
        self.assertEqual(body["thinking"], {"type": self.cfg.model.anthropic.thinking})
        self.assertEqual(body["output_config"], {"effort": self.cfg.model.anthropic.effort})


# ----------------------------------------------------------------------------- 수정 루프: 대화와 dummy 턴 (SDK 불필요)

REPAIR_MESSAGES = [
    {"role": "user", "content": "TURN 0 PROMPT"},
    {"role": "assistant", "content": "TURN 0 REPLY"},
    {"role": "user", "content": "REPAIR REQUEST 1"},
]


def _cfg(provider: str, **model_overrides: Any) -> Config:
    """저장소 설정에서 제공자만 바꾼 것(환경변수의 제공자·모델 덮어쓰기는 무시)."""
    with mock.patch.dict(os.environ):
        os.environ.pop("LLM_PROVIDER", None)
        os.environ.pop("LLM_MODEL", None)
        cfg = load_config()
    return dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, provider=provider, **model_overrides))


class ConversationTest(unittest.TestCase):
    """``conversation``: 보낼 대화의 조립과 형태 검사."""

    def test_single_turn_is_one_user_message(self) -> None:
        self.assertEqual(conversation("PROMPT", None), [{"role": "user", "content": "PROMPT"}])

    def test_messages_replace_the_user_argument(self) -> None:
        self.assertEqual(conversation("NOT SENT", REPAIR_MESSAGES), REPAIR_MESSAGES)
        five = [*REPAIR_MESSAGES, {"role": "assistant", "content": "TURN 1 REPLY"}, {"role": "user", "content": "REPAIR REQUEST 2"}]
        self.assertEqual(conversation("", five), five)

    def test_result_is_a_copy_with_role_and_content_only(self) -> None:
        given = [{"role": "user", "content": "PROMPT", "thinking": "not forwarded", "cache_control": {"type": "ephemeral"}}]
        turns = conversation("", given)
        self.assertEqual(turns, [{"role": "user", "content": "PROMPT"}])
        turns[0]["content"] = "changed"
        self.assertEqual(given[0]["content"], "PROMPT")

    def test_malformed_conversations_are_a_harness_bug(self) -> None:
        user, assistant = {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}
        bad = {
            "empty": [],
            "ends with assistant": [user, assistant],
            "starts with assistant": [assistant, user, assistant],
            "two users in a row": [user, user, user],
            "system role": [{"role": "system", "content": "s"}],
            "content is not text": [{"role": "user", "content": [{"type": "text", "text": "u"}]}],
            "content missing": [{"role": "user"}],
            # 빈 응답(사고 블록만 있던 end_turn 등)을 assistant 턴으로 보내면 Messages API 가 400 으로 거부한다
            "empty assistant reply": [user, {"role": "assistant", "content": ""}, user],
            "whitespace-only assistant reply": [user, {"role": "assistant", "content": "  \n"}, user],
            "empty user message": [user, assistant, {"role": "user", "content": ""}],
        }
        for label, messages in bad.items():
            with self.subTest(label=label), self.assertRaises(ValueError):
                conversation("PROMPT", messages)


class DummyProviderTest(unittest.TestCase):
    """dummy 제공자: ``<k>.md`` 와 수정 턴의 ``<k>.turn<t>.md`` (설계 v3 §9.2). SDK 도 네트워크도 쓰지 않는다."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        directory = self.root / "task" / "lang"
        directory.mkdir(parents=True)
        for name, text in {"0.md": "reply zero", "1.md": "reply one", "2.md": "reply two", "1.turn1.md": "one, repaired once",
                           "1.turn3.md": "one, third repair", "README.md": "not a reply", "notes.turn1.md": "not a reply"}.items():
            (directory / name).write_text(text, encoding="utf-8")
        patcher = mock.patch.object(llm_client, "DUMMY_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.llm = LLMClient(_cfg("dummy"))

    def reply(self, *key: Any, **kwargs: Any) -> Any:
        return self.llm.complete("SYSTEM", "USER", dummy_key=key, **kwargs)

    def test_no_sdk_is_touched(self) -> None:
        self.assertEqual((self.llm.provider, self.llm._client, self.llm.sdk_version, self.llm.base_url), ("dummy", None, None, None))
        self.llm.preflight()

    def test_turn_zero_is_the_numbered_file(self) -> None:
        for key, text in (((0,), "reply zero"), ((1,), "reply one"), ((2,), "reply two"), ((4,), "reply one"),  # 4 mod 3
                          ((1, 0), "reply one")):
            with self.subTest(key=key):
                resp = self.reply("task", "lang", *key)
                self.assertEqual((resp.text, resp.error, resp.stop_reason, resp.model_served), (text, None, "end_turn", "dummy"))

    def test_repair_turn_prefers_the_turn_file_of_the_same_sample(self) -> None:
        self.assertEqual(self.reply("task", "lang", 1, 1).text, "one, repaired once")
        self.assertEqual(self.reply("task", "lang", 4, 1).text, "one, repaired once")  # 턴 파일은 k 를 고르는 파일 수에 들지 않는다
        self.assertEqual(self.reply("task", "lang", 1, 3).text, "one, third repair")
        self.assertEqual(Path(self.reply("task", "lang", 1, 1).raw_usage["source"]), (self.root / "task" / "lang" / "1.turn1.md").resolve())

    def test_repair_turn_without_a_turn_file_repeats_the_first_reply(self) -> None:
        self.assertEqual(self.reply("task", "lang", 1, 2).text, "reply one")  # 1.turn2.md 는 없다: 같은 실패가 반복된다
        self.assertEqual(self.reply("task", "lang", 0, 1).text, "reply zero")
        self.assertEqual(Path(self.reply("task", "lang", 2, 5).raw_usage["source"]), (self.root / "task" / "lang" / "2.md").resolve())

    def test_byte_order_mark_is_not_part_of_the_reply(self) -> None:
        (self.root / "task" / "lang" / "0.md").write_bytes(b"\xef\xbb\xbfreply with a BOM\n")
        (self.root / "task" / "lang" / "0.turn1.md").write_bytes(b"\xef\xbb\xbfrepaired with a BOM\n")
        self.assertEqual(self.reply("task", "lang", 0).text, "reply with a BOM\n")
        self.assertEqual(self.reply("task", "lang", 0, 1).text, "repaired with a BOM\n")

    def test_fake_token_counts_cover_the_whole_conversation(self) -> None:
        single = self.reply("task", "lang", 0)
        self.assertEqual((single.tokens_in, single.tokens_out), (2, 2))  # "SYSTEM USER" / "reply zero"
        multi = self.reply("task", "lang", 1, 1, messages=REPAIR_MESSAGES)
        self.assertEqual((multi.tokens_in, multi.tokens_out), (1 + 3 + 3 + 3, 3))
        self.assertEqual(multi.raw_usage["fake_tokens"], "word_count")

    def test_missing_key_or_files_is_a_fatal_error_response(self) -> None:
        for key in (None, ("task", "lang"), ("task", "lang", 0, 1, 2)):
            with self.subTest(key=key):
                resp = self.llm.complete("SYSTEM", "USER", dummy_key=key)
                self.assertTrue(resp.error.startswith("fatal:DummyKeyMissing:"), resp.error)
        resp = self.reply("task", "other_language", 0)
        self.assertTrue(resp.error.startswith("fatal:DummyResponseMissing:"), resp.error)
        self.assertEqual(resp.text, "")

    def test_malformed_conversation_raises_before_any_reply(self) -> None:
        with self.assertRaises(ValueError):
            self.reply("task", "lang", 0, 1, messages=REPAIR_MESSAGES[:2])


class RepositoryDummyTurnFilesTest(unittest.TestCase):
    """저장소의 gemm 더미: 수정 턴 1 의 응답 파일 둘(설계 v3 §14)."""

    def setUp(self) -> None:
        self.llm = LLMClient(_cfg("dummy"))

    def test_c_seq_sample_one_is_fixed_in_turn_one(self) -> None:
        first = self.llm.complete("s", "u", dummy_key=("gemm", "c_seq", 1))
        fixed = self.llm.complete("s", "u", dummy_key=("gemm", "c_seq", 1, 1))
        self.assertEqual(fixed.raw_usage["source"], "harness/dummy_responses/gemm/c_seq/1.turn1.md")
        self.assertIn("float sum = 0.0f\n", first.text)  # 턴 0: 세미콜론 누락
        self.assertIn("float sum = 0.0f;\n", fixed.text)
        self.assertEqual(fixed.text.replace("\r", "").count("```"), 2)
        self.assertEqual(self.llm.complete("s", "u", dummy_key=("gemm", "c_seq", 1, 2)).text, first.text)  # 턴 2 파일은 없다

    def test_numpy_sample_two_is_fixed_in_turn_one(self) -> None:
        first = self.llm.complete("s", "u", dummy_key=("gemm", "numpy", 2))
        fixed = self.llm.complete("s", "u", dummy_key=("gemm", "numpy", 2, 1))
        self.assertEqual(fixed.raw_usage["source"], "harness/dummy_responses/gemm/numpy/2.turn1.md")
        self.assertIn("1.01", first.text)
        self.assertIn("np.matmul(A, B, out=C)", fixed.text)
        self.assertNotIn("1.01", fixed.text.split("```python")[1])

    def test_turn_files_do_not_change_which_file_a_sample_gets(self) -> None:
        for language, count in (("numpy", 5), ("c_seq", 3), ("c_openmp", 3)):
            directory = ROOT / "harness" / "dummy_responses" / "gemm" / language
            self.assertEqual(sorted(p.name for p in directory.glob("*.md") if p.stem.isdigit()), [f"{k}.md" for k in range(count)])
            for idx in range(count):
                source = self.llm.complete("s", "u", dummy_key=("gemm", language, idx)).raw_usage["source"]
                self.assertEqual(source, f"harness/dummy_responses/gemm/{language}/{idx}.md")


# ----------------------------------------------------------------------------- 요청 조립 (SDK 자리에 가짜 모듈)


class _FakeSdkError(Exception):
    pass


def _fake_sdk_module(name: str, client_class: type) -> types.ModuleType:
    """``LLMClient`` 가 초기화 때 읽는 이름만 가진 가짜 SDK 모듈(예외 클래스, 버전, 클라이언트 클래스)."""
    module = types.ModuleType(name)
    module.__version__ = "0.0-fake"
    for error in ("AuthenticationError", "PermissionDeniedError", "NotFoundError", "BadRequestError", "UnprocessableEntityError",
                  "RateLimitError", "InternalServerError", "APIStatusError", "APIConnectionError"):
        setattr(module, error, type(error, (_FakeSdkError,), {}))
    setattr(module, "Anthropic" if name == "anthropic" else "OpenAI", client_class)
    return module


class _FakeAnthropic:
    """``client.messages.stream(**kwargs)`` 의 인자를 기록하고 고정 응답을 주는 가짜."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.api_key = "fake"
        self.base_url = "https://fake.invalid"
        self.requests: list[dict[str, Any]] = []
        self.messages = types.SimpleNamespace(stream=self._stream)

    @contextlib.contextmanager
    def _stream(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        message = types.SimpleNamespace(
            content=[types.SimpleNamespace(type="thinking", thinking="..."), types.SimpleNamespace(type="text", text="REPLY")],
            usage=types.SimpleNamespace(input_tokens=30, output_tokens=12, output_tokens_details=types.SimpleNamespace(thinking_tokens=4)),
            stop_reason="end_turn", stop_details=None, model="served-model")
        yield types.SimpleNamespace(get_final_message=lambda: message, request_id="req_fake")


class _FakeOpenAI:
    """``client.chat.completions.create(**kwargs)`` 의 인자를 기록하고 고정 응답을 주는 가짜."""

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.base_url = kwargs["base_url"]
        self.requests: list[dict[str, Any]] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        choice = types.SimpleNamespace(message=types.SimpleNamespace(content="REPLY"), finish_reason="stop")
        usage = types.SimpleNamespace(prompt_tokens=30, completion_tokens=12, completion_tokens_details=None)
        return types.SimpleNamespace(choices=[choice], usage=usage, model="served-model", _request_id="req_fake")


class RequestAssemblyTest(unittest.TestCase):
    """``complete`` 가 SDK 에 넘기는 인자: 한 턴 호출은 예전 그대로, ``messages`` 가 있으면 그 대화. SDK 설치 여부와 무관하게 돈다."""

    def client(self, provider: str, **model_overrides: Any) -> tuple[LLMClient, Any]:
        fake = _fake_sdk_module("anthropic", _FakeAnthropic) if provider == "anthropic" else _fake_sdk_module("openai", _FakeOpenAI)
        env = {"ANTHROPIC_API_KEY": FAKE_KEY, "LLM_BASE_URL": "http://mock.invalid/v1", "OPENAI_API_KEY": FAKE_KEY}
        with mock.patch.dict(sys.modules, {fake.__name__: fake}), mock.patch.dict(os.environ, env):
            llm = LLMClient(_cfg(provider, **model_overrides))
        return llm, llm._client

    def test_anthropic_single_turn_request_is_unchanged(self) -> None:
        llm, sdk = self.client("anthropic")
        resp = llm.complete("SYSTEM", "USER")
        m = llm.cfg.model
        self.assertEqual(sdk.requests, [{"model": m.name, "max_tokens": m.max_tokens, "system": "SYSTEM",
                                         "messages": [{"role": "user", "content": "USER"}],
                                         "thinking": {"type": m.anthropic.thinking}, "output_config": {"effort": m.anthropic.effort}}])
        self.assertEqual((resp.text, resp.error, resp.tokens_in, resp.tokens_out, resp.tokens_thinking, resp.tokens_text),
                         ("REPLY", None, 30, 12, 4, 8))
        self.assertEqual((resp.n_thinking_blocks, resp.n_text_blocks, resp.request_id, resp.model_served), (1, 1, "req_fake", "served-model"))

    def test_anthropic_sends_the_conversation(self) -> None:
        llm, sdk = self.client("anthropic")
        resp = llm.complete("SYSTEM", "NOT SENT", messages=REPAIR_MESSAGES, dummy_key=("gemm", "numpy", 0, 1))
        self.assertIsNone(resp.error)
        self.assertEqual(sdk.requests[0]["messages"], REPAIR_MESSAGES)
        self.assertEqual(sdk.requests[0]["system"], "SYSTEM")
        self.assertEqual({k: v for k, v in sdk.requests[0].items() if k != "messages"},
                         {k: v for k, v in self.single_turn_request("anthropic").items() if k != "messages"})

    def test_openai_compat_single_turn_request_is_unchanged(self) -> None:
        llm, sdk = self.client("openai_compat")
        resp = llm.complete("SYSTEM", "USER")
        m = llm.cfg.model
        self.assertEqual(sdk.requests, [{"model": m.name, "max_tokens": m.max_tokens,
                                         "messages": [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "USER"}]}])
        self.assertEqual((resp.text, resp.error, resp.stop_reason, resp.raw_stop_reason, resp.tokens_in, resp.tokens_out),
                         ("REPLY", None, "end_turn", "stop", 30, 12))

    def test_openai_compat_appends_the_conversation_to_the_system_message(self) -> None:
        llm, sdk = self.client("openai_compat", temperature=0.2, seed=7)
        with contextlib.redirect_stderr(io.StringIO()):
            resp = llm.complete("SYSTEM", "NOT SENT", messages=REPAIR_MESSAGES)
        self.assertIsNone(resp.error)
        self.assertEqual(sdk.requests[0]["messages"], [{"role": "system", "content": "SYSTEM"}, *REPAIR_MESSAGES])
        self.assertEqual((sdk.requests[0]["temperature"], sdk.requests[0]["seed"]), (0.2, 7))

    def test_malformed_conversation_is_not_reported_as_an_api_error(self) -> None:
        for provider in ("anthropic", "openai_compat"):
            llm, sdk = self.client(provider)
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                llm.complete("SYSTEM", "USER", messages=[{"role": "assistant", "content": "a"}])
            self.assertEqual(sdk.requests, [])

    def test_missing_sdk_is_a_configuration_error(self) -> None:
        env = {"ANTHROPIC_API_KEY": FAKE_KEY, "LLM_BASE_URL": "http://mock.invalid/v1"}
        for provider, module in (("anthropic", "anthropic"), ("openai_compat", "openai")):
            with self.subTest(provider=provider), mock.patch.dict(sys.modules, {module: None}), mock.patch.dict(os.environ, env):
                with self.assertRaises(llm_client.LLMClientError) as caught:
                    LLMClient(_cfg(provider))
                self.assertIn(module, str(caught.exception))

    def single_turn_request(self, provider: str) -> dict[str, Any]:
        llm, sdk = self.client(provider)
        llm.complete("SYSTEM", "USER")
        return sdk.requests[0]


# ----------------------------------------------------------------------------- openai_compat: SDK 의 실제 HTTP 경로


def _httpx_of(sdk: Any) -> Any:
    """SDK 가 쓰는 HTTP 라이브러리 모듈(``httpx`` 또는 ``httpx2``): 그 SDK 의 ``DefaultHttpxClient`` 의 부모 클래스가 사는 패키지."""
    for base in sdk.DefaultHttpxClient.__mro__[1:]:
        package = base.__module__.split(".")[0]
        if package.startswith("httpx"):
            return importlib.import_module(package)
    raise AssertionError(f"{sdk.__name__}.DefaultHttpxClient 가 httpx 계열 클라이언트가 아니다")


@requires_modules("openai")
class OpenAICompatMockTest(unittest.TestCase):
    """``complete()`` 를 openai SDK 의 실제 요청 경로(MockTransport)로 통과시킨다."""

    COMPLETION = {
        "id": "chatcmpl-mock", "object": "chat.completion", "created": 0, "model": "served-open-model",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "REPLY"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 21, "completion_tokens": 9, "total_tokens": 30},
    }

    def complete(self, **kwargs: Any) -> tuple[Any, list[dict[str, Any]]]:
        http = _httpx_of(openai)
        requests: list[dict[str, Any]] = []

        def handler(request: Any) -> Any:
            requests.append({"url": str(request.url), "json": json.loads(request.content)})
            return http.Response(200, headers={"content-type": "application/json", "x-request-id": REQUEST_ID},
                                 content=json.dumps(self.COMPLETION).encode())

        cfg = _cfg("openai_compat", name="open-mock", max_retries=0, request_timeout_s=5.0)
        with mock.patch.dict(os.environ, {"LLM_BASE_URL": "http://mock.invalid/v1", "OPENAI_API_KEY": FAKE_KEY}):
            llm = LLMClient(cfg)
        llm._client = openai.OpenAI(base_url="http://mock.invalid/v1", api_key=FAKE_KEY, max_retries=0, timeout=5.0,
                                    http_client=openai.DefaultHttpxClient(transport=http.MockTransport(handler)))
        return llm.complete("SYSTEM PROMPT", "USER PROMPT", **kwargs), requests

    def test_single_turn_request_and_response(self) -> None:
        resp, requests = self.complete()
        self.assertIsNone(resp.error)
        self.assertEqual(len(requests), 1)
        self.assertTrue(requests[0]["url"].endswith("/v1/chat/completions"))
        body = requests[0]["json"]
        self.assertEqual(body["model"], "open-mock")
        self.assertEqual(body["messages"], [{"role": "system", "content": "SYSTEM PROMPT"}, {"role": "user", "content": "USER PROMPT"}])
        self.assertEqual((resp.text, resp.model_served, resp.stop_reason, resp.raw_stop_reason), ("REPLY", "served-open-model", "end_turn", "stop"))
        self.assertEqual((resp.tokens_in, resp.tokens_out, resp.tokens_thinking), (21, 9, None))

    def test_messages_follow_the_system_message(self) -> None:
        resp, requests = self.complete(messages=REPAIR_MESSAGES)
        self.assertIsNone(resp.error)
        body = requests[0]["json"]
        self.assertEqual(body["messages"], [{"role": "system", "content": "SYSTEM PROMPT"}, *REPAIR_MESSAGES])
        self.assertNotIn("USER PROMPT", json.dumps(body))


if __name__ == "__main__":
    unittest.main()
