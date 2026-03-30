"""
Tests for llm_wrapper.py
Covers LLMRequest, LLMResponse, GuardedLLMResult, MockLLMProvider,
and GuardedLLM.

GuardedLLM uses engine.evaluate() expecting an object with .action,
.matched_rules attributes (not the dict returned by GuardrailEngine).
We use a MockEngine to simulate this interface in tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from llm_wrapper import (
    LLMRequest,
    LLMResponse,
    GuardedLLMResult,
    LLMProvider,
    MockLLMProvider,
    GuardedLLM,
    OpenAIProvider,
    AnthropicProvider,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_allow_result(matched_rules=None):
    """Simulate engine.evaluate() returning an 'allow' decision."""
    return SimpleNamespace(
        action="allow",
        matched_rules=matched_rules or [],
        severity="",
    )


def make_block_result(matched_rules=None):
    """Simulate engine.evaluate() returning a 'block' decision."""
    return SimpleNamespace(
        action="block",
        matched_rules=matched_rules or ["rule_1"],
        severity="high",
    )


class MockEngine:
    """Mock guardrail engine whose evaluate() returns attribute-based results."""

    def __init__(self, input_result=None, output_result=None):
        self._input_result = input_result or make_allow_result()
        self._output_result = output_result or make_allow_result()
        self._call_count = 0

    def evaluate(self, text: str):
        self._call_count += 1
        # First call = input, second call = output
        if self._call_count == 1:
            return self._input_result
        return self._output_result


# ── LLMRequest ────────────────────────────────────────────────────────────────

class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(prompt="hello")
        assert req.model == "default"
        assert req.temperature == pytest.approx(0.7)
        assert req.max_tokens == 1000
        assert req.metadata == {}

    def test_custom_values(self):
        req = LLMRequest(
            prompt="test",
            model="gpt-4",
            temperature=0.0,
            max_tokens=500,
            metadata={"user": "alice"},
        )
        assert req.prompt == "test"
        assert req.model == "gpt-4"
        assert req.temperature == pytest.approx(0.0)
        assert req.max_tokens == 500
        assert req.metadata == {"user": "alice"}


# ── LLMResponse ───────────────────────────────────────────────────────────────

class TestLLMResponse:
    def test_defaults(self):
        resp = LLMResponse(text="hello", model="gpt-4")
        assert resp.tokens_used == 0
        assert resp.latency_ms == pytest.approx(0.0)
        assert resp.finish_reason == "complete"

    def test_custom_values(self):
        resp = LLMResponse(
            text="some text",
            model="claude-3",
            tokens_used=42,
            latency_ms=123.4,
            finish_reason="stop",
        )
        assert resp.text == "some text"
        assert resp.tokens_used == 42
        assert resp.latency_ms == pytest.approx(123.4)


# ── MockLLMProvider ───────────────────────────────────────────────────────────

class TestMockLLMProvider:
    def test_complete_returns_llm_response(self):
        provider = MockLLMProvider("test response")
        request = LLMRequest(prompt="hello")
        response = provider.complete(request)
        assert isinstance(response, LLMResponse)

    def test_complete_uses_configured_text(self):
        provider = MockLLMProvider("my mock text")
        request = LLMRequest(prompt="hello")
        response = provider.complete(request)
        assert response.text == "my mock text"

    def test_complete_sets_model_from_request(self):
        provider = MockLLMProvider()
        request = LLMRequest(prompt="hello", model="test-model")
        response = provider.complete(request)
        assert response.model == "test-model"

    def test_complete_increments_call_count(self):
        provider = MockLLMProvider()
        request = LLMRequest(prompt="hello")
        provider.complete(request)
        provider.complete(request)
        assert provider.call_count == 2

    def test_complete_sets_tokens_used(self):
        provider = MockLLMProvider("response text")
        request = LLMRequest(prompt="input prompt")
        response = provider.complete(request)
        assert response.tokens_used > 0

    def test_complete_sets_latency(self):
        provider = MockLLMProvider()
        request = LLMRequest(prompt="hello")
        response = provider.complete(request)
        assert response.latency_ms > 0

    def test_default_response_text(self):
        provider = MockLLMProvider()
        request = LLMRequest(prompt="hello")
        response = provider.complete(request)
        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_is_llm_provider(self):
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)


# ── GuardedLLM ────────────────────────────────────────────────────────────────

class TestGuardedLLMPassThrough:
    def setup_method(self):
        self.provider = MockLLMProvider("safe response")
        self.engine = MockEngine(
            input_result=make_allow_result(),
            output_result=make_allow_result(),
        )
        self.guarded = GuardedLLM(self.provider, self.engine)

    def test_allowed_request_not_blocked(self):
        request = LLMRequest(prompt="hello there")
        result = self.guarded.complete(request)
        assert result.blocked is False

    def test_allowed_request_has_response(self):
        request = LLMRequest(prompt="hello")
        result = self.guarded.complete(request)
        assert result.response is not None
        assert result.response.text == "safe response"

    def test_allowed_input_and_output_actions(self):
        request = LLMRequest(prompt="hello")
        result = self.guarded.complete(request)
        assert result.input_action == "allow"
        assert result.output_action == "allow"

    def test_allowed_request_increments_passed_stat(self):
        self.guarded.complete(LLMRequest(prompt="hello"))
        stats = self.guarded.get_stats()
        assert stats["passed"] == 1
        assert stats["input_blocked"] == 0
        assert stats["output_blocked"] == 0

    def test_latency_ms_set(self):
        result = self.guarded.complete(LLMRequest(prompt="hello"))
        assert result.latency_ms >= 0


class TestGuardedLLMInputBlocked:
    def setup_method(self):
        self.provider = MockLLMProvider("safe response")
        self.engine = MockEngine(
            input_result=make_block_result(["ssn_rule"]),
            output_result=make_allow_result(),
        )
        self.guarded = GuardedLLM(self.provider, self.engine)

    def test_blocked_input_returns_blocked_result(self):
        result = self.guarded.complete(LLMRequest(prompt="my SSN is 123-45-6789"))
        assert result.blocked is True

    def test_blocked_input_has_no_response(self):
        result = self.guarded.complete(LLMRequest(prompt="SSN input"))
        assert result.response is None

    def test_blocked_input_has_block_reason(self):
        result = self.guarded.complete(LLMRequest(prompt="SSN input"))
        assert result.block_reason != ""

    def test_blocked_input_action_is_block(self):
        result = self.guarded.complete(LLMRequest(prompt="SSN input"))
        assert result.input_action == "block"

    def test_blocked_input_increments_stat(self):
        self.guarded.complete(LLMRequest(prompt="SSN input"))
        stats = self.guarded.get_stats()
        assert stats["input_blocked"] == 1
        assert stats["passed"] == 0

    def test_blocked_input_provider_not_called(self):
        self.guarded.complete(LLMRequest(prompt="SSN input"))
        assert self.provider.call_count == 0

    def test_blocked_input_matched_rules(self):
        result = self.guarded.complete(LLMRequest(prompt="SSN input"))
        assert "ssn_rule" in result.input_matched_rules


class TestGuardedLLMOutputBlocked:
    def setup_method(self):
        self.provider = MockLLMProvider("blocked output text")
        self.engine = MockEngine(
            input_result=make_allow_result(),
            output_result=make_block_result(["output_rule"]),
        )
        self.guarded = GuardedLLM(self.provider, self.engine)

    def test_blocked_output_returns_blocked_result(self):
        result = self.guarded.complete(LLMRequest(prompt="safe input"))
        assert result.blocked is True

    def test_blocked_output_has_no_response(self):
        result = self.guarded.complete(LLMRequest(prompt="safe input"))
        assert result.response is None

    def test_blocked_output_input_action_is_allow(self):
        result = self.guarded.complete(LLMRequest(prompt="safe input"))
        assert result.input_action == "allow"
        assert result.output_action == "block"

    def test_blocked_output_increments_stat(self):
        self.guarded.complete(LLMRequest(prompt="safe input"))
        stats = self.guarded.get_stats()
        assert stats["output_blocked"] == 1

    def test_blocked_output_provider_was_called(self):
        self.guarded.complete(LLMRequest(prompt="safe input"))
        assert self.provider.call_count == 1


class TestGuardedLLMStats:
    def setup_method(self):
        self.allow_engine = MockEngine(make_allow_result(), make_allow_result())
        self.provider = MockLLMProvider()
        self.guarded = GuardedLLM(self.provider, self.allow_engine)

    def test_initial_stats_all_zero(self):
        stats = self.guarded.get_stats()
        assert stats["total"] == 0
        assert stats["passed"] == 0
        assert stats["input_blocked"] == 0
        assert stats["output_blocked"] == 0
        assert stats["block_rate"] == 0

    def test_block_rate_zero_when_all_pass(self):
        self.guarded.complete(LLMRequest(prompt="hello"))
        stats = self.guarded.get_stats()
        assert stats["block_rate"] == pytest.approx(0.0)

    def test_total_increments_per_call(self):
        self.guarded.complete(LLMRequest(prompt="a"))
        self.guarded.complete(LLMRequest(prompt="b"))
        stats = self.guarded.get_stats()
        assert stats["total"] == 2

    def test_block_rate_calculation(self):
        # Need a new engine that alternates allow/block
        block_engine = MockEngine(make_block_result(), make_allow_result())
        guarded = GuardedLLM(MockLLMProvider(), block_engine)
        guarded.complete(LLMRequest(prompt="bad input"))
        stats = guarded.get_stats()
        # 1 input blocked out of 1 total = 100%
        assert stats["block_rate"] == pytest.approx(100.0)


# ── OpenAIProvider ────────────────────────────────────────────────────────────

class TestOpenAIProvider:
    def test_init_stores_api_key_and_model(self):
        provider = OpenAIProvider(api_key="sk-test", model="gpt-4")
        assert provider.api_key == "sk-test"
        assert provider.model == "gpt-4"

    def test_init_default_model(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.model == "gpt-4o-mini"

    def test_is_llm_provider(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert isinstance(provider, LLMProvider)

    def test_complete_raises_runtime_error_without_openai(self):
        provider = OpenAIProvider(api_key="sk-test")
        request = LLMRequest(prompt="hello")
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises((RuntimeError, ImportError)):
                provider.complete(request)


# ── AnthropicProvider ─────────────────────────────────────────────────────────

class TestAnthropicProvider:
    def test_init_stores_api_key_and_model(self):
        provider = AnthropicProvider(api_key="sk-ant-test", model="claude-3-opus")
        assert provider.api_key == "sk-ant-test"
        assert provider.model == "claude-3-opus"

    def test_init_default_model(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert provider.model == "claude-3-haiku-20240307"

    def test_is_llm_provider(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        assert isinstance(provider, LLMProvider)

    def test_complete_raises_runtime_error_without_anthropic(self):
        provider = AnthropicProvider(api_key="sk-ant-test")
        request = LLMRequest(prompt="hello")
        with patch.dict("sys.modules", {"anthropic": None}):
            with pytest.raises((RuntimeError, ImportError)):
                provider.complete(request)
