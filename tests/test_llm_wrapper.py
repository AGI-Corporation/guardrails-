"""
Tests for llm_wrapper.py
"""

import pytest

from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
)
from llm_wrapper import (
    GuardedLLM,
    GuardedLLMResult,
    LLMRequest,
    LLMResponse,
    MockLLMProvider,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _engine_with_block_keyword(keyword: str) -> GuardrailEngine:
    engine = GuardrailEngine()
    engine.add_rule(
        GuardrailRule(
            id="block_rule",
            name="Block Rule",
            category=GuardrailCategory.HARMFUL_CONTENT,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            keywords=[keyword],
        )
    )
    return engine


def _clean_engine() -> GuardrailEngine:
    return GuardrailEngine()


# ---------------------------------------------------------------------------
# LLMRequest dataclass
# ---------------------------------------------------------------------------

class TestLLMRequest:
    def test_defaults(self):
        req = LLMRequest(prompt="hello")
        assert req.model == "default"
        assert req.temperature == 0.7
        assert req.max_tokens == 1000
        assert req.metadata == {}

    def test_custom_values(self):
        req = LLMRequest(
            prompt="test prompt",
            model="gpt-4",
            temperature=0.0,
            max_tokens=512,
            metadata={"user": "alice"},
        )
        assert req.model == "gpt-4"
        assert req.temperature == 0.0
        assert req.metadata["user"] == "alice"


# ---------------------------------------------------------------------------
# LLMResponse dataclass
# ---------------------------------------------------------------------------

class TestLLMResponse:
    def test_defaults(self):
        resp = LLMResponse(text="hello world", model="test")
        assert resp.tokens_used == 0
        assert resp.latency_ms == 0
        assert resp.finish_reason == "complete"


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------

class TestMockLLMProvider:
    def test_returns_configured_text(self):
        provider = MockLLMProvider(response_text="Mock reply")
        req = LLMRequest(prompt="Say something")
        resp = provider.complete(req)
        assert resp.text == "Mock reply"

    def test_model_echoed_from_request(self):
        provider = MockLLMProvider()
        req = LLMRequest(prompt="hello", model="my-model")
        resp = provider.complete(req)
        assert resp.model == "my-model"

    def test_tokens_used_positive(self):
        provider = MockLLMProvider(response_text="This is a response with words")
        req = LLMRequest(prompt="short prompt")
        resp = provider.complete(req)
        assert resp.tokens_used > 0

    def test_call_count_increments(self):
        provider = MockLLMProvider()
        assert provider.call_count == 0
        provider.complete(LLMRequest(prompt="a"))
        provider.complete(LLMRequest(prompt="b"))
        assert provider.call_count == 2

    def test_default_response_text(self):
        provider = MockLLMProvider()
        resp = provider.complete(LLMRequest(prompt="hello"))
        assert resp.text == "This is a mock response."


# ---------------------------------------------------------------------------
# GuardedLLM — pass-through (no violations)
# ---------------------------------------------------------------------------

class TestGuardedLLMPassThrough:
    def test_clean_request_and_response_passes(self):
        engine = _clean_engine()
        provider = MockLLMProvider(response_text="A safe response.")
        guarded = GuardedLLM(provider=provider, engine=engine)
        req = LLMRequest(prompt="Tell me about the weather.")
        result = guarded.complete(req)
        assert result.blocked is False
        assert result.input_action == "allow"
        assert result.output_action == "allow"
        assert result.response is not None
        assert result.response.text == "A safe response."

    def test_stats_updated_on_pass(self):
        engine = _clean_engine()
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=engine)
        guarded.complete(LLMRequest(prompt="hello"))
        stats = guarded.get_stats()
        assert stats["total"] == 1
        assert stats["passed"] == 1
        assert stats["input_blocked"] == 0
        assert stats["output_blocked"] == 0


# ---------------------------------------------------------------------------
# GuardedLLM — input blocking
# ---------------------------------------------------------------------------

class TestGuardedLLMInputBlocking:
    def test_blocked_input_returns_no_response(self):
        engine = _engine_with_block_keyword("forbidden")
        provider = MockLLMProvider()
        guarded = GuardedLLM(provider=provider, engine=engine)
        result = guarded.complete(LLMRequest(prompt="This is forbidden content"))
        assert result.blocked is True
        assert result.input_action == "block"
        assert result.response is None

    def test_blocked_input_reason_contains_rule_id(self):
        engine = _engine_with_block_keyword("forbidden")
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=engine)
        result = guarded.complete(LLMRequest(prompt="forbidden text"))
        assert "block_rule" in result.block_reason

    def test_blocked_input_provider_not_called(self):
        engine = _engine_with_block_keyword("forbidden")
        provider = MockLLMProvider()
        guarded = GuardedLLM(provider=provider, engine=engine)
        guarded.complete(LLMRequest(prompt="forbidden"))
        assert provider.call_count == 0

    def test_stats_input_blocked_increments(self):
        engine = _engine_with_block_keyword("forbidden")
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=engine)
        guarded.complete(LLMRequest(prompt="forbidden"))
        guarded.complete(LLMRequest(prompt="forbidden"))
        stats = guarded.get_stats()
        assert stats["input_blocked"] == 2
        assert stats["total"] == 2


# ---------------------------------------------------------------------------
# GuardedLLM — output blocking
# ---------------------------------------------------------------------------

class TestGuardedLLMOutputBlocking:
    def test_blocked_output_returns_no_response(self):
        engine = _engine_with_block_keyword("dangerous_output")
        provider = MockLLMProvider(response_text="This contains dangerous_output here")
        guarded = GuardedLLM(provider=provider, engine=engine)
        result = guarded.complete(LLMRequest(prompt="Safe input prompt"))
        assert result.blocked is True
        assert result.output_action == "block"
        assert result.input_action == "allow"
        assert result.response is None

    def test_output_blocked_matched_rules_populated(self):
        engine = _engine_with_block_keyword("dangerous_output")
        provider = MockLLMProvider(response_text="dangerous_output here")
        guarded = GuardedLLM(provider=provider, engine=engine)
        result = guarded.complete(LLMRequest(prompt="hello"))
        assert "block_rule" in result.output_matched_rules

    def test_stats_output_blocked_increments(self):
        engine = _engine_with_block_keyword("bad_output")
        provider = MockLLMProvider(response_text="bad_output content")
        guarded = GuardedLLM(provider=provider, engine=engine)
        guarded.complete(LLMRequest(prompt="hello"))
        stats = guarded.get_stats()
        assert stats["output_blocked"] == 1


# ---------------------------------------------------------------------------
# GuardedLLM — get_stats
# ---------------------------------------------------------------------------

class TestGuardedLLMStats:
    def test_block_rate_zero_when_nothing_blocked(self):
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=_clean_engine())
        guarded.complete(LLMRequest(prompt="hello"))
        stats = guarded.get_stats()
        assert stats["block_rate"] == 0

    def test_block_rate_100_when_all_blocked(self):
        engine = _engine_with_block_keyword("blocked")
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=engine)
        guarded.complete(LLMRequest(prompt="blocked text"))
        stats = guarded.get_stats()
        assert stats["block_rate"] == 100.0

    def test_block_rate_with_zero_calls(self):
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=_clean_engine())
        stats = guarded.get_stats()
        assert stats["block_rate"] == 0

    def test_latency_ms_positive(self):
        guarded = GuardedLLM(provider=MockLLMProvider(), engine=_clean_engine())
        result = guarded.complete(LLMRequest(prompt="hello"))
        assert result.latency_ms >= 0


# ---------------------------------------------------------------------------
# OpenAIProvider / AnthropicProvider — ImportError path
# ---------------------------------------------------------------------------

class TestOpenAIProviderImportError:
    def test_openai_raises_runtime_error_without_package(self):
        """Cover the ImportError path in OpenAIProvider.complete."""
        from llm_wrapper import OpenAIProvider
        import sys
        # Temporarily hide the openai module if installed
        original = sys.modules.get("openai", None)
        sys.modules["openai"] = None  # type: ignore
        try:
            provider = OpenAIProvider(api_key="fake_key")
            with pytest.raises((RuntimeError, ImportError)):
                provider.complete(LLMRequest(prompt="hello"))
        finally:
            if original is None:
                del sys.modules["openai"]
            else:
                sys.modules["openai"] = original


class TestAnthropicProviderImportError:
    def test_anthropic_raises_runtime_error_without_package(self):
        """Cover the ImportError path in AnthropicProvider.complete."""
        from llm_wrapper import AnthropicProvider
        import sys
        original = sys.modules.get("anthropic", None)
        sys.modules["anthropic"] = None  # type: ignore
        try:
            provider = AnthropicProvider(api_key="fake_key")
            with pytest.raises((RuntimeError, ImportError)):
                provider.complete(LLMRequest(prompt="hello"))
        finally:
            if original is None:
                del sys.modules["anthropic"]
            else:
                sys.modules["anthropic"] = original


# ---------------------------------------------------------------------------
# GuardedLLMResult dataclass
# ---------------------------------------------------------------------------

class TestGuardedLLMResult:
    def test_defaults(self):
        req = LLMRequest(prompt="test")
        result = GuardedLLMResult(
            request=req,
            response=None,
            input_action="allow",
            output_action="allow",
            input_matched_rules=[],
            output_matched_rules=[],
            blocked=False,
        )
        assert result.block_reason == ""
        assert result.latency_ms == 0.0
