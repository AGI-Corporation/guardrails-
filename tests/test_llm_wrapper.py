"""
Tests for llm_wrapper.py
Covers: MockLLMProvider, GuardedLLM (pass-through, input block, output block),
        stats tracking.
"""

import pytest

from llm_wrapper import (
    GuardedLLM,
    LLMRequest,
    LLMResponse,
    MockLLMProvider,
    GuardedLLMResult,
)
from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    GuardrailCategory,
    Severity,
    Action,
    create_default_guardrails,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def safe_engine():
    """Engine with default guardrails (PII, self-harm, etc.)."""
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine


@pytest.fixture
def mock_provider():
    return MockLLMProvider("This is a safe mock response.")


@pytest.fixture
def guarded_llm(safe_engine, mock_provider):
    return GuardedLLM(provider=mock_provider, engine=safe_engine)


# ── MockLLMProvider ───────────────────────────────────────────────────────

class TestMockLLMProvider:

    def test_returns_response_with_text(self, mock_provider):
        req = LLMRequest(prompt="Say hello")
        resp = mock_provider.complete(req)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "This is a safe mock response."

    def test_call_count_increments(self, mock_provider):
        req = LLMRequest(prompt="test")
        mock_provider.complete(req)
        mock_provider.complete(req)
        assert mock_provider.call_count == 2

    def test_tokens_used_positive(self, mock_provider):
        req = LLMRequest(prompt="Hello world")
        resp = mock_provider.complete(req)
        assert resp.tokens_used > 0


# ── GuardedLLM — safe pass-through ───────────────────────────────────────

class TestGuardedLLMSafeRequest:

    def test_safe_prompt_passes_through(self, guarded_llm):
        req = LLMRequest(prompt="Tell me about Python programming")
        result = guarded_llm.complete(req)
        assert not result.blocked
        assert result.response is not None
        assert result.response.text == "This is a safe mock response."
        assert result.input_action == "allow"
        assert result.output_action == "allow"

    def test_stats_increment_on_pass(self, guarded_llm):
        req = LLMRequest(prompt="Hello world")
        guarded_llm.complete(req)
        stats = guarded_llm.get_stats()
        assert stats["total"] == 1
        assert stats["passed"] == 1
        assert stats["input_blocked"] == 0
        assert stats["output_blocked"] == 0


# ── GuardedLLM — input blocked ────────────────────────────────────────────

class TestGuardedLLMInputBlock:

    def test_ssn_input_is_blocked(self, guarded_llm):
        req = LLMRequest(prompt="My SSN is 123-45-6789, help me")
        result = guarded_llm.complete(req)
        assert result.blocked
        assert result.input_action == "block"
        assert result.response is None
        assert result.block_reason

    def test_credit_card_input_blocked(self, guarded_llm):
        req = LLMRequest(prompt="Card: 4111 1111 1111 1111")
        result = guarded_llm.complete(req)
        assert result.blocked
        assert result.input_action == "block"

    def test_input_blocked_increments_stats(self, guarded_llm):
        req = LLMRequest(prompt="SSN: 123-45-6789")
        guarded_llm.complete(req)
        stats = guarded_llm.get_stats()
        assert stats["input_blocked"] == 1
        assert stats["passed"] == 0

    def test_mock_provider_not_called_when_input_blocked(self, guarded_llm, mock_provider):
        req = LLMRequest(prompt="SSN: 123-45-6789")
        guarded_llm.complete(req)
        assert mock_provider.call_count == 0


# ── GuardedLLM — output blocked ───────────────────────────────────────────

class TestGuardedLLMOutputBlock:

    def test_output_block_when_llm_returns_harmful(self, safe_engine):
        """Provider returns text with SSN; output should be blocked."""
        bad_provider = MockLLMProvider("Here is an SSN: 123-45-6789")
        llm = GuardedLLM(provider=bad_provider, engine=safe_engine)
        result = llm.complete(LLMRequest(prompt="Tell me something"))
        assert result.blocked
        assert result.output_action == "block"
        assert result.response is None

    def test_output_blocked_increments_stats(self, safe_engine):
        bad_provider = MockLLMProvider("SSN: 123-45-6789")
        llm = GuardedLLM(provider=bad_provider, engine=safe_engine)
        llm.complete(LLMRequest(prompt="hello"))
        stats = llm.get_stats()
        assert stats["output_blocked"] == 1


# ── get_stats ─────────────────────────────────────────────────────────────

class TestGuardedLLMStats:

    def test_block_rate_calculation(self, guarded_llm):
        # 1 blocked + 1 passed
        guarded_llm.complete(LLMRequest(prompt="Hello"))
        guarded_llm.complete(LLMRequest(prompt="SSN: 123-45-6789"))
        stats = guarded_llm.get_stats()
        assert stats["total"] == 2
        assert stats["block_rate"] == 50.0

    def test_zero_total_gives_zero_block_rate(self, guarded_llm):
        stats = guarded_llm.get_stats()
        assert stats["block_rate"] == 0

    def test_latency_non_negative(self, guarded_llm):
        result = guarded_llm.complete(LLMRequest(prompt="hi"))
        assert result.latency_ms >= 0
