"""
Tests for llm_wrapper.py
Covers: MockLLMProvider, GuardedLLM input/output blocking, stats tracking,
        LLMRequest and LLMResponse data classes.
"""

import pytest

from llm_wrapper import (
    GuardedLLM,
    GuardedLLMResult,
    LLMRequest,
    LLMResponse,
    MockLLMProvider,
)
from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
    create_default_guardrails,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_with_block_keyword(keyword="BLOCKED_WORD"):
    engine = GuardrailEngine()
    rule = GuardrailRule(
        id="test_block",
        name="Test Block",
        category=GuardrailCategory.CUSTOM,
        severity=Severity.HIGH,
        action=Action.BLOCK,
        keywords=[keyword],
    )
    engine.add_rule(rule)
    return engine


def _make_permissive_engine():
    """Engine with no rules — allows everything."""
    return GuardrailEngine()


# ---------------------------------------------------------------------------
# MockLLMProvider
# ---------------------------------------------------------------------------

class TestMockLLMProvider:
    def test_returns_llm_response(self):
        provider = MockLLMProvider("hello world")
        request = LLMRequest(prompt="Say hello")
        response = provider.complete(request)
        assert isinstance(response, LLMResponse)

    def test_response_text_matches_configured(self):
        provider = MockLLMProvider("custom response")
        response = provider.complete(LLMRequest(prompt="anything"))
        assert response.text == "custom response"

    def test_call_count_increments(self):
        provider = MockLLMProvider()
        assert provider.call_count == 0
        provider.complete(LLMRequest(prompt="p1"))
        provider.complete(LLMRequest(prompt="p2"))
        assert provider.call_count == 2

    def test_model_echoed(self):
        provider = MockLLMProvider()
        req = LLMRequest(prompt="hi", model="gpt-test")
        resp = provider.complete(req)
        assert resp.model == "gpt-test"

    def test_tokens_used_positive(self):
        provider = MockLLMProvider("short response")
        resp = provider.complete(LLMRequest(prompt="hello world"))
        assert resp.tokens_used > 0

    def test_default_response_text(self):
        provider = MockLLMProvider()
        resp = provider.complete(LLMRequest(prompt="test"))
        assert resp.text  # non-empty


# ---------------------------------------------------------------------------
# LLMRequest / LLMResponse data classes
# ---------------------------------------------------------------------------

class TestLLMDataClasses:
    def test_llm_request_defaults(self):
        req = LLMRequest(prompt="hello")
        assert req.model == "default"
        assert req.temperature == 0.7
        assert req.max_tokens == 1000
        assert req.metadata == {}

    def test_llm_response_defaults(self):
        resp = LLMResponse(text="hi", model="gpt-4")
        assert resp.tokens_used == 0
        assert resp.latency_ms == 0
        assert resp.finish_reason == "complete"

    def test_guarded_llm_result_blocked_default_is_false(self):
        req = LLMRequest(prompt="hi")
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


# ---------------------------------------------------------------------------
# GuardedLLM — safe prompt passes through
# ---------------------------------------------------------------------------

class TestGuardedLLMAllowPath:
    def setup_method(self):
        self.engine = _make_permissive_engine()
        self.provider = MockLLMProvider("safe response")
        self.llm = GuardedLLM(self.provider, self.engine)

    def test_safe_prompt_not_blocked(self):
        req = LLMRequest(prompt="What is the capital of France?")
        result = self.llm.complete(req)
        assert result.blocked is False

    def test_safe_prompt_has_response(self):
        req = LLMRequest(prompt="Hello")
        result = self.llm.complete(req)
        assert result.response is not None
        assert result.response.text == "safe response"

    def test_safe_prompt_actions(self):
        req = LLMRequest(prompt="Hello")
        result = self.llm.complete(req)
        assert result.input_action == "allow"
        assert result.output_action == "allow"

    def test_safe_prompt_no_matched_rules(self):
        req = LLMRequest(prompt="Hello")
        result = self.llm.complete(req)
        assert result.input_matched_rules == []
        assert result.output_matched_rules == []

    def test_provider_called_once_on_safe_prompt(self):
        req = LLMRequest(prompt="Hello")
        self.llm.complete(req)
        assert self.provider.call_count == 1


# ---------------------------------------------------------------------------
# GuardedLLM — input blocking
# ---------------------------------------------------------------------------

class TestGuardedLLMInputBlocking:
    def setup_method(self):
        self.engine = _make_engine_with_block_keyword("BLOCKED_WORD")
        self.provider = MockLLMProvider("response")
        self.llm = GuardedLLM(self.provider, self.engine)

    def test_blocked_input_returns_blocked(self):
        req = LLMRequest(prompt="This contains BLOCKED_WORD in it")
        result = self.llm.complete(req)
        assert result.blocked is True

    def test_blocked_input_action_is_block(self):
        req = LLMRequest(prompt="BLOCKED_WORD here")
        result = self.llm.complete(req)
        assert result.input_action == "block"

    def test_blocked_input_response_is_none(self):
        req = LLMRequest(prompt="BLOCKED_WORD here")
        result = self.llm.complete(req)
        assert result.response is None

    def test_blocked_input_has_matched_rules(self):
        req = LLMRequest(prompt="BLOCKED_WORD here")
        result = self.llm.complete(req)
        assert "test_block" in result.input_matched_rules

    def test_blocked_input_has_block_reason(self):
        req = LLMRequest(prompt="BLOCKED_WORD here")
        result = self.llm.complete(req)
        assert result.block_reason  # non-empty

    def test_llm_not_called_when_input_blocked(self):
        req = LLMRequest(prompt="BLOCKED_WORD here")
        self.llm.complete(req)
        assert self.provider.call_count == 0


# ---------------------------------------------------------------------------
# GuardedLLM — output blocking
# ---------------------------------------------------------------------------

class TestGuardedLLMOutputBlocking:
    def setup_method(self):
        # Engine blocks "badoutput" keyword
        self.engine = _make_engine_with_block_keyword("badoutput")
        # Provider always returns text containing the blocked keyword
        self.provider = MockLLMProvider("This is badoutput content")
        self.llm = GuardedLLM(self.provider, self.engine)

    def test_blocked_output_returns_blocked(self):
        req = LLMRequest(prompt="A safe question")
        result = self.llm.complete(req)
        assert result.blocked is True

    def test_blocked_output_action(self):
        req = LLMRequest(prompt="A safe question")
        result = self.llm.complete(req)
        assert result.output_action == "block"
        assert result.input_action == "allow"

    def test_blocked_output_response_is_none(self):
        req = LLMRequest(prompt="A safe question")
        result = self.llm.complete(req)
        assert result.response is None

    def test_blocked_output_has_matched_rules(self):
        req = LLMRequest(prompt="A safe question")
        result = self.llm.complete(req)
        assert "test_block" in result.output_matched_rules


# ---------------------------------------------------------------------------
# GuardedLLM — statistics
# ---------------------------------------------------------------------------

class TestGuardedLLMStats:
    def test_initial_stats_zeroed(self):
        engine = _make_permissive_engine()
        llm = GuardedLLM(MockLLMProvider(), engine)
        stats = llm.get_stats()
        assert stats["total"] == 0
        assert stats["input_blocked"] == 0
        assert stats["output_blocked"] == 0
        assert stats["passed"] == 0
        assert stats["block_rate"] == 0

    def test_stats_after_safe_request(self):
        engine = _make_permissive_engine()
        llm = GuardedLLM(MockLLMProvider("safe"), engine)
        llm.complete(LLMRequest(prompt="hello"))
        stats = llm.get_stats()
        assert stats["total"] == 1
        assert stats["passed"] == 1
        assert stats["input_blocked"] == 0

    def test_stats_after_input_block(self):
        engine = _make_engine_with_block_keyword("BAD")
        llm = GuardedLLM(MockLLMProvider(), engine)
        llm.complete(LLMRequest(prompt="BAD content"))
        stats = llm.get_stats()
        assert stats["total"] == 1
        assert stats["input_blocked"] == 1
        assert stats["passed"] == 0

    def test_stats_block_rate(self):
        engine = _make_engine_with_block_keyword("BAD")
        llm = GuardedLLM(MockLLMProvider("safe"), engine)
        llm.complete(LLMRequest(prompt="BAD"))  # blocked
        llm.complete(LLMRequest(prompt="good"))  # allowed
        stats = llm.get_stats()
        assert stats["total"] == 2
        assert stats["block_rate"] == 50.0

    def test_latency_ms_positive(self):
        engine = _make_permissive_engine()
        llm = GuardedLLM(MockLLMProvider("hi"), engine)
        result = llm.complete(LLMRequest(prompt="hello"))
        assert result.latency_ms >= 0
