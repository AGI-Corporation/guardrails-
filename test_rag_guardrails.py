"""
Tests for rag_guardrails.py
Covers RAGContext, RAGRequest, RAGGuardrailResult, RAGPipelineResult,
and RAGGuardrailProcessor.

RAGGuardrailProcessor uses engine.evaluate() expecting an object with
.action, .matched_rules, and .severity attributes (not the dict returned
by GuardrailEngine). We use a MockEngine to simulate this interface.
"""

import pytest
from types import SimpleNamespace

from rag_guardrails import (
    RAGStage,
    RAGContext,
    RAGRequest,
    RAGGuardrailResult,
    RAGPipelineResult,
    RAGGuardrailProcessor,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def allow_result(matched_rules=None):
    return SimpleNamespace(
        action="allow",
        matched_rules=matched_rules or [],
        severity="",
    )


def block_result(matched_rules=None):
    return SimpleNamespace(
        action="block",
        matched_rules=matched_rules or ["test_rule"],
        severity="high",
    )


class AlwaysAllowEngine:
    """Mock engine that always returns 'allow'."""
    def evaluate(self, text: str):
        return allow_result()


class AlwaysBlockEngine:
    """Mock engine that always returns 'block'."""
    def evaluate(self, text: str):
        return block_result()


class SelectiveEngine:
    """Blocks only text that contains a trigger word."""
    def __init__(self, trigger: str):
        self.trigger = trigger

    def evaluate(self, text: str):
        if self.trigger in text:
            return block_result(["trigger_rule"])
        return allow_result()


# ── RAGContext ─────────────────────────────────────────────────────────────────

class TestRAGContext:
    def test_defaults(self):
        ctx = RAGContext(content="hello", source="doc1")
        assert ctx.score == 0.0
        assert ctx.metadata == {}

    def test_custom_values(self):
        ctx = RAGContext(content="text", source="doc2", score=0.9,
                        metadata={"page": 1})
        assert ctx.score == pytest.approx(0.9)
        assert ctx.metadata == {"page": 1}


# ── RAGRequest ────────────────────────────────────────────────────────────────

class TestRAGRequest:
    def test_defaults(self):
        req = RAGRequest(query="What is AI?")
        assert req.contexts == []
        assert req.system_prompt == ""

    def test_custom_values(self):
        ctx = RAGContext(content="AI text", source="doc")
        req = RAGRequest(query="test", contexts=[ctx], system_prompt="You are helpful")
        assert len(req.contexts) == 1
        assert req.system_prompt == "You are helpful"


# ── RAGGuardrailResult ────────────────────────────────────────────────────────

class TestRAGGuardrailResult:
    def test_defaults(self):
        result = RAGGuardrailResult(
            stage=RAGStage.QUERY,
            action="allow",
            matched_rules=[],
            severity="",
            original_content="hello",
        )
        assert result.sanitized_content is None
        assert result.details == {}

    def test_custom_values(self):
        result = RAGGuardrailResult(
            stage=RAGStage.CONTEXT,
            action="block",
            matched_rules=["r1"],
            severity="high",
            original_content="bad content",
            sanitized_content="[REDACTED]",
            details={"source": "doc1"},
        )
        assert result.action == "block"
        assert result.sanitized_content == "[REDACTED]"
        assert result.details == {"source": "doc1"}


# ── RAGStage ──────────────────────────────────────────────────────────────────

class TestRAGStage:
    def test_stage_values(self):
        assert RAGStage.QUERY.value == "query"
        assert RAGStage.CONTEXT.value == "context"
        assert RAGStage.RESPONSE.value == "response"


# ── RAGGuardrailProcessor – check_query ──────────────────────────────────────

class TestRAGGuardrailProcessorCheckQuery:
    def test_safe_query_allows(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        result = processor.check_query("What is the capital of France?")
        assert result.action == "allow"
        assert result.stage == RAGStage.QUERY
        assert result.original_content == "What is the capital of France?"

    def test_blocked_query_blocks(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        result = processor.check_query("My SSN is 123-45-6789")
        assert result.action == "block"
        assert len(result.matched_rules) > 0

    def test_check_query_stage_is_query(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        result = processor.check_query("hello")
        assert result.stage == RAGStage.QUERY


# ── RAGGuardrailProcessor – check_contexts ───────────────────────────────────

class TestRAGGuardrailProcessorCheckContexts:
    def test_empty_contexts_returns_empty_list(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        results = processor.check_contexts([])
        assert results == []

    def test_all_safe_contexts_allowed(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        contexts = [
            RAGContext(content="safe text", source="doc1"),
            RAGContext(content="more safe text", source="doc2"),
        ]
        results = processor.check_contexts(contexts)
        assert all(r.action == "allow" for r in results)
        assert len(results) == 2

    def test_all_blocked_contexts(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        contexts = [
            RAGContext(content="bad content", source="doc1"),
        ]
        results = processor.check_contexts(contexts)
        assert results[0].action == "block"

    def test_mixed_contexts(self):
        processor = RAGGuardrailProcessor(SelectiveEngine("secret"))
        contexts = [
            RAGContext(content="safe text", source="doc1"),
            RAGContext(content="this contains secret", source="doc2"),
        ]
        results = processor.check_contexts(contexts)
        assert results[0].action == "allow"
        assert results[1].action == "block"

    def test_check_contexts_stage_is_context(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        contexts = [RAGContext(content="text", source="src")]
        results = processor.check_contexts(contexts)
        assert results[0].stage == RAGStage.CONTEXT

    def test_context_details_include_source(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        contexts = [RAGContext(content="text", source="my_source", score=0.8)]
        results = processor.check_contexts(contexts)
        assert results[0].details.get("source") == "my_source"
        assert results[0].details.get("score") == pytest.approx(0.8)


# ── RAGGuardrailProcessor – check_response ───────────────────────────────────

class TestRAGGuardrailProcessorCheckResponse:
    def test_safe_response_allowed(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        result = processor.check_response("This is a safe response.")
        assert result.action == "allow"
        assert result.stage == RAGStage.RESPONSE

    def test_blocked_response(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        result = processor.check_response("Blocked response text")
        assert result.action == "block"


# ── RAGGuardrailProcessor – process_request ──────────────────────────────────

class TestRAGGuardrailProcessorProcessRequest:
    def test_safe_query_and_contexts_allowed(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        contexts = [
            RAGContext(content="fact 1", source="doc1"),
            RAGContext(content="fact 2", source="doc2"),
        ]
        request = RAGRequest(query="Tell me about AI", contexts=contexts)
        result = processor.process_request(request)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 2

    def test_blocked_query_short_circuits(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        contexts = [RAGContext(content="context", source="doc")]
        request = RAGRequest(query="blocked query", contexts=contexts)
        result = processor.process_request(request)
        assert result.final_action == "block"
        assert result.safe_contexts == []
        assert result.context_checks == []

    def test_blocked_context_excluded_from_safe(self):
        processor = RAGGuardrailProcessor(SelectiveEngine("secret"))
        contexts = [
            RAGContext(content="safe context", source="doc1"),
            RAGContext(content="secret password here", source="doc2"),
        ]
        request = RAGRequest(query="safe query", contexts=contexts)
        result = processor.process_request(request)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 1
        assert result.safe_contexts[0].source == "doc1"

    def test_process_request_no_contexts(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        request = RAGRequest(query="safe query")
        result = processor.process_request(request)
        assert result.final_action == "allow"
        assert result.safe_contexts == []

    def test_blocked_query_has_blocked_reason(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        request = RAGRequest(query="bad query")
        result = processor.process_request(request)
        assert result.blocked_reason != ""

    def test_result_response_check_is_none_before_final_check(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        request = RAGRequest(query="safe query")
        result = processor.process_request(request)
        assert result.response_check is None


# ── RAGGuardrailProcessor – check_final_response ─────────────────────────────

class TestRAGGuardrailProcessorCheckFinalResponse:
    def _make_pipeline_result(self):
        query_check = RAGGuardrailResult(
            stage=RAGStage.QUERY,
            action="allow",
            matched_rules=[],
            severity="",
            original_content="query",
        )
        return RAGPipelineResult(
            query_check=query_check,
            context_checks=[],
            response_check=None,
            final_action="allow",
            safe_contexts=[],
        )

    def test_safe_response_keeps_allow(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_pipeline_result()
        updated = processor.check_final_response(pipeline_result, "safe answer")
        assert updated.final_action == "allow"
        assert updated.response_check is not None
        assert updated.response_check.action == "allow"

    def test_blocked_response_changes_final_action(self):
        processor = RAGGuardrailProcessor(AlwaysBlockEngine())
        pipeline_result = self._make_pipeline_result()
        updated = processor.check_final_response(pipeline_result, "blocked answer")
        assert updated.final_action == "block"
        assert updated.blocked_reason != ""

    def test_response_check_stage_is_response(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_pipeline_result()
        updated = processor.check_final_response(pipeline_result, "answer")
        assert updated.response_check.stage == RAGStage.RESPONSE


# ── RAGGuardrailProcessor – get_summary ──────────────────────────────────────

class TestRAGGuardrailProcessorGetSummary:
    def _make_full_pipeline_result(self, query_action="allow", response_action="allow",
                                   context_actions=None):
        context_actions = context_actions or []
        query_check = RAGGuardrailResult(
            stage=RAGStage.QUERY, action=query_action,
            matched_rules=[], severity="", original_content="q",
        )
        context_checks = [
            RAGGuardrailResult(
                stage=RAGStage.CONTEXT, action=a,
                matched_rules=[], severity="", original_content="c",
            )
            for a in context_actions
        ]
        response_check = RAGGuardrailResult(
            stage=RAGStage.RESPONSE, action=response_action,
            matched_rules=[], severity="", original_content="r",
        )
        safe_contexts = [
            RAGContext(content="c", source="s")
            for a in context_actions if a == "allow"
        ]
        return RAGPipelineResult(
            query_check=query_check,
            context_checks=context_checks,
            response_check=response_check,
            final_action="allow",
            safe_contexts=safe_contexts,
        )

    def test_summary_keys(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_full_pipeline_result()
        summary = processor.get_summary(pipeline_result)
        expected_keys = {
            "final_action", "query_safe", "total_contexts",
            "blocked_contexts", "safe_contexts", "response_safe", "blocked_reason"
        }
        assert expected_keys.issubset(summary.keys())

    def test_all_safe_summary(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_full_pipeline_result(
            context_actions=["allow", "allow"]
        )
        summary = processor.get_summary(pipeline_result)
        assert summary["query_safe"] is True
        assert summary["response_safe"] is True
        assert summary["total_contexts"] == 2
        assert summary["blocked_contexts"] == 0

    def test_blocked_context_in_summary(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_full_pipeline_result(
            context_actions=["allow", "block", "allow"]
        )
        summary = processor.get_summary(pipeline_result)
        assert summary["total_contexts"] == 3
        assert summary["blocked_contexts"] == 1

    def test_no_response_check_returns_none(self):
        processor = RAGGuardrailProcessor(AlwaysAllowEngine())
        pipeline_result = self._make_full_pipeline_result()
        pipeline_result.response_check = None
        summary = processor.get_summary(pipeline_result)
        assert summary["response_safe"] is None
