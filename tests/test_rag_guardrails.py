"""
Tests for rag_guardrails.py
Covers: RAGGuardrailProcessor (check_query, check_contexts, check_response,
        process_request, check_final_response, get_summary).
"""

import pytest

from rag_guardrails import (
    RAGGuardrailProcessor,
    RAGContext,
    RAGRequest,
    RAGStage,
    RAGGuardrailResult,
    RAGPipelineResult,
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
def engine():
    e = GuardrailEngine()
    for rule in create_default_guardrails():
        e.add_rule(rule)
    return e


@pytest.fixture
def processor(engine):
    return RAGGuardrailProcessor(engine)


def _safe_ctx(content="This is safe context.") -> RAGContext:
    return RAGContext(content=content, source="doc_1", score=0.9)


def _unsafe_ctx() -> RAGContext:
    return RAGContext(content="SSN: 123-45-6789", source="doc_evil", score=0.5)


# ── check_query ───────────────────────────────────────────────────────────

class TestCheckQuery:

    def test_safe_query_allowed(self, processor):
        result = processor.check_query("What is machine learning?")
        assert result.action == "allow"
        assert result.stage == RAGStage.QUERY

    def test_unsafe_query_blocked(self, processor):
        result = processor.check_query("My SSN is 123-45-6789")
        assert result.action == "block"
        assert result.stage == RAGStage.QUERY

    def test_result_contains_original_content(self, processor):
        query = "Tell me about AI"
        result = processor.check_query(query)
        assert result.original_content == query


# ── check_contexts ────────────────────────────────────────────────────────

class TestCheckContexts:

    def test_safe_contexts_all_allowed(self, processor):
        contexts = [_safe_ctx("Python is great"), _safe_ctx("AI is useful")]
        results = processor.check_contexts(contexts)
        assert len(results) == 2
        assert all(r.action == "allow" for r in results)

    def test_unsafe_context_blocked(self, processor):
        results = processor.check_contexts([_unsafe_ctx()])
        assert len(results) == 1
        assert results[0].action == "block"

    def test_mixed_contexts_filtered(self, processor):
        contexts = [_safe_ctx(), _unsafe_ctx()]
        results = processor.check_contexts(contexts)
        actions = [r.action for r in results]
        assert "allow" in actions
        assert "block" in actions

    def test_result_stage_is_context(self, processor):
        results = processor.check_contexts([_safe_ctx()])
        for r in results:
            assert r.stage == RAGStage.CONTEXT


# ── check_response ────────────────────────────────────────────────────────

class TestCheckResponse:

    def test_safe_response_allowed(self, processor):
        result = processor.check_response("Machine learning is a subfield of AI.")
        assert result.action == "allow"
        assert result.stage == RAGStage.RESPONSE

    def test_unsafe_response_blocked(self, processor):
        result = processor.check_response("Here is an SSN: 123-45-6789")
        assert result.action == "block"


# ── process_request ───────────────────────────────────────────────────────

class TestProcessRequest:

    def test_safe_request_allowed(self, processor):
        req = RAGRequest(
            query="What is AI?",
            contexts=[_safe_ctx("AI context"), _safe_ctx("More context")],
        )
        result = processor.process_request(req)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 2

    def test_blocked_query_stops_pipeline(self, processor):
        req = RAGRequest(
            query="SSN: 123-45-6789",
            contexts=[_safe_ctx()],
        )
        result = processor.process_request(req)
        assert result.final_action == "block"
        assert result.safe_contexts == []
        assert result.blocked_reason

    def test_unsafe_contexts_filtered_out(self, processor):
        req = RAGRequest(
            query="Safe query",
            contexts=[_safe_ctx(), _unsafe_ctx()],
        )
        result = processor.process_request(req)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 1
        assert result.safe_contexts[0].source == "doc_1"

    def test_empty_contexts_allowed(self, processor):
        req = RAGRequest(query="What is Python?", contexts=[])
        result = processor.process_request(req)
        assert result.final_action == "allow"
        assert result.safe_contexts == []


# ── check_final_response ──────────────────────────────────────────────────

class TestCheckFinalResponse:

    def test_safe_response_keeps_allow(self, processor):
        req = RAGRequest(query="Safe query", contexts=[_safe_ctx()])
        pipeline = processor.process_request(req)
        updated = processor.check_final_response(pipeline, "A safe answer.")
        assert updated.final_action == "allow"
        assert updated.response_check is not None
        assert updated.response_check.action == "allow"

    def test_unsafe_response_blocks_pipeline(self, processor):
        req = RAGRequest(query="Safe query", contexts=[_safe_ctx()])
        pipeline = processor.process_request(req)
        updated = processor.check_final_response(pipeline, "SSN: 123-45-6789")
        assert updated.final_action == "block"
        assert updated.blocked_reason


# ── get_summary ───────────────────────────────────────────────────────────

class TestGetSummary:

    def test_summary_keys_present(self, processor):
        req = RAGRequest(query="Safe query", contexts=[_safe_ctx(), _unsafe_ctx()])
        pipeline = processor.process_request(req)
        updated = processor.check_final_response(pipeline, "Safe answer")
        summary = processor.get_summary(updated)

        assert "final_action" in summary
        assert "query_safe" in summary
        assert "total_contexts" in summary
        assert "blocked_contexts" in summary
        assert "safe_contexts" in summary
        assert "response_safe" in summary

    def test_summary_counts(self, processor):
        req = RAGRequest(query="Safe?", contexts=[_safe_ctx(), _unsafe_ctx()])
        pipeline = processor.process_request(req)
        updated = processor.check_final_response(pipeline, "Good answer")
        summary = processor.get_summary(updated)

        assert summary["total_contexts"] == 2
        assert summary["blocked_contexts"] == 1
        assert summary["safe_contexts"] == 1
        assert summary["query_safe"] is True
        assert summary["response_safe"] is True
