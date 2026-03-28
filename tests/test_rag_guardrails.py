"""
Tests for rag_guardrails.py
"""

import pytest

from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
)
from rag_guardrails import (
    RAGContext,
    RAGGuardrailProcessor,
    RAGGuardrailResult,
    RAGPipelineResult,
    RAGRequest,
    RAGStage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clean_engine() -> GuardrailEngine:
    return GuardrailEngine()


def _engine_blocking(keyword: str) -> GuardrailEngine:
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


def _processor(engine=None) -> RAGGuardrailProcessor:
    return RAGGuardrailProcessor(engine or _clean_engine())


# ---------------------------------------------------------------------------
# RAGContext dataclass
# ---------------------------------------------------------------------------

class TestRAGContext:
    def test_defaults(self):
        ctx = RAGContext(content="some content", source="wiki")
        assert ctx.score == 0.0
        assert ctx.metadata == {}

    def test_custom_values(self):
        ctx = RAGContext(
            content="text",
            source="db",
            score=0.95,
            metadata={"doc_id": "123"},
        )
        assert ctx.score == 0.95
        assert ctx.metadata["doc_id"] == "123"


# ---------------------------------------------------------------------------
# RAGRequest dataclass
# ---------------------------------------------------------------------------

class TestRAGRequest:
    def test_defaults(self):
        req = RAGRequest(query="What is AI?")
        assert req.contexts == []
        assert req.system_prompt == ""


# ---------------------------------------------------------------------------
# RAGStage enum
# ---------------------------------------------------------------------------

class TestRAGStage:
    def test_values(self):
        assert RAGStage.QUERY.value == "query"
        assert RAGStage.CONTEXT.value == "context"
        assert RAGStage.RESPONSE.value == "response"


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — check_query
# ---------------------------------------------------------------------------

class TestCheckQuery:
    def test_safe_query_allowed(self):
        proc = _processor()
        result = proc.check_query("What is the capital of France?")
        assert result.action == "allow"
        assert result.stage == RAGStage.QUERY
        assert result.original_content == "What is the capital of France?"

    def test_blocked_query(self):
        proc = _processor(_engine_blocking("forbidden"))
        result = proc.check_query("forbidden content")
        assert result.action == "block"
        assert result.stage == RAGStage.QUERY

    def test_matched_rules_populated(self):
        proc = _processor(_engine_blocking("forbidden"))
        result = proc.check_query("forbidden content")
        assert "block_rule" in result.matched_rules


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — check_contexts
# ---------------------------------------------------------------------------

class TestCheckContexts:
    def test_empty_contexts(self):
        proc = _processor()
        results = proc.check_contexts([])
        assert results == []

    def test_safe_context_allowed(self):
        proc = _processor()
        ctx = RAGContext(content="Safe information here.", source="wiki")
        results = proc.check_contexts([ctx])
        assert len(results) == 1
        assert results[0].action == "allow"
        assert results[0].stage == RAGStage.CONTEXT

    def test_blocked_context_flagged(self):
        proc = _processor(_engine_blocking("forbidden"))
        ctx = RAGContext(content="forbidden content", source="untrusted")
        results = proc.check_contexts([ctx])
        assert results[0].action == "block"

    def test_mixed_contexts(self):
        proc = _processor(_engine_blocking("forbidden"))
        contexts = [
            RAGContext(content="safe content", source="doc1"),
            RAGContext(content="forbidden content", source="doc2"),
            RAGContext(content="another safe document", source="doc3"),
        ]
        results = proc.check_contexts(contexts)
        assert len(results) == 3
        assert results[0].action == "allow"
        assert results[1].action == "block"
        assert results[2].action == "allow"

    def test_context_details_contains_source(self):
        proc = _processor(_engine_blocking("forbidden"))
        ctx = RAGContext(content="forbidden data", source="my_source", score=0.8)
        results = proc.check_contexts([ctx])
        assert results[0].details["source"] == "my_source"
        assert results[0].details["score"] == 0.8


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — check_response
# ---------------------------------------------------------------------------

class TestCheckResponse:
    def test_safe_response_allowed(self):
        proc = _processor()
        result = proc.check_response("Paris is the capital of France.")
        assert result.action == "allow"
        assert result.stage == RAGStage.RESPONSE

    def test_blocked_response(self):
        proc = _processor(_engine_blocking("forbidden"))
        result = proc.check_response("This is forbidden output")
        assert result.action == "block"


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — process_request
# ---------------------------------------------------------------------------

class TestProcessRequest:
    def test_clean_request_passes(self):
        proc = _processor()
        request = RAGRequest(
            query="What is Python?",
            contexts=[RAGContext(content="Python is a programming language.", source="wiki")],
        )
        result = proc.process_request(request)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 1

    def test_blocked_query_short_circuits(self):
        proc = _processor(_engine_blocking("forbidden"))
        request = RAGRequest(
            query="forbidden query",
            contexts=[RAGContext(content="safe context", source="doc1")],
        )
        result = proc.process_request(request)
        assert result.final_action == "block"
        assert result.safe_contexts == []
        assert result.context_checks == []
        assert result.blocked_reason != ""

    def test_unsafe_context_filtered_out(self):
        proc = _processor(_engine_blocking("malicious"))
        request = RAGRequest(
            query="What is the weather?",
            contexts=[
                RAGContext(content="Safe document", source="doc1"),
                RAGContext(content="malicious document", source="doc2"),
            ],
        )
        result = proc.process_request(request)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 1
        assert result.safe_contexts[0].source == "doc1"

    def test_all_contexts_blocked_pipeline_still_allows(self):
        proc = _processor(_engine_blocking("bad"))
        request = RAGRequest(
            query="safe query",
            contexts=[RAGContext(content="bad content", source="d1")],
        )
        result = proc.process_request(request)
        # Query is safe, so pipeline continues; safe_contexts is empty
        assert result.final_action == "allow"
        assert result.safe_contexts == []

    def test_query_check_included_in_result(self):
        proc = _processor()
        request = RAGRequest(query="hello?", contexts=[])
        result = proc.process_request(request)
        assert result.query_check is not None
        assert result.query_check.action == "allow"


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — check_final_response
# ---------------------------------------------------------------------------

class TestCheckFinalResponse:
    def test_safe_final_response(self):
        proc = _processor()
        request = RAGRequest(query="hello?", contexts=[])
        pipeline_result = proc.process_request(request)
        updated = proc.check_final_response(pipeline_result, "A safe response.")
        assert updated.response_check is not None
        assert updated.response_check.action == "allow"
        assert updated.final_action == "allow"

    def test_blocked_final_response_updates_result(self):
        proc = _processor(_engine_blocking("forbidden"))
        request = RAGRequest(query="hello?", contexts=[])
        pipeline_result = proc.process_request(request)
        updated = proc.check_final_response(pipeline_result, "forbidden response text")
        assert updated.final_action == "block"
        assert updated.blocked_reason != ""


# ---------------------------------------------------------------------------
# RAGGuardrailProcessor — get_summary
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_summary_clean_pipeline(self):
        proc = _processor()
        request = RAGRequest(
            query="safe query",
            contexts=[RAGContext(content="safe context", source="doc1")],
        )
        pipeline = proc.process_request(request)
        pipeline = proc.check_final_response(pipeline, "safe response")
        summary = proc.get_summary(pipeline)

        assert summary["final_action"] == "allow"
        assert summary["query_safe"] is True
        assert summary["total_contexts"] == 1
        assert summary["blocked_contexts"] == 0
        assert summary["safe_contexts"] == 1
        assert summary["response_safe"] is True

    def test_summary_blocked_query(self):
        proc = _processor(_engine_blocking("forbidden"))
        request = RAGRequest(query="forbidden query", contexts=[])
        pipeline = proc.process_request(request)
        summary = proc.get_summary(pipeline)

        assert summary["final_action"] == "block"
        assert summary["query_safe"] is False

    def test_summary_no_response_check(self):
        proc = _processor()
        request = RAGRequest(query="hello", contexts=[])
        pipeline = proc.process_request(request)
        summary = proc.get_summary(pipeline)
        # response_check not yet run
        assert summary["response_safe"] is None
