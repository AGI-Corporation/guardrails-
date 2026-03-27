"""
Tests for rag_guardrails.py
Covers: RAGGuardrailProcessor check_query/check_contexts/check_response/
        process_request/check_final_response/get_summary,
        RAGContext/RAGRequest/RAGGuardrailResult/RAGPipelineResult data classes.
"""

import pytest

from rag_guardrails import (
    RAGContext,
    RAGGuardrailProcessor,
    RAGGuardrailResult,
    RAGPipelineResult,
    RAGRequest,
    RAGStage,
)
from guardrail_framework import (
    Action,
    GuardrailCategory,
    GuardrailEngine,
    GuardrailRule,
    Severity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_engine_block_keyword(keyword="BLOCKED"):
    engine = GuardrailEngine()
    rule = GuardrailRule(
        id="block_rule",
        name="Block Rule",
        category=GuardrailCategory.CUSTOM,
        severity=Severity.HIGH,
        action=Action.BLOCK,
        keywords=[keyword],
    )
    engine.add_rule(rule)
    return engine


def _make_permissive_engine():
    return GuardrailEngine()


def _make_processor(keyword=None):
    engine = _make_engine_block_keyword(keyword) if keyword else _make_permissive_engine()
    return RAGGuardrailProcessor(engine)


# ---------------------------------------------------------------------------
# Data class tests
# ---------------------------------------------------------------------------

class TestRAGDataClasses:
    def test_rag_context_defaults(self):
        ctx = RAGContext(content="doc text", source="db")
        assert ctx.score == 0.0
        assert ctx.metadata == {}

    def test_rag_request_defaults(self):
        req = RAGRequest(query="what is AI?")
        assert req.contexts == []
        assert req.system_prompt == ""

    def test_rag_guardrail_result_defaults(self):
        result = RAGGuardrailResult(
            stage=RAGStage.QUERY,
            action="allow",
            matched_rules=[],
            severity="none",
            original_content="hello",
        )
        assert result.sanitized_content is None
        assert result.details == {}


# ---------------------------------------------------------------------------
# check_query()
# ---------------------------------------------------------------------------

class TestCheckQuery:
    def test_safe_query_allowed(self):
        processor = _make_processor()
        result = processor.check_query("What is the capital of France?")
        assert result.action == "allow"
        assert result.stage == RAGStage.QUERY

    def test_blocked_query_returned(self):
        processor = _make_processor(keyword="BLOCKED")
        result = processor.check_query("This is BLOCKED content")
        assert result.action == "block"
        assert result.stage == RAGStage.QUERY

    def test_matched_rules_populated(self):
        processor = _make_processor(keyword="BLOCKED")
        result = processor.check_query("BLOCKED text")
        assert "block_rule" in result.matched_rules

    def test_original_content_preserved(self):
        processor = _make_processor()
        query = "test query text"
        result = processor.check_query(query)
        assert result.original_content == query


# ---------------------------------------------------------------------------
# check_contexts()
# ---------------------------------------------------------------------------

class TestCheckContexts:
    def test_empty_contexts_returns_empty_list(self):
        processor = _make_processor()
        results = processor.check_contexts([])
        assert results == []

    def test_safe_contexts_allowed(self):
        processor = _make_processor()
        contexts = [
            RAGContext(content="safe doc 1", source="src1"),
            RAGContext(content="safe doc 2", source="src2"),
        ]
        results = processor.check_contexts(contexts)
        assert len(results) == 2
        assert all(r.action == "allow" for r in results)

    def test_blocked_context_identified(self):
        processor = _make_processor(keyword="BLOCKED")
        contexts = [
            RAGContext(content="safe doc", source="src1"),
            RAGContext(content="BLOCKED content", source="src2"),
        ]
        results = processor.check_contexts(contexts)
        assert results[0].action == "allow"
        assert results[1].action == "block"

    def test_context_stage_is_context(self):
        processor = _make_processor()
        contexts = [RAGContext(content="text", source="src")]
        results = processor.check_contexts(contexts)
        assert results[0].stage == RAGStage.CONTEXT

    def test_details_include_source_and_score(self):
        processor = _make_processor()
        contexts = [RAGContext(content="text", source="my_source", score=0.9)]
        results = processor.check_contexts(contexts)
        assert results[0].details["source"] == "my_source"
        assert results[0].details["score"] == 0.9


# ---------------------------------------------------------------------------
# check_response()
# ---------------------------------------------------------------------------

class TestCheckResponse:
    def test_safe_response_allowed(self):
        processor = _make_processor()
        result = processor.check_response("This is a helpful safe response.")
        assert result.action == "allow"
        assert result.stage == RAGStage.RESPONSE

    def test_blocked_response_identified(self):
        processor = _make_processor(keyword="BLOCKED")
        result = processor.check_response("Response with BLOCKED content")
        assert result.action == "block"

    def test_original_content_preserved(self):
        processor = _make_processor()
        response = "a helpful answer"
        result = processor.check_response(response)
        assert result.original_content == response


# ---------------------------------------------------------------------------
# process_request()
# ---------------------------------------------------------------------------

class TestProcessRequest:
    def test_safe_request_allowed(self):
        processor = _make_processor()
        request = RAGRequest(
            query="What is AI?",
            contexts=[
                RAGContext(content="AI is artificial intelligence.", source="s1"),
            ],
        )
        result = processor.process_request(request)
        assert result.final_action == "allow"

    def test_blocked_query_short_circuits(self):
        processor = _make_processor(keyword="BLOCKED")
        request = RAGRequest(
            query="BLOCKED query",
            contexts=[RAGContext(content="safe context", source="s1")],
        )
        result = processor.process_request(request)
        assert result.final_action == "block"
        assert result.context_checks == []
        assert result.safe_contexts == []
        assert result.blocked_reason  # non-empty

    def test_blocked_context_filtered_from_safe_contexts(self):
        processor = _make_processor(keyword="BLOCKED")
        request = RAGRequest(
            query="safe query",
            contexts=[
                RAGContext(content="safe context", source="s1"),
                RAGContext(content="BLOCKED context", source="s2"),
            ],
        )
        result = processor.process_request(request)
        assert result.final_action == "allow"
        assert len(result.safe_contexts) == 1
        assert result.safe_contexts[0].source == "s1"

    def test_all_contexts_safe_all_in_safe_list(self):
        processor = _make_processor()
        request = RAGRequest(
            query="safe query",
            contexts=[
                RAGContext(content="doc 1", source="s1"),
                RAGContext(content="doc 2", source="s2"),
            ],
        )
        result = processor.process_request(request)
        assert len(result.safe_contexts) == 2

    def test_empty_contexts_allowed(self):
        processor = _make_processor()
        request = RAGRequest(query="safe query", contexts=[])
        result = processor.process_request(request)
        assert result.final_action == "allow"
        assert result.safe_contexts == []

    def test_query_check_included_in_result(self):
        processor = _make_processor()
        request = RAGRequest(query="test query")
        result = processor.process_request(request)
        assert isinstance(result.query_check, RAGGuardrailResult)
        assert result.query_check.stage == RAGStage.QUERY


# ---------------------------------------------------------------------------
# check_final_response()
# ---------------------------------------------------------------------------

class TestCheckFinalResponse:
    def _make_pipeline_result(self, processor, query="safe query", contexts=None):
        request = RAGRequest(query=query, contexts=contexts or [])
        return processor.process_request(request)

    def test_safe_response_keeps_allow(self):
        processor = _make_processor()
        pipeline_result = self._make_pipeline_result(processor)
        final = processor.check_final_response(pipeline_result, "a safe answer")
        assert final.final_action == "allow"
        assert final.response_check is not None
        assert final.response_check.action == "allow"

    def test_blocked_response_changes_action_to_block(self):
        processor = _make_processor(keyword="BLOCKED")
        pipeline_result = self._make_pipeline_result(processor)
        final = processor.check_final_response(pipeline_result, "BLOCKED response")
        assert final.final_action == "block"
        assert "block_rule" in final.response_check.matched_rules

    def test_blocked_response_sets_blocked_reason(self):
        processor = _make_processor(keyword="BLOCKED")
        pipeline_result = self._make_pipeline_result(processor)
        final = processor.check_final_response(pipeline_result, "BLOCKED response")
        assert final.blocked_reason  # non-empty

    def test_response_check_stage_is_response(self):
        processor = _make_processor()
        pipeline_result = self._make_pipeline_result(processor)
        final = processor.check_final_response(pipeline_result, "answer")
        assert final.response_check.stage == RAGStage.RESPONSE


# ---------------------------------------------------------------------------
# get_summary()
# ---------------------------------------------------------------------------

class TestGetSummary:
    def test_summary_keys_present(self):
        processor = _make_processor()
        request = RAGRequest(
            query="safe query",
            contexts=[RAGContext(content="context", source="src")],
        )
        pipeline_result = processor.process_request(request)
        pipeline_result = processor.check_final_response(pipeline_result, "safe answer")
        summary = processor.get_summary(pipeline_result)
        for key in ("final_action", "query_safe", "total_contexts",
                    "blocked_contexts", "safe_contexts", "response_safe", "blocked_reason"):
            assert key in summary

    def test_summary_counts_blocked_contexts(self):
        processor = _make_processor(keyword="BLOCKED")
        request = RAGRequest(
            query="safe query",
            contexts=[
                RAGContext(content="safe", source="s1"),
                RAGContext(content="BLOCKED", source="s2"),
            ],
        )
        pipeline_result = processor.process_request(request)
        summary = processor.get_summary(pipeline_result)
        assert summary["total_contexts"] == 2
        assert summary["blocked_contexts"] == 1
        assert summary["safe_contexts"] == 1

    def test_summary_response_safe_none_when_not_checked(self):
        processor = _make_processor()
        request = RAGRequest(query="query", contexts=[])
        pipeline_result = processor.process_request(request)
        summary = processor.get_summary(pipeline_result)
        assert summary["response_safe"] is None

    def test_summary_query_safe_false_when_blocked(self):
        processor = _make_processor(keyword="BLOCKED")
        request = RAGRequest(query="BLOCKED query", contexts=[])
        pipeline_result = processor.process_request(request)
        summary = processor.get_summary(pipeline_result)
        assert summary["query_safe"] is False
        assert summary["final_action"] == "block"
