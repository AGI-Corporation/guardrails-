"""
RAG Guardrails Module
Specialized protection for Retrieval-Augmented Generation (RAG) pipelines.
Guards the query, retrieved context, and generated response stages.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import re


class RAGStage(Enum):
    QUERY = "query"
    CONTEXT = "context"
    RESPONSE = "response"


@dataclass
class RAGContext:
    content: str
    source: str
    score: float = 0.0
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGRequest:
    query: str
    contexts: List[RAGContext] = field(default_factory=list)
    system_prompt: str = ""


@dataclass
class RAGGuardrailResult:
    stage: RAGStage
    action: str  # "allow" or "block"
    matched_rules: List[str]
    severity: str
    original_content: str
    sanitized_content: Optional[str] = None
    details: Dict = field(default_factory=dict)


@dataclass
class RAGPipelineResult:
    query_check: RAGGuardrailResult
    context_checks: List[RAGGuardrailResult]
    response_check: Optional[RAGGuardrailResult]
    final_action: str
    safe_contexts: List[RAGContext]
    blocked_reason: str = ""


class RAGGuardrailProcessor:
    """
    Applies guardrails at each stage of a RAG pipeline:
    1. Query check - validate user input
    2. Context check - validate retrieved documents
    3. Response check - validate generated answer
    """

    def __init__(self, engine):
        self.engine = engine  # GuardrailEngine

    def check_query(self, query: str) -> RAGGuardrailResult:
        result = self.engine.evaluate(query)
        return RAGGuardrailResult(
            stage=RAGStage.QUERY,
            action=result.action,
            matched_rules=result.matched_rules,
            severity=result.severity,
            original_content=query,
        )

    def check_contexts(self, contexts: List[RAGContext]) -> List[RAGGuardrailResult]:
        results = []
        for ctx in contexts:
            result = self.engine.evaluate(ctx.content)
            results.append(RAGGuardrailResult(
                stage=RAGStage.CONTEXT,
                action=result.action,
                matched_rules=result.matched_rules,
                severity=result.severity,
                original_content=ctx.content,
                details={"source": ctx.source, "score": ctx.score},
            ))
        return results

    def check_response(self, response: str) -> RAGGuardrailResult:
        result = self.engine.evaluate(response)
        return RAGGuardrailResult(
            stage=RAGStage.RESPONSE,
            action=result.action,
            matched_rules=result.matched_rules,
            severity=result.severity,
            original_content=response,
        )

    def process_request(self, request: RAGRequest) -> RAGPipelineResult:
        """
        Run full pipeline guardrail checks.
        Returns safe contexts and whether the pipeline should proceed.
        """
        # 1. Check query
        query_check = self.check_query(request.query)
        if query_check.action == "block":
            return RAGPipelineResult(
                query_check=query_check,
                context_checks=[],
                response_check=None,
                final_action="block",
                safe_contexts=[],
                blocked_reason=f"Query blocked: {query_check.matched_rules}",
            )

        # 2. Check contexts - filter unsafe ones
        context_checks = self.check_contexts(request.contexts)
        safe_contexts = [
            ctx for ctx, check in zip(request.contexts, context_checks)
            if check.action != "block"
        ]

        return RAGPipelineResult(
            query_check=query_check,
            context_checks=context_checks,
            response_check=None,
            final_action="allow",
            safe_contexts=safe_contexts,
        )

    def check_final_response(
        self, pipeline_result: RAGPipelineResult, response: str
    ) -> RAGPipelineResult:
        """Check the LLM-generated response and update the pipeline result"""
        response_check = self.check_response(response)
        pipeline_result.response_check = response_check
        if response_check.action == "block":
            pipeline_result.final_action = "block"
            pipeline_result.blocked_reason = f"Response blocked: {response_check.matched_rules}"
        return pipeline_result

    def get_summary(self, result: RAGPipelineResult) -> Dict:
        blocked_contexts = sum(
            1 for c in result.context_checks if c.action == "block"
        )
        return {
            "final_action": result.final_action,
            "query_safe": result.query_check.action == "allow",
            "total_contexts": len(result.context_checks),
            "blocked_contexts": blocked_contexts,
            "safe_contexts": len(result.safe_contexts),
            "response_safe": (
                result.response_check.action == "allow"
                if result.response_check else None
            ),
            "blocked_reason": result.blocked_reason,
        }
