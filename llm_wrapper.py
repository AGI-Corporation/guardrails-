"""
LLM Integration Wrapper
Applies guardrails to LLM inputs and outputs with support for
OpenAI, Anthropic, and mock providers.
"""

import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import json


# ── Core types ─────────────────────────────────────────────────────────────

@dataclass
class LLMRequest:
    prompt: str
    model: str = "default"
    temperature: float = 0.7
    max_tokens: int = 1000
    metadata: Dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0
    finish_reason: str = "complete"


@dataclass
class GuardedLLMResult:
    request: LLMRequest
    response: Optional[LLMResponse]
    input_action: str
    output_action: str
    input_matched_rules: List[str]
    output_matched_rules: List[str]
    blocked: bool
    block_reason: str = ""
    latency_ms: float = 0.0


# ── LLM Provider base ─────────────────────────────────────────────────

class LLMProvider(ABC):
    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        pass


class MockLLMProvider(LLMProvider):
    """Mock provider for testing"""

    def __init__(self, response_text: str = "This is a mock response."):
        self.response_text = response_text
        self.call_count = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.call_count += 1
        time.sleep(0.01)  # Simulate latency
        return LLMResponse(
            text=self.response_text,
            model=request.model,
            tokens_used=len(request.prompt.split()) + len(self.response_text.split()),
            latency_ms=10.0,
        )


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            start = time.time()
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": request.prompt}],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            latency = (time.time() - start) * 1000
            return LLMResponse(
                text=response.choices[0].message.content,
                model=self.model,
                tokens_used=response.usage.total_tokens,
                latency_ms=latency,
                finish_reason=response.choices[0].finish_reason,
            )
        except ImportError:
            raise RuntimeError("openai package not installed. Run: pip install openai")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude API provider"""

    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model

    def complete(self, request: LLMRequest) -> LLMResponse:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            start = time.time()
            response = client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.prompt}],
            )
            latency = (time.time() - start) * 1000
            return LLMResponse(
                text=response.content[0].text,
                model=self.model,
                tokens_used=response.usage.input_tokens + response.usage.output_tokens,
                latency_ms=latency,
                finish_reason=response.stop_reason,
            )
        except ImportError:
            raise RuntimeError("anthropic package not installed. Run: pip install anthropic")


# ── Guarded LLM wrapper ───────────────────────────────────────────────

class GuardedLLM:
    """
    Wraps an LLM provider with guardrail checks on both inputs and outputs.
    Blocks requests that violate input rules and filters outputs that
    violate output rules.
    """

    def __init__(self, provider: LLMProvider, engine):
        self.provider = provider
        self.engine = engine  # GuardrailEngine instance
        self.stats = {"total": 0, "input_blocked": 0, "output_blocked": 0, "passed": 0}

    def complete(self, request: LLMRequest) -> GuardedLLMResult:
        start = time.time()
        self.stats["total"] += 1

        # ─ 1. Check input ──────────────────────────────────────────────────
        input_result = self.engine.evaluate(request.prompt)
        if input_result.action == "block":
            self.stats["input_blocked"] += 1
            return GuardedLLMResult(
                request=request,
                response=None,
                input_action="block",
                output_action="n/a",
                input_matched_rules=input_result.matched_rules,
                output_matched_rules=[],
                blocked=True,
                block_reason=f"Input blocked by rules: {input_result.matched_rules}",
                latency_ms=(time.time() - start) * 1000,
            )

        # ─ 2. Call LLM ───────────────────────────────────────────────────
        llm_response = self.provider.complete(request)

        # ─ 3. Check output ───────────────────────────────────────────────
        output_result = self.engine.evaluate(llm_response.text)
        if output_result.action == "block":
            self.stats["output_blocked"] += 1
            return GuardedLLMResult(
                request=request,
                response=None,
                input_action="allow",
                output_action="block",
                input_matched_rules=[],
                output_matched_rules=output_result.matched_rules,
                blocked=True,
                block_reason=f"Output blocked by rules: {output_result.matched_rules}",
                latency_ms=(time.time() - start) * 1000,
            )

        self.stats["passed"] += 1
        return GuardedLLMResult(
            request=request,
            response=llm_response,
            input_action="allow",
            output_action="allow",
            input_matched_rules=[],
            output_matched_rules=[],
            blocked=False,
            latency_ms=(time.time() - start) * 1000,
        )

    def get_stats(self) -> Dict:
        total = self.stats["total"]
        return {
            **self.stats,
            "block_rate": round((self.stats["input_blocked"] + self.stats["output_blocked"]) / total * 100, 1)
            if total > 0 else 0,
        }
