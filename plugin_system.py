"""
Custom Plugin Architecture
Allows sophisticated guardrail implementations beyond regex/keywords.
Plugins can use ML models, database lookups, or complex heuristics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import math
import time


@dataclass
class PluginResult:
    plugin_name: str
    passed: bool
    score: float  # 0.0 to 1.0, where 1.0 is most confident violation
    action: str   # "allow", "block", "warn"
    details: Dict = field(default_factory=dict)
    execution_time_ms: float = 0.0
    error: Optional[str] = None


class GuardrailPlugin(ABC):
    """
    Abstract base class for guardrail plugins.
    Implement this to create custom guardrail logic.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique plugin name"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        """Evaluate text and return a PluginResult"""
        pass


class PluginEngine:
    """Manages and runs a collection of plugins"""

    def __init__(self):
        self.plugins: Dict[str, GuardrailPlugin] = {}

    def register(self, plugin: GuardrailPlugin):
        self.plugins[plugin.name] = plugin

    def unregister(self, name: str):
        self.plugins.pop(name, None)

    def evaluate_all(self, text: str, context: Optional[Dict] = None) -> List[PluginResult]:
        results = []
        for plugin in self.plugins.values():
            start = time.time()
            try:
                result = plugin.evaluate(text, context)
                result.execution_time_ms = (time.time() - start) * 1000
            except Exception as e:
                result = PluginResult(
                    plugin_name=plugin.name,
                    passed=True,  # Fail-open on error
                    score=0.0,
                    action="allow",
                    error=str(e),
                    execution_time_ms=(time.time() - start) * 1000,
                )
            results.append(result)
        return results

    def get_final_action(self, results: List[PluginResult]) -> str:
        """Get final action from all plugin results (most restrictive wins)"""
        if any(r.action == "block" for r in results):
            return "block"
        if any(r.action == "warn" for r in results):
            return "warn"
        return "allow"


# ── Example Plugins ──────────────────────────────────────────────────────

class EntropyPlugin(GuardrailPlugin):
    """
    Detects high-entropy strings that may indicate secrets, API keys,
    or obfuscated content.
    """

    def __init__(self, threshold: float = 4.5):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "entropy_detector"

    @property
    def description(self) -> str:
        return "Detects high-entropy strings (potential secrets/keys)"

    def _calculate_entropy(self, text: str) -> float:
        if not text:
            return 0.0
        freq = {}
        for c in text:
            freq[c] = freq.get(c, 0) + 1
        length = len(text)
        return -sum((f / length) * math.log2(f / length) for f in freq.values())

    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        words = text.split()
        high_entropy_words = []

        for word in words:
            if len(word) >= 20:  # Only check long strings
                entropy = self._calculate_entropy(word)
                if entropy > self.threshold:
                    high_entropy_words.append((word[:20] + "...", entropy))

        if high_entropy_words:
            return PluginResult(
                plugin_name=self.name,
                passed=False,
                score=min(1.0, len(high_entropy_words) / 3),
                action="warn",
                details={"high_entropy_strings": high_entropy_words},
            )

        return PluginResult(
            plugin_name=self.name,
            passed=True,
            score=0.0,
            action="allow",
        )


class RepetitionPlugin(GuardrailPlugin):
    """
    Detects excessive repetition that may indicate prompt injection
    or DoS attempts.
    """

    def __init__(self, max_repetition_ratio: float = 0.5):
        self.max_repetition_ratio = max_repetition_ratio

    @property
    def name(self) -> str:
        return "repetition_detector"

    @property
    def description(self) -> str:
        return "Detects excessive repetition (potential DoS/injection)"

    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        words = text.lower().split()
        if len(words) < 10:
            return PluginResult(plugin_name=self.name, passed=True, score=0.0, action="allow")

        unique_ratio = len(set(words)) / len(words)
        repetition_ratio = 1 - unique_ratio

        if repetition_ratio > self.max_repetition_ratio:
            return PluginResult(
                plugin_name=self.name,
                passed=False,
                score=repetition_ratio,
                action="warn" if repetition_ratio < 0.8 else "block",
                details={"repetition_ratio": round(repetition_ratio, 2)},
            )

        return PluginResult(plugin_name=self.name, passed=True, score=0.0, action="allow")


class LengthPlugin(GuardrailPlugin):
    """Enforces text length limits"""

    def __init__(self, max_chars: int = 10000):
        self.max_chars = max_chars

    @property
    def name(self) -> str:
        return "length_guard"

    @property
    def description(self) -> str:
        return f"Blocks text exceeding {self.max_chars} characters"

    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        length = len(text)
        if length > self.max_chars:
            return PluginResult(
                plugin_name=self.name,
                passed=False,
                score=1.0,
                action="block",
                details={"length": length, "max_allowed": self.max_chars},
            )
        return PluginResult(plugin_name=self.name, passed=True, score=0.0, action="allow")


class ThreatIntelligencePlugin(GuardrailPlugin):
    """
    Detects known attack patterns derived from common vulnerability classes:
    SQL injection, command injection, path traversal, and malicious prompt
    signatures seen in the wild.
    """

    # Attack signatures — (label, regex_pattern) pairs
    SIGNATURES: List[tuple] = [
        # SQL injection
        ("sql_injection", r"(?i)\b(union\s+select|drop\s+table|insert\s+into|delete\s+from|exec\s*\(|xp_cmdshell)\b"),
        # Command injection
        ("command_injection", r"(?:;|\||\|\|)\s*(?:ls|cat|rm|wget|curl|bash|sh|python|nc|netcat|chmod)\b"),
        # Path traversal
        ("path_traversal", r"\.\.[/\\](?:\.\.[/\\]){1,}"),
        # Known jailbreak markers
        ("jailbreak_marker", r"(?i)\b(DAN|do\s+anything\s+now|developer\s+mode|jailbreak|uncensored\s+mode)\b"),
        # Prompt injection keywords
        ("prompt_injection", r"(?i)(ignore\s+(all\s+)?(previous|prior|above)\s+instructions|disregard\s+(your\s+)?guidelines)"),
        # SSRF / URL exfiltration
        ("ssrf_attempt", r"(?i)(file://|gopher://|dict://|ftp://|ldap://|http://(?:169\.254|127\.|10\.|192\.168|172\.(?:1[6-9]|2\d|3[01]))[^\s]*)"),
        # Known malware-related terms
        ("malware_reference", r"(?i)\b(keylogger|ransomware|rootkit|botnet|c2\s+server|command\s+and\s+control|payload\s+deliver)\b"),
    ]

    def __init__(self, block_on_match: bool = True):
        import re as _re
        self._compiled = [
            (label, _re.compile(pattern))
            for label, pattern in self.SIGNATURES
        ]
        self.block_on_match = block_on_match

    @property
    def name(self) -> str:
        return "threat_intelligence"

    @property
    def description(self) -> str:
        return "Detects SQL injection, command injection, path traversal, and known jailbreak signatures"

    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        matched: List[str] = []
        for label, pattern in self._compiled:
            if pattern.search(text):
                matched.append(label)

        if matched:
            action = "block" if self.block_on_match else "warn"
            return PluginResult(
                plugin_name=self.name,
                passed=False,
                score=min(1.0, len(matched) / len(self._compiled)),
                action=action,
                details={"matched_signatures": matched},
            )
        return PluginResult(plugin_name=self.name, passed=True, score=0.0, action="allow")


class PromptLeakPlugin(GuardrailPlugin):
    """
    Detects attempts to extract the system prompt, hidden instructions, or
    internal tool configurations from an LLM — a critical risk in deployed
    AI products.
    """

    _LEAK_PATTERNS: List[str] = [
        r"(?i)(what\s+(is|are)\s+(your\s+)?(system\s+prompt|instructions?|prompt|directives?))",
        r"(?i)(show\s+me\s+(your\s+)?(system\s+|hidden\s+)?(instructions?|prompt|config))",
        r"(?i)(repeat\s+(the|your)\s+(system\s+)?prompt)",
        r"(?i)(output\s+(the|your)\s+(system\s+)?(instructions?|prompt))",
        r"(?i)(print\s+(the|your)\s+(system\s+)?prompt)",
        r"(?i)(ignore\s+(above|previous)\s+.{0,40}print\s+(original\s+)?(prompt|instructions?))",
        r"(?i)(what\s+were\s+you\s+told)",
        r"(?i)(verbatim\s+(copy|repeat|output|print)\s+.{0,30}(prompt|instruction|context))",
    ]

    def __init__(self):
        import re as _re
        self._compiled = [_re.compile(p) for p in self._LEAK_PATTERNS]

    @property
    def name(self) -> str:
        return "prompt_leak_detector"

    @property
    def description(self) -> str:
        return "Detects attempts to extract system prompts or internal instructions"

    def evaluate(self, text: str, context: Optional[Dict] = None) -> PluginResult:
        matched_count = sum(1 for p in self._compiled if p.search(text))
        if matched_count > 0:
            return PluginResult(
                plugin_name=self.name,
                passed=False,
                score=min(1.0, matched_count / len(self._compiled)),
                action="block",
                details={"matched_patterns": matched_count, "category": "prompt_leak"},
            )
        return PluginResult(plugin_name=self.name, passed=True, score=0.0, action="allow")


def create_default_plugin_engine() -> PluginEngine:
    """Create a plugin engine with default plugins"""
    engine = PluginEngine()
    engine.register(EntropyPlugin())
    engine.register(RepetitionPlugin())
    engine.register(LengthPlugin())
    engine.register(ThreatIntelligencePlugin())
    engine.register(PromptLeakPlugin())
    return engine


class PluginManager:
    """
    High-level façade used by ``quickstart.py`` and external callers.

    Wraps a ``PluginEngine`` and wires it to an optional ``GuardrailEngine``
    so that plugin results can inform guardrail decisions.
    """

    def __init__(self, guardrail_engine=None) -> None:
        self._plugin_engine = create_default_plugin_engine()
        self._guardrail_engine = guardrail_engine

    def register(self, plugin: GuardrailPlugin) -> None:
        """Register a plugin with the underlying engine."""
        self._plugin_engine.register(plugin)

    def unregister(self, name: str) -> None:
        self._plugin_engine.unregister(name)

    def list_plugins(self) -> List[str]:
        """Return names of all registered plugins."""
        return list(self._plugin_engine.plugins.keys())

    def evaluate(self, text: str, context: Optional[Dict] = None) -> List[PluginResult]:
        """Run all plugins against *text* and return their results."""
        return self._plugin_engine.evaluate_all(text, context)

    def get_final_action(self, text: str, context: Optional[Dict] = None) -> str:
        """Convenience method: evaluate and return the most restrictive action."""
        results = self.evaluate(text, context)
        return self._plugin_engine.get_final_action(results)
