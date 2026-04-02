"""
📋 Policy Manager
=================
YAML and JSON-based policy configuration for the Guardrails Framework.

The README references ``config.yaml`` but there was no implementation.
This module provides the full load → validate → apply pipeline.

A *policy file* declares guardrail rules, LLM settings, audit options, and
profiling thresholds in a human-readable format.  ``PolicyManager`` loads
the file, validates every field, and wires the rules directly into a
``GuardrailEngine`` instance.

Supported formats
-----------------
    YAML (recommended)  — ``policy_manager.load("config.yaml")``
    JSON                — ``policy_manager.load("config.json")``
    Python dict         — ``policy_manager.load_dict({...})``

Policy file schema
------------------
    guardrails:
      - id: "pii_ssn"
        name: "SSN Detection"
        severity: "critical"    # low | medium | high | critical
        action: "block"         # allow | block | warn
        patterns:
          - "\\d{3}-\\d{2}-\\d{4}"
        keywords: []
        enabled: true           # optional; default true

    llm:
      provider: "openai"        # openai | anthropic | custom
      model: "gpt-4o"
      api_key: "${OPENAI_API_KEY}"   # env-var expansion supported

    audit:
      enabled: true
      retention_days: 90
      db_path: "./audit_log.db"

    profiling:
      enabled: true
      slow_threshold_ms: 200

    rate_limiting:
      enabled: true
      algorithm: "token_bucket"   # token_bucket | sliding_window
      capacity: 60
      refill_rate: 1.0
      max_requests: 60
      window_s: 60.0

Public surface
--------------
    PolicyValidationError   — raised for invalid policy files
    PolicyConfig            — validated, typed representation of a policy file
    PolicyManager           — loads, validates, and applies policy to the engine

Usage
-----
    from policy_manager import PolicyManager
    from guardrail_framework import GuardrailEngine

    engine = GuardrailEngine()
    pm = PolicyManager(engine)
    config = pm.load("config.yaml")
    print(f"Loaded {len(config.guardrails)} guardrail rules")
    # engine is now populated with all enabled rules

    # Or load from a dict
    config = pm.load_dict({
        "guardrails": [
            {"id": "block_ssn", "name": "SSN", "severity": "critical",
             "action": "block", "patterns": ["\\\\d{3}-\\\\d{2}-\\\\d{4}"]}
        ]
    })
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from guardrail_framework import Action, GuardrailEngine, GuardrailRule, Severity


# ── Exceptions ────────────────────────────────────────────────────────────────

class PolicyValidationError(Exception):
    """Raised when a policy file fails validation."""


# ── Typed sub-configs ─────────────────────────────────────────────────────────

@dataclass
class GuardrailRuleConfig:
    """Validated representation of a single guardrail rule in a policy file."""
    id: str
    name: str
    severity: str
    action: str
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    enabled: bool = True

    def to_rule(self) -> GuardrailRule:
        """Convert to a ``GuardrailRule`` instance."""
        return GuardrailRule(
            id=self.id,
            name=self.name,
            severity=Severity(self.severity),
            action=Action(self.action),
            patterns=self.patterns,
            keywords=self.keywords,
        )


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str = ""
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 30.0


@dataclass
class AuditConfig:
    """Audit logger configuration."""
    enabled: bool = True
    retention_days: int = 90
    db_path: str = "./audit_log.db"
    export_path: str = "./audit_logs/"


@dataclass
class ProfilingConfig:
    """Performance profiler configuration."""
    enabled: bool = True
    slow_threshold_ms: float = 200.0


@dataclass
class RateLimitingConfig:
    """Rate limiter configuration."""
    enabled: bool = False
    algorithm: str = "token_bucket"  # "token_bucket" | "sliding_window"
    # token_bucket params
    capacity: float = 60.0
    refill_rate: float = 1.0
    # sliding_window params
    max_requests: int = 60
    window_s: float = 60.0


@dataclass
class PolicyConfig:
    """
    Full validated policy configuration.

    All fields have sensible defaults so partial policy files are accepted.
    """
    guardrails: List[GuardrailRuleConfig] = field(default_factory=list)
    llm: LLMConfig = field(default_factory=LLMConfig)
    audit: AuditConfig = field(default_factory=AuditConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    rate_limiting: RateLimitingConfig = field(default_factory=RateLimitingConfig)
    raw: Dict = field(default_factory=dict)  # original parsed dict

    @property
    def active_rules(self) -> List[GuardrailRuleConfig]:
        """Only rules that are enabled."""
        return [r for r in self.guardrails if r.enabled]

    def to_dict(self) -> Dict:
        return {
            "guardrails": [
                {
                    "id": r.id, "name": r.name, "severity": r.severity,
                    "action": r.action, "patterns": r.patterns,
                    "keywords": r.keywords, "enabled": r.enabled,
                }
                for r in self.guardrails
            ],
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "temperature": self.llm.temperature,
                "max_tokens": self.llm.max_tokens,
            },
            "audit": {
                "enabled": self.audit.enabled,
                "retention_days": self.audit.retention_days,
                "db_path": self.audit.db_path,
            },
            "profiling": {
                "enabled": self.profiling.enabled,
                "slow_threshold_ms": self.profiling.slow_threshold_ms,
            },
            "rate_limiting": {
                "enabled": self.rate_limiting.enabled,
                "algorithm": self.rate_limiting.algorithm,
                "capacity": self.rate_limiting.capacity,
                "refill_rate": self.rate_limiting.refill_rate,
                "max_requests": self.rate_limiting.max_requests,
                "window_s": self.rate_limiting.window_s,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# ── PolicyManager ─────────────────────────────────────────────────────────────

_VALID_SEVERITIES = {"low", "medium", "high", "critical"}
_VALID_ACTIONS = {"allow", "block", "warn"}
_VALID_ALGORITHMS = {"token_bucket", "sliding_window"}
_VALID_PROVIDERS = {"openai", "anthropic", "custom"}


class PolicyManager:
    """
    Loads, validates, and applies policy configurations.

    Parameters
    ----------
    engine:
        Optional ``GuardrailEngine`` to populate with rules after loading.
        If ``None``, rules are parsed but not applied to any engine.
    replace_rules:
        If ``True`` (default), all existing rules in *engine* are cleared
        before the policy rules are loaded.  Set to ``False`` to append.
    """

    def __init__(
        self,
        engine: Optional[GuardrailEngine] = None,
        replace_rules: bool = True,
    ) -> None:
        self._engine = engine
        self._replace_rules = replace_rules
        self._current_config: Optional[PolicyConfig] = None

    # ── Public API ─────────────────────────────────────────────────────────

    def load(self, path: str) -> PolicyConfig:
        """
        Load a policy file from *path* (YAML or JSON).

        The file format is inferred from the extension:
        ``.yaml`` / ``.yml`` → YAML; ``.json`` → JSON.

        Raises ``PolicyValidationError`` for invalid content.
        Raises ``FileNotFoundError`` if the path does not exist.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Policy file not found: {path}")

        suffix = p.suffix.lower()
        text = p.read_text(encoding="utf-8")

        if suffix in (".yaml", ".yml"):
            raw = self._parse_yaml(text, path)
        elif suffix == ".json":
            raw = self._parse_json(text, path)
        else:
            raise PolicyValidationError(
                f"Unsupported file format '{suffix}'. Use .yaml, .yml, or .json"
            )

        return self.load_dict(raw)

    def load_dict(self, raw: Dict) -> PolicyConfig:
        """
        Load a policy from a Python dict.

        Validates, resolves environment variable expansions, and optionally
        applies rules to the configured ``GuardrailEngine``.
        """
        raw = self._resolve_env_vars(raw)
        config = self._validate(raw)
        self._current_config = config

        if self._engine is not None:
            self._apply_to_engine(config)

        return config

    def load_json_string(self, json_str: str) -> PolicyConfig:
        """Load a policy from a raw JSON string."""
        try:
            raw = json.loads(json_str)
        except json.JSONDecodeError as exc:
            raise PolicyValidationError(f"Invalid JSON: {exc}") from exc
        return self.load_dict(raw)

    def save(self, path: str, config: Optional[PolicyConfig] = None) -> None:
        """
        Serialize the current (or provided) config to a JSON file.

        YAML output is not generated to avoid the PyYAML dependency at
        runtime; use a YAML formatter externally if needed.
        """
        target = config or self._current_config
        if target is None:
            raise RuntimeError("No policy config loaded; call load() first.")
        Path(path).write_text(target.to_json())

    @property
    def current_config(self) -> Optional[PolicyConfig]:
        """The last successfully loaded ``PolicyConfig``, or ``None``."""
        return self._current_config

    def apply_to_engine(self, engine: GuardrailEngine, config: Optional[PolicyConfig] = None) -> int:
        """
        Apply *config* (or the last loaded config) to *engine*.

        Returns the number of rules applied.
        """
        target = config or self._current_config
        if target is None:
            raise RuntimeError("No policy config loaded; call load() first.")
        self._engine = engine
        self._apply_to_engine(target)
        return len(target.active_rules)

    def build_rate_limiter(self, config: Optional[PolicyConfig] = None):
        """
        Build and return a rate limiter configured from the policy.

        Returns ``None`` if rate limiting is disabled in the policy.
        """
        from rate_limiter import SlidingWindowLimiter, TokenBucketLimiter

        target = config or self._current_config
        if target is None or not target.rate_limiting.enabled:
            return None

        rl = target.rate_limiting
        if rl.algorithm == "token_bucket":
            return TokenBucketLimiter(capacity=rl.capacity, refill_rate=rl.refill_rate)
        if rl.algorithm == "sliding_window":
            return SlidingWindowLimiter(max_requests=rl.max_requests, window_s=rl.window_s)
        raise PolicyValidationError(f"Unknown algorithm '{rl.algorithm}'")

    # ── Validation ─────────────────────────────────────────────────────────

    def _validate(self, raw: Dict) -> PolicyConfig:
        if not isinstance(raw, dict):
            raise PolicyValidationError("Policy root must be a YAML/JSON object (dict)")

        guardrail_rules = self._validate_guardrails(raw.get("guardrails", []))
        llm_cfg = self._validate_llm(raw.get("llm", {}))
        audit_cfg = self._validate_audit(raw.get("audit", {}))
        profiling_cfg = self._validate_profiling(raw.get("profiling", {}))
        rl_cfg = self._validate_rate_limiting(raw.get("rate_limiting", {}))

        return PolicyConfig(
            guardrails=guardrail_rules,
            llm=llm_cfg,
            audit=audit_cfg,
            profiling=profiling_cfg,
            rate_limiting=rl_cfg,
            raw=raw,
        )

    def _validate_guardrails(self, rules_raw: Any) -> List[GuardrailRuleConfig]:
        if not isinstance(rules_raw, list):
            raise PolicyValidationError("'guardrails' must be a list")

        configs: List[GuardrailRuleConfig] = []
        seen_ids = set()

        for i, rule in enumerate(rules_raw):
            ctx = f"guardrails[{i}]"
            if not isinstance(rule, dict):
                raise PolicyValidationError(f"{ctx}: each rule must be a dict")

            rule_id = rule.get("id")
            if not rule_id or not isinstance(rule_id, str):
                raise PolicyValidationError(f"{ctx}: 'id' is required and must be a string")
            if rule_id in seen_ids:
                raise PolicyValidationError(f"{ctx}: duplicate rule id '{rule_id}'")
            seen_ids.add(rule_id)

            name = rule.get("name", rule_id)
            if not isinstance(name, str):
                raise PolicyValidationError(f"{ctx}['{rule_id}']: 'name' must be a string")

            severity = rule.get("severity", "medium")
            if severity not in _VALID_SEVERITIES:
                raise PolicyValidationError(
                    f"{ctx}['{rule_id}']: invalid severity '{severity}'. "
                    f"Use: {sorted(_VALID_SEVERITIES)}"
                )

            action = rule.get("action", "block")
            if action not in _VALID_ACTIONS:
                raise PolicyValidationError(
                    f"{ctx}['{rule_id}']: invalid action '{action}'. "
                    f"Use: {sorted(_VALID_ACTIONS)}"
                )

            patterns = rule.get("patterns", [])
            if not isinstance(patterns, list):
                raise PolicyValidationError(f"{ctx}['{rule_id}']: 'patterns' must be a list")
            for j, p in enumerate(patterns):
                if not isinstance(p, str):
                    raise PolicyValidationError(
                        f"{ctx}['{rule_id}'].patterns[{j}]: must be a string"
                    )
                try:
                    re.compile(p)
                except re.error as exc:
                    raise PolicyValidationError(
                        f"{ctx}['{rule_id}'].patterns[{j}]: invalid regex '{p}': {exc}"
                    ) from exc

            keywords = rule.get("keywords", [])
            if not isinstance(keywords, list):
                raise PolicyValidationError(f"{ctx}['{rule_id}']: 'keywords' must be a list")
            for j, kw in enumerate(keywords):
                if not isinstance(kw, str):
                    raise PolicyValidationError(
                        f"{ctx}['{rule_id}'].keywords[{j}]: must be a string"
                    )

            if not patterns and not keywords:
                raise PolicyValidationError(
                    f"{ctx}['{rule_id}']: at least one 'pattern' or 'keyword' is required"
                )

            enabled = rule.get("enabled", True)
            if not isinstance(enabled, bool):
                raise PolicyValidationError(
                    f"{ctx}['{rule_id}']: 'enabled' must be a boolean"
                )

            configs.append(GuardrailRuleConfig(
                id=rule_id,
                name=name,
                severity=severity,
                action=action,
                patterns=patterns,
                keywords=keywords,
                enabled=enabled,
            ))

        return configs

    def _validate_llm(self, raw: Any) -> LLMConfig:
        if not isinstance(raw, dict):
            raise PolicyValidationError("'llm' must be a dict")
        provider = raw.get("provider", "openai")
        if provider not in _VALID_PROVIDERS:
            raise PolicyValidationError(
                f"'llm.provider' must be one of {sorted(_VALID_PROVIDERS)}, got '{provider}'"
            )
        return LLMConfig(
            provider=provider,
            model=str(raw.get("model", "gpt-4o")),
            api_key=str(raw.get("api_key", "")),
            temperature=float(raw.get("temperature", 0.0)),
            max_tokens=int(raw.get("max_tokens", 1024)),
            timeout_s=float(raw.get("timeout_s", 30.0)),
        )

    def _validate_audit(self, raw: Any) -> AuditConfig:
        if not isinstance(raw, dict):
            raise PolicyValidationError("'audit' must be a dict")
        return AuditConfig(
            enabled=bool(raw.get("enabled", True)),
            retention_days=int(raw.get("retention_days", 90)),
            db_path=str(raw.get("db_path", "./audit_log.db")),
            export_path=str(raw.get("export_path", "./audit_logs/")),
        )

    def _validate_profiling(self, raw: Any) -> ProfilingConfig:
        if not isinstance(raw, dict):
            raise PolicyValidationError("'profiling' must be a dict")
        return ProfilingConfig(
            enabled=bool(raw.get("enabled", True)),
            slow_threshold_ms=float(raw.get("slow_threshold_ms", 200.0)),
        )

    def _validate_rate_limiting(self, raw: Any) -> RateLimitingConfig:
        if not isinstance(raw, dict):
            raise PolicyValidationError("'rate_limiting' must be a dict")
        algorithm = raw.get("algorithm", "token_bucket")
        if algorithm not in _VALID_ALGORITHMS:
            raise PolicyValidationError(
                f"'rate_limiting.algorithm' must be one of {sorted(_VALID_ALGORITHMS)}"
            )
        return RateLimitingConfig(
            enabled=bool(raw.get("enabled", False)),
            algorithm=algorithm,
            capacity=float(raw.get("capacity", 60.0)),
            refill_rate=float(raw.get("refill_rate", 1.0)),
            max_requests=int(raw.get("max_requests", 60)),
            window_s=float(raw.get("window_s", 60.0)),
        )

    # ── Apply to engine ────────────────────────────────────────────────────

    def _apply_to_engine(self, config: PolicyConfig) -> None:
        if self._replace_rules:
            # Remove all existing rules
            for rule_id in list(self._engine.rules.keys()):
                self._engine.remove_rule(rule_id)

        for rule_cfg in config.active_rules:
            self._engine.add_rule(rule_cfg.to_rule())

    # ── Parsers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_yaml(text: str, path: str) -> Dict:
        try:
            import yaml  # type: ignore
            result = yaml.safe_load(text)
            if result is None:
                return {}
            if not isinstance(result, dict):
                raise PolicyValidationError(f"YAML file '{path}' must contain a mapping")
            return result
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML policy files: pip install pyyaml"
            )
        except Exception as exc:  # yaml.YAMLError
            raise PolicyValidationError(f"Failed to parse YAML '{path}': {exc}") from exc

    @staticmethod
    def _parse_json(text: str, path: str) -> Dict:
        try:
            result = json.loads(text)
            if not isinstance(result, dict):
                raise PolicyValidationError(f"JSON file '{path}' must contain an object")
            return result
        except json.JSONDecodeError as exc:
            raise PolicyValidationError(f"Failed to parse JSON '{path}': {exc}") from exc

    # ── Env-var expansion ──────────────────────────────────────────────────

    @staticmethod
    def _resolve_env_vars(raw: Any) -> Any:
        """Recursively expand ``${VAR_NAME}`` placeholders with environment variables."""
        _ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")

        def _expand(value: Any) -> Any:
            if isinstance(value, str):
                def _replacer(match: re.Match) -> str:
                    var_name = match.group(1)
                    return os.environ.get(var_name, match.group(0))
                return _ENV_PATTERN.sub(_replacer, value)
            if isinstance(value, dict):
                return {k: _expand(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_expand(item) for item in value]
            return value

        return _expand(raw)


# ── Convenience helpers ───────────────────────────────────────────────────────

def load_policy(path: str, engine: Optional[GuardrailEngine] = None) -> PolicyConfig:
    """
    One-liner helper: load a policy file and optionally apply it to *engine*.

        config = load_policy("config.yaml", engine)
    """
    return PolicyManager(engine=engine).load(path)


def load_policy_dict(raw: Dict, engine: Optional[GuardrailEngine] = None) -> PolicyConfig:
    """Load a policy from a Python dict (useful in tests and notebooks)."""
    return PolicyManager(engine=engine).load_dict(raw)


# ── Default config.yaml example writer ───────────────────────────────────────

_EXAMPLE_YAML = """\
# Guardrails Policy Configuration
# Reference: https://github.com/AGI-Corporation/guardrails-

guardrails:
  - id: "pii_ssn"
    name: "Social Security Number Detection"
    severity: "critical"
    action: "block"
    patterns:
      - "\\\\d{3}-\\\\d{2}-\\\\d{4}"
    keywords: []
    enabled: true

  - id: "pii_credit_card"
    name: "Credit Card Detection"
    severity: "critical"
    action: "block"
    patterns:
      - "\\\\d{4}[\\\\s-]\\\\d{4}[\\\\s-]\\\\d{4}[\\\\s-]\\\\d{4}"
    keywords: []
    enabled: true

  - id: "prompt_injection"
    name: "Prompt Injection"
    severity: "high"
    action: "block"
    patterns: []
    keywords:
      - "ignore all previous instructions"
      - "disregard your guidelines"
      - "forget everything above"
    enabled: true

  - id: "content_safety"
    name: "Harmful Content"
    severity: "high"
    action: "block"
    keywords:
      - "make explosives"
      - "synthesize drugs"
    patterns: []
    enabled: true

llm:
  provider: "openai"
  model: "gpt-4o"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.0
  max_tokens: 1024

audit:
  enabled: true
  retention_days: 90
  db_path: "./audit_log.db"
  export_path: "./audit_logs/"

profiling:
  enabled: true
  slow_threshold_ms: 200

rate_limiting:
  enabled: false
  algorithm: "token_bucket"
  capacity: 60
  refill_rate: 1.0
"""


def write_example_config(path: str = "config.yaml") -> None:
    """Write an example ``config.yaml`` to *path*."""
    Path(path).write_text(_EXAMPLE_YAML, encoding="utf-8")
    print(f"✅  Example policy written to '{path}'")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardrails Policy Manager")
    sub = parser.add_subparsers(dest="command")

    cmd_load = sub.add_parser("load", help="Load and validate a policy file")
    cmd_load.add_argument("path", help="Path to the policy file (.yaml or .json)")
    cmd_load.add_argument("--json", action="store_true", help="Print parsed config as JSON")

    cmd_init = sub.add_parser("init", help="Write an example config.yaml")
    cmd_init.add_argument("--output", default="config.yaml", help="Output path")

    args = parser.parse_args()

    if args.command == "load":
        from guardrail_framework import GuardrailEngine
        engine = GuardrailEngine()
        pm = PolicyManager(engine=engine)
        try:
            config = pm.load(args.path)
            print(f"✅  Loaded {len(config.guardrails)} rules "
                  f"({len(config.active_rules)} enabled)")
            if args.json:
                print(config.to_json())
        except (PolicyValidationError, FileNotFoundError) as exc:
            print(f"❌  {exc}")
            raise SystemExit(1)

    elif args.command == "init":
        write_example_config(args.output)

    else:
        parser.print_help()
