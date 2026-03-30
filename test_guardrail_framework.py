"""
Tests for guardrail_framework.py
Covers GuardrailRule, GuardrailEngine, and create_default_guardrails.
"""

import pytest
from guardrail_framework import (
    GuardrailRule,
    GuardrailEngine,
    Severity,
    Action,
    create_default_guardrails,
)


# ── GuardrailRule ──────────────────────────────────────────────────────────

class TestGuardrailRule:
    def test_matches_pattern_ssn(self):
        rule = GuardrailRule("ssn", "SSN", Severity.CRITICAL, Action.BLOCK,
                             patterns=[r"\d{3}-\d{2}-\d{4}"])
        assert rule.matches("My SSN is 123-45-6789") is True

    def test_no_match_pattern(self):
        rule = GuardrailRule("ssn", "SSN", Severity.CRITICAL, Action.BLOCK,
                             patterns=[r"\d{3}-\d{2}-\d{4}"])
        assert rule.matches("Hello world, no numbers here") is False

    def test_pattern_case_insensitive(self):
        rule = GuardrailRule("greeting", "Greeting", Severity.LOW, Action.WARN,
                             patterns=[r"hello"])
        assert rule.matches("HELLO world") is True

    def test_matches_keyword(self):
        rule = GuardrailRule("bad", "BadWord", Severity.MEDIUM, Action.BLOCK,
                             keywords=["badword"])
        assert rule.matches("This contains badword inside it") is True

    def test_keyword_case_insensitive(self):
        rule = GuardrailRule("bad", "BadWord", Severity.MEDIUM, Action.BLOCK,
                             keywords=["badword"])
        assert rule.matches("BADWORD is here") is True

    def test_no_match_keyword(self):
        rule = GuardrailRule("bad", "BadWord", Severity.MEDIUM, Action.BLOCK,
                             keywords=["badword"])
        assert rule.matches("This is perfectly fine") is False

    def test_empty_rule_never_matches(self):
        rule = GuardrailRule("empty", "Empty", Severity.LOW, Action.ALLOW)
        assert rule.matches("any text whatsoever") is False

    def test_rule_with_both_pattern_and_keyword(self):
        rule = GuardrailRule("combo", "Combo", Severity.HIGH, Action.BLOCK,
                             patterns=[r"\d{3}-\d{2}-\d{4}"],
                             keywords=["danger"])
        # Matches via keyword
        assert rule.matches("This is danger") is True
        # Matches via pattern
        assert rule.matches("SSN 123-45-6789") is True
        # Neither
        assert rule.matches("safe text") is False

    def test_rule_severity_values(self):
        for severity in Severity:
            rule = GuardrailRule("r", "R", severity, Action.BLOCK,
                                 keywords=["x"])
            assert rule.matches("x") is True

    def test_rule_action_values(self):
        for action in Action:
            rule = GuardrailRule("r", "R", Severity.LOW, action,
                                 keywords=["x"])
            assert rule.matches("x") is True

    def test_partial_keyword_no_match(self):
        """Keyword matching should be substring-based (not word boundary)."""
        rule = GuardrailRule("bad", "BadWord", Severity.LOW, Action.BLOCK,
                             keywords=["harm"])
        # "harm" appears inside "pharmacy" - current implementation uses 'in' so it matches
        assert rule.matches("pharmacy sells medicine") is True

    def test_empty_text_no_match(self):
        rule = GuardrailRule("r", "R", Severity.LOW, Action.BLOCK,
                             keywords=["secret"], patterns=[r"\d+"])
        assert rule.matches("") is False


# ── GuardrailEngine ────────────────────────────────────────────────────────

class TestGuardrailEngine:
    def test_empty_engine_allows_everything(self):
        engine = GuardrailEngine()
        result = engine.evaluate("anything goes")
        assert result["action"] == Action.ALLOW
        assert result["matches"] == []

    def test_single_rule_block(self):
        engine = GuardrailEngine()
        rule = GuardrailRule("r1", "R1", Severity.HIGH, Action.BLOCK,
                             keywords=["secret"])
        engine.add_rule(rule)
        result = engine.evaluate("my secret password")
        assert result["action"] == Action.BLOCK
        assert "r1" in result["matches"]

    def test_single_rule_no_match_allows(self):
        engine = GuardrailEngine()
        rule = GuardrailRule("r1", "R1", Severity.HIGH, Action.BLOCK,
                             keywords=["secret"])
        engine.add_rule(rule)
        result = engine.evaluate("safe message")
        assert result["action"] == Action.ALLOW
        assert result["matches"] == []

    def test_multiple_rules_all_match(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule("r1", "R1", Severity.HIGH, Action.BLOCK,
                                      keywords=["bad"]))
        engine.add_rule(GuardrailRule("r2", "R2", Severity.HIGH, Action.BLOCK,
                                      keywords=["evil"]))
        result = engine.evaluate("bad and evil text")
        assert result["action"] == Action.BLOCK
        assert "r1" in result["matches"]
        assert "r2" in result["matches"]

    def test_multiple_rules_partial_match(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule("r1", "R1", Severity.HIGH, Action.BLOCK,
                                      keywords=["bad"]))
        engine.add_rule(GuardrailRule("r2", "R2", Severity.HIGH, Action.BLOCK,
                                      keywords=["evil"]))
        result = engine.evaluate("bad text only")
        assert result["action"] == Action.BLOCK
        assert "r1" in result["matches"]
        assert "r2" not in result["matches"]

    def test_add_rule_overwrites_same_id(self):
        engine = GuardrailEngine()
        rule_v1 = GuardrailRule("r1", "V1", Severity.LOW, Action.BLOCK,
                                keywords=["old"])
        rule_v2 = GuardrailRule("r1", "V2", Severity.HIGH, Action.BLOCK,
                                keywords=["new"])
        engine.add_rule(rule_v1)
        engine.add_rule(rule_v2)
        # Old keyword should no longer be checked
        assert engine.evaluate("old text")["action"] == Action.ALLOW
        assert engine.evaluate("new text")["action"] == Action.BLOCK

    def test_evaluate_with_pattern(self):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule("ssn", "SSN", Severity.CRITICAL, Action.BLOCK,
                                      patterns=[r"\d{3}-\d{2}-\d{4}"]))
        result = engine.evaluate("My SSN is 123-45-6789 please keep it safe")
        assert result["action"] == Action.BLOCK
        assert "ssn" in result["matches"]

    def test_evaluate_returns_action_enum(self):
        engine = GuardrailEngine()
        result = engine.evaluate("hello")
        assert isinstance(result["action"], Action)


# ── create_default_guardrails ──────────────────────────────────────────────

class TestCreateDefaultGuardrails:
    def test_returns_list(self):
        rules = create_default_guardrails()
        assert isinstance(rules, list)
        assert len(rules) >= 1

    def test_ssn_rule_present(self):
        rules = create_default_guardrails()
        assert any(r.id == "ssn" for r in rules)

    def test_ssn_rule_matches_valid_ssn(self):
        rules = create_default_guardrails()
        ssn_rule = next(r for r in rules if r.id == "ssn")
        assert ssn_rule.matches("123-45-6789") is True
        assert ssn_rule.matches("000-00-0000") is True

    def test_ssn_rule_does_not_match_plain_text(self):
        rules = create_default_guardrails()
        ssn_rule = next(r for r in rules if r.id == "ssn")
        assert ssn_rule.matches("hello world") is False

    def test_default_rules_integrate_with_engine(self):
        engine = GuardrailEngine()
        for rule in create_default_guardrails():
            engine.add_rule(rule)
        result = engine.evaluate("My SSN is 123-45-6789")
        assert result["action"] == Action.BLOCK


# ── Enum values ────────────────────────────────────────────────────────────

class TestEnums:
    def test_severity_values(self):
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"
        assert Severity.CRITICAL.value == "critical"

    def test_action_values(self):
        assert Action.ALLOW.value == "allow"
        assert Action.BLOCK.value == "block"
        assert Action.WARN.value == "warn"
