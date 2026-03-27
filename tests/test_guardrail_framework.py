"""
Tests for guardrail_framework.py
Covers: enums, data classes, GuardrailEngine, ReportGenerator,
        create_default_guardrails, create_default_test_cases.
"""

import json
import os
import tempfile

import pytest

from guardrail_framework import (
    Action,
    EvaluationResult,
    GuardrailCategory,
    GuardrailCLI,
    GuardrailEngine,
    GuardrailRule,
    ReportGenerator,
    Severity,
    TestCase,
    TestResult,
    create_default_guardrails,
    create_default_test_cases,
)


# ---------------------------------------------------------------------------
# Enum sanity
# ---------------------------------------------------------------------------

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
        assert Action.TRANSFORM.value == "transform"
        assert Action.LOG.value == "log"

    def test_category_values(self):
        assert GuardrailCategory.PII_PROTECTION.value == "pii_protection"
        assert GuardrailCategory.HATE_SPEECH.value == "hate_speech"
        assert GuardrailCategory.SELF_HARM.value == "self_harm"
        assert GuardrailCategory.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# Data-class construction
# ---------------------------------------------------------------------------

class TestDataClasses:
    def test_guardrail_rule_defaults(self):
        rule = GuardrailRule(
            id="r1",
            name="Test Rule",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.ALLOW,
        )
        assert rule.id == "r1"
        assert rule.enabled is True
        assert rule.patterns == []
        assert rule.keywords == []
        assert rule.metadata == {}
        assert rule.description == ""

    def test_guardrail_rule_with_all_fields(self):
        rule = GuardrailRule(
            id="r2",
            name="Full Rule",
            category=GuardrailCategory.PII_PROTECTION,
            severity=Severity.CRITICAL,
            action=Action.BLOCK,
            patterns=[r"\d+"],
            keywords=["secret"],
            description="blocks secrets",
            enabled=False,
            metadata={"owner": "team"},
        )
        assert rule.patterns == [r"\d+"]
        assert rule.keywords == ["secret"]
        assert rule.enabled is False
        assert rule.metadata == {"owner": "team"}

    def test_test_case_defaults(self):
        tc = TestCase(id="t1", input_text="hello", expected_action="allow")
        assert tc.description == ""
        assert tc.tags == []

    def test_evaluation_result_timestamp_auto(self):
        result = EvaluationResult(
            text="hi",
            action="allow",
            matched_rules=[],
            severity="none",
        )
        assert result.timestamp  # not empty
        assert result.details == {}

    def test_test_result_fields(self):
        tr = TestResult(
            test_case_id="tc1",
            passed=True,
            expected="allow",
            got="allow",
            matched_rules=[],
        )
        assert tr.execution_time_ms == 0.0


# ---------------------------------------------------------------------------
# GuardrailEngine — rule management
# ---------------------------------------------------------------------------

class TestGuardrailEngineRuleManagement:
    def setup_method(self):
        self.engine = GuardrailEngine()

    def _make_rule(self, rule_id, **kwargs):
        defaults = dict(
            id=rule_id,
            name=f"Rule {rule_id}",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.LOW,
            action=Action.ALLOW,
        )
        defaults.update(kwargs)
        return GuardrailRule(**defaults)

    def test_add_rule(self):
        rule = self._make_rule("r1")
        self.engine.add_rule(rule)
        assert "r1" in self.engine.rules

    def test_remove_rule(self):
        rule = self._make_rule("r1")
        self.engine.add_rule(rule)
        self.engine.remove_rule("r1")
        assert "r1" not in self.engine.rules

    def test_remove_nonexistent_rule_is_noop(self):
        self.engine.remove_rule("does_not_exist")  # should not raise

    def test_add_test_case(self):
        tc = TestCase(id="tc1", input_text="hello", expected_action="allow")
        self.engine.add_test_case(tc)
        assert "tc1" in self.engine.test_cases

    def test_multiple_rules(self):
        for i in range(5):
            self.engine.add_rule(self._make_rule(f"r{i}"))
        assert len(self.engine.rules) == 5


# ---------------------------------------------------------------------------
# GuardrailEngine — evaluate()
# ---------------------------------------------------------------------------

class TestGuardrailEngineEvaluate:
    def setup_method(self):
        self.engine = GuardrailEngine()

    def _add_block_rule(self, rule_id, patterns=None, keywords=None,
                        severity=Severity.HIGH):
        rule = GuardrailRule(
            id=rule_id,
            name=rule_id,
            category=GuardrailCategory.CUSTOM,
            severity=severity,
            action=Action.BLOCK,
            patterns=patterns or [],
            keywords=keywords or [],
        )
        self.engine.add_rule(rule)

    def test_empty_engine_allows_everything(self):
        result = self.engine.evaluate("some dangerous text")
        assert result.action == "allow"
        assert result.matched_rules == []
        assert result.severity == "none"

    def test_keyword_match_blocks(self):
        self._add_block_rule("kw_rule", keywords=["badword"])
        result = self.engine.evaluate("this contains badword here")
        assert result.action == "block"
        assert "kw_rule" in result.matched_rules

    def test_keyword_match_case_insensitive(self):
        self._add_block_rule("kw_rule", keywords=["badword"])
        result = self.engine.evaluate("BADWORD in uppercase")
        assert result.action == "block"

    def test_pattern_match_blocks(self):
        self._add_block_rule("pattern_rule", patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])
        result = self.engine.evaluate("SSN: 123-45-6789")
        assert result.action == "block"

    def test_no_match_allows(self):
        self._add_block_rule("kw_rule", keywords=["restricted"])
        result = self.engine.evaluate("totally safe text")
        assert result.action == "allow"
        assert result.matched_rules == []

    def test_disabled_rule_not_applied(self):
        rule = GuardrailRule(
            id="disabled_rule",
            name="disabled",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            keywords=["badword"],
            enabled=False,
        )
        self.engine.add_rule(rule)
        result = self.engine.evaluate("badword is here")
        assert result.action == "allow"

    def test_highest_severity_wins(self):
        self._add_block_rule("low_rule", keywords=["lowrisk"], severity=Severity.LOW)
        self._add_block_rule("critical_rule", keywords=["highrisk"], severity=Severity.CRITICAL)
        result = self.engine.evaluate("text with lowrisk and highrisk content")
        assert result.severity == "critical"
        assert result.action == "block"
        assert "critical_rule" in result.matched_rules
        assert "low_rule" in result.matched_rules

    def test_warn_action_returned(self):
        rule = GuardrailRule(
            id="warn_rule",
            name="warn",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.MEDIUM,
            action=Action.WARN,
            keywords=["caution"],
        )
        self.engine.add_rule(rule)
        result = self.engine.evaluate("use caution here")
        assert result.action == "warn"

    def test_empty_string_input(self):
        self._add_block_rule("kw_rule", keywords=["test"])
        result = self.engine.evaluate("")
        assert result.action == "allow"

    def test_pattern_and_keyword_both_evaluated(self):
        rule = GuardrailRule(
            id="combo_rule",
            name="combo",
            category=GuardrailCategory.CUSTOM,
            severity=Severity.HIGH,
            action=Action.BLOCK,
            patterns=[r"\bsecret\d+\b"],
            keywords=["blocked"],
        )
        self.engine.add_rule(rule)
        # keyword match
        assert self.engine.evaluate("this is blocked").action == "block"
        # pattern match
        assert self.engine.evaluate("secret123 found").action == "block"
        # neither
        assert self.engine.evaluate("safe text").action == "allow"

    def test_ssn_pattern_with_default_rules(self):
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)
        result = self.engine.evaluate("My SSN is 123-45-6789")
        assert result.action == "block"
        assert "pii_ssn" in result.matched_rules

    def test_credit_card_pattern_with_default_rules(self):
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)
        result = self.engine.evaluate("Pay with 4111 1111 1111 1111 today")
        assert result.action == "block"
        assert "pii_credit_card" in result.matched_rules

    def test_self_harm_keyword_with_default_rules(self):
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)
        result = self.engine.evaluate("I want to end my life.")
        assert result.action == "block"
        assert "self_harm" in result.matched_rules

    def test_safe_text_with_default_rules(self):
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)
        result = self.engine.evaluate("Hello, what is the weather today?")
        assert result.action == "allow"


# ---------------------------------------------------------------------------
# GuardrailEngine — run_tests()
# ---------------------------------------------------------------------------

class TestGuardrailEngineRunTests:
    def setup_method(self):
        self.engine = GuardrailEngine()
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)

    def test_all_default_test_cases_pass(self):
        for tc in create_default_test_cases():
            self.engine.add_test_case(tc)
        results = self.engine.run_tests()
        assert len(results) == len(create_default_test_cases())
        for r in results:
            assert r.passed, f"Test {r.test_case_id} failed: expected {r.expected}, got {r.got}"

    def test_failing_test_case_detected(self):
        tc = TestCase(
            id="wrong_expectation",
            input_text="Hello world",
            expected_action="block",  # wrong expectation — safe text
        )
        self.engine.add_test_case(tc)
        results = self.engine.run_tests()
        failing = [r for r in results if r.test_case_id == "wrong_expectation"]
        assert len(failing) == 1
        assert failing[0].passed is False
        assert failing[0].expected == "block"
        assert failing[0].got == "allow"

    def test_execution_time_recorded(self):
        for tc in create_default_test_cases():
            self.engine.add_test_case(tc)
        results = self.engine.run_tests()
        for r in results:
            assert r.execution_time_ms >= 0

    def test_no_test_cases_returns_empty_list(self):
        results = self.engine.run_tests()
        assert results == []


# ---------------------------------------------------------------------------
# GuardrailEngine — export_rules()
# ---------------------------------------------------------------------------

class TestGuardrailEngineExport:
    def setup_method(self):
        self.engine = GuardrailEngine()
        for rule in create_default_guardrails():
            self.engine.add_rule(rule)

    def test_export_yaml(self):
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as f:
            path = f.name
        try:
            self.engine.export_rules(path, fmt="yaml")
            with open(path) as f:
                content = f.read()
            # Verify the file is non-empty and contains expected rule IDs
            assert "pii_ssn" in content
            assert "pii_credit_card" in content
        finally:
            os.unlink(path)

    def test_export_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            self.engine.export_rules(path, fmt="json")
            with open(path) as f:
                data = json.load(f)
            assert "pii_ssn" in data
        finally:
            os.unlink(path)

    def test_export_empty_engine(self):
        empty_engine = GuardrailEngine()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            empty_engine.export_rules(path, fmt="json")
            with open(path) as f:
                data = json.load(f)
            assert data == {}
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# create_default_guardrails / create_default_test_cases
# ---------------------------------------------------------------------------

class TestDefaultFactories:
    def test_default_guardrails_returns_list(self):
        rules = create_default_guardrails()
        assert isinstance(rules, list)
        assert len(rules) > 0

    def test_default_guardrails_all_valid(self):
        rules = create_default_guardrails()
        for rule in rules:
            assert isinstance(rule, GuardrailRule)
            assert rule.id
            assert rule.name

    def test_default_test_cases_returns_list(self):
        cases = create_default_test_cases()
        assert isinstance(cases, list)
        assert len(cases) > 0

    def test_default_test_cases_all_valid(self):
        cases = create_default_test_cases()
        for tc in cases:
            assert isinstance(tc, TestCase)
            assert tc.id
            assert tc.expected_action in ("allow", "block")


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------

class TestReportGenerator:
    def setup_method(self):
        self.rg = ReportGenerator()

    def _make_result(self, test_case_id, passed):
        return TestResult(
            test_case_id=test_case_id,
            passed=passed,
            expected="allow" if passed else "block",
            got="allow" if passed else "allow",
            matched_rules=[],
            execution_time_ms=1.5,
        )

    def test_report_header_present(self):
        report = self.rg.generate([])
        assert "Guardrail Test Report" in report

    def test_all_pass_summary(self):
        results = [self._make_result(f"tc{i}", True) for i in range(3)]
        report = self.rg.generate(results)
        assert "3/3" in report
        assert "100.0%" in report

    def test_partial_pass_summary(self):
        results = [
            self._make_result("tc1", True),
            self._make_result("tc2", False),
        ]
        report = self.rg.generate(results)
        assert "1/2" in report
        assert "50.0%" in report

    def test_empty_results(self):
        report = self.rg.generate([])
        assert "0/0" in report or "0%" in report

    def test_pass_and_fail_labels(self):
        results = [
            self._make_result("pass_test", True),
            self._make_result("fail_test", False),
        ]
        report = self.rg.generate(results)
        assert "PASS" in report
        assert "FAIL" in report

    def test_test_case_ids_in_report(self):
        results = [self._make_result("unique_id_abc", True)]
        report = self.rg.generate(results)
        assert "unique_id_abc" in report


# ---------------------------------------------------------------------------
# GuardrailCLI
# ---------------------------------------------------------------------------

class TestGuardrailCLI:
    def test_init_creates_engine_with_rules(self):
        cli = GuardrailCLI()
        assert len(cli.engine.rules) > 0

    def test_init_loads_test_cases(self):
        cli = GuardrailCLI()
        assert len(cli.engine.test_cases) > 0

    def test_start_evaluate_option(self, monkeypatch, capsys):
        """Test menu option 1: evaluate text."""
        inputs = iter(["1", "Hello world", "4"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli = GuardrailCLI()
        cli.start()
        captured = capsys.readouterr()
        assert "Action:" in captured.out

    def test_start_run_tests_option(self, monkeypatch, capsys):
        """Test menu option 2: run all tests."""
        inputs = iter(["2", "4"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli = GuardrailCLI()
        cli.start()
        captured = capsys.readouterr()
        assert "Guardrail Test Report" in captured.out

    def test_start_list_rules_option(self, monkeypatch, capsys):
        """Test menu option 3: list rules."""
        inputs = iter(["3", "4"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        cli = GuardrailCLI()
        cli.start()
        captured = capsys.readouterr()
        # Rules listed with ON/OFF status
        assert "ON" in captured.out or "OFF" in captured.out

    def test_start_exit_option(self, monkeypatch, capsys):
        """Test menu option 4: exit immediately."""
        monkeypatch.setattr("builtins.input", lambda _: "4")
        cli = GuardrailCLI()
        cli.start()  # should return without hanging
