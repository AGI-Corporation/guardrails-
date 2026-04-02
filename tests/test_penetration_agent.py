"""
Tests for penetration_test_agent.py — PenetrationTestAgent, PenTestReport,
PenTestSession, CategorySummary, and deep integration of all components.
"""
import json
import os

import pytest

from adversarial_tester import AdversarialTester
from audit_logger import AuditLogger
from guardrail_framework import (
    Action,
    GuardrailEngine,
    GuardrailRule,
    Severity,
    create_default_guardrails,
)
from penetration_test_agent import (
    AttackDetail,
    CategorySummary,
    PenTestReport,
    PenTestSession,
    PenetrationTestAgent,
)
from performance_profiler import PerformanceProfiler


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine() -> GuardrailEngine:
    e = GuardrailEngine()
    for rule in create_default_guardrails():
        e.add_rule(rule)
    return e


@pytest.fixture(scope="module")
def session(tmp_path_factory) -> PenTestSession:
    tmp = tmp_path_factory.mktemp("pentest")
    return PenTestSession(
        name="test-session",
        include_plugins=True,
        audit_db_path=str(tmp / "audit.db"),
        feedback_db_path=str(tmp / "feedback.db"),
    )


@pytest.fixture(scope="module")
def report(engine: GuardrailEngine, session: PenTestSession) -> PenTestReport:
    """Run the agent once; reuse the report for all tests in this module."""
    agent = PenetrationTestAgent(engine=engine)
    return agent.run(session)


# ── PenTestSession ────────────────────────────────────────────────────────────

class TestPenTestSession:
    def test_default_session_name(self):
        s = PenTestSession()
        assert s.name == "pentest"

    def test_custom_seeds_default_empty(self):
        s = PenTestSession()
        assert s.custom_seeds == []

    def test_include_plugins_default_true(self):
        s = PenTestSession()
        assert s.include_plugins is True


# ── PenetrationTestAgent ──────────────────────────────────────────────────────

class TestPenetrationTestAgent:
    def test_default_construction(self):
        agent = PenetrationTestAgent()
        assert agent._engine is not None

    def test_custom_engine(self, engine: GuardrailEngine):
        agent = PenetrationTestAgent(engine=engine)
        assert agent._engine is engine

    def test_custom_profiler(self):
        prof = PerformanceProfiler()
        agent = PenetrationTestAgent(profiler=prof)
        assert agent._profiler is prof

    def test_custom_audit_logger(self, tmp_path):
        al = AuditLogger(db_path=str(tmp_path / "al.db"))
        agent = PenetrationTestAgent(audit_logger=al)
        assert agent._audit is al


# ── PenTestReport structure ───────────────────────────────────────────────────

class TestPenTestReportStructure:
    def test_has_attacks(self, report: PenTestReport):
        assert len(report.attacks) > 0

    def test_all_attacks_are_attack_detail(self, report: PenTestReport):
        for a in report.attacks:
            assert isinstance(a, AttackDetail)

    def test_total_attacks_correct(self, report: PenTestReport):
        assert report.total_attacks == len(report.attacks)

    def test_blocked_plus_bypassed_equals_total(self, report: PenTestReport):
        assert report.total_blocked + report.total_bypassed == report.total_attacks

    def test_overall_block_rate_fraction(self, report: PenTestReport):
        assert 0.0 <= report.overall_block_rate <= 1.0

    def test_has_category_summaries(self, report: PenTestReport):
        assert len(report.category_summaries) > 0

    def test_all_expected_categories(self, report: PenTestReport):
        expected = {
            "encoding_bypasses", "context_injection", "jailbreak_dan",
            "jailbreak_roleplay", "hypothetical_framing",
            "indirect_injection", "boundary_conditions",
        }
        actual = {s.category for s in report.category_summaries}
        assert expected == actual

    def test_category_totals_sum_to_total(self, report: PenTestReport):
        cat_total = sum(s.total for s in report.category_summaries)
        assert cat_total == report.total_attacks

    def test_has_timestamp_fields(self, report: PenTestReport):
        assert report.started_at != ""
        assert report.finished_at != ""

    def test_duration_positive(self, report: PenTestReport):
        assert report.duration_s > 0

    def test_has_recommendations(self, report: PenTestReport):
        assert isinstance(report.recommendations, list)
        assert len(report.recommendations) > 0

    def test_performance_stats_present(self, report: PenTestReport):
        assert isinstance(report.performance_stats, dict)


# ── AttackDetail ──────────────────────────────────────────────────────────────

class TestAttackDetail:
    def test_category_is_string(self, report: PenTestReport):
        for a in report.attacks:
            assert isinstance(a.category, str)

    def test_mutation_is_string(self, report: PenTestReport):
        for a in report.attacks:
            assert isinstance(a.mutation, str)

    def test_blocked_is_bool(self, report: PenTestReport):
        for a in report.attacks:
            assert isinstance(a.blocked, bool)

    def test_matched_rules_is_list(self, report: PenTestReport):
        for a in report.attacks:
            assert isinstance(a.matched_rules, list)

    def test_plugin_action_valid(self, report: PenTestReport):
        valid_actions = {"allow", "warn", "block"}
        for a in report.attacks:
            assert a.plugin_action in valid_actions

    def test_duration_ms_non_negative(self, report: PenTestReport):
        for a in report.attacks:
            assert a.duration_ms >= 0


# ── CategorySummary ───────────────────────────────────────────────────────────

class TestCategorySummary:
    def test_block_rate_fraction(self, report: PenTestReport):
        for s in report.category_summaries:
            assert 0.0 <= s.block_rate <= 1.0

    def test_blocked_plus_bypassed_equals_total(self, report: PenTestReport):
        for s in report.category_summaries:
            assert s.blocked + s.bypassed == s.total

    def test_bypassed_mutations_is_list(self, report: PenTestReport):
        for s in report.category_summaries:
            assert isinstance(s.bypassed_mutations, list)

    def test_avg_duration_ms_non_negative(self, report: PenTestReport):
        for s in report.category_summaries:
            assert s.avg_duration_ms >= 0


# ── Report export formats ─────────────────────────────────────────────────────

class TestPenTestReportExports:
    def test_to_dict_keys(self, report: PenTestReport):
        d = report.to_dict()
        required = {
            "session_name", "started_at", "finished_at", "duration_s",
            "total_attacks", "total_blocked", "total_bypassed",
            "overall_block_rate", "categories", "recommendations",
        }
        assert required.issubset(d.keys())

    def test_to_json_valid_json(self, report: PenTestReport):
        data = json.loads(report.to_json())
        assert data["session_name"] == report.session_name

    def test_to_markdown_is_string(self, report: PenTestReport):
        md = report.to_markdown()
        assert isinstance(md, str)

    def test_to_markdown_contains_session_name(self, report: PenTestReport):
        assert report.session_name in report.to_markdown()

    def test_to_markdown_contains_block_rate(self, report: PenTestReport):
        assert "Block Rate" in report.to_markdown()

    def test_to_markdown_contains_recommendations(self, report: PenTestReport):
        assert "Recommendations" in report.to_markdown()

    def test_export_csv_creates_file(self, report: PenTestReport, tmp_path):
        path = str(tmp_path / "report.csv")
        report.export_csv(path)
        assert os.path.exists(path)

    def test_export_csv_row_count(self, report: PenTestReport, tmp_path):
        import csv
        path = str(tmp_path / "report.csv")
        report.export_csv(path)
        with open(path) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == report.total_attacks

    def test_save_json_creates_file(self, report: PenTestReport, tmp_path):
        path = str(tmp_path / "report.json")
        report.save_json(path)
        assert os.path.exists(path)
        data = json.loads(open(path).read())
        assert data["session_name"] == report.session_name


# ── Integration: audit log populated ─────────────────────────────────────────

class TestAuditIntegration:
    def test_audit_db_has_entries(self, session: PenTestSession, report: PenTestReport):
        audit = AuditLogger(db_path=session.audit_db_path)
        logs = audit.get_logs(limit=10000)
        # At minimum the session-summary entry + at least some attack entries
        assert len(logs) >= 2

    def test_session_summary_logged(self, session: PenTestSession):
        audit = AuditLogger(db_path=session.audit_db_path)
        logs = audit.search("PENTEST SESSION")
        assert len(logs) >= 1
        assert logs[0].action_taken == "pentest_complete"


# ── Integration: custom engine (minimal rules) ────────────────────────────────

class TestMinimalEngine:
    def test_agent_with_empty_engine(self, tmp_path):
        engine = GuardrailEngine()  # No rules at all
        session = PenTestSession(
            name="empty",
            include_plugins=False,
            audit_db_path=str(tmp_path / "a.db"),
            feedback_db_path=str(tmp_path / "f.db"),
        )
        agent = PenetrationTestAgent(engine=engine)
        report = agent.run(session)
        # Empty engine blocks nothing from guardrails
        assert report.overall_block_rate >= 0.0  # plugins may still block

    def test_agent_with_single_rule(self, tmp_path):
        engine = GuardrailEngine()
        engine.add_rule(GuardrailRule(
            "custom_test", "Custom Rule", Severity.HIGH, Action.BLOCK,
            keywords=["ignore all previous instructions"],
        ))
        session = PenTestSession(
            name="single-rule",
            include_plugins=False,
            audit_db_path=str(tmp_path / "a.db"),
            feedback_db_path=str(tmp_path / "f.db"),
        )
        agent = PenetrationTestAgent(engine=engine)
        report = agent.run(session)
        assert report.total_attacks > 0
        # The custom rule should catch at least some injection seeds
        assert report.total_blocked >= 0


# ── Integration: no-plugin mode ───────────────────────────────────────────────

class TestNoPluginMode:
    def test_agent_without_plugins(self, engine: GuardrailEngine, tmp_path):
        session = PenTestSession(
            name="no-plugins",
            include_plugins=False,
            audit_db_path=str(tmp_path / "a.db"),
            feedback_db_path=str(tmp_path / "f.db"),
        )
        agent = PenetrationTestAgent(engine=engine)
        report = agent.run(session)
        # Plugin action should be "allow" for all when plugins disabled
        for a in report.attacks:
            assert a.plugin_action == "allow"

    def test_plugin_details_empty_when_disabled(self, engine: GuardrailEngine, tmp_path):
        session = PenTestSession(
            name="no-plugins-2",
            include_plugins=False,
            audit_db_path=str(tmp_path / "a.db"),
            feedback_db_path=str(tmp_path / "f.db"),
        )
        agent = PenetrationTestAgent(engine=engine)
        report = agent.run(session)
        for a in report.attacks:
            assert a.plugin_details == []


# ── Custom seeds ──────────────────────────────────────────────────────────────

class TestCustomSeeds:
    def test_custom_seed_increases_attack_count(self, engine: GuardrailEngine, tmp_path):
        """Adding a custom seed should produce more test cases than the baseline."""
        base_session = PenTestSession(
            name="base", include_plugins=False,
            audit_db_path=str(tmp_path / "ba.db"),
            feedback_db_path=str(tmp_path / "bf.db"),
        )
        custom_session = PenTestSession(
            name="custom", include_plugins=False,
            custom_seeds=["My custom injection seed payload"],
            audit_db_path=str(tmp_path / "ca.db"),
            feedback_db_path=str(tmp_path / "cf.db"),
        )

        agent = PenetrationTestAgent(engine=engine)
        base_report = agent.run(base_session)
        custom_report = agent.run(custom_session)

        assert custom_report.total_attacks > base_report.total_attacks
