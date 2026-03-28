"""
Tests for api_server.py (FastAPI endpoints)
"""

import pytest
from fastapi.testclient import TestClient

# We import the app and reset its shared state before each test
from api_server import api_app, _engine, _audit
from guardrail_framework import create_default_guardrails, create_default_test_cases


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_engine():
    """Reset shared engine state between tests to avoid interference."""
    _engine.rules.clear()
    _engine.test_cases.clear()
    for rule in create_default_guardrails():
        _engine.add_rule(rule)
    for tc in create_default_test_cases():
        _engine.add_test_case(tc)
    yield


@pytest.fixture()
def client():
    return TestClient(api_app)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------

class TestEvaluateEndpoint:
    def test_safe_text_allowed(self, client):
        response = client.post("/evaluate", json={"text": "Hello, how are you?"})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "allow"
        assert data["matched_rules"] == []
        assert "timestamp" in data

    def test_ssn_blocked(self, client):
        response = client.post("/evaluate", json={"text": "My SSN is 123-45-6789."})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "block"
        assert "pii_ssn" in data["matched_rules"]
        assert data["severity"] != "none"

    def test_credit_card_blocked(self, client):
        response = client.post("/evaluate", json={"text": "Card: 4111 1111 1111 1111"})
        assert response.status_code == 200
        assert response.json()["action"] == "block"

    def test_self_harm_blocked(self, client):
        response = client.post("/evaluate", json={"text": "I want to end my life"})
        assert response.status_code == 200
        assert response.json()["action"] == "block"

    def test_evaluate_with_metadata(self, client):
        response = client.post(
            "/evaluate",
            json={
                "text": "Normal text",
                "user_id": "user_1",
                "session_id": "sess_abc",
                "metadata": {"source": "web"},
            },
        )
        assert response.status_code == 200
        assert response.json()["action"] == "allow"

    def test_evaluate_empty_text(self, client):
        response = client.post("/evaluate", json={"text": ""})
        assert response.status_code == 200
        assert response.json()["action"] == "allow"


# ---------------------------------------------------------------------------
# GET /rules
# ---------------------------------------------------------------------------

class TestListRulesEndpoint:
    def test_returns_list(self, client):
        response = client.get("/rules")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_rule_structure(self, client):
        response = client.get("/rules")
        rules = response.json()
        for rule in rules:
            assert "id" in rule
            assert "name" in rule
            assert "category" in rule
            assert "severity" in rule
            assert "action" in rule
            assert "enabled" in rule

    def test_default_rules_present(self, client):
        response = client.get("/rules")
        rule_ids = {r["id"] for r in response.json()}
        assert "pii_ssn" in rule_ids
        assert "self_harm" in rule_ids


# ---------------------------------------------------------------------------
# POST /rules
# ---------------------------------------------------------------------------

class TestAddRuleEndpoint:
    def test_add_valid_rule(self, client):
        response = client.post(
            "/rules",
            json={
                "id": "test_custom_rule",
                "name": "Custom Rule",
                "category": "custom",
                "severity": "medium",
                "action": "block",
                "keywords": ["testword"],
                "patterns": [],
                "description": "A test rule",
                "enabled": True,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "added"
        assert data["rule_id"] == "test_custom_rule"

    def test_added_rule_affects_evaluation(self, client):
        # Add rule
        client.post(
            "/rules",
            json={
                "id": "block_xyz",
                "name": "Block XYZ",
                "category": "custom",
                "severity": "high",
                "action": "block",
                "keywords": ["xyzword"],
            },
        )
        # Evaluate text that contains the keyword
        response = client.post("/evaluate", json={"text": "This contains xyzword"})
        assert response.json()["action"] == "block"

    def test_add_rule_invalid_category(self, client):
        response = client.post(
            "/rules",
            json={
                "id": "bad_rule",
                "name": "Bad Rule",
                "category": "not_a_real_category",
                "severity": "low",
                "action": "block",
            },
        )
        assert response.status_code == 400

    def test_add_rule_invalid_severity(self, client):
        response = client.post(
            "/rules",
            json={
                "id": "bad_rule",
                "name": "Bad Rule",
                "category": "custom",
                "severity": "extreme",
                "action": "block",
            },
        )
        assert response.status_code == 400

    def test_add_rule_invalid_action(self, client):
        response = client.post(
            "/rules",
            json={
                "id": "bad_rule",
                "name": "Bad Rule",
                "category": "custom",
                "severity": "low",
                "action": "destroy",
            },
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /rules/{rule_id}
# ---------------------------------------------------------------------------

class TestDeleteRuleEndpoint:
    def test_delete_existing_rule(self, client):
        response = client.delete("/rules/pii_ssn")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["rule_id"] == "pii_ssn"

    def test_deleted_rule_no_longer_in_list(self, client):
        client.delete("/rules/pii_ssn")
        rules = client.get("/rules").json()
        rule_ids = {r["id"] for r in rules}
        assert "pii_ssn" not in rule_ids

    def test_delete_nonexistent_rule_returns_404(self, client):
        response = client.delete("/rules/does_not_exist")
        assert response.status_code == 404

    def test_delete_then_evaluate_no_longer_blocked(self, client):
        client.delete("/rules/pii_ssn")
        response = client.post("/evaluate", json={"text": "My SSN is 123-45-6789."})
        # Other rules might still match, but SSN rule is gone
        matched = response.json()["matched_rules"]
        assert "pii_ssn" not in matched


# ---------------------------------------------------------------------------
# POST /tests/run
# ---------------------------------------------------------------------------

class TestRunTestsEndpoint:
    def test_run_tests_returns_results(self, client):
        response = client.post("/tests/run")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "passed" in data
        assert "failed" in data
        assert "pass_rate" in data
        assert "report" in data

    def test_run_tests_all_pass_with_defaults(self, client):
        response = client.post("/tests/run")
        data = response.json()
        assert data["passed"] == data["total"]
        assert data["failed"] == 0

    def test_run_tests_pass_rate_format(self, client):
        response = client.post("/tests/run")
        data = response.json()
        assert 0 <= data["pass_rate"] <= 100

    def test_run_tests_report_is_string(self, client):
        response = client.post("/tests/run")
        assert isinstance(response.json()["report"], str)


# ---------------------------------------------------------------------------
# GET /audit/stats
# ---------------------------------------------------------------------------

class TestAuditStatsEndpoint:
    def test_audit_stats_returns_dict(self, client):
        response = client.get("/audit/stats")
        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "blocked" in data
        assert "allowed" in data
        assert "block_rate" in data

    def test_audit_stats_incremented_after_evaluate(self, client):
        initial = client.get("/audit/stats").json()["total"]
        client.post("/evaluate", json={"text": "Hello"})
        updated = client.get("/audit/stats").json()["total"]
        assert updated == initial + 1


# ---------------------------------------------------------------------------
# GET /audit/log
# ---------------------------------------------------------------------------

class TestAuditLogEndpoint:
    def test_audit_log_returns_list(self, client):
        response = client.get("/audit/log")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_audit_log_entry_structure(self, client):
        client.post("/evaluate", json={"text": "Hello world"})
        entries = client.get("/audit/log").json()
        assert len(entries) >= 1
        entry = entries[0]
        assert "id" in entry
        assert "timestamp" in entry
        assert "action_taken" in entry
        assert "severity" in entry
        assert "matched_rules" in entry

    def test_audit_log_filter_by_action(self, client):
        client.post("/evaluate", json={"text": "Hello world"})
        client.post("/evaluate", json={"text": "My SSN is 123-45-6789."})
        blocked = client.get("/audit/log?action=block").json()
        for entry in blocked:
            assert entry["action_taken"] == "block"

    def test_audit_log_limit_param(self, client):
        for _ in range(5):
            client.post("/evaluate", json={"text": "test text"})
        entries = client.get("/audit/log?limit=2").json()
        assert len(entries) <= 2
