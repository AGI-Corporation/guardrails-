"""
Tests for api_server.py
Covers: /health, /evaluate, /rules (GET, POST, DELETE), /tests/run,
        /audit/stats, /audit/log endpoints using FastAPI TestClient.

The module-level shared state (_engine, _audit) means we isolate tests
by using the TestClient with the real app and relying on fresh defaults.
"""

import pytest
from fastapi.testclient import TestClient

from api_server import api_app


@pytest.fixture
def client():
    """Return a TestClient bound to the FastAPI app."""
    with TestClient(api_app) as c:
        yield c


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_timestamp(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data
        assert data["timestamp"]  # non-empty


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------

class TestEvaluateEndpoint:
    def test_safe_text_allowed(self, client):
        response = client.post("/evaluate", json={"text": "Hello, how are you?"})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "allow"

    def test_ssn_text_blocked(self, client):
        response = client.post("/evaluate", json={"text": "My SSN is 123-45-6789"})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "block"
        assert "pii_ssn" in data["matched_rules"]

    def test_credit_card_text_blocked(self, client):
        response = client.post("/evaluate", json={"text": "Card: 4111 1111 1111 1111"})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "block"

    def test_response_has_required_fields(self, client):
        response = client.post("/evaluate", json={"text": "hello"})
        data = response.json()
        for field in ("action", "matched_rules", "severity", "timestamp"):
            assert field in data

    def test_evaluate_with_optional_fields(self, client):
        payload = {
            "text": "test text",
            "user_id": "user123",
            "session_id": "sess456",
            "metadata": {"source": "test"},
        }
        response = client.post("/evaluate", json=payload)
        assert response.status_code == 200

    def test_self_harm_text_blocked(self, client):
        response = client.post("/evaluate", json={"text": "I want to end my life."})
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "block"


# ---------------------------------------------------------------------------
# GET /rules
# ---------------------------------------------------------------------------

class TestListRulesEndpoint:
    def test_list_rules_returns_200(self, client):
        response = client.get("/rules")
        assert response.status_code == 200

    def test_list_rules_returns_list(self, client):
        data = client.get("/rules").json()
        assert isinstance(data, list)

    def test_list_rules_non_empty(self, client):
        data = client.get("/rules").json()
        assert len(data) > 0

    def test_rule_has_required_fields(self, client):
        data = client.get("/rules").json()
        rule = data[0]
        for field in ("id", "name", "category", "severity", "action", "enabled"):
            assert field in rule


# ---------------------------------------------------------------------------
# POST /rules
# ---------------------------------------------------------------------------

class TestAddRuleEndpoint:
    def _valid_rule_payload(self, rule_id="test_api_rule_001"):
        return {
            "id": rule_id,
            "name": "Test API Rule",
            "category": "custom",
            "severity": "low",
            "action": "warn",
            "keywords": ["testword"],
        }

    def test_add_rule_returns_200(self, client):
        response = client.post("/rules", json=self._valid_rule_payload("unique_rule_add_1"))
        assert response.status_code == 200

    def test_add_rule_status_added(self, client):
        payload = self._valid_rule_payload("unique_rule_add_2")
        data = client.post("/rules", json=payload).json()
        assert data["status"] == "added"
        assert data["rule_id"] == "unique_rule_add_2"

    def test_add_rule_invalid_category_returns_400(self, client):
        payload = self._valid_rule_payload("bad_cat_rule")
        payload["category"] = "invalid_category"
        response = client.post("/rules", json=payload)
        assert response.status_code == 400

    def test_add_rule_invalid_severity_returns_400(self, client):
        payload = self._valid_rule_payload("bad_sev_rule")
        payload["severity"] = "invalid_severity"
        response = client.post("/rules", json=payload)
        assert response.status_code == 400

    def test_add_rule_invalid_action_returns_400(self, client):
        payload = self._valid_rule_payload("bad_act_rule")
        payload["action"] = "invalid_action"
        response = client.post("/rules", json=payload)
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /rules/{rule_id}
# ---------------------------------------------------------------------------

class TestDeleteRuleEndpoint:
    def test_delete_existing_rule(self, client):
        # First add a rule to delete
        rule_id = "rule_to_delete_xyz"
        client.post("/rules", json={
            "id": rule_id,
            "name": "Temporary",
            "category": "custom",
            "severity": "low",
            "action": "allow",
        })
        response = client.delete(f"/rules/{rule_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"
        assert data["rule_id"] == rule_id

    def test_delete_nonexistent_rule_returns_404(self, client):
        response = client.delete("/rules/nonexistent_rule_99999")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# POST /tests/run
# ---------------------------------------------------------------------------

class TestRunTestsEndpoint:
    def test_run_tests_returns_200(self, client):
        response = client.post("/tests/run")
        assert response.status_code == 200

    def test_run_tests_response_fields(self, client):
        data = client.post("/tests/run").json()
        for field in ("total", "passed", "failed", "pass_rate", "report"):
            assert field in data

    def test_run_tests_counts_correct(self, client):
        data = client.post("/tests/run").json()
        assert data["total"] == data["passed"] + data["failed"]

    def test_run_tests_report_is_string(self, client):
        data = client.post("/tests/run").json()
        assert isinstance(data["report"], str)
        assert "Guardrail Test Report" in data["report"]

    def test_run_tests_pass_rate_range(self, client):
        data = client.post("/tests/run").json()
        assert 0 <= data["pass_rate"] <= 100


# ---------------------------------------------------------------------------
# GET /audit/stats
# ---------------------------------------------------------------------------

class TestAuditStatsEndpoint:
    def test_audit_stats_returns_200(self, client):
        response = client.get("/audit/stats")
        assert response.status_code == 200

    def test_audit_stats_has_required_fields(self, client):
        data = client.get("/audit/stats").json()
        for field in ("total", "blocked", "allowed", "block_rate", "by_severity"):
            assert field in data

    def test_audit_stats_totals_add_up(self, client):
        data = client.get("/audit/stats").json()
        assert data["allowed"] == data["total"] - data["blocked"]


# ---------------------------------------------------------------------------
# GET /audit/log
# ---------------------------------------------------------------------------

class TestAuditLogEndpoint:
    def test_audit_log_returns_200(self, client):
        response = client.get("/audit/log")
        assert response.status_code == 200

    def test_audit_log_returns_list(self, client):
        data = client.get("/audit/log").json()
        assert isinstance(data, list)

    def test_audit_log_after_evaluate(self, client):
        # Evaluate something to create an entry
        client.post("/evaluate", json={"text": "some test text for audit"})
        data = client.get("/audit/log").json()
        assert len(data) >= 1

    def test_audit_log_entry_fields(self, client):
        client.post("/evaluate", json={"text": "audit entry test text"})
        data = client.get("/audit/log").json()
        if data:
            entry = data[0]
            for field in ("id", "timestamp", "action_taken", "severity",
                          "matched_rules", "user_id"):
                assert field in entry

    def test_audit_log_filter_by_action(self, client):
        # Safe text -> allow
        client.post("/evaluate", json={"text": "completely safe content here"})
        data = client.get("/audit/log", params={"action": "allow"}).json()
        for entry in data:
            assert entry["action_taken"] == "allow"

    def test_audit_log_limit_parameter(self, client):
        # Make several evaluations
        for i in range(5):
            client.post("/evaluate", json={"text": f"text number {i}"})
        data = client.get("/audit/log", params={"limit": 2}).json()
        assert len(data) <= 2
