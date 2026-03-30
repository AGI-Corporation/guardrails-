"""
Tests for api_server.py
Uses FastAPI's TestClient to exercise all REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from api_server import api_app


@pytest.fixture(scope="module")
def client():
    return TestClient(api_app)


# ── /health ───────────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data


# ── /evaluate ─────────────────────────────────────────────────────────────

class TestEvaluateEndpoint:

    def test_safe_text_allowed(self, client):
        resp = client.post("/evaluate", json={"text": "Hello, how are you?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "allow"
        assert data["matched_rules"] == []

    def test_ssn_blocked(self, client):
        resp = client.post("/evaluate", json={"text": "My SSN is 123-45-6789."})
        assert resp.status_code == 200
        data = resp.json()
        assert data["action"] == "block"
        assert data["severity"] == "critical"

    def test_credit_card_blocked(self, client):
        resp = client.post("/evaluate", json={"text": "Card: 4111 1111 1111 1111"})
        assert resp.status_code == 200
        assert resp.json()["action"] == "block"

    def test_evaluate_returns_timestamp(self, client):
        resp = client.post("/evaluate", json={"text": "test"})
        assert resp.status_code == 200
        assert "timestamp" in resp.json()

    def test_evaluate_with_optional_metadata(self, client):
        resp = client.post("/evaluate", json={
            "text": "Hello",
            "user_id": "user_1",
            "session_id": "sess_abc",
            "metadata": {"source": "web"},
        })
        assert resp.status_code == 200
        assert resp.json()["action"] == "allow"


# ── /rules ────────────────────────────────────────────────────────────────

class TestRulesEndpoint:

    def test_list_rules_returns_list(self, client):
        resp = client.get("/rules")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_list_rules_have_expected_keys(self, client):
        resp = client.get("/rules")
        rule = resp.json()[0]
        assert "id" in rule
        assert "name" in rule
        assert "category" in rule
        assert "severity" in rule
        assert "action" in rule
        assert "enabled" in rule

    def test_add_rule(self, client):
        payload = {
            "id": "test_api_rule",
            "name": "API Test Rule",
            "category": "custom",
            "severity": "low",
            "action": "warn",
            "keywords": ["testword"],
            "patterns": [],
            "description": "Added via API test",
            "enabled": True,
        }
        resp = client.post("/rules", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "added"
        assert data["rule_id"] == "test_api_rule"

    def test_add_rule_bad_category(self, client):
        payload = {
            "id": "bad_rule",
            "name": "Bad",
            "category": "NOT_VALID_CATEGORY",
            "severity": "low",
            "action": "warn",
        }
        resp = client.post("/rules", json=payload)
        assert resp.status_code == 400

    def test_delete_rule(self, client):
        # First add a rule to delete
        client.post("/rules", json={
            "id": "rule_to_delete",
            "name": "Delete Me",
            "category": "custom",
            "severity": "low",
            "action": "log",
        })
        resp = client.delete("/rules/rule_to_delete")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_nonexistent_rule_returns_404(self, client):
        resp = client.delete("/rules/does_not_exist_xyz")
        assert resp.status_code == 404


# ── /tests/run ────────────────────────────────────────────────────────────

class TestRunTestsEndpoint:

    def test_run_tests_returns_summary(self, client):
        resp = client.post("/tests/run")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "passed" in data
        assert "failed" in data
        assert "pass_rate" in data
        assert "report" in data

    def test_all_tests_pass(self, client):
        resp = client.post("/tests/run")
        data = resp.json()
        assert data["failed"] == 0


# ── /audit/* ──────────────────────────────────────────────────────────────

class TestAuditEndpoints:

    def test_audit_stats_returns_dict(self, client):
        resp = client.get("/audit/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data

    def test_audit_log_returns_list(self, client):
        resp = client.get("/audit/log")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_audit_log_limit_param(self, client):
        # Evaluate a few items first
        for _ in range(3):
            client.post("/evaluate", json={"text": "hello"})
        resp = client.get("/audit/log?limit=2")
        assert resp.status_code == 200
        assert len(resp.json()) <= 2
