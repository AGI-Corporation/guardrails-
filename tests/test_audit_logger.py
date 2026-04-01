"""Tests for audit_logger.py"""
import os
import pytest
from audit_logger import AuditLogger, AuditEntry


@pytest.fixture
def logger(tmp_path):
    db = str(tmp_path / "test_audit.db")
    return AuditLogger(db_path=db)


class TestAuditLogger:
    def test_log_with_entry_object(self, logger):
        entry = AuditEntry(
            input_text="test input",
            action_taken="allow",
            matched_rules=[],
            severity="low",
            risk_score=0.0,
        )
        row_id = logger.log(entry)
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_log_with_kwargs(self, logger):
        row_id = logger.log(
            input_text="My SSN is 123-45-6789",
            action_taken="block",
            matched_rules=["ssn"],
            severity="critical",
            risk_score=1.0,
        )
        assert isinstance(row_id, int)
        assert row_id > 0

    def test_get_logs_returns_dicts(self, logger):
        logger.log(input_text="hello", action_taken="allow", matched_rules=[], severity="low", risk_score=0.0)
        logs = logger.get_logs()
        assert isinstance(logs, list)
        assert len(logs) > 0
        assert isinstance(logs[0], dict)
        assert "action_taken" in logs[0]

    def test_get_logs_pagination(self, logger):
        for i in range(5):
            logger.log(input_text=f"text {i}", action_taken="allow", matched_rules=[], severity="low", risk_score=0.0)
        page1 = logger.get_logs(limit=3, offset=0)
        page2 = logger.get_logs(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2

    def test_search(self, logger):
        logger.log(input_text="unique_search_term", action_taken="block", matched_rules=["ssn"], severity="critical", risk_score=1.0)
        results = logger.search("unique_search_term")
        assert len(results) > 0
        assert results[0]["input_text"] == "unique_search_term"

    def test_export_csv(self, logger, tmp_path):
        logger.log(input_text="test", action_taken="allow", matched_rules=[], severity="low", risk_score=0.0)
        csv_path = str(tmp_path / "export.csv")
        logger.export_csv(csv_path)
        assert os.path.exists(csv_path)
        with open(csv_path) as f:
            content = f.read()
        assert "action_taken" in content
