"""
Tests for audit_logger.py
"""
import json
import os
import tempfile
import pytest
from audit_logger import AuditEntry, AuditLogger


@pytest.fixture
def logger() -> AuditLogger:
    """In-memory SQLite logger for tests."""
    return AuditLogger(db_path=":memory:")


class TestAuditLogger:
    def test_log_with_entry_object(self, logger):
        entry = AuditEntry(input_text="hello", action_taken="allow", severity="low")
        row_id = logger.log(entry)
        assert row_id == 1

    def test_log_with_keyword_args(self, logger):
        row_id = logger.log(
            input_text="hello",
            action_taken="block",
            matched_rules=["ssn"],
            severity="critical",
            risk_score=1.0,
        )
        assert row_id == 1

    def test_get_logs_returns_dicts(self, logger):
        logger.log(input_text="text", action_taken="allow")
        logs = logger.get_logs()
        assert len(logs) == 1
        assert isinstance(logs[0], dict)
        assert "action_taken" in logs[0]

    def test_get_logs_dict_supports_get(self, logger):
        logger.log(input_text="text", action_taken="block")
        logs = logger.get_logs()
        assert logs[0].get("action_taken") == "block"

    def test_get_logs_pagination(self, logger):
        for i in range(5):
            logger.log(input_text=f"text{i}", action_taken="allow")
        page1 = logger.get_logs(limit=3, offset=0)
        page2 = logger.get_logs(limit=3, offset=3)
        assert len(page1) == 3
        assert len(page2) == 2

    def test_search_by_action(self, logger):
        logger.log(input_text="blocked text", action_taken="block")
        logger.log(input_text="allowed text", action_taken="allow")
        results = logger.search("block")
        assert len(results) == 1
        assert results[0]["action_taken"] == "block"

    def test_matched_rules_serialised(self, logger):
        logger.log(input_text="t", action_taken="block", matched_rules=["ssn", "credit_card"])
        logs = logger.get_logs()
        assert logs[0]["matched_rules"] == ["ssn", "credit_card"]

    def test_metadata_round_trip(self, logger):
        meta = {"key": "value", "nested": {"a": 1}}
        logger.log(input_text="t", action_taken="allow", metadata=meta)
        logs = logger.get_logs()
        assert logs[0]["metadata"] == meta

    def test_get_metrics(self, logger):
        logger.log(input_text="a", action_taken="block")
        logger.log(input_text="b", action_taken="allow")
        logger.log(input_text="c", action_taken="warn")
        metrics = logger.get_metrics()
        assert metrics["total_evaluations"] == 3
        assert metrics["blocked"] == 1
        assert metrics["allowed"] == 1
        assert metrics["warned"] == 1

    def test_get_metrics_empty(self, logger):
        metrics = logger.get_metrics()
        assert metrics["total_evaluations"] == 0
        assert metrics["block_rate"] == 0.0

    def test_export_csv(self, logger):
        logger.log(input_text="text", action_taken="allow")
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            path = tmp.name
        try:
            logger.export_csv(path)
            with open(path) as f:
                content = f.read()
            assert "action_taken" in content
            assert "allow" in content
        finally:
            os.unlink(path)

    def test_get_log_entries_returns_audit_entries(self, logger):
        logger.log(input_text="t", action_taken="allow")
        entries = logger.get_log_entries()
        assert len(entries) == 1
        assert isinstance(entries[0], AuditEntry)

    def test_multiple_logs_ordered_newest_first(self, logger):
        import time
        logger.log(input_text="first", action_taken="allow")
        time.sleep(0.01)
        logger.log(input_text="second", action_taken="block")
        logs = logger.get_logs()
        # Most recent should be first
        assert logs[0]["input_text"] == "second"
