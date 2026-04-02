"""
Tests for audit_logger.py — logging, retrieval, search, export.
"""
import csv
import json
import os
import tempfile

import pytest

from audit_logger import AuditEntry, AuditLogger


@pytest.fixture()
def logger(tmp_path) -> AuditLogger:
    return AuditLogger(db_path=str(tmp_path / "test_audit.db"))


# ── log() — both signatures ───────────────────────────────────────────────────

class TestLog:
    def test_log_with_entry_object(self, logger: AuditLogger):
        entry = AuditEntry(
            input_text="hello",
            action_taken="allow",
            matched_rules=[],
            severity="low",
        )
        row_id = logger.log(entry)
        assert row_id is not None and row_id > 0

    def test_log_with_kwargs(self, logger: AuditLogger):
        row_id = logger.log(
            input_text="blocked text",
            action_taken="block",
            matched_rules=["ssn"],
            severity="critical",
        )
        assert row_id is not None and row_id > 0

    def test_log_with_all_kwargs(self, logger: AuditLogger):
        row_id = logger.log(
            input_text="test",
            action_taken="warn",
            matched_rules=["email_pii"],
            severity="high",
            user_id="user123",
            session_id="sess456",
            metadata={"key": "value"},
        )
        assert row_id is not None


# ── get_logs() ────────────────────────────────────────────────────────────────

class TestGetLogs:
    def test_get_logs_returns_entries(self, logger: AuditLogger):
        logger.log(input_text="a", action_taken="allow", matched_rules=[], severity="low")
        logs = logger.get_logs()
        assert len(logs) == 1

    def test_get_logs_limit(self, logger: AuditLogger):
        for i in range(10):
            logger.log(input_text=f"text{i}", action_taken="allow", matched_rules=[], severity="low")
        logs = logger.get_logs(limit=5)
        assert len(logs) == 5

    def test_get_logs_offset(self, logger: AuditLogger):
        for i in range(5):
            logger.log(input_text=f"text{i}", action_taken="allow", matched_rules=[], severity="low")
        logs_all = logger.get_logs(limit=100, offset=0)
        logs_offset = logger.get_logs(limit=100, offset=2)
        assert len(logs_offset) == len(logs_all) - 2

    def test_get_logs_empty_db(self, logger: AuditLogger):
        assert logger.get_logs() == []

    def test_log_fields_preserved(self, logger: AuditLogger):
        logger.log(
            input_text="my text",
            action_taken="block",
            matched_rules=["ssn", "credit_card"],
            severity="critical",
            user_id="u1",
        )
        entries = logger.get_logs()
        entry = entries[0]
        assert entry.input_text == "my text"
        assert entry.action_taken == "block"
        assert "ssn" in entry.matched_rules
        assert "credit_card" in entry.matched_rules
        assert entry.severity == "critical"
        assert entry.user_id == "u1"


# ── search() ──────────────────────────────────────────────────────────────────

class TestSearch:
    def test_search_by_input_text(self, logger: AuditLogger):
        logger.log(input_text="My SSN is 123-45-6789", action_taken="block", matched_rules=[], severity="critical")
        logger.log(input_text="Hello world", action_taken="allow", matched_rules=[], severity="low")
        results = logger.search("SSN")
        assert len(results) == 1

    def test_search_by_action(self, logger: AuditLogger):
        logger.log(input_text="text1", action_taken="block", matched_rules=[], severity="high")
        logger.log(input_text="text2", action_taken="allow", matched_rules=[], severity="low")
        results = logger.search("block")
        assert len(results) == 1

    def test_search_no_match(self, logger: AuditLogger):
        logger.log(input_text="irrelevant", action_taken="allow", matched_rules=[], severity="low")
        results = logger.search("zzz_no_match")
        assert results == []


# ── export_csv() ──────────────────────────────────────────────────────────────

class TestExportCsv:
    def test_export_creates_file(self, logger: AuditLogger, tmp_path):
        logger.log(input_text="text", action_taken="allow", matched_rules=[], severity="low")
        out = str(tmp_path / "out.csv")
        logger.export_csv(out)
        assert os.path.exists(out)

    def test_export_has_header(self, logger: AuditLogger, tmp_path):
        logger.log(input_text="text", action_taken="allow", matched_rules=[], severity="low")
        out = str(tmp_path / "out.csv")
        logger.export_csv(out)
        with open(out) as f:
            reader = csv.DictReader(f)
            assert "input_text" in reader.fieldnames
            assert "action_taken" in reader.fieldnames

    def test_export_row_count(self, logger: AuditLogger, tmp_path):
        for i in range(3):
            logger.log(input_text=f"text{i}", action_taken="allow", matched_rules=[], severity="low")
        out = str(tmp_path / "out.csv")
        logger.export_csv(out)
        with open(out) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3

    def test_export_empty_db_no_error(self, logger: AuditLogger, tmp_path):
        out = str(tmp_path / "empty.csv")
        logger.export_csv(out)  # Should not raise


# ── AuditEntry defaults ───────────────────────────────────────────────────────

class TestAuditEntryDefaults:
    def test_timestamp_auto_set(self):
        entry = AuditEntry(input_text="x", action_taken="allow")
        assert entry.timestamp != ""

    def test_matched_rules_default_empty(self):
        entry = AuditEntry(input_text="x", action_taken="allow")
        assert entry.matched_rules == []

    def test_metadata_default_empty(self):
        entry = AuditEntry(input_text="x", action_taken="allow")
        assert entry.metadata == {}
