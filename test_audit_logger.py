"""
Tests for audit_logger.py
Covers AuditEntry, AuditLogger CRUD operations, search, and CSV export.
"""

import csv
import os
import tempfile
from pathlib import Path

import pytest

from audit_logger import AuditEntry, AuditLogger


# ── Helpers ─────────────────────────────────────────────────────────────────

def make_temp_db():
    """Return a temp path for an SQLite db that is cleaned up after use."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    return tmp.name


def make_entry(**kwargs):
    defaults = dict(
        input_text="Hello world",
        action_taken="allow",
        matched_rules=[],
        severity="low",
        risk_score=0.1,
        user_id="user1",
        session_id="sess1",
        metadata={"source": "test"},
    )
    defaults.update(kwargs)
    return AuditEntry(**defaults)


# ── AuditEntry ───────────────────────────────────────────────────────────────

class TestAuditEntry:
    def test_defaults_set(self):
        entry = AuditEntry()
        assert entry.timestamp != ""
        assert entry.matched_rules == []
        assert entry.metadata == {}
        assert entry.id is None

    def test_custom_values(self):
        entry = AuditEntry(
            input_text="test",
            action_taken="block",
            matched_rules=["r1"],
            severity="high",
            risk_score=0.9,
        )
        assert entry.input_text == "test"
        assert entry.action_taken == "block"
        assert entry.matched_rules == ["r1"]
        assert entry.severity == "high"
        assert entry.risk_score == 0.9


# ── AuditLogger ──────────────────────────────────────────────────────────────

class TestAuditLoggerInit:
    def test_creates_database_file(self):
        db_path = make_temp_db()
        try:
            AuditLogger(db_path)
            assert os.path.exists(db_path)
        finally:
            os.unlink(db_path)

    def test_idempotent_init(self):
        """Calling __init__ twice should not raise."""
        db_path = make_temp_db()
        try:
            AuditLogger(db_path)
            AuditLogger(db_path)  # Second init — table already exists
        finally:
            os.unlink(db_path)


class TestAuditLoggerLog:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.logger = AuditLogger(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_log_returns_integer_id(self):
        entry = make_entry()
        row_id = self.logger.log(entry)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_log_increments_id(self):
        id1 = self.logger.log(make_entry(input_text="first"))
        id2 = self.logger.log(make_entry(input_text="second"))
        assert id2 > id1

    def test_logged_entry_retrievable(self):
        entry = make_entry(input_text="test input", action_taken="block",
                           matched_rules=["ssn"], severity="critical",
                           risk_score=0.99, user_id="alice", session_id="s1",
                           metadata={"key": "val"})
        self.logger.log(entry)
        logs = self.logger.get_logs()
        assert len(logs) == 1
        log = logs[0]
        assert log.input_text == "test input"
        assert log.action_taken == "block"
        assert log.matched_rules == ["ssn"]
        assert log.severity == "critical"
        assert log.risk_score == pytest.approx(0.99)
        assert log.user_id == "alice"
        assert log.session_id == "s1"
        assert log.metadata == {"key": "val"}

    def test_log_with_none_optional_fields(self):
        entry = AuditEntry(input_text="x", action_taken="allow")
        row_id = self.logger.log(entry)
        logs = self.logger.get_logs()
        assert any(l.id == row_id for l in logs)


class TestAuditLoggerGetLogs:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.logger = AuditLogger(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_empty_returns_empty_list(self):
        assert self.logger.get_logs() == []

    def test_returns_correct_count(self):
        for i in range(5):
            self.logger.log(make_entry(input_text=f"entry {i}"))
        logs = self.logger.get_logs()
        assert len(logs) == 5

    def test_limit_respected(self):
        for i in range(10):
            self.logger.log(make_entry(input_text=f"entry {i}"))
        logs = self.logger.get_logs(limit=3)
        assert len(logs) == 3

    def test_offset_respected(self):
        for i in range(5):
            self.logger.log(make_entry(input_text=f"entry {i}"))
        all_logs = self.logger.get_logs()
        offset_logs = self.logger.get_logs(offset=2)
        assert len(offset_logs) == len(all_logs) - 2

    def test_returns_audit_entry_objects(self):
        self.logger.log(make_entry())
        logs = self.logger.get_logs()
        assert all(isinstance(l, AuditEntry) for l in logs)

    def test_ids_are_set(self):
        self.logger.log(make_entry())
        logs = self.logger.get_logs()
        assert all(l.id is not None for l in logs)


class TestAuditLoggerSearch:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.logger = AuditLogger(self.db_path)

    def teardown_method(self):
        os.unlink(self.db_path)

    def test_search_by_input_text(self):
        self.logger.log(make_entry(input_text="find me here"))
        self.logger.log(make_entry(input_text="irrelevant content"))
        results = self.logger.search("find me")
        assert len(results) == 1
        assert results[0].input_text == "find me here"

    def test_search_by_action_taken(self):
        self.logger.log(make_entry(action_taken="block"))
        self.logger.log(make_entry(action_taken="allow"))
        results = self.logger.search("block")
        assert len(results) == 1
        assert results[0].action_taken == "block"

    def test_search_no_match_returns_empty(self):
        self.logger.log(make_entry(input_text="some text"))
        results = self.logger.search("zzz_no_match_zzz")
        assert results == []

    def test_search_empty_query_returns_all(self):
        for i in range(3):
            self.logger.log(make_entry(input_text=f"entry {i}"))
        results = self.logger.search("")
        assert len(results) == 3


class TestAuditLoggerExportCSV:
    def setup_method(self):
        self.db_path = make_temp_db()
        self.logger = AuditLogger(self.db_path)
        self.csv_path = make_temp_db().replace(".db", ".csv")

    def teardown_method(self):
        os.unlink(self.db_path)
        if os.path.exists(self.csv_path):
            os.unlink(self.csv_path)

    def test_export_creates_file(self):
        self.logger.log(make_entry())
        self.logger.export_csv(self.csv_path)
        assert os.path.exists(self.csv_path)

    def test_export_empty_db_no_header(self):
        self.logger.export_csv(self.csv_path)
        # Empty db — export_csv returns early without writing anything
        if os.path.exists(self.csv_path):
            with open(self.csv_path) as f:
                content = f.read()
            assert content == ""

    def test_export_contains_correct_rows(self):
        self.logger.log(make_entry(input_text="row1", action_taken="allow"))
        self.logger.log(make_entry(input_text="row2", action_taken="block"))
        self.logger.export_csv(self.csv_path)
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        texts = {r["input_text"] for r in rows}
        assert "row1" in texts
        assert "row2" in texts

    def test_export_has_header_row(self):
        self.logger.log(make_entry())
        self.logger.export_csv(self.csv_path)
        with open(self.csv_path) as f:
            header = f.readline()
        assert "input_text" in header
        assert "action_taken" in header

    def test_export_path_as_pathlib(self):
        self.logger.log(make_entry())
        self.logger.export_csv(Path(self.csv_path))
        assert os.path.exists(self.csv_path)
