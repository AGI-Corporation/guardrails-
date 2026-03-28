"""
Tests for audit_logger.py
"""

import csv
import os
import tempfile
import threading
import pytest

from audit_logger import AuditEntry, AuditLogger, create_audit_entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_db(tmp_path):
    """Return a fresh AuditLogger backed by a temp SQLite DB."""
    db_file = str(tmp_path / "test_audit.db")
    return AuditLogger(db_path=db_file)


def _sample_entry(**kwargs) -> AuditEntry:
    defaults = dict(
        input_text="Hello world",
        action_taken="allow",
        matched_rules=[],
        severity="none",
        user_id=None,
        session_id=None,
        metadata={},
    )
    defaults.update(kwargs)
    return create_audit_entry(**defaults)


# ---------------------------------------------------------------------------
# AuditEntry dataclass
# ---------------------------------------------------------------------------

class TestAuditEntry:
    def test_create_audit_entry_helper(self):
        entry = create_audit_entry(
            input_text="test",
            action_taken="block",
            matched_rules=["r1"],
            severity="high",
            user_id="user123",
            session_id="sess456",
            metadata={"key": "value"},
        )
        assert entry.id is None
        assert entry.input_text == "test"
        assert entry.action_taken == "block"
        assert entry.matched_rules == ["r1"]
        assert entry.severity == "high"
        assert entry.user_id == "user123"
        assert entry.session_id == "sess456"
        assert entry.metadata["key"] == "value"
        assert entry.timestamp != ""

    def test_create_audit_entry_minimal(self):
        entry = create_audit_entry(
            input_text="hi",
            action_taken="allow",
            matched_rules=[],
            severity="none",
        )
        assert entry.user_id is None
        assert entry.session_id is None
        assert entry.metadata == {}


# ---------------------------------------------------------------------------
# AuditLogger — log and query
# ---------------------------------------------------------------------------

class TestAuditLoggerLog:
    def test_log_returns_positive_id(self, tmp_db):
        entry = _sample_entry()
        row_id = tmp_db.log(entry)
        assert row_id > 0

    def test_logged_entry_is_retrievable(self, tmp_db):
        entry = _sample_entry(input_text="specific text", action_taken="block", severity="high")
        tmp_db.log(entry)
        entries = tmp_db.query()
        assert len(entries) == 1
        assert entries[0].input_text == "specific text"
        assert entries[0].action_taken == "block"

    def test_multiple_entries(self, tmp_db):
        for i in range(5):
            tmp_db.log(_sample_entry(input_text=f"text {i}"))
        entries = tmp_db.query()
        assert len(entries) == 5

    def test_matched_rules_persisted(self, tmp_db):
        entry = _sample_entry(matched_rules=["rule_a", "rule_b"])
        tmp_db.log(entry)
        retrieved = tmp_db.query()[0]
        assert "rule_a" in retrieved.matched_rules
        assert "rule_b" in retrieved.matched_rules

    def test_metadata_persisted(self, tmp_db):
        entry = _sample_entry(metadata={"source": "api"})
        tmp_db.log(entry)
        retrieved = tmp_db.query()[0]
        assert retrieved.metadata["source"] == "api"

    def test_null_user_and_session(self, tmp_db):
        entry = _sample_entry(user_id=None, session_id=None)
        tmp_db.log(entry)
        retrieved = tmp_db.query()[0]
        assert retrieved.user_id is None
        assert retrieved.session_id is None


# ---------------------------------------------------------------------------
# AuditLogger — query with filters
# ---------------------------------------------------------------------------

class TestAuditLoggerQuery:
    def test_filter_by_action(self, tmp_db):
        tmp_db.log(_sample_entry(action_taken="allow"))
        tmp_db.log(_sample_entry(action_taken="block"))
        tmp_db.log(_sample_entry(action_taken="block"))
        blocked = tmp_db.query(action="block")
        assert len(blocked) == 2
        allowed = tmp_db.query(action="allow")
        assert len(allowed) == 1

    def test_filter_by_severity(self, tmp_db):
        tmp_db.log(_sample_entry(severity="high"))
        tmp_db.log(_sample_entry(severity="low"))
        high = tmp_db.query(severity="high")
        assert len(high) == 1

    def test_filter_by_user_id(self, tmp_db):
        tmp_db.log(_sample_entry(user_id="alice"))
        tmp_db.log(_sample_entry(user_id="bob"))
        alice_entries = tmp_db.query(user_id="alice")
        assert len(alice_entries) == 1

    def test_limit_respected(self, tmp_db):
        for _ in range(10):
            tmp_db.log(_sample_entry())
        entries = tmp_db.query(limit=3)
        assert len(entries) == 3

    def test_combined_filters(self, tmp_db):
        tmp_db.log(_sample_entry(action_taken="block", severity="critical"))
        tmp_db.log(_sample_entry(action_taken="block", severity="low"))
        tmp_db.log(_sample_entry(action_taken="allow", severity="critical"))
        results = tmp_db.query(action="block", severity="critical")
        assert len(results) == 1


# ---------------------------------------------------------------------------
# AuditLogger — get_statistics
# ---------------------------------------------------------------------------

class TestAuditLoggerStatistics:
    def test_empty_stats(self, tmp_db):
        stats = tmp_db.get_statistics()
        assert stats["total"] == 0
        assert stats["blocked"] == 0
        assert stats["allowed"] == 0
        assert stats["block_rate"] == 0

    def test_statistics_counts(self, tmp_db):
        tmp_db.log(_sample_entry(action_taken="allow"))
        tmp_db.log(_sample_entry(action_taken="block"))
        tmp_db.log(_sample_entry(action_taken="block"))
        stats = tmp_db.get_statistics()
        assert stats["total"] == 3
        assert stats["blocked"] == 2
        assert stats["allowed"] == 1

    def test_block_rate_calculation(self, tmp_db):
        for _ in range(3):
            tmp_db.log(_sample_entry(action_taken="block"))
        for _ in range(1):
            tmp_db.log(_sample_entry(action_taken="allow"))
        stats = tmp_db.get_statistics()
        assert stats["block_rate"] == 75.0

    def test_by_severity_grouping(self, tmp_db):
        tmp_db.log(_sample_entry(severity="high"))
        tmp_db.log(_sample_entry(severity="high"))
        tmp_db.log(_sample_entry(severity="low"))
        stats = tmp_db.get_statistics()
        assert stats["by_severity"]["high"] == 2
        assert stats["by_severity"]["low"] == 1


# ---------------------------------------------------------------------------
# AuditLogger — export_csv
# ---------------------------------------------------------------------------

class TestAuditLoggerExportCSV:
    def test_csv_created(self, tmp_db, tmp_path):
        tmp_db.log(_sample_entry(action_taken="allow"))
        csv_path = str(tmp_path / "export.csv")
        tmp_db.export_csv(csv_path)
        assert os.path.exists(csv_path)

    def test_csv_has_header_and_rows(self, tmp_db, tmp_path):
        tmp_db.log(_sample_entry(action_taken="block", severity="high"))
        csv_path = str(tmp_path / "export.csv")
        tmp_db.export_csv(csv_path)
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0][0] == "id"
        assert len(rows) == 2  # header + 1 data row


# ---------------------------------------------------------------------------
# AuditLogger — thread safety
# ---------------------------------------------------------------------------

class TestAuditLoggerThreadSafety:
    def test_concurrent_logs(self, tmp_db):
        errors = []

        def log_entry():
            try:
                tmp_db.log(_sample_entry())
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=log_entry) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        stats = tmp_db.get_statistics()
        assert stats["total"] == 20
