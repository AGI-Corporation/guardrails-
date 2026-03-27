"""
Tests for audit_logger.py
Covers: AuditLogger init, log, query, get_statistics, export_csv,
        create_audit_entry helper.
"""

import csv
import os
import tempfile

import pytest

from audit_logger import AuditEntry, AuditLogger, create_audit_entry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a fresh temporary SQLite database."""
    return str(tmp_path / "test_audit.db")


@pytest.fixture
def logger(tmp_db):
    return AuditLogger(db_path=tmp_db)


def _make_entry(
    input_text="hello world",
    action_taken="allow",
    matched_rules=None,
    severity="none",
    user_id=None,
    session_id=None,
    metadata=None,
):
    return AuditEntry(
        id=None,
        timestamp="2024-01-01T00:00:00",
        input_text=input_text,
        action_taken=action_taken,
        matched_rules=matched_rules or [],
        severity=severity,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# Database initialisation
# ---------------------------------------------------------------------------

class TestAuditLoggerInit:
    def test_creates_db_file(self, tmp_db):
        AuditLogger(db_path=tmp_db)
        assert os.path.exists(tmp_db)

    def test_reinit_is_idempotent(self, tmp_db):
        AuditLogger(db_path=tmp_db)
        AuditLogger(db_path=tmp_db)  # second init should not raise


# ---------------------------------------------------------------------------
# log()
# ---------------------------------------------------------------------------

class TestAuditLoggerLog:
    def test_log_returns_integer_id(self, logger):
        entry = _make_entry()
        row_id = logger.log(entry)
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_log_increments_id(self, logger):
        id1 = logger.log(_make_entry())
        id2 = logger.log(_make_entry())
        assert id2 > id1

    def test_log_preserves_fields(self, logger):
        entry = _make_entry(
            input_text="test input",
            action_taken="block",
            matched_rules=["rule_a"],
            severity="high",
            user_id="user123",
            session_id="sess456",
            metadata={"extra": "data"},
        )
        logger.log(entry)
        results = logger.query()
        assert len(results) == 1
        r = results[0]
        assert r.input_text == "test input"
        assert r.action_taken == "block"
        assert r.matched_rules == ["rule_a"]
        assert r.severity == "high"
        assert r.user_id == "user123"
        assert r.session_id == "sess456"
        assert r.metadata == {"extra": "data"}

    def test_log_multiple_entries(self, logger):
        for i in range(5):
            logger.log(_make_entry(input_text=f"text{i}"))
        results = logger.query()
        assert len(results) == 5

    def test_log_with_null_timestamp_uses_now(self, logger):
        entry = AuditEntry(
            id=None,
            timestamp=None,
            input_text="hi",
            action_taken="allow",
            matched_rules=[],
            severity="none",
            user_id=None,
            session_id=None,
            metadata={},
        )
        row_id = logger.log(entry)
        results = logger.query()
        assert results[0].timestamp  # auto-filled


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------

class TestAuditLoggerQuery:
    def setup_method(self):
        pass

    def test_query_all(self, logger):
        logger.log(_make_entry(action_taken="allow"))
        logger.log(_make_entry(action_taken="block"))
        results = logger.query()
        assert len(results) == 2

    def test_query_filter_action(self, logger):
        logger.log(_make_entry(action_taken="allow"))
        logger.log(_make_entry(action_taken="block"))
        results = logger.query(action="block")
        assert len(results) == 1
        assert results[0].action_taken == "block"

    def test_query_filter_severity(self, logger):
        logger.log(_make_entry(severity="critical"))
        logger.log(_make_entry(severity="low"))
        results = logger.query(severity="critical")
        assert len(results) == 1
        assert results[0].severity == "critical"

    def test_query_filter_user_id(self, logger):
        logger.log(_make_entry(user_id="alice"))
        logger.log(_make_entry(user_id="bob"))
        results = logger.query(user_id="alice")
        assert len(results) == 1
        assert results[0].user_id == "alice"

    def test_query_limit(self, logger):
        for _ in range(10):
            logger.log(_make_entry())
        results = logger.query(limit=3)
        assert len(results) == 3

    def test_query_multiple_filters(self, logger):
        logger.log(_make_entry(action_taken="block", severity="high", user_id="alice"))
        logger.log(_make_entry(action_taken="allow", severity="high", user_id="alice"))
        logger.log(_make_entry(action_taken="block", severity="low", user_id="bob"))
        results = logger.query(action="block", severity="high", user_id="alice")
        assert len(results) == 1

    def test_query_empty_db(self, logger):
        results = logger.query()
        assert results == []

    def test_query_returns_audit_entries(self, logger):
        logger.log(_make_entry())
        results = logger.query()
        assert isinstance(results[0], AuditEntry)


# ---------------------------------------------------------------------------
# get_statistics()
# ---------------------------------------------------------------------------

class TestAuditLoggerStatistics:
    def test_empty_statistics(self, logger):
        stats = logger.get_statistics()
        assert stats["total"] == 0
        assert stats["blocked"] == 0
        assert stats["allowed"] == 0
        assert stats["block_rate"] == 0

    def test_statistics_counts(self, logger):
        logger.log(_make_entry(action_taken="block"))
        logger.log(_make_entry(action_taken="allow"))
        logger.log(_make_entry(action_taken="allow"))
        stats = logger.get_statistics()
        assert stats["total"] == 3
        assert stats["blocked"] == 1
        assert stats["allowed"] == 2

    def test_statistics_block_rate(self, logger):
        logger.log(_make_entry(action_taken="block"))
        logger.log(_make_entry(action_taken="allow"))
        stats = logger.get_statistics()
        assert stats["block_rate"] == 50.0

    def test_statistics_by_severity(self, logger):
        logger.log(_make_entry(severity="critical"))
        logger.log(_make_entry(severity="critical"))
        logger.log(_make_entry(severity="low"))
        stats = logger.get_statistics()
        assert stats["by_severity"]["critical"] == 2
        assert stats["by_severity"]["low"] == 1


# ---------------------------------------------------------------------------
# export_csv()
# ---------------------------------------------------------------------------

class TestAuditLoggerExportCSV:
    def test_csv_creates_file(self, logger, tmp_path):
        logger.log(_make_entry())
        path = str(tmp_path / "export.csv")
        logger.export_csv(path)
        assert os.path.exists(path)

    def test_csv_has_header(self, logger, tmp_path):
        logger.log(_make_entry())
        path = str(tmp_path / "export.csv")
        logger.export_csv(path)
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
        assert "id" in header
        assert "timestamp" in header
        assert "action_taken" in header

    def test_csv_row_count(self, logger, tmp_path):
        for _ in range(3):
            logger.log(_make_entry())
        path = str(tmp_path / "export.csv")
        logger.export_csv(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        # header + 3 data rows
        assert len(rows) == 4

    def test_csv_empty_db(self, logger, tmp_path):
        path = str(tmp_path / "export.csv")
        logger.export_csv(path)
        with open(path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1  # header only


# ---------------------------------------------------------------------------
# create_audit_entry helper
# ---------------------------------------------------------------------------

class TestCreateAuditEntry:
    def test_returns_audit_entry(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert isinstance(entry, AuditEntry)

    def test_id_is_none(self):
        entry = create_audit_entry("text", "block", ["rule1"], "high")
        assert entry.id is None

    def test_timestamp_set(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert entry.timestamp

    def test_fields_stored(self):
        entry = create_audit_entry(
            "sample", "block", ["r1", "r2"], "critical",
            user_id="u1", session_id="s1", metadata={"k": "v"},
        )
        assert entry.input_text == "sample"
        assert entry.action_taken == "block"
        assert entry.matched_rules == ["r1", "r2"]
        assert entry.severity == "critical"
        assert entry.user_id == "u1"
        assert entry.session_id == "s1"
        assert entry.metadata == {"k": "v"}

    def test_metadata_defaults_to_empty_dict(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert entry.metadata == {}

    def test_optional_fields_default_to_none(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert entry.user_id is None
        assert entry.session_id is None
