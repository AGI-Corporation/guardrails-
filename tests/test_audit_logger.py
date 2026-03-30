"""
Tests for audit_logger.py
Covers: AuditLogger (log, query, get_statistics, export_csv), create_audit_entry.
"""

import os
import tempfile
import pytest

from audit_logger import AuditLogger, AuditEntry, create_audit_entry


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def logger(tmp_path):
    """File-backed SQLite audit logger scoped to a temporary directory."""
    return AuditLogger(db_path=str(tmp_path / "test_audit.db"))


@pytest.fixture
def populated_logger(tmp_path):
    lg = AuditLogger(db_path=str(tmp_path / "test_audit_pop.db"))
    # Add a mix of allow / block entries
    for i in range(5):
        lg.log(create_audit_entry(
            input_text=f"safe text {i}",
            action_taken="allow",
            matched_rules=[],
            severity="none",
            user_id=f"user_{i % 2}",
        ))
    for i in range(3):
        lg.log(create_audit_entry(
            input_text=f"bad text {i}",
            action_taken="block",
            matched_rules=["pii_ssn"],
            severity="critical",
            user_id="user_0",
        ))
    return lg


# ── log & query ───────────────────────────────────────────────────────────

class TestAuditLoggerLog:

    def test_log_returns_integer_id(self, logger):
        entry = create_audit_entry("hello", "allow", [], "none")
        entry_id = logger.log(entry)
        assert isinstance(entry_id, int)
        assert entry_id > 0

    def test_logged_entry_is_queryable(self, logger):
        logger.log(create_audit_entry("test input", "block", ["rule_1"], "high"))
        entries = logger.query(limit=10)
        assert len(entries) == 1
        assert entries[0].input_text == "test input"
        assert entries[0].action_taken == "block"
        assert "rule_1" in entries[0].matched_rules
        assert entries[0].severity == "high"

    def test_multiple_entries_stored(self, populated_logger):
        entries = populated_logger.query(limit=100)
        assert len(entries) == 8

    def test_query_filter_by_action(self, populated_logger):
        blocked = populated_logger.query(action="block", limit=100)
        assert all(e.action_taken == "block" for e in blocked)
        assert len(blocked) == 3

    def test_query_filter_by_severity(self, populated_logger):
        critical = populated_logger.query(severity="critical", limit=100)
        assert all(e.severity == "critical" for e in critical)

    def test_query_filter_by_user_id(self, populated_logger):
        user0 = populated_logger.query(user_id="user_0", limit=100)
        assert all(e.user_id == "user_0" for e in user0)

    def test_query_limit(self, populated_logger):
        entries = populated_logger.query(limit=2)
        assert len(entries) <= 2

    def test_query_ordered_by_timestamp_desc(self, populated_logger):
        entries = populated_logger.query(limit=100)
        timestamps = [e.timestamp for e in entries]
        assert timestamps == sorted(timestamps, reverse=True)


# ── get_statistics ────────────────────────────────────────────────────────

class TestAuditStatistics:

    def test_empty_stats(self, logger):
        stats = logger.get_statistics()
        assert stats["total"] == 0
        assert stats["blocked"] == 0
        assert stats["allowed"] == 0
        assert stats["block_rate"] == 0

    def test_stats_counts(self, populated_logger):
        stats = populated_logger.get_statistics()
        assert stats["total"] == 8
        assert stats["blocked"] == 3
        assert stats["allowed"] == 5

    def test_block_rate(self, populated_logger):
        stats = populated_logger.get_statistics()
        expected = round(3 / 8 * 100, 2)
        assert stats["block_rate"] == expected

    def test_stats_by_severity(self, populated_logger):
        stats = populated_logger.get_statistics()
        assert "critical" in stats["by_severity"]
        assert stats["by_severity"]["critical"] == 3


# ── create_audit_entry helper ─────────────────────────────────────────────

class TestCreateAuditEntry:

    def test_creates_entry_with_id_none(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert entry.id is None

    def test_timestamp_is_populated(self):
        entry = create_audit_entry("text", "allow", [], "none")
        assert entry.timestamp

    def test_optional_fields_default(self):
        entry = create_audit_entry("text", "block", ["r1"], "high")
        assert entry.user_id is None
        assert entry.session_id is None
        assert entry.metadata == {}

    def test_optional_fields_set(self):
        entry = create_audit_entry(
            "text", "block", ["r1"], "high",
            user_id="u1", session_id="s1", metadata={"k": "v"}
        )
        assert entry.user_id == "u1"
        assert entry.session_id == "s1"
        assert entry.metadata == {"k": "v"}


# ── export_csv ────────────────────────────────────────────────────────────

class TestExportCsv:

    def test_export_creates_file(self, populated_logger):
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = f.name
        try:
            populated_logger.export_csv(path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_export_has_header_and_rows(self, populated_logger):
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        try:
            populated_logger.export_csv(path)
            with open(path) as f:
                lines = f.readlines()
            assert len(lines) >= 2  # header + at least 1 row
            assert "timestamp" in lines[0].lower() or "action" in lines[0].lower()
        finally:
            os.unlink(path)
