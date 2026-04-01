"""
Audit Logging System
Persistent audit logging with SQLite backend for compliance (CMMC, HIPAA, SOC 2).
Provides search, filtering, and export capabilities.
"""

import json
import sqlite3
import csv
from datetime import datetime, timezone
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading

@dataclass
class AuditEntry:
    """Single audit log entry representing a guardrail evaluation."""
    id: Optional[int] = None
    timestamp: str = ""
    input_text: str = ""
    action_taken: str = ""
    matched_rules: List[str] = None
    severity: str = ""
    risk_score: float = 0.0
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.matched_rules is None:
            self.matched_rules = []
        if self.metadata is None:
            self.metadata = {}

class AuditLogger:
    """Persistent audit logging with SQLite backend and extended features."""

    def __init__(self, db_path: str = "audit_log.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        # For in-memory databases we must reuse a single connection because
        # each sqlite3.connect(":memory:") call creates a brand-new database.
        # check_same_thread=False is safe here because the in-memory path is
        # used for testing only; production uses file-based SQLite with
        # threading.Lock() for synchronisation.
        self._conn: Optional[sqlite3.Connection] = (
            sqlite3.connect(":memory:", check_same_thread=False)
            if db_path == ":memory:"
            else None
        )
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active database connection."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize SQLite database with required schema."""
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT,
                action_taken TEXT NOT NULL,
                matched_rules TEXT,
                severity TEXT,
                risk_score REAL,
                user_id TEXT,
                session_id TEXT,
                metadata TEXT
                )
            """)
        conn.commit()

    def log(
        self,
        entry: Optional["AuditEntry"] = None,
        *,
        input_text: str = "",
        action_taken: str = "",
        matched_rules: Optional[List[str]] = None,
        severity: str = "",
        risk_score: float = 0.0,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> int:
        """Log a new audit entry to the database.

        Can be called with an ``AuditEntry`` object **or** with keyword
        arguments for convenience::

            logger.log(entry)                               # object form
            logger.log(input_text="hi", action_taken="allow")  # keyword form
        """
        if entry is None:
            entry = AuditEntry(
                input_text=input_text,
                action_taken=action_taken,
                matched_rules=matched_rules or [],
                severity=severity,
                risk_score=risk_score,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata or {},
            )
        with self.lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO audit_log (
                    timestamp, input_text, action_taken, matched_rules,
                    severity, risk_score, user_id, session_id, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp,
                entry.input_text,
                entry.action_taken,
                json.dumps(entry.matched_rules),
                entry.severity,
                entry.risk_score,
                entry.user_id,
                entry.session_id,
                json.dumps(entry.metadata),
            ))
            conn.commit()
            return cursor.lastrowid

    def get_logs(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Retrieve paginated audit logs as dictionaries."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [dict(self._row_to_entry(row).__dict__) for row in cursor.fetchall()]

    def get_log_entries(self, limit: int = 100, offset: int = 0) -> List[AuditEntry]:
        """Retrieve paginated audit logs as ``AuditEntry`` objects."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        return [self._row_to_entry(row) for row in cursor.fetchall()]

    def search(self, query: str) -> List[Dict]:
        """Search logs by input text or action."""
        conn = self._get_conn()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log WHERE input_text LIKE ? OR action_taken LIKE ? ORDER BY timestamp DESC",
            (f"%{query}%", f"%{query}%"),
        )
        return [dict(self._row_to_entry(row).__dict__) for row in cursor.fetchall()]

    def get_metrics(self) -> Dict:
        """Return summary metrics for dashboards and API endpoints."""
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM audit_log")
        total = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM audit_log WHERE action_taken = 'block'")
        blocked = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM audit_log WHERE action_taken = 'warn'")
        warned = cursor.fetchone()[0]
        return {
            "total_evaluations": total,
            "blocked": blocked,
            "warned": warned,
            "allowed": total - blocked - warned,
            "block_rate": round(blocked / total, 4) if total else 0.0,
        }

    def export_csv(self, output_path: Union[str, Path]) -> None:
        """Export all logs to a CSV file."""
        from dataclasses import asdict
        logs = self.get_log_entries(limit=10000)
        with open(output_path, "w", newline="") as f:
            if not logs:
                return
            first = asdict(logs[0])
            writer = csv.DictWriter(f, fieldnames=list(first.keys()))
            writer.writeheader()
            for log in logs:
                data = asdict(log)
                data["matched_rules"] = json.dumps(data["matched_rules"])
                data["metadata"] = json.dumps(data["metadata"])
                writer.writerow(data)

    def _row_to_entry(self, row: sqlite3.Row) -> AuditEntry:
        """Convert a database row to an ``AuditEntry`` object."""
        return AuditEntry(
            id=row["id"],
            timestamp=row["timestamp"],
            input_text=row["input_text"],
            action_taken=row["action_taken"],
            matched_rules=json.loads(row["matched_rules"]),
            severity=row["severity"],
            risk_score=row["risk_score"],
            user_id=row["user_id"],
            session_id=row["session_id"],
            metadata=json.loads(row["metadata"]),
        )
