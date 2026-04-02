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
        # In-memory SQLite databases are not shared across connections, so we
        # keep a single persistent connection for :memory: usage (e.g. tests).
        self._conn: Optional[sqlite3.Connection] = None
        if db_path == ":memory:":
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        """Return the appropriate SQLite connection."""
        if self._conn is not None:
            return self._conn
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """Initialize SQLite database with required schema."""
        conn = self._connect()
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
        # Close only file-based connections opened for init
        if self._conn is None:
            conn.close()

    def log(self, entry: Optional[AuditEntry] = None, **kwargs) -> int:
        """Log a new audit entry to the database.

        Accepts either an ``AuditEntry`` object or keyword arguments that map
        directly to ``AuditEntry`` fields (for convenient call-site usage).
        """
        if entry is None:
            entry = AuditEntry(**kwargs)
        with self.lock:
            conn = self._connect()
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
                    json.dumps(entry.metadata)
                ))
            conn.commit()
            row_id = cursor.lastrowid
            if self._conn is None:
                conn.close()
            return row_id

    def get_logs(self, limit: int = 100, offset: int = 0) -> List[AuditEntry]:
        """Retrieve paginated audit logs."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset)
        )
        rows = [self._row_to_entry(row) for row in cursor.fetchall()]
        if self._conn is None:
            conn.close()
        return rows

    def search(self, query: str) -> List[AuditEntry]:
        """Search logs by input text or action."""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM audit_log WHERE input_text LIKE ? OR action_taken LIKE ? ORDER BY timestamp DESC",
            (f"%{query}%", f"%{query}%")
        )
        rows = [self._row_to_entry(row) for row in cursor.fetchall()]
        if self._conn is None:
            conn.close()
        return rows

    def export_csv(self, output_path: Union[str, Path]):
        """Export all logs to a CSV file."""
        logs = self.get_logs(limit=10000)
        with open(output_path, 'w', newline='') as f:
            if not logs:
                return
            writer = csv.DictWriter(f, fieldnames=asdict(logs[0]).keys())
            writer.writeheader()
            for log in logs:
                data = asdict(log)
                data['matched_rules'] = json.dumps(data['matched_rules'])
                data['metadata'] = json.dumps(data['metadata'])
                writer.writerow(data)

    def _row_to_entry(self, row: sqlite3.Row) -> AuditEntry:
        """Convert a database row to an AuditEntry object."""
        return AuditEntry(
            id=row['id'],
            timestamp=row['timestamp'],
            input_text=row['input_text'],
            action_taken=row['action_taken'],
            matched_rules=json.loads(row['matched_rules']),
            severity=row['severity'],
            risk_score=row['risk_score'],
            user_id=row['user_id'],
            session_id=row['session_id'],
            metadata=json.loads(row['metadata'])
        )
