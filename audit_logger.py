"""
Audit Logging System
Persistent audit logging with SQLite backend for compliance and traceability.
"""

import json
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import threading


@dataclass
class AuditEntry:
    """Single audit log entry"""
    id: Optional[int]
    timestamp: str
    input_text: str
    action_taken: str
    matched_rules: List[str]
    severity: str
    user_id: Optional[str]
    session_id: Optional[str]
    metadata: Dict


class AuditLogger:
    """Persistent audit logging with SQLite backend"""

    def __init__(self, db_path: str = "audit_log.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    input_text TEXT,
                    action_taken TEXT NOT NULL,
                    matched_rules TEXT,
                    severity TEXT,
                    user_id TEXT,
                    session_id TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()

    def log(self, entry: AuditEntry) -> int:
        """Log an audit entry and return its ID"""
        with self.lock:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO audit_log
                    (timestamp, input_text, action_taken, matched_rules, severity, user_id, session_id, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.timestamp or datetime.now().isoformat(),
                    entry.input_text,
                    entry.action_taken,
                    json.dumps(entry.matched_rules),
                    entry.severity,
                    entry.user_id,
                    entry.session_id,
                    json.dumps(entry.metadata),
                ))
                conn.commit()
                return cursor.lastrowid

    def query(
        self,
        action: Optional[str] = None,
        severity: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[AuditEntry]:
        """Query audit log entries"""
        conditions = []
        params = []

        if action:
            conditions.append("action_taken = ?")
            params.append(action)
        if severity:
            conditions.append("severity = ?")
            params.append(severity)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM audit_log {where_clause} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()

        return [
            AuditEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                input_text=row["input_text"],
                action_taken=row["action_taken"],
                matched_rules=json.loads(row["matched_rules"] or "[]"),
                severity=row["severity"],
                user_id=row["user_id"],
                session_id=row["session_id"],
                metadata=json.loads(row["metadata"] or "{}"),
            )
            for row in rows
        ]

    def get_statistics(self) -> Dict:
        """Get aggregate statistics from audit log"""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
            blocked = conn.execute(
                "SELECT COUNT(*) FROM audit_log WHERE action_taken = 'block'"
            ).fetchone()[0]
            by_severity = dict(
                conn.execute(
                    "SELECT severity, COUNT(*) FROM audit_log GROUP BY severity"
                ).fetchall()
            )

        return {
            "total": total,
            "blocked": blocked,
            "allowed": total - blocked,
            "block_rate": round(blocked / total * 100, 2) if total > 0 else 0,
            "by_severity": by_severity,
        }

    def export_csv(self, filepath: str):
        """Export audit log to CSV"""
        import csv
        entries = self.query(limit=100000)
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "timestamp", "action_taken", "severity", "matched_rules", "user_id"])
            for e in entries:
                writer.writerow([e.id, e.timestamp, e.action_taken, e.severity, e.matched_rules, e.user_id])


def create_audit_entry(
    input_text: str,
    action_taken: str,
    matched_rules: List[str],
    severity: str,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> AuditEntry:
    """Helper to create an AuditEntry"""
    return AuditEntry(
        id=None,
        timestamp=datetime.now().isoformat(),
        input_text=input_text,
        action_taken=action_taken,
        matched_rules=matched_rules,
        severity=severity,
        user_id=user_id,
        session_id=session_id,
        metadata=metadata or {},
    )
