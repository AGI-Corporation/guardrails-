"""
Feedback Loop & Tuning System
Captures user feedback on guardrail decisions and suggests rule improvements.
"""

import json
import sqlite3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import Counter
import re


class FeedbackType(Enum):
    FALSE_POSITIVE = "false_positive"  # Blocked but should have been allowed
    FALSE_NEGATIVE = "false_negative"  # Allowed but should have been blocked
    CORRECT_BLOCK = "correct_block"
    CORRECT_ALLOW = "correct_allow"


@dataclass
class FeedbackEntry:
    id: Optional[int]
    timestamp: str
    text: str
    original_action: str
    feedback_type: FeedbackType
    user_id: Optional[str]
    matched_rules: List[str]
    expected_action: str
    comment: str = ""


class FeedbackStore:
    """Stores and retrieves feedback entries using SQLite"""

    def __init__(self, db_path: str = "feedback.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    text TEXT,
                    original_action TEXT,
                    feedback_type TEXT,
                    user_id TEXT,
                    matched_rules TEXT,
                    expected_action TEXT,
                    comment TEXT
                )
            """)
            conn.commit()

    def add(self, entry: FeedbackEntry) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO feedback
                (timestamp, text, original_action, feedback_type, user_id, matched_rules, expected_action, comment)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                entry.timestamp or datetime.now().isoformat(),
                entry.text,
                entry.original_action,
                entry.feedback_type.value,
                entry.user_id,
                json.dumps(entry.matched_rules),
                entry.expected_action,
                entry.comment,
            ))
            conn.commit()
            return cursor.lastrowid

    def get_all(self, feedback_type: Optional[FeedbackType] = None) -> List[FeedbackEntry]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if feedback_type:
                rows = conn.execute(
                    "SELECT * FROM feedback WHERE feedback_type = ? ORDER BY timestamp DESC",
                    (feedback_type.value,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM feedback ORDER BY timestamp DESC"
                ).fetchall()

        return [
            FeedbackEntry(
                id=row["id"],
                timestamp=row["timestamp"],
                text=row["text"],
                original_action=row["original_action"],
                feedback_type=FeedbackType(row["feedback_type"]),
                user_id=row["user_id"],
                matched_rules=json.loads(row["matched_rules"] or "[]"),
                expected_action=row["expected_action"],
                comment=row["comment"] or "",
            )
            for row in rows
        ]

    def get_stats(self) -> Dict:
        entries = self.get_all()
        type_counts = Counter(e.feedback_type.value for e in entries)
        return {
            "total": len(entries),
            "false_positives": type_counts.get("false_positive", 0),
            "false_negatives": type_counts.get("false_negative", 0),
            "correct_blocks": type_counts.get("correct_block", 0),
            "correct_allows": type_counts.get("correct_allow", 0),
        }


class TuningSuggester:
    """
    Analyzes feedback to suggest rule improvements.
    Identifies common patterns in false positives/negatives.
    """

    def __init__(self, store: FeedbackStore):
        self.store = store

    def suggest_keyword_additions(self, min_occurrences: int = 3) -> List[Dict]:
        """Suggest new keywords based on false negatives"""
        false_negatives = self.store.get_all(FeedbackType.FALSE_NEGATIVE)
        word_counts: Counter = Counter()

        for entry in false_negatives:
            words = re.findall(r"\b[a-z]{4,}\b", entry.text.lower())
            word_counts.update(words)

        suggestions = []
        for word, count in word_counts.most_common(20):
            if count >= min_occurrences:
                suggestions.append({
                    "keyword": word,
                    "occurrences": count,
                    "suggestion": f"Consider adding '{word}' as a keyword to existing rules",
                })

        return suggestions

    def suggest_rule_relaxation(self, min_occurrences: int = 5) -> List[Dict]:
        """Suggest rules to relax based on false positives"""
        false_positives = self.store.get_all(FeedbackType.FALSE_POSITIVE)
        rule_counts: Counter = Counter()

        for entry in false_positives:
            rule_counts.update(entry.matched_rules)

        suggestions = []
        for rule_id, count in rule_counts.most_common(10):
            if count >= min_occurrences:
                suggestions.append({
                    "rule_id": rule_id,
                    "false_positive_count": count,
                    "suggestion": f"Rule '{rule_id}' has {count} false positives. Consider relaxing or adding exclusions.",
                })

        return suggestions

    def generate_report(self) -> str:
        stats = self.store.get_stats()
        keyword_suggestions = self.suggest_keyword_additions()
        relaxation_suggestions = self.suggest_rule_relaxation()

        lines = [
            "# Guardrail Tuning Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Feedback Summary",
            f"- Total feedback entries: {stats['total']}",
            f"- False positives: {stats['false_positives']}",
            f"- False negatives: {stats['false_negatives']}",
            f"- Correct blocks: {stats['correct_blocks']}",
            f"- Correct allows: {stats['correct_allows']}",
            "",
        ]

        if keyword_suggestions:
            lines.append("## Suggested Keyword Additions")
            for s in keyword_suggestions[:5]:
                lines.append(f"- `{s['keyword']}` ({s['occurrences']} occurrences): {s['suggestion']}")
            lines.append("")

        if relaxation_suggestions:
            lines.append("## Suggested Rule Relaxations")
            for s in relaxation_suggestions[:5]:
                lines.append(f"- **{s['rule_id']}**: {s['suggestion']}")

        return "\n".join(lines)


def create_feedback_entry(
    text: str,
    original_action: str,
    feedback_type: FeedbackType,
    matched_rules: Optional[List[str]] = None,
    user_id: Optional[str] = None,
    comment: str = "",
) -> FeedbackEntry:
    expected = "allow" if feedback_type == FeedbackType.FALSE_POSITIVE else "block"
    return FeedbackEntry(
        id=None,
        timestamp=datetime.now().isoformat(),
        text=text,
        original_action=original_action,
        feedback_type=feedback_type,
        user_id=user_id,
        matched_rules=matched_rules or [],
        expected_action=expected,
        comment=comment,
    )
