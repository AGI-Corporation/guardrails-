"""
🔴 Penetration Test Agent
========================
Deep integration layer that orchestrates every component of the Guardrails
framework into an automated, end-to-end security penetration test cycle:

    AdversarialTester   → generates attack payloads across all categories
    GuardrailEngine     → defends and classifies each attack
    PluginManager       → provides ML-level heuristic checks (entropy, length, …)
    ContentTransformer  → redacts PII before logging
    AuditLogger         → records every pentest event for compliance
    PerformanceProfiler → measures defence throughput under attack load
    FeedbackLoop        → learns from missed attacks (false negatives)

Public surface
--------------
    PenTestSession      – captures configuration for one test run
    PenTestReport       – aggregates results; exports Markdown / JSON / CSV
    PenetrationTestAgent – orchestrates a full session, returns a report

Usage
-----
    from penetration_test_agent import PenetrationTestAgent, PenTestSession

    session = PenTestSession(name="nightly-scan", include_plugins=True)
    agent   = PenetrationTestAgent()
    report  = agent.run(session)
    print(report.to_markdown())
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from adversarial_tester import (
    AdversarialResult,
    AdversarialSuiteResults,
    AdversarialTester,
    MutatedTest,
)
from audit_logger import AuditLogger
from content_transformer import ContentTransformer
from feedback_loop import FeedbackLoop, FeedbackType, create_feedback_entry
from guardrail_framework import (
    Action,
    GuardrailEngine,
    create_default_guardrails,
)
from performance_profiler import PerformanceProfiler
from plugin_system import PluginManager


# ── Data-classes ──────────────────────────────────────────────────────────────

@dataclass
class PenTestSession:
    """Configuration for a single penetration-test run."""
    name: str = "pentest"
    description: str = ""
    include_plugins: bool = True
    audit_db_path: str = "pentest_audit.db"
    feedback_db_path: str = "pentest_feedback.db"
    #: Extra custom attack seeds (appended to built-in seeds).
    custom_seeds: List[str] = field(default_factory=list)
    #: Maximum number of payloads to run per category (0 = unlimited).
    max_per_category: int = 0


@dataclass
class AttackDetail:
    """Fine-grained result for a single attack payload."""
    category: str
    mutation: str
    original: str
    mutated: str
    blocked: bool
    matched_rules: List[str]
    plugin_action: str          # "allow" | "warn" | "block"
    plugin_details: List[Dict]
    duration_ms: float


@dataclass
class CategorySummary:
    """Rolled-up statistics for one attack category."""
    category: str
    total: int
    blocked: int
    bypassed: int
    block_rate: float           # 0.0 – 1.0
    avg_duration_ms: float
    bypassed_mutations: List[str]


@dataclass
class PenTestReport:
    """
    Full report produced by ``PenetrationTestAgent.run()``.

    Provides structured access to results and multiple export formats.
    """
    session_name: str
    started_at: str
    finished_at: str
    duration_s: float

    # Core results
    attacks: List[AttackDetail] = field(default_factory=list)
    category_summaries: List[CategorySummary] = field(default_factory=list)

    # Aggregate metrics
    total_attacks: int = 0
    total_blocked: int = 0
    total_bypassed: int = 0
    overall_block_rate: float = 0.0

    # Performance snapshot
    performance_stats: Dict = field(default_factory=dict)

    # Recommendations derived from results
    recommendations: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.total_attacks = len(self.attacks)
        self.total_blocked = sum(1 for a in self.attacks if a.blocked)
        self.total_bypassed = self.total_attacks - self.total_blocked
        self.overall_block_rate = (
            self.total_blocked / self.total_attacks if self.total_attacks else 0.0
        )

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        """Return the full report as a plain dictionary (JSON-serialisable)."""
        return {
            "session_name": self.session_name,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_s": round(self.duration_s, 3),
            "total_attacks": self.total_attacks,
            "total_blocked": self.total_blocked,
            "total_bypassed": self.total_bypassed,
            "overall_block_rate": round(self.overall_block_rate * 100, 1),
            "categories": [
                {
                    "category": s.category,
                    "total": s.total,
                    "blocked": s.blocked,
                    "bypassed": s.bypassed,
                    "block_rate_pct": round(s.block_rate * 100, 1),
                    "avg_duration_ms": round(s.avg_duration_ms, 3),
                    "bypassed_mutations": s.bypassed_mutations,
                }
                for s in self.category_summaries
            ],
            "performance": self.performance_stats,
            "recommendations": self.recommendations,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        """Render a human-readable Markdown report."""
        lines = [
            f"# 🔴 Penetration Test Report — {self.session_name}",
            "",
            f"**Started:** {self.started_at}  ",
            f"**Finished:** {self.finished_at}  ",
            f"**Duration:** {self.duration_s:.2f}s",
            "",
            "## Executive Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total attacks | {self.total_attacks} |",
            f"| Blocked | {self.total_blocked} |",
            f"| **Bypassed** | **{self.total_bypassed}** |",
            f"| Overall block rate | {self.overall_block_rate * 100:.1f}% |",
            "",
            "## Category Breakdown",
            "",
            "| Category | Total | Blocked | Bypassed | Block Rate | Avg ms |",
            "|----------|-------|---------|----------|------------|--------|",
        ]
        for s in sorted(self.category_summaries, key=lambda x: x.block_rate):
            lines.append(
                f"| {s.category} | {s.total} | {s.blocked} | {s.bypassed} "
                f"| {s.block_rate * 100:.1f}% | {s.avg_duration_ms:.2f} |"
            )

        if self.recommendations:
            lines += ["", "## Recommendations", ""]
            for rec in self.recommendations:
                lines.append(f"- {rec}")

        if self.performance_stats:
            lines += ["", "## Performance Under Attack", ""]
            lines += [
                "| Component | Calls | Avg ms | P95 ms | Max ms |",
                "|-----------|-------|--------|--------|--------|",
            ]
            for comp, s in self.performance_stats.items():
                lines.append(
                    f"| {comp} | {s.get('total_calls', 0)} "
                    f"| {s.get('avg_ms', 0):.3f} "
                    f"| {s.get('p95_ms', 0):.3f} "
                    f"| {s.get('max_ms', 0):.3f} |"
                )
        return "\n".join(lines)

    def export_csv(self, path: str) -> None:
        """Write the attack-level details to a CSV file."""
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "category", "mutation", "original", "mutated",
                    "blocked", "matched_rules", "plugin_action",
                    "duration_ms",
                ],
            )
            writer.writeheader()
            for a in self.attacks:
                writer.writerow(
                    {
                        "category": a.category,
                        "mutation": a.mutation,
                        "original": a.original[:120],
                        "mutated": a.mutated[:120],
                        "blocked": a.blocked,
                        "matched_rules": json.dumps(a.matched_rules),
                        "plugin_action": a.plugin_action,
                        "duration_ms": round(a.duration_ms, 3),
                    }
                )

    def save_json(self, path: str) -> None:
        """Write JSON report to disk."""
        Path(path).write_text(self.to_json())


# ── PenetrationTestAgent ──────────────────────────────────────────────────────

class PenetrationTestAgent:
    """
    Orchestrates a full, automated security penetration test against a
    Guardrails deployment.

    By default the agent builds a fresh ``GuardrailEngine`` with the standard
    default rules.  Pass a pre-configured engine to test a custom deployment.

    Parameters
    ----------
    engine:
        ``GuardrailEngine`` instance to attack.  Created from defaults when
        ``None``.
    audit_logger:
        ``AuditLogger`` to use for compliance logging.  A new in-memory (file)
        logger is created per session when ``None``.
    profiler:
        ``PerformanceProfiler`` instance.  Created fresh when ``None``.
    """

    def __init__(
        self,
        engine: Optional[GuardrailEngine] = None,
        audit_logger: Optional[AuditLogger] = None,
        profiler: Optional[PerformanceProfiler] = None,
    ) -> None:
        if engine is None:
            engine = GuardrailEngine()
            for rule in create_default_guardrails():
                engine.add_rule(rule)
        self._engine = engine
        self._audit = audit_logger
        self._profiler = profiler or PerformanceProfiler()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, session: Optional[PenTestSession] = None) -> PenTestReport:
        """
        Execute the full penetration-test suite described by *session* and
        return a ``PenTestReport``.
        """
        if session is None:
            session = PenTestSession()

        started_at = datetime.utcnow().isoformat()
        t0 = time.perf_counter()

        # ── Per-session helpers ──────────────────────────────────────────────
        audit = self._audit or AuditLogger(db_path=session.audit_db_path)
        feedback = FeedbackLoop(db_path=session.feedback_db_path)
        transformer = ContentTransformer()
        plugins = PluginManager() if session.include_plugins else None

        tester = AdversarialTester(self._engine)

        # ── Inject any custom seeds ──────────────────────────────────────────
        if session.custom_seeds:
            for seed in session.custom_seeds:
                tester.INJECTION_SEEDS.append(seed)

        # ── Run the adversarial suite ────────────────────────────────────────
        suite: AdversarialSuiteResults = tester.run_full_suite()

        # ── Enrich results with plugin + audit data ──────────────────────────
        attacks: List[AttackDetail] = []
        for adv_result in suite.results:
            attack = self._process_result(
                adv_result=adv_result,
                transformer=transformer,
                plugins=plugins,
                audit=audit,
                feedback=feedback,
                session=session,
            )
            attacks.append(attack)

        # ── Summaries ────────────────────────────────────────────────────────
        category_summaries = self._summarise_by_category(attacks)

        # ── Recommendations ──────────────────────────────────────────────────
        recommendations = self._generate_recommendations(category_summaries, feedback)

        finished_at = datetime.utcnow().isoformat()
        duration_s = time.perf_counter() - t0

        report = PenTestReport(
            session_name=session.name,
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            attacks=attacks,
            category_summaries=category_summaries,
            performance_stats=self._profiler.get_stats(),
            recommendations=recommendations,
        )

        # Log the session-level summary to audit
        audit.log(
            input_text=f"[PENTEST SESSION] {session.name}",
            action_taken="pentest_complete",
            matched_rules=[],
            severity="info",
            metadata={
                "total_attacks": report.total_attacks,
                "total_blocked": report.total_blocked,
                "overall_block_rate": round(report.overall_block_rate * 100, 1),
                "duration_s": round(duration_s, 3),
            },
        )

        return report

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _process_result(
        self,
        adv_result: AdversarialResult,
        transformer: ContentTransformer,
        plugins: Optional[PluginManager],
        audit: AuditLogger,
        feedback: FeedbackLoop,
        session: PenTestSession,
    ) -> AttackDetail:
        test: MutatedTest = adv_result.test

        # ── Plugin evaluation ────────────────────────────────────────────────
        plugin_action = "allow"
        plugin_details: List[Dict] = []
        if plugins is not None:
            with self._profiler.time("plugins", "evaluate"):
                plugin_results = plugins.evaluate(test.mutated)
            plugin_action = plugins._plugin_engine.get_final_action(plugin_results)
            plugin_details = [
                {
                    "plugin": r.plugin_name,
                    "action": r.action,
                    "score": round(r.score, 4),
                    "details": r.details,
                }
                for r in plugin_results
            ]

        # ── Redact PII before audit logging ──────────────────────────────────
        redacted_payload = transformer.apply_all_pii(test.mutated).transformed

        # ── Determine final blocked state (guardrail OR plugin blocks) ────────
        blocked = adv_result.blocked or plugin_action == "block"

        # ── Compliance audit entry ────────────────────────────────────────────
        with self._profiler.time("audit", "log"):
            audit.log(
                input_text=redacted_payload,
                action_taken="block" if blocked else "allow",
                matched_rules=adv_result.matched_rules,
                severity="critical" if blocked else "low",
                metadata={
                    "pentest": True,
                    "category": test.category,
                    "mutation": test.mutation.value,
                    "plugin_action": plugin_action,
                },
            )

        # ── Feed misses (bypasses) into the feedback loop ────────────────────
        if not blocked:
            entry = create_feedback_entry(
                text=test.mutated,
                original_action="allow",
                feedback_type=FeedbackType.FALSE_NEGATIVE,
                matched_rules=[],
            )
            feedback._store.add(entry)

        return AttackDetail(
            category=test.category,
            mutation=test.mutation.value,
            original=test.original,
            mutated=test.mutated,
            blocked=blocked,
            matched_rules=adv_result.matched_rules,
            plugin_action=plugin_action,
            plugin_details=plugin_details,
            duration_ms=adv_result.duration_ms,
        )

    @staticmethod
    def _summarise_by_category(attacks: List[AttackDetail]) -> List[CategorySummary]:
        by_cat: Dict[str, List[AttackDetail]] = {}
        for a in attacks:
            by_cat.setdefault(a.category, []).append(a)

        summaries: List[CategorySummary] = []
        for cat, items in by_cat.items():
            blocked_count = sum(1 for a in items if a.blocked)
            bypassed = [a.mutation for a in items if not a.blocked]
            avg_ms = (
                sum(a.duration_ms for a in items) / len(items) if items else 0.0
            )
            summaries.append(
                CategorySummary(
                    category=cat,
                    total=len(items),
                    blocked=blocked_count,
                    bypassed=len(items) - blocked_count,
                    block_rate=blocked_count / len(items) if items else 0.0,
                    avg_duration_ms=avg_ms,
                    bypassed_mutations=list(dict.fromkeys(bypassed)),  # deduplicated
                )
            )
        return sorted(summaries, key=lambda s: s.block_rate)

    @staticmethod
    def _generate_recommendations(
        summaries: List[CategorySummary],
        feedback: FeedbackLoop,
    ) -> List[str]:
        recs: List[str] = []

        # Per-category recommendations
        for s in summaries:
            if s.block_rate == 0.0:
                recs.append(
                    f"Category '{s.category}' had a 0% block rate — add detection "
                    f"rules or plugins for: {', '.join(s.bypassed_mutations[:3]) or 'all mutations'}."
                )
            elif s.block_rate < 0.5:
                recs.append(
                    f"Category '{s.category}' block rate is only {s.block_rate * 100:.0f}% — "
                    f"consider tightening rules for mutations: "
                    f"{', '.join(s.bypassed_mutations[:3])}."
                )

        # Feedback-driven recommendations
        try:
            tuning_report = feedback.get_tuning_report()
            if "Suggested Keyword Additions" in tuning_report:
                recs.append(
                    "Run feedback.get_tuning_report() for specific keyword additions "
                    "derived from bypass payloads."
                )
        except Exception:
            pass

        if not recs:
            recs.append(
                "Block rate ≥ 50% across all categories. "
                "Continue monitoring and run again after rule updates."
            )

        return recs


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardrails Penetration Test Agent")
    parser.add_argument("--name", default="cli-pentest", help="Session name")
    parser.add_argument("--no-plugins", action="store_true", help="Skip plugin checks")
    parser.add_argument("--json", metavar="PATH", help="Export JSON report to PATH")
    parser.add_argument("--csv", metavar="PATH", help="Export CSV report to PATH")
    args = parser.parse_args()

    session = PenTestSession(
        name=args.name,
        include_plugins=not args.no_plugins,
    )

    print(f"\n🔴 Starting penetration test session: {session.name}\n")
    agent = PenetrationTestAgent()
    report = agent.run(session)

    print(report.to_markdown())

    if args.json:
        report.save_json(args.json)
        print(f"\n✅  JSON report saved to {args.json}")

    if args.csv:
        report.export_csv(args.csv)
        print(f"✅  CSV report saved to {args.csv}")
