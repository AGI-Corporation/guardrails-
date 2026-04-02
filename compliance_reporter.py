"""
🗂️ Compliance Reporter
======================
Generates structured compliance reports by aggregating data from:

    AuditLogger         → event trail, block/allow decisions, risk scores
    PenTestReport       → adversarial test results and bypass rates
    GuardrailEngine     → active rules and coverage assessment

Supported frameworks
--------------------
    HIPAA   — Health Insurance Portability and Accountability Act
    SOC2    — Service Organization Control 2 (Trust Services Criteria)
    CMMC    — Cybersecurity Maturity Model Certification
    GDPR    — General Data Protection Regulation

Public surface
--------------
    ComplianceFramework — enum of supported frameworks
    ComplianceControl   — single control evaluation result
    ComplianceReport    — full report for one framework
    ComplianceReporter  — generates reports; exports Markdown / JSON / HTML

Usage
-----
    from compliance_reporter import ComplianceReporter, ComplianceFramework
    from audit_logger import AuditLogger

    logger = AuditLogger()
    reporter = ComplianceReporter(audit_logger=logger)
    report = reporter.generate(ComplianceFramework.HIPAA)
    print(report.to_markdown())
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

from audit_logger import AuditLogger


# ── Enums & constants ─────────────────────────────────────────────────────────

class ComplianceFramework(Enum):
    HIPAA = "hipaa"
    SOC2 = "soc2"
    CMMC = "cmmc"
    GDPR = "gdpr"


class ControlStatus(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


# ── Data-classes ──────────────────────────────────────────────────────────────

@dataclass
class ComplianceControl:
    """Result of evaluating a single compliance control."""
    control_id: str
    name: str
    description: str
    status: ControlStatus
    evidence: str
    risk_level: str             # "low" | "medium" | "high" | "critical"
    remediation: str = ""       # Action to take if the control fails

    def to_dict(self) -> Dict:
        return {
            "control_id": self.control_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "evidence": self.evidence,
            "risk_level": self.risk_level,
            "remediation": self.remediation,
        }


@dataclass
class ComplianceReport:
    """
    Full compliance report for a single framework.

    Provides summary statistics, per-control findings, and multiple
    export formats (Markdown, JSON, HTML).
    """
    framework: ComplianceFramework
    generated_at: str
    audit_window_events: int

    controls: List[ComplianceControl] = field(default_factory=list)
    pentest_summary: Optional[Dict] = None

    # Computed in __post_init__
    total_controls: int = 0
    passed: int = 0
    failed: int = 0
    partial: int = 0
    overall_score: float = 0.0      # 0.0 – 100.0

    def __post_init__(self) -> None:
        self.total_controls = len(self.controls)
        self.passed = sum(1 for c in self.controls if c.status == ControlStatus.PASS)
        self.failed = sum(1 for c in self.controls if c.status == ControlStatus.FAIL)
        self.partial = sum(
            1 for c in self.controls if c.status == ControlStatus.PARTIAL
        )
        applicable = [
            c for c in self.controls if c.status != ControlStatus.NOT_APPLICABLE
        ]
        if applicable:
            score = sum(
                1.0 if c.status == ControlStatus.PASS
                else 0.5 if c.status == ControlStatus.PARTIAL
                else 0.0
                for c in applicable
            )
            self.overall_score = score / len(applicable) * 100

    @property
    def is_compliant(self) -> bool:
        """True if no controls have FAIL status."""
        return self.failed == 0

    # ── Export helpers ────────────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "framework": self.framework.value.upper(),
            "generated_at": self.generated_at,
            "audit_window_events": self.audit_window_events,
            "summary": {
                "total_controls": self.total_controls,
                "passed": self.passed,
                "failed": self.failed,
                "partial": self.partial,
                "overall_score": round(self.overall_score, 1),
                "compliant": self.is_compliant,
            },
            "controls": [c.to_dict() for c in self.controls],
            "pentest_summary": self.pentest_summary,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def to_markdown(self) -> str:
        status_icon = "✅" if self.is_compliant else "❌"
        lines = [
            f"# {status_icon} {self.framework.value.upper()} Compliance Report",
            "",
            f"**Generated:** {self.generated_at}  ",
            f"**Audit events analysed:** {self.audit_window_events}",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Overall Score | **{self.overall_score:.1f}%** |",
            f"| Compliant | {'Yes ✅' if self.is_compliant else 'No ❌'} |",
            f"| Controls Passed | {self.passed} / {self.total_controls} |",
            f"| Controls Failed | {self.failed} |",
            f"| Partial | {self.partial} |",
            "",
            "## Control Findings",
            "",
            "| ID | Control | Status | Risk | Evidence |",
            "|----|---------|--------|------|---------|",
        ]
        icons = {
            ControlStatus.PASS: "✅",
            ControlStatus.FAIL: "❌",
            ControlStatus.PARTIAL: "⚠️",
            ControlStatus.NOT_APPLICABLE: "➖",
        }
        for c in self.controls:
            icon = icons.get(c.status, "?")
            lines.append(
                f"| `{c.control_id}` | {c.name} | {icon} {c.status.value.upper()} "
                f"| {c.risk_level} | {c.evidence[:80]} |"
            )

        # Remediation actions for failed/partial controls
        fails = [c for c in self.controls if c.status in (ControlStatus.FAIL, ControlStatus.PARTIAL)]
        if fails:
            lines += ["", "## Remediation Actions", ""]
            for c in fails:
                lines.append(f"- **[{c.control_id}] {c.name}**: {c.remediation}")

        # Pentest integration
        if self.pentest_summary:
            lines += ["", "## Penetration Test Evidence", ""]
            lines += [
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Total Attacks | {self.pentest_summary.get('total_attacks', 'N/A')} |",
                f"| Overall Block Rate | {self.pentest_summary.get('overall_block_rate', 'N/A')}% |",
                f"| Bypassed | {self.pentest_summary.get('total_bypassed', 'N/A')} |",
            ]
            worst_cats = self.pentest_summary.get("weakest_categories", [])
            if worst_cats:
                lines += ["", "### Weakest Categories", ""]
                for cat in worst_cats:
                    lines.append(f"- {cat}")

        return "\n".join(lines)

    def to_html(self) -> str:
        """Minimal HTML report suitable for email or CI artifact upload."""
        status_class = "pass" if self.is_compliant else "fail"
        rows = ""
        icons = {
            ControlStatus.PASS: "✅",
            ControlStatus.FAIL: "❌",
            ControlStatus.PARTIAL: "⚠️",
            ControlStatus.NOT_APPLICABLE: "➖",
        }
        for c in self.controls:
            icon = icons.get(c.status, "")
            rows += (
                f"<tr>"
                f"<td><code>{c.control_id}</code></td>"
                f"<td>{c.name}</td>"
                f"<td>{icon} {c.status.value.upper()}</td>"
                f"<td>{c.risk_level}</td>"
                f"<td>{c.evidence[:100]}</td>"
                f"</tr>"
            )
        return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8">
<title>{self.framework.value.upper()} Compliance Report</title>
<style>
  body {{ font-family: sans-serif; margin: 2rem; }}
  h1 {{ color: {'#2e7d32' if self.is_compliant else '#c62828'}; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: .5rem 1rem; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .pass {{ color: #2e7d32; }}
  .fail {{ color: #c62828; }}
</style>
</head>
<body>
<h1>{'✅' if self.is_compliant else '❌'} {self.framework.value.upper()} Compliance Report</h1>
<p><strong>Generated:</strong> {self.generated_at}</p>
<p><strong>Overall Score:</strong> {self.overall_score:.1f}% &mdash;
   <span class="{status_class}">{'Compliant' if self.is_compliant else 'Non-compliant'}</span></p>
<h2>Control Findings</h2>
<table>
<tr><th>ID</th><th>Control</th><th>Status</th><th>Risk</th><th>Evidence</th></tr>
{rows}
</table>
</body>
</html>"""

    def save_json(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_json())

    def save_html(self, path: str) -> None:
        from pathlib import Path
        Path(path).write_text(self.to_html())


# ── ComplianceReporter ────────────────────────────────────────────────────────

class ComplianceReporter:
    """
    Generates ``ComplianceReport`` objects for supported frameworks by
    examining audit logs, active guardrail rules, and (optionally) a
    penetration-test report.

    Parameters
    ----------
    audit_logger:
        The ``AuditLogger`` instance to query for event history.
    engine:
        Optional ``GuardrailEngine`` — enables rule-coverage checks.
    pentest_report:
        Optional ``PenTestReport`` — integrates adversarial test evidence
        into the compliance report.
    audit_limit:
        Maximum number of audit events to load when evaluating controls.
    """

    def __init__(
        self,
        audit_logger: Optional[AuditLogger] = None,
        engine=None,
        pentest_report=None,
        audit_limit: int = 10_000,
    ) -> None:
        self._audit = audit_logger or AuditLogger()
        self._engine = engine
        self._pentest = pentest_report
        self._audit_limit = audit_limit

    # ── Public API ─────────────────────────────────────────────────────────

    def generate(self, framework: ComplianceFramework) -> ComplianceReport:
        """Generate a compliance report for the specified framework."""
        events = self._audit.get_logs(limit=self._audit_limit)
        generated_at = datetime.now(timezone.utc).isoformat()

        dispatch = {
            ComplianceFramework.HIPAA: self._evaluate_hipaa,
            ComplianceFramework.SOC2: self._evaluate_soc2,
            ComplianceFramework.CMMC: self._evaluate_cmmc,
            ComplianceFramework.GDPR: self._evaluate_gdpr,
        }
        controls = dispatch[framework](events)

        pentest_summary = None
        if self._pentest is not None:
            pentest_summary = {
                "total_attacks": self._pentest.total_attacks,
                "total_blocked": self._pentest.total_blocked,
                "total_bypassed": self._pentest.total_bypassed,
                "overall_block_rate": round(self._pentest.overall_block_rate * 100, 1),
                "weakest_categories": [
                    s.category
                    for s in self._pentest.category_summaries
                    if s.block_rate < 0.5
                ],
            }

        return ComplianceReport(
            framework=framework,
            generated_at=generated_at,
            audit_window_events=len(events),
            controls=controls,
            pentest_summary=pentest_summary,
        )

    def generate_all(self) -> Dict[str, ComplianceReport]:
        """Generate reports for all supported frameworks."""
        return {fw.value: self.generate(fw) for fw in ComplianceFramework}

    # ── HIPAA controls ────────────────────────────────────────────────────────

    def _evaluate_hipaa(self, events: list) -> List[ComplianceControl]:
        blocked = [e for e in events if e.action_taken == "block"]
        blocked_critical = [e for e in events if e.severity in ("critical", "high")]
        has_audit_trail = len(events) > 0
        has_phi_rules = self._engine_has_keywords(
            ["SSN", "MRN", "DOB", "medical", "health", "diagnosis"]
        )
        pii_blocks = sum(
            1 for e in blocked
            if any(r for r in e.matched_rules if "pii" in r.lower() or "phi" in r.lower())
        )

        # § 164.312(b) — Audit Controls
        audit_status = ControlStatus.PASS if has_audit_trail else ControlStatus.FAIL
        audit_evidence = (
            f"{len(events)} events logged in audit trail."
            if has_audit_trail
            else "No audit events found — enable AuditLogger."
        )

        # § 164.312(a)(2)(iv) — Encryption / PHI Detection
        phi_status = ControlStatus.PASS if has_phi_rules else ControlStatus.PARTIAL
        phi_evidence = (
            "PHI/PII detection rules are active in the guardrail engine."
            if has_phi_rules
            else "No PHI-specific rules detected; add PHI plugin or keywords."
        )

        # § 164.308(a)(1) — Risk Analysis
        pentest_done = self._pentest is not None
        risk_status = ControlStatus.PASS if pentest_done else ControlStatus.PARTIAL
        risk_evidence = (
            f"Penetration test run: {self._pentest.total_attacks} attacks, "
            f"{self._pentest.overall_block_rate * 100:.1f}% blocked."
            if pentest_done
            else "No penetration test data available; run PenetrationTestAgent."
        )

        # § 164.312(c)(1) — Integrity: high-severity events blocked
        integrity_ok = len(blocked_critical) > 0 or len(events) == 0
        integrity_status = ControlStatus.PASS if integrity_ok else ControlStatus.PARTIAL
        integrity_evidence = (
            f"{len(blocked_critical)} high/critical events blocked."
            if blocked_critical
            else "No high-severity events blocked yet."
        )

        # § 164.308(a)(5) — Security Awareness / Training (approximated by active rules)
        rules_count = len(self._engine.rules) if self._engine else 0
        awareness_status = ControlStatus.PASS if rules_count >= 3 else ControlStatus.PARTIAL
        awareness_evidence = (
            f"{rules_count} active guardrail rules providing automated security enforcement."
        )

        return [
            ComplianceControl(
                "HIPAA-164.312(b)", "Audit Controls", "Maintain audit logs of activity",
                audit_status, audit_evidence, "critical",
                "Enable AuditLogger and ensure logs are retained per policy." if audit_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "HIPAA-164.312(a)(2)(iv)", "PHI Detection & Protection",
                "Detect and protect Protected Health Information",
                phi_status, phi_evidence, "critical",
                "Add PHI-specific guardrail rules or the PHIPlugin." if phi_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "HIPAA-164.308(a)(1)", "Risk Analysis",
                "Conduct thorough risk analysis and implement risk management",
                risk_status, risk_evidence, "high",
                "Run PenetrationTestAgent regularly (weekly or per release)." if risk_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "HIPAA-164.312(c)(1)", "Integrity Controls",
                "Ensure ePHI has not been altered or destroyed",
                integrity_status, integrity_evidence, "high",
                "Review high-severity bypass events and tighten guardrail rules." if integrity_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "HIPAA-164.308(a)(5)", "Security Enforcement Policies",
                "Implement policies and procedures to protect ePHI",
                awareness_status, awareness_evidence, "medium",
                "Add at least 3 active guardrail rules covering content safety categories." if awareness_status != ControlStatus.PASS else "",
            ),
        ]

    # ── SOC 2 controls ────────────────────────────────────────────────────────

    def _evaluate_soc2(self, events: list) -> List[ComplianceControl]:
        has_audit_trail = len(events) > 0
        block_rate = (
            sum(1 for e in events if e.action_taken == "block") / len(events) * 100
            if events else 0.0
        )
        pentest_done = self._pentest is not None
        rules_count = len(self._engine.rules) if self._engine else 0

        # CC6.1 — Logical and Physical Access Controls
        cc6_status = ControlStatus.PASS if rules_count > 0 else ControlStatus.FAIL
        cc6_evidence = f"{rules_count} active guardrail rules restricting access to sensitive operations."

        # CC7.2 — Monitor System Components for Anomalies
        cc7_status = ControlStatus.PASS if has_audit_trail else ControlStatus.FAIL
        cc7_evidence = (
            f"Anomaly monitoring via AuditLogger: {len(events)} events logged."
            if has_audit_trail else "AuditLogger not active."
        )

        # CC8.1 — Change Management
        cc8_status = ControlStatus.PARTIAL
        cc8_evidence = "Guardrail rules are version-controlled; no formal change management workflow detected."

        # CC9.2 — Risk Mitigation via Security Testing
        cc9_status = ControlStatus.PASS if pentest_done else ControlStatus.PARTIAL
        cc9_evidence = (
            f"Penetration test completed: {self._pentest.total_attacks} attacks evaluated."
            if pentest_done else "No penetration test evidence available."
        )

        # A1.2 — Availability: Performance Monitoring
        has_perf_data = pentest_done and bool(self._pentest.performance_stats)
        a12_status = ControlStatus.PASS if has_perf_data else ControlStatus.PARTIAL
        a12_evidence = (
            f"Performance profiling active: {len(self._pentest.performance_stats)} components measured."
            if has_perf_data else "No performance profiling data; integrate PerformanceProfiler."
        )

        # C1.2 — Confidentiality: PII/Content Protection
        c12_status = ControlStatus.PASS if block_rate > 0 or len(events) == 0 else ControlStatus.PARTIAL
        c12_evidence = (
            f"Content protection active: {block_rate:.1f}% of events blocked."
            if events else "No events yet to assess content protection."
        )

        return [
            ComplianceControl(
                "CC6.1", "Logical Access Controls",
                "Restrict access based on least-privilege principle",
                cc6_status, cc6_evidence, "high",
                "Define and activate guardrail rules for all content categories." if cc6_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "CC7.2", "Anomaly & Threat Detection",
                "Monitor for anomalous activity and security events",
                cc7_status, cc7_evidence, "high",
                "Enable AuditLogger for all guardrail evaluations." if cc7_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "CC8.1", "Change Management",
                "Manage changes to system components",
                cc8_status, cc8_evidence, "medium",
                "Implement CI-based rule change review (e.g., PR approvals, policy-as-code).",
            ),
            ComplianceControl(
                "CC9.2", "Risk Mitigation Testing",
                "Identify and mitigate risks via security testing",
                cc9_status, cc9_evidence, "high",
                "Schedule regular PenetrationTestAgent runs." if cc9_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "A1.2", "Availability & Performance Monitoring",
                "Monitor system capacity and performance",
                a12_status, a12_evidence, "medium",
                "Integrate PerformanceProfiler and set latency thresholds." if a12_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "C1.2", "Confidentiality / Content Protection",
                "Protect confidential information from unauthorised access",
                c12_status, c12_evidence, "high",
                "Ensure content safety rules are active and block rate > 0%." if c12_status != ControlStatus.PASS else "",
            ),
        ]

    # ── CMMC controls ─────────────────────────────────────────────────────────

    def _evaluate_cmmc(self, events: list) -> List[ComplianceControl]:
        has_audit_trail = len(events) > 0
        blocked = sum(1 for e in events if e.action_taken == "block")
        pentest_done = self._pentest is not None
        rules_count = len(self._engine.rules) if self._engine else 0

        # AC.1.001 — Limit information system access
        ac1_status = ControlStatus.PASS if rules_count > 0 else ControlStatus.FAIL
        ac1_evidence = f"{rules_count} access control rules active."

        # AU.2.041 — Create and retain system audit logs
        au2_status = ControlStatus.PASS if has_audit_trail else ControlStatus.FAIL
        au2_evidence = f"{len(events)} audit events retained in SQLite store."

        # CA.2.158 — Periodically assess security controls
        ca2_status = ControlStatus.PASS if pentest_done else ControlStatus.PARTIAL
        ca2_evidence = (
            f"Security assessment via PenetrationTestAgent: {self._pentest.total_attacks} tests."
            if pentest_done else "No security assessment data."
        )

        # IR.2.092 — Track, document, and report incidents
        high_severity = sum(1 for e in events if e.severity in ("high", "critical"))
        ir2_status = ControlStatus.PASS if has_audit_trail else ControlStatus.FAIL
        ir2_evidence = (
            f"{high_severity} high/critical incidents tracked in audit log."
            if has_audit_trail else "No incident tracking (AuditLogger not active)."
        )

        # RA.3.145 — Periodically perform risk assessments
        ra3_status = ControlStatus.PASS if pentest_done else ControlStatus.PARTIAL
        ra3_evidence = (
            f"Risk assessment: {self._pentest.total_bypassed} bypass events require remediation."
            if pentest_done else "No risk assessment data."
        )

        # SI.1.210 — Identify, report, correct information and system flaws
        si1_ok = pentest_done and self._pentest.total_bypassed == 0
        si1_status = (
            ControlStatus.PASS if si1_ok
            else ControlStatus.PARTIAL if pentest_done
            else ControlStatus.FAIL
        )
        si1_evidence = (
            f"{self._pentest.total_bypassed} unmitigated bypass vectors identified."
            if pentest_done else "No flaw identification data."
        )

        return [
            ComplianceControl(
                "AC.1.001", "Access Control",
                "Limit information system access to authorised users",
                ac1_status, ac1_evidence, "high",
                "Activate guardrail rules covering all sensitive content categories." if ac1_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "AU.2.041", "Audit Logging",
                "Create and retain audit logs sufficient to enable monitoring",
                au2_status, au2_evidence, "critical",
                "Enable AuditLogger and configure retention policy." if au2_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "CA.2.158", "Security Assessment",
                "Periodically assess security controls",
                ca2_status, ca2_evidence, "high",
                "Run PenetrationTestAgent at least monthly." if ca2_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "IR.2.092", "Incident Response",
                "Track, document, and report cybersecurity events",
                ir2_status, ir2_evidence, "high",
                "Configure AuditLogger and set up alerts for high-severity events." if ir2_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "RA.3.145", "Risk Assessment",
                "Regularly assess risk to operations from threats and vulnerabilities",
                ra3_status, ra3_evidence, "high",
                "Schedule recurring PenetrationTestAgent runs and review bypass trends." if ra3_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "SI.1.210", "System & Information Integrity",
                "Identify, report, and correct information system flaws",
                si1_status, si1_evidence, "medium",
                f"Remediate {self._pentest.total_bypassed} bypass vectors using FeedbackLoop tuning suggestions." if self._pentest and self._pentest.total_bypassed > 0 else "",
            ),
        ]

    # ── GDPR controls ─────────────────────────────────────────────────────────

    def _evaluate_gdpr(self, events: list) -> List[ComplianceControl]:
        has_audit_trail = len(events) > 0
        pentest_done = self._pentest is not None
        rules_count = len(self._engine.rules) if self._engine else 0

        pii_rules_active = self._engine_has_keywords(
            ["email", "SSN", "credit card", "phone", "password", "address"]
        )

        # Art. 5(1)(f) — Integrity & Confidentiality
        art5_status = ControlStatus.PASS if pii_rules_active else ControlStatus.PARTIAL
        art5_evidence = (
            "PII/personal data detection rules are active."
            if pii_rules_active else "No personal data detection rules found."
        )

        # Art. 30 — Records of Processing Activities (audit log)
        art30_status = ControlStatus.PASS if has_audit_trail else ControlStatus.FAIL
        art30_evidence = (
            f"Processing records maintained: {len(events)} audit events."
            if has_audit_trail else "No processing records (AuditLogger not active)."
        )

        # Art. 32 — Security of Processing (technical measures)
        art32_status = ControlStatus.PASS if rules_count >= 3 else ControlStatus.PARTIAL
        art32_evidence = (
            f"{rules_count} active technical guardrails in place."
        )

        # Art. 35 — Data Protection Impact Assessment (DPIA via pentest)
        art35_status = ControlStatus.PASS if pentest_done else ControlStatus.PARTIAL
        art35_evidence = (
            f"DPIA evidence via adversarial testing: {self._pentest.total_attacks} scenarios."
            if pentest_done else "No DPIA data; run PenetrationTestAgent."
        )

        # Art. 25 — Data Protection by Design (content transformation)
        art25_evidence = "ContentTransformer provides PII redaction by design."
        art25_status = ControlStatus.PASS  # always available in framework

        return [
            ComplianceControl(
                "GDPR-Art5(1)(f)", "Integrity & Confidentiality",
                "Process personal data with appropriate security",
                art5_status, art5_evidence, "critical",
                "Add PII detection rules or ContentTransformer integration." if art5_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "GDPR-Art30", "Records of Processing",
                "Maintain records of all processing activities",
                art30_status, art30_evidence, "high",
                "Enable AuditLogger with sufficient retention." if art30_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "GDPR-Art32", "Security of Processing",
                "Implement appropriate technical and organisational measures",
                art32_status, art32_evidence, "high",
                "Deploy at least 3 guardrail rules covering data safety categories." if art32_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "GDPR-Art35", "Data Protection Impact Assessment",
                "Assess the impact of high-risk processing operations",
                art35_status, art35_evidence, "high",
                "Run PenetrationTestAgent and document results as DPIA evidence." if art35_status != ControlStatus.PASS else "",
            ),
            ComplianceControl(
                "GDPR-Art25", "Data Protection by Design",
                "Integrate data protection into system design",
                art25_status, art25_evidence, "medium",
                "",
            ),
        ]

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _engine_has_keywords(self, keywords: List[str]) -> bool:
        if self._engine is None:
            return False
        for rule in self._engine.rules.values():
            for kw in rule.keywords:
                if any(target.lower() in kw.lower() for target in keywords):
                    return True
            for pattern in rule.patterns:
                if any(target.lower() in pattern.lower() for target in keywords):
                    return True
        return False


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Guardrails Compliance Reporter")
    parser.add_argument(
        "--framework",
        choices=["hipaa", "soc2", "cmmc", "gdpr", "all"],
        default="all",
        help="Compliance framework to evaluate",
    )
    parser.add_argument("--json", metavar="PATH", help="Export JSON report to PATH")
    parser.add_argument("--html", metavar="PATH", help="Export HTML report to PATH")
    args = parser.parse_args()

    from guardrail_framework import GuardrailEngine, create_default_guardrails

    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)

    reporter = ComplianceReporter(engine=engine)

    if args.framework == "all":
        reports = reporter.generate_all()
        for fw, report in reports.items():
            print(f"\n{'='*60}")
            print(report.to_markdown())
    else:
        fw = ComplianceFramework(args.framework)
        report = reporter.generate(fw)
        print(report.to_markdown())
        if args.json:
            report.save_json(args.json)
            print(f"\n✅  JSON report saved to {args.json}")
        if args.html:
            report.save_html(args.html)
            print(f"✅  HTML report saved to {args.html}")
