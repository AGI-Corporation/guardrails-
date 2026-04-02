"""
🛡️ Guardrails API Server
"""
import os
import tempfile
import uuid

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, List, Optional

from compliance_reporter import ComplianceFramework, ComplianceReporter
from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    Severity,
    Action,
    create_default_guardrails,
)
from audit_logger import AuditLogger
from penetration_test_agent import PenetrationTestAgent, PenTestSession
from performance_profiler import PerformanceProfiler

app = FastAPI(title="Guardrails API", version="1.0.0")

# ── Shared state ──────────────────────────────────────────────────────────────
engine = GuardrailEngine()
for _r in create_default_guardrails():
    engine.add_rule(_r)

audit = AuditLogger()
profiler = PerformanceProfiler()

# In-memory store for pentest reports (keyed by report ID)
_pentest_reports: Dict[str, dict] = {}


# ── Request / Response models ─────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class GuardrailCreateRequest(BaseModel):
    id: str
    name: str
    severity: str  # low | medium | high | critical
    action: str    # allow | block | warn
    patterns: List[str] = []
    keywords: List[str] = []


class PenTestRunRequest(BaseModel):
    name: Optional[str] = None
    include_plugins: bool = True
    custom_seeds: List[str] = []


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    with profiler.time("api", "evaluate"):
        result = engine.evaluate(req.text)

    action_str = result.action.value
    audit.log(
        input_text=req.text,
        action_taken=action_str,
        matched_rules=result.matched_rules,
        severity=result.severity.value,
        user_id=req.user_id,
        session_id=req.session_id,
    )

    return {
        "action": action_str,
        "severity": result.severity.value,
        "matched_rules": result.matched_rules,
        "timestamp": result.timestamp,
    }


@app.post("/guardrails", status_code=201)
async def create_guardrail(req: GuardrailCreateRequest):
    try:
        severity = Severity(req.severity)
        action = Action(req.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    rule = GuardrailRule(
        id=req.id,
        name=req.name,
        severity=severity,
        action=action,
        patterns=req.patterns,
        keywords=req.keywords,
    )
    engine.add_rule(rule)
    return {"id": rule.id, "name": rule.name, "status": "created"}


@app.get("/guardrails")
async def list_guardrails():
    return [
        {
            "id": r.id,
            "name": r.name,
            "severity": r.severity.value,
            "action": r.action.value,
            "patterns": r.patterns,
            "keywords": r.keywords,
        }
        for r in engine.rules.values()
    ]


@app.delete("/guardrails/{rule_id}", status_code=200)
async def delete_guardrail(rule_id: str):
    if rule_id not in engine.rules:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    engine.remove_rule(rule_id)
    return {"id": rule_id, "status": "deleted"}


@app.get("/audit/logs")
async def get_audit_logs(limit: int = 50, offset: int = 0):
    logs = audit.get_logs(limit=limit, offset=offset)
    return [
        {
            "id": entry.id,
            "timestamp": entry.timestamp,
            "input_text": entry.input_text,
            "action_taken": entry.action_taken,
            "matched_rules": entry.matched_rules,
            "severity": entry.severity,
            "risk_score": entry.risk_score,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
        }
        for entry in logs
    ]


@app.get("/metrics")
async def metrics():
    stats = profiler.get_stats()
    total_rules = len(engine.rules)
    return {
        "total_rules": total_rules,
        "performance": stats,
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "rules_loaded": len(engine.rules)}


# ── Penetration Test endpoints ────────────────────────────────────────────────

def _run_pentest_sync(report_id: str, name: str, include_plugins: bool, custom_seeds: list):
    """Execute pentest in background and store result."""
    tmp_dir = tempfile.mkdtemp(prefix="pentest_")
    session = PenTestSession(
        name=name,
        include_plugins=include_plugins,
        custom_seeds=custom_seeds,
        audit_db_path=os.path.join(tmp_dir, "audit.db"),
        feedback_db_path=os.path.join(tmp_dir, "feedback.db"),
    )
    agent = PenetrationTestAgent(engine=engine, profiler=profiler)
    report = agent.run(session)
    _pentest_reports[report_id] = report.to_dict()
    _pentest_reports[report_id]["report_id"] = report_id
    _pentest_reports[report_id]["status"] = "complete"


@app.post("/pentest/run", status_code=202)
async def run_pentest(req: PenTestRunRequest, background_tasks: BackgroundTasks):
    """
    Kick off a penetration test session in the background.
    Returns a ``report_id`` that can be polled via ``GET /pentest/reports/{id}``.
    """
    report_id = str(uuid.uuid4())
    name = req.name or f"api-pentest-{report_id[:8]}"
    _pentest_reports[report_id] = {"report_id": report_id, "status": "running", "name": name}
    background_tasks.add_task(
        _run_pentest_sync, report_id, name, req.include_plugins, req.custom_seeds
    )
    return {"report_id": report_id, "status": "running", "name": name}


@app.get("/pentest/reports")
async def list_pentest_reports():
    """List all pentest reports (summary only)."""
    return [
        {
            "report_id": v.get("report_id"),
            "name": v.get("session_name") or v.get("name"),
            "status": v.get("status"),
            "total_attacks": v.get("total_attacks"),
            "overall_block_rate": v.get("overall_block_rate"),
            "started_at": v.get("started_at"),
        }
        for v in _pentest_reports.values()
    ]


@app.get("/pentest/reports/{report_id}")
async def get_pentest_report(report_id: str):
    """Retrieve a full pentest report by ID."""
    report = _pentest_reports.get(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail=f"Report '{report_id}' not found")
    return report


# ── Compliance endpoints ──────────────────────────────────────────────────────

@app.get("/compliance/{framework}")
async def compliance_report(framework: str):
    """
    Generate a compliance report for the specified framework.
    Supported: ``hipaa``, ``soc2``, ``cmmc``, ``gdpr``.
    """
    try:
        fw = ComplianceFramework(framework.lower())
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown framework '{framework}'. Use: hipaa, soc2, cmmc, gdpr",
        )
    reporter = ComplianceReporter(audit_logger=audit, engine=engine)
    report = reporter.generate(fw)
    return report.to_dict()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
