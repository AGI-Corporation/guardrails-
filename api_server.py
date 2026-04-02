"""
🛡️ Guardrails API Server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    Severity,
    Action,
    create_default_guardrails,
)
from audit_logger import AuditLogger, AuditEntry
from performance_profiler import PerformanceProfiler

app = FastAPI(title="Guardrails API", version="1.0.0")

# ── Shared state ──────────────────────────────────────────────────────────────
engine = GuardrailEngine()
for _r in create_default_guardrails():
    engine.add_rule(_r)

audit = AuditLogger()
profiler = PerformanceProfiler()


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
