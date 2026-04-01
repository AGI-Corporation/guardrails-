"""
🛡️ Guardrails API Server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from guardrail_framework import (
    GuardrailEngine,
    GuardrailRule,
    Severity,
    Action,
    create_default_guardrails,
)
from audit_logger import AuditLogger

app = FastAPI(title="Guardrails API", version="1.0.0")

# ── Shared state ───────────────────────────────────────────────────────────
_engine = GuardrailEngine()
for _r in create_default_guardrails():
    _engine.add_rule(_r)

_audit = AuditLogger()


# ── Request / response models ──────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class EvaluateResponse(BaseModel):
    text: str
    action: str
    matched_rules: List[str]
    severity: str
    risk_score: float
    timestamp: str


class CreateRuleRequest(BaseModel):
    id: str
    name: str
    severity: str  # low | medium | high | critical
    action: str    # allow | block | warn
    patterns: List[str] = []
    keywords: List[str] = []
    description: str = ""


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest):
    res = _engine.evaluate(req.text)
    _audit.log(
        input_text=req.text,
        action_taken=res.action,
        matched_rules=res.matched_rules,
        severity=res.severity,
        risk_score=res.risk_score,
        user_id=req.user_id,
        session_id=req.session_id,
    )
    return EvaluateResponse(
        text=res.text,
        action=res.action,
        matched_rules=res.matched_rules,
        severity=res.severity,
        risk_score=res.risk_score,
        timestamp=res.timestamp,
    )


@app.get("/guardrails")
async def list_guardrails() -> Dict[str, Any]:
    return {"guardrails": _engine.list_rules()}


@app.post("/guardrails", status_code=201)
async def create_guardrail(req: CreateRuleRequest) -> Dict[str, Any]:
    try:
        severity = Severity(req.severity)
        action = Action(req.action)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    if req.id in _engine.rules:
        raise HTTPException(status_code=409, detail=f"Rule '{req.id}' already exists.")

    rule = GuardrailRule(
        id=req.id,
        name=req.name,
        severity=severity,
        action=action,
        patterns=req.patterns,
        keywords=req.keywords,
        description=req.description,
    )
    _engine.add_rule(rule)
    return {"message": f"Rule '{req.id}' created.", "id": req.id}


@app.delete("/guardrails/{rule_id}")
async def delete_guardrail(rule_id: str) -> Dict[str, Any]:
    if not _engine.remove_rule(rule_id):
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found.")
    return {"message": f"Rule '{rule_id}' deleted."}


@app.get("/audit/logs")
async def get_audit_logs(limit: int = 50, offset: int = 0) -> Dict[str, Any]:
    logs = _audit.get_logs(limit=limit, offset=offset)
    return {"logs": logs, "count": len(logs)}


@app.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    return _audit.get_metrics()


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
