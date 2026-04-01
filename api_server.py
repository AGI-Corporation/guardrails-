"""
🛡️ Guardrails API Server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from guardrail_framework import GuardrailEngine, GuardrailRule, Severity, Action, create_default_guardrails
from audit_logger import AuditLogger
from performance_profiler import PerformanceProfiler

app = FastAPI(title="Guardrails API")
engine = GuardrailEngine()
for r in create_default_guardrails():
    engine.add_rule(r)

audit_logger = AuditLogger()
profiler = PerformanceProfiler()


class EvaluateRequest(BaseModel):
    text: str


class GuardrailRuleRequest(BaseModel):
    id: str
    name: str
    severity: str = "medium"
    action: str = "block"
    patterns: List[str] = []
    keywords: List[str] = []


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    with profiler.time("api", "evaluate"):
        res = engine.evaluate(req.text)
    action = res["action"]
    action_str = action.value if hasattr(action, "value") else str(action)
    severity_str = res["severity"]  # already a string from evaluate()
    audit_logger.log(
        input_text=req.text,
        action_taken=action_str,
        matched_rules=res["matches"],
        severity=severity_str,
        risk_score=res["risk_score"],
    )
    return {
        "text": res["text"],
        "action": action_str,
        "matched_rules": res["matches"],
        "severity": severity_str,
        "risk_score": res["risk_score"],
        "timestamp": res["timestamp"],
    }


@app.post("/guardrails")
async def create_guardrail(rule_req: GuardrailRuleRequest):
    try:
        severity = Severity[rule_req.severity.upper()]
        action = Action(rule_req.action)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    rule = GuardrailRule(
        id=rule_req.id,
        name=rule_req.name,
        severity=severity,
        action=action,
        patterns=rule_req.patterns,
        keywords=rule_req.keywords,
    )
    engine.add_rule(rule)
    return {"status": "created", "id": rule.id}


@app.get("/guardrails")
async def list_guardrails():
    return {"guardrails": engine.list_rules()}


@app.delete("/guardrails/{rule_id}")
async def delete_guardrail(rule_id: str):
    if rule_id not in engine.rules:
        raise HTTPException(status_code=404, detail=f"Rule '{rule_id}' not found")
    engine.remove_rule(rule_id)
    return {"status": "deleted", "id": rule_id}


@app.get("/audit/logs")
async def get_audit_logs(limit: int = 100, offset: int = 0):
    logs = audit_logger.get_logs(limit=limit, offset=offset)
    return {"logs": logs, "count": len(logs)}


@app.get("/metrics")
async def get_metrics():
    stats = profiler.get_stats()
    total_rules = len(engine.rules)
    return {
        "total_rules": total_rules,
        "performance": stats,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
