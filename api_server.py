"""
FastAPI REST API Server for Guardrail Framework
Provides endpoints for evaluating text, managing rules, and running tests.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
import uvicorn

from guardrail_framework import (
    GuardrailEngine, GuardrailRule, GuardrailCategory, Severity, Action,
    create_default_guardrails, create_default_test_cases, ReportGenerator
)
from audit_logger import AuditLogger, create_audit_entry


# ── App setup ────────────────────────────────────────────────────────────────

api_app = FastAPI(
    title="Guardrail API",
    description="REST API for AI content guardrail evaluation",
    version="1.0.0",
)

api_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared state
_engine = GuardrailEngine()
_audit = AuditLogger()
for _rule in create_default_guardrails():
    _engine.add_rule(_rule)
for _tc in create_default_test_cases():
    _engine.add_test_case(_tc)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class EvaluateRequest(BaseModel):
    text: str = Field(..., description="Text to evaluate")
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Optional[Dict] = None


class EvaluateResponse(BaseModel):
    action: str
    matched_rules: List[str]
    severity: str
    timestamp: str


class RuleRequest(BaseModel):
    id: str
    name: str
    category: str
    severity: str
    action: str
    patterns: List[str] = []
    keywords: List[str] = []
    description: str = ""
    enabled: bool = True


# ── Endpoints ────────────────────────────────────────────────────────────────

@api_app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@api_app.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    result = _engine.evaluate(req.text)
    _audit.log(create_audit_entry(
        input_text=req.text,
        action_taken=result.action,
        matched_rules=result.matched_rules,
        severity=result.severity,
        user_id=req.user_id,
        session_id=req.session_id,
        metadata=req.metadata or {},
    ))
    return EvaluateResponse(
        action=result.action,
        matched_rules=result.matched_rules,
        severity=result.severity,
        timestamp=result.timestamp,
    )


@api_app.get("/rules")
def list_rules():
    return [
        {
            "id": r.id,
            "name": r.name,
            "category": r.category.value,
            "severity": r.severity.value,
            "action": r.action.value,
            "enabled": r.enabled,
        }
        for r in _engine.rules.values()
    ]


@api_app.post("/rules")
def add_rule(req: RuleRequest):
    try:
        rule = GuardrailRule(
            id=req.id,
            name=req.name,
            category=GuardrailCategory(req.category),
            severity=Severity(req.severity),
            action=Action(req.action),
            patterns=req.patterns,
            keywords=req.keywords,
            description=req.description,
            enabled=req.enabled,
        )
        _engine.add_rule(rule)
        return {"status": "added", "rule_id": req.id}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@api_app.delete("/rules/{rule_id}")
def delete_rule(rule_id: str):
    if rule_id not in _engine.rules:
        raise HTTPException(status_code=404, detail="Rule not found")
    _engine.remove_rule(rule_id)
    return {"status": "deleted", "rule_id": rule_id}


@api_app.post("/tests/run")
def run_tests():
    results = _engine.run_tests()
    report = ReportGenerator().generate(results)
    passed = sum(1 for r in results if r.passed)
    return {
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": round(passed / len(results) * 100, 1) if results else 0,
        "report": report,
    }


@api_app.get("/audit/stats")
def audit_stats():
    return _audit.get_statistics()


@api_app.get("/audit/log")
def audit_log(
    action: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
):
    entries = _audit.query(action=action, severity=severity, limit=limit)
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp,
            "action_taken": e.action_taken,
            "severity": e.severity,
            "matched_rules": e.matched_rules,
            "user_id": e.user_id,
        }
        for e in entries
    ]


# ── Entry point ───────────────────────────────────────────────────────────────

def run_api_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run(api_app, host=host, port=port)


if __name__ == "__main__":
    run_api_server()
