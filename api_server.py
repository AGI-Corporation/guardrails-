"""
🛡️ Guardrails API Server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from guardrail_framework import GuardrailEngine, create_default_guardrails

app = FastAPI(title="Guardrails API", version="2.0.0")
engine = GuardrailEngine()
for r in create_default_guardrails():
    engine.add_rule(r)


class EvaluateRequest(BaseModel):
    text: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    res = engine.evaluate(req.text)
    return {
        "text": res.text,
        "action": res.action,
        "matched_rules": res.matched_rules,
        "severity": res.severity,
        "risk_score": res.risk_score,
        "categories": res.categories,
        "timestamp": res.timestamp,
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/rules")
async def list_rules():
    return [
        {
            "id": r.id,
            "name": r.name,
            "category": r.category.value,
            "severity": r.severity.value,
            "action": r.action.value,
            "description": r.description,
        }
        for r in engine.list_rules()
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
