"""
🛡️ Guardrails API Server
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from guardrail_framework import GuardrailEngine, create_default_guardrails

app = FastAPI(title="Guardrails API")
engine = GuardrailEngine()
for r in create_default_guardrails(): engine.add_rule(r)

class EvaluateRequest(BaseModel):
    text: str

@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    res = engine.evaluate(req.text)
    # Convert Action and Severity enums to strings for JSON
    return {
        "text": res.text,
        "action": res.action.value,
        "matched_rules": res.matched_rules,
        "severity": res.severity.value,
        "risk_score": res.risk_score,
        "timestamp": res.timestamp
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
