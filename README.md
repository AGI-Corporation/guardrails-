# 🛡️ Guardrails Framework

A professional AI safety guardrail testing and definition framework with LLM integration, adversarial testing, audit logging, REST API, and Streamlit dashboard.

## Overview

The Guardrails Framework is a comprehensive toolkit for defining, testing, and enforcing safety guardrails on Large Language Model (LLM) outputs. It provides organizations with the tools needed to ensure AI systems behave safely, ethically, and in alignment with defined policies.

## Features

- **Core Guardrail Engine** - Define and enforce content, topic, format, and custom guardrails
- **Adversarial Testing** - Automated red-teaming with prompt injection, jailbreak, encoding bypass, and boundary testing
- **Red Hat LLM Pen Testing** - Structured attack-probe suite (40+ probes) aligned with OWASP LLM Top-10 and MITRE ATLAS, covering prompt injection, jailbreak, persona hijacking, encoding bypass, indirect injection, system-prompt extraction, data exfiltration, multi-turn escalation, and token confusion
- **Audit Logging** - Immutable audit trail with risk scoring and compliance reporting
- **REST API** - FastAPI-powered server for guardrail evaluation at scale
- **Streamlit Dashboard** - Real-time monitoring and visualization of guardrail metrics
- **LLM Integration** - Supports OpenAI, Anthropic, and custom LLM backends
- **RAG Guardrails** - Retrieval-Augmented Generation safety with source validation
- **Plugin System** - Extensible architecture for custom guardrail plugins
- **Feedback Loop** - Continuous learning from guardrail decisions
- **Performance Profiling** - Latency and throughput monitoring

## Repository Structure

```
guardrails-/
├── guardrail_framework.py    # Core guardrail engine and definitions
├── audit_logger.py           # Immutable audit logging with risk scoring
├── api_server.py             # FastAPI REST API server
├── content_transformer.py    # Content transformation and sanitization
├── adversarial_tester.py     # Automated adversarial testing suite (mutation-based)
├── red_team_tester.py        # Red Hat LLM penetration testing (OWASP/MITRE aligned)
├── llm_wrapper.py            # LLM provider abstraction layer
├── rag_guardrails.py         # RAG-specific guardrail implementations
├── plugin_system.py          # Plugin architecture for extensibility
├── feedback_loop.py          # Continuous improvement feedback system
├── performance_profiler.py   # Performance monitoring and profiling
├── quickstart.py             # Quick start demo and examples
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AGI-Corporation/guardrails-.git
cd guardrails-

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from guardrail_framework import GuardrailFramework, GuardrailConfig, GuardrailType

# Initialize framework
framework = GuardrailFramework()

# Add a content safety guardrail
framework.add_guardrail(GuardrailConfig(
    name="content_safety",
    guardrail_type=GuardrailType.CONTENT,
    rules=["no_hate_speech", "no_violence"],
    action="block"
))

# Evaluate text
result = framework.evaluate("Your text here")
print(result)
```

Or run the quickstart demo:

```bash
python quickstart.py
```

## API Server

```bash
# Start the API server
uvicorn api_server:app --reload --port 8000

# API docs available at:
# http://localhost:8000/docs
```

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/evaluate` | Evaluate text against guardrails |
| POST | `/guardrails` | Create a new guardrail |
| GET | `/guardrails` | List all guardrails |
| GET | `/audit/logs` | Retrieve audit logs |
| GET | `/metrics` | Performance metrics |
| GET | `/health` | Health check |

## Dashboard

```bash
# Launch the Streamlit dashboard
streamlit run dashboard.py
```

## Adversarial Testing

```python
from adversarial_tester import AdversarialTester

tester = AdversarialTester(framework)
results = tester.run_full_suite()
print(results.summary())
```

Test categories include:
- Prompt injection attacks
- Jailbreak attempts
- Boundary condition testing
- Context manipulation
- Encoding bypass attempts
- Base64 / ROT-13 / reversed-text obfuscation
- Token-split and Markdown formatting obfuscation

## Red Hat LLM Penetration Testing

A structured adversarial probing suite aligned with **OWASP LLM Top-10** and **MITRE ATLAS**:

```python
from red_team_tester import RedTeamEngine, build_probe_catalog
from guardrail_framework import GuardrailEngine, create_default_guardrails

engine = GuardrailEngine()
for rule in create_default_guardrails():
    engine.add_rule(rule)

rt = RedTeamEngine(engine)
report = rt.run_full_suite()
print(report.summary())

# Export full Markdown report
with open("red_team_report.md", "w") as f:
    f.write(report.markdown_report())
```

Attack categories covered (40+ probes):

| Category | OWASP ref | MITRE ATLAS |
|----------|-----------|-------------|
| Prompt Injection | LLM01 | AML.T0051 |
| Jailbreak | LLM01 | AML.T0054 |
| Role-Play Attacks | LLM01 | AML.T0054 |
| Encoding Bypass (Base64, ROT-13, homoglyphs, zero-width) | LLM01 | AML.T0051 |
| Indirect Prompt Injection | LLM01 | AML.T0051 |
| System Prompt Extraction | LLM06 | AML.T0057 |
| Data Exfiltration | LLM06 | AML.T0057 |
| Multi-Turn Escalation | LLM01 | AML.T0054 |
| Token Confusion | LLM01 | AML.T0051 |
| Persona Hijacking (DAN, AIM, GOD MODE) | LLM01 | AML.T0054 |

Run interactively:

```bash
python red_team_tester.py
# or via quickstart:
python quickstart.py  # choose option 8
```

After each run the engine outputs a **bypass rate** per category and prioritised hardening recommendations.

## Plugin System

```python
from plugin_system import GuardrailPlugin, plugin_registry

class CustomPlugin(GuardrailPlugin):
    def evaluate(self, text: str) -> dict:
        # Custom logic here
        return {"passed": True, "score": 1.0}

plugin_registry.register("custom", CustomPlugin())
```

## Configuration

Create a `config.yaml` file:

```yaml
guardrails:
  content_safety:
    enabled: true
    action: block
    threshold: 0.8
  topic_restrictions:
    enabled: true
    blocked_topics: ["competitor", "internal"]
llm:
  provider: openai
  model: gpt-4
  api_key: ${OPENAI_API_KEY}
audit:
  enabled: true
  retention_days: 90
```

## Environment Variables

```bash
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GUARDRAILS_LOG_LEVEL=INFO
GUARDRAILS_AUDIT_PATH=./audit_logs
GUARDRAILS_DB_URL=sqlite:///guardrails.db
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run adversarial test suite
python -m pytest tests/test_adversarial.py -v
```

## Architecture

```
┌─────────────────────────────────────────────┐
│              Streamlit Dashboard             │
├─────────────────────────────────────────────┤
│               FastAPI REST API               │
├──────────────┬──────────────┬───────────────┤
│  Guardrail   │     LLM      │     RAG       │
│   Engine     │   Wrapper    │  Guardrails   │
├──────────────┴──────────────┴───────────────┤
│         Audit Logger + Performance          │
├─────────────────────────────────────────────┤
│     Plugin System + Feedback Loop           │
└─────────────────────────────────────────────┘
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-guardrail`)
3. Commit your changes (`git commit -am 'Add new guardrail type'`)
4. Push to the branch (`git push origin feature/my-guardrail`)
5. Open a Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## Built By

**AGI Corporation** - Building safe and responsible AI systems.

---

*Part of the AGI Corporation open-source AI safety toolkit.*
