# 🛡️ Guardrails Framework

A professional AI safety guardrail testing and definition framework with LLM integration, adversarial testing, audit logging, REST API, and Streamlit dashboard.

## Overview

The Guardrails Framework is a comprehensive toolkit for defining, testing, and enforcing safety guardrails on Large Language Model (LLM) outputs. It provides organizations with the tools needed to ensure AI systems behave safely, ethically, and in alignment with defined policies.

## Features

- **Core Guardrail Engine** - Define and enforce content, topic, format, and custom guardrails
- **Adversarial Testing** - Automated red-teaming with prompt injection, jailbreak, and boundary testing
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
├── adversarial_tester.py     # Automated adversarial testing suite
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
