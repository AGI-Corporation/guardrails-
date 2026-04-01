"""
🛡️ Guardrails Framework — Core Engine

Enterprise-grade guardrail evaluation covering PII, prompt injection,
jailbreaks, content safety, security exploits, and compliance rules.
"""
import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GuardrailEngine")


class Severity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Action(Enum):
    ALLOW = "allow"
    BLOCK = "block"
    WARN = "warn"


class RuleCategory(Enum):
    PII = "pii"
    PROMPT_INJECTION = "prompt_injection"
    JAILBREAK = "jailbreak"
    CONTENT_SAFETY = "content_safety"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    SECRETS = "secrets"
    CUSTOM = "custom"


@dataclass
class GuardrailRule:
    id: str
    name: str
    severity: Severity
    action: Action
    category: RuleCategory = RuleCategory.CUSTOM
    patterns: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    description: str = ""

    def matches(self, text: str) -> bool:
        for p in self.patterns:
            if re.search(p, text, re.IGNORECASE):
                return True
        for k in self.keywords:
            if k.lower() in text.lower():
                return True
        return False


@dataclass
class EvaluationResult:
    """Structured result returned by GuardrailEngine.evaluate()."""
    text: str
    action: str                    # "allow", "block", or "warn"
    matched_rules: List[str]
    severity: str                  # highest severity among matched rules
    risk_score: float              # 0.0–1.0
    timestamp: str
    categories: List[str] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.action == "block"


_SEVERITY_RANK = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}

_SEVERITY_SCORE = {
    Severity.LOW: 0.25,
    Severity.MEDIUM: 0.5,
    Severity.HIGH: 0.75,
    Severity.CRITICAL: 1.0,
}


class GuardrailEngine:
    """Evaluate text against a set of GuardrailRules and return an EvaluationResult."""

    def __init__(self):
        self.rules: Dict[str, GuardrailRule] = {}

    def add_rule(self, rule: GuardrailRule):
        self.rules[rule.id] = rule

    def remove_rule(self, rule_id: str):
        self.rules.pop(rule_id, None)

    def list_rules(self) -> List[GuardrailRule]:
        return list(self.rules.values())

    def evaluate(self, text: str) -> EvaluationResult:
        matched_rules = [r for r in self.rules.values() if r.matches(text)]
        matched_ids = [r.id for r in matched_rules]
        categories = list({r.category.value for r in matched_rules})
        now = datetime.now(timezone.utc).isoformat()

        if not matched_rules:
            return EvaluationResult(
                text=text,
                action=Action.ALLOW.value,
                matched_rules=[],
                severity=Severity.LOW.value,
                risk_score=0.0,
                timestamp=now,
            )

        # Determine most severe rule
        most_severe = max(matched_rules, key=lambda r: _SEVERITY_RANK[r.severity])
        # Determine strictest action (block > warn > allow)
        if any(r.action == Action.BLOCK for r in matched_rules):
            final_action = Action.BLOCK
        elif any(r.action == Action.WARN for r in matched_rules):
            final_action = Action.WARN
        else:
            final_action = Action.ALLOW

        risk_score = min(1.0, sum(_SEVERITY_SCORE[r.severity] for r in matched_rules) / 2)

        return EvaluationResult(
            text=text,
            action=final_action.value,
            matched_rules=matched_ids,
            severity=most_severe.severity.value,
            risk_score=round(risk_score, 3),
            timestamp=now,
            categories=categories,
        )


# ── Default enterprise ruleset ────────────────────────────────────────────────

def create_default_guardrails() -> List[GuardrailRule]:
    """
    Return a comprehensive set of enterprise guardrail rules covering PII,
    prompt injection, jailbreaks, content safety, security exploits,
    secrets, and compliance patterns.
    """
    return [
        # ── PII ────────────────────────────────────────────────────────────
        GuardrailRule(
            "pii_ssn", "Social Security Number", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.PII,
            patterns=[r"\b\d{3}-\d{2}-\d{4}\b", r"\b\d{9}\b"],
            description="Detects US Social Security Numbers",
        ),
        GuardrailRule(
            "pii_credit_card", "Credit Card Number", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.PII,
            patterns=[r"\b(?:\d{4}[\s\-]?){3}\d{4}\b"],
            description="Detects credit/debit card numbers",
        ),
        GuardrailRule(
            "pii_email", "Email Address", Severity.MEDIUM, Action.WARN,
            category=RuleCategory.PII,
            patterns=[r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"],
            description="Detects email addresses",
        ),
        GuardrailRule(
            "pii_phone", "Phone Number", Severity.MEDIUM, Action.WARN,
            category=RuleCategory.PII,
            patterns=[r"\b(?:\+?1[\s\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}\b"],
            description="Detects US phone numbers",
        ),
        GuardrailRule(
            "pii_passport", "Passport Number", Severity.HIGH, Action.BLOCK,
            category=RuleCategory.PII,
            patterns=[
                r"(?i)(passport\s+(number|no\.?|#)\s*[:\-]?\s*[A-Z]{1,2}\d{6,9})",
            ],
            keywords=["passport number", "passport no"],
            description="Detects passport numbers (requires keyword context to reduce false positives)",
        ),
        GuardrailRule(
            "pii_drivers_license", "Driver's License", Severity.HIGH, Action.BLOCK,
            category=RuleCategory.PII,
            keywords=["driver's license", "drivers license", "driver license", "dl number", "license number"],
            description="Detects driver's license references",
        ),
        GuardrailRule(
            "pii_ip_address", "IP Address", Severity.LOW, Action.WARN,
            category=RuleCategory.PII,
            patterns=[
                r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b",
            ],
            description="Detects IPv4 addresses",
        ),
        GuardrailRule(
            "pii_dob", "Date of Birth", Severity.MEDIUM, Action.WARN,
            category=RuleCategory.PII,
            patterns=[r"\bdate of birth\b", r"\bdob\b", r"\bborn on\b"],
            keywords=["date of birth", "dob", "born on"],
            description="Detects date-of-birth references",
        ),

        # ── Secrets / API Keys ─────────────────────────────────────────────
        GuardrailRule(
            "secret_api_key", "API Key Pattern", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.SECRETS,
            patterns=[
                r"\bsk-[A-Za-z0-9]{20,}\b",          # OpenAI-style
                r"\bghp_[A-Za-z0-9]{36}\b",            # GitHub personal token
                r"\bAKIA[A-Z0-9]{16}\b",               # AWS access key
                r"(?i)(api[_\-\s]?key|secret[_\-\s]?key)\s*[=:]\s*['\"]?\w{16,}",
            ],
            description="Detects API keys and secret tokens",
        ),
        GuardrailRule(
            "secret_jwt", "JWT Token", Severity.HIGH, Action.BLOCK,
            category=RuleCategory.SECRETS,
            patterns=[r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"],
            description="Detects JSON Web Tokens",
        ),
        GuardrailRule(
            "secret_private_key", "Private Key", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.SECRETS,
            patterns=[r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----"],
            description="Detects PEM-encoded private keys",
        ),

        # ── Prompt Injection ───────────────────────────────────────────────
        GuardrailRule(
            "injection_ignore_instructions", "Ignore Instructions Injection",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.PROMPT_INJECTION,
            patterns=[
                r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+instructions?",
                r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions?",
                r"forget\s+(all\s+)?(previous|prior|above)\s+instructions?",
            ],
            keywords=["ignore all instructions", "ignore previous instructions",
                      "disregard instructions", "forget your instructions"],
            description="Detects attempts to override system instructions",
        ),
        GuardrailRule(
            "injection_system_override", "System Prompt Override",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.PROMPT_INJECTION,
            patterns=[
                r"(?i)(new\s+system\s+prompt|override\s+system|system\s*:\s*you\s+are\s+now)",
                r"(?i)\[SYSTEM\]\s*:.{0,50}(ignore|override|replace)",
                r"(?i)###\s*system\s*###",
            ],
            keywords=["new system prompt", "override system prompt",
                      "replace system message", "you are now a"],
            description="Detects attempts to inject a new system prompt",
        ),
        GuardrailRule(
            "injection_indirect", "Indirect Prompt Injection",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.PROMPT_INJECTION,
            patterns=[
                r"(?i)when\s+(the\s+)?user\s+(asks?|says?|types?).{0,50}(respond|say|reply)\s+with",
                r"(?i)hidden\s+instruction",
                r"(?i)secret\s+(command|instruction|directive)",
            ],
            keywords=["hidden instruction", "secret directive",
                      "do not reveal", "keep this secret from the user",
                      "hidden system prompt", "reveal the system prompt",
                      "ignore and reveal"],
            description="Detects indirect/embedded prompt injection attempts",
        ),
        GuardrailRule(
            "injection_delimiter_attack", "Delimiter Injection Attack",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.PROMPT_INJECTION,
            patterns=[
                r"(?i)<\s*/?\s*system\s*>",
                r"(?i)\[INST\]|\[/INST\]",
                r"(?i)<\|im_start\|>|<\|im_end\|>",
                r"(?i)<<SYS>>|<</SYS>>",
            ],
            description="Detects delimiter tokens used to hijack model context",
        ),

        # ── Jailbreak ──────────────────────────────────────────────────────
        GuardrailRule(
            "jailbreak_dan", "DAN Jailbreak", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.JAILBREAK,
            patterns=[
                r"(?i)\bDAN\b.{0,80}(mode|prompt|jailbreak|unrestricted)",
                r"(?i)do\s+anything\s+now",
                r"(?i)you\s+are\s+now\s+DAN",
            ],
            keywords=["DAN mode", "do anything now", "jailbreak mode",
                      "unrestricted mode", "no restrictions"],
            description="Detects DAN and similar jailbreak prompts",
        ),
        GuardrailRule(
            "jailbreak_roleplay", "Roleplay-based Jailbreak",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.JAILBREAK,
            patterns=[
                r"(?i)(pretend|imagine|act\s+as|role.?play).{0,60}(no\s+rule|unrestricted|without\s+(restriction|limit|filter|safety|guideline))",
                r"(?i)you\s+are\s+(now\s+)?an?\s+(unrestricted|uncensored|unfiltered|evil|malicious)\s+(AI|assistant|model|bot|LLM)",
                r"(?i)(fictional|hypothetical)\s+(scenario|world|universe).{0,80}(allow|permit|do|tell)",
            ],
            keywords=["act as an unrestricted AI", "pretend you have no rules",
                      "imagine you are uncensored"],
            description="Detects roleplay-framed jailbreak attempts",
        ),
        GuardrailRule(
            "jailbreak_token_smuggling", "Token Smuggling / Encoding Attack",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.JAILBREAK,
            patterns=[
                r"(?i)base64\s*decode",
                r"(?i)rot13",
                r"(?i)hex\s*decode",
                r"(?i)url\s*decode.{0,30}(execute|run|eval)",
            ],
            keywords=["decode this", "base64 encoded instruction"],
            description="Detects encoding-based prompt smuggling",
        ),
        GuardrailRule(
            "jailbreak_developer_mode", "Developer/God Mode Jailbreak",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.JAILBREAK,
            patterns=[
                r"(?i)(developer|god|admin|maintenance|debug)\s+mode",
                r"(?i)enable\s+(developer|maintenance|unrestricted)\s+mode",
            ],
            keywords=["developer mode", "god mode", "maintenance mode",
                      "debug mode enabled", "admin override"],
            description="Detects developer/god-mode jailbreak triggers",
        ),

        # ── Content Safety ─────────────────────────────────────────────────
        GuardrailRule(
            "safety_violence", "Violence / Harm Instructions",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.CONTENT_SAFETY,
            keywords=[
                "how to kill", "how to hurt", "how to harm", "make a bomb",
                "build explosive", "instructions to kill", "how to attack",
                "how to assault", "how to poison",
            ],
            patterns=[
                r"(?i)(step.by.step|detailed?)\s+(guide|instructions?|tutorial)\s+(to|for|on)\s+(kill|harm|hurt|attack|poison|injur)",
            ],
            description="Detects requests for violence or harm instructions",
        ),
        GuardrailRule(
            "safety_self_harm", "Self-Harm Content", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.CONTENT_SAFETY,
            keywords=[
                "how to commit suicide", "methods to end my life",
                "how to cut myself", "self harm methods", "how to self harm",
            ],
            patterns=[
                r"(?i)(how\s+to\s+)?(commit\s+)?suicide",
                r"(?i)want\s+to\s+(die|end\s+(my|this)\s+life)",
            ],
            description="Detects self-harm and suicide-related content",
        ),
        GuardrailRule(
            "safety_hate_speech", "Hate Speech", Severity.HIGH, Action.BLOCK,
            category=RuleCategory.CONTENT_SAFETY,
            patterns=[
                r"(?i)(all|those)\s+\w+\s+(are|should\s+be)\s+(subhuman|inferior|eliminated|exterminated)",
                r"(?i)(racial|ethnic|religious)\s+slur",
            ],
            keywords=[
                "white supremacy", "ethnic cleansing", "racial superiority",
                "genocide", "exterminate", "inferior race",
            ],
            description="Detects hate speech and discriminatory content",
        ),
        GuardrailRule(
            "safety_csam", "Child Safety", Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.CONTENT_SAFETY,
            patterns=[
                r"(?i)(sexual|explicit|nude).{0,30}(minor|child|underage|teen\b|youth)",
                r"(?i)(child|minor|underage).{0,30}(sexual|explicit|nude|naked)",
            ],
            description="Detects child safety violations",
        ),
        GuardrailRule(
            "safety_weapons_mass_destruction", "WMD / CBRN Content",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.CONTENT_SAFETY,
            keywords=[
                "bioweapon", "chemical weapon", "nerve agent", "anthrax",
                "sarin", "vx gas", "nuclear device", "dirty bomb",
                "radiological weapon",
            ],
            patterns=[
                r"(?i)(synthesize|create|make|build|weaponize).{0,40}(nerve\s+agent|bioweapon|chemical\s+weapon|pathogen)",
            ],
            description="Detects WMD/CBRN weapon-related requests",
        ),

        # ── Security ───────────────────────────────────────────────────────
        GuardrailRule(
            "security_sql_injection", "SQL Injection Pattern",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.SECURITY,
            patterns=[
                r"(?i)('|\")\s*(OR|AND)\s+('|\")?\d+('|\")?\s*=\s*('|\")?\d+",
                r"(?i)(UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM|UPDATE\s+\w+\s+SET)",
                r"(?i)--\s*$",
                r"(?i);\s*(DROP|DELETE|INSERT|UPDATE|CREATE|ALTER)\s+",
            ],
            description="Detects SQL injection patterns",
        ),
        GuardrailRule(
            "security_xss", "Cross-Site Scripting (XSS)",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.SECURITY,
            patterns=[
                r"(?i)<script[\s>]",
                r"(?i)javascript\s*:",
                r"(?i)on(load|click|mouseover|error|focus)\s*=",
                r"(?i)<iframe[\s>]",
                r"(?i)document\.(cookie|write|location)",
            ],
            description="Detects XSS attack patterns",
        ),
        GuardrailRule(
            "security_path_traversal", "Path Traversal Attack",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.SECURITY,
            patterns=[
                r"\.\./|\.\.\\",
                r"(?i)/etc/(passwd|shadow|sudoers|hosts)",
                r"(?i)C:\\Windows\\System32",
            ],
            description="Detects directory traversal attempts",
        ),
        GuardrailRule(
            "security_code_execution", "Code Execution Request",
            Severity.HIGH, Action.WARN,
            category=RuleCategory.SECURITY,
            patterns=[
                r"(?i)(eval|exec|os\.system|subprocess\.run|__import__)\s*\(",
                r"(?i)\$\(\s*[^)]+\s*\)",  # Shell command substitution
            ],
            keywords=["execute arbitrary code", "run shell command",
                      "system command", "remote code execution"],
            description="Detects potential code execution requests",
        ),

        # ── Compliance ────────────────────────────────────────────────────
        GuardrailRule(
            "compliance_hipaa", "HIPAA – Protected Health Information",
            Severity.HIGH, Action.BLOCK,
            category=RuleCategory.COMPLIANCE,
            keywords=[
                "medical record number", "health insurance", "diagnosis code",
                "icd-10", "patient id", "npi number", "hipaa",
                "protected health information", "phi",
            ],
            patterns=[
                r"(?i)(patient|medical)\s+(record|id|number|identifier)",
            ],
            description="Detects HIPAA-regulated PHI references",
        ),
        GuardrailRule(
            "compliance_pci", "PCI-DSS – Cardholder Data",
            Severity.CRITICAL, Action.BLOCK,
            category=RuleCategory.COMPLIANCE,
            keywords=[
                "cardholder data", "cvv", "cvc", "card verification",
                "primary account number", "pan", "magnetic stripe",
            ],
            patterns=[
                r"(?i)cvv\s*[=:]\s*\d{3,4}",
                r"(?i)expiry\s*[=:]?\s*\d{2}/\d{2,4}",
            ],
            description="Detects PCI-DSS cardholder data",
        ),
        GuardrailRule(
            "compliance_gdpr_data_transfer", "GDPR – Cross-border Data Transfer",
            Severity.MEDIUM, Action.WARN,
            category=RuleCategory.COMPLIANCE,
            keywords=[
                "transfer personal data", "data subject", "data controller",
                "lawful basis", "right to erasure", "right to access", "gdpr",
            ],
            description="Detects GDPR-related personal data transfer language",
        ),
    ]


if __name__ == "__main__":
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    result = e.evaluate("My SSN is 123-45-6789")
    print(f"Action: {result.action} | Rules: {result.matched_rules} | Score: {result.risk_score}")

