"""
Tests for policy_manager.py — PolicyManager, PolicyConfig, validation,
env-var expansion, engine integration, rate-limiter building.
"""
import json
import os

import pytest

from guardrail_framework import GuardrailEngine, create_default_guardrails
from policy_manager import (
    AuditConfig,
    GuardrailRuleConfig,
    LLMConfig,
    PolicyConfig,
    PolicyManager,
    PolicyValidationError,
    ProfilingConfig,
    RateLimitingConfig,
    load_policy_dict,
    write_example_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def engine() -> GuardrailEngine:
    e = GuardrailEngine()
    for r in create_default_guardrails():
        e.add_rule(r)
    return e


@pytest.fixture()
def pm(engine) -> PolicyManager:
    return PolicyManager(engine=engine)


@pytest.fixture()
def pm_no_engine() -> PolicyManager:
    return PolicyManager(engine=None)


_MINIMAL_RULE = {
    "id": "test_ssn",
    "name": "SSN",
    "severity": "critical",
    "action": "block",
    "patterns": [r"\d{3}-\d{2}-\d{4}"],
}

_MINIMAL_POLICY = {
    "guardrails": [_MINIMAL_RULE],
}

_FULL_POLICY = {
    "guardrails": [
        _MINIMAL_RULE,
        {
            "id": "prompt_injection",
            "name": "Prompt Injection",
            "severity": "high",
            "action": "block",
            "keywords": ["ignore all previous instructions"],
            "enabled": True,
        },
        {
            "id": "disabled_rule",
            "name": "Disabled",
            "severity": "low",
            "action": "warn",
            "keywords": ["test"],
            "enabled": False,
        },
    ],
    "llm": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet",
        "api_key": "sk-ant-test",
        "temperature": 0.5,
        "max_tokens": 2048,
    },
    "audit": {
        "enabled": True,
        "retention_days": 30,
        "db_path": ":memory:",
    },
    "profiling": {
        "enabled": False,
        "slow_threshold_ms": 100,
    },
    "rate_limiting": {
        "enabled": True,
        "algorithm": "token_bucket",
        "capacity": 20,
        "refill_rate": 2.0,
    },
}


# ── GuardrailRuleConfig ───────────────────────────────────────────────────────

class TestGuardrailRuleConfig:
    def test_to_rule_basic(self):
        cfg = GuardrailRuleConfig(
            id="r1", name="R1", severity="high", action="block",
            patterns=[r"\btest\b"],
        )
        rule = cfg.to_rule()
        assert rule.id == "r1"
        assert rule.name == "R1"

    def test_to_rule_severity_enum(self):
        from guardrail_framework import Severity
        cfg = GuardrailRuleConfig(
            id="r1", name="R1", severity="critical", action="block",
            keywords=["hi"],
        )
        rule = cfg.to_rule()
        assert rule.severity == Severity.CRITICAL

    def test_to_rule_action_enum(self):
        from guardrail_framework import Action
        cfg = GuardrailRuleConfig(
            id="r1", name="R1", severity="low", action="warn",
            keywords=["hi"],
        )
        rule = cfg.to_rule()
        assert rule.action == Action.WARN

    def test_enabled_default_true(self):
        cfg = GuardrailRuleConfig(
            id="r1", name="R1", severity="medium", action="allow", keywords=["x"]
        )
        assert cfg.enabled is True


# ── PolicyConfig ──────────────────────────────────────────────────────────────

class TestPolicyConfig:
    def test_active_rules_excludes_disabled(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert len(config.active_rules) == 2  # 3 total, 1 disabled

    def test_to_dict_has_required_keys(self):
        pm = PolicyManager()
        config = pm.load_dict(_MINIMAL_POLICY)
        d = config.to_dict()
        assert "guardrails" in d
        assert "llm" in d
        assert "audit" in d
        assert "profiling" in d
        assert "rate_limiting" in d

    def test_to_json_valid(self):
        pm = PolicyManager()
        config = pm.load_dict(_MINIMAL_POLICY)
        data = json.loads(config.to_json())
        assert data["guardrails"][0]["id"] == "test_ssn"

    def test_guardrails_count(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert len(config.guardrails) == 3

    def test_llm_fields(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert config.llm.provider == "anthropic"
        assert config.llm.temperature == pytest.approx(0.5)
        assert config.llm.max_tokens == 2048

    def test_audit_fields(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert config.audit.enabled is True
        assert config.audit.retention_days == 30

    def test_profiling_fields(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert config.profiling.enabled is False
        assert config.profiling.slow_threshold_ms == pytest.approx(100.0)

    def test_rate_limiting_fields(self):
        pm = PolicyManager()
        config = pm.load_dict(_FULL_POLICY)
        assert config.rate_limiting.enabled is True
        assert config.rate_limiting.capacity == pytest.approx(20.0)


# ── PolicyManager.load_dict ───────────────────────────────────────────────────

class TestPolicyManagerLoadDict:
    def test_basic_load(self, pm):
        config = pm.load_dict(_MINIMAL_POLICY)
        assert len(config.guardrails) == 1

    def test_applies_rules_to_engine(self, engine):
        engine_rules_before = len(engine.rules)
        pm = PolicyManager(engine=engine, replace_rules=True)
        pm.load_dict(_MINIMAL_POLICY)
        assert len(engine.rules) == 1  # only the policy rule

    def test_replace_false_appends(self, engine):
        initial = len(engine.rules)
        pm = PolicyManager(engine=engine, replace_rules=False)
        pm.load_dict(_MINIMAL_POLICY)
        assert len(engine.rules) == initial + 1

    def test_disabled_rules_not_in_engine(self, engine):
        pm = PolicyManager(engine=engine, replace_rules=True)
        pm.load_dict(_FULL_POLICY)
        assert "disabled_rule" not in engine.rules

    def test_no_engine_just_parses(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert config is not None
        assert len(config.guardrails) == 1

    def test_current_config_set(self, pm):
        pm.load_dict(_MINIMAL_POLICY)
        assert pm.current_config is not None

    def test_empty_guardrails_accepted(self, pm_no_engine):
        # Valid: no rules defined yet is OK for a minimal config
        # (rules list empty is allowed — no rules to wire)
        # Actually our validator requires at least patterns or keywords per rule,
        # but an empty list of rules is fine
        config = pm_no_engine.load_dict({"guardrails": []})
        assert config.guardrails == []

    def test_default_llm_config(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"

    def test_default_audit_config(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert config.audit.enabled is True
        assert config.audit.retention_days == 90

    def test_default_profiling_config(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert config.profiling.enabled is True

    def test_default_rate_limiting_disabled(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert config.rate_limiting.enabled is False

    def test_raw_dict_preserved(self, pm_no_engine):
        config = pm_no_engine.load_dict(_MINIMAL_POLICY)
        assert "guardrails" in config.raw


# ── Validation errors ─────────────────────────────────────────────────────────

class TestValidation:
    def _pm(self):
        return PolicyManager()

    def test_non_dict_root_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="Policy root"):
            pm.load_dict("not a dict")  # type: ignore

    def test_missing_rule_id_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="'id'"):
            pm.load_dict({"guardrails": [{"severity": "high", "action": "block", "keywords": ["x"]}]})

    def test_duplicate_rule_id_raises(self):
        pm = self._pm()
        rule = {"id": "dup", "severity": "high", "action": "block", "keywords": ["x"]}
        with pytest.raises(PolicyValidationError, match="duplicate"):
            pm.load_dict({"guardrails": [rule, rule]})

    def test_invalid_severity_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="severity"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "extreme", "action": "block", "keywords": ["x"]}
                ]
            })

    def test_invalid_action_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="action"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "high", "action": "delete", "keywords": ["x"]}
                ]
            })

    def test_invalid_regex_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="invalid regex"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "high", "action": "block", "patterns": ["[invalid"]}
                ]
            })

    def test_no_patterns_and_no_keywords_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="at least one"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "high", "action": "block", "patterns": [], "keywords": []}
                ]
            })

    def test_invalid_llm_provider_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="provider"):
            pm.load_dict({
                "guardrails": [_MINIMAL_RULE],
                "llm": {"provider": "unknown"},
            })

    def test_invalid_algorithm_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="algorithm"):
            pm.load_dict({
                "guardrails": [_MINIMAL_RULE],
                "rate_limiting": {"enabled": True, "algorithm": "fixed_window"},
            })

    def test_keywords_not_list_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="keywords"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "high", "action": "block",
                     "keywords": "not a list", "patterns": []}
                ]
            })

    def test_patterns_not_list_raises(self):
        pm = self._pm()
        with pytest.raises(PolicyValidationError, match="patterns"):
            pm.load_dict({
                "guardrails": [
                    {"id": "r1", "severity": "high", "action": "block",
                     "patterns": "not a list"}
                ]
            })


# ── Env-var expansion ─────────────────────────────────────────────────────────

class TestEnvVarExpansion:
    def test_api_key_expanded(self, monkeypatch):
        monkeypatch.setenv("TEST_API_KEY", "sk-test-xyz")
        pm = PolicyManager()
        config = pm.load_dict({
            "guardrails": [_MINIMAL_RULE],
            "llm": {"api_key": "${TEST_API_KEY}"},
        })
        assert config.llm.api_key == "sk-test-xyz"

    def test_missing_env_var_kept_as_literal(self, monkeypatch):
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        pm = PolicyManager()
        config = pm.load_dict({
            "guardrails": [_MINIMAL_RULE],
            "llm": {"api_key": "${NONEXISTENT_VAR}"},
        })
        # Unexpanded: literal ${NONEXISTENT_VAR} is preserved
        assert "${NONEXISTENT_VAR}" in config.llm.api_key

    def test_expansion_in_nested_dict(self, monkeypatch):
        monkeypatch.setenv("AUDIT_DB", "/tmp/mydb.db")
        pm = PolicyManager()
        config = pm.load_dict({
            "guardrails": [_MINIMAL_RULE],
            "audit": {"db_path": "${AUDIT_DB}"},
        })
        assert config.audit.db_path == "/tmp/mydb.db"


# ── File loading (JSON) ───────────────────────────────────────────────────────

class TestPolicyManagerFileLoading:
    def test_load_json_file(self, tmp_path, pm_no_engine):
        policy = {
            "guardrails": [
                {"id": "r1", "name": "Rule 1", "severity": "high",
                 "action": "block", "keywords": ["bad"]}
            ]
        }
        p = tmp_path / "policy.json"
        p.write_text(json.dumps(policy))
        config = pm_no_engine.load(str(p))
        assert len(config.guardrails) == 1
        assert config.guardrails[0].id == "r1"

    def test_file_not_found_raises(self, pm_no_engine):
        with pytest.raises(FileNotFoundError):
            pm_no_engine.load("/nonexistent/path/policy.yaml")

    def test_unsupported_format_raises(self, tmp_path, pm_no_engine):
        p = tmp_path / "policy.toml"
        p.write_text("[guardrails]")
        with pytest.raises(PolicyValidationError, match="Unsupported"):
            pm_no_engine.load(str(p))

    def test_invalid_json_raises(self, tmp_path, pm_no_engine):
        p = tmp_path / "bad.json"
        p.write_text("not valid json }{")
        with pytest.raises(PolicyValidationError, match="parse"):
            pm_no_engine.load(str(p))

    def test_empty_json_file_treated_as_empty(self, tmp_path, pm_no_engine):
        # Empty dict {} is valid
        p = tmp_path / "empty.json"
        p.write_text("{}")
        config = pm_no_engine.load(str(p))
        assert config.guardrails == []

    def test_save_then_reload(self, tmp_path):
        pm = PolicyManager()
        config = pm.load_dict(_MINIMAL_POLICY)
        save_path = str(tmp_path / "out.json")
        pm.save(save_path)
        # Reload
        pm2 = PolicyManager()
        config2 = pm2.load(save_path)
        assert len(config2.guardrails) == len(config.guardrails)
        assert config2.guardrails[0].id == "test_ssn"


# ── build_rate_limiter ────────────────────────────────────────────────────────

class TestBuildRateLimiter:
    def test_returns_none_when_disabled(self):
        pm = PolicyManager()
        pm.load_dict(_MINIMAL_POLICY)  # rate_limiting.enabled=False by default
        assert pm.build_rate_limiter() is None

    def test_returns_token_bucket(self):
        from rate_limiter import TokenBucketLimiter
        pm = PolicyManager()
        pm.load_dict({
            "guardrails": [_MINIMAL_RULE],
            "rate_limiting": {"enabled": True, "algorithm": "token_bucket",
                              "capacity": 10, "refill_rate": 1.0},
        })
        lim = pm.build_rate_limiter()
        assert isinstance(lim, TokenBucketLimiter)
        assert lim.capacity == 10.0

    def test_returns_sliding_window(self):
        from rate_limiter import SlidingWindowLimiter
        pm = PolicyManager()
        pm.load_dict({
            "guardrails": [_MINIMAL_RULE],
            "rate_limiting": {"enabled": True, "algorithm": "sliding_window",
                              "max_requests": 30, "window_s": 60.0},
        })
        lim = pm.build_rate_limiter()
        assert isinstance(lim, SlidingWindowLimiter)
        assert lim.max_requests == 30

    def test_returns_none_without_loaded_config(self):
        pm = PolicyManager()
        # No config loaded yet — returns None (same as disabled rate limiting)
        assert pm.build_rate_limiter() is None


# ── apply_to_engine ───────────────────────────────────────────────────────────

class TestApplyToEngine:
    def test_applies_rules_to_new_engine(self):
        engine = GuardrailEngine()
        pm = PolicyManager()
        config = pm.load_dict(_MINIMAL_POLICY)
        count = pm.apply_to_engine(engine, config)
        assert count == 1
        assert "test_ssn" in engine.rules

    def test_replaces_existing_rules(self):
        engine = GuardrailEngine()
        for r in create_default_guardrails():
            engine.add_rule(r)
        pm = PolicyManager(replace_rules=True)
        config = pm.load_dict(_MINIMAL_POLICY)
        pm.apply_to_engine(engine, config)
        assert list(engine.rules.keys()) == ["test_ssn"]

    def test_rules_functional_in_engine(self):
        from guardrail_framework import Action
        engine = GuardrailEngine()
        pm = PolicyManager(engine=engine)
        pm.load_dict(_MINIMAL_POLICY)
        result = engine.evaluate("My SSN is 123-45-6789")
        assert result.action == Action.BLOCK


# ── load_policy_dict convenience helper ──────────────────────────────────────

class TestLoadPolicyDict:
    def test_loads_without_engine(self):
        config = load_policy_dict(_MINIMAL_POLICY)
        assert len(config.guardrails) == 1

    def test_loads_with_engine(self):
        engine = GuardrailEngine()
        load_policy_dict(_MINIMAL_POLICY, engine=engine)
        assert "test_ssn" in engine.rules


# ── write_example_config ──────────────────────────────────────────────────────

class TestWriteExampleConfig:
    def test_creates_file(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        write_example_config(path)
        assert os.path.exists(path)

    def test_file_is_nonempty(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        write_example_config(path)
        assert len(open(path).read()) > 100

    def test_file_contains_guardrails_key(self, tmp_path):
        path = str(tmp_path / "config.yaml")
        write_example_config(path)
        assert "guardrails:" in open(path).read()
