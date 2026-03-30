"""
Guardrails Framework - Streamlit Monitoring Dashboard
Real-time visualization of guardrail metrics, audit events, and system health.

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List

from audit_logger import AuditLogger
from guardrail_framework import GuardrailEngine, create_default_guardrails, create_default_test_cases
from performance_profiler import profiler


# ── Page configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Guardrails Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Session state ──────────────────────────────────────────────────────────

@st.cache_resource
def get_engine() -> GuardrailEngine:
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    for tc in create_default_test_cases():
        engine.add_test_case(tc)
    return engine


@st.cache_resource
def get_audit_logger() -> AuditLogger:
    return AuditLogger()


# ── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/AGI-Corporation/guardrails-/main/docs/images/guardrails-python-blueprint.png",
        use_container_width=True,
    )
    st.title("🛡️ Guardrails")
    st.caption("AI Safety Monitoring Dashboard")
    st.divider()

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Live Evaluate", "📋 Audit Log", "🧪 Test Runner", "⚙️ Rules Manager"],
        label_visibility="collapsed",
    )

    st.divider()
    refresh = st.slider("Auto-refresh (seconds)", 0, 60, 0, step=5)
    if refresh > 0:
        st.info(f"Page refreshes every {refresh}s")


# ── Helper ─────────────────────────────────────────────────────────────────

def _audit_entries_to_df(entries) -> pd.DataFrame:
    if not entries:
        return pd.DataFrame()
    return pd.DataFrame([
        {
            "id": e.id,
            "timestamp": e.timestamp,
            "action": e.action_taken,
            "severity": e.severity,
            "rules": ", ".join(e.matched_rules) if e.matched_rules else "—",
            "user": e.user_id or "—",
            "text_preview": (e.input_text or "")[:80],
        }
        for e in entries
    ])


# ── Pages ──────────────────────────────────────────────────────────────────

engine = get_engine()
audit = get_audit_logger()

# ─── Overview ──────────────────────────────────────────────────────────────
if page == "📊 Overview":
    st.title("📊 Guardrails Overview")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    stats = audit.get_statistics()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Evaluations", stats.get("total", 0))
    col2.metric("Blocked", stats.get("blocked", 0), delta=None)
    col3.metric("Allowed", stats.get("allowed", 0))
    col4.metric(
        "Block Rate",
        f"{stats.get('block_rate', 0):.1f}%",
    )

    st.divider()

    col_left, col_right = st.columns(2)

    # ── Severity distribution ──────────────────────────────────────────────
    with col_left:
        st.subheader("Events by Severity")
        by_severity = stats.get("by_severity", {})
        if by_severity:
            df_sev = pd.DataFrame(
                list(by_severity.items()), columns=["Severity", "Count"]
            )
            color_map = {
                "critical": "#e74c3c",
                "high": "#e67e22",
                "medium": "#f1c40f",
                "low": "#2ecc71",
                "none": "#95a5a6",
            }
            fig = px.pie(
                df_sev,
                names="Severity",
                values="Count",
                color="Severity",
                color_discrete_map=color_map,
                hole=0.4,
            )
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No audit events yet. Run some evaluations to populate charts.")

    # ── Action distribution ────────────────────────────────────────────────
    with col_right:
        st.subheader("Allow vs Block")
        allowed = stats.get("allowed", 0)
        blocked = stats.get("blocked", 0)
        if allowed + blocked > 0:
            fig2 = go.Figure(go.Bar(
                x=["Allowed", "Blocked"],
                y=[allowed, blocked],
                marker_color=["#2ecc71", "#e74c3c"],
                text=[allowed, blocked],
                textposition="auto",
            ))
            fig2.update_layout(
                yaxis_title="Count",
                margin=dict(t=10, b=0),
                height=300,
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No audit events yet.")

    # ── Active rules ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("Active Guardrail Rules")
    rules_data = [
        {
            "ID": r.id,
            "Name": r.name,
            "Category": r.category.value,
            "Severity": r.severity.value,
            "Action": r.action.value,
            "Enabled": "✅" if r.enabled else "❌",
        }
        for r in engine.rules.values()
    ]
    st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)


# ─── Live Evaluate ────────────────────────────────────────────────────────
elif page == "🔍 Live Evaluate":
    st.title("🔍 Live Text Evaluation")
    st.caption("Evaluate text against the active guardrail rules in real-time.")

    user_text = st.text_area(
        "Enter text to evaluate",
        height=150,
        placeholder="Type or paste text here…",
    )

    col_btn, col_info = st.columns([1, 4])
    evaluate_clicked = col_btn.button("Evaluate", type="primary", use_container_width=True)

    if evaluate_clicked and user_text.strip():
        result = engine.evaluate(user_text)

        audit.log(
            __import__("audit_logger").create_audit_entry(
                input_text=user_text,
                action_taken=result.action,
                matched_rules=result.matched_rules,
                severity=result.severity,
            )
        )

        if result.action == "block":
            st.error(f"🚫 **BLOCKED** — Severity: `{result.severity}`")
        elif result.action == "warn":
            st.warning(f"⚠️ **WARNED** — Severity: `{result.severity}`")
        else:
            st.success("✅ **ALLOWED** — No violations detected")

        if result.matched_rules:
            st.subheader("Matched Rules")
            for rule_id in result.matched_rules:
                rule = engine.rules.get(rule_id)
                if rule:
                    st.write(f"- **{rule.name}** (`{rule_id}`) — {rule.category.value} / {rule.severity.value}")
                else:
                    st.write(f"- `{rule_id}`")

    elif evaluate_clicked:
        st.warning("Please enter some text to evaluate.")

    # ── Content Transformer ────────────────────────────────────────────────
    st.divider()
    st.subheader("🔧 Content Transformer (PII Redaction Preview)")
    transform_text = st.text_area(
        "Enter text to redact PII",
        height=100,
        key="transform",
        placeholder="e.g. My email is alice@example.com and my SSN is 123-45-6789",
    )
    if st.button("Redact PII"):
        from content_transformer import ContentTransformer
        ct = ContentTransformer()
        tr = ct.apply_all_pii(transform_text)
        st.code(tr.transformed, language=None)
        st.caption(f"{tr.changes_made} substitution(s) applied: {tr.transformations_applied}")


# ─── Audit Log ────────────────────────────────────────────────────────────
elif page == "📋 Audit Log":
    st.title("📋 Audit Log")

    col_f1, col_f2, col_f3 = st.columns(3)
    action_filter = col_f1.selectbox("Action", ["all", "allow", "block", "warn"])
    severity_filter = col_f2.selectbox("Severity", ["all", "critical", "high", "medium", "low", "none"])
    limit = col_f3.slider("Max rows", 10, 500, 100, step=10)

    entries = audit.query(
        action=None if action_filter == "all" else action_filter,
        severity=None if severity_filter == "all" else severity_filter,
        limit=limit,
    )

    df = _audit_entries_to_df(entries)
    if df.empty:
        st.info("No audit entries found for the selected filters.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        col_exp, _ = st.columns([1, 4])
        if col_exp.button("Export CSV"):
            csv_path = "/tmp/audit_export.csv"
            audit.export_csv(csv_path)
            with open(csv_path) as f:
                st.download_button(
                    "Download audit_export.csv",
                    f.read(),
                    file_name="audit_export.csv",
                    mime="text/csv",
                )


# ─── Test Runner ──────────────────────────────────────────────────────────
elif page == "🧪 Test Runner":
    st.title("🧪 Guardrail Test Runner")
    st.caption("Run the built-in test suite against the current guardrail rules.")

    if st.button("▶ Run Tests", type="primary"):
        from guardrail_framework import ReportGenerator
        with st.spinner("Running tests…"):
            results = engine.run_tests()
        report = ReportGenerator().generate(results)

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        pct = passed / total * 100 if total else 0

        if passed == total:
            st.success(f"✅ All {total} tests passed!")
        else:
            st.error(f"❌ {total - passed}/{total} tests failed")

        st.progress(pct / 100)

        results_data = [
            {
                "Test ID": r.test_case_id,
                "Status": "✅ PASS" if r.passed else "❌ FAIL",
                "Expected": r.expected,
                "Got": r.got,
                "Matched Rules": ", ".join(r.matched_rules) or "—",
                "Time (ms)": f"{r.execution_time_ms:.2f}",
            }
            for r in results
        ]
        st.dataframe(pd.DataFrame(results_data), use_container_width=True, hide_index=True)

        with st.expander("Full markdown report"):
            st.markdown(report)

    # ── Adversarial quick test ─────────────────────────────────────────────
    st.divider()
    st.subheader("⚔️ Quick Adversarial Test")
    seed_text = st.text_input(
        "Seed phrase (expected to be blocked)",
        value="end my life",
    )
    if st.button("Run Adversarial Mutations"):
        from adversarial_tester import AdversarialTestGenerator
        gen = AdversarialTestGenerator()
        cases = gen.generate_all(seed_text, expected_block=True)
        stats = gen.run_against_engine(cases, engine)

        st.metric("Total mutations", stats["total"])
        col_b, col_e = st.columns(2)
        col_b.metric("Blocked", stats["blocked"])
        col_e.metric("Evaded (gaps)", stats["evaded"])

        if stats["evasions"]:
            st.warning("Evasion attempts that bypassed guardrails:")
            evasion_df = pd.DataFrame(stats["evasions"])
            st.dataframe(evasion_df, use_container_width=True, hide_index=True)
        else:
            st.success("No evasions detected — guardrails are robust to these mutations.")


# ─── Rules Manager ────────────────────────────────────────────────────────
elif page == "⚙️ Rules Manager":
    st.title("⚙️ Rules Manager")

    tab_list, tab_add = st.tabs(["Active Rules", "Add Rule"])

    with tab_list:
        rules_data = [
            {
                "ID": r.id,
                "Name": r.name,
                "Category": r.category.value,
                "Severity": r.severity.value,
                "Action": r.action.value,
                "Patterns": len(r.patterns),
                "Keywords": len(r.keywords),
                "Enabled": r.enabled,
            }
            for r in engine.rules.values()
        ]
        st.dataframe(pd.DataFrame(rules_data), use_container_width=True, hide_index=True)

    with tab_add:
        st.subheader("Add a new guardrail rule")
        from guardrail_framework import GuardrailRule, GuardrailCategory, Severity, Action

        with st.form("add_rule_form"):
            rule_id = st.text_input("Rule ID (unique)", placeholder="my_custom_rule")
            rule_name = st.text_input("Name", placeholder="My Custom Rule")
            rule_category = st.selectbox("Category", [c.value for c in GuardrailCategory])
            rule_severity = st.selectbox("Severity", [s.value for s in Severity])
            rule_action = st.selectbox("Action", [a.value for a in Action])
            rule_keywords = st.text_input(
                "Keywords (comma-separated)", placeholder="bad_word, another_word"
            )
            rule_patterns = st.text_input(
                "Regex patterns (comma-separated)", placeholder=r"\d{3}-\d{2}-\d{4}"
            )
            rule_desc = st.text_area("Description", height=80)
            rule_enabled = st.checkbox("Enabled", value=True)

            submitted = st.form_submit_button("Add Rule", type="primary")

        if submitted:
            if not rule_id or not rule_name:
                st.error("Rule ID and Name are required.")
            elif rule_id in engine.rules:
                st.error(f"Rule ID '{rule_id}' already exists.")
            else:
                keywords = [k.strip() for k in rule_keywords.split(",") if k.strip()]
                patterns = [p.strip() for p in rule_patterns.split(",") if p.strip()]
                new_rule = GuardrailRule(
                    id=rule_id,
                    name=rule_name,
                    category=GuardrailCategory(rule_category),
                    severity=Severity(rule_severity),
                    action=Action(rule_action),
                    keywords=keywords,
                    patterns=patterns,
                    description=rule_desc,
                    enabled=rule_enabled,
                )
                engine.add_rule(new_rule)
                st.success(f"Rule '{rule_id}' added successfully!")
                st.rerun()


# ── Auto-refresh ───────────────────────────────────────────────────────────
if refresh > 0:
    import time
    time.sleep(refresh)
    st.rerun()
