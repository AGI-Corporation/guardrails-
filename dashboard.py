"""
🛡️ Guardrails Framework Dashboard
Interactive Streamlit dashboard for monitoring guardrail events, adversarial
test results, and system performance.
"""
import datetime
import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from audit_logger import AuditLogger
from guardrail_framework import GuardrailEngine, create_default_guardrails
from performance_profiler import PerformanceProfiler

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Guardrails Dashboard",
    page_icon="🛡️",
    layout="wide",
)
st.title("🛡️ Guardrails Framework Dashboard")

# ── Shared state ──────────────────────────────────────────────────────────────
@st.cache_resource
def _get_engine() -> GuardrailEngine:
    engine = GuardrailEngine()
    for rule in create_default_guardrails():
        engine.add_rule(rule)
    return engine


@st.cache_resource
def _get_logger() -> AuditLogger:
    return AuditLogger()


@st.cache_resource
def _get_profiler() -> PerformanceProfiler:
    return PerformanceProfiler()


engine = _get_engine()
logger = _get_logger()
profiler = _get_profiler()

# ── Sidebar navigation ────────────────────────────────────────────────────────
page = st.sidebar.selectbox(
    "View",
    ["Overview", "Live Evaluator", "Audit Log", "Adversarial Testing", "Performance"],
)

# ── Helper: load audit log as DataFrame ──────────────────────────────────────

def _audit_df() -> pd.DataFrame:
    logs = logger.get_logs(limit=5000)
    if not logs:
        return pd.DataFrame()
    rows = [
        {
            "timestamp": e.timestamp,
            "action": e.action_taken,
            "severity": e.severity,
            "matched_rules": ", ".join(e.matched_rules),
            "input_text": e.input_text[:100],
            "user_id": e.user_id or "",
        }
        for e in logs
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


# ═════════════════════════════════════════════════════════════════════════════
#  OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
if page == "Overview":
    st.header("Overview")

    df = _audit_df()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Rules", len(engine.rules))
    if not df.empty:
        col2.metric("Total Events", len(df))
        col3.metric("Blocked", int((df["action"] == "block").sum()))
        col4.metric(
            "Block Rate",
            f"{100 * (df['action'] == 'block').mean():.1f}%",
        )
    else:
        col2.metric("Total Events", 0)
        col3.metric("Blocked", 0)
        col4.metric("Block Rate", "—")

    st.markdown("---")

    if not df.empty:
        st.subheader("Action Distribution")
        action_counts = df["action"].value_counts().reset_index()
        action_counts.columns = ["action", "count"]
        fig = px.pie(action_counts, names="action", values="count", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Events Over Time")
        df_ts = df.set_index("timestamp").resample("1min").size().reset_index()
        df_ts.columns = ["timestamp", "events"]
        fig2 = px.bar(df_ts, x="timestamp", y="events", labels={"events": "Events / min"})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No audit events yet. Evaluate some text to see data here.")

    st.markdown("---")
    st.subheader("Loaded Guardrail Rules")
    rules_data = [
        {
            "ID": r.id,
            "Name": r.name,
            "Severity": r.severity.value,
            "Action": r.action.value,
            "Patterns": len(r.patterns),
            "Keywords": len(r.keywords),
        }
        for r in engine.rules.values()
    ]
    st.dataframe(pd.DataFrame(rules_data), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
#  LIVE EVALUATOR
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Live Evaluator":
    st.header("Live Evaluator")
    st.write("Type any text below to evaluate it against the active guardrail rules.")

    text_input = st.text_area("Input text", height=120, placeholder="Enter text to evaluate…")

    if st.button("Evaluate", type="primary") and text_input.strip():
        with profiler.time("dashboard", "evaluate"):
            result = engine.evaluate(text_input)

        action_str = result.action.value
        severity_str = result.severity.value

        if action_str == "block":
            st.error(f"🚫 **BLOCKED** — Severity: {severity_str.upper()}")
        elif action_str == "warn":
            st.warning(f"⚠️ **WARNING** — Severity: {severity_str.upper()}")
        else:
            st.success("✅ **ALLOWED**")

        col_l, col_r = st.columns(2)
        col_l.metric("Action", action_str.upper())
        col_r.metric("Matched Rules", len(result.matched_rules))

        if result.matched_rules:
            st.subheader("Matched Rules")
            for rule_id in result.matched_rules:
                rule = engine.rules.get(rule_id)
                if rule:
                    st.markdown(
                        f"- **{rule.name}** (`{rule_id}`) — "
                        f"severity: {rule.severity.value}, action: {rule.action.value}"
                    )

        # Log to audit
        logger.log(
            input_text=text_input,
            action_taken=action_str,
            matched_rules=result.matched_rules,
            severity=severity_str,
        )


# ═════════════════════════════════════════════════════════════════════════════
#  AUDIT LOG
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Audit Log":
    st.header("Audit Log")

    df = _audit_df()
    if df.empty:
        st.info("No audit events recorded yet.")
    else:
        # Filters
        col_a, col_b = st.columns(2)
        action_filter = col_a.multiselect(
            "Filter by action", options=df["action"].unique().tolist(),
            default=df["action"].unique().tolist(),
        )
        search_term = col_b.text_input("Search input text", "")

        filtered = df[df["action"].isin(action_filter)]
        if search_term:
            filtered = filtered[
                filtered["input_text"].str.contains(search_term, case=False, na=False)
            ]

        st.write(f"Showing **{len(filtered)}** of {len(df)} events")
        st.dataframe(filtered, use_container_width=True)

        # Download
        csv_bytes = filtered.to_csv(index=False).encode()
        st.download_button("⬇️ Download CSV", csv_bytes, "audit_export.csv", "text/csv")


# ═════════════════════════════════════════════════════════════════════════════
#  ADVERSARIAL TESTING
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Adversarial Testing":
    st.header("🔴 Adversarial / Penetration Testing")

    st.write(
        "Run the full adversarial test suite against the active guardrail engine. "
        "This exercises encoding bypasses, jailbreaks, prompt injection, PII leaks, "
        "boundary conditions, and more."
    )

    include_plugins = st.checkbox("Include plugin-level checks", value=True)

    if st.button("▶ Run Penetration Test Suite", type="primary"):
        with st.spinner("Running full adversarial suite…"):
            from penetration_test_agent import PenTestSession, PenetrationTestAgent

            session = PenTestSession(
                name=f"dashboard-{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')}",
                include_plugins=include_plugins,
                audit_db_path="pentest_audit.db",
                feedback_db_path="pentest_feedback.db",
            )
            agent = PenetrationTestAgent(engine=engine)
            report = agent.run(session)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Attacks", report.total_attacks)
        col2.metric("Blocked", report.total_blocked)
        col3.metric("Bypassed ⚠️", report.total_bypassed)
        col4.metric("Block Rate", f"{report.overall_block_rate * 100:.1f}%")

        st.markdown("---")

        # Category bar chart
        st.subheader("Block Rate by Attack Category")
        cat_data = [
            {"Category": s.category, "Block Rate (%)": round(s.block_rate * 100, 1)}
            for s in report.category_summaries
        ]
        cat_df = pd.DataFrame(cat_data).sort_values("Block Rate (%)")
        fig = px.bar(
            cat_df, x="Block Rate (%)", y="Category", orientation="h",
            color="Block Rate (%)", color_continuous_scale="RdYlGn",
            range_color=[0, 100],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed category table
        st.subheader("Category Details")
        detail_data = [
            {
                "Category": s.category,
                "Total": s.total,
                "Blocked": s.blocked,
                "Bypassed": s.bypassed,
                "Block Rate": f"{s.block_rate * 100:.1f}%",
                "Avg ms": f"{s.avg_duration_ms:.2f}",
                "Bypassed Mutations": ", ".join(s.bypassed_mutations[:5]),
            }
            for s in report.category_summaries
        ]
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True)

        # Recommendations
        if report.recommendations:
            st.subheader("📋 Recommendations")
            for rec in report.recommendations:
                st.warning(rec)

        # Export
        st.subheader("Export Report")
        col_j, col_c = st.columns(2)
        json_bytes = report.to_json().encode()
        col_j.download_button("⬇️ JSON Report", json_bytes, "pentest_report.json", "application/json")

        import io
        csv_df = pd.DataFrame(
            [
                {
                    "category": a.category, "mutation": a.mutation,
                    "blocked": a.blocked, "plugin_action": a.plugin_action,
                    "duration_ms": a.duration_ms,
                }
                for a in report.attacks
            ]
        )
        col_c.download_button(
            "⬇️ CSV Report",
            csv_df.to_csv(index=False).encode(),
            "pentest_attacks.csv",
            "text/csv",
        )


# ═════════════════════════════════════════════════════════════════════════════
#  PERFORMANCE
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Performance":
    st.header("Performance Profiler")

    stats = profiler.get_stats()
    if not stats:
        st.info("No profiler data yet. Run some evaluations first.")
    else:
        st.subheader("Component Statistics")
        perf_rows = [
            {
                "Component": k,
                "Calls": v["total_calls"],
                "Avg ms": v["avg_ms"],
                "P95 ms": v["p95_ms"],
                "Max ms": v["max_ms"],
                "Success %": v["success_rate_pct"],
            }
            for k, v in stats.items()
        ]
        st.dataframe(pd.DataFrame(perf_rows), use_container_width=True)

        st.subheader("Average Latency by Component")
        fig = px.bar(
            pd.DataFrame(perf_rows),
            x="Component",
            y="Avg ms",
            color="Avg ms",
            color_continuous_scale="Blues",
        )
        st.plotly_chart(fig, use_container_width=True)

        bottlenecks = profiler.get_bottlenecks(5)
        if bottlenecks:
            st.subheader("Top Bottlenecks")
            for b in bottlenecks:
                st.markdown(
                    f"- **{b['component']}** — avg {b['avg_ms']}ms, p95 {b['p95_ms']}ms"
                )

