import streamlit as st
import pandas as pd
import plotly.express as px
from audit_logger import AuditLogger
from performance_profiler import PerformanceProfiler
import datetime

st.set_page_config(page_title="Guardrails Dashboard", layout="wide")
st.title("🛡️ Guardrails Framework Dashboard")

logger = AuditLogger()
profiler = PerformanceProfiler()

page = st.sidebar.selectbox("View", ["Overview", "Audit Log", "Performance"])

if page == "Overview":
    st.header("Overview")
    st.write("Welcome to the Guardrails Framework dashboard.")
    logs = logger.get_logs(limit=1000)
    if logs:
        df = pd.DataFrame(logs)
        total = len(df)
        blocked = int((df["action_taken"] == "block").sum()) if "action_taken" in df.columns else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Evaluations", total)
        col2.metric("Blocked", blocked)
        col3.metric("Block Rate", f"{blocked / total * 100:.1f}%" if total else "0%")
    else:
        st.info("No audit log entries yet. Evaluate some text to populate the dashboard.")
elif page == "Audit Log":
    st.header("Audit Log")
    logs = logger.get_logs()
    if logs:
        df = pd.DataFrame(logs)
        st.dataframe(df)
    else:
        st.info("No audit log entries found.")
elif page == "Performance":
    st.header("Performance")
    stats = profiler.get_stats()
    if stats:
        st.json(stats)
    else:
        st.info("No performance data recorded yet.")
