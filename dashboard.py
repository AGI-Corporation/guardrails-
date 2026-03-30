import streamlit as st
import pandas as pd
import plotly.express as px
from audit_logger import AuditLogger
from performance_profiler import PerformanceProfiler
import datetime
from dataclasses import asdict

st.set_page_config(page_title="Guardrails Dashboard", layout="wide")
st.title("🛡️ Guardrails Framework Dashboard")

logger = AuditLogger()
profiler = PerformanceProfiler()

page = st.sidebar.selectbox("View", ["Overview", "Audit Log", "Performance"])

if page == "Overview":
    st.header("Overview")
    st.write("Welcome to the Guardrails Framework dashboard.")
elif page == "Audit Log":
    st.header("Audit Log")
    logs = logger.get_logs()
    if logs:
        df = pd.DataFrame([asdict(entry) for entry in logs])
        st.dataframe(df)
    else:
        st.info("No audit log entries found.")
elif page == "Performance":
    st.header("Performance")
    st.json(profiler.get_stats())
