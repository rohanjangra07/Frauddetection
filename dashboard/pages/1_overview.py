import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from components.sidebar import render_sidebar
from utils.caching import load_processed_data
from components.metrics import render_model_performance_summary
from components.charts import plot_risk_tier_donut, plot_fraud_by_hour

st.set_page_config(page_title="Overview | FraudOps", layout="wide")
render_sidebar()

st.title("📊 Executive Overview")
st.markdown("Macro-level view of system performance and current transaction risk.")

# Load data
with st.spinner("Loading real transaction stream..."):
    df = load_processed_data()

if df.empty:
    st.stop()

# Defensive sanitization: guard against any inf values from engineered features
df = df.replace([np.inf, -np.inf], np.nan)

# =========================
# KPI SECTION
# =========================

total_transactions = len(df)

total_fraud = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0

fraud_prevalence = (total_fraud / total_transactions * 100) if total_transactions > 0 else 0

avg_fraud_amount = (
    df[df["isFraud"] == 1]["TransactionAmt"].mean()
    if "isFraud" in df.columns and total_fraud > 0
    else 0.0
)

high_risk_rate = (
    (df["Risk_Tier"] == "Critical Risk").mean() * 100
    if "Risk_Tier" in df.columns
    else 0.0
)

kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.metric("Total Transactions", f"{total_transactions:,}")

with kpi2:
    st.metric("Total Fraud Cases", f"{total_fraud:,}")

with kpi3:
    st.metric("Fraud Prevalence", f"{fraud_prevalence:.2f}%",
              help="Fraud rows ÷ total rows. Reflects class imbalance in the dataset.")

with kpi4:
    st.metric("Avg Fraud Amount", f"${avg_fraud_amount:,.2f}")

with kpi5:
    st.metric("High-Risk Flag Rate", f"{high_risk_rate:.2f}%",
              help="% of transactions classified as Critical Risk (probability ≥ 0.75).")

st.markdown("---")


# Charts layout
col1, col2 = st.columns([1, 1.5])

with col1:
    st.plotly_chart(plot_risk_tier_donut(df), use_container_width=True)
    render_model_performance_summary()

with col2:
    st.plotly_chart(plot_fraud_by_hour(df), use_container_width=True)

st.markdown("### 📝 Operations Note")
st.info("The threshold for 'Critical Risk' is currently set to 0.75. This threshold is optimized to prioritize Precision to avoid overwhelming the manual review team with false positives.")
