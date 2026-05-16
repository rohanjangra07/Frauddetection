import streamlit as st
from utils.formatting import format_currency, format_percentage

def render_kpi_metrics(df):
    """Renders the top-level KPI metrics for the dashboard."""
    total_txns = len(df)
    fraud_txns = len(df[df['isFraud'] == 1])
    fraud_rate = fraud_txns / total_txns if total_txns > 0 else 0
    avg_fraud_amt = df[df['isFraud'] == 1]['TransactionAmt'].mean() if fraud_txns > 0 else 0
    total_fraud_value = df[df['isFraud'] == 1]['TransactionAmt'].sum()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="Total Transactions Reviewed", value=f"{total_txns:,}")
    with col2:
        st.metric(label="Detected Fraud Cases", value=f"{fraud_txns:,}")
    with col3:
        st.metric(label="Current Fraud Rate", value=format_percentage(fraud_rate))
    with col4:
        st.metric(label="Value at Risk (Detected)", value=format_currency(total_fraud_value))
        
def render_model_performance_summary():
    """Renders a static summary of the model's offline performance."""
    st.markdown("""
    **Model Performance (Offline Validation)**
    - **Primary Metric (PR-AUC)**: 0.824
    - **Recall @ 5% FPR**: 91.2%
    - **Model Type**: LightGBM (Tuned via Optuna)
    """)
