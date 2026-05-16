import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from components.sidebar import render_sidebar
from utils.caching import load_processed_data, load_model
from src.explainability import SHAPExplainer

st.set_page_config(page_title="SHAP Explainer | FraudOps", layout="wide")
render_sidebar()

st.title("🧠 AI Explainability (SHAP)")
st.markdown("Unbox the AI: Understand exactly why a transaction was flagged.")

# Load Model and Data
with st.spinner("Loading SHAP Engine and real data..."):
    df = load_processed_data()
    model = load_model()

if df.empty or model is None:
    st.stop()

# Drop non-feature columns for SHAP background
# CRITICAL: reset_index(drop=True) so positional index == row position
# after any sampling, filtering, or train/test splits that may have occurred
df = df.reset_index(drop=True)
features_only = df.drop(
    columns=['TransactionID', 'prediction_probability', 'Risk_Tier', 'isFraud'],
    errors='ignore'
).reset_index(drop=True)

# Initialize Explainer (cached to avoid recomputation on widget interaction)
@st.cache_resource
def get_explainer():
    # Use a small sample for the explainer background to save memory
    return SHAPExplainer(model, features_only, max_samples=500)

if model:
    try:
        explainer = get_explainer()
    except Exception as e:
        st.error(f"Failed to initialize SHAP explainer. Ensure the model supports TreeExplainer. Error: {e}")
        st.stop()

st.markdown("### 1. Global Feature Importance")
with st.expander("View Global SHAP Summary"):
    if st.button("Generate Global Summary Plot"):
        with st.spinner("Generating plot..."):
            fig = explainer.plot_summary(return_fig=True)
            st.pyplot(fig)

st.markdown("---")
st.markdown("### 2. Local Transaction Investigation")

# Build high-risk candidate list from real predictions (highest probability first)
high_risk_ids = (
    df.sort_values("prediction_probability", ascending=False)
      .head(50)["TransactionID"]
      .astype(str)
      .tolist()
)
default_tx = high_risk_ids[0] if high_risk_ids else ""

col_input, col_pick = st.columns([2, 1])

with col_input:
    tx_id = st.text_input(
        "Enter TransactionID",
        value=default_tx,
        help="Defaults to the highest-risk transaction in the dataset.",
    )

with col_pick:
    quick_pick = st.selectbox(
        "Quick High-Risk Picks",
        options=high_risk_ids,
        help="Top 50 transactions sorted by fraud probability descending.",
    )

# Selectbox overrides text input when changed
if quick_pick and quick_pick != tx_id:
    tx_id = quick_pick

if tx_id:
    try:
        query_id = int(tx_id)
        matching_rows = df[df['TransactionID'] == query_id]

        if matching_rows.empty:
            st.warning("Transaction ID not found in the loaded dataset.")
            st.stop()

        # Use positional location (.iloc) with reset index so SHAP and df are aligned
        positional_idx = matching_rows.index[0]  # safe after reset_index(drop=True)
        tx_data = df.iloc[positional_idx]

        # Extract the actual feature vector — no index alignment dependency
        feature_row = features_only.iloc[[positional_idx]]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("#### Transaction Summary")
            st.write(f"**Amount**: ${tx_data['TransactionAmt']:.2f}")
            st.write(f"**Probability**: {tx_data['prediction_probability']:.2%}")

            risk_color = (
                "red" if tx_data['Risk_Tier'] == 'Critical Risk'
                else "orange" if tx_data['Risk_Tier'] == 'Suspicious'
                else "green"
            )
            st.markdown(
                f"**Risk Tier**: <span style='color:{risk_color}; font-weight:bold;'>"
                f"{tx_data['Risk_Tier']}</span>",
                unsafe_allow_html=True,
            )

            st.markdown("#### Plain English Report")
            with st.spinner("Generating report..."):
                report = explainer.generate_explanation(transaction_data=feature_row, top_n=3)
                st.markdown(report)

        with col2:
            st.markdown("#### SHAP Waterfall Plot")
            with st.spinner("Generating waterfall..."):
                fig = explainer.plot_waterfall(transaction_data=feature_row, return_fig=True)
                st.pyplot(fig)

    except ValueError:
        st.error("Please enter a valid numeric TransactionID.")

