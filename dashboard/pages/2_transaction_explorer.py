import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from components.sidebar import render_sidebar
from utils.caching import load_processed_data
from components.tables import render_transaction_table

st.set_page_config(page_title="Transaction Explorer | FraudOps", layout="wide")
render_sidebar()

st.title("🔍 Transaction Explorer")
st.markdown("Investigate individual transactions or bulk-filter the live risk stream.")

# Load data
with st.spinner("Loading real transaction stream..."):
    df = load_processed_data()

if df.empty:
    st.stop()

# =========================
# 1. TRANSACTION LOOKUP
# =========================
st.markdown("### 🔎 Transaction Lookup")

search_tx = st.text_input(
    "Search TransactionID",
    placeholder="Enter a numeric TransactionID to investigate instantly...",
)

if search_tx:
    try:
        tx_id = int(search_tx)
        matched_tx = df[df["TransactionID"] == tx_id]

        if matched_tx.empty:
            st.warning(f"Transaction {tx_id} not found in the loaded dataset.")
        else:
            tx = matched_tx.iloc[0]

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Transaction Amount", f"${tx['TransactionAmt']:,.2f}")

            with col2:
                prob = tx.get("prediction_probability", 0.0)
                st.metric("Fraud Probability", f"{prob:.2%}")

            with col3:
                st.metric("Risk Tier", tx.get("Risk_Tier", "N/A"))

            with col4:
                fraud_label = (
                    "🚨 Fraud" if tx.get("isFraud", 0) == 1 else "✅ Legitimate"
                )
                st.metric("Actual Label", fraud_label)

            st.dataframe(matched_tx, use_container_width=True, hide_index=True)

    except ValueError:
        st.error("Please enter a valid numeric TransactionID.")

st.markdown("---")

# =========================
# 2. QUICK METRICS (filtered scope)
# =========================
st.markdown("### 📊 Dataset Snapshot")

snap1, snap2, snap3 = st.columns(3)
with snap1:
    st.metric("Total Records Loaded", f"{len(df):,}")
with snap2:
    fraud_count = int(df["isFraud"].sum()) if "isFraud" in df.columns else 0
    st.metric("Known Fraud Cases", f"{fraud_count:,}")
with snap3:
    crit_count = int((df["Risk_Tier"] == "Critical Risk").sum()) if "Risk_Tier" in df.columns else 0
    st.metric("Critical Risk Flags", f"{crit_count:,}")

st.markdown("---")

# =========================
# 3. FILTERS (Sidebar)
# =========================
with st.sidebar:
    st.markdown("### 🎛️ Filters")

    selected_tier = st.multiselect(
        "Risk Tier",
        options=["Critical Risk", "Suspicious", "Clear"],
        default=["Critical Risk", "Suspicious", "Clear"],  # show all by default
    )

    min_prob = st.slider("Min Fraud Probability", 0.0, 1.0, 0.0, step=0.01)

    amt_max = float(min(df["TransactionAmt"].max(), 10000.0))
    amount_range = st.slider(
        "Transaction Amount Range ($)",
        0.0,
        amt_max,
        (0.0, amt_max),
    )

# Apply filters
filtered_df = df[
    (df["Risk_Tier"].isin(selected_tier))
    & (df["prediction_probability"] >= min_prob)
    & (df["TransactionAmt"] >= amount_range[0])
    & (df["TransactionAmt"] <= amount_range[1])
]

# Sort highest risk first — what fraud analysts care about
filtered_df = filtered_df.sort_values("prediction_probability", ascending=False)

# =========================
# 4. RISK-SORTED CURATED TABLE
# =========================
st.markdown(f"### 📋 Filtered Transactions — {len(filtered_df):,} results")
st.caption("Sorted by fraud probability descending. Table displays top 500 highest-risk rows.")

# Pass sorted filtered_df — tables.py caps at 500 and curates columns
render_transaction_table(filtered_df)
