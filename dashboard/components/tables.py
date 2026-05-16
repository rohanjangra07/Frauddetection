import streamlit as st
import pandas as pd
from utils.formatting import apply_risk_styling

# Hard cap: styling hundreds of thousands of cells destroys Streamlit performance.
# Fraud analysts inspect top-risk rows, not entire production datasets.
MAX_DISPLAY_ROWS = 500


def render_transaction_table(df: pd.DataFrame):
    """
    Renders a styled transaction table capped at MAX_DISPLAY_ROWS rows.
    Input df should already be sorted by prediction_probability descending
    so the cap retains the highest-risk rows.
    """
    total_rows = len(df)

    # Cap BEFORE building any Styler object
    display_df = df.head(MAX_DISPLAY_ROWS).copy()

    if display_df.empty:
        st.info("No transactions match the current filters.")
        return

    # Curate columns — gracefully skip missing ones
    preferred_cols = [
        "TransactionID", "TransactionAmt", "prediction_probability",
        "Risk_Tier", "DeviceRisk", "HourOfDay", "isFraud",
    ]
    display_cols = [c for c in preferred_cols if c in display_df.columns]
    display_df = display_df[display_cols]

    # Format probability as percentage string for readability
    if "prediction_probability" in display_df.columns:
        display_df["prediction_probability"] = (
            display_df["prediction_probability"].apply(lambda x: f"{x:.2%}")
        )

    # Apply risk-tier colour styling ONLY to the capped subset
    styled_df = display_df.style.map(apply_risk_styling, subset=["Risk_Tier"])

    # Row count caption — transparent UX
    if total_rows > MAX_DISPLAY_ROWS:
        st.caption(
            f"⚠️ Displaying top **{MAX_DISPLAY_ROWS:,}** highest-risk rows "
            f"out of **{total_rows:,}** matching transactions. "
            f"Refine filters to narrow results."
        )
    else:
        st.caption(f"Showing all **{total_rows:,}** matching transactions.")

    st.dataframe(
        styled_df,
        use_container_width=True,
        height=420,
        hide_index=True,
    )
