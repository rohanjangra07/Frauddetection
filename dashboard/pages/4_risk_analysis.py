import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import plotly.express as px
from components.sidebar import render_sidebar
from utils.caching import load_processed_data
from utils.plot_theme import apply_plotly_theme, COLORS
from src.risk_segmentation import RiskSegmenter

st.set_page_config(page_title="Risk Analysis | FraudOps", layout="wide")
render_sidebar()

st.title("🛡️ Risk Analysis & Fraud Patterns")
st.markdown("Deep dive into the characteristics of the different risk tiers.")

# Load data
with st.spinner("Loading real transaction stream..."):
    df = load_processed_data()

if df.empty:
    st.stop()

# Initialize Segmenter
segmenter = RiskSegmenter(df, proba_col='prediction_probability')

st.markdown("### 1. Risk Tier Performance Summary")
summary_df = segmenter.get_tier_summaries()

# Format the summary df for display
display_summary = summary_df.copy()
display_summary['Total_Value_At_Risk'] = display_summary['Total_Value_At_Risk'].apply(lambda x: f"${x:,.2f}")
display_summary['Avg_TransactionAmt'] = display_summary['Avg_TransactionAmt'].apply(lambda x: f"${x:,.2f}")
display_summary['%_of_Total_Transactions'] = display_summary['%_of_Total_Transactions'].apply(lambda x: f"{x:.1f}%")

st.dataframe(display_summary, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("### 2. Fraud Pattern Insights")
# Extract insights
insights = segmenter.extract_top_fraud_patterns()
st.info(insights)

st.markdown("---")
st.markdown("### 3. Visual Analytics")

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(segmenter.plot_tier_distribution(return_fig=True), use_container_width=True)
    
with col2:
    if 'HourOfDay' in df.columns:
        st.plotly_chart(segmenter.plot_hourly_activity(return_fig=True), use_container_width=True)

st.markdown("#### Probability Distribution")
st.plotly_chart(segmenter.plot_prob_distribution(return_fig=True), use_container_width=True)

st.markdown("#### Feature Correlation (Amount vs Risk)")
# Custom Plotly chart for Amount vs Risk
if 'AmtToMeanRatio' in df.columns:
    fig = px.box(
        df,
        x='Risk_Tier',
        y='AmtToMeanRatio',
        color='Risk_Tier',
        category_orders={"Risk_Tier": ["Clear", "Suspicious", "Critical Risk"]},
        color_discrete_map={"Clear": COLORS["clear"], "Suspicious": COLORS["suspicious"], "Critical Risk": COLORS["fraud"]},
        labels={'AmtToMeanRatio': 'Amount / Card Mean', 'Risk_Tier': 'Risk Tier'},
    )
    fig.update_layout(yaxis=dict(range=[0, df['AmtToMeanRatio'].quantile(0.95)]))
    apply_plotly_theme(fig, title="Amount-to-Mean Ratio by Risk Tier")
    st.plotly_chart(fig, use_container_width=True)
