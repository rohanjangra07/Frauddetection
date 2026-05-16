import sys
from pathlib import Path

# Add project root to path for src imports
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from components.sidebar import render_sidebar

# Must be the first Streamlit command
st.set_page_config(
    page_title="FraudOps | Enterprise Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

render_sidebar()

st.title("🛡️ Enterprise Fraud Operations Center")

st.markdown("""
### Welcome to the AI-Powered Fraud Detection System

This dashboard is designed for Fraud Analysts and Risk Managers to monitor real-time transaction streams, investigate suspicious activity, and understand the AI's decision-making process.

#### 🧭 Navigation Guide (See Sidebar)
1. **Overview**: Executive summary, KPIs, and high-level macro trends.
2. **Transaction Explorer**: Deep-dive into individual transactions, filter by risk tiers, and search by ID.
3. **SHAP Explainer**: Unbox the AI. See exactly *why* a specific transaction was flagged using Explainable AI.
4. **Risk Analysis**: Analyze the defining characteristics of Critical Risk fraud rings.

---
*Built with Streamlit, LightGBM, and SHAP. Optimized for the IEEE-CIS Fraud Detection dataset.*
""")

st.info("👈 Please select a module from the sidebar to begin.")
