import streamlit as st

def render_sidebar():
    """Renders the common sidebar elements across all pages."""
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=50) # Generic security icon
        st.title("FraudOps Center")
        st.markdown("---")
        
        st.markdown("### System Status")
        st.success("🟢 Real-Time Inference: Online")
        st.success("🟢 Model Version: LGBM-v1.2")
        st.success("🟢 SHAP Engine: Active")
        
        st.markdown("---")
        st.markdown("### Quick Links")
        st.markdown("[Documentation](#)")
        st.markdown("[Retrain Model](#)")
        st.markdown("[Alert Configuration](#)")
        
        st.markdown("---")
        st.caption("Powered by Advanced Agentic ML")
