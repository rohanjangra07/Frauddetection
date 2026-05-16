import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("🛡️ Fraud Detection Operations Dashboard")

# Load model
@st.cache_resource
def load_model():
    if os.path.exists('model.pkl'):
        return joblib.load('model.pkl')
    return None

model = load_model()

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Overview", "Transaction Explorer", "Risk Analysis"])

if page == "Overview":
    st.header("System Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Detection Rate", "98.2%")
    col2.metric("False Positive Rate", "1.5%")
    col3.metric("Loss Prevented", "$1.2M")
    
    st.subheader("Fraud Patterns by Hour")
    # Mock data for demonstration
    hours = list(range(24))
    rates = [0.05, 0.06, 0.08, 0.07, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03, 0.03, 0.04, 0.04, 0.05, 0.06, 0.07, 0.06, 0.05, 0.04, 0.04]
    fig = px.line(x=hours, y=rates, labels={'x':'Hour', 'y':'Fraud Probability'})
    st.plotly_chart(fig, use_container_width=True)

elif page == "Transaction Explorer":
    st.header("Transaction Explorer")
    st.write("Search and filter transactions by ID or Risk Tier.")
    # Mock table
    data = pd.DataFrame({
        'TransactionID': [3663549, 3663550, 3663551, 3663552],
        'Amount': [50.0, 120.5, 15.0, 450.0],
        'RiskScore': [0.12, 0.85, 0.05, 0.92],
        'Status': ['Clear', 'Critical', 'Clear', 'Critical']
    })
    st.dataframe(data, use_container_width=True)

elif page == "Risk Analysis":
    st.header("Risk Segmentation")
    st.write("Distribution of transaction risk across the current batch.")
    fig = px.pie(names=['Clear', 'Suspicious', 'Critical'], values=[95, 4, 1], color_discrete_sequence=['#00CC96', '#FECB52', '#EF553B'])
    st.plotly_chart(fig, use_container_width=True)
