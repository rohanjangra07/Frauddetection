import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from src.config import FEATURE_STATS_PATH

st.set_page_config(page_title="Drift & Stability Monitoring", layout="wide")

st.title("🛡️ Drift & Stability Monitoring")
st.markdown("""
This dashboard monitors how fraud patterns and model behavior shift over time (across training folds). 
Significant deviations between 'Actual Fraud Rate' and 'Mean Prediction Probability' indicate temporal concept drift.
""")

DRIFT_MONITOR_PATH = os.path.join(os.path.dirname(FEATURE_STATS_PATH), 'drift_monitoring.json')

if not os.path.exists(DRIFT_MONITOR_PATH):
    st.warning("Drift monitoring data not found. Please run the training pipeline first.")
    st.stop()

with open(DRIFT_MONITOR_PATH, 'r') as f:
    stats = json.load(f)

# ── Metric Overview ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

avg_fraud_rate = np.mean(stats['fold_fraud_rates'])
avg_pred_prob = np.mean(stats['fold_mean_preds'])
avg_pr_auc = np.mean(stats['fold_pr_aucs'])
stability_score = 1.0 - np.std(stats['fold_pr_aucs']) / avg_pr_auc if avg_pr_auc > 0 else 0

col1.metric("Avg Fraud Rate", f"{avg_fraud_rate:.2%}")
col2.metric("Avg Prediction Prob", f"{avg_pred_prob:.2%}")
col3.metric("Avg CV PR-AUC", f"{avg_pr_auc:.4f}")
col4.metric("Stability Score", f"{stability_score:.2%}", help="1 - CV_Std / CV_Mean. Higher means more consistent performance across time.")

# ── Temporal Drift Visualization ──────────────────────────────────────────────
st.subheader("📊 Temporal Concept Drift (Actual vs Predicted)")

folds = [f"Fold {i+1}" for i in range(len(stats['fold_fraud_rates']))]
drift_df = pd.DataFrame({
    "Fold": folds,
    "Actual Fraud Rate": stats['fold_fraud_rates'],
    "Predicted Probability": stats['fold_mean_preds'],
    "PR-AUC": stats['fold_pr_aucs']
})

fig = go.Figure()
fig.add_trace(go.Scatter(x=drift_df["Fold"], y=drift_df["Actual Fraud Rate"], name="Actual Fraud Rate", line=dict(color='#00CC96', width=3)))
fig.add_trace(go.Scatter(x=drift_df["Fold"], y=drift_df["Predicted Probability"], name="Predicted Probability", line=dict(color='#636EFA', dash='dash')))
fig.update_layout(
    title="Fraud Rate Drift Across Temporal Folds",
    yaxis_title="Rate / Probability",
    template="plotly_dark",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

# ── Performance Stability ─────────────────────────────────────────────────────
st.subheader("📈 Performance Stability (PR-AUC)")
fig_auc = px.bar(drift_df, x="Fold", y="PR-AUC", color="PR-AUC", color_continuous_scale="Viridis", text_auto=".4f")
fig_auc.update_layout(template="plotly_dark", title="PR-AUC per Fold (Temporal Split)")
st.plotly_chart(fig_auc, use_container_width=True)

# ── Feature Quality & Pruning ─────────────────────────────────────────────────
st.subheader("✂️ Feature Quality & Pruning Candidates")
col_left, col_right = st.columns(2)

with col_left:
    st.info("The following features show the lowest average importance across all folds. They are prime candidates for removal to reduce noise.")
    weak_df = pd.DataFrame({"Feature": stats['weak_features']})
    st.table(weak_df.head(10))

with col_right:
    st.info("Performance Variance Analysis")
    std_auc = np.std(stats['fold_pr_aucs'])
    if std_auc > 0.05:
        st.error(f"High Variance Detected (Std: {std_auc:.4f}). Your model might be sensitive to specific time periods.")
    else:
        st.success(f"Low Variance (Std: {std_auc:.4f}). The model generalizes well across time slices.")

# ── Feature Distribution Drift (Mock/Static for now) ──────────────────────────
st.subheader("🔍 Top Drifting Features (Input Stats)")
# In a real production system, we'd compare training stats with inference stats here.
# For now, we show the training std/mean ratio as a proxy for feature instability.

feat_instability = {}
for feat, mean in stats['means'].items():
    std = stats['stds'].get(feat, 0)
    if mean != 0:
        feat_instability[feat] = std / abs(mean)
    else:
        feat_instability[feat] = 0

instability_df = pd.DataFrame({
    "Feature": list(feat_instability.keys()),
    "Instability (Std/Mean)": list(feat_instability.values())
}).sort_values("Instability (Std/Mean)", ascending=False).head(15)

fig_drift = px.bar(instability_df, x="Instability (Std/Mean)", y="Feature", orientation='h', color="Instability (Std/Mean)", color_continuous_scale="Reds")
fig_drift.update_layout(template="plotly_dark", title="Feature Coefficient of Variation (Training Period)")
st.plotly_chart(fig_drift, use_container_width=True)
