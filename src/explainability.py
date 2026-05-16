import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from typing import Dict, Any, List, Tuple
from src.utils import logger
from src.config import OUTPUT_DIR

try:
    # Apply dark Matplotlib theme for Streamlit dark-mode compatibility.
    # Imported lazily so this module works outside the dashboard context too.
    import sys, os
    _dash_utils = os.path.join(os.path.dirname(__file__), '..', 'dashboard', 'utils')
    if os.path.isdir(_dash_utils) and _dash_utils not in sys.path:
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from dashboard.utils.plot_theme import apply_matplotlib_dark_theme
    apply_matplotlib_dark_theme()
except Exception:
    pass  # Outside dashboard environment – use default Matplotlib style

# Initialize JS for notebook environments safely
try:
    shap.initjs()
except Exception:
    pass

class SHAPExplainer:
    """
    Production-optimized Explainable AI module using SHAP.
    Designed for memory-efficiency and dashboard compatibility.
    """
    
    def __init__(self, model: Any, background_data: pd.DataFrame, max_samples: int = 1000):
        """
        Initializes the TreeExplainer. Uses a sampled background dataset 
        to prevent RAM crashes with high-cardinality/large-scale data.
        """
        logger.info(f"Initializing SHAP TreeExplainer with max {max_samples} background samples...")
        self.model = model

        # Sample and CRITICAL: reset_index so positional index == row position.
        # Without this, iloc[idx] and shap_values[idx] refer to different rows
        # if background_data came from filtering, sorting, or a train/test split.
        sample_n = min(max_samples, len(background_data))
        self.background_data = (
            background_data
            .sample(n=sample_n, random_state=42)
            .reset_index(drop=True)
        )
            
        self.explainer = shap.TreeExplainer(model)
        # Compute SHAP values once and cache them
        logger.info("Computing global SHAP values. This may take a moment...")
        self.shap_values = self.explainer(self.background_data)
        logger.info("SHAP computation complete.")

    def plot_summary(self, max_display: int = 15, return_fig: bool = False):
        """
        Global SHAP Summary Plot.
        Returns the figure object for Streamlit rendering if requested.
        """
        try:
            from dashboard.utils.plot_theme import apply_matplotlib_dark_theme
            apply_matplotlib_dark_theme()
        except Exception:
            pass
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.background_data, max_display=max_display, show=False)
        plt.title(f"Global Feature Importance (Top {max_display} SHAP)", fontsize=14, color='white')
        plt.tight_layout()
        
        if return_fig:
            return fig
        plt.show()
        plt.close(fig)

    def plot_waterfall(self, transaction_data: 'pd.DataFrame', return_fig: bool = False):
        """
        Local SHAP Waterfall Plot for a specific transaction.
        Accepts a single-row DataFrame of features — no index alignment required.
        """
        try:
            from dashboard.utils.plot_theme import apply_matplotlib_dark_theme
            apply_matplotlib_dark_theme()
        except Exception:
            pass
        # Compute SHAP values live for the supplied feature vector
        shap_vals = self.explainer(transaction_data)
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        plt.title("Local Transaction Explanation", fontsize=14, color='white')
        plt.tight_layout()
        
        if return_fig:
            return fig
        plt.show()
        plt.close(fig)

    def plot_dependence(self, feature_name: str, interaction_feature: str = "auto", return_fig: bool = False):
        """
        SHAP Dependence Plot to visualize feature interactions.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            feature_name, 
            self.shap_values.values, 
            self.background_data, 
            interaction_index=interaction_feature,
            ax=ax,
            show=False
        )
        plt.title(f"SHAP Dependence: {feature_name}", fontsize=14)
        plt.tight_layout()
        
        if return_fig:
            return fig
        plt.show()
        plt.close(fig)

    def compare_feature_importance(self, return_fig: bool = False):
        """
        Compares Model-Native Importance (Split/Gain) vs SHAP Importance.
        """
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not expose feature_importances_.")
            return None
            
        # Native Importance
        native_imp = pd.Series(self.model.feature_importances_, index=self.background_data.columns)
        native_imp = native_imp.sort_values(ascending=False).head(10)
        
        # SHAP Importance (Mean absolute SHAP value)
        shap_imp = pd.Series(np.abs(self.shap_values.values).mean(axis=0), index=self.background_data.columns)
        shap_imp = shap_imp.sort_values(ascending=False).head(10)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        import seaborn as sns
        sns.barplot(x=native_imp.values, y=native_imp.index, ax=axes[0], palette='Blues_r')
        axes[0].set_title("Model-Native Importance (Split/Gain)")
        axes[0].set_xlabel("Importance Score")
        
        sns.barplot(x=shap_imp.values, y=shap_imp.index, ax=axes[1], palette='Oranges_r')
        axes[1].set_title("SHAP Global Importance (Absolute Impact)")
        axes[1].set_xlabel("Mean |SHAP Value|")
        
        plt.tight_layout()
        if return_fig:
            return fig
        plt.show()
        plt.close(fig)

    def generate_explanation(self, transaction_data: 'pd.DataFrame', top_n: int = 4) -> str:
        """
        Generates a detailed, business-friendly explanation for a specific transaction.
        Accepts a single-row DataFrame of features — no index alignment required.
        Handles LightGBM/XGBoost log-odds conversion automatically.
        """
        # Compute SHAP values live for the supplied feature vector
        shap_explanation = self.explainer(transaction_data)
        instance = transaction_data.iloc[0]
        shap_vals = shap_explanation.values[0]
        base_value = shap_explanation.base_values[0]
        
        # Log-odds to Probability conversion
        log_odds = base_value + np.sum(shap_vals)
        probability = 1 / (1 + np.exp(-log_odds))
        
        risk_tier = "🔴 Critical Risk" if probability >= 0.75 else "🟡 Suspicious" if probability >= 0.40 else "🟢 Legitimate"
        
        # Extract impacts
        impacts = []
        for i, (col, val) in enumerate(instance.items()):
            impacts.append({
                'feature': col,
                'value': val,
                'shap_impact': shap_vals[i]
            })
            
        # Sort by absolute impact
        impacts.sort(key=lambda x: abs(x['shap_impact']), reverse=True)
        
        top_positive = [f for f in impacts if f['shap_impact'] > 0][:top_n]
        top_negative = [f for f in impacts if f['shap_impact'] < 0][:top_n]
        
        # Generate Narrative
        tx_label = transaction_data.index[0] if transaction_data.index[0] != 0 else "N/A"
        report = f"### AI Investigation Report: Transaction #{tx_label}\n"
        report += f"**Risk Assessment**: {risk_tier} (Fraud Probability: {probability:.1%})\n\n"
        
        report += "**🚨 Red Flags (Factors driving the risk score UP):**\n"
        if not top_positive:
            report += "- None detected.\n"
        else:
            for f in top_positive:
                feat, val, impact = f['feature'], f['value'], f['shap_impact']
                report += f"- **{feat}** is `{val}`. *(High SHAP impact: +{impact:.2f} to log-odds)*\n"
                
        report += "\n**🛡️ Mitigating Factors (Factors driving the risk score DOWN):**\n"
        if not top_negative:
            report += "- None detected.\n"
        else:
            for f in top_negative:
                feat, val, impact = f['feature'], f['value'], f['shap_impact']
                report += f"- **{feat}** is `{val}`. *(SHAP impact: {impact:.2f})*\n"
                
        report += "\n---\n*Insight: SHAP values represent the mathematical contribution of each feature to the final prediction, isolating them from complex tree interactions.*"
        
        return report
