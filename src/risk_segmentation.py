import pandas as pd
import numpy as np
import plotly.express as px
from typing import Dict, Any, List
from src.utils import logger
from src.config import TARGET_COL

# NOTE: plot_theme is imported inside plotting methods so this module can be
# used independently of the dashboard if needed (e.g. from train.py).
_RISK_COLOR_MAP = {
    'Clear':         '#21C55D',
    'Suspicious':    '#FFA62B',
    'Critical Risk': '#FF4B4B',
}
_FRAUD_COLOR_MAP = {'0': '#96BAFF', '1': '#FF4B4B'}


class RiskSegmenter:
    """
    Module for Risk Segmentation and Fraud Pattern Analysis.
    Translates model probabilities into actionable business tiers.
    All plots return native Plotly figures styled with the FraudOps theme.
    """

    def __init__(self, df: pd.DataFrame, proba_col: str = 'prediction_probability'):
        self.df = df.copy()
        self.proba_col = proba_col
        self._segment_data()

    def _segment_data(self):
        """Applies business logic to create risk tiers."""
        logger.info("Segmenting data into Risk Tiers...")
        conditions = [
            (self.df[self.proba_col] >= 0.75),
            (self.df[self.proba_col] >= 0.40) & (self.df[self.proba_col] < 0.75),
            (self.df[self.proba_col] < 0.40),
        ]
        choices = ['Critical Risk', 'Suspicious', 'Clear']
        self.df['Risk_Tier'] = np.select(conditions, choices, default='Unknown')

        self.df['Risk_Tier'] = pd.Categorical(
            self.df['Risk_Tier'],
            categories=['Clear', 'Suspicious', 'Critical Risk'],
            ordered=True,
        )

    def get_tier_summaries(self) -> pd.DataFrame:
        """Calculates KPIs for each risk tier."""
        summary = self.df.groupby('Risk_Tier', observed=True).agg(
            Transaction_Count=('Risk_Tier', 'count'),
            Avg_TransactionAmt=('TransactionAmt', 'mean'),
            Total_Value_At_Risk=('TransactionAmt', 'sum'),
            Mean_Probability=(self.proba_col, 'mean'),
        ).reset_index()
        summary['%_of_Total_Transactions'] = summary['Transaction_Count'] / len(self.df) * 100
        return summary

    def extract_top_fraud_patterns(self) -> str:
        """Analyzes the 'Critical Risk' tier to extract the Top 3 defining characteristics."""
        critical_df = self.df[self.df['Risk_Tier'] == 'Critical Risk']
        if len(critical_df) == 0:
            return "No Critical Risk transactions found."

        report = "### 🚨 Top 3 Fraud Patterns (Critical Risk Segment)\n\n"

        median_crit = critical_df['TransactionAmt'].median()
        median_clear = self.df[self.df['Risk_Tier'] == 'Clear']['TransactionAmt'].median()
        if median_crit > median_clear * 2:
            report += (
                f"**1. High-Value Targeting**: The median transaction amount for critical fraud "
                f"(${median_crit:.2f}) is significantly higher than normal transactions "
                f"(${median_clear:.2f}). Fraudsters are attempting account takeovers with large ticket sizes.\n"
            )

        if 'HourOfDay' in critical_df.columns:
            peak_hour = critical_df['HourOfDay'].mode()[0]
            report += (
                f"**2. Temporal Anomaly**: A significant cluster of critical risk transactions "
                f"occurs around **{peak_hour}:00**. This suggests automated bot scripts or "
                f"international fraud rings operating in specific timezones.\n"
            )

        if 'AmtToMeanRatio' in critical_df.columns:
            high_ratio_pct = (critical_df['AmtToMeanRatio'] > 5).mean() * 100
            report += (
                f"**3. Spending Velocity**: {high_ratio_pct:.1f}% of critical transactions are "
                f"over 5× the normal spending average for that specific card, indicating sudden "
                f"bursts of irregular spending.\n"
            )
        return report

    # ── Plotting Utilities ────────────────────────────────────────────────────

    def plot_tier_distribution(self, return_fig: bool = False):
        """Donut chart showing the distribution of transactions across risk tiers."""
        from dashboard.utils.plot_theme import apply_plotly_theme_minimal, COLORS
        counts = self.df['Risk_Tier'].value_counts().reset_index()
        counts.columns = ['Risk_Tier', 'Count']

        fig = px.pie(
            counts,
            names='Risk_Tier',
            values='Count',
            hole=0.45,
            color='Risk_Tier',
            color_discrete_map=_RISK_COLOR_MAP,
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(color='white', size=13),
            marker=dict(line=dict(color='rgba(0,0,0,0.4)', width=2)),
        )
        apply_plotly_theme_minimal(fig)
        fig.update_layout(
            title=dict(
                text="Transaction Distribution by Risk Tier",
                font=dict(size=17, color=COLORS["text"]),
                x=0.01, xanchor="left",
            ),
            margin=dict(t=55, b=10, l=10, r=10),
        )
        if return_fig:
            return fig
        fig.show()

    def plot_prob_distribution(self, return_fig: bool = False):
        """Native Plotly histogram — NEVER uses Matplotlib or mpl_to_plotly conversion."""
        fig = px.histogram(
            self.df,
            x="prediction_probability",
            nbins=50,
            color_discrete_sequence=["#EF4444"],
        )
        fig.update_layout(
            template="plotly_dark",
            bargap=0.05,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Arial", size=14, color="white"),
            title=dict(text="Fraud Probability Distribution", font=dict(size=20, color="white")),
            xaxis_title="Fraud Probability",
            yaxis_title="Transaction Count",
        )
        fig.add_vline(
            x=0.40, line_dash="dash", line_color="#FFA62B", line_width=2,
            annotation_text="Suspicious (0.40)",
            annotation_font_color="#FFA62B",
            annotation_position="top right",
        )
        fig.add_vline(
            x=0.75, line_dash="dash", line_color="#FF4B4B", line_width=2,
            annotation_text="Critical (0.75)",
            annotation_font_color="#FF4B4B",
            annotation_position="top right",
        )
        if return_fig:
            return fig
        fig.show()

    def plot_hourly_activity(self, return_fig: bool = False):
        """Stacked bar chart showing risk tier activity by hour of day."""
        from dashboard.utils.plot_theme import apply_plotly_theme
        if 'HourOfDay' not in self.df.columns:
            logger.warning("HourOfDay not in dataframe.")
            return None

        hourly_data = (
            self.df.groupby(['HourOfDay', 'Risk_Tier'], observed=True)
            .size()
            .reset_index(name='Transaction Count')
        )
        fig = px.bar(
            hourly_data,
            x='HourOfDay',
            y='Transaction Count',
            color='Risk_Tier',
            barmode='stack',
            color_discrete_map=_RISK_COLOR_MAP,
            category_orders={'Risk_Tier': ['Clear', 'Suspicious', 'Critical Risk']},
            labels={'HourOfDay': 'Hour of Day'},
        )
        fig.update_traces(marker_line_width=0)
        apply_plotly_theme(fig, title="Transaction Volume by Hour and Risk Tier")
        if return_fig:
            return fig
        fig.show()
