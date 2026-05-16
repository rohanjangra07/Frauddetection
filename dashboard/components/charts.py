import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.formatting import get_risk_color
from utils.plot_theme import apply_plotly_theme, apply_plotly_theme_minimal, RISK_COLOR_MAP, COLORS

def plot_risk_tier_donut(df: pd.DataFrame):
    """Creates a donut chart for risk tier distribution."""
    counts = df['Risk_Tier'].value_counts().reset_index()
    counts.columns = ['Risk_Tier', 'Count']

    color_map = {tier: get_risk_color(tier) for tier in counts['Risk_Tier']}

    fig = px.pie(
        counts,
        values='Count',
        names='Risk_Tier',
        hole=0.45,
        color='Risk_Tier',
        color_discrete_map=color_map,
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont=dict(color='white', size=13),
        marker=dict(line=dict(color='rgba(0,0,0,0.4)', width=2)),
    )
    apply_plotly_theme_minimal(fig)
    fig.update_layout(
        title=dict(text="Transaction Distribution by Risk Tier",
                   font=dict(size=17, color=COLORS["text"]), x=0.01, xanchor="left"),
        margin=dict(t=55, b=10, l=10, r=10),
        showlegend=True,
    )
    return fig


def plot_fraud_by_hour(df: pd.DataFrame):
    """Creates a bar chart showing fraud count by hour of day."""
    if 'HourOfDay' not in df.columns:
        return go.Figure()

    hourly_stats = df.groupby('HourOfDay').agg(
        Total_Txns=('isFraud', 'count'),
        Fraud_Txns=('isFraud', 'sum'),
    ).reset_index()
    hourly_stats['Fraud_Rate'] = hourly_stats['Fraud_Txns'] / hourly_stats['Total_Txns']

    fig = px.bar(
        hourly_stats,
        x='HourOfDay',
        y='Fraud_Txns',
        labels={'HourOfDay': 'Hour (0-23)', 'Fraud_Txns': 'Fraud Count'},
        color_discrete_sequence=[COLORS["fraud"]],
    )
    fig.update_traces(marker_line_width=0, opacity=0.88)
    apply_plotly_theme(fig, title="Detected Fraud Transactions by Hour of Day")
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=1))
    return fig


def plot_transaction_amt_dist(df: pd.DataFrame):
    """Creates a histogram for Transaction Amounts split by fraud status."""
    q99 = df['TransactionAmt'].quantile(0.99)
    filtered_df = df[df['TransactionAmt'] <= q99].copy()
    filtered_df['isFraud'] = filtered_df['isFraud'].astype(str)

    fig = px.histogram(
        filtered_df,
        x='TransactionAmt',
        color='isFraud',
        nbins=60,
        barmode='overlay',
        labels={'isFraud': 'Is Fraud', 'TransactionAmt': 'Amount ($)'},
        color_discrete_map={'0': COLORS["secondary"], '1': COLORS["fraud"]},
        opacity=0.78,
    )
    apply_plotly_theme(fig, title="Transaction Amount Distribution (Top 1% Excluded)")
    return fig


def plot_scatter_amt_vs_hour(df: pd.DataFrame):
    """Scatter plot of Amount vs Hour, colored by fraud probability."""
    if 'HourOfDay' not in df.columns or 'prediction_probability' not in df.columns:
        return go.Figure()

    plot_df = df.sample(n=min(2000, len(df)), random_state=42) if len(df) > 2000 else df

    fig = px.scatter(
        plot_df,
        x='HourOfDay',
        y='TransactionAmt',
        color='prediction_probability',
        color_continuous_scale='Reds',
        hover_data=['TransactionID', 'Risk_Tier'],
        labels={'prediction_probability': 'Fraud Prob.'},
        opacity=0.75,
    )
    apply_plotly_theme(fig, title="Transaction Amount vs Hour (Colored by Fraud Probability)")
    fig.update_layout(xaxis=dict(tickmode='linear', dtick=2))
    return fig
