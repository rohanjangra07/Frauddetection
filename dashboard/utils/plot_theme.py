"""
Centralized Plotly Theme Utility for FraudOps Dashboard.

All dashboard charts MUST be passed through apply_plotly_theme() before
being returned to Streamlit. This guarantees visual consistency across every
page: font family, colors, backgrounds, gridlines, and hover styling.
"""
import plotly.graph_objects as go
import matplotlib as mpl

# ── Brand Palette ────────────────────────────────────────────────────────────
COLORS = {
    "fraud":     "#FF4B4B",   # vivid red  – fraud / critical
    "suspicious":"#FFA62B",   # amber      – suspicious
    "clear":     "#21C55D",   # emerald    – legitimate / safe
    "primary":   "#7C83FD",   # periwinkle – primary accent
    "secondary": "#96BAFF",   # sky blue   – secondary accent
    "text":      "#F0F2F6",   # near-white – all text
    "subtext":   "#9BA3AF",   # muted grey – secondary text / ticks
    "grid":      "rgba(255,255,255,0.08)",
    "bg":        "rgba(0,0,0,0)",   # fully transparent (inherits Streamlit bg)
}

RISK_COLOR_MAP = {
    "Critical Risk": COLORS["fraud"],
    "Suspicious":    COLORS["suspicious"],
    "Clear":         COLORS["clear"],
}

FONT_FAMILY = "Inter, Arial, sans-serif"

# ── Core Layout Defaults ─────────────────────────────────────────────────────
_BASE_LAYOUT = dict(
    template="plotly_dark",
    font=dict(
        family=FONT_FAMILY,
        size=13,
        color=COLORS["text"],
    ),
    title_font=dict(
        family=FONT_FAMILY,
        size=18,
        color=COLORS["text"],
    ),
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    legend=dict(
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.15)",
        borderwidth=1,
        font=dict(color=COLORS["text"], size=12),
        title_font=dict(color=COLORS["text"]),
    ),
    hoverlabel=dict(
        bgcolor="#1E2130",
        bordercolor="rgba(255,255,255,0.25)",
        font=dict(family=FONT_FAMILY, size=12, color=COLORS["text"]),
    ),
    margin=dict(t=55, b=45, l=50, r=20),
)

_AXIS_STYLE = dict(
    showgrid=True,
    gridcolor=COLORS["grid"],
    gridwidth=1,
    zeroline=False,
    linecolor="rgba(255,255,255,0.2)",
    tickcolor=COLORS["subtext"],
    tickfont=dict(color=COLORS["subtext"], size=11),
    title_font=dict(color=COLORS["text"], size=13),
)


def apply_plotly_theme(fig: go.Figure, *, title: str = None) -> go.Figure:
    """
    Apply the FraudOps centralized dark theme to any Plotly figure.

    Usage:
        fig = px.bar(...)
        fig = apply_plotly_theme(fig, title="My Chart")

    Returns the same figure (mutated in-place) for chaining convenience.
    """
    update = dict(**_BASE_LAYOUT)
    if title:
        update["title"] = dict(
            text=title,
            font=dict(family=FONT_FAMILY, size=18, color=COLORS["text"]),
            x=0.01,
            xanchor="left",
        )

    fig.update_layout(
        **update,
        xaxis=_AXIS_STYLE,
        yaxis=_AXIS_STYLE,
    )
    return fig


def apply_plotly_theme_minimal(fig: go.Figure) -> go.Figure:
    """
    Lighter version for charts where axis styling must be preserved
    (e.g. pie / donut where x/y axis don't apply).
    """
    fig.update_layout(**_BASE_LAYOUT)
    return fig


# ── Matplotlib Dark Theme (for SHAP plots) ──────────────────────────────────
def apply_matplotlib_dark_theme():
    """
    Configure Matplotlib rcParams for dark-mode SHAP plots so they
    render with white text and transparent backgrounds on Streamlit.
    Call this once before generating any Matplotlib / SHAP figure.
    """
    mpl.rcParams.update({
        "figure.facecolor":  "none",          # transparent bg
        "axes.facecolor":    "#0E1117",        # Streamlit dark bg
        "axes.edgecolor":    COLORS["subtext"],
        "axes.labelcolor":   COLORS["text"],
        "axes.titlecolor":   COLORS["text"],
        "xtick.color":       COLORS["subtext"],
        "ytick.color":       COLORS["subtext"],
        "text.color":        COLORS["text"],
        "grid.color":        COLORS["grid"],
        "legend.facecolor":  "#1E2130",
        "legend.edgecolor":  "none",
        "legend.labelcolor": COLORS["text"],
        "savefig.transparent": True,
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "DejaVu Sans"],
    })
