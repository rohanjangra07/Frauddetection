def format_currency(value: float) -> str:
    """Formats a float as USD currency."""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Formats a float as a percentage."""
    return f"{value * 100:.2f}%"

def get_risk_color(tier: str) -> str:
    """Returns a hex color code based on the risk tier."""
    color_map = {
        'Critical Risk': '#d62728', # Red
        'Suspicious': '#ff7f0e',    # Orange
        'Clear': '#2ca02c',         # Green
        'Unknown': '#7f7f7f'        # Gray
    }
    return color_map.get(tier, '#7f7f7f')

def apply_risk_styling(val):
    """Pandas styling function for Risk Tiers."""
    color = get_risk_color(val)
    return f'color: {color}; font-weight: bold;'
