"""Display logic for percentile-based ED risk tiers."""

from __future__ import annotations


def get_status_tier_from_percentile(risk_percentile: float) -> str:
    """Map prediction percentile to ED color tier.

    Schema:
    - Highest 15% => red (>85th percentile)
    - 65th to 85th => yellow
    - Lowest 65% => green
    """
    if risk_percentile > 85:
        return "red"
    if risk_percentile > 65:
        return "yellow"
    return "green"


def get_status_ui(tier: str) -> dict:
    """UI tokens for gradient cards and badges."""
    mapping = {
        "red": {
            "label": "Severe",
            "badge_bg": "#fee2e2",
            "badge_fg": "#b91c1c",
            "gradient": "linear-gradient(90deg, rgba(239,68,68,0.18) 0%, rgba(255,255,255,0.96) 50%)",
            "dot": "#ef4444",
        },
        "yellow": {
            "label": "Risky",
            "badge_bg": "#fef3c7",
            "badge_fg": "#b45309",
            "gradient": "linear-gradient(90deg, rgba(245,158,11,0.16) 0%, rgba(255,255,255,0.96) 50%)",
            "dot": "#f59e0b",
        },
        "green": {
            "label": "Stable",
            "badge_bg": "#dcfce7",
            "badge_fg": "#166534",
            "gradient": "linear-gradient(90deg, rgba(16,185,129,0.14) 0%, rgba(255,255,255,0.96) 50%)",
            "dot": "#10b981",
        },
    }
    return mapping.get(tier, mapping["yellow"])
