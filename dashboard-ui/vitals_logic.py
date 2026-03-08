"""Clinical and display status helpers for the ER dashboard."""

from __future__ import annotations


def _abnormal_count(pulse: int, temp: float, resp: int, sys_bp: int, dia_bp: int, o2_sat: int) -> int:
    flags = [
        not (60 <= pulse <= 100),
        not (97.0 <= temp <= 99.0),
        not (12 <= resp <= 20),
        not (90 <= sys_bp <= 140),
        not (55 <= dia_bp <= 90),
        o2_sat < 94,
    ]
    return sum(flags)


def get_status_tier(priority_score: float, pulse: int, temp: float, resp: int, sys_bp: int, dia_bp: int, o2_sat: int) -> str:
    """Return red/yellow/green tier using both ML score and vitals context."""
    abn = _abnormal_count(pulse, temp, resp, sys_bp, dia_bp, o2_sat)

    if priority_score >= 8.0 or abn >= 3:
        return "red"
    if priority_score >= 5.0 or abn >= 1:
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
