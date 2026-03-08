"""
Vitals Logic: Clinical range validation based on NHAMCS ED analysis.
"""

def get_vitals_status(age, pulse, temp, resp, sys, dia, o2):
    """
    Returns a status dictionary for patient vitals.
    Logic derived from ed2022_normal_vitals_analysis.ipynb.
    """
    status = {
        "pulse": "Normal" if 60 <= pulse <= 100 else "Abnormal",
        "temp": "Normal" if 97.0 <= temp <= 99.0 else "Abnormal",
        "resp": "Normal" if 12 <= resp <= 20 else "Abnormal",
        "systolic": "Normal" if 90 <= sys <= 120 else "Abnormal",
        "diastolic": "Normal" if 60 <= dia <= 80 else "Abnormal",
        "o2": "Normal" if o2 >= 95 else "Abnormal"
    }
    
    # Calculate overall clinical status
    abnormal_count = list(status.values()).count("Abnormal")
    if abnormal_count >= 3:
        status["overall"] = "CRITICAL"
        status["color"] = "#ef4444" # Red
        status["icon"] = "🔴"
    elif abnormal_count > 0:
        status["overall"] = "URGENT"
        status["color"] = "#f59e0b" # Yellow/Amber
        status["icon"] = "🟡"
    else:
        status["overall"] = "STABLE"
        status["color"] = "#10b981" # Green
        status["icon"] = "🟢"
        
    return status

def get_triage_color(level):
    """Maps NHAMCS triage levels to UI colors."""
    mapping = {
        1: ("IMMEDIATE", "#ef4444", "🔴"),
        2: ("EMERGENT", "#f97316", "🟠"),
        3: ("URGENT", "#f59e0b", "🟡"),
        4: ("SEMI-URGENT", "#10b981", "🟢"),
        5: ("NON-URGENT", "#6b7280", "⚪")
    }
    return mapping.get(level, ("UNKNOWN", "#374151", "❓"))
