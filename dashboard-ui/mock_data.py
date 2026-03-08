import random
from datetime import datetime, timedelta

import pandas as pd


FIRST_NAMES = [
    "James",
    "Mary",
    "Robert",
    "Patricia",
    "John",
    "Jennifer",
    "Michael",
    "Linda",
    "Daniel",
    "Ava",
]
LAST_NAMES = [
    "Smith",
    "Johnson",
    "Williams",
    "Brown",
    "Jones",
    "Garcia",
    "Miller",
    "Davis",
    "Wilson",
    "Martinez",
]
COMPLAINTS = [
    "Chest Pain",
    "Abdominal Pain",
    "Shortness of Breath",
    "Fever",
    "Fall",
    "Headache",
    "Nausea",
    "Laceration",
    "Syncope",
    "Weakness",
]
RISK_FLAG_OPTIONS = [
    "Sepsis Risk",
    "Cardiac Concern",
    "Respiratory Decline",
    "Fall Risk",
    "Dehydration",
    "High BP",
    "Hypoxia",
]
DISPOSITIONS = ["Admit", "Discharge", "Observe"]
NEXT_ACTIONS = [
    "Reassess in 10 min",
    "Order Labs",
    "Order CT",
    "Notify Provider",
    "Escalate to senior",
    "Place on monitor",
]
WORKFLOW_STAGES = ["Intake", "Triage", "Provider", "Orders", "Disposition"]


def _pick_workflow_stage(priority_score: float) -> str:
    if priority_score >= 8.0:
        return "Provider"
    if priority_score >= 6.0:
        return "Triage"
    if priority_score >= 4.0:
        return "Orders"
    return "Intake"


def _format_vitals_summary(sys_bp: int, dia_bp: int, pulse: int, temp: float, o2: int, resp: int) -> str:
    return f"BP {sys_bp}/{dia_bp}, HR {pulse}, RR {resp}, Temp {temp:.1f}, SpO2 {o2}%"


def generate_patient_data(n: int = 50, seed: int = 42) -> pd.DataFrame:
    """Generate mock patient data with UI-ready ML insight fields."""
    rng = random.Random(seed)
    now = datetime.now()
    rows = []

    for i in range(n):
        age = rng.randint(18, 90)
        gender = rng.choice(["M", "F"])
        arrival_dt = now - timedelta(minutes=rng.randint(5, 320))
        wait_minutes = int((now - arrival_dt).total_seconds() // 60)

        pulse = rng.randint(52, 128)
        temp = round(rng.uniform(96.5, 102.2), 1)
        resp = rng.randint(10, 30)
        sys_bp = rng.randint(85, 172)
        dia_bp = rng.randint(48, 108)
        o2 = rng.randint(86, 100)

        triage_level = rng.randint(1, 5)
        priority_score = round(rng.uniform(1.0, 10.0), 1)

        risk_count = rng.choice([1, 1, 2, 2, 3])
        risk_flags = ", ".join(rng.sample(RISK_FLAG_OPTIONS, k=risk_count))

        top_factors = [
            f"Triage={triage_level}",
            f"Wait={wait_minutes}m",
            rng.choice(["HR trend", "SpO2 trend", "Chief complaint", "Age + comorbid profile"]),
        ]

        rows.append(
            {
                "patient_id": f"P{i+1:03d}",
                "name": f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}",
                "age": age,
                "gender": gender,
                "arrival_time": arrival_dt.strftime("%H:%M"),
                "wait_minutes": wait_minutes,
                "triage_level": triage_level,
                "chief_complaint": rng.choice(COMPLAINTS),
                "location_bed": rng.choice(["Waiting", "Bed 4A", "Bed 6C", "Fast Track", "Hall 2"]),
                "assigned_provider": rng.choice(["Dr. Smith", "Dr. Lee", "Dr. Patel", "Unassigned"]),
                "vitals_summary": _format_vitals_summary(sys_bp, dia_bp, pulse, temp, o2, resp),
                "pulse": pulse,
                "temp": temp,
                "resp": resp,
                "sys_bp": sys_bp,
                "dia_bp": dia_bp,
                "o2_sat": o2,
                "ml_priority_score": priority_score,
                "ml_risk_flags": risk_flags,
                "predicted_disposition": rng.choice(DISPOSITIONS),
                "recommended_next_action": rng.choice(NEXT_ACTIONS),
                "ml_top_factors": " | ".join(top_factors),
                "ml_confidence": round(rng.uniform(0.58, 0.97), 2),
                "workflow_stage": _pick_workflow_stage(priority_score),
            }
        )

    return pd.DataFrame(rows)
