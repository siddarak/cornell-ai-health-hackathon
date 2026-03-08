import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

def generate_patient_data(n=50):
    """
    Generates realistic mock patient data for the ER Dashboard.
    NHAMCS-like fields included.
    """
    first_names = ["James", "Mary", "Robert", "Patricia", "John", "Jennifer", "Michael", "Linda"]
    last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis"]
    complaints = ["Chest Pain", "Abdominal Pain", "Shortness of Breath", "Fever", "Fall", "Headache", "Nausea", "Laceration"]
    
    data = []
    now = datetime.now()
    
    for i in range(n):
        age = random.randint(18, 90)
        gender = random.choice(["M", "F"])
        arrival_time = now - timedelta(minutes=random.randint(10, 300))
        wait_time = (now - arrival_time).seconds // 60
        
        # Vitals generation
        pulse = random.randint(50, 120)
        temp = round(random.uniform(96.5, 101.5), 1)
        resp = random.randint(10, 26)
        sys = random.randint(80, 160)
        dia = random.randint(50, 100)
        o2 = random.randint(88, 100)
        
        # Triage and Priority
        triage = random.randint(1, 5)
        priority_score = round(random.uniform(1.0, 10.0), 1)
        
        data.append({
            "MRN": f"ER-{random.randint(1000, 9999)}",
            "Name": f"{random.choice(last_names)}, {random.choice(first_names)}",
            "Age": age,
            "Gender": gender,
            "Arrival": arrival_time.strftime("%H:%M"),
            "WaitTime": wait_time,
            "TriageLevel": triage,
            "Pulse": pulse,
            "Temp": temp,
            "Resp": resp,
            "Sys": sys,
            "Dia": dia,
            "O2": o2,
            "ChiefComplaint": random.choice(complaints),
            "PriorityScore": priority_score,
            "RiskFlags": random.choice(["None", "Dehydration", "Cardiac Risk", "High BP", "Hypoxia"]),
            "Location": random.choice(["Waiting", "Bay 1", "Bay 2", "Triage", "Fast Track"]),
            "AssignedDoc": random.choice(["Dr. Smith", "Dr. Jones", "Dr. Lee", "Unassigned"])
        })
        
    return pd.DataFrame(data)
