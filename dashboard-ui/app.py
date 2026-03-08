from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mock_data import generate_patient_data
from vitals_logic import get_status_tier, get_status_ui


st.set_page_config(
    page_title="ER Clinical Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
html, body, [class*="css"] {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.stApp {
    background: #f5f7fb;
    color: #111827;
}
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1.4rem;
}
.kpi {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px 14px;
}
.patient-card {
    border-radius: 12px;
    border: 1px solid #e5e7eb;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.patient-card-top {
    display: flex;
    gap: 10px;
}
.avatar {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #0f172a;
    background: #e5e7eb;
    border: 2px solid #cbd5e1;
    flex-shrink: 0;
}
.factor-chip {
    display: inline-block;
    border-radius: 999px;
    border: 1px solid #d1d5db;
    background: #f9fafb;
    color: #374151;
    font-size: 0.72rem;
    padding: 2px 8px;
    margin-right: 6px;
    margin-top: 4px;
}
.patient-card h4 {
    margin: 0;
    font-size: 1rem;
}
.meta-line {
    color: #4b5563;
    font-size: 0.86rem;
}
.badge {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 600;
    display: inline-block;
}
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
.small-muted {
    color: #6b7280;
    font-size: 0.82rem;
}
</style>
""",
    unsafe_allow_html=True,
)


EXPECTED_COLUMNS = [
    "patient_id",
    "name",
    "age",
    "gender",
    "arrival_time",
    "wait_minutes",
    "triage_level",
    "chief_complaint",
    "location_bed",
    "vitals_summary",
    "assigned_provider",
    "pulse",
    "temp",
    "resp",
    "sys_bp",
    "dia_bp",
    "o2_sat",
    "ml_priority_score",
    "ml_risk_flags",
    "predicted_disposition",
    "recommended_next_action",
    "ml_top_factors",
    "ml_confidence",
    "workflow_stage",
]


WORKFLOW_STAGES = ["Intake", "Triage", "Provider", "Orders", "Disposition"]


def _find_col(df: pd.DataFrame, options: Iterable[str]) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for opt in options:
        if opt.lower() in cols:
            return cols[opt.lower()]
    return None


def _to_num(series: pd.Series, default: float = np.nan) -> pd.Series:
    out = pd.to_numeric(series, errors="coerce")
    if np.isnan(default):
        return out
    return out.fillna(default)


def _build_vitals_summary(df: pd.DataFrame) -> pd.Series:
    return (
        "BP "
        + df["sys_bp"].round(0).astype("Int64").astype(str)
        + "/"
        + df["dia_bp"].round(0).astype("Int64").astype(str)
        + ", HR "
        + df["pulse"].round(0).astype("Int64").astype(str)
        + ", RR "
        + df["resp"].round(0).astype("Int64").astype(str)
        + ", Temp "
        + df["temp"].round(1).astype(str)
        + ", SpO2 "
        + df["o2_sat"].round(0).astype("Int64").astype(str)
        + "%"
    )


def normalize_uploaded_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded patient CSV into dashboard schema with sensible defaults."""
    df = raw_df.copy()

    mapped = pd.DataFrame()
    mapped["patient_id"] = df[_find_col(df, ["patient_id", "patientid", "mrn", "id"]) or df.columns[0]].astype(str)
    mapped["name"] = (
        df[_find_col(df, ["name", "patient_name", "patient"]) or df.columns[0]].astype(str)
        if _find_col(df, ["name", "patient_name", "patient"]) is not None
        else "Unknown Patient"
    )
    mapped["age"] = _to_num(df[_find_col(df, ["age"])], default=45) if _find_col(df, ["age"]) else 45

    sex_col = _find_col(df, ["gender", "sex"])
    mapped["gender"] = df[sex_col].astype(str).str.upper().str[0] if sex_col else "U"

    arr_col = _find_col(df, ["arrival_time", "arrival", "arrtime"])
    if arr_col:
        arr_vals = df[arr_col].astype(str)
        mapped["arrival_time"] = arr_vals.str.replace("b'", "", regex=False).str.replace("'", "", regex=False).str.zfill(4).str[:2] + ":" + arr_vals.str.replace("b'", "", regex=False).str.replace("'", "", regex=False).str.zfill(4).str[2:]
    else:
        mapped["arrival_time"] = "--:--"

    wait_col = _find_col(df, ["wait_minutes", "waittime", "wait"])
    mapped["wait_minutes"] = _to_num(df[wait_col], default=0).clip(lower=0) if wait_col else 0

    triage_col = _find_col(df, ["triage_level", "triage", "immedr"])
    mapped["triage_level"] = _to_num(df[triage_col], default=3).clip(lower=1, upper=5) if triage_col else 3

    complaint_col = _find_col(df, ["chief_complaint", "complaint", "rfv1"])
    mapped["chief_complaint"] = df[complaint_col].astype(str) if complaint_col else "General Complaint"

    mapped["location_bed"] = (
        df[_find_col(df, ["location_bed", "location", "bed"])].astype(str)
        if _find_col(df, ["location_bed", "location", "bed"])
        else "Waiting"
    )

    mapped["assigned_provider"] = (
        df[_find_col(df, ["assigned_provider", "provider", "doctor", "assigneddoc"])].astype(str)
        if _find_col(df, ["assigned_provider", "provider", "doctor", "assigneddoc"])
        else "Unassigned"
    )

    mapped["pulse"] = _to_num(df[_find_col(df, ["pulse", "hr"])], default=85) if _find_col(df, ["pulse", "hr"]) else 85
    mapped["temp"] = _to_num(df[_find_col(df, ["temp", "temperature", "tempf"])], default=98.2) if _find_col(df, ["temp", "temperature", "tempf"]) else 98.2
    mapped["resp"] = _to_num(df[_find_col(df, ["resp", "respr", "rr"])], default=16) if _find_col(df, ["resp", "respr", "rr"]) else 16
    mapped["sys_bp"] = _to_num(df[_find_col(df, ["sys_bp", "bpsys", "sbp"])], default=120) if _find_col(df, ["sys_bp", "bpsys", "sbp"]) else 120
    mapped["dia_bp"] = _to_num(df[_find_col(df, ["dia_bp", "bpdias", "dbp"])], default=80) if _find_col(df, ["dia_bp", "bpdias", "dbp"]) else 80
    mapped["o2_sat"] = _to_num(df[_find_col(df, ["o2_sat", "o2", "spo2", "popct"])], default=97) if _find_col(df, ["o2_sat", "o2", "spo2", "popct"]) else 97

    score_col = _find_col(df, ["ml_priority_score", "priorityscore", "icu_risk_probability", "risk_score"])
    if score_col:
        vals = _to_num(df[score_col], default=0)
        if vals.max() <= 1.0:
            vals = vals * 10.0
        mapped["ml_priority_score"] = vals.clip(lower=0, upper=10)
    else:
        mapped["ml_priority_score"] = (11 - mapped["triage_level"] * 1.6 + mapped["wait_minutes"] / 90).clip(1, 10)

    mapped["ml_confidence"] = (
        _to_num(df[_find_col(df, ["ml_confidence", "confidence"])], default=0.75).clip(0.5, 1.0)
        if _find_col(df, ["ml_confidence", "confidence"])
        else 0.75
    )

    mapped["ml_risk_flags"] = (
        df[_find_col(df, ["ml_risk_flags", "riskflags", "risk_flags"])].astype(str)
        if _find_col(df, ["ml_risk_flags", "riskflags", "risk_flags"])
        else "Cardiac Concern"
    )

    mapped["predicted_disposition"] = (
        df[_find_col(df, ["predicted_disposition", "disposition", "predicteddisposition"])].astype(str)
        if _find_col(df, ["predicted_disposition", "disposition", "predicteddisposition"])
        else np.where(mapped["ml_priority_score"] >= 7.5, "Admit", "Observe")
    )

    mapped["recommended_next_action"] = (
        df[_find_col(df, ["recommended_next_action", "next_action", "recommendednextaction"])].astype(str)
        if _find_col(df, ["recommended_next_action", "next_action", "recommendednextaction"])
        else np.where(mapped["ml_priority_score"] >= 7.5, "Notify Provider", "Reassess in 10 min")
    )

    mapped["ml_top_factors"] = (
        df[_find_col(df, ["ml_top_factors", "factors", "reason_summary"])].astype(str)
        if _find_col(df, ["ml_top_factors", "factors", "reason_summary"])
        else "Triage acuity | Wait duration | Vitals pattern"
    )

    mapped["workflow_stage"] = (
        df[_find_col(df, ["workflow_stage", "stage"])].astype(str)
        if _find_col(df, ["workflow_stage", "stage"])
        else "Triage"
    )

    mapped["vitals_summary"] = _build_vitals_summary(mapped)

    for col in EXPECTED_COLUMNS:
        if col not in mapped.columns:
            mapped[col] = ""

    return mapped[EXPECTED_COLUMNS].copy()


def get_status_for_row(row: pd.Series) -> tuple[str, dict]:
    tier = get_status_tier(
        priority_score=float(row["ml_priority_score"]),
        pulse=int(row["pulse"]),
        temp=float(row["temp"]),
        resp=int(row["resp"]),
        sys_bp=int(row["sys_bp"]),
        dia_bp=int(row["dia_bp"]),
        o2_sat=int(row["o2_sat"]),
    )
    return tier, get_status_ui(tier)


def get_initials(name: str) -> str:
    parts = [p for p in str(name).split() if p]
    if not parts:
        return "PT"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()


def split_top_factors(text: str, top_k: int = 3) -> list[str]:
    factors = [f.strip() for f in str(text).replace(",", "|").split("|") if f.strip()]
    return factors[:top_k]


def style_status_gradient(value: str) -> str:
    if value == "Severe":
        return "background: linear-gradient(90deg, rgba(239,68,68,0.22), rgba(255,255,255,0.95)); color: #991b1b; font-weight: 700;"
    if value == "Risky":
        return "background: linear-gradient(90deg, rgba(245,158,11,0.22), rgba(255,255,255,0.95)); color: #92400e; font-weight: 700;"
    if value == "Stable":
        return "background: linear-gradient(90deg, rgba(16,185,129,0.2), rgba(255,255,255,0.95)); color: #065f46; font-weight: 700;"
    return ""


def workflow_bar(stage: str) -> str:
    chips = []
    for s in WORKFLOW_STAGES:
        if s.lower() == str(stage).lower():
            chips.append(f"<span class='badge' style='background:#dbeafe;color:#1d4ed8'>{s}</span>")
        else:
            chips.append(f"<span class='badge' style='background:#f3f4f6;color:#6b7280'>{s}</span>")
    return " ".join(chips)


@st.dialog("ML Insight Explanation")
def show_ml_explanation_modal(patient_row: pd.Series, status_ui: dict) -> None:
    st.markdown(
        f"**{patient_row['name']}** (`{patient_row['patient_id']}`)  \n"
        f"Status: **{status_ui['label']}**  \n"
        f"Priority: **{patient_row['ml_priority_score']:.1f}/10**  \n"
        f"Confidence: **{patient_row['ml_confidence']:.0%}**"
    )
    st.markdown("**Top Model Drivers**")
    for factor in split_top_factors(patient_row["ml_top_factors"]):
        st.markdown(f"- {factor}")
    st.markdown("**Risk Flags**")
    st.markdown(str(patient_row["ml_risk_flags"]))
    st.caption("Decision support only. Use alongside clinical judgment.")
    if st.button("Close"):
        st.session_state.explain_patient_id = None
        st.rerun()


# -----------------------------
# Data Source
# -----------------------------
with st.sidebar:
    st.markdown("### Data Source")
    uploaded = st.file_uploader("Upload cherry-picked patient CSV", type=["csv"])

if uploaded is not None:
    uploaded_df = pd.read_csv(uploaded, low_memory=False)
    patients_df = normalize_uploaded_df(uploaded_df)
    st.caption(f"Loaded {len(patients_df)} patients from uploaded CSV")
else:
    patients_df = generate_patient_data(55)

if "selected_patient_id" not in st.session_state and len(patients_df) > 0:
    st.session_state.selected_patient_id = patients_df.iloc[0]["patient_id"]
if "explain_patient_id" not in st.session_state:
    st.session_state.explain_patient_id = None


# -----------------------------
# Top Bar
# -----------------------------
st.title("ER Clinical Intelligence Dashboard")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Patients in View</div><div style='font-size:1.35rem;font-weight:700'>{len(patients_df)}</div></div>", unsafe_allow_html=True)
with k2:
    avg_wait = int(patients_df["wait_minutes"].mean()) if len(patients_df) else 0
    st.markdown(f"<div class='kpi'><div class='small-muted'>Avg Wait</div><div style='font-size:1.35rem;font-weight:700'>{avg_wait} min</div></div>", unsafe_allow_html=True)
with k3:
    high_risk = int((patients_df["ml_priority_score"] >= 7.5).sum())
    st.markdown(f"<div class='kpi'><div class='small-muted'>High ML Risk</div><div style='font-size:1.35rem;font-weight:700'>{high_risk}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Last Refresh</div><div style='font-size:1.35rem;font-weight:700'>{datetime.now().strftime('%H:%M:%S')}</div></div>", unsafe_allow_html=True)

st.markdown("---")


# -----------------------------
# Main Layout
# -----------------------------
center_col, panel_col = st.columns([3.4, 1.6], gap="large")

with center_col:
    st.subheader("Patient Queue")

    f1, f2, f3 = st.columns([1.5, 1.2, 1.2])
    with f1:
        search = st.text_input("Search name / ID", "")
    with f2:
        status_filter = st.multiselect("Status", ["Severe", "Risky", "Stable"], default=["Severe", "Risky", "Stable"])
    with f3:
        dispo_filter = st.multiselect("Disposition", sorted(patients_df["predicted_disposition"].unique()), default=list(sorted(patients_df["predicted_disposition"].unique())))

    df = patients_df.copy()
    status_cache = []
    for _, r in df.iterrows():
        tier, ui = get_status_for_row(r)
        status_cache.append((tier, ui["label"], ui))

    df["status_tier"] = [x[0] for x in status_cache]
    df["status_label"] = [x[1] for x in status_cache]
    df["_status_ui"] = [x[2] for x in status_cache]

    if search.strip():
        s = search.strip().lower()
        df = df[df["name"].str.lower().str.contains(s) | df["patient_id"].str.lower().str.contains(s)]

    df = df[df["status_label"].isin(status_filter)]
    df = df[df["predicted_disposition"].isin(dispo_filter)]
    df = df.sort_values(["ml_priority_score", "wait_minutes"], ascending=[False, False])

    if len(df) == 0:
        st.warning("No patients match current filters.")

    for _, row in df.iterrows():
        ui = row["_status_ui"]
        factor_chips = "".join([f"<span class='factor-chip'>{f}</span>" for f in split_top_factors(row["ml_top_factors"])])
        initials = get_initials(row["name"])

        card_col, btn_col = st.columns([8.3, 1.1])
        with card_col:
            st.markdown(
                f"""
                <div class='patient-card' style='background:{ui['gradient']}'>
                  <div class='patient-card-top'>
                    <div class='avatar' style='border-color:{ui['dot']}'>{initials}</div>
                    <div style='width:100%'>
                      <div style='display:flex;justify-content:space-between;align-items:center'>
                        <h4>{row['name']}</h4>
                        <span class='badge' style='background:{ui['badge_bg']};color:{ui['badge_fg']}'>{row['status_label']}</span>
                      </div>
                      <div class='meta-line'><span style='color:{ui['dot']};font-weight:700'>Status</span> • {row['patient_id']} • {int(row['age'])}/{row['gender']} • Arrived {row['arrival_time']} • Wait {int(row['wait_minutes'])} min</div>
                      <div class='meta-line'><b>Complaint:</b> {row['chief_complaint']} | <b>Location:</b> {row['location_bed']} | <b>Provider:</b> {row['assigned_provider']}</div>
                      <div class='meta-line'><b>ML:</b> {row['ml_priority_score']:.1f}/10 • {row['predicted_disposition']} • {row['recommended_next_action']}</div>
                      <div>{factor_chips}</div>
                    </div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with btn_col:
            if st.button("View", key=f"view_{row['patient_id']}"):
                st.session_state.selected_patient_id = row["patient_id"]
            if st.button("Why", key=f"why_{row['patient_id']}"):
                st.session_state.explain_patient_id = row["patient_id"]

    with st.expander("Compact Table View (optional)", expanded=False):
        table_df = df[
            [
                "patient_id",
                "name",
                "age",
                "gender",
                "arrival_time",
                "wait_minutes",
                "status_label",
                "chief_complaint",
                "location_bed",
                "vitals_summary",
                "assigned_provider",
                "ml_priority_score",
                "ml_risk_flags",
                "predicted_disposition",
                "recommended_next_action",
            ]
        ].copy()
        styled_df = table_df.style.applymap(style_status_gradient, subset=["status_label"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

    explain_id = st.session_state.get("explain_patient_id")
    if explain_id:
        explain_df = patients_df[patients_df["patient_id"] == explain_id]
        if len(explain_df) > 0:
            explain_row = explain_df.iloc[0]
            _, explain_ui = get_status_for_row(explain_row)
            show_ml_explanation_modal(explain_row, explain_ui)

with panel_col:
    st.subheader("Selected Patient")

    selected_id = st.session_state.get("selected_patient_id")
    selected_df = patients_df[patients_df["patient_id"] == selected_id]
    if len(selected_df) == 0 and len(patients_df) > 0:
        selected_df = patients_df.iloc[[0]]
        st.session_state.selected_patient_id = selected_df.iloc[0]["patient_id"]

    if len(selected_df) == 0:
        st.info("No patient available")
        st.stop()

    patient = selected_df.iloc[0]
    tier, ui = get_status_for_row(patient)

    st.markdown(
        f"""
        <div class='section-card' style='border-top:6px solid {ui['dot']}'>
            <div class='small-muted'>{patient['patient_id']}</div>
            <div style='font-size:1.2rem;font-weight:700'>{patient['name']}</div>
            <div class='meta-line'>{int(patient['age'])} / {patient['gender']} • Allergies: NKDA • Code Status: FULL CODE</div>
            <div style='margin-top:6px'>{workflow_bar(patient['workflow_stage'])}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Priority", f"{patient['ml_priority_score']:.1f}/10")
    m2.metric("Confidence", f"{patient['ml_confidence']:.0%}")
    m3.metric("Wait", f"{int(patient['wait_minutes'])}m")

    st.markdown("<div class='section-card'><b>Chief Complaint</b><br/>" + str(patient["chief_complaint"]) + "</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-card'><b>Vitals</b><br/>" + str(patient["vitals_summary"]) + "</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-card'><b>ML Risk Summary</b><br/>"
        + f"Status: {ui['label']} | Disposition: {patient['predicted_disposition']}<br/>"
        + f"Flags: {patient['ml_risk_flags']}<br/>"
        + f"Recommended Next Action: {patient['recommended_next_action']}"
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='section-card'><b>AI Rationale</b><br/>"
        + f"This status is driven by: {patient['ml_top_factors']}"
        + "</div>",
        unsafe_allow_html=True,
    )
    factor_list = split_top_factors(patient["ml_top_factors"])
    if factor_list:
        st.markdown("**Top Drivers**")
        for factor in factor_list:
            st.markdown(f"- {factor}")

    st.markdown("**Actions**")
    st.button("Assign Bed", use_container_width=True)
    st.button("Order Labs / Imaging", use_container_width=True)
    st.button("Notify Provider", use_container_width=True)
    st.button("Escalate", type="primary", use_container_width=True)

    st.markdown("---")
    st.markdown("**ER-wide Analytics**")

    occ = pd.DataFrame(
        {
            "State": ["Occupied", "Available", "Cleaning"],
            "Count": [max(1, int(len(patients_df) * 0.58)), max(1, int(len(patients_df) * 0.27)), max(1, int(len(patients_df) * 0.15))],
        }
    )
    fig_occ = px.pie(occ, names="State", values="Count", hole=0.55)
    fig_occ.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig_occ, use_container_width=True)

    waits = patients_df["wait_minutes"].clip(lower=0)
    bins = pd.cut(waits, bins=[0, 30, 60, 90, 120, 180, 500], right=False)
    wait_hist = waits.groupby(bins).count().reset_index(name="count")
    wait_hist["bucket"] = wait_hist["wait_minutes"].astype(str)
    fig_wait = px.bar(wait_hist, x="bucket", y="count")
    fig_wait.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Wait bucket", yaxis_title="Patients")
    st.plotly_chart(fig_wait, use_container_width=True)

    top_risks = (
        patients_df["ml_risk_flags"]
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
    )
    if len(top_risks) > 0:
        st.markdown("**Top Active Risks**")
        for risk, count in top_risks.items():
            st.markdown(f"- {risk}: {count}")

    load_forecast = int((patients_df["ml_priority_score"] >= 7.0).sum() * 0.6 + len(patients_df) * 0.15)
    st.info(f"ML Department Load Prediction (next 2-4h): ~{load_forecast} incoming/high-attention cases")


st.markdown(
    f"<div class='small-muted' style='text-align:center;margin-top:18px'>"
    f"Early warning decision support only — not diagnosis. Last updated {datetime.now().strftime('%H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
