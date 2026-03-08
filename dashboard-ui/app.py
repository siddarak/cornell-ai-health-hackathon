from __future__ import annotations

from datetime import datetime
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from mock_data import generate_patient_data
from vitals_logic import get_status_tier_from_percentile, get_status_ui


st.set_page_config(
    page_title="ER Clinical Intelligence Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root {
    color-scheme: light !important;
}
html, body, [class*="css"] {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
.stApp {
    background: #f5f7fb;
    color: #111827;
}
[data-testid="stSidebar"] {
    background: #eef2ff;
    border-right: 1px solid #dbeafe;
}
.main .block-container {
    padding-top: 1.1rem;
    padding-bottom: 1.25rem;
}
.kpi {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 14px;
    padding: 10px 14px;
}
.queue-note {
    color: #4b5563;
    font-size: 0.82rem;
}
.patient-card {
    border-radius: 14px;
    border: 1px solid #e5e7eb;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.patient-card-top {
    display: flex;
    gap: 10px;
}
.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.78rem;
    font-weight: 700;
    color: #0f172a;
    background: #ffffff;
    border: 2px solid #cbd5e1;
    flex-shrink: 0;
}
.patient-card h4 {
    margin: 0;
    font-size: 1rem;
}
.meta-line {
    color: #334155;
    font-size: 0.84rem;
    margin-top: 2px;
}
.badge {
    border-radius: 999px;
    padding: 2px 8px;
    font-size: 0.75rem;
    font-weight: 700;
    display: inline-block;
}
.flag-chip {
    display: inline-block;
    border-radius: 999px;
    border: 1px solid #cbd5e1;
    background: #ffffff;
    color: #334155;
    font-size: 0.70rem;
    padding: 1px 7px;
    margin-right: 5px;
    margin-top: 5px;
}
.mini-grid {
    margin-top: 8px;
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 6px;
}
.mini-tile {
    border: 1px solid #dbeafe;
    background: #ffffff;
    border-radius: 10px;
    padding: 6px;
}
.mini-label {
    color: #64748b;
    font-size: 0.66rem;
    text-transform: uppercase;
    letter-spacing: 0.02em;
}
.mini-value {
    color: #0f172a;
    font-size: 0.88rem;
    font-weight: 700;
}
.section-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 10px 12px;
    margin-bottom: 10px;
}
.small-muted {
    color: #64748b;
    font-size: 0.82rem;
}
.avatar-pill {
    width: 36px;
    height: 36px;
    border-radius: 999px;
    border: 2px solid #94a3b8;
    background: #ffffff;
    color: #0f172a;
    font-weight: 700;
    font-size: 0.78rem;
    display: flex;
    align-items: center;
    justify-content: center;
}
.age-sex {
    font-size: 1.2rem;
    font-weight: 800;
    color: #0f172a;
    line-height: 1.1;
}
.chip-title {
    color: #475569;
    font-size: 0.72rem;
    margin-bottom: 2px;
}
.chip-row {
    margin-top: 4px;
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
    "risk_score",
    "ml_risk_flags",
    "predicted_disposition",
    "recommended_next_action",
    "ml_top_factors",
    "ml_confidence",
    "workflow_stage",
]

AVAILABLE_LABS = [
    "CBC",
    "CMP",
    "BMP",
    "Lactate",
    "Troponin",
    "BNP",
    "D-dimer",
    "CRP",
    "Procalcitonin",
    "PT/INR",
    "ABG",
    "Blood Culture",
    "Urinalysis",
]


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


def _normalize_arrival_time(series: pd.Series) -> pd.Series:
    clean = series.astype(str).str.replace("b'", "", regex=False).str.replace("'", "", regex=False).str.strip()
    has_colon = clean.str.contains(":")
    hhmm = clean.str.replace(":", "", regex=False).str.extract(r"(\d{1,4})", expand=False).fillna("0").str.zfill(4)
    formatted = hhmm.str[:2] + ":" + hhmm.str[2:]
    return pd.Series(np.where(has_colon, clean.str.slice(0, 5), formatted), index=series.index)


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


def _default_risk_score(triage_level: pd.Series, wait_minutes: pd.Series) -> pd.Series:
    triage_component = ((6 - triage_level) / 5).clip(lower=0.0, upper=1.0)
    wait_component = (wait_minutes / 240.0).clip(lower=0.0, upper=1.0)
    return (0.15 + 0.60 * triage_component + 0.25 * wait_component).clip(lower=0.0, upper=1.0)


def _series_with_current(
    current: float,
    *,
    points: int,
    scale: float,
    min_val: float,
    max_val: float,
    rng: np.random.Generator,
) -> list[float]:
    past = current + rng.normal(0.0, scale, points - 1)
    vals = np.append(past, current)
    return np.clip(vals, min_val, max_val).round(2).tolist()


def build_vitals_trend_df(patient_row: pd.Series, points: int = 8) -> pd.DataFrame:
    patient_id = str(patient_row["patient_id"])
    seed = sum((i + 1) * ord(ch) for i, ch in enumerate(patient_id))
    rng = np.random.default_rng(seed)
    time_labels = [f"T-{(points - 1 - i) * 5}m" for i in range(points - 1)] + ["Now"]

    trend = pd.DataFrame(
        {
            "time": time_labels,
            "Pulse": _series_with_current(float(patient_row["pulse"]), points=points, scale=3.8, min_val=40, max_val=180, rng=rng),
            "Resp": _series_with_current(float(patient_row["resp"]), points=points, scale=1.4, min_val=6, max_val=40, rng=rng),
            "SpO2": _series_with_current(float(patient_row["o2_sat"]), points=points, scale=0.9, min_val=80, max_val=100, rng=rng),
            "Systolic BP": _series_with_current(float(patient_row["sys_bp"]), points=points, scale=4.5, min_val=70, max_val=220, rng=rng),
            "Temperature": _series_with_current(float(patient_row["temp"]), points=points, scale=0.2, min_val=94, max_val=106, rng=rng),
        }
    )
    return trend


def normalize_uploaded_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize uploaded CSV into dashboard schema with safe defaults."""
    df = raw_df.copy()

    mapped = pd.DataFrame()

    id_col = _find_col(df, ["patient_id", "patientid", "mrn", "id"])
    name_col = _find_col(df, ["name", "patient_name", "patient"])
    age_col = _find_col(df, ["age"])
    sex_col = _find_col(df, ["gender", "sex"])
    arr_col = _find_col(df, ["arrival_time", "arrival", "arrtime"])
    wait_col = _find_col(df, ["wait_minutes", "waittime", "wait"])
    triage_col = _find_col(df, ["triage_level", "triage", "immedr"])
    complaint_col = _find_col(df, ["chief_complaint", "complaint", "rfv1"])
    location_col = _find_col(df, ["location_bed", "location", "bed"])
    provider_col = _find_col(df, ["assigned_provider", "provider", "doctor", "assigneddoc"])

    mapped["patient_id"] = df[id_col].astype(str) if id_col else pd.Series(range(1, len(df) + 1), dtype=str)
    mapped["name"] = df[name_col].astype(str) if name_col else "Unknown Patient"
    mapped["age"] = _to_num(df[age_col], default=45) if age_col else 45
    mapped["gender"] = df[sex_col].astype(str).str.upper().str[0] if sex_col else "U"
    mapped["arrival_time"] = _normalize_arrival_time(df[arr_col]) if arr_col else "--:--"
    mapped["wait_minutes"] = _to_num(df[wait_col], default=0).clip(lower=0) if wait_col else 0
    mapped["triage_level"] = _to_num(df[triage_col], default=3).clip(lower=1, upper=5) if triage_col else 3
    mapped["chief_complaint"] = df[complaint_col].astype(str) if complaint_col else "General complaint"
    mapped["location_bed"] = df[location_col].astype(str) if location_col else "Waiting"
    mapped["assigned_provider"] = df[provider_col].astype(str) if provider_col else "Unassigned"

    pulse_col = _find_col(df, ["pulse", "hr"])
    temp_col = _find_col(df, ["temp", "temperature", "tempf"])
    resp_col = _find_col(df, ["resp", "respr", "rr"])
    sys_col = _find_col(df, ["sys_bp", "bpsys", "sbp"])
    dia_col = _find_col(df, ["dia_bp", "bpdias", "dbp"])
    o2_col = _find_col(df, ["o2_sat", "o2", "spo2", "popct"])

    mapped["pulse"] = _to_num(df[pulse_col], default=85) if pulse_col else 85
    mapped["temp"] = _to_num(df[temp_col], default=98.2) if temp_col else 98.2
    mapped["resp"] = _to_num(df[resp_col], default=16) if resp_col else 16
    mapped["sys_bp"] = _to_num(df[sys_col], default=120) if sys_col else 120
    mapped["dia_bp"] = _to_num(df[dia_col], default=80) if dia_col else 80
    mapped["o2_sat"] = _to_num(df[o2_col], default=97) if o2_col else 97

    risk_col = _find_col(df, ["risk_score", "icu_risk_probability", "ml_risk_probability", "ml_priority_score", "priorityscore"])
    if risk_col:
        risk_vals = _to_num(df[risk_col], default=0.45)
        if risk_vals.max() > 1.0:
            risk_vals = risk_vals / 10.0
        mapped["risk_score"] = risk_vals.clip(lower=0.0, upper=1.0)
    else:
        mapped["risk_score"] = _default_risk_score(mapped["triage_level"], mapped["wait_minutes"])

    mapped["ml_confidence"] = (
        _to_num(df[_find_col(df, ["ml_confidence", "confidence"])], default=0.75).clip(0.5, 0.99)
        if _find_col(df, ["ml_confidence", "confidence"])
        else 0.75
    )

    mapped["ml_risk_flags"] = (
        df[_find_col(df, ["ml_risk_flags", "riskflags", "risk_flags"])].astype(str)
        if _find_col(df, ["ml_risk_flags", "riskflags", "risk_flags"])
        else "Cardiac concern"
    )

    mapped["predicted_disposition"] = (
        df[_find_col(df, ["predicted_disposition", "disposition", "predicteddisposition"])].astype(str)
        if _find_col(df, ["predicted_disposition", "disposition", "predicteddisposition"])
        else np.select(
            [mapped["risk_score"] >= 0.80, mapped["risk_score"] >= 0.50],
            ["Admit", "Observe"],
            default="Discharge",
        )
    )

    mapped["recommended_next_action"] = (
        df[_find_col(df, ["recommended_next_action", "next_action", "recommendednextaction"])].astype(str)
        if _find_col(df, ["recommended_next_action", "next_action", "recommendednextaction"])
        else np.select(
            [mapped["risk_score"] >= 0.85, mapped["risk_score"] >= 0.65],
            ["Escalate", "Notify Provider"],
            default="Reassess in 15 min",
        )
    )

    mapped["ml_top_factors"] = (
        df[_find_col(df, ["ml_top_factors", "factors", "reason_summary", "top_features"])].astype(str)
        if _find_col(df, ["ml_top_factors", "factors", "reason_summary", "top_features"])
        else "Triage acuity | Arrival context | Vital trend"
    )

    mapped["workflow_stage"] = (
        df[_find_col(df, ["workflow_stage", "stage"])].astype(str)
        if _find_col(df, ["workflow_stage", "stage"])
        else np.select(
            [mapped["risk_score"] >= 0.85, mapped["risk_score"] >= 0.65, mapped["risk_score"] >= 0.40],
            ["Provider", "Triage", "Orders"],
            default="Intake",
        )
    )

    mapped["vitals_summary"] = _build_vitals_summary(mapped)

    for col in EXPECTED_COLUMNS:
        if col not in mapped.columns:
            mapped[col] = ""

    return mapped[EXPECTED_COLUMNS].copy()


def add_risk_bands(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["risk_percentile"] = (out["risk_score"].rank(pct=True, method="average") * 100).round(2)
    out["status_tier"] = out["risk_percentile"].apply(get_status_tier_from_percentile)
    out["status_label"] = out["status_tier"].map(
        {
            "red": "Severe",
            "yellow": "Risky",
            "green": "Stable",
        }
    )
    return out


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


def css_safe_key(raw: str) -> str:
    keep = []
    for ch in str(raw):
        if ch.isalnum() or ch == "_":
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def apply_tile_gradient(tile_key: str, ui: dict) -> None:
    st.markdown(
        f"""
        <style>
        .st-key-{tile_key} {{
            background: {ui['gradient']};
            border: 1px solid #dbeafe;
            border-radius: 14px;
            padding: 10px 12px;
            margin-bottom: 8px;
            position: relative;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_tile_click_overlay(click_key: str) -> None:
    st.markdown(
        f"""
        <style>
        .st-key-{click_key} {{
            height: 0;
            margin: 0;
            padding: 0;
        }}
        .st-key-{click_key} button {{
            position: absolute;
            inset: 0;
            width: 100%;
            height: 100%;
            opacity: 0;
            z-index: 25;
            cursor: pointer;
            border: none;
            background: transparent;
            margin: 0;
            padding: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.dialog("Patient Severity Detail", width="large")
def show_patient_modal(patient_row: pd.Series) -> None:
    ui = get_status_ui(str(patient_row["status_tier"]))
    risk_percentile = float(patient_row["risk_percentile"])
    safe_patient_key = css_safe_key(patient_row["patient_id"])

    st.markdown(f"### {patient_row['name']}")
    st.caption(
        f"{patient_row['patient_id']} • {int(patient_row['age'])}/{patient_row['gender']} • "
        f"Arrived {patient_row['arrival_time']} • Wait {int(patient_row['wait_minutes'])} min"
    )
    st.info(f"Severity: {ui['label']} | Percentile banding: red >85, yellow 65-85, green <=65")

    c1, c2, c3 = st.columns(3)
    c1.metric("Model Score", f"{patient_row['risk_score']:.3f}")
    c2.metric("Percentile", f"{risk_percentile:.1f}th")
    c3.metric("Confidence", f"{patient_row['ml_confidence']:.0%}")
    s1, s2 = st.columns(2)
    with s1.container(border=True):
        st.caption("Chief Complaint")
        st.write(f"**{patient_row['chief_complaint']}**")
    with s2.container(border=True):
        st.caption("Disposition")
        st.write(f"**{patient_row['predicted_disposition']}**")

    s3, s4 = st.columns(2)
    with s3.container(border=True):
        st.caption("Recommended Action")
        st.write(f"**{patient_row['recommended_next_action']}**")
    with s4.container(border=True):
        st.caption("Current Vitals")
        st.write(str(patient_row["vitals_summary"]))

    st.markdown("**Vitals Trend**")
    trend_df = build_vitals_trend_df(patient_row)
    metric_options = ["Pulse", "Resp", "SpO2", "Systolic BP", "Temperature"]
    selected_metric = st.selectbox(
        "Metric",
        options=metric_options,
        index=0,
        key=f"vital_metric_{safe_patient_key}",
    )
    fig_vitals = px.line(trend_df, x="time", y=selected_metric, markers=True)
    fig_vitals.update_layout(
        height=220,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="",
        yaxis_title=selected_metric,
    )
    fig_vitals.update_traces(line_color="#1d4ed8", marker_size=6)
    st.plotly_chart(fig_vitals, use_container_width=True)

    st.markdown("**Model Rationale**")
    st.write(
        f"Score is at the {risk_percentile:.1f}th percentile in current queue. "
        f"Band mapping: red >85, yellow 65-85, green <=65."
    )
    factors = split_top_factors(patient_row["ml_top_factors"])
    if factors:
        chips = "".join([f"<span class='flag-chip'>{f}</span>" for f in factors])
        st.markdown(f"<div class='chip-row'>{chips}</div>", unsafe_allow_html=True)
    flags = [f.strip() for f in str(patient_row["ml_risk_flags"]).split(",") if f.strip()]
    if flags:
        flag_html = "".join([f"<span class='flag-chip'>{f}</span>" for f in flags])
        st.markdown("<div class='chip-title'>Risk Flags</div>" + flag_html, unsafe_allow_html=True)

    st.markdown("**Lab Orders**")
    patient_id = str(patient_row["patient_id"])
    current_orders = st.session_state.patient_lab_orders.get(patient_id, [])
    already_ordered_tests = [entry["test"] for entry in current_orders]
    available_to_order = [lab for lab in AVAILABLE_LABS if lab not in already_ordered_tests]

    st.caption("Click any test below to place an order.")
    quick_cols = st.columns(3)
    if available_to_order:
        for idx, lab in enumerate(available_to_order):
            with quick_cols[idx % 3]:
                if st.button(
                    f"Order {lab}",
                    key=f"order_btn_{safe_patient_key}_{css_safe_key(lab)}",
                    use_container_width=True,
                ):
                    ordered_at = datetime.now().strftime("%H:%M:%S")
                    new_entry = {"test": lab, "status": "Ordered", "ordered_at": ordered_at}
                    st.session_state.patient_lab_orders[patient_id] = current_orders + [new_entry]
                    st.rerun()
    else:
        st.write("All available demo labs are already ordered.")

    refreshed_orders = st.session_state.patient_lab_orders.get(patient_id, [])
    st.caption("Ordered Tests")
    if refreshed_orders:
        for idx, entry in enumerate(refreshed_orders):
            r1, r2, r3 = st.columns([3.0, 1.5, 1.2])
            with r1:
                st.write(f"**{entry['test']}**")
            with r2:
                st.caption(f"{entry['status']} • {entry['ordered_at']}")
            with r3:
                if st.button("Remove", key=f"remove_lab_{safe_patient_key}_{idx}", use_container_width=True):
                    remaining = [e for j, e in enumerate(refreshed_orders) if j != idx]
                    st.session_state.patient_lab_orders[patient_id] = remaining
                    st.rerun()
        if st.button("Clear All Orders", key=f"clear_all_labs_{safe_patient_key}", use_container_width=True):
            st.session_state.patient_lab_orders[patient_id] = []
            st.rerun()
    else:
        st.write("No labs ordered yet. Use quick-order buttons above.")

    st.markdown("**Clinical Notes**")
    existing_note = st.session_state.patient_notes.get(patient_id, "")
    note_key = f"note_input_{safe_patient_key}"
    note_text = st.text_area(
        "Add or update note for this patient",
        value=existing_note,
        height=130,
        key=note_key,
        placeholder="Enter assessment notes, follow-up plan, and handoff details...",
    )
    n1, n2 = st.columns(2)
    with n1:
        if st.button("Save Note", use_container_width=True):
            st.session_state.patient_notes[patient_id] = note_text.strip()
            st.success("Note saved for this patient.")
    with n2:
        if st.button("Clear Note", use_container_width=True):
            st.session_state.patient_notes[patient_id] = ""
            st.rerun()
    st.caption("Decision support only. Use with clinician judgment.")

    if st.button("Close", use_container_width=True):
        st.session_state.modal_patient_id = None
        st.rerun()


# -----------------------------
# Demo Data
# -----------------------------
patients_df = generate_patient_data(55)

patients_df = add_risk_bands(patients_df)

if "modal_patient_id" not in st.session_state:
    st.session_state.modal_patient_id = None
if "patient_notes" not in st.session_state:
    st.session_state.patient_notes = {}
if "patient_lab_orders" not in st.session_state:
    st.session_state.patient_lab_orders = {}

with st.sidebar:
    st.markdown("### Queue Filters")
    search = st.text_input("Search name / ID", "")
    status_filter = st.multiselect("Severity", ["Severe", "Risky", "Stable"], default=["Severe", "Risky", "Stable"])

    dispositions = sorted(patients_df["predicted_disposition"].astype(str).unique().tolist())
    disposition_filter = st.multiselect("Predicted Disposition", dispositions, default=dispositions)

    min_risk = st.slider("Minimum Model Score", min_value=0.00, max_value=1.00, value=0.00, step=0.01)

    st.caption("Banding schema: Red >85th percentile, Yellow 65-85th, Green <=65th.")


# Fixed queue layout settings (max density / max viewport)
compact_tiles = True
max_cards = 30
queue_height = 900


# -----------------------------
# Queue Filtering
# -----------------------------
queue_df = patients_df.copy()
if search.strip():
    query = search.strip().lower()
    queue_df = queue_df[
        queue_df["name"].str.lower().str.contains(query)
        | queue_df["patient_id"].str.lower().str.contains(query)
    ]

queue_df = queue_df[queue_df["status_label"].isin(status_filter)]
queue_df = queue_df[queue_df["predicted_disposition"].astype(str).isin(disposition_filter)]
queue_df = queue_df[queue_df["risk_score"] >= min_risk]
queue_df = queue_df.sort_values(["risk_score", "wait_minutes"], ascending=[False, False])


# -----------------------------
# Top Hospital Summary
# -----------------------------
st.title("ER Clinical Intelligence Dashboard")
st.markdown("<div class='queue-note'>Model output is 0-1 and mapped to percentile-based severity bands for queue coloring.</div>", unsafe_allow_html=True)

red_count = int((patients_df["status_tier"] == "red").sum())
yellow_count = int((patients_df["status_tier"] == "yellow").sum())
green_count = int((patients_df["status_tier"] == "green").sum())
score_85 = float(np.percentile(patients_df["risk_score"], 85)) if len(patients_df) else 0.0
provider_clean = (
    patients_df["assigned_provider"]
    .astype(str)
    .str.strip()
)
team_on_round_today = int(
    provider_clean[
        ~provider_clean.str.lower().isin({"", "unassigned", "none", "nan"})
    ]
    .nunique()
)

k1, k2, k3, k4, k5, k6 = st.columns(6)
with k1:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Patients</div><div style='font-size:1.3rem;font-weight:700'>{len(patients_df)}</div></div>", unsafe_allow_html=True)
with k2:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Severe (Red)</div><div style='font-size:1.3rem;font-weight:700;color:#b91c1c'>{red_count}</div></div>", unsafe_allow_html=True)
with k3:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Risky (Yellow)</div><div style='font-size:1.3rem;font-weight:700;color:#b45309'>{yellow_count}</div></div>", unsafe_allow_html=True)
with k4:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Stable (Green)</div><div style='font-size:1.3rem;font-weight:700;color:#166534'>{green_count}</div></div>", unsafe_allow_html=True)
with k5:
    st.markdown(f"<div class='kpi'><div class='small-muted'>85th Score Cutoff</div><div style='font-size:1.3rem;font-weight:700'>{score_85:.3f}</div></div>", unsafe_allow_html=True)
with k6:
    st.markdown(f"<div class='kpi'><div class='small-muted'>Team On Round Today</div><div style='font-size:1.3rem;font-weight:700'>{team_on_round_today}</div></div>", unsafe_allow_html=True)

st.markdown("---")


# -----------------------------
# Main Content
# -----------------------------
queue_col, dept_col = st.columns([3.8, 1.6], gap="large")

with queue_col:
    st.subheader("Patient Queue")

    if len(queue_df) == 0:
        st.warning("No patients match current filters.")

    with st.container(height=queue_height):
        for _, row in queue_df.head(max_cards).iterrows():
            ui = get_status_ui(str(row["status_tier"]))
            flags = [f.strip() for f in str(row["ml_risk_flags"]).split(",") if f.strip()][:3]
            top_factors = split_top_factors(row["ml_top_factors"])
            safe_patient_key = css_safe_key(row["patient_id"])
            tile_key = f"tile_{safe_patient_key}"
            click_key = f"click_{safe_patient_key}"
            apply_tile_gradient(tile_key, ui)
            apply_tile_click_overlay(click_key)

            with st.container(border=False, key=tile_key):
                if st.button(f"Open {row['patient_id']}", key=click_key, use_container_width=True):
                    st.session_state.modal_patient_id = row["patient_id"]
                if compact_tiles:
                    h1, h2, h3 = st.columns([5.2, 1.5, 1.1])
                    with h1:
                        top_driver = top_factors[0] if top_factors else "N/A"
                        st.markdown(
                            f"""
                            <div style='display:flex;align-items:center;gap:8px'>
                                <div class='avatar-pill'>{get_initials(row['name'])}</div>
                                <div>
                                    <div style='font-weight:700;color:#0f172a'>{row['name']}</div>
                                    <div class='small-muted'>
                                        {row['patient_id']} • {int(row['age'])}/{row['gender']} • Arr {row['arrival_time']} • Wait {int(row['wait_minutes'])}m • {top_driver}
                                    </div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with h2:
                        st.markdown(
                            f"<div style='font-size:1.05rem;font-weight:800;color:#0f172a'>{row['predicted_disposition']}</div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(str(row["location_bed"]))
                    with h3:
                        st.markdown(
                            f"<div style='height:12px;border-radius:999px;background:{ui['dot']};margin-top:8px;margin-bottom:4px'></div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{row['risk_percentile']:.1f}th")

                    m1, m2, m3, m4, m5 = st.columns(5)
                    with m1:
                        st.caption("Score")
                        st.markdown(f"**{row['risk_score']:.3f}**")
                    with m2:
                        st.caption("Confidence")
                        st.markdown(f"**{row['ml_confidence']:.0%}**")
                    with m3:
                        st.caption("Flags")
                        st.markdown(f"**{min(2, len(flags))}**")
                    with m4:
                        st.caption("Wait")
                        st.markdown(f"**{int(row['wait_minutes'])}m**")
                    with m5:
                        st.caption("Complaint")
                        st.markdown(
                            f"<div style='font-size:0.82rem;font-weight:700;color:#0f172a;line-height:1.15'>{row['chief_complaint']}</div>",
                            unsafe_allow_html=True,
                        )
                else:
                    h1, h2, h3 = st.columns([4.4, 1.6, 1.2])
                    with h1:
                        st.markdown(
                            f"""
                            <div style='display:flex;align-items:center;gap:8px'>
                                <div class='avatar-pill'>{get_initials(row['name'])}</div>
                                <div>
                                    <div style='font-weight:700;color:#0f172a'>{row['name']}</div>
                                    <div class='small-muted'>{row['patient_id']} • Arrived {row['arrival_time']} • Wait {int(row['wait_minutes'])} min</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                    with h2:
                        with st.container(border=True):
                            st.caption("Patient")
                            st.markdown(f"<div class='age-sex'>{int(row['age'])}/{row['gender']}</div>", unsafe_allow_html=True)
                            st.caption(str(row["location_bed"]))
                    with h3:
                        st.markdown(
                            f"<div style='height:14px;border-radius:999px;background:{ui['dot']};margin-top:10px;margin-bottom:6px'></div>",
                            unsafe_allow_html=True,
                        )
                        st.caption(f"{row['risk_percentile']:.1f}th")

                    m1, m2, m3, m4 = st.columns(4)
                    with m1.container(border=True):
                        st.metric("Model Score", f"{row['risk_score']:.3f}")
                    with m2.container(border=True):
                        st.metric("Percentile", f"{row['risk_percentile']:.1f}th")
                    with m3.container(border=True):
                        st.caption("Disposition")
                        st.markdown(
                            f"<div style='font-size:0.95rem;font-weight:700;color:#0f172a;line-height:1.2'>{row['predicted_disposition']}</div>",
                            unsafe_allow_html=True,
                        )
                    with m4.container(border=True):
                        st.metric("Confidence", f"{row['ml_confidence']:.0%}")

                    d1, d2 = st.columns(2)
                    with d1.container(border=True):
                        st.caption("Risk Flags")
                        if flags:
                            chips = "".join([f"<span class='flag-chip'>{f}</span>" for f in flags[:2]])
                            st.markdown(chips, unsafe_allow_html=True)
                        else:
                            st.write("None")
                    with d2.container(border=True):
                        st.caption("Top Driver")
                        if top_factors:
                            st.markdown(f"<span class='flag-chip'>{top_factors[0]}</span>", unsafe_allow_html=True)
                        else:
                            st.write("N/A")

    if len(queue_df) > max_cards:
        st.caption(f"Showing top {max_cards} of {len(queue_df)} matching patients in the scrollable queue.")

    with st.expander("Compact Table (optional)", expanded=False):
        table_df = queue_df[
            [
                "patient_id",
                "name",
                "arrival_time",
                "wait_minutes",
                "status_label",
                "risk_score",
                "risk_percentile",
                "predicted_disposition",
                "recommended_next_action",
            ]
        ].copy()
        styled_df = table_df.style.applymap(style_status_gradient, subset=["status_label"])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)

with dept_col:
    st.subheader("Department Snapshot")

    occ = pd.DataFrame(
        {
            "State": ["Occupied", "Available", "Cleaning"],
            "Count": [max(1, int(len(patients_df) * 0.58)), max(1, int(len(patients_df) * 0.27)), max(1, int(len(patients_df) * 0.15))],
        }
    )
    fig_occ = px.pie(occ, names="State", values="Count", hole=0.55)
    fig_occ.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    st.plotly_chart(fig_occ, use_container_width=True)

    top_risks = (
        patients_df["ml_risk_flags"]
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .value_counts()
        .head(3)
    )
    st.markdown("**Top Active Risk Signals**")
    if len(top_risks) > 0:
        for risk, count in top_risks.items():
            st.markdown(f"- {risk}: {count}")

    waits = patients_df["wait_minutes"].clip(lower=0)
    bins = pd.cut(waits, bins=[0, 30, 60, 90, 120, 180, 500], right=False)
    wait_hist = waits.groupby(bins).count().reset_index(name="count")
    wait_hist["bucket"] = wait_hist["wait_minutes"].astype(str)
    fig_wait = px.bar(wait_hist, x="bucket", y="count")
    fig_wait.update_layout(height=220, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Wait bucket", yaxis_title="Patients")
    st.plotly_chart(fig_wait, use_container_width=True)


modal_id = st.session_state.get("modal_patient_id")
if modal_id:
    modal_df = patients_df[patients_df["patient_id"] == modal_id]
    if len(modal_df) > 0:
        show_patient_modal(modal_df.iloc[0])


st.markdown(
    f"<div class='small-muted' style='text-align:center;margin-top:16px'>"
    f"Early warning decision support only — not diagnosis. Last updated {datetime.now().strftime('%H:%M:%S')}"
    f"</div>",
    unsafe_allow_html=True,
)
