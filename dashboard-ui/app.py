import streamlit as st
import pandas as pd
from mock_data import generate_patient_data
from vitals_logic import get_vitals_status, get_triage_color
import plotly.express as px
from datetime import datetime
import numpy as np

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ER Intel Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM THEME (Safe Palette) ---
st.markdown("""
<style>
    /* Data-Ink Maximization: Neutral UI */
    .stApp {
        background-color: #f8fafc;
        color: #1e3a8a;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* KPI Card Styling */
    .kpi-card {
        background: white;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        border-left: 4px solid #3b82f6;
    }
    
    /* Professional Table (No Zebra) */
    .stDataFrame {
        border: none !important;
    }
    
    /* Utility Styles */
    .small-text { font-size: 0.8rem; color: #64748b; }
    .bold-text { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# --- INITIALIZE STATE ---
if 'selected_patient_idx' not in st.session_state:
    st.session_state.selected_patient_idx = 0
if 'patients_df' not in st.session_state:
    st.session_state.patients_df = generate_patient_data(40)

df = st.session_state.patients_df

# --- 1. TOP BAR (5% Height) ---
col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
with col1:
    st.subheader("🏥 EMERGENCY DEPARTMENT SYSTEM")
with col2:
    st.metric("Beds Occupied", "22/30", delta="-2", delta_color="normal")
with col3:
    st.metric("Avg Wait Time", "42 min", delta="+5", delta_color="inverse")
with col4:
    st.metric("Triage Queue", "8 patients", delta="1", delta_color="inverse")

st.divider()

# --- 2. MAIN AREA (LEFT RAIL | CENTER | RIGHT PANEL) ---
rail_col, center_col, panel_col = st.columns([1.5, 6, 2.5])

# ZONE 4: LEFT RAIL (Filters)
with rail_col:
    st.markdown("### 🔍 FILTERS")
    with st.expander("TRIAGE LEVEL", expanded=True):
        st.checkbox("🔴 Level 1 (Immediate)", value=True)
        st.checkbox("🟠 Level 2 (Emergent)", value=True)
        st.checkbox("🟡 Level 3 (Urgent)", value=True)
        st.checkbox("🟢 Level 4 (Semi-urgent)", value=True)
        
    with st.expander("ACUITY", expanded=False):
        st.slider("Min Acuity Score", 1, 5, 2)
        
    st.markdown("---")
    st.markdown("**Role**: Charge Nurse")
    st.button("⚙️ Settings")

# ZONE 5: CENTER PATIENT LIST (The Workhorse)
with center_col:
    st.markdown("### 📋 PATIENT OVERVIEW (Multi-Patient)")
    
    # Process vitals status for display
    def get_row_status(row):
        v = get_vitals_status(row['Age'], row['Pulse'], row['Temp'], row['Resp'], row['Sys'], row['Dia'], row['O2'])
        return f"{v['icon']} {v['overall']}"

    # Pre-calculate display fields for better st.dataframe integration
    df_display = df.copy()
    df_display['STATUS'] = df_display.apply(get_row_status, axis=1)
    
    # Priority 1: The Workhorse Table
    event = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        column_order=["TriageLevel", "MRN", "Name", "Age", "Gender", "Arrival", "WaitTime", "STATUS", "PriorityScore"],
        column_config={
            "TriageLevel": st.column_config.NumberColumn("TRIAGE", format="%d 🏥", help="1: Immediate, 5: Non-urgent"),
            "WaitTime": st.column_config.ProgressColumn("WAIT (min)", min_value=0, max_value=300, format="%d"),
            "PriorityScore": st.column_config.NumberColumn("AI SCORE", format="%.1f 🔺"),
            "Name": st.column_config.TextColumn("PATIENT NAME", width="medium"),
            "Arrival": "TIME",
            "Age": "AGE",
            "Gender": st.column_config.TextColumn("S"),
        }
    )
    
    if len(event.selection.rows):
        st.session_state.selected_patient_idx = event.selection.rows[0]

# ZONE 6: RIGHT PANEL (Selection Driven Dive)
selected_idx = st.session_state.selected_patient_idx
patient = df.iloc[selected_idx]
v_status = get_vitals_status(patient['Age'], patient['Pulse'], patient['Temp'], patient['Resp'], patient['Sys'], patient['Dia'], patient['O2'])

with panel_col:
    st.markdown("### 📁 PATIENT SUMMARY")
    
    # Patient Banner (Material 3 Density)
    st.markdown(f"""
    <div style="background: #1e293b; color: white; padding: 1rem; border-radius: 4px; border-top: 5px solid {v_status['color']};">
        <div style="font-size: 0.8rem; opacity: 0.8; letter-spacing: 0.05rem;">{patient['MRN']}</div>
        <div style="font-size: 1.2rem; font-weight: bold;">{patient['Name'].upper()}</div>
        <div style="font-size: 0.9rem; margin-bottom: 0.5rem;">{patient['Age']}y {patient['Gender']} | {patient['Location']}</div>
        <div style="background: #ef4444; color: white; display: inline-block; padding: 2px 6px; border-radius: 2px; font-size: 0.7rem; font-weight: bold;">NKDA</div>
        <div style="background: #475569; color: white; display: inline-block; padding: 2px 6px; border-radius: 2px; font-size: 0.7rem; font-weight: bold; margin-left: 4px;">FULL CODE</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Vitals Scan (Small Multiples)
    st.markdown("---")
    v_cols = st.columns(3)
    v_cols[0].metric("Pulse", f"{patient['Pulse']}", delta=None)
    v_cols[1].metric("O2 Sat", f"{patient['O2']}%", delta=None)
    v_cols[2].metric("Temp", f"{patient['Temp']}°", delta=None)

    # AI Rationale (Glass-Box)
    st.markdown("#### 🤖 AI RATIONALE")
    st.info(f"**Flag**: {patient['RiskFlags']} ({patient['PriorityScore']}/10)\n\n**Calculated from**: Abnormal {patient['ChiefComplaint']} pattern and current vital deviations.")
    
    # Intrinsic Controls (One-Gesture)
    st.markdown("#### ⚡ ACTIONS")
    st.button(f"Assign {patient['Name'].split(',')[0]} to Bed", use_container_width=True)
    st.button("Escalate Priority", type="primary", use_container_width=True)

# --- FOOTER ---
st.markdown(f"<div class='small-text' style='text-align: center; margin-top: 2rem;'>ED Intel Dashboard v1.0 | Last updated: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)
