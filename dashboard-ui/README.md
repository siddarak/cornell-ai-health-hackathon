# ER Clinical Intelligence Dashboard

Clinician-facing ED dashboard built with **Streamlit** for triage and model-driven decision support.

## What It Does
- Uses a **card-first patient queue** with gradient status tiers (`Severe`, `Risky`, `Stable`).
- Supports uploading a **cherry-picked patient CSV** from your model pipeline.
- Shows a fixed right panel with:
  - patient banner and workflow stage
  - vitals and complaint summary
  - ML risk summary, confidence, and top drivers
  - quick actions and ER-wide analytics
- Includes optional compact table mode with status-gradient styling.

## Data Expectations
`app.py` normalizes uploaded files into a standard schema. If columns are missing, defaults are applied.

Preferred fields include:
- identifiers: `patient_id`, `name`
- triage context: `age`, `gender`, `arrival_time`, `wait_minutes`, `triage_level`, `chief_complaint`
- vitals: `pulse`, `temp`, `resp`, `sys_bp`, `dia_bp`, `o2_sat`
- ML outputs: `ml_priority_score`, `ml_risk_flags`, `predicted_disposition`,
  `recommended_next_action`, `ml_top_factors`, `ml_confidence`, `workflow_stage`

## Run Locally
1. Install dependencies:
   ```bash
   pip install streamlit pandas plotly numpy
   ```
2. Start dashboard:
   ```bash
   cd dashboard-ui
   streamlit run app.py
   ```

## Files
- `app.py`: Main UI, uploaded-data normalization, queue/panel rendering.
- `vitals_logic.py`: Status-tier engine + UI color/gradient tokens.
- `mock_data.py`: Synthetic patient generator with ML-style output fields.
