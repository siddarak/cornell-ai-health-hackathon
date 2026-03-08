# ER Clinical Intelligence Dashboard

Clinician-facing ED dashboard built with **Streamlit** for triage and model-driven decision support.

## What It Does
- Uses a **card-first patient queue** with gradient status tiers (`Severe`, `Risky`, `Stable`).
- Applies percentile banding from model scores in `0-1`:
  - `red`: highest 15% (`>85th percentile`)
  - `yellow`: `65th-85th percentile`
  - `green`: lowest 65% (`<=65th percentile`)
- Supports uploading a **cherry-picked patient CSV** from your model pipeline.
- Uses a single **Open** action on each patient card to launch a popup with:
  - key patient snapshot
  - exact model score + percentile
  - model rationale/top drivers
  - recommended next action
- Keeps queue height fixed and scrollable so the page does not grow indefinitely.
- Includes optional compact table mode with status-gradient styling.

## Data Expectations
`app.py` normalizes uploaded files into a standard schema. If columns are missing, defaults are applied.

Preferred fields include:
- identifiers: `patient_id`, `name`
- triage context: `age`, `gender`, `arrival_time`, `wait_minutes`, `triage_level`, `chief_complaint`
- vitals: `pulse`, `temp`, `resp`, `sys_bp`, `dia_bp`, `o2_sat`
- ML outputs: `risk_score` (preferred), `ml_risk_flags`, `predicted_disposition`,
  `recommended_next_action`, `ml_top_factors`, `ml_confidence`, `workflow_stage`

If `risk_score` is missing, the app generates a temporary fallback score for UI testing only.

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
