# ER Intel Dashboard

This is a high-density, clinician-centric Emergency Department dashboard built with **Streamlit**.

## Features
- **5-Zone Layout**: Top Bar stats, Left Rail filters, Center Patient Table, and Right Detail Panel.
- **Clinical Logic**: Vitals validation based on NHAMCS 2022 dataset analysis.
- **Design Principles**: Data-Ink Maximization, Progressive Disclosure, and Redundant Encoding.
- **AI Insights**: Glass-Box AI rationale for patient priority scores.

## How to Run locally
1. Ensure you have the dependencies installed:
   ```bash
   pip install streamlit pandas plotly numpy
   ```
2. Navigate to this directory:
   ```bash
   cd dashboard-ui
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## File Structure
- `app.py`: Main dashboard interface.
- `vitals_logic.py`: Clinical range validation and triage coloring.
- `mock_data.py`: Generates realistic patient records for demonstration.
