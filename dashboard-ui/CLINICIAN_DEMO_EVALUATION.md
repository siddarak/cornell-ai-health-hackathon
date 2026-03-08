# Clinician-Style Demo Evaluation (ED Dashboard)

Date: 2026-03-08
Scope reviewed: queue view, tile density, color-based severity, popup rationale, and notes workflow.

## Participants (Simulated Reviewer Roles)
- ED attending physician perspective
- Triage nurse perspective
- Charge nurse / flow coordinator perspective
- ED operations perspective

## What Was Discussed
1. Clinical usefulness of percentile severity bands.
- Red/yellow/green is fast to parse and helps prioritize attention.
- Percentile framing is useful for relative queue prioritization, but does not equal diagnosis.

2. Cognitive load and scanability.
- Compact tile layout is much better for high census periods.
- Gradient tile severity helps avoid reading extra text.
- Top-bar KPIs support quick situational awareness.

3. Explainability and trust.
- Popup with model score + percentile + top drivers is enough for demo trust-building.
- Showing rationale in one click is preferred over separate "what" and "why" interactions.

4. Workflow fit.
- Per-patient notes in popup are valuable for handoff and tracking clinician thinking.
- Team-on-round KPI helps staff orient to current coverage.

5. Safety and deployment concerns.
- Current model output should remain decision support only.
- No evidence yet of prospective validation, calibration monitoring, or harm audit.
- Must avoid claiming diagnostic certainty or replacement of triage clinician judgment.

## Determinations
- Demo readiness for hackathon storytelling: **High**.
- Real-world clinical readiness: **Not production-ready** without validation and governance.
- Most useful immediate value: **queue prioritization, shared team awareness, and faster handoff context**.

## Recommended Demo Messaging
Use this line in the demo:
- "This system prioritizes attention, not diagnosis. It highlights who needs reassessment sooner and why."

## Minimum Next Steps Before Any Real Pilot
1. Validate model discrimination/calibration on held-out real ED data.
2. Define escalation thresholds with clinical leadership and measure alert burden.
3. Log all model outputs and clinician overrides for retrospective safety review.
4. Add policy language for intended use, contraindications, and downtime behavior.

## Suggested 2-Minute Demo Flow
1. Show top KPIs and red/yellow/green queue distribution.
2. Click a red tile to open rationale and review score percentile + drivers.
3. Add a clinical note in popup to demonstrate handoff utility.
4. Close by emphasizing decision-support positioning and safety boundaries.
