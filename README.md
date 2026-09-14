# PURVADRISHTI
## Predictive Intelligence for Cyber-Fraud Cash-Out Response

> "See the risk before the cash-out."

PURVADRISHTI is a human-in-the-loop decision-support prototype for predicting high-risk H3 geographic zones where cyber-fraud proceeds may be withdrawn. It uses a frozen Phase 7 XGBoost Learning-to-Rank model trained on synthetic cyber-fraud data.

---

## Quick Start

```powershell
# From project root
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

| URL | Description |
|-----|-------------|
| http://localhost:8000 | PURVADRISHTI Dashboard |
| http://localhost:8000/docs | Swagger UI |
| http://localhost:8000/redoc | ReDoc |

---

## Final Primary Model

```
FINAL PRIMARY MODEL
===================
Model:      Phase 7 XGBoost Learning-to-Rank
Model file: data/output/phase7/model.ubj
Features:   22
Objective:  rank:ndcg

Official Held-Out Test Performance (Phase 6 V2 test split):
  T3 Hit@5:   65.26%
  T3 MRR:     0.4971
  T3 NDCG@5:  0.5135

Phase 8:  Experimental mule-network features — no verified end-to-end improvement
Phase 9:  Experimental KDE — no demonstrated utility on synthetic dataset
Final scoring: Phase 7 XGBoost ONLY
KDE is NOT used in final scoring.
```

---

## Architecture

```
app/
  main.py              ← FastAPI app + startup (loads model once)
  api/
    routes.py          ← All API endpoints
    schemas.py         ← Pydantic request/response models
  services/
    predictor.py       ← PredictionService (wraps HIVEPredictor)
  core/
    config.py          ← Constants, model metadata, alert thresholds

src/
  phase10_pipeline.py  ← HIVEPredictor (core prediction logic, unchanged)
  candidate_generator.py
  feature_engineering.py
  ...

templates/
  index.html           ← PURVADRISHTI dashboard (professional light theme)

data/output/
  phase7/model.ubj     ← FROZEN primary model (do not modify)
  phase6_v2/           ← Pre-computed candidate features
  phase9/              ← Experimental KDE outputs (not used)
  phase10/             ← Validation outputs
```

---

## API Endpoints

### `GET /api/health`
```json
{"status": "ok", "model": "Phase 7 XGBoost", "model_frozen": true, "model_loaded": true}
```

### `GET /api/model-info`
Returns frozen model metadata, official test metrics, and KDE note.

### `GET /api/complaints`
Returns list of available demo complaint IDs (from Phase 6 V2 test/validation split).

### `POST /api/predict`
**Request:**
```json
{"complaint_id": "CMP00007160"}
```
**Response:** Full prediction including top-5 H3 zones, relative model scores, ATM candidates, SHAP explanation, alert priority, and (separately) validation ground truth.

---

## Prediction Flow

```
Complaint ID
  ↓
Feature lookup (pre-computed Phase 6 V2 snapshot)
  ↓
Phase 7 XGBoost LTR scoring (22 features)
  ↓
Top-5 ranked H3 zones
  ↓
H3 centroid + boundary coordinates (Leaflet polygons)
  ↓
Nearby ATM mapping (by proximity to H3 centroid)
  ↓
SHAP explanation (XGBoost native pred_contribs)
  ↓
Alert priority (UI heuristic — see below)
  ↓
Dashboard
```

---

## Alert Priority — Implemented Rule

**Rule type:** UI prioritization heuristic. NOT model probability. NOT calibrated confidence.

```
score_margin = (rank1_score - rank2_score) / rank1_score
HIGH:   margin > 0.20  (strong score separation)
MEDIUM: 0.05 < margin <= 0.20
LOW:    margin <= 0.05
```

Documented as: *"Prototype decision-support priority — human review required."*

---

## H3 Zone Output

Each of the top-5 predictions includes:
- H3 cell ID
- **Relative Model Score** (0–100, normalized within complaint — NOT probability)
- Distance from victim location
- ATM count in zone
- H3 centroid (lat/lon) + boundary polygon
- SHAP explanation

> **Important:** Do not interpret the Relative Model Score as a probability or calibrated confidence. It is a normalized ranking score for display only.

---

## ATM Mapping

ATMs are mapped **after** H3 prediction, not predicted directly:
1. Rank-1 H3 zone selected
2. ATM reference searched by proximity to H3 centroid (ring-1, ring-2 expansion)
3. Nearest ATMs returned, ranked by distance to centroid

> ATM candidates are not direct model prediction targets.

---

## SHAP Explanation

XGBoost native `pred_contribs` used to compute SHAP values for each top-5 candidate.
Dashboard shows: top positive and negative contributors.

> "SHAP explains the model's ranking for the selected candidate; it is not causal evidence of criminal activity."

---

## Official Evaluation Metrics

| Metric | Value |
|--------|------:|
| T3 Hit@5 | 65.26% |
| T3 MRR | 0.4971 |
| T3 NDCG@5 | 0.5135 |

**Demo subset result: 9/10 Top-5 hits.**
This demo subset result is NOT the official model accuracy.

---

## Limitations

- Demonstration uses **synthetic data**. Production requires authorized, governed real data.
- Predictions are **candidate H3 zones**, not exact locations.
- Relative Model Score is **not a probability**.
- KDE spatial component was evaluated but provided no demonstrated utility on this synthetic dataset.
- Alert priority is a **UI heuristic**, not an operational threshold.

---

## Disclaimer

> Decision-support prototype. Predictions identify ranked high-risk H3 zones for human review. They are not guarantees of cash-out location and do not trigger automatic enforcement, account freezing or other action. Human-in-the-loop review is required before any operational action.
