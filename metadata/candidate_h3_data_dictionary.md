# HIVE-Predict — Candidate H3 Ranking Dataset Data Dictionary

> Phase 6 | Version 1.0 | Seed: 42 | Resolution: H3 res-8

---

## Overview

The candidate-H3 ranking dataset supports XGBoost Learning-to-Rank (LTR) training for HIVE-Predict.

**ML Task**: Given a fraud complaint and all information observable at the feature cutoff, rank candidate H3 cells by likelihood of being the future cash-out location.

**Dataset structure**: One row per `(complaint_id, candidate_h3_cell)` pair.
Each complaint forms one **ranking group** (query). The model ranks candidates within each group.

---

## Files

| File | Rows | Description |
|------|------|-------------|
| `candidate_h3_dataset.csv` | full | All splits combined |
| `train.csv` | ~70% | Training split (earlier complaints) |
| `validation.csv` | ~15% | Validation split |
| `test.csv` | ~15% | Test split (latest complaints) |
| `group_info.json` | — | XGBoost group sizes per split |
| `candidate_validation_report.csv` | 15 rows | Automated validation results |
| `candidate_generation_report.md` | — | Full methodology report |

---

## Schema

### Non-Model Identifier Columns

> ⚠️ These columns MUST NOT be passed to XGBoost as features.
> Use for grouping, filtering, and evaluation only.

| Column | Type | Description |
|--------|------|-------------|
| `complaint_id` | string | FK → complaints.complaint_id. XGBoost group key. |
| `candidate_h3_cell` | string | H3 cell index at res-8. Candidate location. |
| `split` | string | `train` / `validation` / `test` |
| `victim_h3_res8` | string | Victim location H3 cell. Reference only. |
| `feature_cutoff_timestamp` | ISO datetime | Complaint registration time = feature cutoff. |

---

### Target Column

| Column | Type | Values | Description |
|--------|------|--------|-------------|
| `relevance` | int | 0 or 1 | **1** = this candidate is an actual future cash-out H3 cell. **0** = negative candidate. |

**Relevance assignment rule**:
- `relevance = 1` if `candidate_h3_cell ∈ actual_h3_set(complaint_id)`
- `actual_h3_set` = set of unique `actual_h3_cell` values from `cashout_labels.csv` for this complaint
- Multiple withdrawals at the same H3 → **one positive row** (deduplicated by set)
- NO_CASHOUT complaints → **excluded** from this dataset (no fake positive)

---

### Complaint-Level Feature Columns (15 features)

Same value for all candidate rows of the same complaint.
All features are observable at `feature_cutoff_timestamp`.

| Column | Type | Source | Description |
|--------|------|--------|-------------|
| `fraud_amount` | float | complaints | Fraud amount in INR |
| `amount_log` | float | complaints | log1p(fraud_amount) |
| `hour` | int | complaints | Hour of incident (0–23) |
| `day_of_week` | int | complaints | Day of week: 0=Mon, 6=Sun |
| `is_weekend` | int | complaints | 1 if Saturday or Sunday |
| `is_night` | int | complaints | 1 if hour ∈ {22, 23, 0, 1, 2, 3, 4, 5} |
| `fraud_type_encoded` | int | complaints | Fraud type code: 0=DIGITAL_PAYMENT, 1=INVESTMENT, 2=DIGITAL_ARREST, 3=PHISHING, 4=OTHER |
| `mule_chain_depth` | int | mule_chains (obs. at cutoff) | Number of mule hops observed by feature cutoff |
| `mule_velocity` | float | mule_chains (obs. at cutoff) | Mule hops per minute (0 if <2 hops) |
| `amount_velocity` | float | mule_chains (obs. at cutoff) | INR per minute through mule chain; min-time floor=60s |
| `distance_from_victim` | float | victim coords + atm_reference | Haversine km from victim to state average ATM location |
| `historical_hotspot_density` | float | seed events + past cases | Density of historical withdrawals in victim H3 ring-6 neighbourhood |
| `atm_density` | int | atm_reference | ATMs in victim H3 + ring-1 (7 cells) |
| `complaint_cluster` | int | rolling DBSCAN | Spatiotemporal complaint cluster ID (–1 = noise) |
| `time_since_transaction` | float | complaint timestamps | Seconds from incident_datetime to feature_cutoff_timestamp |

---

### Candidate-Level Feature Columns (7 features)

Vary per candidate. All use only information observable at feature cutoff.
No future withdrawal data is used in any of these features.

| Column | Type | Source | Description | Leakage risk |
|--------|------|--------|-------------|-------------|
| `cand_dist_km_from_victim` | float | h3.cell_to_latlng + haversine | Haversine km from victim centroid to candidate H3 centroid | None |
| `cand_h3_grid_dist` | int | h3.grid_distance | H3 grid ring distance from victim_h3 to candidate_h3; capped at 9999 | None |
| `cand_atm_count` | int | atm_reference | Number of ATMs whose h3_cell_res8 == candidate_h3_cell | None |
| `cand_atm_density` | int | atm_reference | ATMs in candidate_h3 + ring-1 neighbours (7 cells total) | None |
| `cand_hotspot_density` | float | seed events (pre-2024-01-01) | Mean withdrawal density in candidate ring-3 neighbourhood (37 cells); uses seed counter only | None |
| `cand_in_victim_state` | int | atm_reference state labels | 1 if any ATM in candidate ring-1 is labelled with victim's state | None |
| `cand_is_victim_h3` | int | h3 equality | 1 if candidate_h3_cell == victim_h3_res8; almost always 0 | None |

---

## XGBoost Usage Guide

```python
import pandas as pd, json, xgboost as xgb

MODEL_FEATURES = [
    # Complaint-level
    "fraud_amount", "amount_log", "hour", "day_of_week", "is_weekend",
    "is_night", "fraud_type_encoded", "mule_chain_depth", "mule_velocity",
    "amount_velocity", "distance_from_victim", "historical_hotspot_density",
    "atm_density", "complaint_cluster", "time_since_transaction",
    # Candidate-level
    "cand_dist_km_from_victim", "cand_h3_grid_dist", "cand_atm_count",
    "cand_atm_density", "cand_hotspot_density", "cand_in_victim_state",
    "cand_is_victim_h3",
]

train = pd.read_csv("data/output/phase6/train.csv")
with open("data/output/phase6/group_info.json") as f:
    ginfo = json.load(f)

X_train = train[MODEL_FEATURES].values
y_train = train["relevance"].values
groups  = [g["group_size"] for g in ginfo["train"]]

dtrain = xgb.DMatrix(X_train, label=y_train,
                      feature_names=MODEL_FEATURES)
dtrain.set_group(groups)
```

> **Never pass** `complaint_id`, `candidate_h3_cell`, `split`, `victim_h3_res8`,
> or `feature_cutoff_timestamp` to XGBoost.

---

## NO_CASHOUT Alert Threshold Calibration

The 1,925 NO_CASHOUT complaints are excluded from the ranking dataset but can be used
for alert threshold calibration at inference time:

1. Apply trained ranker to each NO_CASHOUT complaint with a full candidate set
2. Record `max_score` (highest ranking score across all candidates)
3. Plot ROC: `cashout_complaints max_score` vs `no_cashout_complaints max_score`
4. Choose alert threshold = score above which HIVE-Predict raises an alert

This requires no modification to the ranking model — only post-hoc threshold tuning.

---

## Temporal Split Boundaries

| Split | Complaints | Cutoff |
|-------|-----------|--------|
| Train | ~5,652 | Earlier 70% by feature_cutoff_timestamp |
| Validation | ~1,211 | Middle 15% |
| Test | ~1,212 | Latest 15% |

**Split unit**: `complaint_id` — a complaint's rows NEVER span multiple splits.

---

*Generated by HIVE-Predict Phase 6 pipeline. Seed=42. Do not manually edit.*
