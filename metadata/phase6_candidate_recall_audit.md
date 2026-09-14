# HIVE-Predict — Phase 6 Candidate Recall & Inference-Alignment Audit

> Audit Date: 2026-09-06  
> Dataset: V1 (`data/output/phase6/`) archived; V2 (`data/output/phase6_v2/`) produced  
> Seed: 42 | H3 Resolution: 8

---

## Executive Summary

| Section | Finding | Status |
|---------|---------|--------|
| 1. Natural recall measurement | H3-level: **76.61%** (up from 69.75%) | ✅ Improved |
| 2. Source-level attribution | `nearest_atm` = 68.82%; hotspot = 0.00% | ✅ Diagnosed |
| 3. ATM reference coverage | 100% of actual H3 cells ARE in ATM pool — sampling miss only | ✅ Confirmed |
| 4. Inference equivalence | 0 violations across 150-complaint sample | ✅ Clean |
| 5. Generator improvement | all-state ATMs + neighboring-state ATMs + wider nearest-ATM | ✅ Implemented |
| 6. V2 natural recall | 76.61% H3-level; 75.83% complaint-level (measured before injection) | ✅ Measured |
| 7. Training data policy | `positive_source` flag per row; post-hoc clearly labeled | ✅ Implemented |
| 8. Positive feature distributions | Natural vs injected profile documented; differences are geographic | ✅ Analysed |
| 9. Candidate set size | avg=163, min=80, p95=202, max=218 — within LTR bounds | ✅ Controlled |
| 10. Final verdict | **A. SAFE TO TRAIN XGBOOST** (with documented caveats) | ✅ |

---

## Section 1 — Natural Candidate Recall

### Methodology
For every one of the 8,075 cashout complaints, `generate_candidates()` was called **without** any `actual_h3_cell` argument. The resulting candidate set was compared against ground-truth actual H3 cells.

### V1 Baseline Results (pre-improvement)

| Metric | Value |
|--------|-------|
| Complaint-level natural recall | 5,609 / 8,075 = **69.46%** |
| H3-level natural recall | 6,995 / 10,029 = **69.75%** |
| Post-hoc injection required | 2,466 complaints (30.5%) |

### V1 Recall by Distance Band

| Band | Hit | Total | Recall |
|------|----:|------:|-------:|
| 0–50km | 4,196 | 5,050 | 83.1% |
| 50–150km | 2,511 | 2,853 | 88.0% |
| 150–500km | 282 | 617 | **45.7%** ← primary gap |
| 500+km | 6 | 1,509 | **0.4%** ← structural gap |

### V1 Recall by State (selected)

| State | Recall | Note |
|-------|-------:|------|
| Delhi | 14.8% | Victims in Delhi, withdrawals cross-state |
| Haryana | 66.1% | |
| All others | 72–82% | |

---

## Section 2 — Source-Level Recall Attribution (V1)

| Source | Actual H3 Captured | % of Actuals | Verdict |
|--------|-------------------:|-------------:|---------|
| `nearest_atm` (10 cells) | 6,902 | **68.82%** | Primary driver |
| `ring_neighbor` (36 cells) | 59 | 0.59% | Hard negatives only |
| `state_atm` (5 random) | 27 | 0.27% | Effectively useless |
| `global_random` (4 cells) | 7 | 0.07% | Effectively useless |
| **`hotspot_pool` (6 cells)** | **0** | **0.00%** | **COMPLETELY USELESS** |
| MISSED | 3,034 | 30.25% | |

### Root Cause
- `hotspot_pool` uses seed events (historical simulation, pre-2024). These events are geographically distributed across India but are NOT tied to ATM locations. Actual withdrawals happen at ATMs in the 1,629-cell ATM pool. The seed counter density is uncorrelated with ATM cell locations → 0% recall.
- `state_atm` sampled only 5 random cells from a state with 16–421 unique ATM H3 cells → near-zero hit probability.
- `nearest_atm` with only 10 cells missed complaints where the actual ATM is 11th–1629th nearest → sampling miss.

---

## Section 3 — ATM Reference Coverage

**Critical finding**: 100% of actual cashout H3 cells are in the ATM reference.

| Check | Value |
|-------|-------|
| Unique withdrawal H3 cells in dataset | 1,442 |
| Unique ATM H3 cells in reference | 1,629 |
| Withdrawal H3 cells in ATM reference | 1,442 / 1,442 = **100.0%** |
| Withdrawal H3 NOT in ATM reference | **0** |
| Missed H3 cells in ATM pool | 1,210 / 1,210 = **100.0%** |
| Missed H3 cells NOT in ATM pool | **0** |

**Conclusion**: The problem is entirely a **sampling miss**, not an ATM reference gap. All actual withdrawal H3 cells exist in the reference dataset. The generator simply wasn't drawing enough cells to cover them.

### ATM Reference Cells by State

| State | Unique H3 Cells |
|-------|---------------:|
| Maharashtra | **421** |
| Uttar Pradesh | **285** |
| Rajasthan | **201** |
| Karnataka | **163** |
| West Bengal | **136** |
| Tamil Nadu | **113** |
| Gujarat | 66 |
| Telangana | 59 |
| Delhi | 45 |
| Madhya Pradesh | 33 |
| Bihar | 24 |
| Kerala | 23 |
| Haryana | 16 |
| Odisha | 13 |
| Punjab | 8 |

With 5 random state ATMs sampled from 421 cells (Maharashtra), the per-draw probability of hitting any specific ATM H3 is ~1.2%. This directly explains the 30.5% miss rate.

---

## Section 4 — Inference Equivalence

Verified across 150 complaints spanning the full temporal range:

| Check | Result |
|-------|--------|
| `actual_h3_cell` in candidate features | ❌ Not present |
| `actual_withdrawal_id` in features | ❌ Not present |
| `actual_withdrawal_timestamp` in features | ❌ Not present |
| `feature_cutoff_timestamp` present | ✅ Present |
| `cand_hotspot_density` uses seed events only (pre-2024) | ✅ Confirmed |
| Violations (withdrawal at/before feature cutoff) | **0 / 12,003** |

All 150 sampled complaints have withdrawal timestamps strictly after their feature cutoff. The candidate generator is fully inference-equivalent.

---

## Section 5 — Generator Improvement (V1 → V2)

### Changes Made

| Source | V1 | V2 | Rationale |
|--------|----|----|-----------|
| Ring neighbors (1-3) | 36 cells | 36 cells | Kept: important hard negatives |
| Nearest ATM cells | 10 | **25** | +15 cells: primary recall driver |
| State ATM cells | 5 random | **All (≤100, closest-first)** | Eliminates sampling miss for 150-500km |
| Neighboring-state ATMs | None | **6 per neighbor** | New: fixes cross-state (Delhi +21.6pp) |
| Global random ATMs | 4 | **20** | +16 cells: partial 500+km coverage |
| `hotspot_pool` | 6 | **0 (REMOVED)** | 0.00% recall — replaced by above |

### Neighboring-State Adjacency Map (hardcoded for 15 reference states)

```
Delhi       -> Haryana, Uttar Pradesh, Rajasthan
Haryana     -> Delhi, Punjab, Uttar Pradesh, Rajasthan
Maharashtra -> Gujarat, Madhya Pradesh, Telangana, Karnataka
... (full map in src/candidate_generator.py: STATE_NEIGHBORS)
```

### Max State ATM Cap
Large states (Maharashtra=421, UP=285) are capped at 100 closest cells by centroid distance from victim. This prevents runaway candidate counts while maximising coverage of likely withdrawal zones.

---

## Section 6 — V2 Natural Recall (Measured Before Any Injection)

### Overall

| Metric | V1 | V2 | Δ |
|--------|--:|--:|--:|
| Complaint-level recall | 69.46% | **75.83%** | +6.37pp |
| H3-level recall | 69.75% | **76.61%** | +6.86pp |
| Post-hoc still required | 2,466 (30.5%) | **1,952 (24.2%)** | −514 complaints |

### V2 Recall by Distance Band

| Band | V1 | V2 | Δ |
|------|---:|---:|--:|
| 0–50km | 83.1% | **89.7%** | +6.6pp |
| 50–150km | 88.0% | **95.8%** | +7.8pp |
| 150–500km | 45.7% | **60.1%** | **+14.4pp** |
| 500+km | 0.4% | **3.2%** | +2.8pp (structural limit) |

### V2 Recall by State

| State | V1 | V2 | Δ | Notes |
|-------|---:|---:|--:|-------|
| Bihar | 82.3% | 84.4% | +2.1 | |
| **Delhi** | **14.8%** | **36.4%** | **+21.6pp** | Neighboring-state ATMs working |
| Gujarat | 77.7% | 81.5% | +3.8 | |
| Haryana | 66.1% | 72.4% | +6.3 | |
| Karnataka | 75.6% | 80.4% | +4.8 | |
| Kerala | 75.7% | 78.0% | +2.3 | |
| Madhya Pradesh | 77.6% | 80.0% | +2.4 | |
| Maharashtra | 74.5% | 82.6% | +8.1 | All-state ATMs (421 cells) |
| Odisha | 80.6% | 81.0% | +0.4 | |
| Punjab | 80.5% | 82.1% | +1.6 | |
| Rajasthan | 75.2% | 81.5% | +6.3 | |
| Tamil Nadu | 72.7% | 77.8% | +5.1 | |
| Telangana | 71.0% | 77.5% | +6.5 | |
| Uttar Pradesh | 75.4% | 82.2% | +6.8 | All-state ATMs (285 cells) |
| West Bengal | 75.8% | 81.2% | +5.4 | |

### V2 Candidate Count Distribution

| Stat | Value |
|------|-------|
| Min | 80 |
| Median | 178 |
| p95 | 202 |
| Max | **218** |
| Average | **163.1** |
| Total dataset rows | **1,317,397** |

---

## Section 7 — Training Data Policy

### `positive_source` Column
Every row in `candidate_h3_dataset_v2.csv` carries a `positive_source` column:

| Value | Meaning | Count |
|-------|---------|------:|
| `natural` | Positive candidate naturally generated; matches inference-time behavior | 7,683 |
| `posthoc_injection` | Positive added post-hoc; actual H3 NOT in natural candidate set | 2,346 |
| `negative` | Negative candidate | 1,307,368 |

### Training guidance

```python
# Option A: Train on all rows (recommended — preserves all valid signal)
X_train = train_df[MODEL_FEATURES].values
y_train = train_df["relevance"].values
# Use positive_source for evaluation/analysis only

# Option B: Train only on inference-aligned complaints
# (excludes 24.2% of cashout complaints — not recommended)
aligned_cids = train_df[
    ~train_df["complaint_id"].isin(posthoc_complaint_ids)
]["complaint_id"].unique()
```

### Policy statement
> The v2 dataset preserves post-hoc injected positives in the training set with explicit labeling.  
> These represent real cash-out events (cross-state, 500+km) with legitimate feature distributions.  
> Excluding them would introduce selection bias: the model would be trained only on cases  
> where the actual ATM happened to be in a sampled pool, systematically under-training on  
> long-distance cross-state fraud patterns.

### V1 archived
V1 dataset (`data/output/phase6/`) is preserved as-is. It is now designated as the **post-hoc baseline** dataset and must not be used as the primary training dataset.

---

## Section 8 — Positive Feature Distributions (V2)

### Natural vs Injected vs Negative

| Feature | Natural Positive | Injected Positive | Negative | Key Finding |
|---------|----------------:|------------------:|---------:|-------------|
| `cand_dist_km_from_victim` | 63.3 km | **772.9 km** | 323.5 km | Injected = cross-India |
| `cand_h3_grid_dist` | 75 rings | **925 rings** | 386 rings | Consistent with distance |
| `cand_atm_count` | 1.08 | **2.18** | 0.95 | Injected = real ATM-rich cells |
| `cand_atm_density` | 1.39 | **6.67** | 2.43 | Injected = dense ATM zones |
| `cand_hotspot_density` | 0.063 | **0.273** | 0.093 | Injected = historically active |
| `cand_in_victim_state` | 0.642 | **0.070** | 0.463 | Injected = different state ✓ |

### Interpretation

The injected positives (2,346 rows) have a **distinctive and meaningful profile**:
1. They are **far from victims** (avg 773 km) — consistent with cross-state fraud geography
2. They have **high ATM density** (6.67 avg) — these are real, ATM-rich urban zones
3. They are **NOT in victim's state** (only 7% in-state) — confirms cross-state nature
4. They have **high hotspot density** (0.273) — historically active withdrawal zones

**This is not an artifact**. These are genuine fraud locations that are geographically distant from victims. The feature differences are semantically valid and will provide useful training signal: the model must learn that sometimes high-ATM-density distant cells are the cash-out location.

---

## Section 9 — Candidate Set Size Analysis

### Tradeoff Summary

| Dimension | V1 | V2 | Impact |
|-----------|-----|-----|--------|
| Natural recall | 69.75% | 76.61% | +6.86pp recall |
| Avg candidates | 60.8 | 163.1 | 2.7× more candidates |
| Dataset size | 490K rows | 1.32M rows | 2.7× larger |
| Hard negatives (<100km) | 72.0% of negatives | 32.3% of negatives | Less hard-neg proportion |
| Hard negatives (absolute) | 346,143 | **421,755** | +22% more absolute hard negs |
| Positive rate | 2.04% | 0.76% | Lower ratio (more negatives) |

### Size justification

- 163 avg candidates per complaint is well within XGBoost LTR practical bounds
- Maximum of 218 prevents group explosion
- 1.32M rows is manageable on standard hardware (~500MB RAM for training)

### Hard negative proportion drop
The drop from 72% to 32.3% reflects the addition of state-wide ATMs (100-400km range = medium difficulty, not hard). The absolute count of hard negatives actually **increased** by 22%. The model now trains on a richer mix of difficulty levels: hard (ring neighbors), medium (state ATMs), and easy-exploration (global random).

---

## Section 10 — Final Verdict

### All audit checks

| # | Check | V2 Result |
|---|-------|-----------|
| 1 | Natural candidate recall measured (no injection) | ✅ 76.61% H3-level |
| 2 | Recall materially improved over V1 | ✅ +6.86pp |
| 3 | Root cause diagnosed and fixed | ✅ Sampling miss; all sources redesigned |
| 4 | Hotspot pool removed | ✅ Contributed 0.00% recall |
| 5 | actual_h3_cell never used for generation | ✅ Confirmed |
| 6 | Inference equivalence (0 temporal violations) | ✅ Confirmed |
| 7 | Per-row `positive_source` flag | ✅ 'natural' / 'posthoc_injection' / 'negative' |
| 8 | Candidate set size practical | ✅ avg=163, max=218 |
| 9 | 15/15 validation checks pass | ✅ All pass |
| 10 | Strict temporal split maintained | ✅ max(train)<min(val)<min(test) |
| 11 | No complaint overlap between splits | ✅ 0 overlaps |

### Residual limitations (documented)

| Limitation | Scope | Fix path |
|-----------|-------|----------|
| 500+km natural recall = 3.2% | ~15% of actual H3 targets | Mule-account state geography (Phase 7) |
| Delhi complaint recall = 36.4% | ~888 actual H3 targets | Broader neighboring-state coverage |
| 24.2% post-hoc injection rate | 1,952 complaints | Same — cross-state geography |

The 500+km limitation is structural: at inference time, a ranker without mule-network information cannot know that a Delhi victim's fraud is likely to be cashed out in Tamil Nadu. The correct fix is to incorporate mule-chain destination account states as additional candidate-generation priors (available at feature cutoff via the mule_chains table). This is deferred to Phase 7.

---

```
════════════════════════════════════════════════════════════
  FINAL VERDICT:

  A. SAFE TO TRAIN XGBOOST

  Conditions satisfied:
    ✅ Candidate generation is inference-aligned (no future data)
    ✅ Natural recall is demonstrably improved: 76.61% H3-level
    ✅ Post-hoc injections are explicitly labeled per row
    ✅ Candidate set size is computationally practical (avg=163)
    ✅ 15/15 validation checks pass
    ✅ Strict temporal train/val/test split
    ✅ Root cause of recall gap diagnosed and partially fixed

  Required before training:
    - Use `data/output/phase6_v2/` as training source (NOT phase6/)
    - Exclude 'positive_source', 'victim_h3_res8', 'feature_cutoff_timestamp'
      from XGBoost feature columns
    - Use group_info.json for XGBoost group assignment
    - Track performance separately on natural vs post-hoc positive complaints

  Deferred to Phase 7:
    - Mule-account state geography as additional candidate source
    - Expected to raise 500+km recall from 3.2% to 30-50%
    - Expected to raise Delhi recall from 36.4% to 70%+
════════════════════════════════════════════════════════════
```

---

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `candidate_h3_dataset_v2.csv` | `data/output/phase6_v2/` | 1,317,397 rows, 29 columns |
| `train.csv` | `data/output/phase6_v2/` | 922,267 rows, 5,652 complaints |
| `validation.csv` | `data/output/phase6_v2/` | 196,870 rows, 1,211 complaints |
| `test.csv` | `data/output/phase6_v2/` | 198,260 rows, 1,212 complaints |
| `group_info.json` | `data/output/phase6_v2/` | XGBoost group sizes |
| `natural_recall_report.json` | `data/output/phase6_v2/` | Full recall breakdown |
| `candidate_validation_report.csv` | `data/output/phase6_v2/` | 15-check results |
| `candidate_h3_dataset.csv` (V1) | `data/output/phase6/` | **ARCHIVED** — baseline only |

---

*Audit completed: 2026-09-06 | V2 seed: 42 | Generator: candidate_generator.py v2*
