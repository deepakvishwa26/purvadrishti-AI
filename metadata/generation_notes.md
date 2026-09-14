# HIVE-Predict Synthetic Data Generation Notes

## Overview

This synthetic dataset was generated for the HIVE-Predict ML system (Project GoldenHour, SIH Problem Statement 26184).

It provides connected, temporally consistent, geographically realistic, and ML-safe synthetic fraud case data for training and evaluating the HIVE-Predict cash-out location prediction model.

## Key Design Decisions

### Connected Event Model
Each synthetic case generates a full event chain:
```
Complaint → Victim Account → Initial Transaction → Mule Chain → Cash-out → Withdrawal → ATM → H3 Cell → Features → Labels
```

This ensures all foreign keys are valid and features can be derived with complete data lineage.

### Temporal Leakage Prevention
- Every feature snapshot has a `feature_cutoff_timestamp` set to `complaint_registered_at`.
- Only data available at or before the cutoff is used for feature computation.
- Future withdrawal information is NEVER included in features.
- Historical hotspot density only considers withdrawals from EARLIER cases.

### Financial Consistency
- Amounts flow from victim through mule chain with configurable friction (2-10% per hop).
- Withdrawal amounts never exceed available funds at the cashout account.
- All amounts are positive.

### H3 Implementation
- Uses the official `h3` Python library (not fake strings).
- Resolution 8 for primary prediction, resolution 9 for dense urban refinement.
- H3 cells are derived from real ATM coordinates, ensuring spatial consistency.
- Validated: stored H3 cell == H3(latitude, longitude) for every withdrawal.

### Behavioral Profiles
Six configurable profiles control fraud case characteristics:
1. **FAST_CASHOUT** (30%): Short chain, quick withdrawal
2. **DEEP_MULE_CHAIN** (15%): Many hops, complex laundering
3. **SLOW_CASHOUT** (15%): Delayed withdrawal
4. **NO_CASHOUT** (15%): No withdrawal (negative class for prediction)
5. **MULTIPLE_CASHOUT** (15%): Multiple withdrawal events
6. **COORDINATED_CASE** (10%): Shared characteristics across complaints

### Feature Derivation
All ML features are derived from the event data, not independently generated:
- `hour`, `day_of_week` ← incident timestamp
- `mule_chain_depth` ← count of mule chain records
- `mule_velocity` ← chain timestamps
- `amount_velocity` ← chain amounts and timestamps
- `distance_from_victim` ← Haversine distance
- `historical_hotspot_density` ← past withdrawal spatial density
- `atm_density` ← ATM reference data
- `complaint_cluster` ← DBSCAN clustering

## Synthetic Assumptions

> **IMPORTANT**: The following are SIMULATION ASSUMPTIONS, not official statistics.

1. Fraud type distribution (35% digital payment, 20% investment, etc.)
2. Fraud amount distributions (lognormal parameters per type)
3. Behavioral profile probabilities
4. State-level fraud reporting weights
5. Hourly fraud distribution weights
6. Mule chain friction rates (2-10%)
7. ATM clustering patterns (70% urban, 30% rural)
8. Geographic proximity weighting for ATM selection

## Limitations

1. **Not real data**: This is synthetic data that does not capture real-world fraud behavior complexity.
2. **Simplified geography**: ATM locations are synthetic approximations, not real ATM coordinates.
3. **Simplified network**: Each case is independent (except COORDINATED_CASE profile), whereas real fraud networks may share accounts across cases.
4. **No real calibration**: Without access to real NCRP/CFCFRMS data, distributions are approximations.
5. **Model performance on synthetic data does not guarantee real-world accuracy.**

## Intended Use

This dataset supports Phases 2-5 of HIVE-Predict development:
- Phase 2: Synthetic generator implementation (CURRENT)
- Phase 3: 10,000-case generation (CURRENT)
- Phase 4: Data integrity validation (CURRENT)
- Phase 5: Feature engineering validation (CURRENT)
- Phase 6: Candidate-H3 training dataset preparation (NEXT)

## Ethical Considerations

- All PII is synthetic (no real individuals are represented)
- Synthetic email addresses use `@example.test` domain
- Bank accounts, mobile numbers, and addresses are fabricated
- The system predicts geographic zones, not individuals
- Human decision-making remains in the loop
