# HIVE-PREDICT PROJECT CONTEXT
## Project GoldenHour — SIH Problem Statement 26184

This document is the authoritative implementation context for the HIVE-Predict ML core and synthetic-data generation system.

It consolidates the decisions made during the project discussion and should be treated as the implementation specification unless a later explicitly approved document supersedes it.

---

# 1. PROJECT IDENTITY

Project:
Project GoldenHour

SIH Problem Statement:
26184

ML Component:
HIVE-Predict

Expanded meaning:
Hexagonal Intelligence for Victim-fund Extraction — Predictive Engine

Primary purpose:

HIVE-Predict is a spatio-temporal, explainable predictive analytics engine designed to predict where stolen funds are most likely to be physically withdrawn as cash during the next 0–4 hours after a cyber-fraud complaint becomes actionable.

The system does NOT simply classify whether fraud occurred.

It predicts:

1. Top-5 H3 geographic zones where cash-out is likely.
2. Top-5 nearby ATMs associated with those zones.
3. Probability/ranking information.
4. Alert priority.
5. SHAP-based explanation of the prediction.

The prediction is intended as decision-support intelligence for law-enforcement and banking personnel.

The model does not autonomously freeze accounts, dispatch police, or take enforcement action.

Human decision-making remains in the loop.

---

# 2. CORE PROBLEM

Cyber-fraud investigation generally begins from a digital event:

Complaint
→ transaction
→ recipient account
→ mule accounts
→ money movement

The important operational question is:

"Where is the stolen money likely to be physically withdrawn next?"

HIVE-Predict addresses the geographic cash-out layer.

The central prediction target is therefore:

Future withdrawal location represented as an H3 cell.

The Golden Hour prediction window is:

0–4 hours.

---

# 3. IMPORTANT DISTINCTION: EXISTING SYSTEM VS PROPOSED SYSTEM

HIVE-Predict must NOT be described as a replacement for NCRP, CFCFRMS, bank systems, or existing government infrastructure.

The proposed architecture is an additional predictive intelligence layer.

Conceptually:

Existing NCRP / CFCFRMS ecosystem
            |
            | authorized data/API/event integration
            ↓
Integration Layer
            ↓
HIVE Analytical Data Layer
            ↓
Feature Engineering
            ↓
HIVE-Predict
            ↓
Prediction / Alert Dashboard

The project does not claim knowledge of the exact internal production database schema of NCRP/CFCFRMS.

The architecture described here is a PROPOSED integration/data architecture.

---

# 4. NCRP INPUT ALIGNMENT

The complaint-registration information relevant to HIVE-Predict includes information such as:

- incident date
- incident time
- incident details
- bank/wallet/merchant
- transaction ID / UTR
- transaction date
- fraud amount
- optional suspect information

Optional suspect information may include:

- mobile number
- email
- bank account
- address
- website/social-media handle

Not every HIVE-Predict feature comes directly from the complaint form.

This distinction is critical.

Complaint-level information:
- incident information
- fraud amount
- transaction information
- bank/wallet/merchant
- available suspect information

Financial-system enrichment:
- source account
- destination account
- transaction sequence
- mule-chain structure
- transaction timing
- transaction velocity
- account relationships

Geographic enrichment:
- victim coordinates
- ATM coordinates
- H3 cells
- geographic distances
- ATM density

Historical analytical data:
- withdrawal density
- hotspot density
- historical cash-out patterns
- complaint clustering

Future outcome:
- actual withdrawal
- withdrawal timestamp
- withdrawal location
- actual H3 cell

The system must never claim that NCRP directly supplies fields that actually require downstream authorized enrichment.

---

# 5. OFFICIAL PUBLIC DATA VS SYNTHETIC DATA

The project currently has official/public aggregate datasets.

These are reference/calibration datasets.

They are NOT individual complaint-level training records.

Official/public datasets currently include:

1. State/UT financial cyber-fraud reporting derived from NCRP-related official public reporting.
2. NCRB State/UT cybercrime statistics for 2021–2023.
3. Associated official State/UT metrics.

These official datasets are used for:

- contextual analysis
- geographic weighting
- distribution calibration where appropriate
- sanity checking
- official baseline statistics

They must not be mixed with synthetic individual observations.

Raw official source files must remain immutable.

Every processed official dataset must retain provenance.

Synthetic data must be clearly marked as synthetic.

Never describe synthetic observations as real NCRP records.

Preferred terminology:

"calibrated synthetic dataset"

or:

"synthetically generated fraud-event dataset informed by public/official aggregate statistics."

---

# 6. CORE DATA ARCHITECTURE

The proposed analytical data architecture is:

NCRP / CFCFRMS
        |
        | Authorized data
        ↓
Integration Layer
        |
        +------------------+
        |                  |
        ↓                  ↓
Complaint            Transaction
        |                  |
        |                  ↓
        |               Account
        |                  |
        |                  ↓
        |              Mule Chain
        |                  |
        |                  ↓
        |              Withdrawal
        |                  |
        |                  ↓
        |                  ATM
        |                  |
        +--------+---------+
                 ↓
        Feature Engineering
                 ↓
        Feature Snapshot
                 ↓
        HIVE-Predict
                 |
          +------+------+
          |             |
          ↓             ↓
      XGBoost          KDE
       Ranker          Spatial
          |             |
          +------+------+
                 ↓
             Score Fusion
                 ↓
           Top-5 H3 Zones
                 ↓
            Top-5 ATMs
                 ↓
          Police/Bank UI
                 ↓
          Actual Outcome
                 ↓
            Label Store
                 ↓
          Retraining Loop

---

# 7. FINALIZED CONCEPTUAL DATA MODEL

The finalized conceptual data model contains:

1. complaint
2. transaction
3. suspect
4. mule_chain
5. withdrawal
6. account
7. ATM/reference geography
8. feature_snapshot
9. cashout_labels

The ACCOUNT entity was deliberately added because financial relationships should not be represented only as raw strings inside other tables.

---

# 8. COMPLAINT TABLE

Conceptual schema:

complaint
---------
complaint_id
incident_date
incident_time
incident_details
fraud_type
fraud_amount
victim_state
victim_pincode

Implementation-level additions may include:

complaint_registered_at
status
source_system
data_version

Reason for complaint_registered_at:

Incident time and complaint-registration time are not necessarily identical.

Example:

Fraud occurs:
22:15

Victim notices:
22:40

Complaint registered:
22:47

The ML system must know what information was available at prediction time.

---

# 9. TRANSACTION TABLE

Conceptual schema:

transaction
-----------
transaction_id
utr
transaction_date
transaction_time
bank_wallet_merchant
amount

Implementation should additionally support:

source_account_id
destination_account_id
complaint_id

Reason:

The transaction table must be linkable to the account graph and complaint.

---

# 10. ACCOUNT TABLE

Recommended implementation schema:

account
-------
account_id
bank_id
account_type
account_status
first_seen_timestamp

Possible account types:

VICTIM
MULE
MERCHANT
OTHER

The account table provides the entity layer connecting:

Complaint
→ Transaction
→ Mule chain
→ Withdrawal

---

# 11. SUSPECT TABLE

Conceptual schema:

suspect
-------
suspect_id
suspect_mobile
suspect_email
suspect_bank_account
suspect_address
suspect_url_social_handle

These fields are operational/linkage information.

Raw PII must NOT directly become ordinary ML features.

For synthetic development:

- mobile numbers must be synthetic
- email addresses must use safe synthetic domains
- bank account identifiers must be synthetic
- addresses must be synthetic
- social handles must be synthetic

Do not intentionally generate real personal information.

---

# 12. MULE CHAIN TABLE

Conceptual schema:

mule_chain
----------
chain_id
complaint_id
source_account
destination_account
hop_number
transaction_amount
transaction_timestamp

Implementation may use:

source_account_id
destination_account_id

instead of ambiguous raw account strings.

Example:

Victim Account
      |
      | ₹420,000
      ↓
Mule A
      |
      | ₹380,000
      ↓
Mule B
      |
      | ₹350,000
      ↓
Mule C
      |
      ↓
Cash-out

mule_chain_depth must be derived from the actual chain.

It should NOT be independently fabricated in the feature table.

---

# 13. WITHDRAWAL TABLE

Conceptual schema:

withdrawal
----------
withdrawal_id
account_id
withdrawal_timestamp
amount
atm_id
latitude
longitude
h3_cell

The withdrawal represents the actual cash-out event.

The H3 cell should be derived from latitude/longitude using a real H3 implementation.

Do NOT fabricate H3 identifiers as arbitrary strings.

H3 resolution 8 is the primary city-level prediction resolution.

H3 resolution 9 can be used for dense-urban refinement.

---

# 14. ATM REFERENCE DATA

Create an ATM/reference geography dataset containing at least:

atm_id
latitude
longitude
state

Optional:

city
district
source
availability/status

If public geographic data is used, preserve its provenance.

OpenStreetMap may be used as a public geographic reference where appropriate.

If real geographic reference data is unavailable during development, synthetic ATM reference data may be generated.

Synthetic ATM locations must be explicitly marked synthetic.

Do not claim synthetic ATM coordinates represent real ATM locations.

---

# 15. FINAL ML FEATURE SNAPSHOT

Conceptual finalized ML feature table:

feature_snapshot
----------------
complaint_id
feature_cutoff_timestamp
fraud_amount
hour
day_of_week
fraud_type
mule_chain_depth
mule_velocity
amount_velocity
distance_from_victim
historical_hotspot_density
atm_density
complaint_cluster

Additional engineered fields may be added when justified by the HIVE-Predict model design.

Potential additional fields include:

is_weekend
is_night
amount_log
mule_velocity_zscore
time_since_transaction
time_since_last_complaint_same_bank
victim_h3_res8
victim_h3_res9
distance_to_nearest_known_hotspot
risk_score_base
time_decay_weight

These are model/feature-layer fields.

They are NOT raw complaint fields.

---

# 16. FEATURE DERIVATION RULE

ML features must be derived from the underlying connected event data.

Do NOT independently generate feature values.

Examples:

hour
← incident timestamp

day_of_week
← incident timestamp

mule_chain_depth
← mule-chain graph

mule_velocity
← transaction timestamps

amount_velocity
← transaction amounts + timestamps

distance_from_victim
← geographic coordinates

historical_hotspot_density
← historical withdrawal events

atm_density
← ATM reference dataset

complaint_cluster
← clustering of relevant complaints/events

H3
← latitude + longitude

This creates feature lineage and prevents inconsistent synthetic records.

---

# 17. SYNTHETIC DATA GENERATION PRINCIPLE

The synthetic generator must simulate a connected fraud ecosystem.

Do NOT independently generate random CSV files.

Generate a synthetic CASE first.

Conceptually:

CASE
 ↓
Complaint
 ↓
Victim account
 ↓
Fraud transaction
 ↓
Mule account(s)
 ↓
Mule transactions
 ↓
Mule chain
 ↓
Cash-out decision
 ↓
Withdrawal
 ↓
ATM / coordinates
 ↓
H3
 ↓
Feature snapshot
 ↓
Ground-truth label

Every generated complaint should be traceable through the associated records.

---

# 18. SYNTHETIC BEHAVIORAL PROFILES

The simulator should support configurable fraud-ring behavior profiles.

Recommended profiles:

FAST_CASHOUT

Characteristics:
- short chain
- high transaction velocity
- short time between final transfer and withdrawal

DEEP_MULE_CHAIN

Characteristics:
- many account hops
- multiple intermediary accounts

SLOW_CASHOUT

Characteristics:
- longer transaction intervals
- delayed withdrawal

NO_CASHOUT

Characteristics:
- complaint exists
- transaction/mule activity exists
- no withdrawal within the defined prediction/evaluation window

MULTIPLE_CASHOUT

Characteristics:
- multiple withdrawal events
- withdrawals remain logically related to available funds

COORDINATED_CASE

Characteristics:
- multiple complaints exhibit shared temporal/geographic/network characteristics
- useful for complaint clustering

Profile probabilities must be configurable.

Do not claim these probabilities are official statistics unless explicitly supported by an official source.

---

# 19. FRAUD TYPES

Support at least:

DIGITAL_PAYMENT_FRAUD
INVESTMENT_FRAUD
DIGITAL_ARREST
PHISHING
OTHER

The simulator may map these to the fraud-type vocabulary used by the HIVE-Predict technical model.

The exact category mapping must be documented.

Fraud amount distributions should be conditioned on fraud type.

Amounts should generally be right-skewed rather than uniformly distributed.

---

# 20. TEMPORAL GENERATION

Synthetic timestamps must be logically connected.

Example:

Complaint:
22:47:00

Initial transaction:
22:48:12

Mule hop 1:
22:49:03

Mule hop 2:
22:50:41

Mule hop 3:
22:52:18

Withdrawal:
23:08:44

Do not independently randomize these timestamps.

The simulator should generate event sequences.

This allows valid derivation of:

mule_velocity
amount_velocity
time_to_cashout
time_since_transaction

---

# 21. GEOGRAPHIC GENERATION

Geographic behavior should not be uniformly random.

The simulator may use:

- calibrated hotspot priors
- geographic proximity
- ATM density
- historical withdrawal density
- fraud-type behavior
- synthetic behavioral profiles
- temporal behavior

The exact weights must be configurable.

These weights are simulation assumptions unless directly supported by data.

---

# 22. H3 GENERATION

For any coordinate:

latitude + longitude
        ↓
real H3 library
        ↓
H3 cell

Primary resolution:

H3 resolution 8

Dense urban refinement:

H3 resolution 9

The H3 label for an actual withdrawal is generated from the actual withdrawal coordinate.

---

# 23. LABEL DATASET

Keep training features and future outcome labels separate.

Recommended label table:

cashout_labels
--------------
complaint_id
actual_withdrawal_id
actual_h3_cell
actual_atm_id
actual_withdrawal_timestamp
actual_withdrawal_amount
time_to_cashout_seconds

For NO_CASHOUT:

actual_withdrawal_id = null
actual_h3_cell = null
actual_atm_id = null
actual_withdrawal_timestamp = null

The label represents future outcome.

It must never be included as an input feature during inference.

---

# 24. AS-OF / FEATURE CUTOFF PRINCIPLE

This is a critical ML requirement.

Every feature snapshot must have:

feature_cutoff_timestamp

Only information available at or before this timestamp may be used to construct the features.

Example:

Complaint registered:
22:47

Feature cutoff:
22:47

Actual withdrawal:
23:18

Features:
information available <= 22:47

Label:
withdrawal occurring after 22:47

Never include future withdrawal information in the feature vector.

Otherwise the training dataset suffers from target leakage.

---

# 25. HIVE-PREDICT MODEL ARCHITECTURE

HIVE-Predict is a two-stage ensemble.

STAGE 1:
XGBoost Learning-to-Rank

STAGE 2:
Kernel Density Estimation

Fusion:

Final_Score(cell)
=
alpha × XGBoost_score(cell)
+
(1-alpha) × KDE_density(cell)

Initial design value:

alpha ≈ 0.65

The final value should be tuned on validation data.

---

# 26. WHY XGBOOST RANKING

The operational output is:

"Which geographic cells should be ranked highest?"

Therefore the problem is framed as Learning-to-Rank rather than simple multi-class classification.

Use candidate H3 cells for each complaint.

The model learns to rank the true withdrawal cell above incorrect candidate cells.

Candidate negatives should include:

1. random geographic cells
2. nearby hard-negative cells

This improves local discrimination.

---

# 27. WHY KDE

KDE provides a feature-independent historical spatial prior.

It answers:

"Where has cash-out historically concentrated?"

KDE helps prevent isolated and unsupported geographic predictions.

Historical withdrawal coordinates are used to estimate the density surface.

Time decay should allow recent events to influence the density more strongly.

The continuous KDE surface can then be aggregated to H3 cell scores.

---

# 28. SHAP EXPLAINABILITY

SHAP is a first-class component.

For every prediction, provide per-feature contributions.

Example concept:

fraud_type
mule_velocity_zscore
distance_to_nearest_known_hotspot
hour_of_day
mule_chain_depth
amount_log

The system should explain why a geographic candidate received a high score.

---

# 29. MODEL OUTPUT CONTRACT

Expected output:

{
  "top_k_zones": [
    {
      "h3_cell": "...",
      "probability": "...",
      "confidence_band": "..."
    }
  ],
  "top_k_atms": [
    {
      "atm_id": "...",
      "lat": "...",
      "lon": "...",
      "rank_score": "..."
    }
  ],
  "shap_explanation": {
    "feature": "contribution"
  },
  "alert_priority": "HIGH | MEDIUM | LOW"
}

Default K:

5

---

# 30. INFERENCE FLOW

STEP 0:
Incoming complaint.

STEP 1:
Feature transformation.

STEP 2:
Generate candidate H3 cells.

STEP 3:
XGBoost ranking.

STEP 4:
KDE density lookup.

STEP 5:
Score fusion.

STEP 6:
Rank Top-5 H3 cells.

STEP 7:
Map Top-5 H3 cells to nearby ATMs.

STEP 8:
Generate SHAP explanation.

STEP 9:
Produce alert/dashboard payload.

---

# 31. TRAIN / VALIDATION / TEST SPLIT

Use temporal splitting.

Do NOT use a simple random train/test split.

Primary split:

70% train
15% validation
15% test

ordered chronologically.

Reason:

Fraud-ring behavior may persist across time.

Random splitting can leak future behavior into training.

Also consider spatial holdout evaluation to test geographic generalization.

---

# 32. EVALUATION METRICS

Primary:

Top-5 Zone Hit Rate

NDCG@5

Secondary:

Precision@K

False Positive Rate for HIGH alerts

Mean Haversine Error

Precision-Recall analysis is preferred over ROC-AUC for heavily imbalanced zone/time prediction.

---

# 33. SYNTHETIC GENERATOR REQUIREMENTS

Initial generation target:

10,000 complaints.

The generator must later support:

50,000
100,000
500,000

through configuration.

Example:

NUM_CASES=10000

The code should not need modification to change dataset size.

Default random seed:

42

Generation should be reproducible.

---

# 34. SYNTHETIC GENERATOR OUTPUT

Expected datasets:

complaints.csv
accounts.csv
transactions.csv
mule_chains.csv
suspects.csv
withdrawals.csv
atm_reference.csv
feature_snapshots.csv
cashout_labels.csv

Additional metadata:

generation_config.yaml
provenance.json
data_dictionary.md
generation_notes.md

A dataset summary should report row counts and key statistics.

---

# 35. DATA VALIDATION

The generator must validate:

1. Every transaction references valid accounts.
2. Every mule-chain edge references valid accounts.
3. Hop numbers are logically ordered.
4. Transaction timestamps are ordered.
5. Withdrawal account exists.
6. Withdrawal occurs after relevant transaction activity.
7. Withdrawal amount is positive.
8. Coordinates are valid.
9. H3 corresponds to coordinates.
10. Feature values are reproducible.
11. No future information leaks into features.
12. Synthetic PII is clearly synthetic.
13. Amounts are positive.
14. Transfer amounts are logically related.
15. NO_CASHOUT cases contain no withdrawal in the evaluation window.
16. MULTIPLE_CASHOUT cases remain financially consistent.
17. Foreign-key relationships are valid.

---

# 36. IMPORTANT: DO NOT FABRICATE MODEL FEATURES

Do not write:

historical_hotspot_density = random()

or:

atm_density = random()

or:

mule_velocity = random()

if the underlying event data exists.

Instead:

mule_velocity
← transaction timestamps

atm_density
← ATM reference data

historical_hotspot_density
← historical withdrawal data

H3
← coordinates

mule_chain_depth
← actual chain

Features must have data lineage.

---

# 37. HISTORICAL HOTSPOT GENERATION

Historical hotspot density should be computed from historical withdrawal events.

The synthetic system may begin with calibrated hotspot priors.

However, the simulator must clearly distinguish:

HOTSPOT PRIOR
from
OBSERVED SYNTHETIC WITHDRAWAL DENSITY

The purpose is to avoid circularly fabricating a feature and then claiming it was learned from observations.

---

# 38. COMPLAINT CLUSTERING

Complaint clustering may use:

- latitude
- longitude
- timestamp
- same-day activity
- relevant behavioral/network similarities

DBSCAN is compatible with the existing HIVE-Predict design.

The resulting cluster identifier is a derived feature.

---

# 39. PII SEPARATION

Operational identifiers such as:

mobile
email
bank account
address
social handle

belong to the secure operational/linkage layer.

They should not directly be fed into the ML model.

Where appropriate, the ML layer may use derived aggregate features such as:

linked_cases_count
linked_accounts_count
previous_cashout_count
network_degree
identifier_risk_score

These must be carefully evaluated for leakage and fairness.

---

# 40. FREQUENT MODEL TRAINING ARCHITECTURE

Do not retrain the model on every incoming complaint.

Real-time:

Complaint
→ feature generation
→ prediction

Continuous:

new events
→ event/data store

Periodic:

data quality
→ feature generation
→ training dataset
→ model training
→ validation
→ model registry
→ production

Retraining can be:

- scheduled weekly
- triggered by drift detection

---

# 41. CLOSED LEARNING LOOP

The long-term architecture is:

Complaint
   ↓
Trace
   ↓
Predict
   ↓
Intervene
   ↓
Observe outcome
   ↓
Create training label
   ↓
Retrain
   ↓
Improve prediction

This closed loop is one of the core strategic ideas of Project GoldenHour.

---

# 42. DRIFT MONITORING

The existing technical design proposes monitoring:

Data drift:
feature distribution changes.

Concept/spatial drift:
new withdrawal clusters or emerging hotspots.

Potential mechanisms:

KS test for feature drift.

Rolling DBSCAN for emerging spatial clusters.

Human review before introducing new hotspots into the KDE prior.

Model promotion should use a champion/challenger validation gate.

---

# 43. MODEL SERIALIZATION

XGBoost model:

native Booster model format / JSON.

KDE:

precomputed density representation.

Preprocessing artifacts:

encoders
scalers
other transformations

All should be versioned.

Training-data hash and model version should be recorded.

---

# 44. IMPORTANT LIMITATIONS

Synthetic data does not equal real-world data.

The key assumption is:

real fraud-ring withdrawal behavior has sufficient structural similarity to the calibrated synthetic simulator.

This must eventually be validated using authorized real data.

Until then, model performance is proof-of-concept evidence only.

Do not present synthetic test results as real-world accuracy.

---

# 45. ETHICAL AND PRIVACY REQUIREMENTS

The MVP/training system should use synthetic data.

Future real-data integration must be:

- authorized
- privacy controlled
- data minimized
- access controlled
- auditable

The model predicts geographic zones.

It does not profile communities or demographic groups.

The system must maintain human oversight.

---

# 46. IMPLEMENTATION PRIORITY

The development sequence is:

PHASE 1
Synthetic data specification

PHASE 2
Synthetic generator

PHASE 3
10,000-case generation

PHASE 4
Data integrity validation

PHASE 5
Feature engineering validation

PHASE 6
Candidate-H3 training dataset

PHASE 7
XGBoost ranking model

PHASE 8
KDE spatial model

PHASE 9
Score fusion

PHASE 10
SHAP explanation

PHASE 11
Evaluation

PHASE 12
Scale synthetic generation to 100K / 500K

PHASE 13
Inference API

PHASE 14
Dashboard integration

Do NOT skip directly to model training before validating the synthetic generator.

---

# 47. CURRENT IMMEDIATE TASK

The immediate task is:

BUILD A ROBUST SYNTHETIC DATA GENERATOR.

First target:

10,000 connected synthetic fraud cases.

The generator must produce all required relational/event datasets and validate them.

Do not train the final HIVE-Predict model yet.

The generator must be good enough that the resulting data can later support:

XGBoost Learning-to-Rank
+
KDE
+
SHAP
+
Top-5 H3 prediction
+
Top-5 ATM mapping

---

# 48. IMPLEMENTATION PHILOSOPHY

The synthetic generator is not merely a CSV faker.

It is a simplified agent/event simulator.

The generator should simulate:

Victim
Fraud event
Money transfer
Mule network
Cash-out behavior
Geographic behavior
ATM selection
Future outcome

The important property is:

CONNECTED + TEMPORALLY CONSISTENT + GEOGRAPHICALLY CONSISTENT + ML-SAFE

---

# 49. REQUIRED DEVELOPMENT QUALITY

The implementation should be:

- modular
- configurable
- reproducible
- testable
- documented
- scalable
- privacy-safe
- leakage-aware

Use clean Python architecture.

Prefer small modules over one huge script.

Include unit/integrity tests.

Include a clear README.

---

# 50. AUTHORITATIVE DESIGN SUMMARY

The final conceptual system is:

                 NCRP / CFCFRMS
                       |
                Authorized data
                       ↓
                Integration Layer
                       |
        +--------------+--------------+
        |              |              |
        ↓              ↓              ↓
    Complaint      Transaction      Suspect
                       |
                       ↓
                    Account
                       |
                       ↓
                  Mule Chain
                       |
                       ↓
                   Withdrawal
                       |
                       ↓
                      ATM
                       |
                       ↓
              Feature Engineering
                       |
                       ↓
               Feature Snapshot
                       |
                +------+------+
                |             |
                ↓             ↓
             XGBoost          KDE
             Ranker         Spatial Prior
                |             |
                +------+------+
                       ↓
                  Score Fusion
                       ↓
                  Top-5 H3 Zones
                       ↓
                   Top-5 ATMs
                       ↓
                 Human Dashboard
                       ↓
                 Actual Outcome
                       ↓
                   Label Store
                       ↓
                Retraining Loop

Core principle:

COMPLAINT → TRACE → PREDICT → INTERVENE → OBSERVE → LABEL → RETRAIN

This is the intended HIVE-Predict architecture.

END OF PROJECT CONTEXT