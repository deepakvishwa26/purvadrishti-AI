# HIVE-Predict Synthetic Data Dictionary

This document defines every field in the generated synthetic dataset.

> **IMPORTANT**: All data is synthetically generated. No real PII or individual records are included.

---

## complaints.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| complaint_id | string | Unique complaint identifier | Generated | Generated (sequential) | Non-PII | Join key | Format: CMP00000001 |
| incident_date | date | Date when fraud occurred | Generated | Generated | Non-PII | Indirect (via hour, day_of_week) | YYYY-MM-DD |
| incident_time | time | Time when fraud occurred | Generated | Generated | Non-PII | Indirect (via hour) | HH:MM:SS |
| incident_datetime | datetime | Full incident timestamp | Generated | Generated | Non-PII | Feature derivation source | ISO 8601 |
| complaint_registered_at | datetime | When complaint was filed | Generated | Generated | Non-PII | Feature cutoff timestamp | Always after incident_datetime |
| incident_details | string | Description of fraud event | Generated | Generated (templated) | Non-PII | Not used directly | Synthetic text |
| fraud_type | string | Category of fraud | Generated | Generated (weighted random) | Non-PII | Direct feature | One of 5 types |
| fraud_amount | float | Amount lost (INR) | Generated | Generated (lognormal, type-conditioned) | Non-PII | Direct feature | Always positive |
| victim_state | string | Victim's state | Generated | Generated (weighted random) | Low-risk | Indirect | Indian state name |
| victim_pincode | string | Victim's postal code | Generated | Generated | Low-risk | Not used directly | Synthetic pincode |
| victim_lat | float | Victim approximate latitude | Generated | Generated | Low-risk | Distance computation | Within state bounds |
| victim_lon | float | Victim approximate longitude | Generated | Generated | Low-risk | Distance computation | Within state bounds |
| victim_account_id | string | FK to accounts table | Generated | Generated | Non-PII | Join key | Links to account entity |
| source_system | string | Data source identifier | Generated | Generated | Non-PII | Not used | Always "SYNTHETIC" |
| data_version | string | Schema version | Generated | Generated | Non-PII | Not used | From config |

---

## accounts.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| account_id | string | Unique account identifier | Generated | Generated (sequential) | Non-PII | Join key | Format: ACCT00000001 |
| bank_id | string | Bank identifier | Generated | Generated (random from bank list) | Non-PII | Not used directly | Synthetic bank |
| account_type | string | Role of account | Generated | Generated | Non-PII | Not used directly | VICTIM, MULE, MERCHANT, OTHER |
| account_status | string | Account status | Generated | Generated | Non-PII | Not used | Always "ACTIVE" |
| first_seen_timestamp | datetime | When account first appeared | Generated | Generated | Non-PII | Not used directly | ISO 8601 |

---

## transactions.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| transaction_id | string | Unique transaction identifier | Generated | Generated (sequential) | Non-PII | Join key | Format: TXN0000000001 |
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| utr | string | Unique Transaction Reference | Generated | Generated (random digits) | Non-PII | Not used in ML | Synthetic UTR |
| transaction_date | date | Transaction date | Generated | Derived from incident time | Non-PII | Indirect | YYYY-MM-DD |
| transaction_time | time | Transaction time | Generated | Derived from incident time | Non-PII | Indirect | HH:MM:SS |
| transaction_datetime | datetime | Full transaction timestamp | Generated | Derived | Non-PII | Velocity computation | ISO 8601 |
| bank_wallet_merchant | string | Payment channel | Generated | Generated (random) | Non-PII | Not used directly | Synthetic bank name |
| amount | float | Transaction amount (INR) | Generated | Derived from fraud_amount | Non-PII | Velocity computation | Always positive |
| source_account_id | string | FK to accounts (sender) | Generated | Generated | Non-PII | Join key | Links to account entity |
| destination_account_id | string | FK to accounts (receiver) | Generated | Generated | Non-PII | Join key | Links to account entity |

---

## mule_chains.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| chain_id | string | Unique chain identifier | Generated | Generated (sequential) | Non-PII | Grouping | Format: CHN00000001 |
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| source_account | string | FK to accounts (sender) | Generated | Generated | Non-PII | Chain graph | Linked account |
| destination_account | string | FK to accounts (receiver) | Generated | Generated | Non-PII | Chain graph | Linked account |
| hop_number | int | Position in chain (1-indexed) | Generated | Generated (sequential) | Non-PII | mule_chain_depth derivation | Starts at 1 |
| transaction_amount | float | Transfer amount after friction (INR) | Generated | Derived (amount × (1-friction)) | Non-PII | amount_velocity derivation | Decreases each hop |
| transaction_timestamp | datetime | When this hop occurred | Generated | Derived (previous + delay) | Non-PII | mule_velocity derivation | Strictly increasing |

---

## suspects.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| suspect_id | string | Unique suspect identifier | Generated | Generated (sequential) | Non-PII | Join key | Format: SUS00000001 |
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| suspect_mobile | string | Synthetic mobile number | Generated | Generated (random digits) | SYNTHETIC PII | Not used in ML | 10-digit Indian format |
| suspect_email | string | Synthetic email | Generated | Generated | SYNTHETIC PII | Not used in ML | Uses @example.test domain |
| suspect_bank_account | string | Synthetic bank account | Generated | Generated (random digits) | SYNTHETIC PII | Not used in ML | 16-digit synthetic |
| suspect_address | string | Synthetic address | Generated | Generated (Faker) | SYNTHETIC PII | Not used in ML | Synthetic Indian address |
| suspect_url_social_handle | string | Synthetic social handle | Generated | Generated | SYNTHETIC PII | Not used in ML | Prefixed with @syn_ etc. |

---

## withdrawals.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| withdrawal_id | string | Unique withdrawal identifier | Generated | Generated (sequential) | Non-PII | Label key | Format: WDR00000001 |
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| account_id | string | FK to accounts (cashout account) | Generated | Generated | Non-PII | Join key | Final mule account |
| withdrawal_timestamp | datetime | When cash-out occurred | Generated | Derived (chain end + delay) | Non-PII | Ground truth | ISO 8601 |
| amount | float | Withdrawal amount (INR) | Generated | Derived (available × fraction) | Non-PII | Ground truth | Always positive, ≤ available |
| atm_id | string | FK to atm_reference | Generated | Selected by proximity | Non-PII | Ground truth | Nearby ATM |
| latitude | float | ATM latitude | Generated | From ATM reference | Non-PII | Ground truth (H3 source) | Valid India bounds |
| longitude | float | ATM longitude | Generated | From ATM reference | Non-PII | Ground truth (H3 source) | Valid India bounds |
| h3_cell | string | H3 cell at resolution 8 | Derived | Derived from lat/lon via H3 library | Non-PII | Ground truth label | Real H3 cell ID |

---

## atm_reference.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| atm_id | string | Unique ATM identifier | Generated | Generated (sequential) | Non-PII | Reference | Format: ATM000001 |
| latitude | float | ATM latitude | Generated | Generated (clustered around cities) | Non-PII | Density computation | SYNTHETIC location |
| longitude | float | ATM longitude | Generated | Generated | Non-PII | Density computation | SYNTHETIC location |
| state | string | Indian state | Generated | Derived from coordinates | Non-PII | Geographic analysis | State name |
| city | string | Nearest city (if urban) | Generated | From urban center | Non-PII | Geographic analysis | May be null for rural |
| h3_cell_res8 | string | H3 cell at resolution 8 | Derived | Via H3 library | Non-PII | ATM density computation | Real H3 cell |
| h3_cell_res9 | string | H3 cell at resolution 9 | Derived | Via H3 library | Non-PII | Dense urban analysis | Real H3 cell |
| source | string | Data source marker | Generated | Generated | Non-PII | Provenance | Always "SYNTHETIC_REFERENCE" |

---

## feature_snapshots.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| feature_cutoff_timestamp | datetime | Point-in-time cutoff | Derived | = complaint_registered_at | Non-PII | Leakage prevention | Only data ≤ this time used |
| fraud_amount | float | Fraud amount (INR) | Derived | From complaint | Non-PII | XGBoost feature | Direct copy |
| amount_log | float | Log-transformed amount | Derived | log1p(fraud_amount) | Non-PII | XGBoost feature | Reduces skew |
| hour | int | Hour of incident (0-23) | Derived | From incident_datetime | Non-PII | XGBoost feature | Temporal feature |
| day_of_week | int | Day of week (0=Mon, 6=Sun) | Derived | From incident_datetime | Non-PII | XGBoost feature | Temporal feature |
| is_weekend | int | Weekend indicator (0/1) | Derived | From day_of_week | Non-PII | XGBoost feature | Binary |
| is_night | int | Night indicator (0/1) | Derived | From hour | Non-PII | XGBoost feature | 22:00-05:00 |
| fraud_type | string | Fraud category | Derived | From complaint | Non-PII | XGBoost feature (encoded) | Categorical |
| fraud_type_encoded | int | Numeric fraud type | Derived | Encoding map | Non-PII | XGBoost feature | 0-4 |
| mule_chain_depth | int | Number of mule hops | Derived | Count of mule_chain records | Non-PII | XGBoost feature | From actual chain |
| mule_velocity | float | Hops per minute | Derived | chain hops / time span | Non-PII | XGBoost feature | 0 if single hop |
| amount_velocity | float | INR per minute through chain | Derived | amount / time span | Non-PII | XGBoost feature | Financial velocity |
| distance_from_victim | float | Distance to ATM region (km) | Derived | Haversine(victim, avg ATM) | Non-PII | XGBoost feature | Geographic |
| historical_hotspot_density | float | Past withdrawal density near victim | Derived | Historical withdrawals in H3 neighborhood | Non-PII | XGBoost feature | Temporal-aware |
| atm_density | int | ATMs in victim's H3 neighborhood | Derived | Count from ATM reference | Non-PII | XGBoost feature | From reference data |
| complaint_cluster | int | DBSCAN cluster label | Derived | Spatial-temporal clustering | Non-PII | XGBoost feature | -1 = noise |
| victim_h3_res8 | string | Victim location H3 cell | Derived | H3(victim_lat, victim_lon) | Non-PII | Candidate generation | Resolution 8 |
| time_since_transaction | float | Seconds from incident to cutoff | Derived | cutoff - incident time | Non-PII | XGBoost feature | Registration delay proxy |

---

## cashout_labels.csv

| Field | Type | Description | Source | Derived/Generated | PII Status | ML Usage | Notes |
|-------|------|-------------|--------|-------------------|------------|----------|-------|
| complaint_id | string | FK to complaints | Generated | Generated | Non-PII | Join key | Links to complaint |
| actual_withdrawal_id | string | FK to withdrawals (or null) | Derived | From withdrawal | Non-PII | Ground truth | Null for NO_CASHOUT |
| actual_h3_cell | string | True withdrawal H3 cell (or null) | Derived | From withdrawal | Non-PII | Ground truth target | Null for NO_CASHOUT |
| actual_atm_id | string | True ATM used (or null) | Derived | From withdrawal | Non-PII | Ground truth | Null for NO_CASHOUT |
| actual_withdrawal_timestamp | datetime | When withdrawal occurred (or null) | Derived | From withdrawal | Non-PII | Ground truth | Null for NO_CASHOUT |
| actual_withdrawal_amount | float | Amount withdrawn (or null) | Derived | From withdrawal | Non-PII | Ground truth | Null for NO_CASHOUT |
| time_to_cashout_seconds | float | Seconds from complaint to cashout (or null) | Derived | withdrawal_ts - registered_at | Non-PII | Analysis | Null for NO_CASHOUT |
| cashout_occurred | bool | Whether cashout happened | Derived | Presence of withdrawal | Non-PII | Classification | True/False |

---

## Financial Simulation Rules

1. **Fraud amount**: Sampled from lognormal distribution conditioned on fraud type.
2. **Initial transaction**: Full fraud amount transferred from victim to first mule account.
3. **Mule hop friction**: Each hop loses 2-10% of the transferred amount (configurable).
4. **Withdrawal fraction**: 50-95% of available funds at cashout account are withdrawn.
5. **Multiple withdrawals**: Split using Dirichlet distribution.
6. **ATM limit**: Per-transaction limit of INR 50,000 (not strictly enforced in synthetic data as single withdrawal events may represent multiple ATM transactions).
7. **Constraint**: Total withdrawn ≤ fraud amount (validated).

---

## Temporal Simulation Rules

1. **Incident time**: Weighted by hour-of-day distribution (peaks in evening).
2. **Registration delay**: 1 minute to 2 hours after incident.
3. **Initial transaction**: 10s to 2min after incident.
4. **Mule hop delay**: Profile-dependent (30s to 30min per hop).
5. **Cashout delay**: Profile-dependent (5min to 4 hours after last hop).
6. **Multiple withdrawal spacing**: 5-30 minutes between withdrawals.
7. **All timestamps are strictly monotonically ordered per case.**

---

## Geographic Simulation Rules

1. **Victim location**: Random within state bounding box, state selected by weight.
2. **ATM placement**: 70% clustered near 10 major urban centers, 30% scattered across states.
3. **ATM selection for withdrawal**: Weighted by inverse distance to victim location.
4. **H3 cells**: Derived from real coordinates using H3 library at resolution 8.
5. **ATM density**: Count of ATMs in H3 cell + 1-ring neighbors.
6. **Historical hotspot density**: Fraction of past withdrawals in victim's 2-ring H3 neighborhood.
