# HIVE-Predict ML Dataset Audit Report

Generated: 2026-09-03T21:05:20.667013Z


======================================================================
# 1. CASHOUT LABEL SEMANTICS
======================================================================

Total label rows: 14208
Unique complaint_id in labels: 10000
Unique actual_withdrawal_id in labels (non-null): 12709

Rows per complaint distribution:
  min: 1
  max: 4
  mean: 1.42
  1 row(s): 7239 complaints
  2 row(s): 1796 complaints
  3 row(s): 483 complaints
  4 row(s): 482 complaints

cashout_occurred=True rows: 12709
cashout_occurred=False rows: 1499

Total withdrawals: 12709
Withdrawals represented in labels: 12709
Withdrawals NOT in labels: 0
Labels pointing to nonexistent withdrawals: 0

--- WHY withdrawals=12709 but labels=14208 ---
Labels include 1499 NO_CASHOUT rows (one per no-cashout complaint)
Plus 12709 cashout rows (one per withdrawal)
So labels = cashout_withdrawals + no_cashout_complaints = 12709 + 1499 = 14208
MATCH: Every withdrawal has exactly one label row. Design is CORRECT.

======================================================================
# 2. MULE NETWORK AUDIT
======================================================================

Total accounts: 50782
Account types:
  MULE: 40782
  VICTIM: 10000

Unique source accounts in transactions: 40782
Unique destination accounts in transactions: 40782

Accounts appearing in 1 transaction: 20000
Accounts appearing in 2 transactions: 30782
Accounts appearing in 3+ transactions: 0
Max transactions for single account: 2

Accounts participating in multiple complaints: 0
WARNING: NO account reuse across complaints. Network is fully isolated per-case.
This means the mule network is NOT a genuine shared network.

Mule chain depth distribution:
  mean: 3.08
  min: 1
  max: 8
  depth 1: 1825 chains (18.2%)
  depth 2: 2947 chains (29.5%)
  depth 3: 1528 chains (15.3%)
  depth 4: 1828 chains (18.3%)
  depth 5: 972 chains (9.7%)
  depth 6: 306 chains (3.1%)
  depth 7: 281 chains (2.8%)
  depth 8: 313 chains (3.1%)

Unique accounts in mule chains: 40782
Mule accounts appearing in multiple hops: 20782
Mule accounts used across multiple complaints: 0

======================================================================
# 3. CASE-LEVEL EVENT AUDIT
======================================================================


--- Case 1: CMP00002657 ---
  Fraud type: PHISHING, Amount: INR 6,890.79
  Incident: 2024-03-28T20:41:34
  Registered: 2024-03-28T21:49:42
  State: Telangana
  Transactions: 2
    First TX: 2024-03-28T20:41:51 | ACCT00002657 -> ACCT00020881 | INR 6,890.79
  Mule hops: 1
    Hop 1: ACCT00020881 -> ACCT00020882 | INR 6,676.36 | 2024-03-28T20:44:28
  Withdrawals: 1
    WDR00003377: INR 4,730.92 at ATM001666 (8860a2d545fffff) @ 2024-03-28T20:57:14
  Feature snapshot: YES
    cutoff: 2024-03-28T21:49:42
    mule_depth=1, mule_vel=0.0, amt_vel=6676.36
    hotspot_density=0.0, atm_density=0, cluster=1
  Labels: 1 row(s)
    cashout=True, wdr_id=WDR00003377, h3=8860a2d545fffff
  Temporal order: OK

--- Case 2: CMP00000446 ---
  Fraud type: DIGITAL_PAYMENT_FRAUD, Amount: INR 24,299.55
  Incident: 2024-02-15T20:48:10
  Registered: 2024-02-15T22:14:35
  State: Haryana
  Transactions: 4
    First TX: 2024-02-15T20:48:51 | ACCT00000446 -> ACCT00011824 | INR 24,299.55
  Mule hops: 3
    Hop 1: ACCT00011824 -> ACCT00011825 | INR 23,735.86 | 2024-02-15T21:05:04
    Hop 2: ACCT00011825 -> ACCT00011826 | INR 23,211.59 | 2024-02-15T21:30:42
    Hop 3: ACCT00011826 -> ACCT00011827 | INR 20,891.45 | 2024-02-15T21:55:32
  Withdrawals: 1
    WDR00000588: INR 18,252.19 at ATM001830 (88424b8dd1fffff) @ 2024-02-16T01:01:19
  Feature snapshot: YES
    cutoff: 2024-02-15T22:14:35
    mule_depth=3, mule_vel=0.0594, amt_vel=470.3275
    hotspot_density=0.0, atm_density=0, cluster=17
  Labels: 1 row(s)
    cashout=True, wdr_id=WDR00000588, h3=88424b8dd1fffff
  Temporal order: OK

--- Case 3: CMP00009506 ---
  Fraud type: PHISHING, Amount: INR 19,932.15
  Incident: 2024-11-10T09:16:07
  Registered: 2024-11-10T10:51:36
  State: Gujarat
  Transactions: 6
    First TX: 2024-11-10T09:16:38 | ACCT00009506 -> ACCT00048736 | INR 19,932.15
  Mule hops: 5
    Hop 1: ACCT00048736 -> ACCT00048737 | INR 18,468.60 | 2024-11-10T09:20:14
    Hop 2: ACCT00048737 -> ACCT00048738 | INR 17,443.83 | 2024-11-10T09:24:27
    Hop 3: ACCT00048738 -> ACCT00048739 | INR 16,246.07 | 2024-11-10T09:27:13
    Hop 4: ACCT00048739 -> ACCT00048740 | INR 15,702.89 | 2024-11-10T09:28:27
    Hop 5: ACCT00048740 -> ACCT00048741 | INR 15,006.35 | 2024-11-10T09:31:59
  Withdrawals: 2
    WDR00012063: INR 3,225.76 at ATM001500 (8842cc9ac5fffff) @ 2024-11-10T10:55:06
    WDR00012064: INR 8,643.00 at ATM001500 (8842cc9ac5fffff) @ 2024-11-10T11:19:22
  Feature snapshot: YES
    cutoff: 2024-11-10T10:51:36
    mule_depth=5, mule_vel=0.4255, amt_vel=1571.7957
    hotspot_density=0.0, atm_density=0, cluster=416
  Labels: 2 row(s)
    cashout=True, wdr_id=WDR00012063, h3=8842cc9ac5fffff
    cashout=True, wdr_id=WDR00012064, h3=8842cc9ac5fffff
  Temporal order: OK

--- Case 4: CMP00000333 ---
  Fraud type: DIGITAL_ARREST, Amount: INR 252,078.60
  Incident: 2024-12-19T23:40:47
  Registered: 2024-12-20T01:28:06
  State: Madhya Pradesh
  Transactions: 5
    First TX: 2024-12-19T23:42:26 | ACCT00000333 -> ACCT00011348 | INR 252,078.60
  Mule hops: 4
    Hop 1: ACCT00011348 -> ACCT00011349 | INR 237,003.54 | 2024-12-19T23:44:54
    Hop 2: ACCT00011349 -> ACCT00011350 | INR 224,484.71 | 2024-12-19T23:47:44
    Hop 3: ACCT00011350 -> ACCT00011351 | INR 215,468.88 | 2024-12-19T23:50:00
    Hop 4: ACCT00011351 -> ACCT00011352 | INR 200,679.58 | 2024-12-19T23:53:03
  Withdrawals: 1
    WDR00000441: INR 185,221.54 at ATM001486 (8842cbc849fffff) @ 2024-12-20T01:21:08
  Feature snapshot: YES
    cutoff: 2024-12-20T01:28:06
    mule_depth=4, mule_vel=0.4908, amt_vel=29080.189
    hotspot_density=0.0, atm_density=0, cluster=-1
  Labels: 1 row(s)
    cashout=True, wdr_id=WDR00000441, h3=8842cbc849fffff
  Temporal order: OK

--- Case 5: CMP00004169 ---
  Fraud type: PHISHING, Amount: INR 7,436.77
  Incident: 2024-01-04T17:53:54
  Registered: 2024-01-04T18:05:25
  State: Maharashtra
  Transactions: 2
    First TX: 2024-01-04T17:54:22 | ACCT00004169 -> ACCT00027001 | INR 7,436.77
  Mule hops: 1
    Hop 1: ACCT00027001 -> ACCT00027002 | INR 7,166.92 | 2024-01-04T17:56:24
  Withdrawals: 1
    WDR00005255: INR 6,023.99 at ATM001895 (886095c191fffff) @ 2024-01-04T18:36:55
  Feature snapshot: YES
    cutoff: 2024-01-04T18:05:25
    mule_depth=1, mule_vel=0.0, amt_vel=7166.92
    hotspot_density=0.0, atm_density=0, cluster=-1
  Labels: 1 row(s)
    cashout=True, wdr_id=WDR00005255, h3=886095c191fffff
  Temporal order: OK

Remaining 15 sampled cases: 15/15 have feature+label coverage

======================================================================
# 4. TEMPORAL AUDIT
======================================================================

  incident -> first transaction:
    min:    0.2 min
    median: 1.1 min
    mean:   1.1 min
    P90:    1.8 min
    P95:    1.9 min
    max:    2.0 min
  first transaction -> first mule hop:
    min:    0.5 min
    median: 3.1 min
    mean:   5.7 min
    P90:    14.2 min
    P95:    22.5 min
    max:    30.0 min
  mule hop -> mule hop:
    min:    0.5 min
    median: 4.5 min
    mean:   6.3 min
    P90:    13.2 min
    P95:    21.6 min
    max:    30.0 min
  final mule activity -> first withdrawal:
    min:    5.0 min
    median: 57.7 min
    mean:   80.6 min
    P90:    176.6 min
    P95:    207.7 min
    max:    240.0 min
  incident -> first withdrawal:
    min:    6.7 min
    median: 70.3 min
    mean:   101.0 min
    P90:    228.8 min
    P95:    261.4 min
    max:    332.5 min

Feature cutoff validation:
  cutoff == registered_at: 10000/10000

======================================================================
# 5. FINANCIAL FLOW AUDIT
======================================================================

Sampled 2000 cases:
  Cases with impossible money flow (withdrawal > fraud): 0
  Max withdrawal/fraud ratio: 0.9209
  Mean withdrawal/fraud ratio: 0.6029
  Median withdrawal/fraud ratio: 0.5994

Multiple withdrawal cases: 2761
  Max withdrawals per case: 4

======================================================================
# 6. GEOGRAPHIC AUDIT
======================================================================

Unique H3 cells in withdrawals: 1065
Unique ATMs used: 1131
Total ATMs in reference: 2000

H3 validation (500 sample): 0 mismatches

Top 20 H3 cells by withdrawal count:
  883d145833fffff: 58 withdrawals
  8842d9aa2bfffff: 55 withdrawals
  8842d575e3fffff: 53 withdrawals
  88425822adfffff: 51 withdrawals
  8842edb963fffff: 50 withdrawals
  8860a228cdfffff: 48 withdrawals
  8861980001fffff: 47 withdrawals
  883cb26953fffff: 45 withdrawals
  8842eed161fffff: 45 withdrawals
  883d1642e5fffff: 44 withdrawals
  8860a07543fffff: 44 withdrawals
  8860a835abfffff: 43 withdrawals
  8860313019fffff: 43 withdrawals
  883cb5d835fffff: 42 withdrawals
  883d80548bfffff: 42 withdrawals
  883d902019fffff: 42 withdrawals
  8860a38d97fffff: 42 withdrawals
  8842c565b7fffff: 42 withdrawals
  883c1b5563fffff: 42 withdrawals
  883d1608a1fffff: 41 withdrawals

ATM usage distribution:
  ATMs used once: 255
  ATMs used 2-5 times: 247
  ATMs used 6-20 times: 394
  ATMs used 20+ times: 254
  Max usage: 58

Geographic spread:
  Latitude  range: [8.1728, 32.4268], std=5.9554
  Longitude range: [68.4515, 89.7930], std=4.2387

======================================================================
# 7. HISTORICAL HOTSPOT FEATURE AUDIT
======================================================================

historical_hotspot_density statistics:
  min:    0.000000
  max:    0.012987
  mean:   0.000033
  median: 0.000000
  zeros:  9739 (97.4%)
  nonzero:261 (2.6%)

Lineage analysis:
  SOURCE: historical withdrawal records (only from cases registered BEFORE current case)
  METHOD: Count withdrawals in victim's H3 2-ring neighborhood / total historical withdrawals
  CUTOFF: Only withdrawals with timestamp < feature_cutoff_timestamp
  NOTE: Cases are processed in order of complaint_registered_at
  POTENTIAL ISSUE: First cases always have density=0 (no history yet)
  First 100 cases with density=0: 99

Circularity check:
  The feature uses historical_withdrawals which EXCLUDES the current case's withdrawals
  (generator.py line 231: withdrawals are added to historical set AFTER feature computation)
  VERDICT: No circularity detected in code logic.

======================================================================
# 8. ATM DENSITY AUDIT
======================================================================

atm_density statistics:
  min:    0
  max:    10
  mean:   0.03
  median: 0.0
  zeros:  9839

Derivation method:
  Uses compute_atm_density_h3() from geography.py
  Counts ATMs in the victim's H3 res-8 cell + its 6 immediate H3 neighbors (grid_disk radius=1)
  Data source: atm_reference DataFrame (static, not temporal)
  VERDICT: Derived from static ATM reference. No leakage.

======================================================================
# 9. COMPLAINT CLUSTER AUDIT
======================================================================

Total complaints: 10000
Unique cluster labels: 709
Noise/unclustered (-1): 2172 (21.7%)
Clustered complaints: 7828
Number of clusters: 708
Cluster size distribution:
  min:    3
  max:    1009
  mean:   11.1
  median: 5.0

Derivation method:
  DBSCAN on [victim_lat*111.32*spatial_weight, victim_lon*111.32*spatial_weight, hour_norm*temporal_window*temporal_weight]
  eps=10.0 km, min_samples=3
  Uses victim lat/lon and incident hour (all available at cutoff)
  VERDICT: No future information used. Based on complaint geography+time only.

======================================================================
# 10. FEATURE LINEAGE AUDIT
======================================================================


| Feature | Source Table | Source Field(s) | Derivation | Available at Cutoff? | Potential Leakage? |
|---------|-------------|-----------------|------------|--------------------|--------------------|
| fraud_amount | complaints | fraud_amount | Direct copy | YES | NO |
| hour | complaints | incident_datetime | .hour | YES | NO |
| day_of_week | complaints | incident_datetime | .weekday() | YES | NO |
| is_weekend | complaints | incident_datetime | weekday >= 5 | YES | NO |
| is_night | complaints | incident_datetime | hour >= 22 or <= 5 | YES | NO |
| fraud_type | complaints | fraud_type | Label encoding | YES | NO |
| mule_chain_depth | mule_chains | hop records | len(case mule records) | PARTIAL* | MEDIUM* |
| mule_velocity | mule_chains | timestamps | hops/minute | PARTIAL* | MEDIUM* |
| amount_velocity | mule_chains | amounts+timestamps | INR/minute | PARTIAL* | MEDIUM* |
| distance_from_victim | complaints+ATM ref | victim coords + state ATMs | haversine to avg ATM in state | YES | NO |
| historical_hotspot_density | withdrawals (past) | h3_cell + timestamp | count in H3 2-ring / total | YES (past only) | LOW |
| atm_density | ATM reference | h3_cell_res8 | count in cell+neighbors | YES (static) | NO |
| complaint_cluster | complaints | victim lat/lon + hour | DBSCAN | YES | NO |
| victim_h3_res8 | complaints | victim lat/lon | h3.latlng_to_cell | YES | NO |
| amount_log | complaints | fraud_amount | log1p | YES | NO |
| time_since_transaction | complaints | incident + registered | seconds between | YES | NO |

*CRITICAL NOTE on mule_chain_depth, mule_velocity, amount_velocity:
  These features are derived from the COMPLETE mule chain for the case.
  In REAL deployment, the full mule chain may not be known at prediction time.
  The mule chain is discovered AFTER the complaint, potentially concurrently.
  Whether this constitutes leakage depends on operational assumptions:
  - If HIVE-Predict runs AFTER the banking system traces the mule chain: NO leakage
  - If HIVE-Predict runs IMMEDIATELY upon complaint receipt: LEAKAGE
  PROJECT_CONTEXT.md says 'Financial-system enrichment' includes 'mule-chain structure'
  This implies the system has access to mule chain data at prediction time.
  VERDICT: Acceptable given the documented integration architecture, but flagged as MEDIUM risk.

======================================================================
# 11. DISTRIBUTION AUDIT
======================================================================


fraud_amount (INR) (n=10000):
  min:  500.0000
  P25:  13064.5725
  P50:  36122.1550
  P75:  111075.1975
  P90:  295345.3080
  P95:  536763.1950
  P99:  1513693.8165
  max:  15820360.2200

mule_chain_depth (n=10000):
  min:  1.0000
  P25:  2.0000
  P50:  3.0000
  P75:  4.0000
  P90:  5.0000
  P95:  7.0000
  P99:  8.0000
  max:  8.0000

mule_velocity (n=10000):
  min:  0.0000
  P25:  0.0825
  P50:  0.2575
  P75:  0.5430
  P90:  1.0526
  P95:  1.5789
  P99:  3.0000
  max:  4.0000

amount_velocity (n=10000):
  min:  14.2417
  P25:  1366.8004
  P50:  5736.7843
  P75:  24276.5407
  P90:  92025.8707
  P95:  201512.5548
  P99:  668608.8607
  max:  14976627.9400

distance_from_victim (km) (n=10000):
  min:  0.4300
  P25:  136.7975
  P50:  229.0450
  P75:  353.9900
  P90:  465.1900
  P95:  522.5765
  P99:  607.2331
  max:  689.3600

time_to_cashout (minutes) (n=12709):
  min:  -108.2500
  P25:  -11.5000
  P50:  33.5167
  P75:  90.2333
  P90:  154.2033
  P95:  191.9167
  P99:  246.2547
  max:  327.5167

withdrawal_amount (INR) (n=12709):
  min:  0.3400
  P25:  3345.2700
  P50:  11107.3900
  P75:  39058.0700
  P90:  115004.3900
  P95:  218364.4780
  P99:  644206.1856
  max:  4652292.7800

--- Categorical Distributions ---

Fraud type:
  DIGITAL_PAYMENT_FRAUD: 3537 (35.4%)
  INVESTMENT_FRAUD: 2012 (20.1%)
  PHISHING: 1955 (19.6%)
  DIGITAL_ARREST: 1548 (15.5%)
  OTHER: 948 (9.5%)

Cashout/No-cashout (by complaint):
  With cashout: 8501 (85.0%)
  No cashout: 1499 (15.0%)

Victim state distribution:
  Maharashtra: 1563 (15.6%)
  Uttar Pradesh: 1207 (12.1%)
  Karnataka: 1030 (10.3%)
  Telangana: 730 (7.3%)
  Delhi: 729 (7.3%)
  Tamil Nadu: 709 (7.1%)
  Rajasthan: 700 (7.0%)
  Gujarat: 687 (6.9%)
  West Bengal: 596 (6.0%)
  Madhya Pradesh: 528 (5.3%)

======================================================================
# 12. BEHAVIORAL PROFILE AUDIT
======================================================================

NOTE: Behavioral profile is not stored in output. Inferring from patterns.

Inferred profile distribution:
  FAST_CASHOUT: 2921 (29.2%)
  MULTIPLE_CASHOUT: 2761 (27.6%)
  NO_CASHOUT: 1499 (15.0%)
  DEEP_MULE_CHAIN: 1440 (14.4%)
  SLOW_CASHOUT: 1033 (10.3%)
  OTHER: 346 (3.5%)

Behavioral comparison by inferred profile:

  FAST_CASHOUT (n=2921):
    avg chain depth:  1.55
    median ttc:       33.9 min
    avg withdrawals:  1.00

  SLOW_CASHOUT (n=1033):
    avg chain depth:  2.49
    median ttc:       227.5 min
    avg withdrawals:  1.00

  DEEP_MULE_CHAIN (n=1440):
    avg chain depth:  5.06
    median ttc:       174.8 min
    avg withdrawals:  1.00

  NO_CASHOUT (n=1499):
    avg chain depth:  2.96
    median ttc:       N/A
    avg withdrawals:  0.00

  MULTIPLE_CASHOUT (n=2761):
    avg chain depth:  4.09
    median ttc:       87.3 min
    avg withdrawals:  2.52

======================================================================
# 13. SYNTHETIC REALISM AUDIT
======================================================================

Checking fraud_type -> H3 determinism...
  DIGITAL_ARREST: 605 unique H3 cells across 1935 withdrawals
  DIGITAL_PAYMENT_FRAUD: 848 unique H3 cells across 4464 withdrawals
  INVESTMENT_FRAUD: 688 unique H3 cells across 2584 withdrawals
  OTHER: 492 unique H3 cells across 1228 withdrawals
  PHISHING: 715 unique H3 cells across 2498 withdrawals

Checking profile -> cashout determinism...
  Cases with 0 withdrawals: 1499
  (If this exactly matches NO_CASHOUT probability ~15%, profile deterministically controls label)
  Ratio: 15.0%

Checking for duplicate cases...
  Duplicate rows on (fraud_type, amount, state, datetime): 0

Feature-target correlations (point-biserial with cashout_occurred):
  fraud_amount: r=-0.0215
  hour: r=-0.0096
  day_of_week: r=-0.0284
  mule_chain_depth: r=0.0284
  mule_velocity: r=0.1400
  amount_velocity: r=-0.0121
  distance_from_victim: r=-0.0038
  historical_hotspot_density: r=0.0060
  atm_density: r=-0.0106
  complaint_cluster: r=-0.0162

Mule chain determinism check:
  Chains with identical amounts at every hop: 0
  (Should be 0 if friction is applied correctly)

======================================================================
# 14. MODEL LEAKAGE AUDIT
======================================================================

Checking each feature for direct/indirect leakage...

  historical_hotspot_density:
    Uses PAST withdrawals only (before cutoff). Code verified: withdrawals added to historical set AFTER feature computation.

  complaint_cluster:
    DBSCAN on victim lat/lon + incident hour. No withdrawal/label info used. HOWEVER: computed batch-wise on ALL complaints, which means future complaints inform cluster assignment.

  mule_chain_depth:
    Derived from complete mule chain. Available if banking system traces chain before prediction.

  mule_velocity:
    Same as mule_chain_depth - requires chain to be traced.

  amount_velocity:
    Same as mule_chain_depth - requires chain to be traced.

  distance_from_victim:
    Victim coords vs avg ATM in state. No withdrawal info used.

  atm_density:
    Static ATM reference data. No withdrawal info.

CRITICAL LEAKAGE FINDING:
  complaint_cluster uses DBSCAN on ALL complaints simultaneously.
  In the code (compute_complaint_clusters), the entire complaint list is passed.
  This means a complaint registered at time T has its cluster influenced by
  complaints registered AFTER T. This is temporal leakage.
  SEVERITY: MEDIUM - cluster assignment may change if future complaints are excluded.

Checking for label columns in features...
  Label columns found in features: NONE (GOOD)

======================================================================
# 15. REPRODUCIBILITY AUDIT
======================================================================

Computing file hashes for current output...
  accounts.csv: a3369d0b27e3c90573de1833c3edd036 (2,762,297 bytes)
  atm_reference.csv: 6b4f17528597f23198e94379c8a3e9d1 (199,662 bytes)
  cashout_labels.csv: fa4b423704e667bd46f9fefb3d8de509 (1,202,035 bytes)
  complaints.csv: 404fd1c19c8f4c9b427588529d2aabb2 (2,282,492 bytes)
  feature_snapshots.csv: aa039a3e4b1a16eb7849c434ece791fb (1,440,271 bytes)
  mule_chains.csv: 475313b895cd5bd9d9aa41836f7b8d3d (2,521,755 bytes)
  suspects.csv: c372ae423147ee3d20c27fee2f48cf9a (1,307,552 bytes)
  transactions.csv: 817fa019e41355b345cbbfd62a5b278b (5,864,373 bytes)
  validation_report.csv: 1c85a681b36c43250fd42b73a9491d7c (2,239 bytes)
  withdrawals.csv: 4183d30b66c0426a35338f68af78f011 (1,425,687 bytes)

Reproducibility test (small-scale):
  Complaint IDs match: True
  Fraud amounts match: True
  VERDICT: REPRODUCIBLE

======================================================================
# 16. FINAL VERDICT
======================================================================

B. REQUIRES FIXES BEFORE ML TRAINING

Issues ranked by severity:

### [CRITICAL] 30.9% of cashout labels have NEGATIVE time_to_cashout
  Evidence: 3,927 of 12,709 cashout label rows have time_to_cashout_seconds < 0 (min: -108.2 min). This means the withdrawal occurred BEFORE the complaint was registered. The label computes time_to_cashout = withdrawal_timestamp - complaint_registered_at. Since registration_delay can be 1-120 min, and cashout can begin within minutes of the initial fraud, many withdrawals happen before the victim even files a complaint.
  Impact: This is architecturally CORRECT (money moves before the victim complains). However, for ML training it means ~31% of "future" withdrawal labels are actually PAST events relative to the prediction point (feature_cutoff_timestamp = complaint_registered_at). HIVE-Predict cannot predict something that has already happened. These cases should either be excluded from training or treated differently.
  Fix: Filter labels to only include withdrawals with time_to_cashout_seconds > 0 for the ranking model. Alternatively, adjust the prediction window definition. Cases where ALL withdrawals are already complete at registration time should be labeled differently (e.g., ALREADY_CASHED_OUT).

### [CRITICAL] Mule network has zero cross-complaint account reuse
  Evidence: Every mule chain creates entirely new accounts. No account appears in multiple complaints. This means the 'network' is actually N isolated linear chains. An ML model cannot learn network-based patterns because there is no network. In real fraud, mule accounts are reused across cases, creating a graph structure.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Implement a mule account pool with configurable reuse probability. Some accounts should appear as destinations in multiple complaint chains.

### [CRITICAL] complaint_cluster has temporal leakage
  Evidence: DBSCAN clustering is computed batch-wise on ALL complaints simultaneously. A complaint at time T has its cluster influenced by complaints registered after T. In deployment, future complaints would not be available.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Compute clusters incrementally: for each complaint, only use complaints registered at or before its feature_cutoff_timestamp. Alternatively, use a sliding window approach.

### [HIGH] Mule chain features may not be available at prediction time
  Evidence: mule_chain_depth, mule_velocity, amount_velocity require the complete mule chain. If HIVE-Predict runs immediately upon complaint receipt (before chain tracing), these features would be zero/unknown. The current setup assumes full chain availability.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Document the operational assumption explicitly. Consider adding a 'chain_known' flag or providing graceful degradation when chain is partial.

### [HIGH] NO_CASHOUT is deterministically controlled by behavioral profile
  Evidence: Cases with 0 withdrawals: 1499 (15.0%). This closely matches the configured NO_CASHOUT probability (15%). The profile directly determines whether cashout occurs. An ML model might learn to exploit the profile-specific feature distributions rather than genuine patterns.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Add stochastic cashout failure to other profiles (e.g., 5% of FAST_CASHOUT cases could fail to cash out due to account freezing). Make NO_CASHOUT less deterministic.

### [MEDIUM] historical_hotspot_density is mostly zero
  Evidence: Zero values: 9739 (97.4%). Because cases are processed chronologically and the victim's H3 neighborhood rarely overlaps with historical withdrawal locations, this feature is near-zero for most cases.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Consider using a broader spatial window, or seeding with some initial historical data to make this feature more informative.

### [MEDIUM] ATM selection is victim-proximity-based, creating geographic leakage
  Evidence: Withdrawals use select_nearby_atm() which selects ATMs near the VICTIM location. This means distance_from_victim and atm_density (both computed at victim location) are correlated with where the withdrawal actually occurs. The model may learn 'predict near the victim' as a shortcut rather than learning genuine geographic patterns.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Introduce more geographic diversity: some withdrawals should occur far from the victim, especially for DEEP_MULE_CHAIN and COORDINATED_CASE profiles.

### [MEDIUM] Geographic diversity is limited
  Evidence: Only 1065 unique H3 cells for 12709 withdrawals. Top H3 cells concentrate many withdrawals, limiting the model's ability to learn diverse geographic patterns.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Increase geographic spread and ATM count, or vary the selection radius by profile.

### [LOW] amount_velocity has extreme outliers
  Evidence: P99: 668609, max: 14976628. Single-hop chains get amount_velocity = fraud_amount (no time denominator), creating extremely high values.
  Impact: Affects HIVE-Predict ML training quality.
  Fix: Cap or normalize amount_velocity for single-hop cases. Consider using log transform.
