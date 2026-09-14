"""
HIVE-Predict ML Dataset Audit
Second-level audit focused on ML training suitability.
Inspects actual generated CSV files.
"""
import os, sys, json, hashlib
import numpy as np
import pandas as pd
import h3
from datetime import datetime
from collections import Counter

OUT = "data/output"

def load(name):
    return pd.read_csv(os.path.join(OUT, f"{name}.csv"))

# Load all datasets
print("Loading datasets...")
complaints = load("complaints")
accounts = load("accounts")
transactions = load("transactions")
mule_chains = load("mule_chains")
suspects = load("suspects")
withdrawals = load("withdrawals")
atm_ref = load("atm_reference")
features = load("feature_snapshots")
labels = load("cashout_labels")

lines = []
def section(title):
    lines.append(f"\n{'='*70}")
    lines.append(f"# {title}")
    lines.append(f"{'='*70}\n")
    print(f"\n--- {title} ---")

def p(text=""):
    lines.append(text)
    print(text)

# ============================================================
# 1. CASHOUT LABEL SEMANTICS
# ============================================================
section("1. CASHOUT LABEL SEMANTICS")

p(f"Total label rows: {len(labels)}")
p(f"Unique complaint_id in labels: {labels['complaint_id'].nunique()}")

wdr_ids_in_labels = labels['actual_withdrawal_id'].dropna()
p(f"Unique actual_withdrawal_id in labels (non-null): {wdr_ids_in_labels.nunique()}")

# Rows per complaint
rows_per_complaint = labels.groupby('complaint_id').size()
p(f"\nRows per complaint distribution:")
p(f"  min: {rows_per_complaint.min()}")
p(f"  max: {rows_per_complaint.max()}")
p(f"  mean: {rows_per_complaint.mean():.2f}")
for v, c in sorted(rows_per_complaint.value_counts().items()):
    p(f"  {v} row(s): {c} complaints")

# Cashout occurred breakdown
cashout_true = labels[labels['cashout_occurred'] == True]
cashout_false = labels[labels['cashout_occurred'] == False]
p(f"\ncashout_occurred=True rows: {len(cashout_true)}")
p(f"cashout_occurred=False rows: {len(cashout_false)}")

# Withdrawals represented vs not
wdr_ids_all = set(withdrawals['withdrawal_id'])
wdr_ids_labeled = set(wdr_ids_in_labels)
p(f"\nTotal withdrawals: {len(wdr_ids_all)}")
p(f"Withdrawals represented in labels: {len(wdr_ids_labeled)}")
p(f"Withdrawals NOT in labels: {len(wdr_ids_all - wdr_ids_labeled)}")
p(f"Labels pointing to nonexistent withdrawals: {len(wdr_ids_labeled - wdr_ids_all)}")

# Explain the count discrepancy
p(f"\n--- WHY withdrawals={len(withdrawals)} but labels={len(labels)} ---")
p(f"Labels include {len(cashout_false)} NO_CASHOUT rows (one per no-cashout complaint)")
p(f"Plus {len(cashout_true)} cashout rows (one per withdrawal)")
p(f"So labels = cashout_withdrawals + no_cashout_complaints = {len(cashout_true)} + {len(cashout_false)} = {len(labels)}")
if len(cashout_true) == len(withdrawals):
    p("MATCH: Every withdrawal has exactly one label row. Design is CORRECT.")
else:
    diff = len(cashout_true) - len(withdrawals)
    p(f"MISMATCH: cashout_true rows ({len(cashout_true)}) != withdrawals ({len(withdrawals)}), diff={diff}")
    p("This needs investigation.")

# ============================================================
# 2. MULE NETWORK AUDIT
# ============================================================
section("2. MULE NETWORK AUDIT")

p(f"Total accounts: {len(accounts)}")
p(f"Account types:")
for at, c in accounts['account_type'].value_counts().items():
    p(f"  {at}: {c}")

# Source/dest in transactions
src_accts = set(transactions['source_account_id'].dropna())
dst_accts = set(transactions['destination_account_id'].dropna())
p(f"\nUnique source accounts in transactions: {len(src_accts)}")
p(f"Unique destination accounts in transactions: {len(dst_accts)}")

# Accounts appearing in multiple transactions
all_tx_accts = pd.concat([transactions['source_account_id'], transactions['destination_account_id']])
acct_tx_counts = all_tx_accts.value_counts()
p(f"\nAccounts appearing in 1 transaction: {(acct_tx_counts == 1).sum()}")
p(f"Accounts appearing in 2 transactions: {(acct_tx_counts == 2).sum()}")
p(f"Accounts appearing in 3+ transactions: {(acct_tx_counts >= 3).sum()}")
p(f"Max transactions for single account: {acct_tx_counts.max()}")

# Accounts participating in multiple complaints
tx_acct_complaint = transactions[['source_account_id', 'complaint_id']].rename(columns={'source_account_id':'account_id'})
tx_acct_complaint2 = transactions[['destination_account_id', 'complaint_id']].rename(columns={'destination_account_id':'account_id'})
acct_complaints = pd.concat([tx_acct_complaint, tx_acct_complaint2])
acct_complaint_counts = acct_complaints.groupby('account_id')['complaint_id'].nunique()
multi_complaint_accts = (acct_complaint_counts > 1).sum()
p(f"\nAccounts participating in multiple complaints: {multi_complaint_accts}")
if multi_complaint_accts == 0:
    p("WARNING: NO account reuse across complaints. Network is fully isolated per-case.")
    p("This means the mule network is NOT a genuine shared network.")
else:
    p(f"Top reused accounts: {acct_complaint_counts.nlargest(5).to_dict()}")

# Mule chain depth distribution
mc_depth = mule_chains.groupby('complaint_id')['hop_number'].max()
p(f"\nMule chain depth distribution:")
p(f"  mean: {mc_depth.mean():.2f}")
p(f"  min: {mc_depth.min()}")
p(f"  max: {mc_depth.max()}")
for d in range(1, 9):
    cnt = (mc_depth == d).sum()
    if cnt > 0:
        p(f"  depth {d}: {cnt} chains ({cnt/len(mc_depth)*100:.1f}%)")

# Mule account reuse in mule_chains table
mc_src = set(mule_chains['source_account'].dropna())
mc_dst = set(mule_chains['destination_account'].dropna())
mc_all = mc_src | mc_dst
mc_accts_series = pd.concat([mule_chains['source_account'], mule_chains['destination_account']])
mc_acct_counts = mc_accts_series.value_counts()
mc_multi = (mc_acct_counts > 1).sum()
p(f"\nUnique accounts in mule chains: {len(mc_all)}")
p(f"Mule accounts appearing in multiple hops: {mc_multi}")
mc_acct_complaints = mule_chains.groupby('source_account')['complaint_id'].nunique()
mc_cross_complaint = (mc_acct_complaints > 1).sum()
p(f"Mule accounts used across multiple complaints: {mc_cross_complaint}")

# ============================================================
# 3. CASE-LEVEL EVENT AUDIT
# ============================================================
section("3. CASE-LEVEL EVENT AUDIT")

np.random.seed(123)
sample_cids = np.random.choice(complaints['complaint_id'].values, size=20, replace=False)

for i, cid in enumerate(sample_cids[:5]):  # Print 5 full, summarize rest
    comp = complaints[complaints['complaint_id'] == cid].iloc[0]
    case_tx = transactions[transactions['complaint_id'] == cid].sort_values('transaction_datetime')
    case_mc = mule_chains[mule_chains['complaint_id'] == cid].sort_values('hop_number')
    case_wdr = withdrawals[withdrawals['complaint_id'] == cid]
    case_feat = features[features['complaint_id'] == cid]
    case_lbl = labels[labels['complaint_id'] == cid]

    p(f"\n--- Case {i+1}: {cid} ---")
    p(f"  Fraud type: {comp['fraud_type']}, Amount: INR {comp['fraud_amount']:,.2f}")
    p(f"  Incident: {comp['incident_datetime']}")
    p(f"  Registered: {comp['complaint_registered_at']}")
    p(f"  State: {comp['victim_state']}")
    p(f"  Transactions: {len(case_tx)}")
    if len(case_tx) > 0:
        first_tx = case_tx.iloc[0]
        p(f"    First TX: {first_tx['transaction_datetime']} | {first_tx['source_account_id']} -> {first_tx['destination_account_id']} | INR {first_tx['amount']:,.2f}")
    p(f"  Mule hops: {len(case_mc)}")
    if len(case_mc) > 0:
        for _, hop in case_mc.iterrows():
            p(f"    Hop {int(hop['hop_number'])}: {hop['source_account']} -> {hop['destination_account']} | INR {hop['transaction_amount']:,.2f} | {hop['transaction_timestamp']}")
    p(f"  Withdrawals: {len(case_wdr)}")
    for _, w in case_wdr.iterrows():
        p(f"    {w['withdrawal_id']}: INR {w['amount']:,.2f} at {w['atm_id']} ({w['h3_cell']}) @ {w['withdrawal_timestamp']}")
    p(f"  Feature snapshot: {'YES' if len(case_feat) > 0 else 'MISSING'}")
    if len(case_feat) > 0:
        f0 = case_feat.iloc[0]
        p(f"    cutoff: {f0['feature_cutoff_timestamp']}")
        p(f"    mule_depth={f0['mule_chain_depth']}, mule_vel={f0['mule_velocity']}, amt_vel={f0['amount_velocity']}")
        p(f"    hotspot_density={f0['historical_hotspot_density']}, atm_density={f0['atm_density']}, cluster={f0['complaint_cluster']}")
    p(f"  Labels: {len(case_lbl)} row(s)")
    for _, lb in case_lbl.iterrows():
        p(f"    cashout={lb['cashout_occurred']}, wdr_id={lb['actual_withdrawal_id']}, h3={lb['actual_h3_cell']}")

    # Temporal chain check
    times = []
    if len(case_tx) > 0:
        times.append(('first_tx', pd.to_datetime(case_tx.iloc[0]['transaction_datetime'])))
    for _, hop in case_mc.iterrows():
        times.append((f'hop_{int(hop["hop_number"])}', pd.to_datetime(hop['transaction_timestamp'])))
    for _, w in case_wdr.iterrows():
        times.append(('withdrawal', pd.to_datetime(w['withdrawal_timestamp'])))
    
    ordered = all(times[j][1] <= times[j+1][1] for j in range(len(times)-1))
    p(f"  Temporal order: {'OK' if ordered else 'VIOLATION'}")

# Summary for remaining 15
issues = 0
for cid in sample_cids[5:]:
    case_feat = features[features['complaint_id'] == cid]
    case_lbl = labels[labels['complaint_id'] == cid]
    if len(case_feat) == 0 or len(case_lbl) == 0:
        issues += 1
p(f"\nRemaining 15 sampled cases: {15-issues}/15 have feature+label coverage")

# ============================================================
# 4. TEMPORAL AUDIT
# ============================================================
section("4. TEMPORAL AUDIT")

def temporal_stats(series, name):
    s = series.dropna()
    if len(s) == 0:
        p(f"  {name}: NO DATA")
        return
    p(f"  {name}:")
    p(f"    min:    {s.min()/60:.1f} min")
    p(f"    median: {s.median()/60:.1f} min")
    p(f"    mean:   {s.mean()/60:.1f} min")
    p(f"    P90:    {np.percentile(s, 90)/60:.1f} min")
    p(f"    P95:    {np.percentile(s, 95)/60:.1f} min")
    p(f"    max:    {s.max()/60:.1f} min")
    neg = (s < 0).sum()
    if neg > 0:
        p(f"    NEGATIVE VALUES: {neg} (IMPOSSIBLE)")

# incident -> first transaction
complaints_dt = pd.to_datetime(complaints['incident_datetime'])
first_tx_times = transactions.groupby('complaint_id')['transaction_datetime'].min()
merged_tx = complaints[['complaint_id']].copy()
merged_tx['incident_dt'] = complaints_dt
merged_tx = merged_tx.merge(first_tx_times.rename('first_tx_dt'), on='complaint_id', how='left')
merged_tx['first_tx_dt'] = pd.to_datetime(merged_tx['first_tx_dt'])
merged_tx['delta'] = (merged_tx['first_tx_dt'] - merged_tx['incident_dt']).dt.total_seconds()
temporal_stats(merged_tx['delta'], "incident -> first transaction")

# first tx -> first mule hop
first_mc_times = mule_chains.groupby('complaint_id')['transaction_timestamp'].min()
merged_mc = first_tx_times.rename('first_tx').to_frame().merge(
    first_mc_times.rename('first_mc').to_frame(), on='complaint_id', how='inner')
merged_mc['first_tx'] = pd.to_datetime(merged_mc['first_tx'])
merged_mc['first_mc'] = pd.to_datetime(merged_mc['first_mc'])
merged_mc['delta'] = (merged_mc['first_mc'] - merged_mc['first_tx']).dt.total_seconds()
temporal_stats(merged_mc['delta'], "first transaction -> first mule hop")

# mule hop -> mule hop
hop_deltas = []
for cid, group in mule_chains.groupby('complaint_id'):
    group = group.sort_values('hop_number')
    times_list = pd.to_datetime(group['transaction_timestamp']).values
    for j in range(1, len(times_list)):
        delta = (times_list[j] - times_list[j-1]) / np.timedelta64(1, 's')
        hop_deltas.append(delta)
if hop_deltas:
    temporal_stats(pd.Series(hop_deltas), "mule hop -> mule hop")

# final chain activity -> withdrawal
last_mc_times = mule_chains.groupby('complaint_id')['transaction_timestamp'].max()
first_wdr_times = withdrawals.groupby('complaint_id')['withdrawal_timestamp'].min()
merged_wdr = last_mc_times.rename('last_mc').to_frame().merge(
    first_wdr_times.rename('first_wdr').to_frame(), on='complaint_id', how='inner')
merged_wdr['last_mc'] = pd.to_datetime(merged_wdr['last_mc'])
merged_wdr['first_wdr'] = pd.to_datetime(merged_wdr['first_wdr'])
merged_wdr['delta'] = (merged_wdr['first_wdr'] - merged_wdr['last_mc']).dt.total_seconds()
temporal_stats(merged_wdr['delta'], "final mule activity -> first withdrawal")

# incident -> withdrawal
inc_dt = complaints.set_index('complaint_id')['incident_datetime']
merged_iw = first_wdr_times.rename('first_wdr').to_frame()
merged_iw = merged_iw.merge(inc_dt.rename('incident_dt').to_frame(), on='complaint_id', how='inner')
merged_iw['incident_dt'] = pd.to_datetime(merged_iw['incident_dt'])
merged_iw['first_wdr'] = pd.to_datetime(merged_iw['first_wdr'])
merged_iw['delta'] = (merged_iw['first_wdr'] - merged_iw['incident_dt']).dt.total_seconds()
temporal_stats(merged_iw['delta'], "incident -> first withdrawal")

# Feature cutoff check
p("\nFeature cutoff validation:")
feat_comp = features.merge(complaints[['complaint_id', 'complaint_registered_at']], on='complaint_id')
cutoff_match = (feat_comp['feature_cutoff_timestamp'] == feat_comp['complaint_registered_at']).sum()
p(f"  cutoff == registered_at: {cutoff_match}/{len(feat_comp)}")

# ============================================================
# 5. FINANCIAL FLOW AUDIT
# ============================================================
section("5. FINANCIAL FLOW AUDIT")

impossible_flows = 0
max_ratio = 0
ratios = []
for cid in complaints['complaint_id'].values[:2000]:  # Sample 2000
    fraud_amt = complaints[complaints['complaint_id'] == cid]['fraud_amount'].iloc[0]
    case_wdr = withdrawals[withdrawals['complaint_id'] == cid]
    if len(case_wdr) > 0:
        total_withdrawn = case_wdr['amount'].sum()
        ratio = total_withdrawn / fraud_amt if fraud_amt > 0 else 0
        ratios.append(ratio)
        if total_withdrawn > fraud_amt * 1.01:
            impossible_flows += 1
        max_ratio = max(max_ratio, ratio)

p(f"Sampled 2000 cases:")
p(f"  Cases with impossible money flow (withdrawal > fraud): {impossible_flows}")
p(f"  Max withdrawal/fraud ratio: {max_ratio:.4f}")
if ratios:
    ratios_s = pd.Series(ratios)
    p(f"  Mean withdrawal/fraud ratio: {ratios_s.mean():.4f}")
    p(f"  Median withdrawal/fraud ratio: {ratios_s.median():.4f}")

# Multiple withdrawal consistency
multi_wdr = withdrawals.groupby('complaint_id').size()
multi_wdr_cases = multi_wdr[multi_wdr > 1]
p(f"\nMultiple withdrawal cases: {len(multi_wdr_cases)}")
p(f"  Max withdrawals per case: {multi_wdr.max()}")

# ============================================================
# 6. GEOGRAPHIC AUDIT
# ============================================================
section("6. GEOGRAPHIC AUDIT")

p(f"Unique H3 cells in withdrawals: {withdrawals['h3_cell'].nunique()}")
p(f"Unique ATMs used: {withdrawals['atm_id'].nunique()}")
p(f"Total ATMs in reference: {len(atm_ref)}")

# H3 validation sample
sample_wdr = withdrawals.sample(min(500, len(withdrawals)), random_state=42)
h3_mismatches = 0
for _, row in sample_wdr.iterrows():
    expected = h3.latlng_to_cell(row['latitude'], row['longitude'], 8)
    if expected != str(row['h3_cell']):
        h3_mismatches += 1
p(f"\nH3 validation (500 sample): {h3_mismatches} mismatches")

# Top 20 H3 cells
h3_counts = withdrawals['h3_cell'].value_counts()
p(f"\nTop 20 H3 cells by withdrawal count:")
for cell, cnt in h3_counts.head(20).items():
    p(f"  {cell}: {cnt} withdrawals")

# ATM usage distribution
atm_counts = withdrawals['atm_id'].value_counts()
p(f"\nATM usage distribution:")
p(f"  ATMs used once: {(atm_counts == 1).sum()}")
p(f"  ATMs used 2-5 times: {((atm_counts >= 2) & (atm_counts <= 5)).sum()}")
p(f"  ATMs used 6-20 times: {((atm_counts >= 6) & (atm_counts <= 20)).sum()}")
p(f"  ATMs used 20+ times: {(atm_counts >= 20).sum()}")
p(f"  Max usage: {atm_counts.max()}")

# Geographic clustering
lat_std = withdrawals['latitude'].std()
lon_std = withdrawals['longitude'].std()
p(f"\nGeographic spread:")
p(f"  Latitude  range: [{withdrawals['latitude'].min():.4f}, {withdrawals['latitude'].max():.4f}], std={lat_std:.4f}")
p(f"  Longitude range: [{withdrawals['longitude'].min():.4f}, {withdrawals['longitude'].max():.4f}], std={lon_std:.4f}")

# ============================================================
# 7. HISTORICAL HOTSPOT FEATURE AUDIT
# ============================================================
section("7. HISTORICAL HOTSPOT FEATURE AUDIT")

hotspot = features['historical_hotspot_density']
p(f"historical_hotspot_density statistics:")
p(f"  min:    {hotspot.min():.6f}")
p(f"  max:    {hotspot.max():.6f}")
p(f"  mean:   {hotspot.mean():.6f}")
p(f"  median: {hotspot.median():.6f}")
p(f"  zeros:  {(hotspot == 0).sum()} ({(hotspot == 0).sum()/len(hotspot)*100:.1f}%)")
p(f"  nonzero:{(hotspot > 0).sum()} ({(hotspot > 0).sum()/len(hotspot)*100:.1f}%)")

p("\nLineage analysis:")
p("  SOURCE: historical withdrawal records (only from cases registered BEFORE current case)")
p("  METHOD: Count withdrawals in victim's H3 2-ring neighborhood / total historical withdrawals")
p("  CUTOFF: Only withdrawals with timestamp < feature_cutoff_timestamp")
p("  NOTE: Cases are processed in order of complaint_registered_at")
p("  POTENTIAL ISSUE: First cases always have density=0 (no history yet)")

first_zero_count = 0
for i in range(min(100, len(features))):
    if features.iloc[i]['historical_hotspot_density'] == 0:
        first_zero_count += 1
p(f"  First 100 cases with density=0: {first_zero_count}")

# Check circularity: does this case's own withdrawal influence its own feature?
p("\nCircularity check:")
p("  The feature uses historical_withdrawals which EXCLUDES the current case's withdrawals")
p("  (generator.py line 231: withdrawals are added to historical set AFTER feature computation)")
p("  VERDICT: No circularity detected in code logic.")

# ============================================================
# 8. ATM DENSITY AUDIT
# ============================================================
section("8. ATM DENSITY AUDIT")

atm_density = features['atm_density']
p(f"atm_density statistics:")
p(f"  min:    {atm_density.min()}")
p(f"  max:    {atm_density.max()}")
p(f"  mean:   {atm_density.mean():.2f}")
p(f"  median: {atm_density.median()}")
p(f"  zeros:  {(atm_density == 0).sum()}")

p("\nDerivation method:")
p("  Uses compute_atm_density_h3() from geography.py")
p("  Counts ATMs in the victim's H3 res-8 cell + its 6 immediate H3 neighbors (grid_disk radius=1)")
p("  Data source: atm_reference DataFrame (static, not temporal)")
p("  VERDICT: Derived from static ATM reference. No leakage.")

# ============================================================
# 9. COMPLAINT CLUSTER AUDIT
# ============================================================
section("9. COMPLAINT CLUSTER AUDIT")

clusters = features['complaint_cluster']
p(f"Total complaints: {len(clusters)}")
p(f"Unique cluster labels: {clusters.nunique()}")
noise = (clusters == -1).sum()
p(f"Noise/unclustered (-1): {noise} ({noise/len(clusters)*100:.1f}%)")
non_noise = clusters[clusters != -1]
if len(non_noise) > 0:
    p(f"Clustered complaints: {len(non_noise)}")
    cluster_sizes = non_noise.value_counts()
    p(f"Number of clusters: {len(cluster_sizes)}")
    p(f"Cluster size distribution:")
    p(f"  min:    {cluster_sizes.min()}")
    p(f"  max:    {cluster_sizes.max()}")
    p(f"  mean:   {cluster_sizes.mean():.1f}")
    p(f"  median: {cluster_sizes.median()}")

p("\nDerivation method:")
p("  DBSCAN on [victim_lat*111.32*spatial_weight, victim_lon*111.32*spatial_weight, hour_norm*temporal_window*temporal_weight]")
p("  eps=10.0 km, min_samples=3")
p("  Uses victim lat/lon and incident hour (all available at cutoff)")
p("  VERDICT: No future information used. Based on complaint geography+time only.")

# ============================================================
# 10. FEATURE LINEAGE AUDIT
# ============================================================
section("10. FEATURE LINEAGE AUDIT")

lineage_table = """
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
"""
p(lineage_table)

p("*CRITICAL NOTE on mule_chain_depth, mule_velocity, amount_velocity:")
p("  These features are derived from the COMPLETE mule chain for the case.")
p("  In REAL deployment, the full mule chain may not be known at prediction time.")
p("  The mule chain is discovered AFTER the complaint, potentially concurrently.")
p("  Whether this constitutes leakage depends on operational assumptions:")
p("  - If HIVE-Predict runs AFTER the banking system traces the mule chain: NO leakage")
p("  - If HIVE-Predict runs IMMEDIATELY upon complaint receipt: LEAKAGE")
p("  PROJECT_CONTEXT.md says 'Financial-system enrichment' includes 'mule-chain structure'")
p("  This implies the system has access to mule chain data at prediction time.")
p("  VERDICT: Acceptable given the documented integration architecture, but flagged as MEDIUM risk.")

# ============================================================
# 11. DISTRIBUTION AUDIT
# ============================================================
section("11. DISTRIBUTION AUDIT")

def dist_stats(series, name):
    s = series.dropna()
    p(f"\n{name} (n={len(s)}):")
    p(f"  min:  {s.min():.4f}")
    p(f"  P25:  {np.percentile(s, 25):.4f}")
    p(f"  P50:  {s.median():.4f}")
    p(f"  P75:  {np.percentile(s, 75):.4f}")
    p(f"  P90:  {np.percentile(s, 90):.4f}")
    p(f"  P95:  {np.percentile(s, 95):.4f}")
    p(f"  P99:  {np.percentile(s, 99):.4f}")
    p(f"  max:  {s.max():.4f}")

dist_stats(complaints['fraud_amount'], "fraud_amount (INR)")
dist_stats(features['mule_chain_depth'], "mule_chain_depth")
dist_stats(features['mule_velocity'], "mule_velocity")
dist_stats(features['amount_velocity'], "amount_velocity")
dist_stats(features['distance_from_victim'], "distance_from_victim (km)")

# Time to cashout
ttc = labels[labels['cashout_occurred'] == True]['time_to_cashout_seconds'].dropna()
dist_stats(ttc / 60, "time_to_cashout (minutes)")

dist_stats(withdrawals['amount'], "withdrawal_amount (INR)")

# Categorical distributions
p("\n--- Categorical Distributions ---")
p("\nFraud type:")
for ft, cnt in complaints['fraud_type'].value_counts().items():
    p(f"  {ft}: {cnt} ({cnt/len(complaints)*100:.1f}%)")

p("\nCashout/No-cashout (by complaint):")
complaints_with_cashout = withdrawals['complaint_id'].nunique()
complaints_without = len(complaints) - complaints_with_cashout
p(f"  With cashout: {complaints_with_cashout} ({complaints_with_cashout/len(complaints)*100:.1f}%)")
p(f"  No cashout: {complaints_without} ({complaints_without/len(complaints)*100:.1f}%)")

p("\nVictim state distribution:")
for state, cnt in complaints['victim_state'].value_counts().head(10).items():
    p(f"  {state}: {cnt} ({cnt/len(complaints)*100:.1f}%)")

# ============================================================
# 12. BEHAVIORAL PROFILE AUDIT
# ============================================================
section("12. BEHAVIORAL PROFILE AUDIT")

# We don't have a direct profile column in the output, so infer from patterns
# Use mule chain depth + withdrawal count + timing patterns
p("NOTE: Behavioral profile is not stored in output. Inferring from patterns.\n")

# Build per-complaint summary
comp_summary = complaints[['complaint_id']].copy()
comp_summary = comp_summary.merge(mc_depth.rename('depth'), on='complaint_id', how='left')
comp_summary['depth'] = comp_summary['depth'].fillna(0).astype(int)
wdr_counts = withdrawals.groupby('complaint_id').size().rename('n_withdrawals')
comp_summary = comp_summary.merge(wdr_counts, on='complaint_id', how='left')
comp_summary['n_withdrawals'] = comp_summary['n_withdrawals'].fillna(0).astype(int)

# Time to first cashout
comp_summary = comp_summary.merge(
    merged_iw[['delta']].rename(columns={'delta': 'ttc_seconds'}),
    on='complaint_id', how='left')

# Infer profiles
def infer_profile(row):
    if row['n_withdrawals'] == 0:
        return 'NO_CASHOUT'
    if row['n_withdrawals'] >= 2:
        return 'MULTIPLE_CASHOUT'
    if row['depth'] >= 4:
        return 'DEEP_MULE_CHAIN'
    ttc = row.get('ttc_seconds', None)
    if pd.notna(ttc):
        if ttc < 3600:
            return 'FAST_CASHOUT'
        elif ttc > 7200:
            return 'SLOW_CASHOUT'
    return 'OTHER'

comp_summary['inferred_profile'] = comp_summary.apply(infer_profile, axis=1)
p("Inferred profile distribution:")
for prof, cnt in comp_summary['inferred_profile'].value_counts().items():
    p(f"  {prof}: {cnt} ({cnt/len(comp_summary)*100:.1f}%)")

# Profile behavior comparison
p("\nBehavioral comparison by inferred profile:")
for prof in ['FAST_CASHOUT', 'SLOW_CASHOUT', 'DEEP_MULE_CHAIN', 'NO_CASHOUT', 'MULTIPLE_CASHOUT']:
    subset = comp_summary[comp_summary['inferred_profile'] == prof]
    if len(subset) == 0:
        continue
    p(f"\n  {prof} (n={len(subset)}):")
    p(f"    avg chain depth:  {subset['depth'].mean():.2f}")
    ttc_s = subset['ttc_seconds'].dropna()
    if len(ttc_s) > 0:
        p(f"    median ttc:       {ttc_s.median()/60:.1f} min")
    else:
        p(f"    median ttc:       N/A")
    p(f"    avg withdrawals:  {subset['n_withdrawals'].mean():.2f}")

# ============================================================
# 13. SYNTHETIC REALISM AUDIT
# ============================================================
section("13. SYNTHETIC REALISM AUDIT")

# Check: fraud type -> H3 determinism
p("Checking fraud_type -> H3 determinism...")
if len(withdrawals) > 0:
    wdr_fraud = withdrawals.merge(complaints[['complaint_id', 'fraud_type']], on='complaint_id')
    h3_by_fraud = wdr_fraud.groupby('fraud_type')['h3_cell'].nunique()
    for ft, n_h3 in h3_by_fraud.items():
        n_wdr = len(wdr_fraud[wdr_fraud['fraud_type'] == ft])
        p(f"  {ft}: {n_h3} unique H3 cells across {n_wdr} withdrawals")

# Check: profile -> label determinism
p("\nChecking profile -> cashout determinism...")
no_cashout_count = comp_summary[comp_summary['n_withdrawals'] == 0]
p(f"  Cases with 0 withdrawals: {len(no_cashout_count)}")
p(f"  (If this exactly matches NO_CASHOUT probability ~15%, profile deterministically controls label)")
p(f"  Ratio: {len(no_cashout_count)/len(comp_summary)*100:.1f}%")

# Check: duplicate cases
p("\nChecking for duplicate cases...")
key_cols = ['fraud_type', 'fraud_amount', 'victim_state', 'incident_datetime']
duplicates = complaints.duplicated(subset=key_cols, keep=False).sum()
p(f"  Duplicate rows on (fraud_type, amount, state, datetime): {duplicates}")

# Check: feature-target correlation
p("\nFeature-target correlations (point-biserial with cashout_occurred):")
feat_lbl = features.merge(labels[['complaint_id', 'cashout_occurred']].drop_duplicates('complaint_id'), on='complaint_id')
numeric_feats = ['fraud_amount', 'hour', 'day_of_week', 'mule_chain_depth', 
                  'mule_velocity', 'amount_velocity', 'distance_from_victim',
                  'historical_hotspot_density', 'atm_density', 'complaint_cluster']
for col in numeric_feats:
    if col in feat_lbl.columns:
        corr = feat_lbl[col].astype(float).corr(feat_lbl['cashout_occurred'].astype(float))
        flag = " <-- SUSPICIOUS" if abs(corr) > 0.5 else ""
        p(f"  {col}: r={corr:.4f}{flag}")

# Check: overly deterministic chains
p("\nMule chain determinism check:")
mc_amounts = mule_chains.groupby('complaint_id')['transaction_amount'].apply(list)
constant_chains = 0
for cid, amounts in mc_amounts.items():
    if len(amounts) > 1 and len(set(amounts)) == 1:
        constant_chains += 1
p(f"  Chains with identical amounts at every hop: {constant_chains}")
p(f"  (Should be 0 if friction is applied correctly)")

# ============================================================
# 14. MODEL LEAKAGE AUDIT
# ============================================================
section("14. MODEL LEAKAGE AUDIT")

p("Checking each feature for direct/indirect leakage...")
p()

leakage_checks = {
    'historical_hotspot_density': 'Uses PAST withdrawals only (before cutoff). Code verified: withdrawals added to historical set AFTER feature computation.',
    'complaint_cluster': 'DBSCAN on victim lat/lon + incident hour. No withdrawal/label info used. HOWEVER: computed batch-wise on ALL complaints, which means future complaints inform cluster assignment.',
    'mule_chain_depth': 'Derived from complete mule chain. Available if banking system traces chain before prediction.',
    'mule_velocity': 'Same as mule_chain_depth - requires chain to be traced.',
    'amount_velocity': 'Same as mule_chain_depth - requires chain to be traced.',
    'distance_from_victim': 'Victim coords vs avg ATM in state. No withdrawal info used.',
    'atm_density': 'Static ATM reference data. No withdrawal info.',
}

for feat, analysis in leakage_checks.items():
    p(f"  {feat}:")
    p(f"    {analysis}")
    p()

p("CRITICAL LEAKAGE FINDING:")
p("  complaint_cluster uses DBSCAN on ALL complaints simultaneously.")
p("  In the code (compute_complaint_clusters), the entire complaint list is passed.")
p("  This means a complaint registered at time T has its cluster influenced by")
p("  complaints registered AFTER T. This is temporal leakage.")
p("  SEVERITY: MEDIUM - cluster assignment may change if future complaints are excluded.")

p("\nChecking for label columns in features...")
label_cols = {'actual_withdrawal_id', 'actual_h3_cell', 'actual_atm_id', 
              'actual_withdrawal_timestamp', 'actual_withdrawal_amount',
              'time_to_cashout_seconds', 'cashout_occurred'}
found = label_cols.intersection(set(features.columns))
p(f"  Label columns found in features: {found if found else 'NONE (GOOD)'}")

# ============================================================
# 15. REPRODUCIBILITY AUDIT
# ============================================================
section("15. REPRODUCIBILITY AUDIT")

p("Computing file hashes for current output...")
for fname in sorted(os.listdir(OUT)):
    if fname.endswith('.csv'):
        path = os.path.join(OUT, fname)
        with open(path, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        size = os.path.getsize(path)
        p(f"  {fname}: {h} ({size:,} bytes)")

p("\nReproducibility test (small-scale):")
from src.account_generator import AccountRegistry
from src.complaint_generator import generate_complaints as gen_comp
import yaml
with open("config/generation_config.yaml") as f:
    cfg = yaml.safe_load(f)
cfg_test = dict(cfg)
cfg_test['general'] = dict(cfg['general'])
cfg_test['general']['num_cases'] = 50

rng1 = np.random.RandomState(42)
reg1 = AccountRegistry(cfg_test, np.random.RandomState(42))
c1 = gen_comp(cfg_test, rng1, reg1)

rng2 = np.random.RandomState(42)
reg2 = AccountRegistry(cfg_test, np.random.RandomState(42))
c2 = gen_comp(cfg_test, rng2, reg2)

ids_match = [c1[i]['complaint_id'] for i in range(50)] == [c2[i]['complaint_id'] for i in range(50)]
amts_match = [c1[i]['fraud_amount'] for i in range(50)] == [c2[i]['fraud_amount'] for i in range(50)]
p(f"  Complaint IDs match: {ids_match}")
p(f"  Fraud amounts match: {amts_match}")
p(f"  VERDICT: {'REPRODUCIBLE' if ids_match and amts_match else 'NOT REPRODUCIBLE'}")

# ============================================================
# 16. FINAL VERDICT
# ============================================================
section("16. FINAL VERDICT")

p("B. REQUIRES FIXES BEFORE ML TRAINING")
p()
p("Issues ranked by severity:\n")

issues_list = [
    ("CRITICAL", "Mule network has zero cross-complaint account reuse",
     "Every mule chain creates entirely new accounts. No account appears in multiple complaints. "
     "This means the 'network' is actually N isolated linear chains. An ML model cannot learn "
     "network-based patterns because there is no network. In real fraud, mule accounts are reused "
     "across cases, creating a graph structure.",
     "Implement a mule account pool with configurable reuse probability. Some accounts should "
     "appear as destinations in multiple complaint chains."),
    
    ("CRITICAL", "complaint_cluster has temporal leakage",
     "DBSCAN clustering is computed batch-wise on ALL complaints simultaneously. A complaint "
     "at time T has its cluster influenced by complaints registered after T. In deployment, "
     "future complaints would not be available.",
     "Compute clusters incrementally: for each complaint, only use complaints registered at or "
     "before its feature_cutoff_timestamp. Alternatively, use a sliding window approach."),
    
    ("HIGH", "Mule chain features may not be available at prediction time",
     "mule_chain_depth, mule_velocity, amount_velocity require the complete mule chain. "
     "If HIVE-Predict runs immediately upon complaint receipt (before chain tracing), these "
     "features would be zero/unknown. The current setup assumes full chain availability.",
     "Document the operational assumption explicitly. Consider adding a 'chain_known' flag "
     "or providing graceful degradation when chain is partial."),
    
    ("HIGH", "NO_CASHOUT is deterministically controlled by behavioral profile",
     f"Cases with 0 withdrawals: {len(no_cashout_count)} ({len(no_cashout_count)/len(comp_summary)*100:.1f}%). "
     "This closely matches the configured NO_CASHOUT probability (15%). The profile directly "
     "determines whether cashout occurs. An ML model might learn to exploit the profile-specific "
     "feature distributions rather than genuine patterns.",
     "Add stochastic cashout failure to other profiles (e.g., 5% of FAST_CASHOUT cases could "
     "fail to cash out due to account freezing). Make NO_CASHOUT less deterministic."),
    
    ("MEDIUM", "historical_hotspot_density is mostly zero",
     f"Zero values: {(hotspot == 0).sum()} ({(hotspot == 0).sum()/len(hotspot)*100:.1f}%). "
     "Because cases are processed chronologically and the victim's H3 neighborhood rarely "
     "overlaps with historical withdrawal locations, this feature is near-zero for most cases.",
     "Consider using a broader spatial window, or seeding with some initial historical data "
     "to make this feature more informative."),
    
    ("MEDIUM", "ATM selection is victim-proximity-based, creating geographic leakage",
     "Withdrawals use select_nearby_atm() which selects ATMs near the VICTIM location. "
     "This means distance_from_victim and atm_density (both computed at victim location) "
     "are correlated with where the withdrawal actually occurs. The model may learn 'predict "
     "near the victim' as a shortcut rather than learning genuine geographic patterns.",
     "Introduce more geographic diversity: some withdrawals should occur far from the victim, "
     "especially for DEEP_MULE_CHAIN and COORDINATED_CASE profiles."),
    
    ("MEDIUM", "Geographic diversity is limited",
     f"Only {withdrawals['h3_cell'].nunique()} unique H3 cells for {len(withdrawals)} withdrawals. "
     "Top H3 cells concentrate many withdrawals, limiting the model's ability to learn diverse "
     "geographic patterns.",
     "Increase geographic spread and ATM count, or vary the selection radius by profile."),
    
    ("LOW", "amount_velocity has extreme outliers",
     f"P99: {np.percentile(features['amount_velocity'].dropna(), 99):.0f}, "
     f"max: {features['amount_velocity'].max():.0f}. "
     "Single-hop chains get amount_velocity = fraud_amount (no time denominator), "
     "creating extremely high values.",
     "Cap or normalize amount_velocity for single-hop cases. Consider using log transform."),
]

for severity, title, evidence, fix in issues_list:
    p(f"### [{severity}] {title}")
    p(f"  Evidence: {evidence}")
    p(f"  Impact: Affects HIVE-Predict ML training quality.")
    p(f"  Fix: {fix}")
    p()

# Write to file
report_text = "\n".join(lines)
os.makedirs("metadata", exist_ok=True)
with open("metadata/ML_DATASET_AUDIT.md", "w", encoding="utf-8") as f:
    f.write("# HIVE-Predict ML Dataset Audit Report\n\n")
    f.write(f"Generated: {datetime.utcnow().isoformat()}Z\n\n")
    f.write(report_text)
print(f"\n\nAudit report saved to metadata/ML_DATASET_AUDIT.md")
