import pandas as pd
import numpy as np

mc  = pd.read_csv('data/output/mule_chains.csv', parse_dates=['transaction_timestamp'])
wdr = pd.read_csv('data/output/withdrawals.csv', parse_dates=['withdrawal_timestamp'])
fs  = pd.read_csv('data/output/feature_snapshots.csv', parse_dates=['feature_cutoff_timestamp'])
atm = pd.read_csv('data/output/atm_reference.csv')

cashout_cids = list(set(fs['complaint_id']) & set(wdr['complaint_id']))[:5]

print("=== MULE CHAIN TEMPORAL TRACE ===")
for cid in cashout_cids:
    cutoff   = fs.loc[fs['complaint_id']==cid, 'feature_cutoff_timestamp'].iloc[0]
    wdr_row  = wdr.loc[wdr['complaint_id']==cid].iloc[0]
    chain    = mc[mc['complaint_id']==cid].sort_values('hop_number')
    pre      = chain[chain['transaction_timestamp'] <= cutoff]
    post     = chain[chain['transaction_timestamp'] > cutoff]

    cashout_acct = wdr_row['account_id']
    cashout_h3   = wdr_row['h3_cell']

    in_pre = cashout_acct in set(pre['destination_account'])
    in_post= cashout_acct in set(post['destination_account'])

    print(f"  {cid}")
    print(f"    cutoff    : {cutoff}")
    print(f"    wdr_ts    : {wdr_row['withdrawal_timestamp']}")
    print(f"    full hops : {len(chain)} (max={chain['hop_number'].max()})")
    print(f"    pre-cutoff: {len(pre)} hops | dest accts: {pre['destination_account'].tolist()}")
    print(f"    post-cutoff:{len(post)} hops")
    print(f"    cashout acct: {cashout_acct}")
    print(f"    cashout H3  : {cashout_h3}")
    print(f"    cashout in pre-cutoff chain: {in_pre}")
    print(f"    cashout in post-cutoff chain:{in_post}")
    print()

# Also check: do accounts have any location info in any related table?
print("=== ATM REFERENCE STATES ===")
print("Unique states:", sorted(atm['state'].unique()))

# Check if mule chain destination accounts appear in withdrawals for DIFFERENT complaints
# (this would represent historical cashout geography)
print("\n=== HISTORICAL CASHOUT GEOGRAPHY ===")
# For each destination account in mule chains, find if that account has withdrawals
# (potentially from previous/different complaints)
wdr_by_acct = wdr.groupby('account_id')['h3_cell'].apply(list).to_dict()
mc_sample = mc.head(100)
accts_with_hist = [(a, wdr_by_acct.get(a, [])) for a in mc_sample['destination_account']
                   if wdr_by_acct.get(a)]
print(f"Sample: {len(accts_with_hist)} of 100 dest accounts have historical withdrawals")
for acct, h3s in accts_with_hist[:5]:
    print(f"  {acct}: {h3s[:3]}")
