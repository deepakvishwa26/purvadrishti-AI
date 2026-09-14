"""
Phase 8 deep inspection:
1. Does accounts table have state/location? (NO based on schema)
2. Do mule chain destination accounts link to ATMs via withdrawals?
3. What % of cashout chain hops are pre-cutoff vs post-cutoff?
4. Synthetic shortcut check: does the cashout account appear in pre-cutoff chain?
5. Is there a suspects.csv with location info?
6. What is the distribution of chain hops pre-cutoff?
"""
import pandas as pd
import numpy as np

mc  = pd.read_csv('data/output/mule_chains.csv', parse_dates=['transaction_timestamp'])
wdr = pd.read_csv('data/output/withdrawals.csv', parse_dates=['withdrawal_timestamp'])
fs  = pd.read_csv('data/output/feature_snapshots.csv', parse_dates=['feature_cutoff_timestamp'])
atm = pd.read_csv('data/output/atm_reference.csv')
sus = pd.read_csv('data/output/suspects.csv', nrows=3)

print("=== suspects.csv columns ===")
print(sus.columns.tolist())
print(sus.head(3).to_string())
print()

# Full suspects schema
sus_full = pd.read_csv('data/output/suspects.csv')
print(f"suspects: {len(sus_full)} rows, {sus_full.columns.tolist()}")
print()

# Build cutoff timestamp lookup
cutoff_map = fs.set_index('complaint_id')['feature_cutoff_timestamp'].to_dict()

# CRITICAL SYNTHETIC CHECK:
# For ALL cashout complaints, check if cashout_account is in pre-cutoff chain
cashout_cids = list(set(mc['complaint_id']) & set(wdr['complaint_id']))
print(f"Total cashout complaints in mule_chains: {len(cashout_cids)}")

n_in_pre  = 0
n_in_post = 0
n_not_in  = 0
pre_hops_dist = []
post_hops_dist = []
cashout_in_pre_by_hop = {}

for cid in cashout_cids[:500]:  # sample 500
    cutoff = cutoff_map.get(cid)
    if cutoff is None:
        continue
    chain    = mc[mc['complaint_id']==cid].sort_values('hop_number')
    pre      = chain[chain['transaction_timestamp'] <= cutoff]
    post     = chain[chain['transaction_timestamp'] > cutoff]
    wdr_acct = wdr.loc[wdr['complaint_id']==cid, 'account_id'].iloc[0]

    pre_hops_dist.append(len(pre))
    post_hops_dist.append(len(post))

    if wdr_acct in set(pre['destination_account']):
        n_in_pre += 1
        hop = chain.loc[chain['destination_account']==wdr_acct, 'hop_number'].iloc[0]
        cashout_in_pre_by_hop[hop] = cashout_in_pre_by_hop.get(hop, 0) + 1
    elif wdr_acct in set(post['destination_account']):
        n_in_post += 1
    else:
        n_not_in += 1

print(f"\nSynthetic shortcut check (500 cashout complaints):")
print(f"  Cashout account in PRE-cutoff chain: {n_in_pre} ({n_in_pre/500*100:.1f}%)")
print(f"  Cashout account in POST-cutoff chain:{n_in_post} ({n_in_post/500*100:.1f}%)")
print(f"  Cashout account not in any chain:    {n_not_in} ({n_not_in/500*100:.1f}%)")
print(f"  Hop distribution of cashout accounts in pre-chain: {cashout_in_pre_by_hop}")
print()
print(f"  Pre-cutoff hops: mean={np.mean(pre_hops_dist):.2f} min={min(pre_hops_dist)} max={max(pre_hops_dist)}")
print(f"  Post-cutoff hops: mean={np.mean(post_hops_dist):.2f} min={min(post_hops_dist)} max={max(post_hops_dist)}")

# Now: are mule-chain destination accounts linked to ATM locations?
# Via withdrawal H3 cells
wdr_by_acct = wdr.groupby('account_id').agg({'h3_cell': list, 'latitude': 'mean', 'longitude': 'mean'})
atm_h3_state = atm.set_index('h3_cell_res8')['state'].to_dict()

# For a sample complaint, what state is the cashout account's withdrawal H3 in?
print("\n=== MULE DESTINATION STATE VIA ATM LOOKUP ===")
sample_cids = cashout_cids[:10]
for cid in sample_cids:
    cutoff = cutoff_map.get(cid)
    if cutoff is None: continue
    chain  = mc[mc['complaint_id']==cid]
    pre    = chain[chain['transaction_timestamp'] <= cutoff]
    dest_accts = pre['destination_account'].tolist()

    # For each dest account, look up its withdrawal state
    dest_states = []
    for acct in dest_accts:
        if acct in wdr_by_acct.index:
            h3s = wdr_by_acct.loc[acct, 'h3_cell']
            states = [atm_h3_state.get(h3, 'unknown') for h3 in h3s if h3 in atm_h3_state]
            if states: dest_states.extend(states)

    cashout_acct = wdr.loc[wdr['complaint_id']==cid, 'account_id'].iloc[0]
    cashout_h3   = wdr.loc[wdr['complaint_id']==cid, 'h3_cell'].iloc[0]
    cashout_state= atm_h3_state.get(cashout_h3, 'unknown')
    comp_state   = pd.read_csv('data/output/complaints.csv').set_index('complaint_id')['victim_state'].get(cid, 'unknown')

    print(f"  {cid}: victim={comp_state} | cashout_state={cashout_state}")
    print(f"    pre-cutoff dest states (via wdr lookup): {dest_states}")
    # LEAKAGE CHECK: is the cashout_acct's withdrawal used here?
    # wdr_by_acct includes ALL withdrawals of that account, including this one
    this_wdr_ts = wdr.loc[wdr['complaint_id']==cid, 'withdrawal_timestamp'].iloc[0]
    print(f"    [LEAKAGE RISK] cashout_acct={cashout_acct} withdrawal at {this_wdr_ts} (after cutoff={cutoff})")
    other_wdr = wdr[(wdr['account_id']==cashout_acct) & (wdr['complaint_id']!=cid)]
    print(f"    cashout_acct other complaints: {len(other_wdr)}")
