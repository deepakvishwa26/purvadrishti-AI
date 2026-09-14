"""
V2 recall pre-flight: measure natural recall on 200 complaints using v2 config.
NO post-hoc injection. Reports recall by source and distance band.
"""
import sys, time
sys.path.insert(0, "")

import numpy as np, pandas as pd, h3, yaml
from collections import Counter, defaultdict
from math import radians, sin, cos, sqrt, atan2

OUT = "data/output"
labels_df     = pd.read_csv(f"{OUT}/cashout_labels.csv")
features_df   = pd.read_csv(f"{OUT}/feature_snapshots.csv")
complaints_df = pd.read_csv(f"{OUT}/complaints.csv")
atm_df        = pd.read_csv(f"{OUT}/atm_reference.csv")

with open("config/phase6_v2_config.yaml") as f:
    p6_cfg = yaml.safe_load(f)
with open("config/generation_config.yaml") as f:
    gen_cfg = yaml.safe_load(f)

from src.generator import generate_historical_seed_events
from src.candidate_generator import CandidateGenerator

cashout_labels = labels_df[labels_df["cashout_occurred"] == True]
actual_h3_by   = cashout_labels.groupby("complaint_id")["actual_h3_cell"].apply(lambda x: set(x.dropna())).to_dict()
state_lookup   = complaints_df.set_index("complaint_id")["victim_state"].to_dict()

rng      = np.random.RandomState(42)
seed_rng = np.random.RandomState(42)
events   = generate_historical_seed_events(gen_cfg, atm_df, seed_rng)
seed_ctr = Counter(e["h3_cell"] for e in events)

t0 = time.time()
gen = CandidateGenerator(atm_df, seed_ctr, p6_cfg, rng)
print(f"Generator init: {time.time()-t0:.2f}s")

features_sorted = features_df[features_df["complaint_id"].isin(actual_h3_by)].sort_values("feature_cutoff_timestamp")
sample = features_sorted.head(500)   # test on first 500 for speed

all_cand_sizes = []
all_posthoc    = 0
recall_hit     = 0; recall_tot = 0
dist_bands     = {(0,50):([],0),(50,150):([],0),(150,500):([],0),(500,99999):([],0)}

t1 = time.time()
for feat_row in sample.itertuples(index=False):
    cid       = feat_row.complaint_id
    v_h3      = feat_row.victim_h3_res8
    v_state   = state_lookup[cid]
    v_lat, v_lon = h3.cell_to_latlng(v_h3)
    actual_set = actual_h3_by.get(cid, set())

    cands = gen.generate_candidates(v_h3, v_state, v_lat, v_lon)
    all_cand_sizes.append(len(cands))

    for ah3 in actual_set:
        recall_tot += 1
        in_cands = ah3 in cands
        if in_cands:
            recall_hit += 1
        else:
            all_posthoc += 1
        # distance band
        a_lat, a_lon = h3.cell_to_latlng(ah3)
        dlat=radians(a_lat-v_lat); dlon=radians(a_lon-v_lon)
        km = 6371*2*atan2(sqrt(sin(dlat/2)**2+cos(radians(v_lat))*cos(radians(a_lat))*sin(dlon/2)**2),
                          sqrt(1-sin(dlat/2)**2-cos(radians(v_lat))*cos(radians(a_lat))*sin(dlon/2)**2))
        for (lo,hi),(hits, _) in dist_bands.items():
            if lo<=km<hi:
                hits.append(in_cands)
                break

elapsed = time.time()-t1

print(f"\n500-complaint pre-flight (v2, {elapsed:.1f}s):")
print(f"  H3-level recall (no injection): {recall_hit}/{recall_tot} = {recall_hit/recall_tot*100:.2f}%")
print(f"  Still post-hoc needed:          {all_posthoc}/{recall_tot} = {all_posthoc/recall_tot*100:.2f}%")
print(f"  Avg candidates/complaint:       {sum(all_cand_sizes)/len(all_cand_sizes):.1f}")
print(f"  Min/Median/Max:                 {min(all_cand_sizes)}/{sorted(all_cand_sizes)[len(all_cand_sizes)//2]}/{max(all_cand_sizes)}")
print(f"  Projected dataset size:         {int(sum(all_cand_sizes)/len(all_cand_sizes)*8075):,} rows")
print(f"\nRecall by distance band:")
for (lo,hi),(hits,_) in dist_bands.items():
    if hits:
        print(f"  {lo}-{hi}km: {sum(hits)}/{len(hits)} = {sum(hits)/len(hits)*100:.1f}%")
print(f"\nProjected full-run time: {elapsed/500*8075:.0f}s ({elapsed/500*8075/60:.1f}min)")
