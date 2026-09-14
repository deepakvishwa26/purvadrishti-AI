"""Quick 200-case timing test for Phase 6 pipeline performance."""
import time, yaml, numpy as np, pandas as pd, h3
from collections import Counter
from src.candidate_generator import CandidateGenerator
from src.generator import generate_historical_seed_events

t0 = time.time()
with open("config/phase6_config.yaml") as f:
    p6_cfg = yaml.safe_load(f)
with open("config/generation_config.yaml") as f:
    gen_cfg = yaml.safe_load(f)

OUT = "data/output"
features_df   = pd.read_csv(f"{OUT}/feature_snapshots.csv")
labels_df     = pd.read_csv(f"{OUT}/cashout_labels.csv")
complaints_df = pd.read_csv(f"{OUT}/complaints.csv")
atm_df        = pd.read_csv(f"{OUT}/atm_reference.csv")

cashout_labels = labels_df[labels_df["cashout_occurred"] == True]
cashout_cids   = set(cashout_labels["complaint_id"])

rng = np.random.RandomState(42)
seed_rng = np.random.RandomState(42)
print(f"Load: {time.time()-t0:.2f}s")

t1 = time.time()
events = generate_historical_seed_events(gen_cfg, atm_df, seed_rng)
seed_counter = Counter(e["h3_cell"] for e in events)
print(f"Seed counter: {time.time()-t1:.2f}s")

t2 = time.time()
generator = CandidateGenerator(atm_df, seed_counter, p6_cfg, rng)
print(f"Generator init: {time.time()-t2:.2f}s")
print(f"  Hotspot cache: {len(generator._cand_hotspot_cache)} entries")

actual_h3_by = cashout_labels.groupby("complaint_id")["actual_h3_cell"].apply(lambda x: set(x.dropna())).to_dict()
state_lookup = complaints_df.set_index("complaint_id")["victim_state"].to_dict()

from src.phase6_runner import build_candidate_rows, COMPLAINT_FEAT_COLS

t3 = time.time()
complaints_sorted = features_df[features_df["complaint_id"].isin(cashout_cids)].sort_values("feature_cutoff_timestamp")
sample = complaints_sorted.head(200)
all_rows = []
for feat_row in sample.itertuples(index=False):
    cid = feat_row.complaint_id
    victim_h3 = feat_row.victim_h3_res8
    victim_state = state_lookup[cid]
    victim_lat, victim_lon = h3.cell_to_latlng(victim_h3)
    actual_h3_set = actual_h3_by.get(cid, set())
    comp_feats = {col: getattr(feat_row, col) for col in COMPLAINT_FEAT_COLS if hasattr(feat_row, col)}
    comp_feats["victim_h3_res8"] = victim_h3
    comp_feats["feature_cutoff_timestamp"] = feat_row.feature_cutoff_timestamp
    rows, added = build_candidate_rows(cid, victim_h3, victim_state, victim_lat, victim_lon,
                                        actual_h3_set, comp_feats, generator)
    all_rows.extend(rows)

elapsed = time.time() - t3
total_rows = len(all_rows)
avg_cands = total_rows / 200
print(f"200 cases: {elapsed:.2f}s ({elapsed/200*1000:.1f}ms/case)")
print(f"Total rows: {total_rows} | avg per complaint: {avg_cands:.1f}")
print(f"Projected 8075 cases: {elapsed/200*8075:.0f}s ({elapsed/200*8075/60:.1f}min)")

df = pd.DataFrame(all_rows)
pos = (df["relevance"]==1).sum()
neg = (df["relevance"]==0).sum()
hard = df[(df["relevance"]==0) & (df["cand_dist_km_from_victim"] < 100)]
print(f"Pos: {pos} | Neg: {neg} | Hard neg (<100km): {len(hard)} ({len(hard)/neg*100:.1f}%)")
print(f"Columns: {list(df.columns)}")
print(f"TOTAL: {time.time()-t0:.2f}s")
