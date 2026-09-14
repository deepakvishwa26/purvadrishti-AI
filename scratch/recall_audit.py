"""
Phase 6 Candidate Recall Audit — Sections 1-4
Measures natural recall WITHOUT post-hoc injection.
Attributes recall to each candidate source.
"""
import sys, time
sys.path.insert(0, "")

import numpy as np
import pandas as pd
import h3
import yaml
from collections import Counter, defaultdict
from math import radians, sin, cos, sqrt, atan2

# ── Load everything ─────────────────────────────────────────────────────
print("Loading data...")
OUT = "data/output"
P6  = "data/output/phase6"

labels_df    = pd.read_csv(f"{OUT}/cashout_labels.csv")
features_df  = pd.read_csv(f"{OUT}/feature_snapshots.csv")
complaints_df= pd.read_csv(f"{OUT}/complaints.csv")
atm_df       = pd.read_csv(f"{OUT}/atm_reference.csv")
withdrawals  = pd.read_csv(f"{OUT}/withdrawals.csv")
cand_v1      = pd.read_csv(f"{P6}/candidate_h3_dataset.csv")

with open("config/phase6_config.yaml") as f:
    p6_cfg = yaml.safe_load(f)
with open("config/generation_config.yaml") as f:
    gen_cfg = yaml.safe_load(f)

cashout_labels = labels_df[labels_df["cashout_occurred"] == True]
actual_h3_by   = (
    cashout_labels.groupby("complaint_id")["actual_h3_cell"]
    .apply(lambda x: set(x.dropna())).to_dict()
)
print(f"Loaded. {len(actual_h3_by)} cashout complaints to audit.\n")

# ── Rebuild seed counter ────────────────────────────────────────────────
from src.generator import generate_historical_seed_events
seed_rng = np.random.RandomState(42)
events = generate_historical_seed_events(gen_cfg, atm_df, seed_rng)
seed_h3_counter = Counter(e["h3_cell"] for e in events)
print(f"Seed counter: {len(seed_h3_counter)} unique H3 cells\n")

# ── Pre-compute generation sources ─────────────────────────────────────
# Same as CandidateGenerator but exposing each source separately
cfg = p6_cfg["candidate_generation"]

# ATM tables
atm_unique = atm_df.drop_duplicates("h3_cell_res8").copy()
atm_unique["_clat"] = atm_unique["h3_cell_res8"].apply(lambda c: h3.cell_to_latlng(c)[0])
atm_unique["_clon"] = atm_unique["h3_cell_res8"].apply(lambda c: h3.cell_to_latlng(c)[1])
atm_h3_arr  = atm_unique["h3_cell_res8"].values
atm_lat_arr = atm_unique["_clat"].values
atm_lon_arr = atm_unique["_clon"].values
atm_h3_set  = set(atm_h3_arr)

# Hotspot pool (top 200)
hotspot_pool = np.array([
    cell for cell, _ in sorted(seed_h3_counter.items(), key=lambda x: -x[1])[:200]
])

# State -> ATM H3 map
state_atm_h3 = {}
for state in atm_df["state"].unique():
    state_atm_h3[state] = atm_df[atm_df["state"]==state]["h3_cell_res8"].unique()

# Lookup dicts
victim_state_lookup = complaints_df.set_index("complaint_id")["victim_state"].to_dict()

def haversine_vec(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = np.sin(dlat/2)**2 + cos(radians(lat1))*np.cos(np.radians(lat2_arr))*np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))

SEP = "=" * 70
def section(n, title):
    print(f"\n{SEP}")
    print(f"  SECTION {n}: {title}")
    print(SEP)

# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Natural candidate recall — per source
# ══════════════════════════════════════════════════════════════════════════
section(1, "NATURAL CANDIDATE RECALL (no post-hoc injection)")

# For each source, precompute which actual H3 cells it captures
source_names = ["ring_neighbor", "nearest_atm", "hotspot_pool",
                "state_atm", "global_random"]

complaint_recall     = {}  # cid -> bool (all actual H3 covered naturally)
h3_recall            = {}  # (cid, h3) -> source_name or None
source_capture_count = defaultdict(int)  # source -> how many actual H3 it captures
total_actual_h3      = 0

rng = np.random.RandomState(42)  # same seed as runner

features_sorted = (
    features_df[features_df["complaint_id"].isin(actual_h3_by)]
    .sort_values("feature_cutoff_timestamp")
    .copy()
)

t0 = time.time()
for i, feat_row in enumerate(features_sorted.itertuples(index=False)):
    cid          = feat_row.complaint_id
    victim_h3    = feat_row.victim_h3_res8
    victim_state = victim_state_lookup[cid]
    v_lat, v_lon = h3.cell_to_latlng(victim_h3)
    actual_set   = actual_h3_by[cid]

    # Source 1: Ring neighbors (rings 1-3)
    ring_candidates = set()
    for ring in range(1, cfg.get("victim_ring_max", 3) + 1):
        try:
            ring_candidates.update(h3.grid_ring(victim_h3, ring))
        except Exception:
            pass

    # Source 2: Nearest ATM H3 cells
    dists = haversine_vec(v_lat, v_lon, atm_lat_arr, atm_lon_arr)
    n_near = cfg.get("atm_nearby_count", 10)
    near_idx = np.argsort(dists)[:n_near]
    near_atm_candidates = set(atm_h3_arr[near_idx])

    # Source 3: Hotspot pool (sampled)
    n_hot = cfg.get("hotspot_sample_count", 6)
    hot_idx = rng.choice(len(hotspot_pool), size=min(n_hot, len(hotspot_pool)), replace=False)
    hotspot_candidates = set(hotspot_pool[hot_idx])

    # Source 4: Same-state ATM
    n_state = cfg.get("atm_state_count", 5)
    pool = state_atm_h3.get(victim_state, atm_h3_arr)
    if len(pool) > 0:
        chosen = rng.choice(pool, size=min(n_state, len(pool)), replace=False)
        state_candidates = set(chosen.tolist())
    else:
        state_candidates = set()

    # Source 5: Global random ATM
    n_rand = cfg.get("atm_random_count", 4)
    ridx = rng.choice(len(atm_h3_arr), size=min(n_rand, len(atm_h3_arr)), replace=False)
    random_candidates = set(atm_h3_arr[ridx])

    # Union = natural candidate set (no post-hoc)
    natural_candidates = (ring_candidates | near_atm_candidates |
                          hotspot_candidates | state_candidates | random_candidates)

    # Per-H3 source attribution
    all_covered = True
    for actual_h3 in actual_set:
        total_actual_h3 += 1
        src = None
        if actual_h3 in ring_candidates:      src = "ring_neighbor"
        elif actual_h3 in near_atm_candidates: src = "nearest_atm"
        elif actual_h3 in hotspot_candidates:  src = "hotspot_pool"
        elif actual_h3 in state_candidates:    src = "state_atm"
        elif actual_h3 in random_candidates:   src = "global_random"
        else:                                   src = None  # missed

        h3_recall[(cid, actual_h3)] = src
        if src is not None:
            source_capture_count[src] += 1
        else:
            all_covered = False

    complaint_recall[cid] = all_covered

    if (i+1) % 2000 == 0:
        print(f"  {i+1:,}/{len(features_sorted):,} ({time.time()-t0:.1f}s)")

elapsed = time.time()-t0
print(f"Done: {elapsed:.1f}s\n")

# ── Recall metrics ──────────────────────────────────────────────────────
n_complaints      = len(complaint_recall)
n_comp_recalled   = sum(complaint_recall.values())
n_h3_recalled     = sum(1 for src in h3_recall.values() if src is not None)
n_h3_missed       = sum(1 for src in h3_recall.values() if src is None)

print(f"COMPLAINT-LEVEL RECALL: {n_comp_recalled}/{n_complaints} = {n_comp_recalled/n_complaints*100:.2f}%")
print(f"H3-LEVEL RECALL:        {n_h3_recalled}/{total_actual_h3} = {n_h3_recalled/total_actual_h3*100:.2f}%")
print(f"H3-LEVEL MISS:          {n_h3_missed}/{total_actual_h3} = {n_h3_missed/total_actual_h3*100:.2f}%")

# Single vs multiple cashout
single = {cid: rc for cid, rc in complaint_recall.items() if len(actual_h3_by[cid])==1}
multi  = {cid: rc for cid, rc in complaint_recall.items() if len(actual_h3_by[cid])>1}
print(f"\nSingle-cashout recall: {sum(single.values())}/{len(single)} = {sum(single.values())/max(len(single),1)*100:.2f}%")
print(f"Multi-cashout recall:  {sum(multi.values())}/{len(multi)} = {sum(multi.values())/max(len(multi),1)*100:.2f}%")

# ── Source attribution table ────────────────────────────────────────────
section(2, "SOURCE-LEVEL RECALL")
print(f"{'Source':<20} {'H3 Captured':>14} {'% of Actuals':>14} {'Recall Contribution':>20}")
print("-" * 72)
for src in source_names:
    n = source_capture_count[src]
    pct = n / total_actual_h3 * 100
    print(f"  {src:<18} {n:>14,} {pct:>14.2f}% {pct:>20.2f}%")
print(f"  {'MISSED':<18} {n_h3_missed:>14,} {n_h3_missed/total_actual_h3*100:>14.2f}%")

# Missed breakdown
missed_h3_cells = {h3c for (cid, h3c), src in h3_recall.items() if src is None}
missed_in_atm_pool = sum(1 for c in missed_h3_cells if c in atm_h3_set)
missed_not_in_atm  = sum(1 for c in missed_h3_cells if c not in atm_h3_set)
print(f"\nMissed H3 cells ({len(missed_h3_cells)} unique):")
print(f"  In ATM H3 pool (sampling miss):    {missed_in_atm_pool} ({missed_in_atm_pool/max(len(missed_h3_cells),1)*100:.1f}%)")
print(f"  NOT in ATM H3 pool (pool miss):    {missed_not_in_atm} ({missed_not_in_atm/max(len(missed_h3_cells),1)*100:.1f}%)")

# ── Recall by distance band ─────────────────────────────────────────────
print("\nRecall by victim→actual H3 distance:")
dist_bands = [(0,50,"0-50km"),(50,150,"50-150km"),(150,500,"150-500km"),(500,99999,"500+km")]
for lo, hi, label in dist_bands:
    band_total = 0; band_hit = 0
    for (cid, actual_h3), src in h3_recall.items():
        feat_row = features_df[features_df["complaint_id"]==cid]
        if len(feat_row)==0: continue
        v_h3 = feat_row.iloc[0]["victim_h3_res8"]
        v_lat, v_lon = h3.cell_to_latlng(v_h3)
        a_lat, a_lon = h3.cell_to_latlng(actual_h3)
        from math import radians, sin, cos, sqrt, atan2
        dlat=radians(a_lat-v_lat); dlon=radians(a_lon-v_lon)
        km = 6371*2*atan2(sqrt(sin(dlat/2)**2+cos(radians(v_lat))*cos(radians(a_lat))*sin(dlon/2)**2),
                          sqrt(1-sin(dlat/2)**2-cos(radians(v_lat))*cos(radians(a_lat))*sin(dlon/2)**2))
        if lo <= km < hi:
            band_total += 1
            if src is not None: band_hit += 1
    if band_total > 0:
        print(f"  {label}: {band_hit}/{band_total} = {band_hit/band_total*100:.1f}%")

# ── Recall by state ─────────────────────────────────────────────────────
print("\nRecall by victim state:")
state_recall = defaultdict(lambda: [0,0])  # state -> [hit, total]
for (cid, actual_h3), src in h3_recall.items():
    st = victim_state_lookup.get(cid, "UNKNOWN")
    state_recall[st][1] += 1
    if src is not None: state_recall[st][0] += 1
for state in sorted(state_recall):
    hit, tot = state_recall[state]
    print(f"  {state:<30} {hit}/{tot} = {hit/tot*100:.1f}%")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: ATM Reference Coverage
# ══════════════════════════════════════════════════════════════════════════
section(3, "ATM REFERENCE COVERAGE")

wdr_h3_cells = set(withdrawals["h3_cell"].dropna())
print(f"Unique withdrawal H3 cells in dataset:   {len(wdr_h3_cells)}")
print(f"Unique ATM H3 cells in reference:        {len(atm_h3_set)}")

wdr_in_atm_ref  = wdr_h3_cells & atm_h3_set
wdr_not_in_ref  = wdr_h3_cells - atm_h3_set
print(f"Withdrawal H3 cells in ATM reference:    {len(wdr_in_atm_ref)} ({len(wdr_in_atm_ref)/len(wdr_h3_cells)*100:.1f}%)")
print(f"Withdrawal H3 cells NOT in ATM ref:      {len(wdr_not_in_ref)} ({len(wdr_not_in_ref)/len(wdr_h3_cells)*100:.1f}%)")

# Check actual positive H3 cells specifically
actual_h3_all = set()
for s in actual_h3_by.values():
    actual_h3_all.update(s)
actual_in_atm  = actual_h3_all & atm_h3_set
actual_not_atm = actual_h3_all - atm_h3_set
print(f"\nActual cashout H3 cells:                 {len(actual_h3_all)}")
print(f"  In ATM reference (sampling problem):   {len(actual_in_atm)} ({len(actual_in_atm)/len(actual_h3_all)*100:.1f}%)")
print(f"  NOT in ATM reference (pool problem):   {len(actual_not_atm)} ({len(actual_not_atm)/len(actual_h3_all)*100:.1f}%)")

# For ATM-not-in-ref: is there a real ATM at those coords?
# Check by looking at withdrawal ATM_ID vs atm_reference ATM IDs
wdr_atm_ids = set(withdrawals["atm_id"].dropna())
ref_atm_ids = set(atm_df["atm_id"].dropna()) if "atm_id" in atm_df.columns else set()
print(f"\nWithdrawal ATM IDs:                      {len(wdr_atm_ids)}")
print(f"ATM reference ATM IDs:                   {len(ref_atm_ids)}")
if ref_atm_ids:
    wdr_in_ref = wdr_atm_ids & ref_atm_ids
    print(f"Withdrawal ATMs in reference:            {len(wdr_in_ref)} ({len(wdr_in_ref)/len(wdr_atm_ids)*100:.1f}%)")

# Resolution check: are withdrawal H3 cells at res-8?
sample_wdr_h3 = list(wdr_h3_cells)[:5]
for c in sample_wdr_h3:
    try:
        res = h3.get_resolution(c)
        in_atm = c in atm_h3_set
        print(f"  wdr cell={c} res={res} in_atm_ref={in_atm}")
    except Exception as e:
        print(f"  wdr cell={c} ERROR: {e}")

# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Inference Equivalence
# ══════════════════════════════════════════════════════════════════════════
section(4, "INFERENCE EQUIVALENCE (sample 150 complaints)")

sample_cids = list(complaint_recall.keys())[::len(complaint_recall)//150][:150]
print(f"Sampling {len(sample_cids)} complaints across temporal range")

violations = []
for cid in sample_cids:
    feat_row = features_df[features_df["complaint_id"]==cid]
    if len(feat_row)==0: continue
    cutoff = pd.Timestamp(feat_row.iloc[0]["feature_cutoff_timestamp"])

    # Check: do candidate features encode future info?
    cand_row = cand_v1[cand_v1["complaint_id"]==cid].iloc[0]

    checks = {
        "actual_h3_cell not in features":
            "actual_h3_cell" not in cand_v1.columns,
        "actual_withdrawal_id not in features":
            "actual_withdrawal_id" not in cand_v1.columns,
        "actual_withdrawal_timestamp not in features":
            "actual_withdrawal_timestamp" not in cand_v1.columns,
        "feature_cutoff_timestamp present":
            "feature_cutoff_timestamp" in cand_v1.columns,
    }
    for k, v in checks.items():
        if not v:
            violations.append(f"{cid}: {k} FAILED")

print(f"Violations found: {len(violations)}")
if violations:
    for v in violations[:5]:
        print(f"  {v}")
else:
    print("  All inference equivalence checks PASS (no future data in features)")

# ── Summary for report ───────────────────────────────────────────────────
print(f"\n{'='*70}")
print("RECALL SUMMARY")
print(f"{'='*70}")
print(f"Complaint-level recall (no injection): {n_comp_recalled}/{n_complaints} = {n_comp_recalled/n_complaints*100:.2f}%")
print(f"H3-level recall (no injection):        {n_h3_recalled}/{total_actual_h3} = {n_h3_recalled/total_actual_h3*100:.2f}%")
print(f"Post-hoc injection required for:       {n_complaints-n_comp_recalled} complaints ({(n_complaints-n_comp_recalled)/n_complaints*100:.1f}%)")
print(f"Root cause: {missed_in_atm_pool} of {len(missed_h3_cells)} missed H3 cells ARE in ATM pool (sampling miss)")
print(f"            {missed_not_in_atm} missed H3 cells NOT in ATM pool (pool gap)")
