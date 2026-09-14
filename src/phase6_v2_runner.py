"""
Phase 6 v2 — Optimised Single-Pass Runner
==========================================
Single pass over 8,075 complaints:
  - Measures natural recall (no injection)
  - Generates candidate rows with positive_source flag
  - Builds DataFrame from per-column dict (fast, not list-of-dicts)

Key optimisations vs previous runner:
  1. Single pass — recall measurement + row generation combined
  2. DataFrame built column-by-column via extend lists (not list-of-dicts)
  3. No redundant generate_candidates calls
"""

import os, sys, time, json, argparse
import numpy as np, pandas as pd, yaml, h3
from collections import Counter, defaultdict
from datetime import datetime
from math import radians, sin, cos, sqrt, atan2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.candidate_generator import CandidateGenerator
from src.phase6_validation import run_phase6_validation

COMPLAINT_FEAT_COLS = [
    "fraud_amount","amount_log","hour","day_of_week","is_weekend","is_night",
    "fraud_type_encoded","mule_chain_depth","mule_velocity","amount_velocity",
    "distance_from_victim","historical_hotspot_density","atm_density",
    "complaint_cluster","time_since_transaction",
]

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2-lat1); dlon = radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R*2*atan2(sqrt(a), sqrt(1-a))

def load_cfg(p):
    with open(p) as f: return yaml.safe_load(f)

def run_v2(config_path="config/phase6_v2_config.yaml"):
    print("="*60)
    print("  HIVE-PREDICT PHASE 6 v2 — SINGLE-PASS OPTIMISED RUNNER")
    print("="*60)

    p6_cfg  = load_cfg(config_path)
    gen_cfg = load_cfg(p6_cfg["source"]["seed_config"])
    seed    = p6_cfg["general"]["seed"]
    out_dir = p6_cfg["general"]["output_dir"]
    src_dir = p6_cfg["source"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()

    print("\n[1/8] Loading source tables...")
    features_df   = pd.read_csv(f"{src_dir}/feature_snapshots.csv")
    labels_df     = pd.read_csv(f"{src_dir}/cashout_labels.csv")
    complaints_df = pd.read_csv(f"{src_dir}/complaints.csv")
    atm_df        = pd.read_csv(f"{src_dir}/atm_reference.csv")

    cashout_labels = labels_df[labels_df["cashout_occurred"] == True].copy()
    cashout_cids   = set(cashout_labels["complaint_id"])
    print(f"  feature_snapshots: {len(features_df):,} | cashout complaints: {len(cashout_cids):,}")

    print("\n[2/8] Rebuilding seed H3 counter...")
    from src.generator import generate_historical_seed_events
    seed_h3_counter = Counter(
        e["h3_cell"] for e in
        generate_historical_seed_events(gen_cfg, atm_df, np.random.RandomState(seed))
    )
    print(f"  {sum(seed_h3_counter.values()):,} seed events, {len(seed_h3_counter):,} unique H3")

    print("\n[3/8] Building v2 CandidateGenerator...")
    rng = np.random.RandomState(seed)
    gen = CandidateGenerator(atm_df, seed_h3_counter, p6_cfg, rng)
    print(f"  ATM H3 pool: {len(gen._atm_h3_cells):,} unique cells")
    print(f"  All-state-ATMs: {gen._use_all_state_atms} | cap: {gen._max_state_atm_count}")
    print(f"  Neighboring-state count: {gen._nbr_state_count} per neighbor")

    actual_h3_by = (
        cashout_labels.groupby("complaint_id")["actual_h3_cell"]
        .apply(lambda x: set(x.dropna())).to_dict()
    )
    state_lookup = complaints_df.set_index("complaint_id")["victim_state"].to_dict()
    complaints_sorted = (
        features_df[features_df["complaint_id"].isin(cashout_cids)]
        .sort_values("feature_cutoff_timestamp").copy()
    )
    n_complaints = len(complaints_sorted)

    print(f"\n[4/8] Single-pass generation + recall measurement ({n_complaints:,} complaints)...")

    # ── Column-wise accumulators (fast DataFrame build) ──────────────────
    col_cid       = []
    col_ch3       = []
    col_rel       = []
    col_psrc      = []
    col_cutoff    = []
    col_vh3       = []
    feat_arrays   = {c: [] for c in COMPLAINT_FEAT_COLS}
    cand_arrays   = {
        "cand_dist_km_from_victim": [],
        "cand_h3_grid_dist":        [],
        "cand_atm_count":           [],
        "cand_atm_density":         [],
        "cand_hotspot_density":     [],
        "cand_in_victim_state":     [],
        "cand_is_victim_h3":        [],
    }

    # ── Recall tracking ──────────────────────────────────────────────────
    recall_comp_hit = 0; recall_h3_hit = 0; recall_h3_tot = 0
    n_posthoc_complaints = 0
    cand_sizes = []
    band_stats  = defaultdict(lambda: [0,0])
    state_stats = defaultdict(lambda: [0,0])

    for i, feat_row in enumerate(complaints_sorted.itertuples(index=False)):
        if (i+1) % 2000 == 0:
            print(f"  {i+1:,}/{n_complaints:,} ({time.time()-t0:.1f}s)")

        cid      = feat_row.complaint_id
        v_h3     = feat_row.victim_h3_res8
        v_state  = state_lookup[cid]
        v_lat, v_lon = h3.cell_to_latlng(v_h3)
        actual_set   = actual_h3_by.get(cid, set())
        cutoff       = feat_row.feature_cutoff_timestamp

        # Generate candidates ONCE
        gen_cands = gen.generate_candidates(v_h3, v_state, v_lat, v_lon)
        added_posthoc = not actual_set.issubset(gen_cands)
        all_cands = sorted(set(gen_cands) | actual_set)  # sorted for reproducibility
        cand_sizes.append(len(all_cands))

        # Recall tracking
        comp_all_hit = True
        for ah3 in actual_set:
            recall_h3_tot += 1
            in_nat = ah3 in gen_cands
            if in_nat:
                recall_h3_hit += 1
            else:
                comp_all_hit = False
            state_stats[v_state][1] += 1
            if in_nat: state_stats[v_state][0] += 1
            a_lat, a_lon = h3.cell_to_latlng(ah3)
            km = _haversine_km(v_lat, v_lon, a_lat, a_lon)
            for lo, hi in [(0,50),(50,150),(150,500),(500,99999)]:
                if lo <= km < hi:
                    band_stats[(lo,hi)][1] += 1
                    if in_nat: band_stats[(lo,hi)][0] += 1
                    break
        if comp_all_hit:
            recall_comp_hit += 1
        if added_posthoc:
            n_posthoc_complaints += 1

        # Complaint-level feature values (same for all candidates of this complaint)
        feat_vals = {c: getattr(feat_row, c, None) for c in COMPLAINT_FEAT_COLS}

        # Build candidate rows (column-wise append — faster than dict-list)
        for ch3 in all_cands:
            relevance = 1 if ch3 in actual_set else 0
            if relevance == 1:
                psrc = "natural" if ch3 in gen_cands else "posthoc_injection"
            else:
                psrc = "negative"

            cf = gen.compute_candidate_features(ch3, v_h3, v_lat, v_lon, v_state)

            col_cid.append(cid)
            col_ch3.append(ch3)
            col_rel.append(relevance)
            col_psrc.append(psrc)
            col_cutoff.append(cutoff)
            col_vh3.append(v_h3)
            for c in COMPLAINT_FEAT_COLS:
                feat_arrays[c].append(feat_vals[c])
            for k, v in cf.items():
                cand_arrays[k].append(v)

    print(f"  Done. Building DataFrame...")
    t_df = time.time()
    candidate_df = pd.DataFrame({
        "complaint_id"             : col_cid,
        "candidate_h3_cell"        : col_ch3,
        "relevance"                : col_rel,
        "positive_source"          : col_psrc,
        "feature_cutoff_timestamp" : col_cutoff,
        "victim_h3_res8"           : col_vh3,
        **{c: feat_arrays[c] for c in COMPLAINT_FEAT_COLS},
        **cand_arrays,
    })
    print(f"  DataFrame built in {time.time()-t_df:.1f}s — {len(candidate_df):,} rows")

    # ── Print recall results ──────────────────────────────────────────────
    recall_comp_pct = recall_comp_hit / n_complaints * 100
    recall_h3_pct   = recall_h3_hit   / recall_h3_tot  * 100
    print(f"\n  === NATURAL RECALL (no injection) ===")
    print(f"  Complaint-level: {recall_comp_hit}/{n_complaints} = {recall_comp_pct:.2f}%")
    print(f"  H3-level:        {recall_h3_hit}/{recall_h3_tot} = {recall_h3_pct:.2f}%")
    print(f"  Post-hoc needed: {n_posthoc_complaints} complaints ({n_posthoc_complaints/n_complaints*100:.1f}%)")

    cs = sorted(cand_sizes)
    print(f"\n  Candidate count: min={cs[0]} median={cs[len(cs)//2]} "
          f"p95={cs[int(len(cs)*0.95)]} max={cs[-1]} avg={sum(cs)/len(cs):.1f}")

    print(f"\n  Recall by distance band:")
    for lo,hi in [(0,50),(50,150),(150,500),(500,99999)]:
        h,t = band_stats[(lo,hi)]
        if t: print(f"    {lo}-{hi}km: {h}/{t} = {h/t*100:.1f}%")

    print(f"\n  Recall by state:")
    for st in sorted(state_stats):
        h,t = state_stats[st]
        print(f"    {st:<30} {h}/{t} = {h/t*100:.1f}%")

    # ── Positive / negative breakdown ─────────────────────────────────────
    pos_df = candidate_df[candidate_df["relevance"]==1]
    neg_df = candidate_df[candidate_df["relevance"]==0]
    nat_pos  = (candidate_df["positive_source"]=="natural").sum()
    phoc_pos = (candidate_df["positive_source"]=="posthoc_injection").sum()
    hard_neg = (neg_df["cand_dist_km_from_victim"] < 100).sum()
    print(f"\n  Positives: {len(pos_df):,} | natural: {nat_pos:,} | post-hoc: {phoc_pos:,}")
    print(f"  Negatives: {len(neg_df):,}")
    print(f"  Positive rate: {len(pos_df)/len(candidate_df)*100:.3f}%")
    print(f"  Hard negatives (<100km): {hard_neg:,} ({hard_neg/len(neg_df)*100:.1f}%)")

    # ── Section 8: Feature distributions ─────────────────────────────────
    print(f"\n  Feature distributions (natural pos / injected pos / negative):")
    feat_cols = ["cand_dist_km_from_victim","cand_h3_grid_dist","cand_atm_count",
                 "cand_atm_density","cand_hotspot_density","cand_in_victim_state"]
    nat_pos_df  = candidate_df[candidate_df["positive_source"]=="natural"]
    phoc_pos_df = candidate_df[candidate_df["positive_source"]=="posthoc_injection"]
    print(f"  {'Feature':<30} {'NatPos':>10} {'InjPos':>10} {'Neg':>10}")
    for col in feat_cols:
        np_m  = nat_pos_df[col].mean()
        pp_m  = phoc_pos_df[col].mean() if len(phoc_pos_df)>0 else 0.0
        ng_m  = neg_df[col].mean()
        print(f"  {col:<30} {np_m:>10.4f} {pp_m:>10.4f} {ng_m:>10.4f}")

    # ── Temporal split ────────────────────────────────────────────────────
    print("\n[5/8] Temporal split...")
    split_cfg = p6_cfg["split"]
    cids_sorted = (
        candidate_df[["complaint_id","feature_cutoff_timestamp"]]
        .drop_duplicates("complaint_id")
        .sort_values("feature_cutoff_timestamp")["complaint_id"].tolist()
    )
    n = len(cids_sorted)
    n_tr = int(n * split_cfg["train_fraction"])
    n_va = int(n * split_cfg["validation_fraction"])
    train_cids = set(cids_sorted[:n_tr])
    val_cids   = set(cids_sorted[n_tr:n_tr+n_va])
    test_cids  = set(cids_sorted[n_tr+n_va:])

    split_map = {}
    for c in train_cids: split_map[c] = "train"
    for c in val_cids:   split_map[c] = "validation"
    for c in test_cids:  split_map[c] = "test"
    candidate_df["split"] = candidate_df["complaint_id"].map(split_map)

    train_df = candidate_df[candidate_df["split"]=="train"].copy()
    val_df   = candidate_df[candidate_df["split"]=="validation"].copy()
    test_df  = candidate_df[candidate_df["split"]=="test"].copy()

    for name, df in [("train",train_df),("validation",val_df),("test",test_df)]:
        ts = pd.to_datetime(df["feature_cutoff_timestamp"])
        pr = df["relevance"].mean()*100
        print(f"  {name:<12}: {df['complaint_id'].nunique():,} complaints | "
              f"{len(df):,} rows | {ts.min().date()} -> {ts.max().date()} | pos_rate={pr:.3f}%")

    # Strict temporal ordering
    tr_max = pd.to_datetime(train_df["feature_cutoff_timestamp"]).max()
    va_min = pd.to_datetime(val_df["feature_cutoff_timestamp"]).min()
    va_max = pd.to_datetime(val_df["feature_cutoff_timestamp"]).max()
    te_min = pd.to_datetime(test_df["feature_cutoff_timestamp"]).min()
    print(f"  max(train)<min(val): {tr_max} < {va_min} -> {tr_max < va_min}")
    print(f"  max(val)<min(test):  {va_max} < {te_min} -> {va_max < te_min}")

    # ── Validation ────────────────────────────────────────────────────────
    print("\n[6/8] Validation (15 checks)...")
    val_report = run_phase6_validation(
        candidate_df=candidate_df, labels_df=labels_df,
        complaints_df=complaints_df,
        train_df=train_df, val_df=val_df, test_df=test_df,
    )
    print(val_report.summary())

    # ── Save ──────────────────────────────────────────────────────────────
    print("\n[7/8] Saving outputs...")
    candidate_df.to_csv(f"{out_dir}/candidate_h3_dataset_v2.csv", index=False)
    train_df.to_csv(f"{out_dir}/train.csv", index=False)
    val_df.to_csv(f"{out_dir}/validation.csv", index=False)
    test_df.to_csv(f"{out_dir}/test.csv", index=False)
    val_report.to_dataframe().to_csv(f"{out_dir}/candidate_validation_report.csv", index=False)

    def grp(df):
        return (df.groupby("complaint_id", sort=False).size()
                .reset_index(name="group_size")
                [["complaint_id","group_size"]].to_dict(orient="records"))
    with open(f"{out_dir}/group_info.json","w") as f:
        json.dump({"description":"XGBoost LTR group sizes v2",
                   "split_unit":"complaint_id",
                   "train":grp(train_df),"validation":grp(val_df),"test":grp(test_df)}, f, indent=2)

    # ── Natural recall JSON ───────────────────────────────────────────────
    recall_report = {
        "version": "6.2.0", "generated": datetime.utcnow().isoformat()+"Z",
        "complaint_level_natural_recall": {"hit": recall_comp_hit, "total": n_complaints,
                                            "pct": round(recall_comp_pct,4)},
        "h3_level_natural_recall": {"hit": recall_h3_hit, "total": recall_h3_tot,
                                     "pct": round(recall_h3_pct,4)},
        "posthoc_injection_complaints": n_posthoc_complaints,
        "recall_by_distance_band": {
            f"{lo}-{hi}km": {"hit": band_stats[(lo,hi)][0], "total": band_stats[(lo,hi)][1],
                             "pct": round(band_stats[(lo,hi)][0]/max(band_stats[(lo,hi)][1],1)*100,2)}
            for lo,hi in [(0,50),(50,150),(150,500),(500,99999)]},
        "recall_by_state": {st: {"hit": h,"total": t,"pct": round(h/t*100,2)}
                            for st,(h,t) in state_stats.items()},
        "candidate_count_stats": {"min": cs[0], "mean": round(sum(cs)/len(cs),1),
                                   "median": cs[len(cs)//2],
                                   "p95": cs[int(len(cs)*0.95)], "max": cs[-1]},
    }
    with open(f"{out_dir}/natural_recall_report.json","w") as f:
        json.dump(recall_report, f, indent=2)
    print(f"  natural_recall_report.json, candidate_h3_dataset_v2.csv, splits, group_info saved")

    elapsed = time.time()-t0
    print(f"\n{'='*60}")
    print(f"  PHASE 6 v2 COMPLETE in {elapsed:.1f}s")
    print(f"  Natural H3 recall:   {recall_h3_pct:.2f}%  (v1: 69.75%)")
    print(f"  Complaint recall:    {recall_comp_pct:.2f}% (v1: 69.46%)")
    print(f"  Post-hoc remaining:  {n_posthoc_complaints} complaints ({n_posthoc_complaints/n_complaints*100:.1f}%)")
    if val_report.all_passed():
        print("  Validation: 15/15 PASS")
    else:
        n_fail = sum(1 for r in val_report.results if not r.passed())
        print(f"  Validation: {n_fail} FAILURES — see report")
    print("="*60)
    return candidate_df, recall_report, val_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase6_v2_config.yaml")
    args = parser.parse_args()
    run_v2(args.config)
