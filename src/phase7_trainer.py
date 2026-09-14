"""
HIVE-Predict Phase 7 — XGBoost Learning-to-Rank Trainer
=========================================================
1.  Pre-training sanity checks (posthoc counts, feature list, temporal validity)
2.  Data loading from phase6_v2 temporal splits
3.  XGBoost rank:ndcg training with complaint-level groups
4.  Three-tier evaluation framework:
      Tier 1 — Generator recall (candidate set membership, no model)
      Tier 2 — Conditional ranking (given actual H3 in candidate set)
      Tier 3 — End-to-end (inference-aligned, natural positives only)
5.  Overall + stratified metrics (fraud_type, state, distance, time, amount, depth)
6.  Complete output suite saved to data/output/phase7/
"""

import os, sys, time, json, warnings
import numpy as np, pandas as pd, yaml
from itertools import groupby as itr_groupby
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
except ImportError:
    sys.exit("ERROR: xgboost not installed. Run: pip install xgboost")

# ── Constants ─────────────────────────────────────────────────────────────────

COMPLAINT_FEATS = [
    "fraud_amount","amount_log","hour","day_of_week","is_weekend","is_night",
    "fraud_type_encoded","mule_chain_depth","mule_velocity","amount_velocity",
    "distance_from_victim","historical_hotspot_density","atm_density",
    "complaint_cluster","time_since_transaction",
]
CANDIDATE_FEATS = [
    "cand_dist_km_from_victim","cand_h3_grid_dist","cand_atm_count",
    "cand_atm_density","cand_hotspot_density","cand_in_victim_state",
    "cand_is_victim_h3",
]
MODEL_FEATURES = COMPLAINT_FEATS + CANDIDATE_FEATS  # 22 total
assert len(MODEL_FEATURES) == 22

EXCLUDE_COLS = {
    "complaint_id","candidate_h3_cell","relevance","split",
    "positive_source","victim_h3_res8","feature_cutoff_timestamp",
}

# ── Metric helpers ────────────────────────────────────────────────────────────

def _ndcg_at_k(sorted_rel, ideal_rel, k=5):
    """NDCG@k given already-sorted relevance (desc) and ideal relevance vector."""
    dcg  = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(sorted_rel[:k]))
    idcg = sum((2**r - 1) / np.log2(i+2) for i, r in enumerate(sorted(ideal_rel, reverse=True)[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_complaints(df, model, split_name):
    """
    Per-complaint three-tier evaluation.
    Returns a DataFrame with one row per complaint, all metric columns.
    """
    rows = []
    for cid, grp in df.groupby("complaint_id", sort=False):
        X = grp[MODEL_FEATURES].values.astype(np.float32)
        scores = model.predict(xgb.DMatrix(X, feature_names=MODEL_FEATURES))

        rel      = grp["relevance"].values
        psrc     = grp["positive_source"].values

        # Binary arrays
        is_pos     = rel == 1
        is_nat_pos = (rel == 1) & (psrc == "natural")
        is_inj_pos = (rel == 1) & (psrc == "posthoc_injection")

        # Sorted indices (descending score)
        sidx = np.argsort(-scores)
        sorted_rel     = rel[sidx]
        sorted_nat_rel = is_nat_pos.astype(int)[sidx]

        # First rank of each positive type (0-indexed)
        all_pos_ranks = [i for i, j in enumerate(sidx) if is_pos[j]]
        nat_pos_ranks = [i for i, j in enumerate(sidx) if is_nat_pos[j]]

        # Complaint metadata
        has_natural   = bool(is_nat_pos.any())
        has_injected  = bool(is_inj_pos.any())
        n_pos         = int(is_pos.sum())
        n_nat_pos     = int(is_nat_pos.sum())

        row = {
            "complaint_id"     : cid,
            "split"            : split_name,
            "n_candidates"     : len(grp),
            "n_positives"      : n_pos,
            "n_nat_positives"  : n_nat_pos,
            "has_natural"      : has_natural,
            "has_injected"     : has_injected,
            "posthoc_only"     : has_injected and not has_natural,
            "is_multi_cashout" : n_pos > 1,
            # Feature values for stratification
            "hour"             : int(grp["hour"].iloc[0]),
            "fraud_amount"     : float(grp["fraud_amount"].iloc[0]),
            "mule_chain_depth" : int(grp["mule_chain_depth"].iloc[0]),
            "fraud_type_enc"   : int(grp["fraud_type_encoded"].iloc[0]),
        }

        # Distance from victim to actual H3 (natural positives only; else all)
        pos_dist = grp.loc[grp["relevance"]==1, "cand_dist_km_from_victim"]
        nat_dist = grp.loc[is_nat_pos, "cand_dist_km_from_victim"]
        row["cashout_dist_km"] = float(nat_dist.mean() if has_natural else pos_dist.mean())

        # ── TIER 1: Candidate recall (is actual H3 naturally in set?) ─────
        # Pre-computed globally; per complaint it's just has_natural
        row["tier1_recall"]   = int(has_natural)

        # ── TIER 2: Conditional ranking (model quality, all positives) ────
        # Computed over ALL positives (as model was trained on them)
        for k in [1, 3, 5, 10]:
            row[f"t2_hit{k}"]  = int(any(r < k for r in all_pos_ranks))
        row["t2_mrr"]   = 1/(all_pos_ranks[0]+1) if all_pos_ranks else 0.0
        row["t2_ndcg5"] = _ndcg_at_k(sorted_rel, rel)

        # ── TIER 3: End-to-end (inference-aligned: natural positives only) ─
        # If complaint has no natural positive → e2e = 0 (bounded by generator)
        for k in [1, 3, 5, 10]:
            row[f"t3_hit{k}"] = int(any(r < k for r in nat_pos_ranks)) if has_natural else 0
        row["t3_mrr"]   = (1/(nat_pos_ranks[0]+1) if nat_pos_ranks else 0.0) if has_natural else 0.0
        row["t3_ndcg5"] = _ndcg_at_k(sorted_nat_rel, is_nat_pos.astype(int)) if has_natural else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(results_df, mask=None, label="ALL"):
    """Aggregate per-complaint metrics to summary dict."""
    df = results_df[mask] if mask is not None else results_df
    n  = len(df)
    if n == 0:
        return {"label": label, "n": 0}

    m = {"label": label, "n": n}
    # Tier 1
    m["tier1_complaint_recall"] = round(df["tier1_recall"].mean() * 100, 2)
    m["tier1_n_natural"]        = int(df["tier1_recall"].sum())
    # Tier 2 (conditional, all positives)
    for k in [1, 3, 5, 10]:
        m[f"t2_hit{k}"] = round(df[f"t2_hit{k}"].mean() * 100, 2)
    m["t2_mrr"]   = round(df["t2_mrr"].mean(), 4)
    m["t2_ndcg5"] = round(df["t2_ndcg5"].mean(), 4)
    # Tier 3 (end-to-end, natural only)
    for k in [1, 3, 5, 10]:
        m[f"t3_hit{k}"] = round(df[f"t3_hit{k}"].mean() * 100, 2)
    m["t3_mrr"]   = round(df["t3_mrr"].mean(), 4)
    m["t3_ndcg5"] = round(df["t3_ndcg5"].mean(), 4)
    return m


def bucket_distance(km):
    if km < 50:   return "0-50km"
    if km < 150:  return "50-150km"
    if km < 500:  return "150-500km"
    return "500+km"

def bucket_hour(h):
    if 6  <= h < 12: return "morning (6-12)"
    if 12 <= h < 18: return "afternoon (12-18)"
    if 18 <= h < 22: return "evening (18-22)"
    return "night (22-6)"

def bucket_amount(a):
    if a <  5000:  return "<5K"
    if a < 25000:  return "5-25K"
    if a < 100000: return "25-100K"
    return "100K+"

def bucket_depth(d):
    if d == 0: return "0"
    if d == 1: return "1"
    if d == 2: return "2"
    return "3+"

# ── Ordered group sizes ───────────────────────────────────────────────────────

def get_group_sizes(df):
    """Group sizes in DataFrame row order (complaints are contiguous)."""
    return np.array([sum(1 for _ in g) for _, g in itr_groupby(df["complaint_id"].values)])


# ── Main ──────────────────────────────────────────────────────────────────────

def run_phase7(config_path="config/phase7_config.yaml"):
    cfg     = yaml.safe_load(open(config_path))
    seed    = cfg["general"]["seed"]
    out_dir = cfg["general"]["output_dir"]
    src_dir = cfg["general"]["source_dir"]
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)
    t0 = time.time()

    print("="*65)
    print("  HIVE-PREDICT PHASE 7 — XGBOOST LEARNING-TO-RANK")
    print("="*65)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 1: Pre-training sanity checks
    # ═══════════════════════════════════════════════════════════════════
    print("\n[1/7] Pre-training sanity checks...")
    full_df = pd.read_csv(f"{src_dir}/candidate_h3_dataset_v2.csv")

    # Check 1: posthoc counts
    ph_rows = (full_df["positive_source"] == "posthoc_injection").sum()
    ph_comp = full_df[full_df["positive_source"] == "posthoc_injection"]["complaint_id"].nunique()
    nat_rows= (full_df["positive_source"] == "natural").sum()

    sanity = {
        "posthoc_complaints_expected"    : 1952,
        "posthoc_complaints_actual"      : int(ph_comp),
        "posthoc_complaints_match"       : int(ph_comp) == 1952,
        "posthoc_positive_rows_expected" : 2346,
        "posthoc_positive_rows_actual"   : int(ph_rows),
        "posthoc_positive_rows_match"    : int(ph_rows) == 2346,
    }
    delta = int(ph_rows) - int(ph_comp)
    sanity["posthoc_rows_minus_complaints"] = delta
    sanity["delta_explanation"] = (
        f"{delta} extra post-hoc rows come from multi-cashout complaints that have "
        f">=2 actual H3 cells all requiring post-hoc injection. "
        f"Each additional post-hoc H3 per complaint adds 1 row but does not "
        f"add 1 new complaint to the posthoc_complaints count."
    )

    # Verify by checking: posthoc complaints with >1 post-hoc positive
    ph_pos_per_comp = (
        full_df[full_df["positive_source"]=="posthoc_injection"]
        .groupby("complaint_id").size()
    )
    multi_ph = (ph_pos_per_comp > 1).sum()
    extra_rows = (ph_pos_per_comp - 1).sum()
    sanity["posthoc_complaints_with_multiple_ph_positives"] = int(multi_ph)
    sanity["extra_rows_from_multiple_ph_positives"]         = int(extra_rows)
    sanity["delta_verified"]                                = int(extra_rows) == delta

    # Check 2: Feature list
    all_cols = set(full_df.columns)
    remaining = [c for c in full_df.columns if c not in EXCLUDE_COLS]
    sanity["excluded_columns"]         = list(EXCLUDE_COLS)
    sanity["model_feature_count"]      = len(MODEL_FEATURES)
    sanity["model_features"]           = MODEL_FEATURES
    sanity["feature_count_match"]      = len(MODEL_FEATURES) == 22
    sanity["all_features_in_csv"]      = all(c in all_cols for c in MODEL_FEATURES)
    sanity["actual_* columns in data"] = [c for c in full_df.columns if "actual_" in c.lower()]

    # Check 3: Temporal validity (no actual_ columns)
    forbidden = [c for c in full_df.columns if "actual_" in c.lower() or
                 "withdrawal_id" in c.lower() or "cashout_timestamp" in c.lower()]
    sanity["forbidden_columns_found"] = forbidden
    sanity["temporal_validity_ok"]    = len(forbidden) == 0

    for k, v in sanity.items():
        if isinstance(v, bool):
            sym = "OK" if v else "FAIL"
            print(f"  [{sym}] {k}: {v}")
        elif k in ("posthoc_complaints_actual","posthoc_positive_rows_actual","model_feature_count"):
            ok = sanity.get(k.replace("_actual","_match"), True)
            sym = "OK" if ok else "FAIL"
            print(f"  [{sym}] {k} = {v}")

    print(f"\n  Delta explanation: {sanity['delta_explanation']}")
    assert sanity["posthoc_complaints_match"],    "posthoc_complaints mismatch!"
    assert sanity["posthoc_positive_rows_match"], "posthoc_positive_rows mismatch!"
    assert sanity["feature_count_match"],         "Feature count != 22!"
    assert sanity["temporal_validity_ok"],        f"Forbidden columns found: {forbidden}"
    print("  All sanity checks PASSED.")
    del full_df  # free memory

    # ═══════════════════════════════════════════════════════════════════
    # STEP 2: Load temporal splits
    # ═══════════════════════════════════════════════════════════════════
    print("\n[2/7] Loading temporal splits...")
    train_df = pd.read_csv(f"{src_dir}/train.csv")
    val_df   = pd.read_csv(f"{src_dir}/validation.csv")
    test_df  = pd.read_csv(f"{src_dir}/test.csv")
    print(f"  train:      {len(train_df):>9,} rows | {train_df['complaint_id'].nunique():,} complaints")
    print(f"  validation: {len(val_df):>9,} rows | {val_df['complaint_id'].nunique():,} complaints")
    print(f"  test:       {len(test_df):>9,} rows | {test_df['complaint_id'].nunique():,} complaints")

    # Load complaints for fraud_type mapping
    complaints_df = pd.read_csv("data/output/complaints.csv")
    ft_map = {}
    if "fraud_type" in complaints_df.columns:
        ft_map = complaints_df.set_index("complaint_id")["fraud_type"].to_dict()

    # ═══════════════════════════════════════════════════════════════════
    # STEP 3: Build DMatrix + group sizes
    # ═══════════════════════════════════════════════════════════════════
    print("\n[3/7] Building DMatrix objects...")
    X_tr = train_df[MODEL_FEATURES].values.astype(np.float32)
    y_tr = train_df["relevance"].values.astype(np.float32)
    X_va = val_df[MODEL_FEATURES].values.astype(np.float32)
    y_va = val_df["relevance"].values.astype(np.float32)

    g_tr = get_group_sizes(train_df)
    g_va = get_group_sizes(val_df)
    assert g_tr.sum() == len(train_df), "Group size mismatch: train"
    assert g_va.sum() == len(val_df),   "Group size mismatch: val"
    print(f"  train groups: {len(g_tr):,} | val groups: {len(g_va):,}")

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=MODEL_FEATURES)
    dtrain.set_group(g_tr)
    dval   = xgb.DMatrix(X_va, label=y_va, feature_names=MODEL_FEATURES)
    dval.set_group(g_va)
    del X_tr, y_tr, X_va, y_va  # free memory

    # ═══════════════════════════════════════════════════════════════════
    # STEP 4: XGBoost Training
    # ═══════════════════════════════════════════════════════════════════
    print("\n[4/7] Training XGBoost rank:ndcg...")
    mcfg = cfg["model"]
    params = {
        "objective"        : mcfg["objective"],
        "eval_metric"      : mcfg["eval_metric"],
        "eta"              : mcfg["learning_rate"],
        "max_depth"        : mcfg["max_depth"],
        "min_child_weight" : mcfg["min_child_weight"],
        "subsample"        : mcfg["subsample"],
        "colsample_bytree" : mcfg["colsample_bytree"],
        "gamma"            : mcfg["gamma"],
        "reg_alpha"        : mcfg["reg_alpha"],
        "reg_lambda"       : mcfg["reg_lambda"],
        "seed"             : mcfg["seed"],
        "verbosity"        : mcfg["verbosity"],
        "device"           : mcfg["device"],
    }
    evals_result = {}
    t_train = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round       = mcfg["n_estimators"],
        evals                 = [(dtrain, "train"), (dval, "val")],
        early_stopping_rounds = mcfg["early_stopping_rounds"],
        evals_result          = evals_result,
        verbose_eval          = 100,
    )
    train_time = time.time() - t_train
    best_round  = model.best_iteration
    best_val    = model.best_score
    print(f"  Training done in {train_time:.1f}s | best_round={best_round} | best_val_ndcg5={best_val:.6f}")

    # Training curve
    train_curve = pd.DataFrame({
        "round"     : list(range(len(evals_result["train"]["ndcg@5-"]))),
        "train_ndcg5": evals_result["train"]["ndcg@5-"],
        "val_ndcg5"  : evals_result["val"]["ndcg@5-"],
    })
    train_curve.to_csv(f"{out_dir}/train_eval_history.csv", index=False)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 5: Evaluation
    # ═══════════════════════════════════════════════════════════════════
    print("\n[5/7] Evaluating (three-tier framework)...")

    # Evaluate all three splits
    val_results  = evaluate_complaints(val_df,  model, "validation")
    test_results = evaluate_complaints(test_df, model, "test")

    # Attach fraud_type string
    for df_r in [val_results, test_results]:
        df_r["fraud_type"] = df_r["complaint_id"].map(
            lambda c: ft_map.get(c, "unknown"))
        df_r["dist_band"]  = df_r["cashout_dist_km"].apply(bucket_distance)
        df_r["hour_band"]  = df_r["hour"].apply(bucket_hour)
        df_r["amt_band"]   = df_r["fraud_amount"].apply(bucket_amount)
        df_r["depth_band"] = df_r["mule_chain_depth"].apply(bucket_depth)
        df_r["cashout_type"] = df_r["n_positives"].apply(
            lambda n: "single" if n == 1 else "multiple")

    def print_summary(results, label):
        print(f"\n  ── {label} ({len(results):,} complaints) ──")
        ov = aggregate_metrics(results)
        nat = aggregate_metrics(results, results["has_natural"], "natural")
        inj = aggregate_metrics(results, results["has_injected"], "injected")
        print(f"  {'Metric':<18} {'Overall':>10} {'Natural':>10} {'PostHoc':>10}  [T2=conditional, T3=e2e-inference]")
        print(f"  {'-'*62}")
        for k in [1,3,5,10]:
            print(f"  T2 Hit@{k:<11} {ov.get(f't2_hit{k}',0):>9.2f}% {nat.get(f't2_hit{k}',0):>9.2f}% {inj.get(f't2_hit{k}',0):>9.2f}%")
        print(f"  T2 MRR            {ov['t2_mrr']:>10.4f} {nat['t2_mrr']:>10.4f} {inj['t2_mrr']:>10.4f}")
        print(f"  T2 NDCG@5         {ov['t2_ndcg5']:>10.4f} {nat['t2_ndcg5']:>10.4f} {inj['t2_ndcg5']:>10.4f}")
        print(f"  {'-'*62}")
        for k in [1,3,5,10]:
            print(f"  T3 Hit@{k:<11} {ov.get(f't3_hit{k}',0):>9.2f}% {nat.get(f't3_hit{k}',0):>9.2f}% {inj.get(f't3_hit{k}',0):>9.2f}%")
        print(f"  T3 MRR            {ov['t3_mrr']:>10.4f} {nat['t3_mrr']:>10.4f} {inj['t3_mrr']:>10.4f}")
        print(f"  T3 NDCG@5         {ov['t3_ndcg5']:>10.4f} {nat['t3_ndcg5']:>10.4f} {inj['t3_ndcg5']:>10.4f}")
        print(f"\n  Candidate recall (Tier 1): {ov['tier1_complaint_recall']:.2f}% of all {label} complaints")
        print(f"  [NOTE] End-to-end T3 metrics are bounded by {ov['tier1_complaint_recall']:.2f}% generator recall")
        return ov, nat, inj

    print("\n  === VALIDATION ===")
    val_ov, val_nat, val_inj = print_summary(val_results, "validation")
    print("\n  === TEST (HELD-OUT) ===")
    tst_ov, tst_nat, tst_inj = print_summary(test_results, "test")

    # ═══════════════════════════════════════════════════════════════════
    # STEP 6: Stratified evaluation (test set)
    # ═══════════════════════════════════════════════════════════════════
    print("\n[6/7] Stratified evaluation (test set)...")

    strat_rows = []
    strat_dims = {
        "fraud_type"   : test_results["fraud_type"].unique(),
        "dist_band"    : ["0-50km","50-150km","150-500km","500+km"],
        "hour_band"    : ["morning (6-12)","afternoon (12-18)","evening (18-22)","night (22-6)"],
        "amt_band"     : ["<5K","5-25K","25-100K","100K+"],
        "depth_band"   : ["0","1","2","3+"],
        "cashout_type" : ["single","multiple"],
    }
    # Add victim_state from complaints
    if "victim_state" in complaints_df.columns:
        vs_map = complaints_df.set_index("complaint_id")["victim_state"].to_dict()
        test_results["victim_state"] = test_results["complaint_id"].map(vs_map)
        strat_dims["victim_state"] = test_results["victim_state"].dropna().unique()

    for dim, vals in strat_dims.items():
        if dim not in test_results.columns: continue
        for val in vals:
            mask = test_results[dim] == val
            if mask.sum() == 0: continue
            m = aggregate_metrics(test_results, mask, f"{dim}={val}")
            m["dimension"] = dim
            m["stratum"]   = str(val)
            strat_rows.append(m)
            print(f"  {dim}={val:<25} n={m['n']:>5} | "
                  f"recall={m['tier1_complaint_recall']:>5.1f}% | "
                  f"T2-Hit5={m.get('t2_hit5',0):>5.1f}% | "
                  f"T3-Hit5={m.get('t3_hit5',0):>5.1f}%")

    strat_df = pd.DataFrame(strat_rows)

    # ═══════════════════════════════════════════════════════════════════
    # STEP 7: Save outputs
    # ═══════════════════════════════════════════════════════════════════
    print("\n[7/7] Saving outputs...")

    # Model
    model.save_model(f"{out_dir}/model.ubj")
    print(f"  model.ubj saved (best_round={best_round})")

    # Feature list
    with open(f"{out_dir}/feature_list.json","w") as f:
        json.dump({"complaint_features": COMPLAINT_FEATS,
                   "candidate_features": CANDIDATE_FEATS,
                   "model_features"    : MODEL_FEATURES,
                   "n_features"        : len(MODEL_FEATURES)}, f, indent=2)

    # Hyperparameters
    with open(f"{out_dir}/hyperparameters.json","w") as f:
        json.dump({**params, "n_estimators": mcfg["n_estimators"],
                   "early_stopping_rounds": mcfg["early_stopping_rounds"],
                   "best_round": best_round, "best_val_ndcg5": best_val}, f, indent=2)

    # Model metadata
    import datetime
    with open(f"{out_dir}/model_metadata.json","w") as f:
        json.dump({
            "phase": "7.0.0",
            "generated": datetime.datetime.utcnow().isoformat()+"Z",
            "seed": seed,
            "training_data": src_dir,
            "model_file": "model.ubj",
            "feature_file": "feature_list.json",
            "n_train_rows": len(train_df),
            "n_train_complaints": int(g_tr.sum() > 0) * len(g_tr),
            "n_val_complaints": len(g_va),
            "best_round": best_round,
            "best_val_ndcg5": float(best_val),
            "training_time_seconds": round(train_time, 1),
            "xgboost_version": xgb.__version__,
            "objective": params["objective"],
        }, f, indent=2)

    # Sanity check report
    with open(f"{out_dir}/sanity_check_report.json","w") as f:
        json.dump(sanity, f, indent=2)

    # Validation metrics
    with open(f"{out_dir}/validation_metrics.json","w") as f:
        json.dump({"overall": val_ov, "natural": val_nat, "posthoc": val_inj}, f, indent=2)

    # Test metrics
    with open(f"{out_dir}/test_metrics.json","w") as f:
        json.dump({"overall": tst_ov, "natural": tst_nat, "posthoc": tst_inj}, f, indent=2)

    # Stratified metrics
    strat_df.to_csv(f"{out_dir}/stratified_metrics.csv", index=False)

    # Feature importance
    fi = model.get_score(importance_type="gain")
    fi_df = (pd.DataFrame(list(fi.items()), columns=["feature","gain"])
             .sort_values("gain", ascending=False))
    fi_df["weight"] = fi_df["feature"].map(model.get_score(importance_type="weight"))
    fi_df["cover"]  = fi_df["feature"].map(model.get_score(importance_type="cover"))
    fi_df.to_csv(f"{out_dir}/feature_importance.csv", index=False)
    print("  Feature importance (top 10 by gain):")
    for _, r in fi_df.head(10).iterrows():
        print(f"    {r['feature']:<35} gain={r['gain']:>10.2f}")

    # Per-complaint results
    test_results.to_csv(f"{out_dir}/per_complaint_test_results.csv", index=False)

    # Prediction samples (top-10 predictions for 100 complaints from test)
    sample_cids = test_df["complaint_id"].unique()[:100]
    sample_rows = []
    for cid in sample_cids:
        grp = test_df[test_df["complaint_id"]==cid].copy()
        X_s = grp[MODEL_FEATURES].values.astype(np.float32)
        scores = model.predict(xgb.DMatrix(X_s, feature_names=MODEL_FEATURES))
        grp["score"] = scores
        top10 = grp.nlargest(10, "score")[
            ["complaint_id","candidate_h3_cell","relevance","positive_source","score",
             "cand_dist_km_from_victim","cand_atm_count","cand_hotspot_density"]
        ]
        sample_rows.append(top10)
    pd.concat(sample_rows).to_csv(f"{out_dir}/prediction_samples.csv", index=False)

    # Error analysis: complaints where T3 Hit@5 = 0 but had natural positive
    missed = test_results[(test_results["t3_hit5"]==0) & (test_results["has_natural"])]
    missed.to_csv(f"{out_dir}/error_analysis.csv", index=False)
    print(f"  Error analysis: {len(missed):,} complaints missed at T3-Hit@5 (had natural positive, ranked outside top-5)")

    elapsed = time.time()-t0
    print(f"\n{'='*65}")
    print(f"  PHASE 7 COMPLETE in {elapsed:.1f}s")
    print(f"\n  TEST RESULTS (end-to-end / inference-aligned):")
    print(f"    T3 Hit@1:   {tst_ov['t3_hit1']:>6.2f}%  (upper bound: {tst_ov['tier1_complaint_recall']:.2f}%)")
    print(f"    T3 Hit@3:   {tst_ov['t3_hit3']:>6.2f}%")
    print(f"    T3 Hit@5:   {tst_ov['t3_hit5']:>6.2f}%")
    print(f"    T3 Hit@10:  {tst_ov['t3_hit10']:>6.2f}%")
    print(f"    T3 MRR:     {tst_ov['t3_mrr']:>8.4f}")
    print(f"    T3 NDCG@5:  {tst_ov['t3_ndcg5']:>8.4f}")
    print(f"\n  RANKING QUALITY (conditional on actual H3 in candidate set):")
    print(f"    T2 Hit@5:   {tst_ov['t2_hit5']:>6.2f}%")
    print(f"    T2 NDCG@5:  {tst_ov['t2_ndcg5']:>8.4f}")
    print(f"\n  Output: {out_dir}/")
    print("="*65)

    return model, tst_ov, test_results, strat_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/phase7_config.yaml")
    args = parser.parse_args()
    run_phase7(args.config)
