import pandas as pd, json, os

print("=== PHASE 7 OUTPUT FILES ===")
for f in sorted(os.listdir("data/output/phase7")):
    fp = os.path.join("data/output/phase7", f)
    if os.path.isfile(fp):
        print(f"  {f}  ({os.path.getsize(fp)//1024}KB)")

print("\n=== GROUP INFO SEARCH ===") 
for gi_path in [
    "data/output/phase7/group_info.json",
    "data/output/phase6_v2/group_info.json",
    "data/output/group_info.json",
]:
    if os.path.exists(gi_path):
        gi = json.load(open(gi_path))
        print(f"Found: {gi_path}")
        print("Keys:", list(gi.keys()))
        for k in ["train","validation","test"]:
            if k in gi:
                v = gi[k]
                if isinstance(v, list):
                    print(f"  {k}: {len(v)} complaints")
                elif isinstance(v, dict):
                    print(f"  {k}: {list(v.keys())[:5]}...")
        break
    else:
        print(f"  NOT FOUND: {gi_path}")

print("\n=== PHASE 7 CANDIDATE DATASET SEARCH ===")
for cand_path in [
    "data/output/phase6_v2/candidate_h3_dataset.csv",
    "data/output/phase7/candidate_h3_dataset.csv",
    "data/output/phase6_v2/train.csv",
    "data/output/phase6_v2/validation.csv",
]:
    if os.path.exists(cand_path):
        print(f"  FOUND: {cand_path}  ({os.path.getsize(cand_path)//1024}KB)")
        df = pd.read_csv(cand_path, nrows=5)
        print("  Columns:", list(df.columns))
        # Check if feature_cutoff_timestamp exists
        if "feature_cutoff_timestamp" in df.columns:
            print("  Has feature_cutoff_timestamp: YES")
        if "split" in df.columns:
            print("  Split values:", df["split"].unique().tolist())
        break
    else:
        print(f"  NOT FOUND: {cand_path}")

# Also check phase6_v2 directory
print("\n=== PHASE6_V2 DIRECTORY ===")
if os.path.exists("data/output/phase6_v2"):
    for f in sorted(os.listdir("data/output/phase6_v2")):
        fp = os.path.join("data/output/phase6_v2", f)
        if os.path.isfile(fp):
            print(f"  {f}  ({os.path.getsize(fp)//1024}KB)")
else:
    print("  phase6_v2 directory not found")

print("\n=== WITHDRAWAL SCHEMA + STATS ===")
wdr = pd.read_csv("data/output/withdrawals.csv", parse_dates=["withdrawal_timestamp"])
print("Columns:", list(wdr.columns))
print(f"Rows: {len(wdr):,}")
print(f"Unique complaints: {wdr['complaint_id'].nunique():,}")
print(f"Date range: {wdr['withdrawal_timestamp'].min()} -> {wdr['withdrawal_timestamp'].max()}")
print(f"lat: {wdr['latitude'].min():.3f} to {wdr['latitude'].max():.3f}")
print(f"lon: {wdr['longitude'].min():.3f} to {wdr['longitude'].max():.3f}")
print("Sample:")
print(wdr[["withdrawal_timestamp","latitude","longitude","h3_cell","complaint_id"]].head(3).to_string())

print("\n=== FEATURE SNAPSHOT CUTOFF RANGE ===")
fs = pd.read_csv("data/output/feature_snapshots.csv", parse_dates=["feature_cutoff_timestamp"])
cashout_cids = set(wdr["complaint_id"])
fs_cash = fs[fs["complaint_id"].isin(cashout_cids)]
print(f"Cashout complaints with snapshots: {len(fs_cash):,}")
print(f"Cutoff range: {fs_cash['feature_cutoff_timestamp'].min()} to {fs_cash['feature_cutoff_timestamp'].max()}")

print("\n=== PHASE 7 TEST METRICS ===")
m = json.load(open("data/output/phase7/test_metrics.json"))
overall = m.get("overall", m)
for k,v in overall.items():
    print(f"  {k}: {v}")

print("\n=== ATM REFERENCE ===")
atm = pd.read_csv("data/output/atm_reference.csv")
print("Columns:", list(atm.columns))
print(f"Rows: {len(atm):,}")
print(atm.head(3).to_string())
