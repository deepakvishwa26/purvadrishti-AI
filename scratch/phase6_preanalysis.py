"""Pre-analysis for Phase 6 candidate generation design decisions."""
import pandas as pd
import numpy as np
import h3
from math import radians, sin, cos, sqrt, atan2

OUT = "data/output"
labels   = pd.read_csv(f"{OUT}/cashout_labels.csv")
features = pd.read_csv(f"{OUT}/feature_snapshots.csv")
atm_df   = pd.read_csv(f"{OUT}/atm_reference.csv")
wdr      = pd.read_csv(f"{OUT}/withdrawals.csv")

cashout = labels[labels["cashout_occurred"] == True]

def km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))

print("=== CASHOUT COMPLAINT STATS ===")
print(f"Cashout complaints:        {cashout['complaint_id'].nunique()}")
print(f"Total cashout label rows:  {len(cashout)}")
print(f"Total NO_CASHOUT:          {(labels['cashout_occurred']==False).sum()}")

pos_per = cashout.groupby("complaint_id")["actual_h3_cell"].nunique()
print(f"\nUnique positive H3 per cashout complaint:")
for v, c in pos_per.value_counts().sort_index().items():
    print(f"  {v} positive H3(s): {c:,} complaints")

print("\n=== ATM REFERENCE ===")
print(f"Total ATMs:              {len(atm_df)}")
print(f"Unique H3 cells (res-8): {atm_df['h3_cell_res8'].nunique()}")
print(f"States covered:          {atm_df['state'].nunique()}")
atm_per_state = atm_df.groupby("state").size()
print(f"ATMs per state: min={atm_per_state.min()} mean={atm_per_state.mean():.0f} max={atm_per_state.max()}")

print("\n=== FEATURE SNAPSHOT COLUMNS ===")
print(features.columns.tolist())

print("\n=== VICTIM -> ACTUAL H3 DISTANCE DISTRIBUTION (sample 500) ===")
feat_wdr = features.merge(
    cashout[["complaint_id", "actual_h3_cell"]].drop_duplicates("complaint_id"),
    on="complaint_id", how="inner"
)
sample = feat_wdr.sample(500, random_state=42)
dists_km = []
grid_dists = []
for _, row in sample.iterrows():
    v_lat, v_lon = h3.cell_to_latlng(row["victim_h3_res8"])
    a_lat, a_lon = h3.cell_to_latlng(row["actual_h3_cell"])
    dists_km.append(km(v_lat, v_lon, a_lat, a_lon))
    try:
        grid_dists.append(h3.grid_distance(row["victim_h3_res8"], row["actual_h3_cell"]))
    except Exception:
        grid_dists.append(None)

s_km = pd.Series(dists_km)
s_gd = pd.Series([g for g in grid_dists if g is not None])
print(f"Distance victim->actual withdrawal H3 (km):")
print(f"  min={s_km.min():.1f}  p10={s_km.quantile(.1):.1f}  p25={s_km.quantile(.25):.1f}  "
      f"median={s_km.median():.1f}  p75={s_km.quantile(.75):.1f}  "
      f"p90={s_km.quantile(.90):.1f}  max={s_km.max():.1f}")
print(f"H3 grid distance victim->actual:")
print(f"  min={s_gd.min()}  median={s_gd.median():.0f}  p90={s_gd.quantile(.9):.0f}  max={s_gd.max()}")

print("\n=== SAME-STATE CHECK ===")
feat_wdr2 = features.merge(wdr[["complaint_id","latitude","longitude"]], on="complaint_id")
same_state = 0
for _, row in feat_wdr2.sample(200, random_state=42).iterrows():
    v_lat, v_lon = h3.cell_to_latlng(row["victim_h3_res8"])
    same_state += (km(v_lat, v_lon, row["latitude"], row["longitude"]) < 200)
print(f"Withdrawals within 200km of victim: {same_state}/200 ({same_state/2:.0f}%)")

print("\n=== RING-2 NEIGHBOR COVERAGE ESTIMATE ===")
# How often is actual H3 within ring-2 of victim? ring-1? ring-3?
hits = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0, 20: 0}
for _, row in sample.iterrows():
    for ring in hits:
        try:
            gd = h3.grid_distance(row["victim_h3_res8"], row["actual_h3_cell"])
            if gd <= ring:
                hits[ring] += 1
        except Exception:
            pass
n = len(sample)
for ring, hit in hits.items():
    print(f"  Actual H3 within ring-{ring:2d} of victim: {hit}/{n} ({hit/n*100:.1f}%)")

print("\n=== H3 ATM CELL POOL ===")
atm_h3_unique = atm_df["h3_cell_res8"].unique()
print(f"Unique ATM H3 cells (res-8): {len(atm_h3_unique)}")
print("These are the primary negative candidate pool.")

print("\n=== MULTIPLE CASHOUT H3 DEDUP CHECK ===")
# For complaints with multiple withdrawals, how many unique H3 cells?
multi = cashout.groupby("complaint_id")["actual_h3_cell"].agg(["count", "nunique"])
multi_complaints = multi[multi["count"] > 1]
print(f"Complaints with >1 withdrawal: {len(multi_complaints)}")
print(f"  Of those, with >1 unique H3:  {(multi_complaints['nunique']>1).sum()}")
print(f"  Of those, with =1 unique H3:  {(multi_complaints['nunique']==1).sum()} (all same cell)")
