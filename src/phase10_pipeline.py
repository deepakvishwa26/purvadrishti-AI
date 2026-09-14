"""
Phase 10 — HIVE-Predict Demo Pipeline
======================================
End-to-end prediction pipeline for a single cyber-fraud complaint.

Input:  complaint_id (from synthetic test set)
Output: structured prediction dict with:
  - top-5 ranked H3 zones
  - XGBoost scores
  - nearby ATM candidates per zone
  - SHAP explanation
  - alert priority
  - map-ready coordinates

Uses ONLY:
  - Phase 7 frozen XGBoost model (data/output/phase7/model.ubj)
  - Phase 6 V2 pre-computed candidate features (for speed/reliability)
  - ATM reference data

KDE is NOT integrated (Phase 9: no demonstrated utility).
"""

import os, sys, json, warnings
import numpy as np, pandas as pd, h3
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import xgboost as xgb

# ── Paths ─────────────────────────────────────────────────────────────
BASE_DATA_DIR = os.environ.get("PURVADRISHTI_DATA_DIR", "data/output")
P7   = f"{BASE_DATA_DIR}/phase7"
P6V2 = f"{BASE_DATA_DIR}/phase6_v2"
SRC  = BASE_DATA_DIR
OUT  = f"{BASE_DATA_DIR}/phase10"

P7_FEATURES = [
    "fraud_amount","amount_log","hour","day_of_week","is_weekend","is_night",
    "fraud_type_encoded","mule_chain_depth","mule_velocity","amount_velocity",
    "distance_from_victim","historical_hotspot_density","atm_density",
    "complaint_cluster","time_since_transaction",
    "cand_dist_km_from_victim","cand_h3_grid_dist","cand_atm_count",
    "cand_atm_density","cand_hotspot_density","cand_in_victim_state",
    "cand_is_victim_h3",
]

FEATURE_DISPLAY_NAMES = {
    "fraud_amount":             "Fraud Amount (INR)",
    "amount_log":               "Log Fraud Amount",
    "hour":                     "Hour of Incident",
    "day_of_week":              "Day of Week",
    "is_weekend":               "Weekend Flag",
    "is_night":                 "Night-time Flag",
    "fraud_type_encoded":       "Fraud Type",
    "mule_chain_depth":         "Mule Chain Depth",
    "mule_velocity":            "Mule Velocity (hops/min)",
    "amount_velocity":          "Amount Velocity (INR/min)",
    "distance_from_victim":     "Distance from Victim (km)",
    "historical_hotspot_density":"Historical Hotspot Density",
    "atm_density":              "ATM Density (victim area)",
    "complaint_cluster":        "Complaint Cluster ID",
    "time_since_transaction":   "Time Since Transaction (min)",
    "cand_dist_km_from_victim": "Candidate Distance from Victim (km)",
    "cand_h3_grid_dist":        "H3 Grid Distance",
    "cand_atm_count":           "ATMs in Candidate Zone",
    "cand_atm_density":         "ATM Density in Zone",
    "cand_hotspot_density":     "Historical Hotspot Density in Zone",
    "cand_in_victim_state":     "In Victim's State",
    "cand_is_victim_h3":        "Is Victim's H3 Zone",
}

# Alert priority thresholds (score-based, transparent prototype thresholds)
ALERT_PRIORITY_CONFIG = {
    "HIGH":   {"min_score": 0.7,  "description": "Strong spatial match — immediate attention recommended"},
    "MEDIUM": {"min_score": 0.4,  "description": "Moderate match — review and verify"},
    "LOW":    {"min_score": 0.0,  "description": "Weak match — low-confidence zone"},
}

def _km(lat1, lon1, lat2, lon2):
    R = 6371.0; d = radians
    a = sin((d(lat2-lat1))/2)**2 + cos(d(lat1))*cos(d(lat2))*sin((d(lon2-lon1))/2)**2
    return R*2*atan2(sqrt(a), sqrt(1-a))

def _alert_priority(score: float) -> str:
    if score >= ALERT_PRIORITY_CONFIG["HIGH"]["min_score"]:   return "HIGH"
    if score >= ALERT_PRIORITY_CONFIG["MEDIUM"]["min_score"]: return "MEDIUM"
    return "LOW"


class HIVEPredictor:
    """
    End-to-end HIVE-Predict demo predictor.

    Loads all static data once at __init__ time.
    Prediction is a single call to predict(complaint_id).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)
        self._loaded = False

    def load(self):
        """Load all static assets. Call once before prediction."""
        print("[HIVE] Loading Phase 7 model...")
        self.model = xgb.Booster()
        self.model.load_model(f"{P7}/model.ubj")

        print("[HIVE] Loading reference data...")
        self.atm_df     = pd.read_csv(f"{SRC}/atm_reference.csv")
        self.comp_df    = pd.read_csv(f"{SRC}/complaints.csv",
                                      parse_dates=["incident_datetime",
                                                    "complaint_registered_at"])
        # Note: withdrawals.csv and feature_snapshots.csv are NOT loaded here
        # — not needed for prediction (only used by Phase 9 KDE and training pipelines)
        self.labels_df  = pd.read_csv(f"{SRC}/cashout_labels.csv")
        # Normalize cashout_occurred to bool
        if "cashout_occurred" in self.labels_df.columns:
            self.labels_df["cashout_occurred"] = self.labels_df["cashout_occurred"].astype(str).str.lower().isin(["true","1","yes"])

        # Phase 6 v2 pre-computed candidate datasets (test + validation splits)
        # Load ONLY the columns needed for inference to minimise RAM.
        print("[HIVE] Loading pre-computed candidate features...")
        NEED_COLS = (
            ["complaint_id", "candidate_h3_cell", "feature_cutoff_timestamp"] +
            P7_FEATURES +
            ["cand_dist_km_from_victim", "cand_atm_count", "cand_in_victim_state"]
        )
        # Remove duplicates while preserving order
        seen_c = set(); NEED_COLS = [c for c in NEED_COLS if not (c in seen_c or seen_c.add(c))]

        # dtype map: float32 for numeric features to halve memory vs float64
        f32_cols = {c: "float32" for c in P7_FEATURES + ["cand_dist_km_from_victim", "cand_atm_count", "cand_hotspot_density", "cand_atm_density"]}

        te = pd.read_csv(f"{P6V2}/test.csv",
                         usecols=[c for c in NEED_COLS if c != "feature_cutoff_timestamp"] + ["feature_cutoff_timestamp"],
                         dtype=f32_cols)
        va = pd.read_csv(f"{P6V2}/validation.csv",
                         usecols=[c for c in NEED_COLS if c != "feature_cutoff_timestamp"] + ["feature_cutoff_timestamp"],
                         dtype=f32_cols)
        self._cand_df = pd.concat([te, va], ignore_index=True)
        del te, va  # free immediately

        # Build complaint_id lookup set (before potentially large .unique())
        self._known_complaints = set(self._cand_df["complaint_id"].tolist())
        self._atm_by_h3 = defaultdict(list)
        for _, row in self.atm_df.iterrows():
            self._atm_by_h3[row["h3_cell_res8"]].append({
                "atm_id": row["atm_id"],
                "lat": row["latitude"],
                "lon": row["longitude"],
                "state": row["state"],
                "city": row["city"],
                "h3_cell": row["h3_cell_res8"],
            })

        # h3 centroid cache
        self._h3_centroid = {}

        # SHAP explainer
        try:
            import shap
            self._shap_explainer = shap.TreeExplainer(self.model)
            self._shap_available = True
            print("[HIVE] SHAP explainer ready.")
        except Exception as e:
            self._shap_available = False
            print(f"[HIVE] SHAP not available: {e}")

        self._loaded = True
        print(f"[HIVE] Ready. {len(self._known_complaints):,} complaints available.")

    def _h3_centroid_coords(self, h3_cell: str):
        if h3_cell not in self._h3_centroid:
            lat, lon = h3.cell_to_latlng(h3_cell)
            self._h3_centroid[h3_cell] = (lat, lon)
        return self._h3_centroid[h3_cell]

    def _h3_boundary_coords(self, h3_cell: str):
        """Returns list of [lat,lon] pairs for H3 polygon boundary."""
        boundary = h3.cell_to_boundary(h3_cell)
        return [[lat, lon] for lat, lon in boundary]

    def _find_nearby_atms(self, h3_cell: str, victim_lat: float,
                           victim_lon: float, top_n: int = 5):
        """Find top_n ATMs near a predicted H3 cell."""
        cand_lat, cand_lon = self._h3_centroid_coords(h3_cell)

        # Step 1: ATMs in same H3 cell
        atms = list(self._atm_by_h3.get(h3_cell, []))

        # Step 2: ATMs in neighboring H3 cells (ring 1)
        if len(atms) < top_n:
            try:
                neighbors = h3.grid_disk(h3_cell, 1) - {h3_cell}
                for nbr in neighbors:
                    atms.extend(self._atm_by_h3.get(nbr, []))
            except Exception:
                pass

        # Step 3: ATMs in ring 2 if still insufficient
        if len(atms) < top_n:
            try:
                ring2 = h3.grid_disk(h3_cell, 2) - h3.grid_disk(h3_cell, 1)
                for nbr in ring2:
                    atms.extend(self._atm_by_h3.get(nbr, []))
            except Exception:
                pass

        if not atms:
            return []

        # Deduplicate by atm_id and sort by distance to H3 centroid
        seen = set(); unique_atms = []
        for a in atms:
            if a["atm_id"] not in seen:
                seen.add(a["atm_id"])
                a = dict(a)
                a["dist_to_h3_centroid_km"] = round(_km(cand_lat, cand_lon, a["lat"], a["lon"]), 2)
                a["dist_to_victim_km"]       = round(_km(victim_lat, victim_lon, a["lat"], a["lon"]), 2)
                unique_atms.append(a)

        unique_atms.sort(key=lambda x: x["dist_to_h3_centroid_km"])
        return unique_atms[:top_n]

    def _compute_shap(self, feature_df: pd.DataFrame, top_n_features: int = 5):
        """Compute SHAP values for all rows; return structured per-row explanation."""
        if not self._shap_available:
            return None
        try:
            import shap
            X = feature_df[P7_FEATURES].fillna(0).values.astype(np.float32)
            dmat = xgb.DMatrix(X, feature_names=P7_FEATURES)
            shap_values = self.model.predict(dmat, pred_contribs=True)
            # shap_values shape: (n_rows, n_features + 1) — last col is bias
            results = []
            for i in range(len(feature_df)):
                sv = shap_values[i, :-1]  # exclude bias
                bias = float(shap_values[i, -1])
                fvals = X[i]
                # Sort by |SHAP| descending
                order = np.argsort(-np.abs(sv))
                factors = []
                for j in order[:top_n_features]:
                    factors.append({
                        "feature": P7_FEATURES[j],
                        "display_name": FEATURE_DISPLAY_NAMES.get(P7_FEATURES[j], P7_FEATURES[j]),
                        "shap_value": round(float(sv[j]), 4),
                        "feature_value": round(float(fvals[j]), 4),
                        "direction": "increases_score" if sv[j] > 0 else "decreases_score",
                    })
                results.append({
                    "bias": round(bias, 4),
                    "top_factors": factors,
                    "positive_factors": [f for f in factors if f["shap_value"] > 0],
                    "negative_factors": [f for f in factors if f["shap_value"] < 0],
                })
            return results
        except Exception as e:
            print(f"[SHAP] Error: {e}")
            return None

    def predict(self, complaint_id: str, top_n: int = 5, seed: int = 42):
        """
        Run the full HIVE-Predict pipeline for one complaint.

        Returns
        -------
        dict with keys:
          complaint_id, complaint_info, cutoff,
          candidates (ranked list), top_predictions,
          actual_h3_cells, model_metadata
        """
        assert self._loaded, "Call load() first"
        assert complaint_id in self._known_complaints, \
            f"{complaint_id} not in pre-computed candidate set"

        # ── Complaint info ────────────────────────────────────────────
        comp_row = self.comp_df[self.comp_df["complaint_id"] == complaint_id]
        if len(comp_row) == 0:
            raise ValueError(f"Complaint {complaint_id} not found")
        comp = comp_row.iloc[0].to_dict()

        # ── Candidate features ────────────────────────────────────────
        cands = self._cand_df[self._cand_df["complaint_id"] == complaint_id].copy()
        cands = cands.reset_index(drop=True)

        # Enforce no actual_* columns leak
        leak_cols = [c for c in cands.columns if "actual" in c.lower()]
        assert len(leak_cols) == 0, f"Leakage columns found: {leak_cols}"

        # ── XGBoost scoring ───────────────────────────────────────────
        X = cands[P7_FEATURES].fillna(0).values.astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=P7_FEATURES)
        raw_scores = self.model.predict(dmat)
        cands["xgb_score"] = [float(x) if x == x else 0.0 for x in raw_scores]  # NaN-safe

        # Normalize to [0,1] per complaint (rank percentile for interpretability)
        n = len(cands)
        ranks = np.argsort(np.argsort(raw_scores))
        cands["score_pct"] = ranks / max(n-1, 1)

        # Rank: 1 = best
        cands["rank"] = n - ranks

        # Sort by score descending
        cands = cands.sort_values("xgb_score", ascending=False).reset_index(drop=True)

        # ── SHAP for top-N ────────────────────────────────────────────
        top_df  = cands.head(top_n)
        shap_rs = self._compute_shap(top_df)

        # ── Actual H3 cells (ground truth — for validation only) ──────
        act_rows = self.labels_df[
            (self.labels_df["complaint_id"] == complaint_id) &
            (self.labels_df["cashout_occurred"] == True)
        ]
        actual_h3 = act_rows["actual_h3_cell"].tolist() if len(act_rows) else []

        # ── Top predictions with ATMs ─────────────────────────────────
        victim_lat = float(comp.get("victim_lat", 0))
        victim_lon = float(comp.get("victim_lon", 0))
        cutoff = cands["feature_cutoff_timestamp"].iloc[0] if "feature_cutoff_timestamp" in cands.columns else None

        top_predictions = []
        for i, (_, row) in enumerate(top_df.iterrows()):
            h3_cell = row["candidate_h3_cell"]
            clat, clon = self._h3_centroid_coords(h3_cell)
            score = float(row["xgb_score"])
            is_correct = h3_cell in actual_h3

            top_predictions.append({
                "rank": i + 1,
                "h3_cell": h3_cell,
                "xgb_score": round(score, 4),
                "score_percentile": round(float(row["score_pct"]), 3),
                "alert_priority": _alert_priority(row["score_pct"]),
                "centroid_lat": round(clat, 6),
                "centroid_lon": round(clon, 6),
                "boundary": self._h3_boundary_coords(h3_cell),
                "dist_from_victim_km": round(float(row.get("cand_dist_km_from_victim", 0)), 1),
                "atm_count_in_zone": int(row.get("cand_atm_count", 0)),
                "in_victim_state": bool(row.get("cand_in_victim_state", False)),
                "is_actual_h3": is_correct,
                "nearby_atms": self._find_nearby_atms(h3_cell, victim_lat, victim_lon),
                "shap": shap_rs[i] if shap_rs else None,
            })

        # Overall alert: highest priority among top-5
        priority_order = {"HIGH": 2, "MEDIUM": 1, "LOW": 0}
        overall_priority = max(
            (p["alert_priority"] for p in top_predictions),
            key=lambda x: priority_order[x], default="LOW"
        )

        # Actual rank of true H3 in sorted candidates
        actual_rank = None
        if actual_h3:
            match_rows = cands[cands["candidate_h3_cell"].isin(actual_h3)]
            if len(match_rows):
                actual_rank = int(match_rows.index[0] + 1)

        return {
            "complaint_id": complaint_id,
            "complaint_info": {
                "fraud_type": comp.get("fraud_type",""),
                "fraud_amount": comp.get("fraud_amount",0),
                "victim_state": comp.get("victim_state",""),
                "victim_lat":   victim_lat,
                "victim_lon":   victim_lon,
                "incident_datetime": str(comp.get("incident_datetime","")),
                "complaint_registered_at": str(comp.get("complaint_registered_at","")),
            },
            "feature_cutoff": str(cutoff) if cutoff else "",
            "n_candidates": n,
            "top_predictions": top_predictions,
            "overall_alert_priority": overall_priority,
            "alert_description": ALERT_PRIORITY_CONFIG[overall_priority]["description"],
            "actual_h3_cells": actual_h3,
            "actual_rank_in_candidates": actual_rank,
            "top5_hit": any(p["is_actual_h3"] for p in top_predictions),
            "model_metadata": {
                "model": "Phase 7 XGBoost LTR (FROZEN)",
                "model_path": f"{P7}/model.ubj",
                "features": 22,
                "training_complaints": 5652,
                "test_T3_hit5": 65.26,
                "test_T3_mrr":  0.4971,
                "kde_note": "KDE evaluated in Phase 9 but provided no demonstrated utility. Not used.",
                "seed": seed,
            }
        }
