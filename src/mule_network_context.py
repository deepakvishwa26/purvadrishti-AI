"""
HIVE-Predict Phase 8 — Mule-Network Geography Loader
=====================================================
Builds pre-computation tables from mule_chains + withdrawals + suspects
that are CUTOFF-VALID (no future leakage).

Safety rules:
1. Only mule_chain hops with transaction_timestamp <= feature_cutoff_timestamp
2. For withdrawal geography lookups: only withdrawals from OTHER complaints
   with withdrawal_timestamp < current complaint's feature_cutoff_timestamp
3. Suspects address parsed for state-level signals only (prediction-time valid)

Design: All data is loaded once and indexed; per-complaint lookups are O(1)/O(k).
"""

import re
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from collections import defaultdict


# ─── Haversine ──────────────────────────────────────────────────────────────

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1); dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


# ─── Pincode → State lookup (rough mapping for Indian pincodes) ─────────────

PINCODE_STATE = {
    range(110001, 110110): "Delhi",
    range(120001, 135001): "Haryana",
    range(140001, 161000): "Punjab",
    range(160001, 161000): "Punjab",
    range(200001, 204000): "Uttar Pradesh",
    range(201001, 284000): "Uttar Pradesh",
    range(226001, 285000): "Uttar Pradesh",
    range(800001, 856000): "Bihar",
    range(700001, 743000): "West Bengal",
    range(560001, 597000): "Karnataka",
    range(600001, 643000): "Tamil Nadu",
    range(500001, 534000): "Telangana",
    range(380001, 396000): "Gujarat",
    range(400001, 445000): "Maharashtra",
    range(302001, 344000): "Rajasthan",
    range(462001, 490000): "Madhya Pradesh",
    range(670001, 696000): "Kerala",
    range(751001, 770000): "Odisha",
    range(500001, 509000): "Telangana",
}


def pincode_to_state(pincode_str):
    """Parse a 6-digit pincode from address text → approximate state."""
    m = re.search(r'\b(\d{6})\b', str(pincode_str))
    if not m:
        return None
    pin = int(m.group(1))
    for r, state in PINCODE_STATE.items():
        if pin in r:
            return state
    return None


# ─── MuleNetworkContext ───────────────────────────────────────────────────────

class MuleNetworkContext:
    """
    Pre-computed, cutoff-safe mule-network geography.

    Loaded once; queried per complaint.

    Safety:
      - All withdrawal lookups exclude current complaint and enforce ts < cutoff.
      - All mule chain traversals filter transaction_timestamp <= cutoff.
      - No actual_h3_cell from the current complaint is ever read.
    """

    def __init__(self, mc_path, wdr_path, sus_path, atm_path, complaints_df):
        print("  Loading mule network context...")
        mc  = pd.read_csv(mc_path,  parse_dates=["transaction_timestamp"])
        wdr = pd.read_csv(wdr_path, parse_dates=["withdrawal_timestamp"])
        sus = pd.read_csv(sus_path)
        atm = pd.read_csv(atm_path)

        self._mc  = mc
        self._wdr = wdr
        self._atm = atm

        # ATM H3 → state
        self._h3_state = atm.set_index("h3_cell_res8")["state"].to_dict()
        self._h3_latlon = {
            row["h3_cell_res8"]: (row["latitude"], row["longitude"])
            for _, row in atm.iterrows()
        }

        # ATM cells by state (all, for candidate generation)
        self._state_atm_h3 = defaultdict(list)
        for _, row in atm.iterrows():
            self._state_atm_h3[row["state"]].append(row["h3_cell_res8"])

        # Victim state per complaint
        self._victim_state = complaints_df.set_index("complaint_id")["victim_state"].to_dict()
        self._victim_lat   = complaints_df.set_index("complaint_id")["victim_lat"].to_dict()
        self._victim_lon   = complaints_df.set_index("complaint_id")["victim_lon"].to_dict()

        # Build: account → list of (withdrawal_timestamp, h3_cell, complaint_id)
        # Used for historical cashout geography lookups
        self._acct_history = defaultdict(list)
        for _, row in wdr.iterrows():
            self._acct_history[row["account_id"]].append(
                (row["withdrawal_timestamp"], row["h3_cell"], row["complaint_id"])
            )

        # Suspects: complaint → parsed state from address
        self._suspect_state = {}
        for _, row in sus.iterrows():
            state = pincode_to_state(row.get("suspect_address", ""))
            if state:
                self._suspect_state[row["complaint_id"]] = state

        # Mule chain grouped by complaint
        self._mc_by_cid = self._mc.groupby("complaint_id")

        print(f"    Mule chain records: {len(mc):,}")
        print(f"    Withdrawal records: {len(wdr):,}")
        print(f"    Accounts with withdrawal history: {len(self._acct_history):,}")
        print(f"    Suspect state extracted: {len(self._suspect_state):,} / {len(sus):,}")

    # ── Primary interface ─────────────────────────────────────────────────────

    def get_mule_geography(self, complaint_id, feature_cutoff_ts, max_states=3, atm_per_state=15):
        """
        For a given complaint + cutoff, return:
          - mule_states: ranked list of states inferred from pre-cutoff mule chain
          - candidate_h3: ATM H3 cells in those states (distance-sorted from victim)
          - source_tags: dict mapping h3 → source label
          - features: complaint-level mule context features
        """
        # Ensure timestamp comparison works regardless of input type
        feature_cutoff_ts = pd.Timestamp(feature_cutoff_ts)
        victim_lat = self._victim_lat.get(complaint_id)
        victim_lon = self._victim_lon.get(complaint_id)

        # Get pre-cutoff mule chain for this complaint
        if complaint_id not in self._mc_by_cid.groups:
            return [], {}, {}

        chain = self._mc_by_cid.get_group(complaint_id)
        pre_chain = chain[chain["transaction_timestamp"] <= feature_cutoff_ts]

        if len(pre_chain) == 0:
            return [], {}, {}

        # Sort hops
        pre_chain = pre_chain.sort_values("hop_number")
        max_hop = int(pre_chain["hop_number"].max())

        # All destination accounts in pre-cutoff chain (by hop number)
        dest_by_hop = {}
        for _, row in pre_chain.iterrows():
            h = int(row["hop_number"])
            dest_by_hop[h] = row["destination_account"]

        # State vote accumulation (weighted by tx amount and hop position)
        state_tx_amount = defaultdict(float)
        state_tx_count  = defaultdict(int)
        state_sources   = defaultdict(list)

        # ── Process each dest account (SAFE: use only other complaints pre-cutoff) ──
        for hop, acct in dest_by_hop.items():
            # Look up historical withdrawals of this account from OTHER complaints
            # only before feature_cutoff_timestamp
            for ts, h3, cid in self._acct_history.get(acct, []):
                if cid == complaint_id:
                    continue  # SAFETY: skip current complaint's withdrawal
                if ts >= feature_cutoff_ts:
                    continue  # SAFETY: skip future withdrawals
                state = self._h3_state.get(h3)
                if state:
                    # Weight by amount and inverse hop position (closer hops = more weight)
                    hop_weight = 1.0 / (hop + 1)
                    tx_amt = float(pre_chain.loc[pre_chain["hop_number"]==hop, "transaction_amount"].iloc[0]
                                   if len(pre_chain[pre_chain["hop_number"]==hop]) > 0 else 1000)
                    state_tx_amount[state] += tx_amt * hop_weight
                    state_tx_count[state]  += 1
                    source = ("intermediate" if hop < max_hop else "terminal")
                    state_sources[state].append(source)

        # ── Supplement with suspect address state ──────────────────────────────
        sus_state = self._suspect_state.get(complaint_id)
        if sus_state:
            state_tx_amount[sus_state] += 5000  # moderate prior
            state_tx_count[sus_state]  += 1
            state_sources[sus_state].append("suspect_address")

        # ── Rank states by weighted tx amount ─────────────────────────────────
        mule_states = sorted(state_tx_amount, key=lambda s: -state_tx_amount[s])[:max_states]

        # ── Generate ATM H3 candidates in top mule states ─────────────────────
        candidate_h3   = []
        source_tags    = {}

        for st in mule_states:
            pool = self._state_atm_h3.get(st, [])
            if not pool:
                continue
            # Sort by distance from victim
            if victim_lat is not None:
                dists = [
                    (h3, _haversine_km(victim_lat, victim_lon,
                                       *self._h3_latlon.get(h3, (victim_lat, victim_lon))))
                    for h3 in pool
                ]
                dists.sort(key=lambda x: x[1])
                selected = [h3 for h3, _ in dists[:atm_per_state]]
            else:
                selected = pool[:atm_per_state]

            for h3 in selected:
                candidate_h3.append(h3)
                source_tags[h3] = f"mule_state_atm:{st}"

        # ── Complaint-level mule context features ─────────────────────────────
        features = {
            "n_pre_cutoff_hops"    : len(pre_chain),
            "max_pre_cutoff_hop"   : max_hop,
            "mule_state_diversity" : len(state_tx_amount),
            "mule_identified_states": list(mule_states),
            "mule_state_tx_amounts" : dict(state_tx_amount),
            "mule_state_tx_counts"  : dict(state_tx_count),
            "has_suspect_address_state": int(sus_state is not None),
        }

        return candidate_h3, source_tags, features

    def compute_candidate_mule_features(self, candidate_h3, mule_geo_result, victim_lat, victim_lon):
        """
        Compute candidate-level mule features for a single candidate H3 cell.
        mule_geo_result = (candidate_h3_list, source_tags, features) from get_mule_geography()
        """
        _, source_tags, feat = mule_geo_result

        mule_states         = feat.get("mule_identified_states", [])
        mule_state_amounts  = feat.get("mule_state_tx_amounts", {})
        mule_state_counts   = feat.get("mule_state_tx_counts", {})

        # Is this candidate in a mule-identified state?
        cand_state = self._h3_state.get(candidate_h3)
        in_mule_state = int(cand_state in mule_states) if cand_state else 0

        # Transaction amount / count flowing to this candidate's state
        cand_mule_tx_amount = mule_state_amounts.get(cand_state, 0.0)
        cand_mule_tx_count  = mule_state_counts.get(cand_state, 0)

        # Distance from candidate H3 centroid to nearest mule-chain account centroid
        # (using historical withdrawal locations of pre-cutoff dest accounts)
        # We use the source_tags from candidate gen: if candidate was generated
        # via mule_state_atm, its source tag contains the state — we can't easily
        # get exact centroid here without extra lookup, so we use 0 as default
        # (kept simple; state-level features above capture the key signal)

        return {
            "cand_in_mule_state"       : in_mule_state,
            "cand_mule_state_tx_amount": float(cand_mule_tx_amount),
            "cand_mule_state_tx_count" : int(cand_mule_tx_count),
        }

    def get_complaint_mule_features(self, complaint_id, feature_cutoff_ts):
        """Complaint-level mule context features (same value across all candidates)."""
        feature_cutoff_ts = pd.Timestamp(feature_cutoff_ts)
        if complaint_id not in self._mc_by_cid.groups:
            return {"n_pre_cutoff_hops": 0, "max_pre_cutoff_hop": 0,
                    "mule_state_diversity": 0, "has_suspect_address_state": 0}
        chain = self._mc_by_cid.get_group(complaint_id)
        pre   = chain[chain["transaction_timestamp"] <= feature_cutoff_ts]
        sus_state = self._suspect_state.get(complaint_id)
        return {
            "n_pre_cutoff_hops"         : len(pre),
            "max_pre_cutoff_hop"        : int(pre["hop_number"].max()) if len(pre) > 0 else 0,
            "mule_state_diversity"      : 0,  # computed in get_mule_geography
            "has_suspect_address_state" : int(sus_state is not None),
        }
