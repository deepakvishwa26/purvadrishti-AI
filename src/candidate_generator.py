"""
HIVE-Predict Phase 6 — Candidate H3 Generator (v2)

Generates candidate H3 cells for XGBoost Learning-to-Rank training.

Design principles:
- Actual future H3 is NEVER used for candidate generation (no leakage).
  It may only be added post-hoc by the caller for label assignment.
- All candidate-specific features use only information observable at feature cutoff.
- Candidate generation is fully reproducible given seed=42.

v2 changes (vs v1):
  - Hotspot pool REMOVED (contributed 0.00% natural recall in audit)
  - Nearest ATM count: 10 -> 30 (primary recall driver: 68.82% of captures)
  - Random state ATMs REPLACED by all-victim-state ATMs (fixes 150-500km)
  - Neighboring-state ATMs ADDED (fixes Delhi 14.8% cross-state recall)
  - Global random: 4 -> 15

Candidate sources per complaint (v2):
  1. Ring neighbors of VICTIM H3 (rings 1-3): 36 cells — hard negatives
  2. 30 nearest ATM H3 cells by centroid proximity to victim
  3. ALL ATM H3 cells in victim's state (avg ~100 cells)
  4. 8 random ATMs per neighboring state (2-3 neighbors typical)
  5. 15 random global ATM H3 cells (exploration)
  6. [POST-HOC, CALLER ONLY] Actual future H3 cell(s) — label assignment only

After deduplication, typical candidate count: 150-200 per complaint.
"""

import numpy as np
import pandas as pd
import h3
from math import radians, sin, cos, sqrt, atan2
from collections import Counter
from typing import Dict, FrozenSet, List


# ---------------------------------------------------------------------------
# Indian state adjacency — for neighboring-state candidate generation.
# Based on the 15 states present in the ATM reference dataset.
# ---------------------------------------------------------------------------

STATE_NEIGHBORS: Dict[str, List[str]] = {
    "Bihar"           : ["Uttar Pradesh", "West Bengal", "Odisha"],
    "Delhi"           : ["Haryana", "Uttar Pradesh", "Rajasthan"],
    "Gujarat"         : ["Rajasthan", "Madhya Pradesh", "Maharashtra"],
    "Haryana"         : ["Delhi", "Punjab", "Uttar Pradesh", "Rajasthan"],
    "Karnataka"       : ["Maharashtra", "Telangana", "Tamil Nadu", "Kerala"],
    "Kerala"          : ["Karnataka", "Tamil Nadu"],
    "Madhya Pradesh"  : ["Rajasthan", "Gujarat", "Maharashtra", "Uttar Pradesh"],
    "Maharashtra"     : ["Gujarat", "Madhya Pradesh", "Telangana", "Karnataka"],
    "Odisha"          : ["West Bengal", "Bihar", "Telangana"],
    "Punjab"          : ["Haryana"],
    "Rajasthan"       : ["Delhi", "Haryana", "Punjab", "Uttar Pradesh",
                         "Madhya Pradesh", "Gujarat"],
    "Tamil Nadu"      : ["Karnataka", "Kerala", "Telangana"],
    "Telangana"       : ["Maharashtra", "Karnataka", "Odisha", "Tamil Nadu"],
    "Uttar Pradesh"   : ["Delhi", "Haryana", "Rajasthan", "Madhya Pradesh",
                         "Bihar", "West Bengal"],
    "West Bengal"     : ["Bihar", "Odisha", "Uttar Pradesh"],
}


# ---------------------------------------------------------------------------
# Haversine helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Scalar haversine distance in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _haversine_km_vec(lat1: float, lon1: float,
                      lat2_arr: np.ndarray, lon2_arr: np.ndarray) -> np.ndarray:
    """Vectorised haversine: one query point vs array of targets."""
    R = 6371.0
    dlat = np.radians(lat2_arr - lat1)
    dlon = np.radians(lon2_arr - lon1)
    a = (np.sin(dlat / 2) ** 2
         + cos(radians(lat1)) * np.cos(np.radians(lat2_arr)) * np.sin(dlon / 2) ** 2)
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def _h3_grid_distance(cell1: str, cell2: str) -> int:
    """H3 grid distance with km-based fallback for disconnected cells."""
    try:
        return h3.grid_distance(cell1, cell2)
    except Exception:
        lat1, lon1 = h3.cell_to_latlng(cell1)
        lat2, lon2 = h3.cell_to_latlng(cell2)
        dist_km = _haversine_km(lat1, lon1, lat2, lon2)
        return min(int(dist_km / 0.46), 9999)  # H3 res-8 edge ≈ 0.46 km


# ---------------------------------------------------------------------------
# CandidateGenerator — v2
# ---------------------------------------------------------------------------

class CandidateGenerator:
    """
    Generates candidate H3 cells for HIVE-Predict Learning-to-Rank.

    Call generate_candidates() to get the candidate set (no actual_h3 used).
    Call compute_candidate_features() to compute per-cell features.

    v2 improvements:
      - All victim-state ATMs included (not sampled)
      - Neighboring-state ATMs added (cross-state recall)
      - Nearest ATM count increased to 30
      - Hotspot pool removed (0% recall contribution)
    """

    def __init__(self, atm_df: pd.DataFrame,
                 seed_h3_counter: Counter,
                 config: dict,
                 rng: np.random.RandomState):
        self.atm_df          = atm_df
        self.seed_h3_counter = seed_h3_counter
        self.config          = config
        self.rng             = rng
        self.cfg             = config["candidate_generation"]
        self.h3_res          = config["general"]["h3_resolution"]

        # ── Pre-compute ATM H3 unique cell table with centroids ──────────
        self._build_atm_tables()

        # ── Pre-build ATM-per-H3 index: h3_cell -> count ─────────────────
        self._atm_h3_count: Dict[str, int] = (
            atm_df.groupby("h3_cell_res8").size().to_dict()
        )

        # ── Pre-build state set per H3 cell: h3_cell -> set of states ─────
        self._h3_cell_states: Dict[str, set] = (
            atm_df.groupby("h3_cell_res8")["state"]
            .apply(set)
            .to_dict()
        )

        # ── Pre-build state -> ATM H3 cells mapping ──────────────────────
        # Used for all-victim-state-ATMs and neighboring-state-ATMs (v2)
        self._state_atm_h3: Dict[str, np.ndarray] = {}
        for state in atm_df["state"].unique():
            cells = atm_df[atm_df["state"] == state]["h3_cell_res8"].unique()
            self._state_atm_h3[state] = cells

        # ── Pre-compute hotspot densities for all ATM cells ──────────────
        # Kept for compute_candidate_features() — NOT used in generation (v2)
        self._cand_hotspot_ring = 3
        self._precompute_hotspot_density()

        # ── v2 mode flags ────────────────────────────────────────────────
        self._use_all_state_atms  = self.cfg.get("use_all_victim_state_atms", False)
        self._nbr_state_count     = self.cfg.get("neighboring_state_atm_count", 0)
        self._max_state_atm_count = self.cfg.get("max_state_atm_count", None)  # None=unlimited

        # ── Pre-compute state ATM centroids for distance-sorted capping ───
        # Only needed if max_state_atm_count is set
        if self._max_state_atm_count is not None:
            # Build dict: state -> (h3_cells, lats, lons)
            self._state_atm_centroids: Dict[str, tuple] = {}
            for state in atm_df["state"].unique():
                subset = atm_df[atm_df["state"]==state].drop_duplicates("h3_cell_res8")
                cells = subset["h3_cell_res8"].values
                lats  = np.array([h3.cell_to_latlng(c)[0] for c in cells])
                lons  = np.array([h3.cell_to_latlng(c)[1] for c in cells])
                self._state_atm_centroids[state] = (cells, lats, lons)

    # ── Private helpers ──────────────────────────────────────────────────

    def _build_atm_tables(self):
        """Pre-compute unique ATM H3 cells with centroids for vectorised lookup."""
        uniq = self.atm_df.drop_duplicates("h3_cell_res8").copy()
        centroids = uniq["h3_cell_res8"].apply(h3.cell_to_latlng)
        uniq["_clat"] = centroids.apply(lambda x: x[0])
        uniq["_clon"] = centroids.apply(lambda x: x[1])
        self._atm_h3_cells  = uniq["h3_cell_res8"].values   # str array
        self._atm_h3_clat   = uniq["_clat"].values           # float array
        self._atm_h3_clon   = uniq["_clon"].values           # float array

    def _precompute_hotspot_density(self):
        """
        Pre-compute historical seed-event hotspot density for every ATM H3 cell.
        density = mean(seed_counter[c] for c in grid_disk(cell, ring))
        Uses SEED events only (pre-2024-01-01) — no future data.
        """
        ring = self._cand_hotspot_ring
        self._cand_hotspot_cache: Dict[str, float] = {}
        for cell in self._atm_h3_cells:
            try:
                nbrs = h3.grid_disk(cell, ring)
                cnt  = sum(self.seed_h3_counter.get(c, 0) for c in nbrs)
                self._cand_hotspot_cache[cell] = round(cnt / max(len(nbrs), 1), 6)
            except Exception:
                self._cand_hotspot_cache[cell] = 0.0

    def _hotspot_density(self, cell: str) -> float:
        """Look up pre-computed hotspot density; compute on-the-fly if missing."""
        if cell in self._cand_hotspot_cache:
            return self._cand_hotspot_cache[cell]
        try:
            ring = self._cand_hotspot_ring
            nbrs = h3.grid_disk(cell, ring)
            val  = round(sum(self.seed_h3_counter.get(c, 0) for c in nbrs)
                         / max(len(nbrs), 1), 6)
        except Exception:
            val = 0.0
        self._cand_hotspot_cache[cell] = val
        return val

    # ── Public: candidate generation ─────────────────────────────────────

    def generate_candidates(self, victim_h3: str, victim_state: str,
                            victim_lat: float, victim_lon: float) -> frozenset:
        """
        Generate candidate H3 cells for a single complaint.

        DOES NOT use actual_h3_cell — leakage-free by design.
        Caller adds actual_h3 after this call for label assignment only.

        v2 sources:
          1. Ring neighbours of victim H3 (rings 1-3): ~36 cells
          2. N nearest ATM H3 cells (N=30 in v2)
          3. ALL ATM H3 cells in victim's state (v2: replaces 5-random)
          4. K random ATMs per neighboring state (v2: new)
          5. M global random ATM H3 cells

        Returns:
            frozenset of candidate H3 cell strings (deduped).
        """
        cfg = self.cfg
        candidates: set = set()

        # ── Source 1: Ring neighbours of victim H3 ───────────────────────
        ring_max = cfg.get("victim_ring_max", 3)
        for ring in range(1, ring_max + 1):
            try:
                candidates.update(h3.grid_ring(victim_h3, ring))
            except Exception:
                pass

        # ── Source 2: Nearest ATM H3 cells by distance to victim ─────────
        n_near = cfg.get("atm_nearby_count", 30)  # v2 default: 30
        dists  = _haversine_km_vec(victim_lat, victim_lon,
                                   self._atm_h3_clat, self._atm_h3_clon)
        nearest_idx = np.argsort(dists)[:n_near]
        for idx in nearest_idx:
            candidates.add(self._atm_h3_cells[idx])

        # ── Source 3: State ATM H3 cells ──────────────────────────────────
        # v2 (use_all_victim_state_atms=true): include all cells in victim state.
        #   If max_state_atm_count is set, take the closest N by centroid distance.
        # v1 fallback: sample atm_state_count random cells.
        if self._use_all_state_atms:
            cap = self._max_state_atm_count
            if cap is not None and victim_state in self._state_atm_centroids:
                cells, lats, lons = self._state_atm_centroids.get(
                    victim_state, (np.array([]), np.array([]), np.array([]))
                )
                if len(cells) > cap:
                    d = _haversine_km_vec(victim_lat, victim_lon, lats, lons)
                    idx = np.argsort(d)[:cap]
                    candidates.update(cells[idx].tolist())
                else:
                    candidates.update(cells.tolist())
            else:
                state_pool = self._state_atm_h3.get(victim_state, np.array([]))
                candidates.update(state_pool.tolist())
        else:
            n_state = cfg.get("atm_state_count", 5)
            pool    = self._state_atm_h3.get(victim_state, self._atm_h3_cells)
            if len(pool) > 0:
                chosen = self.rng.choice(pool, size=min(n_state, len(pool)), replace=False)
                candidates.update(chosen.tolist())

        # ── Source 4: Neighboring-state ATM H3 cells (v2 only) ───────────
        if self._nbr_state_count > 0:
            neighbors = STATE_NEIGHBORS.get(victim_state, [])
            for nbr_state in neighbors:
                nbr_pool = self._state_atm_h3.get(nbr_state, np.array([]))
                if len(nbr_pool) > 0:
                    n = min(self._nbr_state_count, len(nbr_pool))
                    chosen = self.rng.choice(nbr_pool, size=n, replace=False)
                    candidates.update(chosen.tolist())

        # ── Source 5: Global random ATM H3 cells (exploration) ───────────
        n_rand = cfg.get("atm_random_count", 15)  # v2 default: 15
        if n_rand > 0:
            idx = self.rng.choice(len(self._atm_h3_cells),
                                  size=min(n_rand, len(self._atm_h3_cells)),
                                  replace=False)
            for i in idx:
                candidates.add(self._atm_h3_cells[i])

        # ── Source 6 (REMOVED in v2): Global hotspot pool ────────────────
        # Contributed 0.00% natural recall in audit (seed events ≠ ATM cells).
        # Kept as dead code for v1 compatibility:
        n_hot = cfg.get("hotspot_sample_count", 0)
        # n_hot is 0 in v2 config — no-op.

        return frozenset(candidates)

    # ── Public: per-candidate features ───────────────────────────────────

    def compute_candidate_features(self, candidate_h3: str,
                                   victim_h3: str,
                                   victim_lat: float,
                                   victim_lon: float,
                                   victim_state: str) -> dict:
        """
        Compute features for a single (complaint, candidate_h3) pair.

        All inputs are observable at feature cutoff.
        actual_h3 is NOT a parameter — no leakage possible.

        Features (7 candidate-level):
            cand_dist_km_from_victim   : haversine km, victim -> candidate
            cand_h3_grid_dist          : H3 grid ring distance
            cand_atm_count             : ATMs whose h3_cell_res8 == candidate_h3
            cand_atm_density           : ATMs in candidate + ring-1 neighbors (7 cells)
            cand_hotspot_density       : seed withdrawal density in candidate ring-3
            cand_in_victim_state       : 1 if candidate ring-1 has a victim-state ATM
            cand_is_victim_h3          : 1 if candidate == victim_h3

        Returns:
            dict of feature name -> value
        """
        # Candidate centroid
        try:
            c_lat, c_lon = h3.cell_to_latlng(candidate_h3)
        except Exception:
            c_lat, c_lon = victim_lat, victim_lon

        # Distance victim -> candidate
        dist_km = round(_haversine_km(victim_lat, victim_lon, c_lat, c_lon), 2)

        # H3 grid distance
        grid_dist = _h3_grid_distance(victim_h3, candidate_h3)

        # ATM counts — O(1) dict lookups
        cand_atm_count = self._atm_h3_count.get(candidate_h3, 0)

        try:
            ring1_nbrs       = h3.grid_disk(candidate_h3, 1)   # 7 cells incl. self
            cand_atm_density = sum(self._atm_h3_count.get(c, 0) for c in ring1_nbrs)
        except Exception:
            cand_atm_density = cand_atm_count

        # Historical hotspot density (seed events only — pre-2024)
        cand_hotspot = self._hotspot_density(candidate_h3)

        # Is candidate in victim's state? — O(1) dict lookup
        try:
            ring1_nbrs = h3.grid_disk(candidate_h3, 1)
            in_state   = int(any(
                victim_state in self._h3_cell_states.get(c, set())
                for c in ring1_nbrs
            ))
        except Exception:
            in_state = 0

        # Is candidate the same H3 as victim?
        is_victim_h3 = int(candidate_h3 == victim_h3)

        return {
            "cand_dist_km_from_victim" : dist_km,
            "cand_h3_grid_dist"        : grid_dist,
            "cand_atm_count"           : cand_atm_count,
            "cand_atm_density"         : cand_atm_density,
            "cand_hotspot_density"     : cand_hotspot,
            "cand_in_victim_state"     : in_state,
            "cand_is_victim_h3"        : is_victim_h3,
        }
