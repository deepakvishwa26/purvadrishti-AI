"""
Phase 9 — KDE Engine (Corrected)
==================================
Temporal, per-complaint Gaussian KDE over historical withdrawal coordinates.

Coordinate system
-----------------
Raw lat/lon are converted to approximate metric (km) coordinates using an
equirectangular projection centred on India:

    x = R * lon_rad * cos(lat0_rad)
    y = R * lat_rad

    R    = 6371 km
    lat0 = 22.0 degrees N  (approximate centre of India study region)

This makes the bandwidth parameter directly interpretable in kilometres.
No heavy GIS dependency required — only numpy.

Temporal leakage rule (enforced inside score_complaint)
-------------------------------------------------------
For every complaint at cutoff T, ONLY events satisfying:

    withdrawal_timestamp < T   (strict less-than)
    AND
    complaint_id != current_complaint_id

are used. No future withdrawals, no current complaint withdrawal.

Fallback behaviour for sparse history
--------------------------------------
count >= 5  → fit Gaussian KDE on available pre-cutoff pool   ("local")
count 1–4   → fit Gaussian KDE on sparse pool                 ("sparse_local")
count == 0  → assign log(epsilon) to all candidates            ("uniform_fallback")

No fixed global prior is used. The fallback always uses the temporally valid
pre-cutoff pool, so the rule above holds for every complaint regardless of
its position in the temporal sequence.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity


# ── Constants ────────────────────────────────────────────────────────
R         = 6371.0       # Earth radius, km
LAT0_DEG  = 22.0         # Reference latitude — approximate centre of India
LAT0_RAD  = np.radians(LAT0_DEG)
LOG_EPSILON = 1e-30      # floor before taking log


def latlon_to_xy(lats, lons):
    """
    Equirectangular projection → (x_km, y_km).
    x = R * lon_rad * cos(lat0)
    y = R * lat_rad
    """
    lat_rad = np.radians(np.asarray(lats, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lons, dtype=np.float64))
    x = R * lon_rad * np.cos(LAT0_RAD)
    y = R * lat_rad
    return x, y


def _fit_kde(lats, lons, bandwidth_km):
    """Fit sklearn Gaussian KDE in metric (km) space."""
    x, y = latlon_to_xy(lats, lons)
    X = np.column_stack([x, y])
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth_km)
    kde.fit(X)
    return kde


def _score_kde(kde, lats, lons):
    """
    Evaluate KDE at query lat/lon positions.
    Returns (densities, log_densities) — both arrays of shape (n,).
    """
    x, y = latlon_to_xy(lats, lons)
    Q = np.column_stack([x, y])
    log_dens = kde.score_samples(Q)       # sklearn returns log-density
    dens     = np.exp(log_dens)
    return dens, log_dens


class KDEEngine:
    """
    Per-complaint temporal KDE scorer.

    All temporal and identity leakage rules enforced inside score_complaint().
    No global prior is kept — each complaint uses its own temporally valid pool.

    Parameters
    ----------
    wdr_df : DataFrame with columns
        withdrawal_timestamp (datetime-like), latitude, longitude, complaint_id
    bandwidth_km : float
        Gaussian KDE bandwidth in kilometres
    min_local_events : int
        Threshold below which a sparse-local KDE is used (still valid).
        Set to 5 as default.  count==0 → uniform_fallback.
    """

    def __init__(self, wdr_df: pd.DataFrame, bandwidth_km: float,
                 min_local_events: int = 5):
        self.bandwidth_km   = bandwidth_km
        self.min_events     = min_local_events

        wdr = wdr_df[["complaint_id","withdrawal_timestamp",
                       "latitude","longitude"]].copy()
        wdr["withdrawal_timestamp"] = pd.to_datetime(wdr["withdrawal_timestamp"])
        self._wdr = wdr.sort_values("withdrawal_timestamp").reset_index(drop=True)

        # Counters for reporting
        self._cnt_local   = 0    # count >= min_events
        self._cnt_sparse  = 0    # 1 <= count < min_events
        self._cnt_uniform = 0    # count == 0

    # ── Sparsity audit ────────────────────────────────────────────────
    def measure_sparsity(self, complaints: pd.DataFrame) -> dict:
        """
        For each complaint in `complaints` (must have complaint_id,
        feature_cutoff_timestamp), count the number of valid pre-cutoff
        OTHER-complaint withdrawals.

        Returns distribution dict and per-complaint series.
        """
        counts = {}
        for row in complaints.itertuples(index=False):
            cid = row.complaint_id
            cut = pd.Timestamp(row.feature_cutoff_timestamp)
            mask = (self._wdr["withdrawal_timestamp"] < cut) & \
                   (self._wdr["complaint_id"] != cid)
            counts[cid] = int(mask.sum())

        dist = {"0":0, "1-4":0, "5-9":0, "10-49":0, "50-99":0, "100+":0}
        for n in counts.values():
            if n == 0:          dist["0"]     += 1
            elif n < 5:         dist["1-4"]   += 1
            elif n < 10:        dist["5-9"]   += 1
            elif n < 50:        dist["10-49"] += 1
            elif n < 100:       dist["50-99"] += 1
            else:               dist["100+"]  += 1
        return {"distribution": dist, "per_complaint": counts}

    # ── Per-complaint scoring ─────────────────────────────────────────
    def score_complaint(self, complaint_id: str, feature_cutoff_ts,
                        candidate_lats: np.ndarray,
                        candidate_lons: np.ndarray):
        """
        Score candidate H3 centroids for a single complaint.

        Enforced rules
        --------------
        1. withdrawal_timestamp < feature_cutoff_ts  (strict)
        2. complaint_id != complaint_id (exclude current complaint)

        Returns
        -------
        densities     : ndarray (n_candidates,)
        log_densities : ndarray (n_candidates,)
        source        : str — "local" | "sparse_local" | "uniform_fallback"
        n_events      : int — number of historical events used
        """
        feature_cutoff_ts = pd.Timestamp(feature_cutoff_ts)

        mask = (self._wdr["withdrawal_timestamp"] < feature_cutoff_ts) & \
               (self._wdr["complaint_id"] != complaint_id)
        pool = self._wdr[mask]
        n    = len(pool)

        if n == 0:
            # Absolute fallback — no historical data at all
            self._cnt_uniform += 1
            dens     = np.full(len(candidate_lats), LOG_EPSILON)
            log_dens = np.full(len(candidate_lats), np.log(LOG_EPSILON))
            return dens, log_dens, "uniform_fallback", 0

        # Fit KDE on the temporally valid pool (local or sparse-local)
        kde = _fit_kde(pool["latitude"].values,
                       pool["longitude"].values,
                       self.bandwidth_km)
        dens, log_dens = _score_kde(kde, candidate_lats, candidate_lons)

        if n >= self.min_events:
            self._cnt_local  += 1
            source = "local"
        else:
            self._cnt_sparse += 1
            source = "sparse_local"

        return dens, log_dens, source, n

    def stats(self):
        total = self._cnt_local + self._cnt_sparse + self._cnt_uniform
        return {
            "bandwidth_km": self.bandwidth_km,
            "coordinate_system": "equirectangular (x=R*lon*cos(lat0), y=R*lat), lat0=22N",
            "reference_latitude_deg": LAT0_DEG,
            "earth_radius_km": R,
            "min_events_for_local_kde": self.min_events,
            "complaints_scored": total,
            "local_kde_complaints": self._cnt_local,
            "sparse_local_complaints": self._cnt_sparse,
            "uniform_fallback_complaints": self._cnt_uniform,
            "pct_local": round(self._cnt_local/total*100, 1) if total else 0,
            "pct_sparse": round(self._cnt_sparse/total*100, 1) if total else 0,
            "pct_fallback": round(self._cnt_uniform/total*100, 1) if total else 0,
        }

    def reset_counts(self):
        self._cnt_local = self._cnt_sparse = self._cnt_uniform = 0
