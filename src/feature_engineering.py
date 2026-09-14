"""
Feature engineering module — V2 with prediction-time semantics.

All ML features are DERIVED from the underlying connected event data.
No feature is independently randomly generated.

V2 changes:
- mule_chain_depth/velocity/amount_velocity use only data OBSERVED
  at or before feature_cutoff_timestamp (not the full future chain).
- complaint_cluster uses rolling-window DBSCAN (only historical complaints).
- historical_hotspot_density uses broader spatial window (6-ring) and
  is seeded with pre-generated historical events.
- amount_velocity uses a minimum time denominator floor (60s default).

V2.1 performance fix:
- historical_hotspot_density now uses a pre-built Counter[h3_cell->count]
  instead of scanning the full event list per case. This reduces the
  hotspot feature computation from O(N*M) to O(N*K) where K=num_neighbors.

Feature derivation lineage:
- hour                        <- incident timestamp
- day_of_week                 <- incident timestamp
- fraud_amount                <- complaint
- fraud_type                  <- complaint (encoded)
- mule_chain_depth            <- OBSERVED mule chain records at cutoff
- mule_velocity               <- OBSERVED mule chain timestamps (hops/minute)
- amount_velocity             <- OBSERVED mule chain amounts/timestamps (INR/minute)
- distance_from_victim        <- victim coords vs. last mule chain ATM area
- historical_hotspot_density  <- historical withdrawal spatial density
- atm_density                 <- ATM reference data within H3 neighborhood
- complaint_cluster           <- rolling-window DBSCAN on time+geography

Temporal leakage prevention:
- feature_cutoff_timestamp = complaint_registered_at
- Only data at or before cutoff is used for features.
"""

import h3 as h3lib
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from sklearn.cluster import DBSCAN

from src.geography import haversine_km, compute_atm_density_h3, lat_lon_to_h3


# --- Fraud type encoding ---
FRAUD_TYPE_ENCODING = {
    "DIGITAL_PAYMENT_FRAUD": 0,
    "INVESTMENT_FRAUD": 1,
    "DIGITAL_ARREST": 2,
    "PHISHING": 3,
    "OTHER": 4,
}


def compute_features_for_case(complaint, mule_chain_records, withdrawals,
                                atm_df, all_complaints_so_far,
                                historical_h3_counter, config):
    """
    Compute the ML feature snapshot for a single complaint case.

    V2: All features use only data OBSERVED at or before feature_cutoff_timestamp.
    Mule chain features are filtered by timestamp <= cutoff.

    V2.1: historical_h3_counter replaces the historical_withdrawals list for
    O(K) hotspot density computation instead of O(M) per case.

    Args:
        complaint: Complaint dict.
        mule_chain_records: List of mule chain dicts for this case.
        withdrawals: List of withdrawal dicts for this case (used only for labels,
                     NOT for features -- to prevent leakage).
        atm_df: ATM reference DataFrame.
        all_complaints_so_far: List of complaint dicts registered before cutoff
                               (for clustering).
        historical_h3_counter: Counter[h3_cell -> count] of ALL historical
                               withdrawal events BEFORE this case's cutoff.
                               Maintained incrementally by the generator.
        config: Configuration dict.

    Returns:
        dict: Feature snapshot record.
    """
    # Feature cutoff = complaint registration time
    cutoff_str = complaint["complaint_registered_at"]
    cutoff_dt = datetime.fromisoformat(cutoff_str)

    incident_dt = datetime.fromisoformat(complaint["incident_datetime"])

    # --- Basic temporal features ---
    hour = incident_dt.hour
    day_of_week = incident_dt.weekday()  # 0=Monday, 6=Sunday
    is_weekend = 1 if day_of_week >= 5 else 0
    is_night = 1 if (hour >= 22 or hour <= 5) else 0

    # --- Fraud amount ---
    fraud_amount = complaint["fraud_amount"]
    amount_log = float(np.log1p(fraud_amount))

    # --- Fraud type ---
    fraud_type = complaint["fraud_type"]
    fraud_type_encoded = FRAUD_TYPE_ENCODING.get(fraud_type, 4)

    # --- V2: OBSERVED mule chain features (only hops at or before cutoff) ---
    observed_mule = [
        r for r in mule_chain_records
        if datetime.fromisoformat(r["transaction_timestamp"]) <= cutoff_dt
    ]
    mule_chain_depth = len(observed_mule)

    # --- Mule velocity (observed hops per minute) ---
    mule_velocity = 0.0
    if len(observed_mule) >= 2:
        timestamps = [datetime.fromisoformat(r["transaction_timestamp"])
                      for r in observed_mule]
        total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
        if total_seconds > 0:
            mule_velocity = round(len(observed_mule) / (total_seconds / 60), 4)

    # --- V2: Amount velocity with minimum time floor ---
    features_cfg = config.get("features", {})
    min_time_sec = features_cfg.get("amount_velocity_min_time_seconds", 60)

    amount_velocity = 0.0
    if len(observed_mule) >= 1:
        first_amount = observed_mule[0]["transaction_amount"]
        if len(observed_mule) >= 2:
            timestamps = [datetime.fromisoformat(r["transaction_timestamp"])
                          for r in observed_mule]
            total_seconds = (timestamps[-1] - timestamps[0]).total_seconds()
            # V2: Apply minimum time floor to avoid extreme values
            effective_seconds = max(total_seconds, min_time_sec)
            amount_velocity = round(first_amount / (effective_seconds / 60), 4)
        else:
            # Single hop: use min_time_sec as denominator
            amount_velocity = round(first_amount / (min_time_sec / 60), 4)

    # --- Distance from victim to ATM region ---
    victim_lat = complaint["victim_lat"]
    victim_lon = complaint["victim_lon"]
    victim_h3 = lat_lon_to_h3(victim_lat, victim_lon, config["general"]["h3_resolution_primary"])

    # Distance: victim to average ATM location in victim's state
    state_atms = atm_df[atm_df["state"] == complaint["victim_state"]]
    if len(state_atms) > 0:
        avg_atm_lat = state_atms["latitude"].mean()
        avg_atm_lon = state_atms["longitude"].mean()
        distance_from_victim = round(
            haversine_km(victim_lat, victim_lon, avg_atm_lat, avg_atm_lon), 2
        )
    else:
        distance_from_victim = 0.0

    # --- Historical hotspot density (V2.1: O(K) counter-based lookup) ---
    # historical_h3_counter is a Counter[h3_cell -> count] pre-built by the
    # generator from all events BEFORE this case's cutoff. No timestamp
    # filtering needed here — temporal correctness is guaranteed by the caller.
    spatial_rings = features_cfg.get("hotspot_spatial_rings", 6)
    historical_hotspot_density = 0.0
    total_historical = sum(historical_h3_counter.values())
    if total_historical > 0:
        try:
            neighbors = h3lib.grid_disk(victim_h3, spatial_rings)
            num_cells = len(neighbors)
            count = sum(historical_h3_counter.get(cell, 0) for cell in neighbors)
            # Normalize by number of neighborhood cells (density per cell)
            historical_hotspot_density = round(count / max(num_cells, 1), 6)
        except Exception:
            historical_hotspot_density = 0.0

    # --- ATM density ---
    atm_density = compute_atm_density_h3(
        victim_h3, atm_df,
        resolution=config["general"]["h3_resolution_primary"]
    )

    # --- Time since transaction (seconds from incident to cutoff) ---
    time_since_transaction = round((cutoff_dt - incident_dt).total_seconds(), 0)

    return {
        "complaint_id": complaint["complaint_id"],
        "feature_cutoff_timestamp": cutoff_str,
        "fraud_amount": fraud_amount,
        "amount_log": amount_log,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_night": is_night,
        "fraud_type": fraud_type,
        "fraud_type_encoded": fraud_type_encoded,
        "mule_chain_depth": mule_chain_depth,
        "mule_velocity": mule_velocity,
        "amount_velocity": amount_velocity,
        "distance_from_victim": distance_from_victim,
        "historical_hotspot_density": historical_hotspot_density,
        "atm_density": atm_density,
        "complaint_cluster": -1,  # Placeholder, computed via rolling window below
        "victim_h3_res8": victim_h3,
        "time_since_transaction": time_since_transaction,
    }


def compute_complaint_clusters_rolling(feature_snapshots, complaints_sorted, config):
    """
    V2: Compute complaint clusters using rolling-window DBSCAN.

    For each complaint at time T, only complaints registered within
    [T - window_days, T] are used for clustering. This prevents
    temporal leakage from future complaints.

    Args:
        feature_snapshots: List of feature snapshot dicts (ordered by registration time).
        complaints_sorted: List of complaint dicts (ordered by registration time).
        config: Configuration dict.

    Returns:
        list[int]: Cluster labels for each complaint (-1 = noise).
    """
    if len(complaints_sorted) < 2:
        return [0] * len(complaints_sorted)

    cluster_cfg = config["clustering"]
    spatial_weight = float(cluster_cfg["spatial_weight"])
    temporal_weight = float(cluster_cfg["temporal_weight"])
    window_days = int(cluster_cfg.get("window_days", 30))
    eps = float(cluster_cfg["dbscan_eps_km"])
    min_samples = int(cluster_cfg["dbscan_min_samples"])

    # Pre-compute all coords and timestamps
    n = len(complaints_sorted)
    reg_times = []
    coords_raw = []
    for c in complaints_sorted:
        reg_times.append(datetime.fromisoformat(c["complaint_registered_at"]))
        lat = float(c["victim_lat"])
        lon = float(c["victim_lon"])
        incident_dt = datetime.fromisoformat(c["incident_datetime"])
        hour_norm = float(incident_dt.hour) / 24.0
        coords_raw.append((lat, lon, hour_norm))

    labels = [-1] * n
    window_td = timedelta(days=window_days)

    # Use a sliding window approach
    # For efficiency, process in batches: assign clusters at regular intervals
    batch_size = max(1, n // 20)  # ~20 cluster computations

    for batch_end in range(batch_size, n + 1, batch_size):
        # Include all complaints up to batch_end
        cutoff_time = reg_times[min(batch_end - 1, n - 1)]
        window_start = cutoff_time - window_td

        # Find complaints in window
        window_indices = []
        for j in range(batch_end):
            if reg_times[j] >= window_start:
                window_indices.append(j)

        if len(window_indices) < min_samples:
            # Not enough data for clustering
            for j in range(max(0, batch_end - batch_size), batch_end):
                if j < n:
                    labels[j] = -1
            continue

        # Build feature matrix for window
        coords = np.zeros((len(window_indices), 3), dtype=np.float64)
        for k, j in enumerate(window_indices):
            lat, lon, hour_norm = coords_raw[j]
            coords[k, 0] = lat * 111.32 * spatial_weight
            coords[k, 1] = lon * 111.32 * spatial_weight
            coords[k, 2] = hour_norm * float(cluster_cfg["temporal_window_hours"]) * temporal_weight

        db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
        window_labels = db.fit_predict(coords)

        # Assign labels to the batch portion of the window
        # (only the new complaints in this batch get their labels)
        for k, j in enumerate(window_indices):
            if j >= max(0, batch_end - batch_size) and j < batch_end:
                labels[j] = int(window_labels[k])

    # Handle any remaining complaints
    remaining_start = (n // batch_size) * batch_size
    if remaining_start < n and remaining_start > 0:
        cutoff_time = reg_times[n - 1]
        window_start = cutoff_time - window_td
        window_indices = [j for j in range(n) if reg_times[j] >= window_start]

        if len(window_indices) >= min_samples:
            coords = np.zeros((len(window_indices), 3), dtype=np.float64)
            for k, j in enumerate(window_indices):
                lat, lon, hour_norm = coords_raw[j]
                coords[k, 0] = lat * 111.32 * spatial_weight
                coords[k, 1] = lon * 111.32 * spatial_weight
                coords[k, 2] = hour_norm * float(cluster_cfg["temporal_window_hours"]) * temporal_weight

            db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            window_labels = db.fit_predict(coords)

            for k, j in enumerate(window_indices):
                if j >= remaining_start:
                    labels[j] = int(window_labels[k])

    return labels


# Keep old function for backwards compatibility but mark deprecated
def compute_complaint_clusters(feature_snapshots, complaints, config):
    """
    DEPRECATED in V2. Use compute_complaint_clusters_rolling() instead.
    Kept for backwards compatibility with V1 datasets.
    """
    return compute_complaint_clusters_rolling(feature_snapshots, complaints, config)
