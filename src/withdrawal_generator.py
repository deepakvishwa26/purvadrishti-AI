"""
Withdrawal generator module — V2 with temporal correctness and ATM diversity.

Generates cash-out withdrawal events at ATMs.
Withdrawals are the ground-truth outcome that HIVE-Predict learns to predict.

V2 changes:
- ALL withdrawals occur AFTER feature_cutoff_timestamp (= complaint_registered_at).
  This enforces the prediction-time timeline:
  incident < registration = cutoff < withdrawal.
- ATM selection uses a weighted mixture:
  - victim_proximity_weight: select ATM near victim
  - mule_drift_weight: select ATM near drifted mule chain location
  - random_selection_weight: select random ATM anywhere
- NO_CASHOUT cases produce zero withdrawals.
"""

import numpy as np
from datetime import datetime, timedelta

from src.geography import select_nearby_atm, select_random_atm, lat_lon_to_h3


def generate_withdrawals(complaint, final_account_id, available_amount,
                          last_chain_timestamp, profile_name, profile_cfg,
                          atm_df, config, rng, withdrawal_counter,
                          drifted_lat=None, drifted_lon=None,
                          do_cashout=True):
    """
    Generate withdrawal(s) for a fraud case.

    V2: Enforces all withdrawals occur after feature_cutoff_timestamp.
    V2: Uses weighted ATM selection mixture for geographic diversity.

    Args:
        complaint: Complaint dict (contains victim_lat/lon, complaint_registered_at).
        final_account_id: The account holding funds at end of mule chain.
        available_amount: Funds available for withdrawal.
        last_chain_timestamp: Timestamp of last mule chain activity (datetime).
        profile_name: Behavioral profile name.
        profile_cfg: Profile configuration dict.
        atm_df: ATM reference DataFrame.
        config: Full configuration dict.
        rng: numpy RandomState.
        withdrawal_counter: Current counter for unique IDs.
        drifted_lat: V2 — drifted latitude from mule chain geographic drift.
        drifted_lon: V2 — drifted longitude from mule chain geographic drift.
        do_cashout: V2 — whether cashout actually occurs (profile decoupling).

    Returns:
        tuple: (list of withdrawal dicts, updated withdrawal_counter)
    """
    if not do_cashout:
        return [], withdrawal_counter

    # V2: Get feature cutoff (= complaint_registered_at)
    feature_cutoff = datetime.fromisoformat(complaint["complaint_registered_at"])

    # Cashout delay config
    cashout_delay_cfg = profile_cfg.get("cashout_delay_seconds")
    if cashout_delay_cfg is None:
        # Fallback for NO_CASHOUT surprise cashout
        cashout_delay_cfg = {"min": 900, "max": 7200}

    delay_min = cashout_delay_cfg["min"]
    delay_max = cashout_delay_cfg["max"]

    financial_cfg = config["financial"]
    h3_res = config["general"]["h3_resolution_primary"]
    withdrawal_frac_min = financial_cfg["withdrawal_fraction"]["min"]
    withdrawal_frac_max = financial_cfg["withdrawal_fraction"]["max"]

    # V2: ATM selection weights
    atm_sel_cfg = config.get("atm_selection", {})
    victim_weight = atm_sel_cfg.get("victim_proximity_weight", 0.4)
    drift_weight = atm_sel_cfg.get("mule_drift_weight", 0.45)
    random_weight = atm_sel_cfg.get("random_selection_weight", 0.15)
    total_w = victim_weight + drift_weight + random_weight
    victim_weight /= total_w
    drift_weight /= total_w
    random_weight /= total_w

    # How many withdrawals
    num_w_min = max(1, profile_cfg["num_withdrawals"].get("min", 1))
    num_w_max = max(1, profile_cfg["num_withdrawals"].get("max", 1))
    num_withdrawals = int(rng.randint(num_w_min, num_w_max + 1))

    # Total withdrawal amount
    total_frac = rng.uniform(withdrawal_frac_min, withdrawal_frac_max)
    total_withdraw = round(available_amount * total_frac, 2)

    # Split across withdrawals
    if num_withdrawals == 1:
        amounts = [total_withdraw]
    else:
        splits = rng.dirichlet(np.ones(num_withdrawals))
        amounts = [round(total_withdraw * s, 2) for s in splits]

    # V2: CRITICAL — earliest withdrawal must be AFTER feature_cutoff
    # Use max(feature_cutoff, last_chain_timestamp) + delay
    earliest_start = max(feature_cutoff, last_chain_timestamp)
    current_time = earliest_start

    # Use drifted location if available, else victim location
    if drifted_lat is None:
        drifted_lat = complaint["victim_lat"]
    if drifted_lon is None:
        drifted_lon = complaint["victim_lon"]

    victim_lat = complaint["victim_lat"]
    victim_lon = complaint["victim_lon"]

    withdrawals = []

    for i, w_amount in enumerate(amounts):
        withdrawal_counter += 1
        withdrawal_id = f"WDR{str(withdrawal_counter).zfill(8)}"

        # Time progression: each withdrawal after the previous
        if i == 0:
            delay = int(rng.uniform(delay_min, delay_max))
        else:
            delay = int(rng.uniform(300, 1800))  # 5-30 min between withdrawals
        current_time = current_time + timedelta(seconds=delay)

        # Ensure amount respects available funds
        effective_amount = round(min(w_amount, available_amount), 2)
        if effective_amount <= 0:
            continue

        # V2: Weighted ATM selection
        selection = rng.choice(
            ["victim", "drift", "random"],
            p=[victim_weight, drift_weight, random_weight]
        )

        if selection == "victim":
            atm = select_nearby_atm(victim_lat, victim_lon, atm_df, rng)
        elif selection == "drift":
            atm = select_nearby_atm(drifted_lat, drifted_lon, atm_df, rng,
                                     max_distance_km=80)
        else:
            atm = select_random_atm(atm_df, rng)

        lat = atm["latitude"]
        lon = atm["longitude"]
        h3_cell = lat_lon_to_h3(lat, lon, h3_res)

        withdrawals.append({
            "withdrawal_id": withdrawal_id,
            "complaint_id": complaint["complaint_id"],
            "account_id": final_account_id,
            "withdrawal_timestamp": current_time.isoformat(),
            "amount": effective_amount,
            "atm_id": atm["atm_id"],
            "latitude": lat,
            "longitude": lon,
            "h3_cell": h3_cell,
        })

        available_amount -= effective_amount
        # Update reference for next withdrawal
        drifted_lat = lat
        drifted_lon = lon

    return withdrawals, withdrawal_counter
