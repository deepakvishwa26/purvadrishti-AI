"""
Mule chain generator module — V2 with account reuse and geographic drift.

Generates the money-laundering chain: a sequence of account hops
moving stolen funds from the initial destination account through
one or more intermediary mule accounts.

V2 changes:
- Uses get_or_create_mule() for account reuse across complaints.
- Adds geographic drift: each hop shifts the reference location,
  so the final cash-out location is not anchored to victim location.
- Returns the drifted geographic reference for the withdrawal generator.

Financial rules:
- Each hop loses a configurable friction percentage (2-10%).
- Amounts are always positive.
- The chain depth is determined by the behavioral profile.
- Timestamps are strictly monotonically increasing.
"""

import numpy as np
from datetime import datetime, timedelta
from math import radians, cos

from src.transaction_generator import generate_mule_transaction


def generate_mule_chain(complaint, initial_dest_account_id, initial_amount,
                         initial_tx_datetime, profile_cfg, config,
                         account_registry, rng, tx_counter):
    """
    Generate the mule chain for a fraud case.

    V2: Returns drifted geographic reference (lat, lon) for withdrawal selection.

    Args:
        complaint: Complaint dict.
        initial_dest_account_id: Account that received the initial fraud transfer.
        initial_amount: Amount that arrived at initial_dest_account_id.
        initial_tx_datetime: Timestamp of the initial transaction (datetime).
        profile_cfg: Behavioral profile configuration for this case.
        config: Full configuration dict.
        account_registry: AccountRegistry instance.
        rng: numpy RandomState.
        tx_counter: Current transaction counter.

    Returns:
        tuple: (mule_chain_records: list[dict],
                mule_transactions: list[dict],
                final_account_id: str,
                final_amount: float,
                final_timestamp: datetime,
                chain_depth: int,
                updated tx_counter,
                drifted_lat: float,
                drifted_lon: float)
    """
    financial_cfg = config["financial"]
    friction_min = financial_cfg["mule_hop_friction"]["min"]
    friction_max = financial_cfg["mule_hop_friction"]["max"]

    # V2: Geographic drift config
    atm_sel_cfg = config.get("atm_selection", {})
    drift_km = atm_sel_cfg.get("mule_chain_drift_km", 50)

    # Determine chain depth
    depth_min = profile_cfg["mule_chain_depth"]["min"]
    depth_max = profile_cfg["mule_chain_depth"]["max"]
    chain_depth = int(rng.randint(depth_min, depth_max + 1))

    hop_delay_min = profile_cfg["hop_delay_seconds"]["min"]
    hop_delay_max = profile_cfg["hop_delay_seconds"]["max"]

    mule_chain_records = []
    mule_transactions = []

    current_account = initial_dest_account_id
    current_amount = initial_amount
    current_time = initial_tx_datetime

    # V2: Geographic drift — start from victim location, drift with each hop
    current_lat = complaint["victim_lat"]
    current_lon = complaint["victim_lon"]

    complaint_id = complaint["complaint_id"]

    for hop in range(chain_depth):
        hop_number = hop + 1

        # Apply friction
        friction = rng.uniform(friction_min, friction_max)
        transfer_amount = round(current_amount * (1 - friction), 2)

        # Time progression
        delay_sec = int(rng.uniform(hop_delay_min, hop_delay_max))
        hop_time = current_time + timedelta(seconds=delay_sec)

        # V2: Get mule account from pool (with reuse) instead of always creating new
        next_account = account_registry.get_or_create_mule(
            hop_time.isoformat(), complaint_id=complaint_id
        )

        # V2: Geographic drift — shift reference location
        drift_angle = rng.uniform(0, 2 * np.pi)
        drift_dist = rng.exponential(drift_km / (chain_depth + 1))
        drift_dist = min(drift_dist, drift_km)
        dlat = (drift_dist * np.cos(drift_angle)) / 111.32
        dlon = (drift_dist * np.sin(drift_angle)) / (111.32 * cos(radians(current_lat)))
        current_lat = current_lat + dlat
        current_lon = current_lon + dlon
        # Clamp to India bounds
        current_lat = max(6.0, min(35.0, current_lat))
        current_lon = max(68.0, min(98.0, current_lon))

        # Create mule chain record
        chain_record = {
            "complaint_id": complaint_id,
            "source_account": current_account,
            "destination_account": next_account,
            "hop_number": hop_number,
            "transaction_amount": transfer_amount,
            "transaction_timestamp": hop_time.isoformat(),
        }
        mule_chain_records.append(chain_record)

        # Create corresponding transaction
        tx, tx_counter = generate_mule_transaction(
            source_account_id=current_account,
            dest_account_id=next_account,
            amount=transfer_amount,
            timestamp=hop_time,
            complaint_id=complaint_id,
            config=config,
            rng=rng,
            tx_counter=tx_counter,
        )
        mule_transactions.append(tx)

        current_account = next_account
        current_amount = transfer_amount
        current_time = hop_time

    return (mule_chain_records, mule_transactions, current_account,
            current_amount, current_time, chain_depth, tx_counter,
            current_lat, current_lon)
