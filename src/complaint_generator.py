"""
Complaint generator module.

Generates synthetic complaint records as the root entity of each fraud case.
Each complaint triggers the downstream event chain.

Fields include incident time (when fraud occurred), complaint registration time
(when victim reported it), fraud type, amount, and victim geography.

All PII and identifiers are synthetic.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_complaints(config, rng, account_registry):
    """
    Generate complaint records for all cases.

    Each complaint creates:
    - A victim account (via account_registry)
    - Incident timestamp
    - Registration timestamp (after incident)
    - Fraud type and amount (type-conditioned distribution)
    - Victim state and pincode

    Args:
        config: Parsed YAML configuration dict.
        rng: numpy RandomState.
        account_registry: AccountRegistry instance.

    Returns:
        list[dict]: List of complaint records (one per case).
    """
    num_cases = config["general"]["num_cases"]
    fraud_types_cfg = config["fraud_types"]
    fraud_amount_cfg = config["fraud_amount"]
    temporal_cfg = config["temporal"]
    geo_cfg = config["geography"]

    # --- Prepare distributions ---
    fraud_type_names = list(fraud_types_cfg.keys())
    fraud_type_probs = np.array([fraud_types_cfg[ft] for ft in fraud_type_names])
    fraud_type_probs = fraud_type_probs / fraud_type_probs.sum()

    # Date range
    start_date = datetime.strptime(temporal_cfg["start_date"], "%Y-%m-%d")
    end_date = datetime.strptime(temporal_cfg["end_date"], "%Y-%m-%d")
    date_range_days = (end_date - start_date).days

    # Hourly weights
    hourly_weights = temporal_cfg["hourly_weights"]
    hours = sorted(hourly_weights.keys())
    hour_probs = np.array([hourly_weights[h] for h in hours], dtype=float)
    hour_probs = hour_probs / hour_probs.sum()

    # State distribution
    state_names = list(geo_cfg["states"].keys())
    state_weights = np.array([geo_cfg["states"][s]["weight"] for s in state_names])
    state_weights = state_weights / state_weights.sum()

    complaints = []

    for i in range(num_cases):
        complaint_id = f"CMP{str(i + 1).zfill(8)}"

        # --- Fraud type ---
        fraud_type = rng.choice(fraud_type_names, p=fraud_type_probs)

        # --- Fraud amount (type-conditioned lognormal) ---
        amt_cfg = fraud_amount_cfg[fraud_type]
        raw_amount = rng.lognormal(mean=amt_cfg["mu"], sigma=amt_cfg["sigma"])
        fraud_amount = round(max(amt_cfg["min"], min(raw_amount, amt_cfg["max"])), 2)

        # --- Temporal: incident timestamp ---
        day_offset = rng.randint(0, date_range_days + 1)
        incident_date = start_date + timedelta(days=int(day_offset))

        hour = int(rng.choice(hours, p=hour_probs))
        minute = int(rng.randint(0, 60))
        second = int(rng.randint(0, 60))
        incident_datetime = incident_date.replace(hour=hour, minute=minute, second=second)

        # --- Registration timestamp (after incident) ---
        reg_delay_sec = int(rng.uniform(
            temporal_cfg["registration_delay"]["min"],
            temporal_cfg["registration_delay"]["max"]
        ))
        registered_at = incident_datetime + timedelta(seconds=reg_delay_sec)

        # --- Victim geography ---
        state_idx = rng.choice(len(state_names), p=state_weights)
        victim_state = state_names[state_idx].replace("_", " ")
        state_cfg = geo_cfg["states"][state_names[state_idx]]

        # Generate victim pincode
        prefix = rng.choice(state_cfg["pincode_prefix"])
        suffix = str(rng.randint(1000, 9999))
        victim_pincode = prefix + suffix

        # Generate victim approximate coordinates (for distance calculations)
        victim_lat = rng.uniform(state_cfg["lat_range"][0], state_cfg["lat_range"][1])
        victim_lon = rng.uniform(state_cfg["lon_range"][0], state_cfg["lon_range"][1])

        # --- Incident details (synthetic text) ---
        incident_details = _generate_incident_text(fraud_type, fraud_amount, rng)

        # --- Create victim account ---
        victim_account_id = account_registry.create_account(
            "VICTIM", incident_datetime.isoformat()
        )

        complaints.append({
            "complaint_id": complaint_id,
            "incident_date": incident_datetime.strftime("%Y-%m-%d"),
            "incident_time": incident_datetime.strftime("%H:%M:%S"),
            "incident_datetime": incident_datetime.isoformat(),
            "complaint_registered_at": registered_at.isoformat(),
            "incident_details": incident_details,
            "fraud_type": fraud_type,
            "fraud_amount": fraud_amount,
            "victim_state": victim_state,
            "victim_pincode": victim_pincode,
            "victim_lat": round(victim_lat, 6),
            "victim_lon": round(victim_lon, 6),
            "victim_account_id": victim_account_id,
            "source_system": "SYNTHETIC",
            "data_version": config["general"]["schema_version"],
        })

    return complaints


def _generate_incident_text(fraud_type, amount, rng):
    """Generate plausible synthetic incident description text."""
    templates = {
        "DIGITAL_PAYMENT_FRAUD": [
            f"Victim reported unauthorized UPI transaction of INR {amount:.0f}.",
            f"Fraudulent digital payment of INR {amount:.0f} detected from victim account.",
            f"Victim received fake payment link; INR {amount:.0f} debited without authorization.",
        ],
        "INVESTMENT_FRAUD": [
            f"Victim lured into fake investment scheme; lost INR {amount:.0f}.",
            f"Fraudulent trading platform collected INR {amount:.0f} from victim.",
            f"Victim transferred INR {amount:.0f} to fake mutual fund agent.",
        ],
        "DIGITAL_ARREST": [
            f"Victim threatened with digital arrest; paid INR {amount:.0f} under duress.",
            f"Impersonation of law enforcement; victim coerced into transferring INR {amount:.0f}.",
            f"Fake police/CBI call led victim to transfer INR {amount:.0f}.",
        ],
        "PHISHING": [
            f"Victim's banking credentials phished; INR {amount:.0f} stolen.",
            f"Phishing email led to unauthorized transfer of INR {amount:.0f}.",
            f"Victim shared OTP after phishing call; INR {amount:.0f} debited.",
        ],
        "OTHER": [
            f"Cyber fraud reported; INR {amount:.0f} lost through unspecified method.",
            f"Victim reported loss of INR {amount:.0f} via online fraud.",
            f"Online fraud complaint filed for INR {amount:.0f}.",
        ],
    }
    options = templates.get(fraud_type, templates["OTHER"])
    return rng.choice(options)
