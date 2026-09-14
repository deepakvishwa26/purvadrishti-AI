"""
Transaction generator module.

Generates the initial fraud transaction for each complaint case.
The initial transaction represents the first movement of stolen funds
from the victim's account.

All UTRs and identifiers are synthetic.
"""

import numpy as np
from datetime import datetime, timedelta


def generate_initial_transaction(complaint, account_registry, config, rng, tx_counter):
    """
    Generate the initial fraud transaction for a complaint.

    The initial transaction moves funds from victim to the first mule/destination
    account. Timing is slightly after the incident time.

    Args:
        complaint: dict with complaint fields.
        account_registry: AccountRegistry instance.
        config: Parsed configuration dict.
        rng: numpy RandomState.
        tx_counter: Current transaction counter (for unique IDs).

    Returns:
        tuple: (transaction_dict, destination_account_id, updated tx_counter)
    """
    tx_counter += 1
    transaction_id = f"TXN{str(tx_counter).zfill(10)}"

    # UTR: synthetic 12-digit reference
    utr_digits = "".join([str(rng.randint(0, 10)) for _ in range(12)])
    utr = f"SYNUTR{utr_digits}"

    # Transaction occurs shortly after the incident
    incident_dt = datetime.fromisoformat(complaint["incident_datetime"])
    tx_delay = timedelta(seconds=int(rng.uniform(10, 120)))  # 10s to 2min after
    tx_datetime = incident_dt + tx_delay

    # The full fraud amount moves in the initial transaction
    amount = complaint["fraud_amount"]

    # Source: victim account; Destination: first mule
    source_account_id = complaint["victim_account_id"]
    dest_account_id = account_registry.create_account(
        "MULE", tx_datetime.isoformat()
    )

    # Bank/wallet/merchant
    banks = config["banks"]
    bank = banks[rng.randint(0, len(banks))]

    tx = {
        "transaction_id": transaction_id,
        "complaint_id": complaint["complaint_id"],
        "utr": utr,
        "transaction_date": tx_datetime.strftime("%Y-%m-%d"),
        "transaction_time": tx_datetime.strftime("%H:%M:%S"),
        "transaction_datetime": tx_datetime.isoformat(),
        "bank_wallet_merchant": bank["name"],
        "amount": round(amount, 2),
        "source_account_id": source_account_id,
        "destination_account_id": dest_account_id,
    }

    return tx, dest_account_id, tx_counter


def generate_mule_transaction(source_account_id, dest_account_id, amount,
                               timestamp, complaint_id, config, rng, tx_counter):
    """
    Generate a mule-hop transaction.

    Args:
        source_account_id: Sending mule account.
        dest_account_id: Receiving mule account.
        amount: Transfer amount (after friction).
        timestamp: Transaction timestamp (datetime).
        complaint_id: Associated complaint.
        config: Configuration dict.
        rng: numpy RandomState.
        tx_counter: Current counter.

    Returns:
        tuple: (transaction_dict, updated tx_counter)
    """
    tx_counter += 1
    transaction_id = f"TXN{str(tx_counter).zfill(10)}"
    utr_digits = "".join([str(rng.randint(0, 10)) for _ in range(12)])
    utr = f"SYNUTR{utr_digits}"

    banks = config["banks"]
    bank = banks[rng.randint(0, len(banks))]

    tx = {
        "transaction_id": transaction_id,
        "complaint_id": complaint_id,
        "utr": utr,
        "transaction_date": timestamp.strftime("%Y-%m-%d"),
        "transaction_time": timestamp.strftime("%H:%M:%S"),
        "transaction_datetime": timestamp.isoformat(),
        "bank_wallet_merchant": bank["name"],
        "amount": round(amount, 2),
        "source_account_id": source_account_id,
        "destination_account_id": dest_account_id,
    }

    return tx, tx_counter
