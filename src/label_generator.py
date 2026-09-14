"""
Label generator module — V2 with prediction-time temporal validation.

Creates the cashout_labels table, which contains ground-truth outcome
information for each complaint case.

The label represents the FUTURE withdrawal event (after the feature cutoff).
It must NEVER be included as an input feature during inference.

V2 changes:
- time_to_cashout_seconds is computed as withdrawal_timestamp - feature_cutoff_timestamp.
- Assertion: all cashout labels have time_to_cashout_seconds > 0.
- For NO_CASHOUT cases, all withdrawal fields are NULL/None.
- For MULTIPLE_CASHOUT cases, one label per withdrawal is created.
"""

from datetime import datetime


def generate_labels(complaint, withdrawals):
    """
    Generate ground-truth label records for a complaint case.

    V2: Validates that all withdrawals occur after feature cutoff.

    Args:
        complaint: Complaint dict.
        withdrawals: List of withdrawal dicts for this case (may be empty).

    Returns:
        list[dict]: Label records. One per withdrawal, or one NULL record
                    for NO_CASHOUT.
    """
    complaint_id = complaint["complaint_id"]
    registered_at = datetime.fromisoformat(complaint["complaint_registered_at"])

    if not withdrawals:
        # NO_CASHOUT case
        return [{
            "complaint_id": complaint_id,
            "actual_withdrawal_id": None,
            "actual_h3_cell": None,
            "actual_atm_id": None,
            "actual_withdrawal_timestamp": None,
            "actual_withdrawal_amount": None,
            "time_to_cashout_seconds": None,
            "cashout_occurred": False,
        }]

    labels = []
    for w in withdrawals:
        w_timestamp = datetime.fromisoformat(w["withdrawal_timestamp"])
        time_to_cashout = round((w_timestamp - registered_at).total_seconds(), 0)

        # V2: Assert positive time_to_cashout
        assert time_to_cashout > 0, (
            f"TEMPORAL VIOLATION: withdrawal {w['withdrawal_id']} at {w['withdrawal_timestamp']} "
            f"occurs before/at complaint registration {complaint['complaint_registered_at']} "
            f"(time_to_cashout={time_to_cashout}s). "
            f"All withdrawals must occur AFTER feature_cutoff_timestamp."
        )

        labels.append({
            "complaint_id": complaint_id,
            "actual_withdrawal_id": w["withdrawal_id"],
            "actual_h3_cell": w["h3_cell"],
            "actual_atm_id": w["atm_id"],
            "actual_withdrawal_timestamp": w["withdrawal_timestamp"],
            "actual_withdrawal_amount": w["amount"],
            "time_to_cashout_seconds": time_to_cashout,
            "cashout_occurred": True,
        })

    return labels
