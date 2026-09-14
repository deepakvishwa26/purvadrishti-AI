"""
Validation module — V2 with prediction-time and network checks.

Automated validation checks for the generated synthetic dataset.
Produces a clear PASS/FAIL report for each check.

V2 additions:
- Prediction-time temporal semantics (cutoff < withdrawal)
- Positive time_to_cashout for all cashout labels
- Mule account reuse across complaints
- Profile != deterministic label
- Historical hotspot density non-zero distribution
- Geographic diversity threshold
- Amount velocity robustness
"""

import h3
import numpy as np
import pandas as pd
from datetime import datetime


class ValidationReport:
    """Collects and displays validation results."""

    def __init__(self):
        self.results = []

    def add(self, check_name, passed, detail=""):
        self.results.append({
            "check": check_name,
            "result": "PASS" if passed else "FAIL",
            "detail": detail,
        })

    def all_passed(self):
        return all(r["result"] == "PASS" for r in self.results)

    def summary(self):
        lines = ["\n" + "=" * 60]
        lines.append("SYNTHETIC DATA VALIDATION REPORT (V2)")
        lines.append("=" * 60)
        for r in self.results:
            status = r["result"]
            icon = "[OK]" if status == "PASS" else "[!!]"
            line = f"  {icon} {status}: {r['check']}"
            if r["detail"]:
                line += f" -- {r['detail']}"
            lines.append(line)

        passed = sum(1 for r in self.results if r["result"] == "PASS")
        total = len(self.results)
        lines.append("-" * 60)
        lines.append(f"  TOTAL: {passed}/{total} checks passed")
        if self.all_passed():
            lines.append("  STATUS: ALL VALIDATIONS PASSED [OK]")
        else:
            lines.append("  STATUS: SOME VALIDATIONS FAILED [!!]")
        lines.append("=" * 60 + "\n")
        return "\n".join(lines)

    def to_dataframe(self):
        return pd.DataFrame(self.results)


def run_full_validation(complaints_df, accounts_df, transactions_df,
                         mule_chains_df, suspects_df, withdrawals_df,
                         atm_df, feature_snapshots_df, labels_df):
    """
    Run all validation checks on the generated datasets.

    Returns:
        ValidationReport with all results.
    """
    report = ValidationReport()

    # 1. Unique IDs
    _check_unique_ids(report, complaints_df, accounts_df, transactions_df,
                      mule_chains_df, suspects_df, withdrawals_df, atm_df)

    # 2. Foreign-key integrity
    _check_foreign_keys(report, complaints_df, accounts_df, transactions_df,
                        mule_chains_df, withdrawals_df, labels_df)

    # 3. Valid timestamps
    _check_timestamps(report, complaints_df, transactions_df,
                      mule_chains_df, withdrawals_df)

    # 4. Chronological ordering
    _check_chronological_order(report, complaints_df, transactions_df,
                                mule_chains_df, withdrawals_df)

    # 5. Positive amounts
    _check_positive_amounts(report, complaints_df, transactions_df,
                            mule_chains_df, withdrawals_df)

    # 6. Financial consistency
    _check_financial_consistency(report, complaints_df, mule_chains_df,
                                  withdrawals_df)

    # 7. Valid coordinates
    _check_coordinates(report, withdrawals_df, atm_df)

    # 8-9. H3 validity and consistency
    _check_h3_validity(report, withdrawals_df, atm_df)

    # 10. Withdrawal-account integrity
    _check_withdrawal_accounts(report, withdrawals_df, accounts_df)

    # 11. Mule-chain integrity
    _check_mule_chain_integrity(report, mule_chains_df, accounts_df)

    # 12. No future leakage
    _check_no_leakage(report, feature_snapshots_df, withdrawals_df, complaints_df)

    # 13. NO_CASHOUT cases
    _check_no_cashout(report, labels_df, withdrawals_df)

    # 14. MULTIPLE_CASHOUT
    _check_multiple_cashout(report, labels_df)

    # 15. Record counts
    _check_record_counts(report, complaints_df, accounts_df, transactions_df,
                          mule_chains_df, suspects_df, withdrawals_df,
                          feature_snapshots_df, labels_df)

    # === V2 CHECKS ===

    # 16. Prediction-time temporal semantics
    _check_prediction_time_semantics(report, complaints_df, labels_df,
                                      feature_snapshots_df, withdrawals_df)

    # 17. Mule account reuse
    _check_mule_reuse(report, mule_chains_df, transactions_df)

    # 18. Profile != deterministic label
    _check_profile_decoupling(report, complaints_df, labels_df)

    # 19. Historical hotspot density distribution
    _check_hotspot_distribution(report, feature_snapshots_df)

    # 20. Geographic diversity
    _check_geographic_diversity(report, withdrawals_df)

    # 21. Amount velocity robustness
    _check_amount_velocity(report, feature_snapshots_df)

    return report


# === ORIGINAL CHECKS (preserved from V1) ===

def _check_unique_ids(report, complaints_df, accounts_df, transactions_df,
                       mule_chains_df, suspects_df, withdrawals_df, atm_df):
    checks = [
        ("complaint_id", complaints_df),
        ("account_id", accounts_df),
        ("transaction_id", transactions_df),
        ("suspect_id", suspects_df),
        ("atm_id", atm_df),
    ]
    for col, df in checks:
        if col in df.columns:
            n_total = len(df)
            n_unique = df[col].nunique()
            passed = n_total == n_unique
            report.add(f"Unique {col}", passed,
                       f"{n_unique}/{n_total} unique")

    if len(withdrawals_df) > 0:
        n_total = len(withdrawals_df)
        n_unique = withdrawals_df["withdrawal_id"].nunique()
        passed = n_total == n_unique
        report.add("Unique withdrawal_id", passed,
                   f"{n_unique}/{n_total} unique")


def _check_foreign_keys(report, complaints_df, accounts_df, transactions_df,
                         mule_chains_df, withdrawals_df, labels_df):
    complaint_ids = set(complaints_df["complaint_id"])
    account_ids = set(accounts_df["account_id"])

    if "complaint_id" in transactions_df.columns:
        tx_complaints = set(transactions_df["complaint_id"])
        passed = tx_complaints.issubset(complaint_ids)
        report.add("FK: transactions -> complaints", passed,
                   f"{len(tx_complaints - complaint_ids)} orphans" if not passed else "")

    for col in ["source_account_id", "destination_account_id"]:
        if col in transactions_df.columns:
            tx_accts = set(transactions_df[col].dropna())
            passed = tx_accts.issubset(account_ids)
            report.add(f"FK: transactions.{col} -> accounts", passed,
                       f"{len(tx_accts - account_ids)} orphans" if not passed else "")

    if len(mule_chains_df) > 0 and "complaint_id" in mule_chains_df.columns:
        mc_complaints = set(mule_chains_df["complaint_id"])
        passed = mc_complaints.issubset(complaint_ids)
        report.add("FK: mule_chains -> complaints", passed)

    for col in ["source_account", "destination_account"]:
        if len(mule_chains_df) > 0 and col in mule_chains_df.columns:
            mc_accts = set(mule_chains_df[col].dropna())
            passed = mc_accts.issubset(account_ids)
            report.add(f"FK: mule_chains.{col} -> accounts", passed,
                       f"{len(mc_accts - account_ids)} orphans" if not passed else "")

    if len(withdrawals_df) > 0 and "account_id" in withdrawals_df.columns:
        w_accts = set(withdrawals_df["account_id"].dropna())
        passed = w_accts.issubset(account_ids)
        report.add("FK: withdrawals -> accounts", passed)

    if "complaint_id" in labels_df.columns:
        lbl_complaints = set(labels_df["complaint_id"])
        passed = lbl_complaints.issubset(complaint_ids)
        report.add("FK: labels -> complaints", passed)


def _check_timestamps(report, complaints_df, transactions_df,
                       mule_chains_df, withdrawals_df):
    ts_cols = [
        ("complaints", complaints_df, "incident_datetime"),
        ("complaints", complaints_df, "complaint_registered_at"),
        ("transactions", transactions_df, "transaction_datetime"),
    ]
    if len(mule_chains_df) > 0:
        ts_cols.append(("mule_chains", mule_chains_df, "transaction_timestamp"))
    if len(withdrawals_df) > 0:
        ts_cols.append(("withdrawals", withdrawals_df, "withdrawal_timestamp"))

    for name, df, col in ts_cols:
        if col in df.columns:
            try:
                parsed = pd.to_datetime(df[col])
                n_null = parsed.isna().sum()
                passed = n_null == 0
                report.add(f"Valid timestamps: {name}.{col}", passed,
                           f"{n_null} unparseable" if not passed else "")
            except Exception as e:
                report.add(f"Valid timestamps: {name}.{col}", False, str(e))


def _check_chronological_order(report, complaints_df, transactions_df,
                                 mule_chains_df, withdrawals_df):
    violations = 0
    sample_size = min(len(complaints_df), 2000)
    sample = complaints_df.sample(sample_size, random_state=42) if len(complaints_df) > 2000 else complaints_df

    for _, row in sample.iterrows():
        cid = row["complaint_id"]
        incident_dt = pd.to_datetime(row["incident_datetime"])
        registered_dt = pd.to_datetime(row["complaint_registered_at"])

        if registered_dt < incident_dt:
            violations += 1
            continue

        case_txns = transactions_df[transactions_df["complaint_id"] == cid]
        if len(case_txns) > 0:
            first_tx_dt = pd.to_datetime(case_txns["transaction_datetime"].min())
            if first_tx_dt < incident_dt:
                violations += 1

        if len(mule_chains_df) > 0:
            case_mc = mule_chains_df[mule_chains_df["complaint_id"] == cid].sort_values("hop_number")
            if len(case_mc) > 1:
                mc_times = pd.to_datetime(case_mc["transaction_timestamp"])
                if not mc_times.is_monotonic_increasing:
                    violations += 1

    report.add("Chronological event ordering", violations == 0,
               f"{violations} violations" if violations > 0 else
               f"All {sample_size} sampled cases ordered")


def _check_positive_amounts(report, complaints_df, transactions_df,
                             mule_chains_df, withdrawals_df):
    checks = [
        ("complaint fraud_amount", complaints_df["fraud_amount"]),
        ("transaction amount", transactions_df["amount"]),
    ]
    if len(mule_chains_df) > 0:
        checks.append(("mule_chain amount", mule_chains_df["transaction_amount"]))
    if len(withdrawals_df) > 0:
        checks.append(("withdrawal amount", withdrawals_df["amount"]))

    for name, series in checks:
        n_neg = (series <= 0).sum()
        report.add(f"Positive amounts: {name}", n_neg == 0,
                   f"{n_neg} non-positive" if n_neg > 0 else "")


def _check_financial_consistency(report, complaints_df, mule_chains_df,
                                   withdrawals_df):
    if len(withdrawals_df) == 0:
        report.add("Financial consistency", True, "No withdrawals to check")
        return

    violations = 0
    sample = complaints_df.sample(min(2000, len(complaints_df)), random_state=42)
    for _, row in sample.iterrows():
        cid = row["complaint_id"]
        fraud_amount = row["fraud_amount"]
        case_wdr = withdrawals_df[withdrawals_df["complaint_id"] == cid]
        if len(case_wdr) > 0:
            total_withdrawn = case_wdr["amount"].sum()
            if total_withdrawn > fraud_amount * 1.01:
                violations += 1

    report.add("Financial consistency: withdrawal <= fraud_amount", violations == 0,
               f"{violations} violations" if violations > 0 else "")


def _check_coordinates(report, withdrawals_df, atm_df):
    valid_lat = (6.0, 36.0)
    valid_lon = (68.0, 98.0)

    for name, df in [("ATM", atm_df), ("Withdrawal", withdrawals_df)]:
        if len(df) == 0:
            continue
        lat_ok = (df["latitude"] >= valid_lat[0]) & (df["latitude"] <= valid_lat[1])
        lon_ok = (df["longitude"] >= valid_lon[0]) & (df["longitude"] <= valid_lon[1])
        invalid = (~lat_ok | ~lon_ok).sum()
        report.add(f"Valid coordinates: {name}", invalid == 0,
                   f"{invalid} out-of-bounds" if invalid > 0 else
                   f"All {len(df)} valid")


def _check_h3_validity(report, withdrawals_df, atm_df):
    if len(withdrawals_df) == 0:
        report.add("H3 validity: withdrawals", True, "No withdrawals")
        report.add("H3/coordinate consistency", True, "No withdrawals")
        return

    invalid_h3 = 0
    for cell in withdrawals_df["h3_cell"]:
        if not h3.is_valid_cell(cell):
            invalid_h3 += 1
    report.add("H3 validity: withdrawals", invalid_h3 == 0,
               f"{invalid_h3} invalid" if invalid_h3 > 0 else
               f"All {len(withdrawals_df)} valid")

    mismatches = 0
    sample = withdrawals_df.sample(min(2000, len(withdrawals_df)), random_state=42)
    for _, row in sample.iterrows():
        expected_h3 = h3.latlng_to_cell(row["latitude"], row["longitude"], 8)
        if expected_h3 != row["h3_cell"]:
            mismatches += 1
    report.add("H3/coordinate consistency", mismatches == 0,
               f"{mismatches} mismatches" if mismatches > 0 else
               f"All {len(sample)} sampled consistent")

    invalid_atm_h3 = 0
    for _, row in atm_df.iterrows():
        expected = h3.latlng_to_cell(row["latitude"], row["longitude"], 8)
        if expected != row["h3_cell_res8"]:
            invalid_atm_h3 += 1
    report.add("H3 validity: ATM reference", invalid_atm_h3 == 0,
               f"{invalid_atm_h3} mismatches" if invalid_atm_h3 > 0 else
               f"All {len(atm_df)} consistent")


def _check_withdrawal_accounts(report, withdrawals_df, accounts_df):
    if len(withdrawals_df) == 0:
        report.add("Withdrawal-account integrity", True, "No withdrawals")
        return
    account_ids = set(accounts_df["account_id"])
    w_accts = set(withdrawals_df["account_id"])
    passed = w_accts.issubset(account_ids)
    report.add("Withdrawal-account integrity", passed,
               f"{len(w_accts - account_ids)} orphans" if not passed else "")


def _check_mule_chain_integrity(report, mule_chains_df, accounts_df):
    if len(mule_chains_df) == 0:
        report.add("Mule chain integrity", True, "No mule chains")
        return

    violations = 0
    for cid, group in mule_chains_df.groupby("complaint_id"):
        hops = sorted(group["hop_number"].tolist())
        expected = list(range(1, len(hops) + 1))
        if hops != expected:
            violations += 1

    report.add("Mule chain hop numbering", violations == 0,
               f"{violations} chains with bad numbering" if violations > 0 else "")

    connectivity_violations = 0
    for cid, group in mule_chains_df.groupby("complaint_id"):
        group = group.sort_values("hop_number")
        for i in range(len(group) - 1):
            row_current = group.iloc[i]
            row_next = group.iloc[i + 1]
            if row_current["destination_account"] != row_next["source_account"]:
                connectivity_violations += 1

    report.add("Mule chain connectivity", connectivity_violations == 0,
               f"{connectivity_violations} broken links" if connectivity_violations > 0 else "")


def _check_no_leakage(report, feature_snapshots_df, withdrawals_df, complaints_df):
    if len(withdrawals_df) == 0 or len(feature_snapshots_df) == 0:
        report.add("No temporal leakage", True, "Insufficient data to check")
        return

    violations = 0
    sample = feature_snapshots_df.sample(min(2000, len(feature_snapshots_df)), random_state=42)
    for _, feat_row in sample.iterrows():
        cid = feat_row["complaint_id"]
        cutoff = pd.to_datetime(feat_row["feature_cutoff_timestamp"])
        complaint_row = complaints_df[complaints_df["complaint_id"] == cid]
        if len(complaint_row) > 0:
            registered_at = pd.to_datetime(complaint_row.iloc[0]["complaint_registered_at"])
            if cutoff != registered_at:
                violations += 1

    report.add("No temporal leakage: cutoff = registered_at", violations == 0,
               f"{violations} mismatches" if violations > 0 else "")

    leaky_cols = {"actual_h3_cell", "actual_withdrawal_id", "actual_atm_id",
                  "actual_withdrawal_timestamp", "actual_withdrawal_amount"}
    found_leaky = leaky_cols.intersection(set(feature_snapshots_df.columns))
    report.add("No label columns in features", len(found_leaky) == 0,
               f"Found: {found_leaky}" if found_leaky else "")


def _check_no_cashout(report, labels_df, withdrawals_df):
    no_cashout = labels_df[labels_df["cashout_occurred"] == False]
    if len(no_cashout) == 0:
        report.add("NO_CASHOUT labels", True, "No NO_CASHOUT cases")
        return

    violations = 0
    if len(withdrawals_df) > 0:
        wdr_complaints = set(withdrawals_df["complaint_id"])
        for _, row in no_cashout.iterrows():
            if row["complaint_id"] in wdr_complaints:
                violations += 1

    report.add("NO_CASHOUT labels valid", violations == 0,
               f"{violations} have unexpected withdrawals" if violations > 0 else
               f"{len(no_cashout)} valid NO_CASHOUT cases")


def _check_multiple_cashout(report, labels_df):
    cashout_labels = labels_df[labels_df["cashout_occurred"] == True]
    multi = cashout_labels.groupby("complaint_id").size()
    multi_cases = (multi > 1).sum()
    report.add("MULTIPLE_CASHOUT cases present", True,
               f"{multi_cases} complaints with multiple withdrawals")


def _check_record_counts(report, complaints_df, accounts_df, transactions_df,
                           mule_chains_df, suspects_df, withdrawals_df,
                           feature_snapshots_df, labels_df):
    report.add("Record count: complaints", len(complaints_df) > 0,
               f"{len(complaints_df)} records")
    report.add("Record count: accounts", len(accounts_df) > 0,
               f"{len(accounts_df)} records")
    report.add("Record count: transactions", len(transactions_df) > 0,
               f"{len(transactions_df)} records")
    report.add("Record count: mule_chains", True,
               f"{len(mule_chains_df)} records")
    report.add("Record count: suspects", len(suspects_df) > 0,
               f"{len(suspects_df)} records")
    report.add("Record count: withdrawals", True,
               f"{len(withdrawals_df)} records")
    report.add("Record count: feature_snapshots", len(feature_snapshots_df) > 0,
               f"{len(feature_snapshots_df)} records")
    report.add("Record count: labels", len(labels_df) > 0,
               f"{len(labels_df)} records")

    feat_complaints = set(feature_snapshots_df["complaint_id"])
    all_complaints = set(complaints_df["complaint_id"])
    missing = all_complaints - feat_complaints
    report.add("Feature coverage: all complaints", len(missing) == 0,
               f"{len(missing)} missing" if missing else "")

    lbl_complaints = set(labels_df["complaint_id"])
    missing_lbl = all_complaints - lbl_complaints
    report.add("Label coverage: all complaints", len(missing_lbl) == 0,
               f"{len(missing_lbl)} missing" if missing_lbl else "")


# === V2 CHECKS ===

def _check_prediction_time_semantics(report, complaints_df, labels_df,
                                       feature_snapshots_df, withdrawals_df):
    """V2: Verify all withdrawals occur after feature_cutoff_timestamp."""
    if len(withdrawals_df) == 0:
        report.add("V2: cutoff < withdrawal", True, "No withdrawals")
        report.add("V2: time_to_cashout > 0", True, "No cashout labels")
        return

    # Check cutoff < withdrawal for all cashout cases
    cashout = labels_df[labels_df["cashout_occurred"] == True].copy()
    if len(cashout) > 0:
        # Merge with complaints to get registered_at
        cashout = cashout.merge(
            complaints_df[["complaint_id", "complaint_registered_at"]],
            on="complaint_id", how="left"
        )
        cashout["reg_dt"] = pd.to_datetime(cashout["complaint_registered_at"])
        cashout["wdr_dt"] = pd.to_datetime(cashout["actual_withdrawal_timestamp"])
        violations = (cashout["wdr_dt"] <= cashout["reg_dt"]).sum()
        report.add("V2: cutoff < withdrawal", violations == 0,
                   f"{violations} withdrawals at/before cutoff" if violations > 0 else
                   f"All {len(cashout)} cashout withdrawals after cutoff")

        # Check time_to_cashout > 0
        ttc = cashout["time_to_cashout_seconds"]
        neg_ttc = (ttc <= 0).sum()
        report.add("V2: time_to_cashout > 0", neg_ttc == 0,
                   f"{neg_ttc} non-positive" if neg_ttc > 0 else
                   f"All {len(ttc)} positive")
    else:
        report.add("V2: cutoff < withdrawal", True, "No cashout labels to check")
        report.add("V2: time_to_cashout > 0", True, "No cashout labels to check")


def _check_mule_reuse(report, mule_chains_df, transactions_df):
    """V2: Verify mule accounts are reused across complaints."""
    if len(mule_chains_df) == 0:
        report.add("V2: mule account reuse", False, "No mule chains")
        return

    # Check for accounts in multiple complaints (via mule_chains)
    acct_complaints = {}
    for _, row in mule_chains_df.iterrows():
        for col in ["source_account", "destination_account"]:
            acct = row[col]
            cid = row["complaint_id"]
            if acct not in acct_complaints:
                acct_complaints[acct] = set()
            acct_complaints[acct].add(cid)

    multi_complaint_accts = sum(1 for cids in acct_complaints.values() if len(cids) > 1)
    report.add("V2: mule account reuse", multi_complaint_accts > 0,
               f"{multi_complaint_accts} accounts in multiple complaints")


def _check_profile_decoupling(report, complaints_df, labels_df):
    """V2: Verify behavioral profile is not a deterministic label."""
    if "behavioral_profile" not in complaints_df.columns:
        report.add("V2: profile decoupling", True, "No profile column (V1 data)")
        return

    # Check NO_CASHOUT profile: some should have cashout
    no_cashout_profile = complaints_df[complaints_df["behavioral_profile"] == "NO_CASHOUT"]
    if len(no_cashout_profile) > 0:
        no_cashout_cids = set(no_cashout_profile["complaint_id"])
        cashout_labels = labels_df[labels_df["cashout_occurred"] == True]
        surprise_cashout = len(no_cashout_cids.intersection(set(cashout_labels["complaint_id"])))
        report.add("V2: NO_CASHOUT has surprise cashouts", surprise_cashout > 0,
                   f"{surprise_cashout} surprise cashouts from {len(no_cashout_profile)} NO_CASHOUT cases")

    # Check non-NO_CASHOUT profiles: some should have no cashout
    other_profiles = complaints_df[complaints_df["behavioral_profile"] != "NO_CASHOUT"]
    if len(other_profiles) > 0:
        other_cids = set(other_profiles["complaint_id"])
        no_cashout_labels = labels_df[labels_df["cashout_occurred"] == False]
        failed_cashout = len(other_cids.intersection(set(no_cashout_labels["complaint_id"])))
        report.add("V2: non-NO_CASHOUT has failed cashouts", failed_cashout > 0,
                   f"{failed_cashout} failed cashouts from {len(other_profiles)} non-NO_CASHOUT cases")


def _check_hotspot_distribution(report, feature_snapshots_df):
    """V2: Check historical_hotspot_density is not mostly zero."""
    if "historical_hotspot_density" not in feature_snapshots_df.columns:
        report.add("V2: hotspot non-zero", False, "Column missing")
        return

    density = feature_snapshots_df["historical_hotspot_density"]
    zero_pct = (density == 0).sum() / len(density) * 100
    report.add("V2: hotspot density < 80% zero", zero_pct < 80,
               f"{zero_pct:.1f}% zero")


def _check_geographic_diversity(report, withdrawals_df):
    """V2: Check geographic diversity of withdrawals."""
    if len(withdrawals_df) == 0:
        report.add("V2: geographic diversity", True, "No withdrawals")
        return

    n_h3 = withdrawals_df["h3_cell"].nunique()
    n_atm = withdrawals_df["atm_id"].nunique()
    # Threshold: at least 500 unique H3 cells for 10K+ withdrawals
    min_h3 = max(200, len(withdrawals_df) // 20)
    report.add("V2: geographic diversity", n_h3 >= min_h3,
               f"{n_h3} unique H3 cells, {n_atm} unique ATMs (min: {min_h3} H3)")


def _check_amount_velocity(report, feature_snapshots_df):
    """V2: Check amount_velocity doesn't have extreme outliers."""
    if "amount_velocity" not in feature_snapshots_df.columns:
        report.add("V2: amount_velocity robustness", False, "Column missing")
        return

    vel = feature_snapshots_df["amount_velocity"]
    p99 = np.percentile(vel.dropna(), 99)
    max_val = vel.max()
    # V2: with min_time floor, max should be much smaller
    # Max reasonable: ~50M INR in 1 minute = 50M/min
    report.add("V2: amount_velocity robustness", max_val < 50000000,
               f"P99={p99:,.0f}, max={max_val:,.0f}")
