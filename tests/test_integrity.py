"""
HIVE-Predict Synthetic Data Integrity Tests.

Tests cover:
1. Schema validation
2. Foreign key integrity
3. Temporal consistency
4. Financial consistency
5. H3 correctness
6. Feature derivation
7. Leakage prevention
8. No-cashout labels
9. Reproducibility
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
import h3
import yaml
from datetime import datetime


# ---- Fixtures ----

@pytest.fixture(scope="session")
def config():
    """Load generation config."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "generation_config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def datasets():
    """Load generated datasets from output directory."""
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "output")
    dfs = {}
    for name in ["complaints", "accounts", "transactions", "mule_chains",
                  "suspects", "withdrawals", "atm_reference",
                  "feature_snapshots", "cashout_labels"]:
        csv_path = os.path.join(output_dir, f"{name}.csv")
        if os.path.exists(csv_path):
            dfs[name] = pd.read_csv(csv_path)
        else:
            dfs[name] = pd.DataFrame()
    return dfs


# ---- 1. Schema Tests ----

class TestSchema:
    def test_complaints_schema(self, datasets):
        df = datasets["complaints"]
        required = {"complaint_id", "incident_date", "incident_time",
                     "incident_datetime", "complaint_registered_at",
                     "fraud_type", "fraud_amount", "victim_state", "victim_pincode"}
        assert required.issubset(set(df.columns)), \
            f"Missing columns: {required - set(df.columns)}"

    def test_accounts_schema(self, datasets):
        df = datasets["accounts"]
        required = {"account_id", "bank_id", "account_type",
                     "account_status", "first_seen_timestamp"}
        assert required.issubset(set(df.columns))

    def test_transactions_schema(self, datasets):
        df = datasets["transactions"]
        required = {"transaction_id", "utr", "transaction_date",
                     "transaction_time", "amount", "source_account_id",
                     "destination_account_id", "complaint_id"}
        assert required.issubset(set(df.columns))

    def test_mule_chains_schema(self, datasets):
        df = datasets["mule_chains"]
        if len(df) == 0:
            pytest.skip("No mule chain records")
        required = {"chain_id", "complaint_id", "source_account",
                     "destination_account", "hop_number",
                     "transaction_amount", "transaction_timestamp"}
        assert required.issubset(set(df.columns))

    def test_suspects_schema(self, datasets):
        df = datasets["suspects"]
        required = {"suspect_id", "complaint_id", "suspect_mobile",
                     "suspect_email", "suspect_bank_account"}
        assert required.issubset(set(df.columns))

    def test_withdrawals_schema(self, datasets):
        df = datasets["withdrawals"]
        if len(df) == 0:
            pytest.skip("No withdrawal records")
        required = {"withdrawal_id", "account_id", "withdrawal_timestamp",
                     "amount", "atm_id", "latitude", "longitude", "h3_cell"}
        assert required.issubset(set(df.columns))

    def test_feature_snapshots_schema(self, datasets):
        df = datasets["feature_snapshots"]
        required = {"complaint_id", "feature_cutoff_timestamp",
                     "fraud_amount", "hour", "day_of_week", "fraud_type",
                     "mule_chain_depth", "mule_velocity", "amount_velocity",
                     "distance_from_victim", "historical_hotspot_density",
                     "atm_density", "complaint_cluster"}
        assert required.issubset(set(df.columns))

    def test_labels_schema(self, datasets):
        df = datasets["cashout_labels"]
        required = {"complaint_id", "actual_withdrawal_id",
                     "actual_h3_cell", "actual_atm_id",
                     "cashout_occurred"}
        assert required.issubset(set(df.columns))


# ---- 2. Foreign Key Tests ----

class TestForeignKeys:
    def test_transactions_reference_valid_complaints(self, datasets):
        complaint_ids = set(datasets["complaints"]["complaint_id"])
        tx_complaints = set(datasets["transactions"]["complaint_id"])
        assert tx_complaints.issubset(complaint_ids)

    def test_transactions_reference_valid_accounts(self, datasets):
        account_ids = set(datasets["accounts"]["account_id"])
        for col in ["source_account_id", "destination_account_id"]:
            tx_accts = set(datasets["transactions"][col].dropna())
            assert tx_accts.issubset(account_ids), \
                f"Orphan accounts in transactions.{col}"

    def test_mule_chains_reference_valid_accounts(self, datasets):
        if len(datasets["mule_chains"]) == 0:
            pytest.skip("No mule chains")
        account_ids = set(datasets["accounts"]["account_id"])
        for col in ["source_account", "destination_account"]:
            mc_accts = set(datasets["mule_chains"][col].dropna())
            assert mc_accts.issubset(account_ids)

    def test_labels_reference_valid_complaints(self, datasets):
        complaint_ids = set(datasets["complaints"]["complaint_id"])
        label_complaints = set(datasets["cashout_labels"]["complaint_id"])
        assert label_complaints.issubset(complaint_ids)

    def test_all_complaints_have_labels(self, datasets):
        complaint_ids = set(datasets["complaints"]["complaint_id"])
        label_complaints = set(datasets["cashout_labels"]["complaint_id"])
        missing = complaint_ids - label_complaints
        assert len(missing) == 0, f"{len(missing)} complaints missing labels"


# ---- 3. Temporal Consistency Tests ----

class TestTemporalConsistency:
    def test_registration_after_incident(self, datasets):
        df = datasets["complaints"]
        incident_dt = pd.to_datetime(df["incident_datetime"])
        registered_dt = pd.to_datetime(df["complaint_registered_at"])
        violations = (registered_dt < incident_dt).sum()
        assert violations == 0, f"{violations} complaints registered before incident"

    def test_first_transaction_after_incident(self, datasets):
        complaints = datasets["complaints"]
        transactions = datasets["transactions"]
        violations = 0
        # Sample 100 cases for performance
        sample = complaints.sample(min(100, len(complaints)), random_state=42)
        for _, row in sample.iterrows():
            cid = row["complaint_id"]
            incident_dt = pd.to_datetime(row["incident_datetime"])
            case_txns = transactions[transactions["complaint_id"] == cid]
            if len(case_txns) > 0:
                first_tx = pd.to_datetime(case_txns["transaction_datetime"].min())
                if first_tx < incident_dt:
                    violations += 1
        assert violations == 0

    def test_mule_chain_monotonic(self, datasets):
        mc = datasets["mule_chains"]
        if len(mc) == 0:
            pytest.skip("No mule chains")
        violations = 0
        for cid, group in mc.groupby("complaint_id"):
            group = group.sort_values("hop_number")
            times = pd.to_datetime(group["transaction_timestamp"])
            if not times.is_monotonic_increasing:
                violations += 1
        assert violations == 0

    def test_withdrawal_after_transactions(self, datasets):
        withdrawals = datasets["withdrawals"]
        transactions = datasets["transactions"]
        if len(withdrawals) == 0:
            pytest.skip("No withdrawals")
        violations = 0
        sample = withdrawals.sample(min(100, len(withdrawals)), random_state=42)
        for _, wdr in sample.iterrows():
            cid = wdr["complaint_id"]
            wdr_dt = pd.to_datetime(wdr["withdrawal_timestamp"])
            case_txns = transactions[transactions["complaint_id"] == cid]
            if len(case_txns) > 0:
                last_tx = pd.to_datetime(case_txns["transaction_datetime"].max())
                if wdr_dt < last_tx:
                    violations += 1
        assert violations == 0


# ---- 4. Financial Consistency Tests ----

class TestFinancialConsistency:
    def test_all_amounts_positive(self, datasets):
        for name in ["complaints", "transactions", "mule_chains", "withdrawals"]:
            df = datasets[name]
            if len(df) == 0:
                continue
            col = "fraud_amount" if name == "complaints" else \
                  "transaction_amount" if name == "mule_chains" else "amount"
            assert (df[col] > 0).all(), f"Non-positive amount in {name}"

    def test_withdrawal_not_exceed_fraud(self, datasets):
        complaints = datasets["complaints"]
        withdrawals = datasets["withdrawals"]
        if len(withdrawals) == 0:
            pytest.skip("No withdrawals")
        violations = 0
        for _, row in complaints.iterrows():
            cid = row["complaint_id"]
            fraud = row["fraud_amount"]
            case_wdr = withdrawals[withdrawals["complaint_id"] == cid]
            if len(case_wdr) > 0:
                total_withdrawn = case_wdr["amount"].sum()
                if total_withdrawn > fraud * 1.01:
                    violations += 1
        assert violations == 0, f"{violations} cases: withdrawal > fraud amount"


# ---- 5. H3 Correctness Tests ----

class TestH3Correctness:
    def test_h3_cells_valid(self, datasets):
        withdrawals = datasets["withdrawals"]
        if len(withdrawals) == 0:
            pytest.skip("No withdrawals")
        invalid = sum(1 for cell in withdrawals["h3_cell"]
                      if not h3.is_valid_cell(str(cell)))
        assert invalid == 0, f"{invalid} invalid H3 cells"

    def test_h3_matches_coordinates(self, datasets):
        withdrawals = datasets["withdrawals"]
        if len(withdrawals) == 0:
            pytest.skip("No withdrawals")
        mismatches = 0
        sample = withdrawals.sample(min(200, len(withdrawals)), random_state=42)
        for _, row in sample.iterrows():
            expected = h3.latlng_to_cell(row["latitude"], row["longitude"], 8)
            if expected != str(row["h3_cell"]):
                mismatches += 1
        assert mismatches == 0, f"{mismatches} H3/coordinate mismatches"

    def test_atm_h3_consistency(self, datasets):
        atm = datasets["atm_reference"]
        mismatches = 0
        sample = atm.sample(min(200, len(atm)), random_state=42)
        for _, row in sample.iterrows():
            expected = h3.latlng_to_cell(row["latitude"], row["longitude"], 8)
            if expected != str(row["h3_cell_res8"]):
                mismatches += 1
        assert mismatches == 0


# ---- 6. Feature Derivation Tests ----

class TestFeatureDerivation:
    def test_hour_derived_from_incident(self, datasets):
        features = datasets["feature_snapshots"]
        complaints = datasets["complaints"]
        merged = features.merge(complaints[["complaint_id", "incident_datetime"]],
                                on="complaint_id")
        sample = merged.sample(min(200, len(merged)), random_state=42)
        mismatches = 0
        for _, row in sample.iterrows():
            expected_hour = pd.to_datetime(row["incident_datetime"]).hour
            if int(row["hour"]) != expected_hour:
                mismatches += 1
        assert mismatches == 0

    def test_day_of_week_derived(self, datasets):
        features = datasets["feature_snapshots"]
        complaints = datasets["complaints"]
        merged = features.merge(complaints[["complaint_id", "incident_datetime"]],
                                on="complaint_id")
        sample = merged.sample(min(200, len(merged)), random_state=42)
        mismatches = 0
        for _, row in sample.iterrows():
            expected_dow = pd.to_datetime(row["incident_datetime"]).weekday()
            if int(row["day_of_week"]) != expected_dow:
                mismatches += 1
        assert mismatches == 0

    def test_fraud_amount_matches_complaint(self, datasets):
        features = datasets["feature_snapshots"]
        complaints = datasets["complaints"]
        merged = features.merge(complaints[["complaint_id", "fraud_amount"]],
                                on="complaint_id", suffixes=("_feat", "_cmp"))
        diff = (merged["fraud_amount_feat"] - merged["fraud_amount_cmp"]).abs()
        assert (diff < 0.01).all(), "Feature fraud_amount doesn't match complaint"

    def test_mule_chain_depth_derived(self, datasets):
        """V2: depth is observable at cutoff, which is <= total chain depth."""
        features = datasets["feature_snapshots"]
        mule_chains = datasets["mule_chains"]
        complaints = datasets["complaints"]
        if len(mule_chains) == 0:
            pytest.skip("No mule chains")
        sample = features.sample(min(100, len(features)), random_state=42)
        for _, row in sample.iterrows():
            cid = row["complaint_id"]
            case_mc = mule_chains[mule_chains["complaint_id"] == cid]
            total_depth = len(case_mc)
            observed_depth = int(row["mule_chain_depth"])
            # V2: observed_depth <= total_depth (some hops may be after cutoff)
            assert observed_depth <= total_depth, \
                f"Observed depth {observed_depth} > total {total_depth} for {cid}"
            assert observed_depth >= 0, \
                f"Negative depth for {cid}"


# ---- 7. Leakage Prevention Tests ----

class TestLeakagePrevention:
    def test_feature_cutoff_equals_registration(self, datasets):
        features = datasets["feature_snapshots"]
        complaints = datasets["complaints"]
        merged = features.merge(
            complaints[["complaint_id", "complaint_registered_at"]],
            on="complaint_id"
        )
        cutoff = pd.to_datetime(merged["feature_cutoff_timestamp"])
        registered = pd.to_datetime(merged["complaint_registered_at"])
        mismatches = (cutoff != registered).sum()
        assert mismatches == 0, \
            f"{mismatches} cases: cutoff ≠ registration time"

    def test_no_future_columns_in_features(self, datasets):
        features = datasets["feature_snapshots"]
        leaky_cols = {"actual_h3_cell", "actual_withdrawal_id",
                      "actual_atm_id", "actual_withdrawal_timestamp",
                      "actual_withdrawal_amount", "time_to_cashout_seconds"}
        found = leaky_cols.intersection(set(features.columns))
        assert len(found) == 0, f"Leaky columns in features: {found}"


# ---- 8. No-Cashout Label Tests ----

class TestNoCashout:
    def test_no_cashout_has_null_withdrawal(self, datasets):
        labels = datasets["cashout_labels"]
        no_cashout = labels[labels["cashout_occurred"] == False]
        if len(no_cashout) == 0:
            pytest.skip("No NO_CASHOUT cases")

        # All withdrawal fields should be null
        for col in ["actual_withdrawal_id", "actual_h3_cell", "actual_atm_id"]:
            if col in no_cashout.columns:
                assert no_cashout[col].isna().all(), \
                    f"Non-null {col} in NO_CASHOUT labels"

    def test_no_cashout_no_withdrawals(self, datasets):
        labels = datasets["cashout_labels"]
        withdrawals = datasets["withdrawals"]
        no_cashout = labels[labels["cashout_occurred"] == False]
        if len(no_cashout) == 0 or len(withdrawals) == 0:
            pytest.skip("Insufficient data")

        no_cashout_cids = set(no_cashout["complaint_id"])
        wdr_cids = set(withdrawals["complaint_id"])
        overlap = no_cashout_cids.intersection(wdr_cids)
        assert len(overlap) == 0, \
            f"{len(overlap)} NO_CASHOUT cases have withdrawals"


# ---- 9. Reproducibility Test ----

class TestReproducibility:
    def test_seed_produces_same_complaints(self, config):
        """Verify that running with same seed produces identical complaint IDs."""
        from src.account_generator import AccountRegistry
        from src.complaint_generator import generate_complaints

        # Use small case count for speed
        test_config = dict(config)
        test_config["general"] = dict(config["general"])
        test_config["general"]["num_cases"] = 50

        rng1 = np.random.RandomState(config["general"]["seed"])
        reg1 = AccountRegistry(test_config, np.random.RandomState(config["general"]["seed"]))
        complaints1 = generate_complaints(test_config, rng1, reg1)

        rng2 = np.random.RandomState(config["general"]["seed"])
        reg2 = AccountRegistry(test_config, np.random.RandomState(config["general"]["seed"]))
        complaints2 = generate_complaints(test_config, rng2, reg2)

        ids1 = [c["complaint_id"] for c in complaints1]
        ids2 = [c["complaint_id"] for c in complaints2]
        assert ids1 == ids2, "Complaint IDs differ between runs"

        amounts1 = [c["fraud_amount"] for c in complaints1]
        amounts2 = [c["fraud_amount"] for c in complaints2]
        assert amounts1 == amounts2, "Fraud amounts differ between runs"


# ---- V2: 10. Prediction-Time Semantics Tests ----

class TestPredictionTimeSemantics:
    """V2: Every cashout withdrawal must occur AFTER feature_cutoff_timestamp."""

    def test_all_withdrawals_after_cutoff(self, datasets):
        """Core temporal contract: cutoff < withdrawal for all cashout labels."""
        labels = datasets["cashout_labels"]
        complaints = datasets["complaints"]
        cashout = labels[labels["cashout_occurred"] == True].copy()
        if len(cashout) == 0:
            pytest.skip("No cashout labels")

        cashout = cashout.merge(
            complaints[["complaint_id", "complaint_registered_at"]],
            on="complaint_id", how="left"
        )
        cashout["reg_dt"] = pd.to_datetime(cashout["complaint_registered_at"])
        cashout["wdr_dt"] = pd.to_datetime(cashout["actual_withdrawal_timestamp"])
        violations = (cashout["wdr_dt"] <= cashout["reg_dt"]).sum()
        assert violations == 0, \
            f"{violations} withdrawals at/before feature cutoff"

    def test_time_to_cashout_always_positive(self, datasets):
        """time_to_cashout_seconds must be > 0 for all cashout labels."""
        labels = datasets["cashout_labels"]
        cashout = labels[labels["cashout_occurred"] == True]
        if len(cashout) == 0:
            pytest.skip("No cashout labels")
        ttc = cashout["time_to_cashout_seconds"]
        neg_or_zero = (ttc <= 0).sum()
        assert neg_or_zero == 0, \
            f"{neg_or_zero} non-positive time_to_cashout values"

    def test_no_cashout_has_null_ttc(self, datasets):
        """NO_CASHOUT labels must have null time_to_cashout_seconds."""
        labels = datasets["cashout_labels"]
        no_cashout = labels[labels["cashout_occurred"] == False]
        if len(no_cashout) == 0:
            pytest.skip("No NO_CASHOUT cases")
        assert no_cashout["time_to_cashout_seconds"].isna().all()


# ---- V2: 11. Mule Network Reuse Tests ----

class TestMuleNetworkReuse:
    """V2: Mule accounts must be reused across multiple complaints."""

    def test_some_accounts_in_multiple_complaints(self, datasets):
        mc = datasets["mule_chains"]
        if len(mc) == 0:
            pytest.skip("No mule chains")

        acct_complaints = {}
        for _, row in mc.iterrows():
            for col in ["source_account", "destination_account"]:
                acct = row[col]
                cid = row["complaint_id"]
                if acct not in acct_complaints:
                    acct_complaints[acct] = set()
                acct_complaints[acct].add(cid)

        multi = sum(1 for cids in acct_complaints.values() if len(cids) > 1)
        assert multi > 0, "No mule account reuse across complaints"

    def test_reuse_not_total(self, datasets):
        """Not every account should be reused — mix of single-use and multi-use."""
        mc = datasets["mule_chains"]
        if len(mc) == 0:
            pytest.skip("No mule chains")

        acct_complaints = {}
        for _, row in mc.iterrows():
            for col in ["source_account", "destination_account"]:
                acct = row[col]
                cid = row["complaint_id"]
                if acct not in acct_complaints:
                    acct_complaints[acct] = set()
                acct_complaints[acct].add(cid)

        single = sum(1 for cids in acct_complaints.values() if len(cids) == 1)
        multi = sum(1 for cids in acct_complaints.values() if len(cids) > 1)
        assert single > 0 and multi > 0, \
            f"Expected mix: single={single}, multi={multi}"


# ---- V2: 12. Profile Decoupling Tests ----

class TestProfileDecoupling:
    """V2: Behavioral profile must not deterministically control outcome."""

    def test_no_cashout_has_surprise_cashouts(self, datasets):
        """Some NO_CASHOUT profile cases should actually have cashout."""
        complaints = datasets["complaints"]
        labels = datasets["cashout_labels"]
        if "behavioral_profile" not in complaints.columns:
            pytest.skip("No behavioral_profile column (V1 data)")

        no_cashout_cases = complaints[complaints["behavioral_profile"] == "NO_CASHOUT"]
        if len(no_cashout_cases) == 0:
            pytest.skip("No NO_CASHOUT profile cases")

        no_cashout_cids = set(no_cashout_cases["complaint_id"])
        cashout_labels = labels[labels["cashout_occurred"] == True]
        surprises = len(no_cashout_cids.intersection(set(cashout_labels["complaint_id"])))
        assert surprises > 0, "NO_CASHOUT profile is 100% deterministic"

    def test_other_profiles_have_failures(self, datasets):
        """Some non-NO_CASHOUT profiles should fail to cash out."""
        complaints = datasets["complaints"]
        labels = datasets["cashout_labels"]
        if "behavioral_profile" not in complaints.columns:
            pytest.skip("No behavioral_profile column (V1 data)")

        other = complaints[complaints["behavioral_profile"] != "NO_CASHOUT"]
        if len(other) == 0:
            pytest.skip("No non-NO_CASHOUT cases")

        other_cids = set(other["complaint_id"])
        no_cashout_labels = labels[labels["cashout_occurred"] == False]
        failures = len(other_cids.intersection(set(no_cashout_labels["complaint_id"])))
        assert failures > 0, "Non-NO_CASHOUT profiles always succeed"


# ---- V2: 13. Hotspot Distribution Test ----

class TestHotspotDistribution:
    """V2: historical_hotspot_density should not be mostly zero."""

    def test_hotspot_has_nonzero_values(self, datasets):
        features = datasets["feature_snapshots"]
        if "historical_hotspot_density" not in features.columns:
            pytest.skip("No hotspot column")
        density = features["historical_hotspot_density"]
        nonzero_pct = (density > 0).sum() / len(density) * 100
        assert nonzero_pct > 5, \
            f"Only {nonzero_pct:.1f}% non-zero (expected >5%)"
