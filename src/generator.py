"""
HIVE-Predict Synthetic Data Generator — Main Orchestrator V2

V2 changes:
- Initializes mule account pool before case generation (CRITICAL #2).
- Enforces prediction-time temporal semantics (CRITICAL #1):
  incident < registration = cutoff < withdrawal.
- Uses rolling-window clustering (CRITICAL #3).
- Generates historical seed events for hotspot density (MEDIUM #6).
- Decouples behavioral profile from cashout outcome (HIGH #5).
- Passes drifted coordinates from mule chain to withdrawal (MEDIUM #7).
- Stores behavioral_profile in complaint (for audit, NOT in features).

Usage:
    python -m src.generator [--config config/generation_config.yaml]
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import pandas as pd
import yaml
from collections import Counter
from datetime import datetime, timedelta

from src.geography import generate_atm_reference, lat_lon_to_h3
from src.account_generator import AccountRegistry
from src.complaint_generator import generate_complaints
from src.transaction_generator import generate_initial_transaction
from src.mule_chain_generator import generate_mule_chain
from src.suspect_generator import create_suspect_generator, generate_suspect
from src.withdrawal_generator import generate_withdrawals
from src.feature_engineering import (compute_features_for_case,
                                      compute_complaint_clusters_rolling)
from src.label_generator import generate_labels
from src.validation import run_full_validation


def load_config(config_path):
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def select_behavioral_profile(config, rng):
    """Select a behavioral profile based on configured probabilities."""
    profiles = config["behavioral_profiles"]
    names = list(profiles.keys())
    probs = np.array([profiles[n]["probability"] for n in names])
    probs = probs / probs.sum()
    return rng.choice(names, p=probs)


def generate_historical_seed_events(config, atm_df, rng):
    """
    V2.1: Generate synthetic historical withdrawal events BEFORE the dataset window.
    These seed the historical_hotspot_density feature so it is not always zero.

    Uses VICTIM-DISTRIBUTION sampling: state-weighted, uniform-within-state lat/lon,
    converted to H3 cells. This matches where future victim complaints will originate,
    guaranteeing H3 neighborhood overlap and non-zero hotspot density for most cases.

    Args:
        config: Configuration dict.
        atm_df: ATM reference DataFrame (unused directly, kept for API compatibility).
        rng: numpy RandomState.

    Returns:
        list[dict]: Historical withdrawal event dicts with h3_cell and timestamp.
    """
    import h3 as h3lib

    seed_cfg = config.get("historical_seed", {})
    num_events = seed_cfg.get("num_seed_withdrawals", 50000)
    window_days = seed_cfg.get("seed_window_days", 180)
    h3_res = config["general"]["h3_resolution_primary"]

    start_date = datetime.strptime(config["temporal"]["start_date"], "%Y-%m-%d")
    seed_end = start_date - timedelta(days=1)
    seed_start = start_date - timedelta(days=window_days)
    total_seconds = (seed_end - seed_start).total_seconds()

    # State-weighted sampling — same distribution as complaint generator
    geo_cfg = config["geography"]
    states_cfg = geo_cfg["states"]
    state_names = list(states_cfg.keys())
    state_weights = np.array([states_cfg[s]["weight"] for s in state_names])
    state_weights = state_weights / state_weights.sum()

    # Vectorised batch: sample states, then random coords within each state
    state_indices = rng.choice(len(state_names), size=num_events, p=state_weights)
    offsets_sec = rng.uniform(0, total_seconds, size=num_events).astype(int)

    events = []
    for i in range(num_events):
        sname = state_names[state_indices[i]]
        scfg = states_cfg[sname]
        lat = rng.uniform(scfg["lat_range"][0], scfg["lat_range"][1])
        lon = rng.uniform(scfg["lon_range"][0], scfg["lon_range"][1])
        cell = h3lib.latlng_to_cell(lat, lon, h3_res)
        ts = seed_start + timedelta(seconds=int(offsets_sec[i]))
        events.append({
            "h3_cell": cell,
            "withdrawal_timestamp": ts.isoformat(),
            "latitude": lat,
            "longitude": lon,
        })

    return events


def generate_dataset(config_path="config/generation_config.yaml"):
    """
    Main generation function. Orchestrates the full synthetic dataset creation.

    Returns:
        dict: DataFrames for all generated tables.
    """
    print("=" * 60)
    print("HIVE-PREDICT SYNTHETIC DATA GENERATOR V2")
    print("=" * 60)

    # --- Load config ---
    config = load_config(config_path)
    seed = config["general"]["seed"]
    num_cases = config["general"]["num_cases"]
    output_dir = config["general"]["output_dir"]
    output_format = config["general"]["output_format"]

    print(f"\nConfiguration loaded from: {config_path}")
    print(f"  Cases to generate: {num_cases}")
    print(f"  Random seed: {seed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Output format: {output_format}")
    print(f"  Generator version: {config['general']['generator_version']}")

    rng = np.random.RandomState(seed)
    start_time = time.time()

    # --- Step 1: Generate ATM reference data ---
    print("\n[1/9] Generating ATM reference data...")
    atm_df = generate_atm_reference(config, rng)
    print(f"  -> {len(atm_df)} ATMs generated")

    # --- Step 2: Initialize registries + mule pool ---
    print("[2/9] Initializing account registry and mule pool...")
    account_registry = AccountRegistry(config, rng)
    # V2: Initialize mule pool before case generation
    pool_base_ts = config["temporal"]["start_date"] + "T00:00:00"
    account_registry.initialize_mule_pool(pool_base_ts)
    pool_stats = account_registry.get_network_stats()
    print(f"  -> Mule pool initialized: {pool_stats['total_mule_accounts']} accounts")
    fake = create_suspect_generator(seed)

    # --- Step 3: Generate historical seed events (V2) ---
    print("[3/9] Generating historical seed events...")
    historical_seed_events = generate_historical_seed_events(config, atm_df, rng)
    print(f"  -> {len(historical_seed_events)} historical seed withdrawals generated")

    # --- Step 4: Generate complaints ---
    print("[4/9] Generating complaints...")
    complaints = generate_complaints(config, rng, account_registry)
    print(f"  -> {len(complaints)} complaints generated")

    # --- Step 5-7: Generate connected event chains ---
    print("[5/9] Generating connected fraud event chains (V2)...")
    all_transactions = []
    all_mule_chains = []
    all_suspects = []
    all_withdrawals = []
    all_labels = []

    tx_counter = 0
    chain_counter = 0
    suspect_counter = 0
    withdrawal_counter = 0

    # Progress tracking
    profile_counts = {}
    cashout_overrides = {"profile_to_nocashout": 0, "nocashout_to_cashout": 0}

    for case_idx, complaint in enumerate(complaints):
        if (case_idx + 1) % 1000 == 0:
            elapsed = time.time() - start_time
            print(f"  Processing case {case_idx + 1}/{num_cases} "
                  f"({elapsed:.1f}s elapsed)")

        # Select behavioral profile
        profile_name = select_behavioral_profile(config, rng)
        profile_cfg = config["behavioral_profiles"][profile_name]
        profile_counts[profile_name] = profile_counts.get(profile_name, 0) + 1

        # V2: Store profile in complaint for audit (NOT in features)
        complaint["behavioral_profile"] = profile_name

        # --- Generate initial transaction ---
        tx, dest_account_id, tx_counter = generate_initial_transaction(
            complaint, account_registry, config, rng, tx_counter
        )
        all_transactions.append(tx)
        initial_tx_datetime = datetime.fromisoformat(tx["transaction_datetime"])

        # --- Generate mule chain (V2: with geo drift) ---
        (mule_records, mule_txns, final_account, final_amount,
         final_time, chain_depth, tx_counter,
         drifted_lat, drifted_lon) = generate_mule_chain(
            complaint, dest_account_id, complaint["fraud_amount"],
            initial_tx_datetime, profile_cfg, config,
            account_registry, rng, tx_counter
        )

        chain_counter += 1
        for mr in mule_records:
            mr["chain_id"] = f"CHN{str(chain_counter).zfill(8)}"

        all_mule_chains.extend(mule_records)
        all_transactions.extend(mule_txns)

        # --- Generate suspect ---
        suspect, suspect_counter = generate_suspect(
            complaint, fake, rng, suspect_counter
        )
        all_suspects.append(suspect)

        # --- V2: Determine cashout with stochastic profile decoupling ---
        cashout_prob = profile_cfg.get("cashout_probability", 1.0)
        do_cashout = rng.random() < cashout_prob

        # Override: NO_CASHOUT profile might still cash out (surprise)
        if profile_name == "NO_CASHOUT" and do_cashout:
            cashout_overrides["nocashout_to_cashout"] += 1
        elif profile_name != "NO_CASHOUT" and not do_cashout:
            cashout_overrides["profile_to_nocashout"] += 1

        # --- Generate withdrawals (V2: temporal + geographic fixes) ---
        withdrawals, withdrawal_counter = generate_withdrawals(
            complaint, final_account, final_amount, final_time,
            profile_name, profile_cfg, atm_df, config,
            rng, withdrawal_counter,
            drifted_lat=drifted_lat, drifted_lon=drifted_lon,
            do_cashout=do_cashout,
        )
        all_withdrawals.extend(withdrawals)

        # --- Generate labels ---
        labels = generate_labels(complaint, withdrawals)
        all_labels.extend(labels)

    # --- Convert to DataFrames ---
    print("[6/9] Converting to DataFrames...")
    complaints_df = pd.DataFrame(complaints)
    accounts_df = account_registry.to_dataframe()
    transactions_df = pd.DataFrame(all_transactions)
    mule_chains_df = pd.DataFrame(all_mule_chains) if all_mule_chains else pd.DataFrame(
        columns=["chain_id", "complaint_id", "source_account", "destination_account",
                 "hop_number", "transaction_amount", "transaction_timestamp"]
    )
    suspects_df = pd.DataFrame(all_suspects)
    withdrawals_df = pd.DataFrame(all_withdrawals) if all_withdrawals else pd.DataFrame(
        columns=["withdrawal_id", "complaint_id", "account_id",
                 "withdrawal_timestamp", "amount", "atm_id",
                 "latitude", "longitude", "h3_cell"]
    )
    labels_df = pd.DataFrame(all_labels)

    # --- Step 6: Feature engineering (V2) ---
    print("[7/9] Computing ML features (V2: observable-at-cutoff)...")
    # Sort complaints by time for historical hotspot density calculation
    complaints_sorted = sorted(complaints, key=lambda c: c["complaint_registered_at"])

    # Pre-build lookup dicts for O(1) access
    mule_chains_by_complaint = {}
    for m in all_mule_chains:
        mule_chains_by_complaint.setdefault(m["complaint_id"], []).append(m)

    withdrawals_by_complaint = {}
    for w in all_withdrawals:
        withdrawals_by_complaint.setdefault(w["complaint_id"], []).append(w)

    feature_snapshots = []
    # V2.1: Use a Counter[h3_cell->count] instead of a growing list.
    # Seed with ALL historical seed events (they are all before the dataset window,
    # so no per-case timestamp filtering is needed).
    historical_h3_counter = Counter(e["h3_cell"] for e in historical_seed_events)
    print(f"  Seeded hotspot counter with {sum(historical_h3_counter.values())} events "
          f"across {len(historical_h3_counter)} unique H3 cells")

    for case_idx, complaint in enumerate(complaints_sorted):
        if (case_idx + 1) % 2000 == 0:
            elapsed = time.time() - start_time
            print(f"  Features for case {case_idx + 1}/{num_cases} ({elapsed:.1f}s)")

        cid = complaint["complaint_id"]
        case_mule = mule_chains_by_complaint.get(cid, [])
        case_withdrawals = withdrawals_by_complaint.get(cid, [])

        feat = compute_features_for_case(
            complaint=complaint,
            mule_chain_records=case_mule,
            withdrawals=case_withdrawals,
            atm_df=atm_df,
            all_complaints_so_far=complaints_sorted[:case_idx],
            historical_h3_counter=historical_h3_counter,
            config=config,
        )
        feature_snapshots.append(feat)

        # V2.1: Add this case's withdrawals to counter (for future cases)
        for w in case_withdrawals:
            historical_h3_counter[w["h3_cell"]] += 1

    # V2: Compute complaint clusters using rolling window (no temporal leakage)
    print("  Computing complaint clusters (V2: rolling-window DBSCAN)...")
    cluster_labels = compute_complaint_clusters_rolling(
        feature_snapshots, complaints_sorted, config
    )
    for i, cl in enumerate(cluster_labels):
        feature_snapshots[i]["complaint_cluster"] = cl

    feature_snapshots_df = pd.DataFrame(feature_snapshots)

    # --- Step 7: Validation ---
    print("[8/9] Running validation...")
    report = run_full_validation(
        complaints_df, accounts_df, transactions_df,
        mule_chains_df, suspects_df, withdrawals_df,
        atm_df, feature_snapshots_df, labels_df
    )
    print(report.summary())

    # --- Step 8: Save outputs ---
    print("[9/9] Saving outputs...")
    os.makedirs(output_dir, exist_ok=True)

    datasets = {
        "complaints": complaints_df,
        "accounts": accounts_df,
        "transactions": transactions_df,
        "mule_chains": mule_chains_df,
        "suspects": suspects_df,
        "withdrawals": withdrawals_df,
        "atm_reference": atm_df,
        "feature_snapshots": feature_snapshots_df,
        "cashout_labels": labels_df,
    }

    for name, df in datasets.items():
        if output_format == "parquet":
            path = os.path.join(output_dir, f"{name}.parquet")
            df.to_parquet(path, index=False)
        else:
            path = os.path.join(output_dir, f"{name}.csv")
            df.to_csv(path, index=False)
        print(f"  -> {name}: {len(df)} rows -> {path}")

    # --- Save validation report ---
    report_df = report.to_dataframe()
    report_path = os.path.join(output_dir, "validation_report.csv")
    report_df.to_csv(report_path, index=False)

    # --- Summary statistics ---
    elapsed = time.time() - start_time
    _print_summary(datasets, profile_counts, cashout_overrides, elapsed, config,
                   account_registry)

    # --- Save provenance ---
    _save_provenance(config, output_dir, datasets, elapsed, account_registry)

    # --- Save summary report ---
    _save_summary_report(datasets, profile_counts, config, output_dir,
                         account_registry, cashout_overrides)

    return datasets


def _print_summary(datasets, profile_counts, cashout_overrides, elapsed, config,
                   account_registry):
    """Print dataset summary statistics."""
    print("\n" + "=" * 60)
    print("GENERATION COMPLETE (V2)")
    print("=" * 60)
    print(f"\n  Time elapsed: {elapsed:.1f} seconds")
    print(f"\n  Record counts:")
    for name, df in datasets.items():
        print(f"    {name}: {len(df):,}")

    print(f"\n  Behavioral profiles:")
    for name, count in sorted(profile_counts.items()):
        pct = count / sum(profile_counts.values()) * 100
        print(f"    {name}: {count:,} ({pct:.1f}%)")

    print(f"\n  V2: Profile decoupling overrides:")
    print(f"    Non-NO_CASHOUT -> no cashout: {cashout_overrides['profile_to_nocashout']}")
    print(f"    NO_CASHOUT -> surprise cashout: {cashout_overrides['nocashout_to_cashout']}")

    # Network stats
    net_stats = account_registry.get_network_stats()
    print(f"\n  V2: Mule network statistics:")
    print(f"    Total mule accounts: {net_stats['total_mule_accounts']:,}")
    print(f"    Used once: {net_stats['accounts_used_once']:,}")
    print(f"    Used in multiple cases: {net_stats['accounts_used_multi']:,}")
    print(f"    Multi-complaint accounts: {net_stats['accounts_multi_complaint']:,}")
    print(f"    Max reuse: {net_stats['max_reuse']}")
    print(f"    Mean reuse: {net_stats['mean_reuse']:.2f}")

    complaints_df = datasets["complaints"]
    print(f"\n  Fraud amount statistics:")
    print(f"    Mean:   INR {complaints_df['fraud_amount'].mean():,.2f}")
    print(f"    Median: INR {complaints_df['fraud_amount'].median():,.2f}")
    print(f"    Min:    INR {complaints_df['fraud_amount'].min():,.2f}")
    print(f"    Max:    INR {complaints_df['fraud_amount'].max():,.2f}")

    labels_df = datasets["cashout_labels"]
    cashout = labels_df[labels_df["cashout_occurred"] == True]
    no_cashout = labels_df[labels_df["cashout_occurred"] == False]

    print(f"\n  Cashout statistics:")
    print(f"    Cases with cashout: {cashout['complaint_id'].nunique():,}")
    print(f"    Cases NO cashout:   {no_cashout['complaint_id'].nunique():,}")
    if len(cashout) > 0 and "time_to_cashout_seconds" in cashout.columns:
        ttc = cashout["time_to_cashout_seconds"].dropna()
        if len(ttc) > 0:
            print(f"    Min time-to-cashout:  {ttc.min() / 60:.1f} min")
            print(f"    Avg time-to-cashout:  {ttc.mean() / 60:.1f} min")
            print(f"    Med time-to-cashout:  {ttc.median() / 60:.1f} min")
            neg = (ttc <= 0).sum()
            print(f"    Negative/zero ttc:    {neg} {'(GOOD: 0)' if neg == 0 else '(BAD!)'}")

    print(f"\n  Geographic coverage:")
    print(f"    States: {complaints_df['victim_state'].nunique()}")
    wdr = datasets["withdrawals"]
    if len(wdr) > 0:
        print(f"    Unique H3 cells:  {wdr['h3_cell'].nunique()}")
        print(f"    Unique ATMs used: {wdr['atm_id'].nunique()}")

    # Provenance
    print(f"\n  Provenance saved to metadata/provenance.json")


def _save_provenance(config, output_dir, datasets, elapsed, account_registry):
    """Save provenance metadata."""
    net_stats = account_registry.get_network_stats()
    provenance = {
        "generator_version": config["general"]["generator_version"],
        "schema_version": config["general"]["schema_version"],
        "generation_timestamp": datetime.utcnow().isoformat() + "Z",
        "random_seed": config["general"]["seed"],
        "num_cases": config["general"]["num_cases"],
        "h3_resolution": config["general"]["h3_resolution_primary"],
        "output_format": config["general"]["output_format"],
        "generation_time_seconds": round(elapsed, 1),
        "record_counts": {name: len(df) for name, df in datasets.items()},
        "v2_changes": [
            "Prediction-time temporal semantics enforced",
            "Mule account pool with reuse across complaints",
            "Rolling-window DBSCAN clustering (no temporal leakage)",
            "Observable-at-cutoff mule chain features",
            "Stochastic profile-outcome decoupling",
            "Historical seed events for hotspot density",
            "Weighted ATM selection (victim/drift/random)",
            "Amount velocity minimum time floor",
        ],
        "mule_network_stats": {
            "total_mule_accounts": net_stats["total_mule_accounts"],
            "accounts_multi_complaint": net_stats["accounts_multi_complaint"],
            "max_reuse": net_stats["max_reuse"],
        },
        "configuration_file": "config/generation_config.yaml",
        "reference_datasets": ["SYNTHETIC_ATM_REFERENCE"],
        "synthetic_assumptions": [
            "Fraud type distribution is a simulation assumption",
            "Fraud amount lognormal parameters are simulation assumptions",
            "Behavioral profile probabilities are simulation assumptions",
            "State-level fraud weights are approximate simulation assumptions",
            "ATM locations are synthetic and do NOT represent real ATM locations",
            "Hourly fraud distribution is a simulation assumption",
            "Mule chain friction rates are simulation assumptions",
            "Mule account reuse probabilities are simulation assumptions",
            "Historical seed event distribution is a simulation assumption",
        ],
        "data_sources": [
            "All data is synthetically generated",
            "No real PII or individual records are included",
            "Geographic bounds are approximate Indian state boundaries",
        ],
    }

    os.makedirs("metadata", exist_ok=True)
    with open(os.path.join("metadata", "provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2)


def _save_summary_report(datasets, profile_counts, config, output_dir,
                         account_registry, cashout_overrides):
    """Save a detailed summary report."""
    complaints_df = datasets["complaints"]
    withdrawals_df = datasets["withdrawals"]
    labels_df = datasets["cashout_labels"]
    mule_chains_df = datasets["mule_chains"]
    feature_snapshots_df = datasets["feature_snapshots"]

    lines = []
    lines.append("# HIVE-Predict Synthetic Data V2 — Generation Summary Report")
    lines.append(f"\nGenerated: {datetime.utcnow().isoformat()}Z")
    lines.append(f"Seed: {config['general']['seed']}")
    lines.append(f"Cases: {config['general']['num_cases']}")
    lines.append(f"Version: {config['general']['generator_version']}")

    lines.append("\n## Record Counts")
    for name, df in datasets.items():
        lines.append(f"- {name}: {len(df):,}")

    lines.append("\n## V2 Remediation Summary")
    lines.append("- Prediction-time temporal semantics enforced")
    lines.append("- Mule account pool with cross-complaint reuse")
    lines.append("- Rolling-window DBSCAN clustering (no temporal leakage)")
    lines.append("- Observable-at-cutoff mule chain features")
    lines.append("- Stochastic profile-outcome decoupling")
    lines.append("- Historical seed events for hotspot density")
    lines.append("- Weighted ATM selection (victim/drift/random)")
    lines.append("- Amount velocity minimum time floor")

    lines.append("\n## Behavioral Profiles")
    for name, count in sorted(profile_counts.items()):
        pct = count / sum(profile_counts.values()) * 100
        lines.append(f"- {name}: {count:,} ({pct:.1f}%)")

    lines.append("\n## Profile Decoupling (V2)")
    lines.append(f"- Non-NO_CASHOUT -> no cashout: {cashout_overrides['profile_to_nocashout']}")
    lines.append(f"- NO_CASHOUT -> surprise cashout: {cashout_overrides['nocashout_to_cashout']}")

    # Network stats
    net_stats = account_registry.get_network_stats()
    lines.append("\n## Mule Network Statistics (V2)")
    lines.append(f"- Total mule accounts: {net_stats['total_mule_accounts']:,}")
    lines.append(f"- Used once: {net_stats['accounts_used_once']:,}")
    lines.append(f"- Multi-case accounts: {net_stats['accounts_used_multi']:,}")
    lines.append(f"- Multi-complaint accounts: {net_stats['accounts_multi_complaint']:,}")
    lines.append(f"- Max reuse: {net_stats['max_reuse']}")
    lines.append(f"- Mean reuse: {net_stats['mean_reuse']:.2f}")

    lines.append("\n## Cashout Statistics")
    cashout = labels_df[labels_df["cashout_occurred"] == True]
    no_cashout = labels_df[labels_df["cashout_occurred"] == False]
    lines.append(f"- Cashout: {cashout['complaint_id'].nunique():,} cases")
    lines.append(f"- No cashout: {no_cashout['complaint_id'].nunique():,} cases")

    if len(cashout) > 0:
        ttc = cashout["time_to_cashout_seconds"].dropna()
        if len(ttc) > 0:
            lines.append(f"- Min time-to-cashout: {ttc.min() / 60:.1f} min")
            lines.append(f"- Mean time-to-cashout: {ttc.mean() / 60:.1f} min")
            lines.append(f"- Median time-to-cashout: {ttc.median() / 60:.1f} min")
            lines.append(f"- Negative/zero ttc: {(ttc <= 0).sum()}")

    lines.append("\n## Geographic Distribution")
    lines.append(f"- States: {complaints_df['victim_state'].nunique()}")
    if len(withdrawals_df) > 0:
        lines.append(f"- Unique H3 cells: {withdrawals_df['h3_cell'].nunique()}")
        lines.append(f"- Unique ATMs used: {withdrawals_df['atm_id'].nunique()}")

    lines.append("\n## Feature Snapshot Statistics")
    for col in ["fraud_amount", "mule_chain_depth", "mule_velocity",
                "amount_velocity", "atm_density", "historical_hotspot_density"]:
        if col in feature_snapshots_df.columns:
            series = feature_snapshots_df[col]
            lines.append(f"- {col}: mean={series.mean():.4f}, std={series.std():.4f}")

    report_text = "\n".join(lines)
    report_path = os.path.join(output_dir, "summary_report.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"  Summary report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HIVE-Predict Synthetic Data Generator V2"
    )
    parser.add_argument("--config", default="config/generation_config.yaml",
                        help="Path to configuration YAML file")
    args = parser.parse_args()

    generate_dataset(args.config)
