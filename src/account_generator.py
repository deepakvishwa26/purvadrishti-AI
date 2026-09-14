"""
Account entity generator — V2 with Mule Account Pool.

Creates synthetic bank accounts that form the entity layer connecting
complaints -> transactions -> mule chains -> withdrawals.

V2 changes:
- Pre-generates a mule account pool before case generation.
- Mule accounts can be REUSED across multiple complaints.
- Configurable reuse probability and max reuse per account.
- Tracks per-account participation for network statistics.

Account types: VICTIM, MULE, MERCHANT, OTHER
All identifiers are fully synthetic.
"""

import numpy as np
import pandas as pd


class AccountRegistry:
    """
    Maintains a registry of all generated accounts for a generation run.
    V2: Includes a pre-generated mule pool with reuse support.
    """

    def __init__(self, config, rng):
        self.config = config
        self.rng = rng
        self.banks = config["banks"]
        self._accounts = []
        self._counter = 0

        # V2: Mule pool
        self._mule_pool = []            # list of mule account_ids
        self._mule_reuse_count = {}     # account_id -> number of complaints using it
        self._mule_complaints = {}      # account_id -> set of complaint_ids
        self._pool_initialized = False

        # Pool config
        pool_cfg = config.get("mule_pool", {})
        self._pool_size = pool_cfg.get("pool_size", 5000)
        self._reuse_probability = pool_cfg.get("reuse_probability", 0.35)
        self._max_reuse = pool_cfg.get("max_reuse_per_account", 8)

    def initialize_mule_pool(self, base_timestamp):
        """
        Pre-generate a pool of mule accounts before case generation.
        These accounts exist in the synthetic world and can be reused.

        Args:
            base_timestamp: ISO timestamp string for first_seen of pool accounts.
        """
        if self._pool_initialized:
            return

        for _ in range(self._pool_size):
            acct_id = self.create_account("MULE", base_timestamp)
            self._mule_pool.append(acct_id)
            self._mule_reuse_count[acct_id] = 0
            self._mule_complaints[acct_id] = set()

        self._pool_initialized = True

    def get_or_create_mule(self, timestamp, complaint_id=None):
        """
        Get a mule account — either reuse from pool or create new.

        With probability `reuse_probability`, selects an existing mule
        from the pool (if one is available under the reuse cap).
        Otherwise creates a fresh mule account.

        Args:
            timestamp: ISO timestamp string for first_seen if creating new.
            complaint_id: Optional complaint_id for tracking reuse.

        Returns:
            str: The mule account_id.
        """
        # Try to reuse from pool
        if self._pool_initialized and self.rng.random() < self._reuse_probability:
            # Find reusable candidates (under max reuse cap)
            candidates = [
                aid for aid in self._mule_pool
                if self._mule_reuse_count[aid] < self._max_reuse
            ]
            if candidates:
                # Weight toward less-used accounts (Zipf-like)
                counts = np.array([self._mule_reuse_count[a] + 1 for a in candidates])
                weights = 1.0 / counts
                weights = weights / weights.sum()
                chosen = self.rng.choice(candidates, p=weights)
                self._mule_reuse_count[chosen] += 1
                if complaint_id:
                    self._mule_complaints[chosen].add(complaint_id)
                return chosen

        # Create new mule account
        acct_id = self.create_account("MULE", timestamp)
        self._mule_reuse_count[acct_id] = 1
        self._mule_complaints[acct_id] = set()
        if complaint_id:
            self._mule_complaints[acct_id].add(complaint_id)
        # Also add to pool for future reuse
        self._mule_pool.append(acct_id)
        return acct_id

    def create_account(self, account_type, first_seen_timestamp, bank_id=None):
        """
        Create a new synthetic account.

        Args:
            account_type: One of VICTIM, MULE, MERCHANT, OTHER.
            first_seen_timestamp: When this account first appeared in the system.
            bank_id: Optional specific bank. If None, randomly assigned.

        Returns:
            str: The new account_id.
        """
        self._counter += 1
        account_id = f"ACCT{str(self._counter).zfill(8)}"

        if bank_id is None:
            bank = self.banks[self.rng.randint(0, len(self.banks))]
            bank_id = bank["bank_id"]

        self._accounts.append({
            "account_id": account_id,
            "bank_id": bank_id,
            "account_type": account_type,
            "account_status": "ACTIVE",
            "first_seen_timestamp": first_seen_timestamp,
        })
        return account_id

    def create_mule_accounts(self, count, first_seen_timestamp):
        """Create multiple mule accounts. Returns list of account_ids."""
        return [
            self.create_account("MULE", first_seen_timestamp)
            for _ in range(count)
        ]

    def get_network_stats(self):
        """
        Compute mule network statistics.

        Returns:
            dict: Network statistics including reuse counts, degree, components.
        """
        reuse_counts = list(self._mule_reuse_count.values())
        multi_complaint = {
            aid: cids for aid, cids in self._mule_complaints.items()
            if len(cids) > 1
        }

        return {
            "total_mule_accounts": len(self._mule_pool),
            "accounts_used_once": sum(1 for c in reuse_counts if c <= 1),
            "accounts_used_multi": sum(1 for c in reuse_counts if c > 1),
            "max_reuse": max(reuse_counts) if reuse_counts else 0,
            "mean_reuse": np.mean(reuse_counts) if reuse_counts else 0,
            "accounts_multi_complaint": len(multi_complaint),
            "top_reused": sorted(
                [(aid, len(cids)) for aid, cids in multi_complaint.items()],
                key=lambda x: -x[1]
            )[:10],
        }

    def to_dataframe(self):
        """Export all accounts as a DataFrame."""
        return pd.DataFrame(self._accounts)
