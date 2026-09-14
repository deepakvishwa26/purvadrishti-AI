"""
Run tests after verifying output files exist.
Waits up to 15 minutes for files to appear.
"""
import os
import sys
import time
import subprocess

output_dir = "data/output"
required_files = [
    "complaints.csv", "accounts.csv", "transactions.csv",
    "mule_chains.csv", "suspects.csv", "withdrawals.csv",
    "atm_reference.csv", "feature_snapshots.csv", "cashout_labels.csv"
]

print("Waiting for output files to appear...")
max_wait = 900  # 15 minutes
waited = 0
while waited < max_wait:
    missing = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
    if not missing:
        break
    if waited % 60 == 0:
        print(f"  [{waited}s] Still waiting for {len(missing)} files...")
    time.sleep(10)
    waited += 10

# Final check
missing = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
if missing:
    print(f"ERROR: After {waited}s, still missing: {missing}")
    sys.exit(1)

# Verify non-empty
print("\nOutput files found:")
for f in required_files:
    path = os.path.join(output_dir, f)
    size = os.path.getsize(path)
    print(f"  {f}: {size:,} bytes")
    if size == 0:
        print(f"  ERROR: {f} is empty!")
        sys.exit(1)

print("\nAll files present and non-empty. Running tests...\n")
result = subprocess.run(
    [sys.executable, "-m", "pytest", "tests/test_integrity.py", "-v", "--cache-clear"],
    cwd=os.getcwd()
)
sys.exit(result.returncode)
