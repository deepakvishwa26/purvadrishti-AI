"""Quick check that output files exist and are non-empty."""
import os

output_dir = "data/output"
if not os.path.exists(output_dir):
    print(f"ERROR: {output_dir} does not exist")
else:
    files = os.listdir(output_dir)
    print(f"Files found: {len(files)}")
    for f in sorted(files):
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path)
        print(f"  {f}: {size:,} bytes")
    if len(files) == 0:
        print("ERROR: No output files found. Generator may not have finished yet.")
