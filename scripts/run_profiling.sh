#!/bin/bash
# PRISM — Run profiling across all bags for paper experiments
# Discovers .db3 files via Python to avoid filesystem encoding issues with
# directory names containing non-standard space characters.
#
# Usage: bash scripts/run_profiling.sh

PYTHON=/home/koushik-test/cad_pipeline_env/bin/python3
SCRIPT=$HOME/prism/scripts/run_bag_dashboard.py
LOG_DIR=$HOME/prism_data/logs

mkdir -p "$LOG_DIR"

echo "Discovering bags..."

# Use Python to walk the filesystem — avoids encoding issues with path names
BAGFILES=$($PYTHON - <<'EOF'
import os, glob
from pathlib import Path

search_roots = [
    Path.home() / "Downloads",
    Path.home() / "Downloads" / "ROSBAG FILES",
]

bags = []
for root in search_roots:
    if root.exists():
        for dirpath, dirs, files in os.walk(str(root)):
            for f in files:
                if f.endswith(".db3"):
                    bags.append(os.path.join(dirpath, f))

for b in sorted(set(bags)):
    print(b)
EOF
)

if [ -z "$BAGFILES" ]; then
    echo "No .db3 files found under ~/Downloads. Exiting."
    exit 1
fi

echo "Found bags:"
echo "$BAGFILES"
echo ""

while IFS= read -r BAGFILE; do
    BAGNAME=$(basename "$BAGFILE" .db3)
    echo "========================================================"
    echo "Processing: $BAGNAME"
    echo "File: $BAGFILE"
    echo "========================================================"
    $PYTHON "$SCRIPT" "$BAGFILE" --no-depth --profile --no-show \
        2>&1 | tee "$LOG_DIR/profile_${BAGNAME}.txt"
    echo ""
done <<< "$BAGFILES"

echo "========================================================"
echo "All bags done. Logs saved to: $LOG_DIR"
echo "========================================================"
