#!/bin/bash
# PRISM — Run all bags with VLM enabled for paper trigger experiments.
# Saves per-bag trigger CSV logs to ~/prism_data/logs/triggers/
# Usage: bash scripts/run_vlm_bags.sh

PYTHON=/home/koushik-test/cad_pipeline_env/bin/python3
SCRIPT=$HOME/prism/scripts/run_bag_dashboard.py
CONFIG=$HOME/prism/configs/config.yaml
LOG_DIR=$HOME/prism_data/logs/triggers

mkdir -p "$LOG_DIR"

echo "PRISM VLM Trigger Experiment"
echo "VLM model: $(ls $HOME/prism_data/models/qwen2_5_vl_7b/ | head -1 2>/dev/null || echo 'NOT FOUND')"
echo ""

# Discover bags via Python to handle encoding issues
BAGFILES=$($PYTHON - <<'EOF'
import os
from pathlib import Path
bags = []
for root, dirs, files in os.walk(str(Path.home() / "Downloads")):
    for f in files:
        if f.endswith(".db3"):
            bags.append(os.path.join(root, f))
for b in sorted(set(bags)):
    print(b)
EOF
)

if [ -z "$BAGFILES" ]; then
    echo "No bags found. Exiting."
    exit 1
fi

COUNT=0
while IFS= read -r BAGFILE; do
    BAGNAME=$(basename "$BAGFILE" .db3)
    COUNT=$((COUNT + 1))
    echo "========================================================"
    echo "[$COUNT] $BAGNAME"
    echo "========================================================"
    $PYTHON "$SCRIPT" "$BAGFILE" \
        --config "$CONFIG" \
        --no-depth \
        --no-show \
        --profile \
        2>&1 | tee "$LOG_DIR/../vlm_run_${BAGNAME}.txt"
    echo ""
done <<< "$BAGFILES"

echo "========================================================"
echo "Done. Trigger CSVs in: $LOG_DIR"
echo "========================================================"
