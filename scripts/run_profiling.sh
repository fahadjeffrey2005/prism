#!/bin/bash
# PRISM — Run profiling across all bags for paper experiments
# Usage: bash scripts/run_profiling.sh

PYTHON=/home/koushik-test/cad_pipeline_env/bin/python3
SCRIPT=~/prism/scripts/run_bag_dashboard.py
BAG_ROOT=~/Downloads/"ROSBAG FILES"
LOG_DIR=~/prism_data/logs

mkdir -p "$LOG_DIR"

BAGS=(rosbag001 rosbag002 rosbag003 rosbag004 rosbag005 rosbag006 rosbag007)

for bag in "${BAGS[@]}"; do
    BAGFILE=$(find "$BAG_ROOT/$bag" -name "*.db3" | head -1)
    if [ -z "$BAGFILE" ]; then
        echo "[SKIP] No .db3 found in $BAG_ROOT/$bag"
        continue
    fi
    echo ""
    echo "========================================================"
    echo "Processing: $bag"
    echo "File: $BAGFILE"
    echo "========================================================"
    $PYTHON "$SCRIPT" "$BAGFILE" --no-depth --profile --no-show \
        2>&1 | tee "$LOG_DIR/profile_${bag}.txt"
done

echo ""
echo "========================================================"
echo "All bags done. Logs saved to: $LOG_DIR"
echo "========================================================"
