#!/bin/bash
# PRISM — TensorRT baseline benchmark (VLM disabled, no depth, TRT engine)
# Run: bash scripts/run_trt_benchmark.sh

PYTHON=/home/koushik-test/cad_pipeline_env/bin/python
BAG=~/Downloads/test12-002.db3
CONFIG=~/prism/configs/config.yaml
LOG=/tmp/trt_baseline.log

echo "Disabling VLM for clean baseline..."
sed -i 's/enabled: true/enabled: false/' $CONFIG

echo "Running 200-frame TRT benchmark..."
$PYTHON ~/prism/scripts/run_bag_dashboard.py $BAG \
    --config $CONFIG \
    --max-frames 200 \
    --no-show \
    --no-depth \
    --width 640 \
    --profile 2>&1 | tee $LOG

echo ""
echo "Re-enabling VLM..."
sed -i 's/enabled: false/enabled: true/' $CONFIG

echo ""
echo "=== RESULTS ==="
grep "PROFILE" $LOG | awk -F'[=ms ]' '{for(i=1;i<=NF;i++) if($i=="TOTAL") print $(i+1)"ms"}' | sort -n | awk '
    BEGIN{n=0; sum=0}
    {vals[n++]=$1; sum+=$1}
    END{
        print "Frames measured: " n
        print "Mean total:  " sum/n " ms  ->  " 1000/(sum/n) " fps"
        print "Min total:   " vals[0] " ms"
        print "Max total:   " vals[n-1] " ms"
        print "p50 total:   " vals[int(n*0.5)] " ms"
    }'

grep "avg" $LOG | tail -3
