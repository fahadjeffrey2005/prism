#!/bin/bash
# Fix cv2 for Jetson — symlinks system OpenCV into the venv

SRC="/usr/lib/python3/dist-packages/cv2.cpython-312-aarch64-linux-gnu.so"
DST="/home/koushik-test/cad_pipeline_env/lib/python3.12/site-packages/cv2.cpython-312-aarch64-linux-gnu.so"

if [ ! -f "$SRC" ]; then
    echo "System cv2 not found at $SRC"
    echo "Trying to find it..."
    SRC=$(find /usr -name "cv2*.so" 2>/dev/null | grep python3 | head -1)
    echo "Found: $SRC"
fi

echo "Linking: $SRC -> $DST"
ln -sf "$SRC" "$DST"

echo "Testing..."
/home/koushik-test/cad_pipeline_env/bin/python scripts/check_cv2.py
