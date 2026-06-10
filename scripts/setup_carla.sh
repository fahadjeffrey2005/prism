#!/usr/bin/env bash
# PRISM — CARLA Setup Script
# Installs CARLA 0.9.15 and Python client on Ubuntu / Parrot OS
# Run once on your Linux x86_64 machine (not Jetson)
#
# Usage:
#   chmod +x scripts/setup_carla.sh
#   ./scripts/setup_carla.sh

set -e

CARLA_VERSION="0.9.15"
CARLA_DIR="$(cd "$(dirname "$0")/.." && pwd)/carla_server"
CARLA_URL="https://github.com/carla-simulator/carla/releases/download/${CARLA_VERSION}/CARLA_${CARLA_VERSION}.tar.gz"
ADDMAPS_URL="https://github.com/carla-simulator/carla/releases/download/${CARLA_VERSION}/AdditionalMaps_${CARLA_VERSION}.tar.gz"

echo "============================================================"
echo "PRISM — CARLA ${CARLA_VERSION} Setup"
echo "============================================================"
echo ""
echo "Target directory: ${CARLA_DIR}"
echo ""

# Check for GPU
if ! command -v nvidia-smi &>/dev/null; then
    echo "WARNING: nvidia-smi not found. CARLA requires an NVIDIA GPU."
    echo "         Continuing anyway — it may still work with CPU rendering."
    echo ""
fi

# Install system deps
echo "[1/5] Installing system dependencies..."
sudo apt-get update -qq
sudo apt-get install -y wget python3-pip python3-venv libpng16-16 libtiff5 \
    libjpeg8 fontconfig xdg-utils 2>/dev/null || true

# Create CARLA directory
mkdir -p "$CARLA_DIR"

# Download CARLA
if [ ! -f "$CARLA_DIR/CarlaUE4.sh" ]; then
    echo "[2/5] Downloading CARLA ${CARLA_VERSION} (~9GB)..."
    echo "      This takes 10-20 minutes depending on connection speed."
    wget -q --show-progress -O "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz" "$CARLA_URL"
    echo "[3/5] Extracting CARLA..."
    tar -xf "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz" -C "$CARLA_DIR"
    rm "$CARLA_DIR/CARLA_${CARLA_VERSION}.tar.gz"
    echo "      CARLA extracted."
else
    echo "[2/5] CARLA already downloaded — skipping."
    echo "[3/5] Skipping extraction."
fi

# Install Python carla package
echo "[4/5] Installing Python carla package..."
pip3 install carla=="${CARLA_VERSION}" pygame numpy 2>/dev/null || \
    pip install carla=="${CARLA_VERSION}" pygame numpy

# Install PRISM dependencies on this machine
echo "[5/5] Installing PRISM dependencies..."
cd "$(dirname "$0")/.."
pip3 install -r requirements.txt 2>/dev/null || true
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118 2>/dev/null || \
    pip3 install torch torchvision  # CPU fallback

echo ""
echo "============================================================"
echo "CARLA setup complete."
echo ""
echo "To start CARLA server (headless, no display needed):"
echo "  ~/prism/carla_server/CarlaUE4.sh -RenderOffScreen"
echo ""
echo "Or with display:"
echo "  ~/prism/carla_server/CarlaUE4.sh"
echo ""
echo "Once server is running, in a new terminal:"
echo "  cd ~/prism && python3 scripts/run_carla_eval.py"
echo "============================================================"
