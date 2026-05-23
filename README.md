# PRISM
### Predictive Reasoning and Intuition System for Mobility

> *Camera-only autonomous mobility with human-like scene understanding*

---

## What is PRISM?

PRISM is a camera-only autonomous driving stack that replicates Tesla Vision's core philosophy — pure camera input, no LiDAR — and extends it with a VLM-powered semantic reasoning layer that gives the system human-like intuition and explainable decisions.

Built for edge deployment on Jetson hardware. Designed for Indian road conditions.

---

## Architecture

```
Camera(s)
    │
    ▼
SENSORY CORE          ← you are here (Month 1)
Feature extraction, detection, depth, optical flow
    │
    ▼
WORLD MODEL           ← Month 1-2
4D persistent representation of the scene
    │
    ▼
PREDICTIVE ENGINE     ← Month 2
Short + medium horizon trajectory prediction
    │
    ▼
SEMANTIC REASONER     ← Month 3
VLM-powered scene understanding (event-driven)
    │
    ▼
ARBITRATION CORE      ← Month 4
Risk-weighted decision layer (the "intuition")
    │
    ▼
ADAPTIVE PLANNER      ← Month 4
Context-aware trajectory planning
    │
    ▼
CONTROL OUTPUT
```

---

## Setup

```bash
# Clone and enter project
cd prism

# Create environment (Python 3.11)
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Verify Metal (M4) is working
python -c "import torch; print(torch.backends.mps.is_available())"
```

---

## Dataset Setup

1. Register at [nuscenes.org](https://www.nuscenes.org)
2. Download `v1.0-mini` (~4GB)
3. Extract to your T7 drive:

```bash
# Create data directory on T7
mkdir -p /Volumes/T7/prism_data/datasets/nuscenes

# Extract downloaded archive there
# Then symlink for easy access
ln -s /Volumes/T7/prism_data ~/prism_data
```

---

## Running

### Validate Sensory Core (start here)
```bash
python scripts/run_sensory_core.py
```

Options:
```bash
python scripts/run_sensory_core.py --scene 1          # different scene
python scripts/run_sensory_core.py --camera CAM_FRONT_LEFT
python scripts/run_sensory_core.py --max-frames 20    # quick test
python scripts/run_sensory_core.py --show             # live display
```

Output saved to `~/prism_data/experiments/visualizations/sensory_core/`

---

## Milestone Targets

| Component | Target Metric | Status |
|---|---|---|
| Sensory Core | mAP > 0.65 on nuScenes | 🔄 Building |
| World Model | BEV accuracy validated | ⏳ Pending |
| Predictive Engine | ADE < 1.5m @ 3s horizon | ⏳ Pending |
| VLM Reasoner | Scene accuracy > 80% | ⏳ Pending |
| Full Stack (CARLA) | Collision rate < 5% | ⏳ Pending |
| Hardware (Jetson) | Real-time @ 10fps | ⏳ Pending |

---

## Novel Contributions

1. **Event-driven VLM triggering** — VLM fires on prediction divergence, not a timer
2. **Confidence-aware arbitration** — system knows when it doesn't know
3. **India-specific fine-tuning** — semantic reasoner tuned for Indian road scenarios
4. **Unified 4D World Model** — single shared truth, all components read from it

---

## Stack

| Component | Model | Target FPS |
|---|---|---|
| Detection | YOLOv8n | 30 |
| Depth | Depth Anything v2 Small | 10 |
| Segmentation | FastSAM-s | 10 |
| Optical Flow | Farneback | 30 |
| BEV Transform | LSS | 10 |
| Trajectory Pred | AgentFormer-lite | 10 |
| VLM Reasoning | Qwen-VL 2B (quantized) | event-driven |
| Simulation | CARLA 0.9.15 | — |

---

*Built at Centre of Excellence on Autonomous Mobility*
