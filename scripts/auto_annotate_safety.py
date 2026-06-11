"""
PRISM — Automatic Safety Event Annotation
==========================================
Reads trigger CSVs and auto-detects safety-critical events based on:
  - T3 (risk_jump) triggers — sudden scene escalation
  - T4 (collision_risk / TTC) triggers
  - Sustained high risk_level (>=2) windows
  - Actor spikes (rapid increase in actor count)

Writes ~/prism_data/logs/safety_events.csv ready for simulate_ablation.py

Usage:
    /home/koushik-test/cad_pipeline_env/bin/python scripts/auto_annotate_safety.py

    # Then run ablation with coverage:
    /home/koushik-test/cad_pipeline_env/bin/python scripts/simulate_ablation.py \
        --annotations ~/prism_data/logs/safety_events.csv
"""

import csv
from pathlib import Path
from collections import defaultdict

TRIGGER_LOG_DIR = Path.home() / "prism_data/logs/triggers"
OUT_FILE        = Path.home() / "prism_data/logs/safety_events.csv"

# ── Detection parameters ──────────────────────────────────────────────────────
RISK_JUMP_LEVELS   = 1        # T3: risk jumped by this many levels
HIGH_RISK_LEVEL    = 2        # sustained risk at or above this level
HIGH_RISK_WINDOW_S = 1.0      # seconds of sustained high risk to count as event
ACTOR_SPIKE_THRESH = 3        # sudden increase in actor count
MERGE_WINDOW_S     = 2.0      # merge events within this window into one


def detect_safety_events(frames: list, bag_name: str) -> list:
    """Returns list of safety event dicts from one bag's frame data."""
    events = []
    prev_risk   = 0
    prev_actors = 0
    high_risk_start = None

    for i, f in enumerate(frames):
        t           = f["timestamp"]
        risk_level  = int(f.get("risk_level", 0))
        risk_score  = float(f.get("risk_score", 0))
        actor_count = int(f.get("actor_count", 0))
        condition   = f.get("trigger_condition", "")
        triggered   = int(f.get("triggered", 0))

        # T4: explicit collision risk trigger
        if triggered and "collision" in condition.lower():
            events.append({
                "safety_event_timestamp": round(t, 4),
                "description":            f"collision_risk:{condition}",
                "bag_name":               bag_name,
                "auto_source":            "T4_ttc",
            })

        # T3: explicit risk jump trigger
        if triggered and "risk_jump" in condition.lower():
            events.append({
                "safety_event_timestamp": round(t, 4),
                "description":            f"risk_jump:{condition}",
                "bag_name":               bag_name,
                "auto_source":            "T3_risk_jump",
            })

        # Sustained high risk
        if risk_level >= HIGH_RISK_LEVEL:
            if high_risk_start is None:
                high_risk_start = t
        else:
            if high_risk_start is not None:
                duration = t - high_risk_start
                if duration >= HIGH_RISK_WINDOW_S:
                    events.append({
                        "safety_event_timestamp": round(high_risk_start, 4),
                        "description":            f"sustained_risk:{duration:.1f}s_level{HIGH_RISK_LEVEL}",
                        "bag_name":               bag_name,
                        "auto_source":            "sustained_high_risk",
                    })
                high_risk_start = None

        # Actor spike
        if actor_count - prev_actors >= ACTOR_SPIKE_THRESH:
            events.append({
                "safety_event_timestamp": round(t, 4),
                "description":            f"actor_spike:+{actor_count - prev_actors}",
                "bag_name":               bag_name,
                "auto_source":            "actor_spike",
            })

        prev_risk   = risk_level
        prev_actors = actor_count

    return events


def merge_nearby_events(events: list, window_s: float) -> list:
    """Merge events within window_s of each other (keep earliest timestamp)."""
    if not events:
        return []
    events.sort(key=lambda e: e["safety_event_timestamp"])
    merged = [events[0]]
    for ev in events[1:]:
        if ev["safety_event_timestamp"] - merged[-1]["safety_event_timestamp"] > window_s:
            merged.append(ev)
    return merged


def load_trigger_log(csv_path: Path) -> list:
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append({
                "timestamp":         float(row.get("timestamp", 0)),
                "frame_idx":         int(row.get("frame_idx", 0)),
                "risk_level":        int(float(row.get("risk_level", 0))),
                "risk_score":        float(row.get("risk_score", 0)),
                "actor_count":       int(float(row.get("actor_count", 0))),
                "divergence_score":  float(row.get("divergence_score", 0)),
                "triggered":         int(row.get("triggered", 0)),
                "trigger_condition": row.get("trigger_condition", ""),
                "vlm_caution":       row.get("vlm_caution", ""),
            })
    return rows


def main():
    logs = sorted(TRIGGER_LOG_DIR.glob("triggers_*.csv"))
    if not logs:
        print(f"No trigger logs found in {TRIGGER_LOG_DIR}")
        print("Run scripts/run_vlm_bags.sh first.")
        return

    all_events = []
    for log_path in logs:
        bag_name = log_path.stem.replace("triggers_", "")
        frames   = load_trigger_log(log_path)
        if not frames:
            continue
        events = detect_safety_events(frames, bag_name)
        events = merge_nearby_events(events, MERGE_WINDOW_S)
        all_events.extend(events)
        print(f"  {bag_name:<50} {len(frames):>6} frames  →  {len(events):>3} safety events")

    all_events.sort(key=lambda e: e["safety_event_timestamp"])

    # Write output
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "safety_event_timestamp", "description", "bag_name", "auto_source"
        ])
        writer.writeheader()
        writer.writerows(all_events)

    print(f"\nTotal safety events detected: {len(all_events)}")
    print(f"Saved to: {OUT_FILE}")
    print()
    print("Source breakdown:")
    by_source = defaultdict(int)
    for ev in all_events:
        by_source[ev["auto_source"]] += 1
    for src, count in sorted(by_source.items(), key=lambda x: -x[1]):
        print(f"  {src:<30} {count}")
    print()
    print("Now run:")
    print(f"  python scripts/simulate_ablation.py --annotations {OUT_FILE}")


if __name__ == "__main__":
    main()
