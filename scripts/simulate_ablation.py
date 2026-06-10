"""
PRISM — Ablation Simulation
============================
Reads trigger CSVs from run_vlm_bags.sh and simulates fixed-rate baselines.
Optionally loads a safety event annotation file to compute coverage.

Usage:
    # After run_vlm_bags.sh completes:
    python scripts/simulate_ablation.py

    # With safety event annotations:
    python scripts/simulate_ablation.py --annotations ~/prism_data/logs/safety_events.csv

Outputs a summary table suitable for direct copy into the paper.
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict


TRIGGER_LOG_DIR = Path.home() / "prism_data/logs/triggers"
OUT_FILE        = Path.home() / "prism_data/logs/ablation_results.txt"


# ── Load trigger logs ─────────────────────────────────────────────────────────

def load_trigger_log(csv_path: Path) -> list:
    """Returns list of dicts, one per frame."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "timestamp":   float(row.get("timestamp", 0)),
                "frame_idx":   int(row.get("frame_idx", 0)),
                "triggered":   int(row.get("triggered", 0)),
                "condition":   row.get("trigger_condition", ""),
                "divergence":  float(row.get("divergence_score", 0)),
                "actor_count": int(row.get("actor_count", 0)),
                "risk_level":  int(row.get("risk_level", 0)),
                "vlm_caution": row.get("vlm_caution", ""),
            })
    return rows


# ── Simulate fixed-rate baseline ──────────────────────────────────────────────

def simulate_fixed_rate(frames: list, hz: float) -> list:
    """
    Returns list of frame indices that would be triggered at a fixed rate.
    Simulates a timer-based trigger firing every 1/hz seconds.
    """
    if not frames:
        return []
    t0        = frames[0]["timestamp"]
    interval  = 1.0 / hz
    triggers  = []
    next_fire = t0
    for f in frames:
        if f["timestamp"] >= next_fire:
            triggers.append(f["frame_idx"])
            next_fire += interval
    return triggers


# ── Coverage analysis ─────────────────────────────────────────────────────────

def compute_coverage(
    safety_events: list,       # list of timestamps
    trigger_frames: list,      # list of frame dicts with 'timestamp'
    window_s: float = 1.0,
) -> float:
    """
    What fraction of safety events had a trigger within ±window_s?
    """
    if not safety_events:
        return None
    trigger_times = [f["timestamp"] for f in trigger_frames if f["triggered"]]
    covered = 0
    for event_t in safety_events:
        if any(abs(t - event_t) <= window_s for t in trigger_times):
            covered += 1
    return covered / len(safety_events)


def compute_fixed_coverage(
    safety_events: list,
    all_frames: list,
    fixed_trigger_indices: list,
    window_s: float = 1.0,
) -> float:
    if not safety_events:
        return None
    idx_set = set(fixed_trigger_indices)
    trigger_times = [f["timestamp"] for f in all_frames if f["frame_idx"] in idx_set]
    covered = 0
    for event_t in safety_events:
        if any(abs(t - event_t) <= window_s for t in trigger_times):
            covered += 1
    return covered / len(safety_events)


# ── Condition breakdown ───────────────────────────────────────────────────────

def condition_breakdown(frames: list) -> dict:
    counts = defaultdict(int)
    total_triggered = 0
    for f in frames:
        if f["triggered"]:
            total_triggered += 1
            cond = f["condition"]
            # Normalise condition label
            if "divergence" in cond:       key = "T1_divergence"
            elif "new_actor" in cond:      key = "T2_new_actor"
            elif "risk_jump" in cond:      key = "T3_risk_jump"
            elif "collision" in cond:      key = "T4_ttc"
            elif "periodic" in cond:       key = "T5_keepalive"
            else:                          key = "other"
            counts[key] += 1
    if total_triggered == 0:
        return {}
    return {k: round(v / total_triggered * 100, 1) for k, v in counts.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default=None,
                        help="CSV file with safety_event_timestamp column")
    args = parser.parse_args()

    logs = sorted(TRIGGER_LOG_DIR.glob("triggers_*.csv"))
    if not logs:
        print(f"No trigger logs found in {TRIGGER_LOG_DIR}")
        print("Run scripts/run_vlm_bags.sh first.")
        return

    # Load safety event annotations if provided
    safety_events = []
    if args.annotations:
        ann_path = Path(args.annotations)
        if ann_path.exists():
            with open(ann_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    t = row.get("safety_event_timestamp") or row.get("timestamp")
                    if t:
                        safety_events.append(float(t))
            print(f"Loaded {len(safety_events)} safety event annotations")

    # Aggregate across all bags
    all_frames       = []
    total_triggered  = 0
    all_conditions   = defaultdict(int)

    per_bag_results = []

    for log_path in logs:
        frames = load_trigger_log(log_path)
        if not frames:
            continue

        n_frames     = len(frames)
        n_triggered  = sum(f["triggered"] for f in frames)
        trigger_rate = n_triggered / max(n_frames, 1) * 100

        # Fixed-rate simulations
        fixed_2hz = simulate_fixed_rate(frames, 2.0)
        fixed_1hz = simulate_fixed_rate(frames, 1.0)
        fixed_05hz= simulate_fixed_rate(frames, 0.5)

        # Per-bag condition breakdown
        conds = condition_breakdown(frames)

        per_bag_results.append({
            "bag":          log_path.stem.replace("triggers_", ""),
            "frames":       n_frames,
            "triggered":    n_triggered,
            "rate_pct":     round(trigger_rate, 1),
            "fixed_2hz_n":  len(fixed_2hz),
            "fixed_1hz_n":  len(fixed_1hz),
            "fixed_05hz_n": len(fixed_05hz),
            "conditions":   conds,
        })

        all_frames.extend(frames)
        total_triggered += n_triggered
        for k, v in conds.items():
            all_conditions[k] += v

    # ── Overall stats ─────────────────────────────────────────────────────────
    total_frames     = len(all_frames)
    overall_rate     = total_triggered / max(total_frames, 1) * 100

    fixed_2hz_all    = simulate_fixed_rate(all_frames, 2.0)
    fixed_1hz_all    = simulate_fixed_rate(all_frames, 1.0)
    fixed_05hz_all   = simulate_fixed_rate(all_frames, 0.5)

    # Coverage (only if annotations provided)
    cov_event    = compute_coverage(safety_events, all_frames) if safety_events else None
    cov_2hz      = compute_fixed_coverage(safety_events, all_frames, fixed_2hz_all)  if safety_events else None
    cov_1hz      = compute_fixed_coverage(safety_events, all_frames, fixed_1hz_all)  if safety_events else None
    cov_05hz     = compute_fixed_coverage(safety_events, all_frames, fixed_05hz_all) if safety_events else None

    # ── Output ────────────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 70)
    lines.append("PRISM — VLM Trigger Ablation Results")
    lines.append("=" * 70)
    lines.append(f"\nTotal frames analysed : {total_frames:,}")
    lines.append(f"Total VLM triggers    : {total_triggered:,}")
    lines.append(f"Overall trigger rate  : {overall_rate:.1f}%")
    lines.append("")

    lines.append("Per-bag summary:")
    lines.append(f"  {'Bag':<45} {'Frames':>7} {'Triggers':>9} {'Rate':>6}")
    lines.append("  " + "-" * 70)
    for r in per_bag_results:
        lines.append(f"  {r['bag']:<45} {r['frames']:>7,} {r['triggered']:>9,} {r['rate_pct']:>5.1f}%")

    # Condition breakdown
    lines.append("\nTrigger condition breakdown (grand average):")
    n_bags = max(len(per_bag_results), 1)
    for k in ["T1_divergence", "T2_new_actor", "T3_risk_jump", "T4_ttc", "T5_keepalive"]:
        avg = all_conditions.get(k, 0) / n_bags
        lines.append(f"  {k:<20}: {avg:.1f}%")

    # Ablation table
    lines.append("\nAblation — trigger rate vs safety coverage:")
    lines.append(f"  {'Strategy':<30} {'Trigger Rate':>14} {'Safety Coverage':>16}")
    lines.append("  " + "-" * 62)

    def fmt_cov(v):
        return f"{v*100:.1f}%" if v is not None else "N/A (annotate first)"

    lines.append(f"  {'Every frame (100%)':<30} {'100.0%':>14} {'100.0%':>16}")
    lines.append(f"  {'Fixed 2 Hz':<30} {len(fixed_2hz_all)/max(total_frames,1)*100:>13.1f}% {fmt_cov(cov_2hz):>16}")
    lines.append(f"  {'Fixed 1 Hz':<30} {len(fixed_1hz_all)/max(total_frames,1)*100:>13.1f}% {fmt_cov(cov_1hz):>16}")
    lines.append(f"  {'Fixed 0.5 Hz':<30} {len(fixed_05hz_all)/max(total_frames,1)*100:>13.1f}% {fmt_cov(cov_05hz):>16}")
    lines.append(f"  {'Event-driven (ours)':<30} {overall_rate:>13.1f}% {fmt_cov(cov_event):>16}")
    lines.append("")

    if not safety_events:
        lines.append("NOTE: Run with --annotations to fill in Safety Coverage column.")
        lines.append("      Create ~/prism_data/logs/safety_events.csv with columns:")
        lines.append("      safety_event_timestamp,description,bag_name")

    lines.append("=" * 70)

    output = "\n".join(lines)
    print(output)
    OUT_FILE.write_text(output)
    print(f"\nSaved to: {OUT_FILE}")


if __name__ == "__main__":
    main()
