"""
PRISM — Extract profiling summary from log files
Run: python scripts/extract_profiling.py
Output saved to ~/prism_data/logs/profiling_summary.txt
"""

import os
from pathlib import Path

log_dir = Path.home() / "prism_data/logs"
out_file = log_dir / "profiling_summary.txt"

results = []

for f in sorted(log_dir.glob("profile_*.txt")):
    lines = f.read_text(errors="replace").splitlines()
    profile_lines = [l for l in lines if "PROFILE" in l]
    fps_lines     = [l for l in lines if "avg" in l and "fps" in l]
    done_lines    = [l for l in lines if "Done" in l and "frames" in l]

    if not profile_lines:
        continue

    # Average all PROFILE lines for stable numbers
    fields = {}
    for pl in profile_lines:
        for match in __import__("re").findall(r"(\w+)=(\d+\.?\d*)ms", pl):
            key, val = match[0], float(match[1])
            fields.setdefault(key, []).append(val)

    avg = {k: round(sum(v) / len(v), 1) for k, v in fields.items()}

    fps_str  = fps_lines[-1].split("|")[-1].strip()  if fps_lines  else "N/A"
    done_str = done_lines[-1].strip()                 if done_lines else "N/A"

    results.append({
        "bag":     f.stem.replace("profile_", ""),
        "avg":     avg,
        "fps":     fps_str,
        "summary": done_str,
    })

# Build output
lines_out = []
lines_out.append("=" * 70)
lines_out.append("PRISM — Profiling Summary across all bags")
lines_out.append("=" * 70)

all_vals = {}
for r in results:
    lines_out.append(f"\n{r['bag']}")
    lines_out.append(f"  {r['summary']}")
    lines_out.append(f"  {r['fps']}")
    for k, v in r["avg"].items():
        lines_out.append(f"  {k}: {v} ms")
        all_vals.setdefault(k, []).append(v)

# Grand average across all bags
lines_out.append("\n" + "=" * 70)
lines_out.append("GRAND AVERAGE (across all bags):")
for k, v in sorted(all_vals.items()):
    mean = round(sum(v) / len(v), 1)
    lines_out.append(f"  {k}: {mean} ms")

total_keys = ["sensory", "metric", "lidar", "world", "pred", "decision", "render"]
total = sum(
    round(sum(all_vals[k]) / len(all_vals[k]), 1)
    for k in total_keys if k in all_vals
)
lines_out.append(f"\n  TOTAL pipeline (sum of above): {round(total, 1)} ms")
lines_out.append(f"  Theoretical max FPS: {round(1000/total, 1) if total > 0 else 'N/A'}")
lines_out.append("=" * 70)

output = "\n".join(lines_out)
print(output)
out_file.write_text(output)
print(f"\nSaved to: {out_file}")
