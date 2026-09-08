"""Read-only, CPU-only summaries of sampled training logs; not validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--run-dir", type=Path, required=True)
parser.add_argument("--owner-session", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start", type=int, default=150000)
parser.add_argument("--end", type=int, default=151750)
args = parser.parse_args()
meta = yaml.safe_load((args.run_dir / "run.yaml").read_text())
assert meta["owner"]["session_id"] == args.owner_session
with (args.run_dir / "console.log").open("rb") as handle:
    raw = handle.read(os.fstat(handle.fileno()).st_size)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "console-prefix.log").write_bytes(raw)
rows = {}
for line in raw.decode("utf8", errors="replace").splitlines():
    marker = '[stdout] {"global_step":'
    if marker not in line:
        continue
    value = json.loads(line.split("[stdout] ", 1)[1])
    if "train/loss" in value:
        value["timestamp"] = line.split("]", 1)[0][1:]
        rows[int(value["global_step"])] = value
metrics = ("loss", "xyz_loss", "raw_epe_m", "cycle_reprojection_loss", "cycle_reprojection_pixel_error", "gradient_norm")
report = {
    "source_run": meta["run_id"], "source_commit": meta["git"]["commit"],
    "log_prefix_sha256": hashlib.sha256(raw).hexdigest(), "log_prefix_bytes": len(raw),
    "latest_logged_step": max(rows), "latest_logged_at": rows[max(rows)]["timestamp"],
    "caveat": "Sampled training batches, not all updates, matched examples, or validation; per-dataset unweighted batch means.",
    "windows": [],
}
for lo in range(args.start + 1, args.end + 1, 500):
    hi = min(lo + 499, args.end)
    window = {"start": lo, "end": hi, "datasets": {}}
    for dataset_id, name in enumerate(("kubric", "pointodyssey", "dynamic_replica")):
        selected = [r for s, r in rows.items() if lo <= s <= hi and r["train/dataset"] == dataset_id]
        if not selected:
            continue
        out = {"logged_batches": len(selected)}
        for metric in metrics:
            values = [r["train/" + metric] for r in selected]
            assert all(math.isfinite(x) for x in values), (name, metric)
            out[metric] = {"mean": st.mean(values), "median": st.median(values), "std": st.stdev(values) if len(values) > 1 else 0.0}
        out["fraction_grad_norm_gt1"] = st.mean(r["train/gradient_norm"] > 1 for r in selected)
        out["lr_factor_range"] = [min(r["train/lr_factor"] for r in selected), max(r["train/lr_factor"] for r in selected)]
        window["datasets"][name] = out
        print(json.dumps({"window": [lo, hi], "dataset": name, "n": len(selected),
                          "xyz": out["xyz_loss"], "epe": out["raw_epe_m"],
                          "cycle_mean": out["cycle_reprojection_loss"]["mean"],
                          "cycle_px_mean": out["cycle_reprojection_pixel_error"]["mean"],
                          "grad_median": out["gradient_norm"]["median"],
                          "clip_fraction": out["fraction_grad_norm_gt1"]}), flush=True)
    report["windows"].append(window)
last_full = next(w for w in reversed(report["windows"]) if w["end"] - w["start"] == 499)
report["first_vs_last_full_percent_change"] = {}
for name, first in report["windows"][0]["datasets"].items():
    last = last_full["datasets"][name]
    report["first_vs_last_full_percent_change"][name] = {
        metric: 100 * (last[metric]["mean"] / first[metric]["mean"] - 1) for metric in metrics
    }
(args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"latest_logged_step": report["latest_logged_step"], "latest_logged_at": report["latest_logged_at"],
                  "first_vs_last_full_percent_change": report["first_vs_last_full_percent_change"]}), flush=True)
print("TRAINING_WINDOW_SUMMARY_OK", flush=True)
