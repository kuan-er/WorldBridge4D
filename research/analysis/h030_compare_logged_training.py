"""CPU-only paired logged-update comparison; sampling fingerprints, not validation."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics as st

import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--baseline", type=Path, required=True)
parser.add_argument("--candidate", type=Path, required=True)
parser.add_argument("--owner-session", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--start", type=int, default=150000)
parser.add_argument("--end", type=int, required=True)
args = parser.parse_args()
assert args.end > args.start + 500
args.output.mkdir(parents=True, exist_ok=True)


def load(run_dir, label):
    meta = yaml.safe_load((run_dir / "run.yaml").read_text())
    assert meta["owner"]["session_id"] == args.owner_session
    with (run_dir / "console.log").open("rb") as handle:
        data = handle.read(os.fstat(handle.fileno()).st_size)
    (args.output / f"{label}-console-prefix.log").write_bytes(data)
    rows = {}
    for line in data[:data.rfind(b"\n") + 1].decode("utf8", errors="replace").splitlines():
        if '[stdout] {"global_step":' not in line:
            continue
        value = json.loads(line.split("[stdout] ", 1)[1])
        step = int(value["global_step"])
        if "train/loss" in value and args.start < step <= args.end:
            if step in rows:
                assert rows[step] == value, (label, "conflicting duplicate", step)
            rows[step] = value
    return rows, {"run_id": meta["run_id"], "commit": meta["git"]["commit"],
                  "log_prefix_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


baseline, baseline_meta = load(args.baseline, "baseline")
candidate, candidate_meta = load(args.candidate, "candidate")
assert baseline.keys() == candidate.keys() and max(candidate) == args.end, "logged-step coverage mismatch"
fixed = {"train/dataset", "train/pairs", "system/world_size", "train/lr_factor"}
for step in sorted(baseline):
    a, b = baseline[step], candidate[step]
    keys = fixed | {k for k in a.keys() | b.keys() if k.startswith(("sampling/", "train/clips_seen"))}
    mismatch = {k: [a.get(k), b.get(k)] for k in keys if a.get(k) != b.get(k)}
    assert not mismatch, ("sampling/protocol fingerprint mismatch", step, mismatch)
metrics = ("xyz_loss", "raw_epe_m", "cycle_reprojection_loss", "cycle_reprojection_pixel_error",
           "cycle_reprojection_valid_points", "gradient_norm")
windows = [(args.start + 1, args.start + 500)]
windows += [(lo, min(lo + 999, args.end)) for lo in range(args.start + 501, args.end + 1, 1000)]
report = {"baseline": baseline_meta, "candidate": candidate_meta, "paired_logged_updates": len(baseline),
          "alignment": "exact logged step/dataset/clip counters/source-target-gap histograms/world/pairs/LR factor",
          "caveat": "Sampled training updates, not validation; histograms are not full per-clip identity records. Cycle coverage may vary with predictions. No significance or generalization claim.",
          "windows": []}
for lo, hi in windows:
    block = {"start": lo, "end": hi, "datasets": {}}
    for dataset_id, dataset in enumerate(("kubric", "pointodyssey", "dynamic_replica")):
        steps = [s for s, r in baseline.items() if lo <= s <= hi and r["train/dataset"] == dataset_id]
        assert steps, (lo, hi, dataset)
        result = {"paired_logged_batches": len(steps)}
        for metric in metrics:
            av = [baseline[s]["train/" + metric] for s in steps]
            bv = [candidate[s]["train/" + metric] for s in steps]
            assert all(math.isfinite(x) for x in av + bv), (dataset, metric)
            ma, mb = st.mean(av), st.mean(bv)
            result[metric] = {"baseline_mean": ma, "candidate_mean": mb,
                              "change_pct": 100 * (mb / ma - 1) if ma != 0 else None,
                              "median_paired_delta": st.median(b - a for a, b in zip(av, bv))}
        block["datasets"][dataset] = result
        print(json.dumps({"window": [lo, hi], "dataset": dataset, **result}), flush=True)
    report["windows"].append(block)
(args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"paired_logged_updates": len(baseline), "alignment_passed": True}), flush=True)
print("MATCHED_TRAINING_COMPARISON_OK", flush=True)
