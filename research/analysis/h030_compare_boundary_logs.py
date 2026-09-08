"""Count-gated boundary review of frozen, paired full-LR training log prefixes.

Training nonboundary is >2px, not the >10px held-out interior. Point-weighted
means below reconstruct sums from logged EPE means; no new pixel inference,
clip/mask-identity proof, bootstrap significance, or optimized-loss quality claim.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as st

DATASETS = ("kubric", "pointodyssey", "dynamic_replica")
STRATA = {
    "boundary": ("boundary_raw_epe_m", "boundary_valid_points"),
    "nonboundary_gt2px": ("nonboundary_raw_epe_m", "nonboundary_valid_points"),
    "occluded_boundary": ("boundary_occluded_raw_epe_m", "boundary_occluded_valid_points"),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(isinstance(value, (int, float)) and math.isfinite(value) and value >= 0,
            "metrics must be finite and nonnegative")
    return value


def count(value):
    require(type(value) is int and value >= 0, "counts must be nonnegative integers")
    return value


def change(a, b):
    return 100 * (b / a - 1) if a else None


def paired_mean(a, b):
    if not a:
        return {"baseline_mean": None, "candidate_mean": None, "change_pct": None}
    ma, mb = st.mean(a), st.mean(b)
    return {"baseline_mean": ma, "candidate_mean": mb, "change_pct": change(ma, mb)}


def compare_rows(baseline, candidate):
    require(bool(baseline) and baseline.keys() == candidate.keys(), "paired coverage differs")
    for step, a in baseline.items():
        b = candidate[step]
        require(a["global_step"] == b["global_step"] == step, "step identity differs")
        keys = {"train/dataset", "train/pairs", "system/world_size", "train/lr_factor",
                "train/boundary_multiplier", "train/boundary_uses_dense_instance_gt",
                "train/boundary_pair_mean_fraction", "train/boundary_valid_fraction"}
        keys |= {k for k in a.keys() | b.keys() if k.startswith(("sampling/", "train/clips_seen"))}
        require(all(k in a and k in b and a[k] == b[k] for k in keys), "fixed fingerprint differs")
        require(a["train/lr_factor"] == 1.0 and a["system/world_size"] == 2
                and a["train/pairs"] == 120 and a["train/boundary_multiplier"] == 2.0,
                "require full-LR world2 B2/A2/K15 boundary2x")
        require(type(a["train/dataset"]) is int and 0 <= a["train/dataset"] < 3, "dataset id")
        for row in (a, b):
            for metric in ("xyz_loss", "raw_epe_m", "boundary_weighted_xyz_loss", "gradient_norm"):
                finite(row["train/" + metric])
            for metric, population in STRATA.values():
                n, value = count(row["train/" + population]), finite(row["train/" + metric])
                require(n > 0 or value == 0, "nonzero EPE for empty logged population")
            require(row["train/boundary_occluded_valid_points"] <= row["train/boundary_valid_points"],
                    "occluded boundary is not a subset")
        for _, population in STRATA.values():
            require(a["train/" + population] == b["train/" + population], "fixed GT population differs")
    result = {"paired_logged_updates": len(baseline), "datasets": {}}
    for dataset_id, dataset in enumerate(DATASETS):
        steps = sorted(s for s, r in baseline.items() if r["train/dataset"] == dataset_id)
        block = {"paired_logged_batches": len(steps), "strata": {}}
        for metric in ("xyz_loss", "raw_epe_m", "gradient_norm"):
            block[metric] = paired_mean([baseline[s]["train/" + metric] for s in steps],
                                        [candidate[s]["train/" + metric] for s in steps])
        for name, (metric, population) in STRATA.items():
            eligible = [s for s in steps if baseline[s]["train/" + population] > 0]
            counts = [baseline[s]["train/" + population] for s in eligible]
            av = [baseline[s]["train/" + metric] for s in eligible]
            bv = [candidate[s]["train/" + metric] for s in eligible]
            total = sum(counts)
            ma = math.fsum(v * n for v, n in zip(av, counts)) / total if total else None
            mb = math.fsum(v * n for v, n in zip(bv, counts)) / total if total else None
            block["strata"][name] = {"points": total, "eligible_batches": len(eligible),
                "empty_batches": len(steps) - len(eligible), "batch_macro": paired_mean(av, bv),
                "reconstructed_point_weighted": {"baseline_mean": ma, "candidate_mean": mb,
                                                 "change_pct": change(ma, mb)}}
        result["datasets"][dataset] = block
    return result


def parse_rows(data, start, end):
    rows = {}
    for line in data[:data.rfind(b"\n") + 1].decode("utf8", errors="replace").splitlines():
        if '[stdout] {"global_step":' not in line:
            continue
        row = json.loads(line.split("[stdout] ", 1)[1])
        step = int(row["global_step"])
        if "train/loss" in row and start < step <= end:
            require(step not in rows or rows[step] == row, "conflicting duplicate row")
            rows[step] = row
    return rows


def analyze(root, output, start, end):
    require(not output.exists(), "require fresh output file")
    manifest = json.loads((root / "summary.json").read_text())
    require(min(w["start"] for w in manifest["windows"]) == start + 1
            and max(w["end"] for w in manifest["windows"]) == end, "frozen window differs")
    rows = []
    for label in ("baseline", "candidate"):
        data = (root / f"{label}-console-prefix.log").read_bytes()
        require(len(data) == manifest[label]["bytes"]
                and hashlib.sha256(data).hexdigest() == manifest[label]["log_prefix_sha256"],
                "frozen prefix identity differs")
        rows.append(parse_rows(data, start, end))
    require(max(rows[0]) == max(rows[1]) == end, "requested endpoint absent")
    report = {"baseline": manifest["baseline"], "candidate": manifest["candidate"],
              "window": [start + 1, end], "full_window": compare_rows(*rows), "halves": []}
    require(report["full_window"]["paired_logged_updates"] == manifest["paired_logged_updates"],
            "frozen coverage differs")
    middle = start + (end - start) // 2
    for lo, hi in ((start + 1, middle), (middle + 1, end)):
        subset = [{s: r for s, r in records.items() if lo <= s <= hi} for records in rows]
        report["halves"].append({"window": [lo, hi], **compare_rows(*subset)})
    report["caveat"] = __doc__
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report["full_window"], allow_nan=False), flush=True)
    print("BOUNDARY_COUNT_GATED_TRAINING_REVIEW_OK", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    args = parser.parse_args()
    analyze(args.root, args.output, args.start, args.end)
