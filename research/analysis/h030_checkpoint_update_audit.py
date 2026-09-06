"""CPU-only audit of stored trainable weights, Adam state dtype and sampled deltas."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("--before", type=Path, required=True)
parser.add_argument("--after", type=Path, required=True)
parser.add_argument("--before-sha256", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--samples-per-tensor", type=int, default=1024)
args = parser.parse_args()
assert args.samples_per_tensor > 0
assert not torch.cuda.is_initialized()
torch.set_num_threads(2)


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


before_sha = digest(args.before)
assert before_sha == args.before_sha256, "origin checkpoint checksum mismatch"
after_sha = digest(args.after)
a = torch.load(args.before, map_location="cpu", mmap=True, weights_only=False)
b = torch.load(args.after, map_location="cpu", mmap=True, weights_only=False)
names = sorted({name for group in a["optimizer"]["param_groups"] for name in group["params"]})
assert names and set(names) <= a["model"].keys() and set(names) <= b["model"].keys()
results = []
for name in names:
    old, new = a["model"][name], b["model"][name]
    assert old.shape == new.shape and old.numel() > 0
    n = min(args.samples_per_tensor, old.numel())
    indices = torch.linspace(0, old.numel() - 1, n, dtype=torch.float64).round().long()
    x, y = old.reshape(-1)[indices].float(), new.reshape(-1)[indices].float()
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    delta = y - x
    results.append({"name": name, "numel": old.numel(), "dtype_before": str(old.dtype),
                    "dtype_after": str(new.dtype), "sample_n": n,
                    "sample_changed": int(torch.count_nonzero(delta)),
                    "sample_relative_l2": float(delta.norm() / x.norm().clamp_min(1e-30)),
                    "sample_max_abs_delta": float(delta.abs().max())})


def optimizer_summary(checkpoint):
    groups = [{k: g.get(k) for k in ("name", "lr", "_base_lr", "betas", "eps", "weight_decay")}
              for g in checkpoint["optimizer"]["param_groups"]]
    dtypes, steps = Counter(), []
    for value in checkpoint["optimizer"]["state"].values():
        for key in ("exp_avg", "exp_avg_sq"):
            if key in value:
                dtypes[f"{key}:{value[key].dtype}"] += 1
        if "step" in value:
            steps.append(float(value["step"]))
    return {"groups": groups, "state_dtype_counts": dict(dtypes),
            "adam_step_range": [min(steps), max(steps)] if steps else None}


summary = {"before": str(args.before), "after": str(args.after), "before_sha256": before_sha,
           "after_sha256": after_sha, "trainable_tensor_count": len(names),
           "trainable_dtype_counts": dict(Counter(r["dtype_after"] for r in results)),
           "sample_count": sum(r["sample_n"] for r in results),
           "changed_sample_count": sum(r["sample_changed"] for r in results),
           "tensors_with_no_sampled_change": sum(r["sample_changed"] == 0 for r in results),
           "optimizer_before": optimizer_summary(a), "optimizer_after": optimizer_summary(b),
           "caveat": "Evenly spaced samples per trainable tensor, not all elements. Stored state dtype plus implementation must be checked before inferring update rounding or master-weight precision."}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({**summary, "tensors": results}, indent=2) + "\n")
print(json.dumps(summary), flush=True)
print("CHECKPOINT_UPDATE_AUDIT_OK", flush=True)
