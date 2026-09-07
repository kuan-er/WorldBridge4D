"""CPU metadata guard/preservation before an H030 cycle0 continuation.

PRL must gate this command on the producer's success and hash its checkpoint
before process spawn. Do not rehash here: preserve the same inode plus the
matching planning sidecar before the training subprocess starts.

For an owned, still-running producer, gate on a successful CPU preflight plus
its complete checkpoint identity. If the live sidecar has advanced, explicit
restore_planning_status may reconstruct only the three canonical planning fields
from that checkpoint, cross-checked with an independently observed historical
status. Never replace the producer's live sidecar or silently relax strict mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

RATES = {"dense_decoder": 3e-6, "source_rgb_decay": 3e-5, "source_rgb_no_decay": 3e-5}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def preserve(source: Path, destination: Path, expected_step: int, *,
             restore_planning_status: dict | None = None) -> dict:
    require(expected_step >= 150500, "handoff must follow the original warmup")
    checkpoint = source / f"checkpoint-{expected_step:07d}.pt"
    status_bytes = (source / "train_status.json").read_bytes()
    live_status_bytes = status_bytes
    status = json.loads(status_bytes)
    live_step = status["completed_steps"]
    payload = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    state, config, optimizer = payload["training_state"], payload["config"], payload["optimizer"]
    require(payload["format"] == 3, "full format3 checkpoint required")
    if restore_planning_status is not None:
        reconstructed = {"completed_steps": state["global_step"],
                         "world_size": state["world_size"], "clips_seen": state["clips_seen"]}
        require(restore_planning_status == reconstructed,
                "observed historical planning status differs from checkpoint")
        require(live_step > expected_step and status["world_size"] == state["world_size"],
                "restore requires an advanced same-world live sidecar")
        old_counts, live_counts = state["clips_seen"], status["clips_seen"]
        require(set(old_counts) == set(live_counts)
                and all(live_counts[k] >= old_counts[k] for k in old_counts)
                and sum(live_counts.values()) - sum(old_counts.values()) == 8 * (live_step - expected_step),
                "advanced B2/A2 planning counters mismatch")
        status = reconstructed
        status_bytes = (json.dumps(status, indent=2) + "\n").encode("utf-8")
    require(state["global_step"] == status["completed_steps"] == expected_step, "handoff step mismatch")
    require(state["world_size"] == status["world_size"] == len(state["rng_states"]) == 2, "two-rank RNG required")
    require(state["clips_seen"] == status["clips_seen"], "planning counters mismatch")
    require(config["precision"] == "bf16" and config["fsdp_master_precision"] == "fp32", "FP32 masters/BF16 compute required")
    require(config["cycle_reprojection_enabled"] is True and config["cycle_reprojection_weight"] == 0.0, "cycle0 input path required")
    groups = optimizer["param_groups"]
    require({g["name"] for g in groups} == set(RATES), "optimizer group mismatch")
    names = [name for group in groups for name in group["params"]]
    require(bool(names) and set(names) == set(optimizer["state"]), "full Adam state required")
    for group in groups:
        require(abs(group["lr"] - RATES[group["name"]]) < 1e-12, "full LR required")
    for name in names:
        require(payload["model"][name].dtype == torch.float32, "FP32 master lost")
        require(all(optimizer["state"][name][key].dtype == torch.float32 for key in ("exp_avg", "exp_avg_sq")), "FP32 Adam lost")
    ages = [int(optimizer["state"][name]["step"].item()) for name in names]
    require(min(ages) == expected_step - 100000 and max(ages) == expected_step - 85000, "original Adam age offsets lost")
    destination.mkdir(parents=True, exist_ok=True)
    protected = destination / checkpoint.name
    if protected.exists():
        require(os.path.samefile(checkpoint, protected), "refuse to overwrite another checkpoint")
    else:
        os.link(checkpoint, protected)
    sidecar = destination / "train_status.json"
    if sidecar.exists():
        require(json.loads(sidecar.read_bytes()) == status, "refuse to overwrite another sidecar")
    else:
        temp = destination / "train_status.json.tmp"
        temp.write_bytes(status_bytes)
        os.replace(temp, sidecar)
    report = {"event": "fp32_cycle0_checkpoint_preserved", "checkpoint": str(protected),
              "source_checkpoint": str(checkpoint), "checkpoint_step": expected_step,
              "bytes": protected.stat().st_size, "world_size": 2,
              "adam_age_range": [min(ages), max(ages)], "actual_lrs": RATES,
              "sha256_identity_field": "Run.dependency.checkpoint_checksum"}
    if restore_planning_status is not None:
        report.update(sidecar_provenance="reconstructed_from_full_checkpoint_training_state_crosschecked_with_observed_historical_status_not_copied_live_sidecar",
                      live_sidecar_step_at_read=live_step,
                      live_sidecar_sha256_at_read=hashlib.sha256(live_status_bytes).hexdigest())
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(preserve(args.source, args.destination, args.expected_step)), flush=True)


if __name__ == "__main__":
    main()
