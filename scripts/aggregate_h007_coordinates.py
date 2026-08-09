#!/usr/bin/env python3
"""Aggregate matched H007 anchor-vs-source-frame evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METRICS = (
    "pointmap", "first_frame_tracking", "first_frame_tracking_visible",
    "first_frame_tracking_occluded_valid", "arbitrary_all_st",
    "arbitrary_all_st_visible", "arbitrary_all_st_occluded_valid",
    "late_appearing", "late_appearing_visible", "late_appearing_occluded_valid",
    "tracking", "tracking_visible", "tracking_occluded_valid",
    "tracking_source_zero", "tracking_source_gt_zero",
)


def load(path: str, expected_frame: str) -> dict:
    value = json.loads(Path(path).read_text())
    if value["split"] != "validation" or value["clips"] != 147:
        raise ValueError(f"{path}: validation protocol mismatch")
    if value["pixel_stride"] != 16:
        raise ValueError(f"{path}: pixel_stride must be 16")
    if value["coordinate_frame"] != expected_frame:
        raise ValueError(f"{path}: expected coordinate_frame={expected_frame}")
    if value["motion_memory_mode"] != "learned":
        raise ValueError(f"{path}: zero-shot motion perturbation is not a matched quality eval")
    audit = value["common_anchor_audit_arbitrary_all_st"]
    native = value["arbitrary_all_st"]
    if audit["points"] != native["points"]:
        raise ValueError(f"{path}: common-anchor audit point count mismatch")
    if value["coordinate_invariance_max_abs_epe"] > 1e-3:
        raise ValueError(f"{path}: rigid coordinate audit failed: {value['coordinate_invariance_max_abs_epe']}")
    return value


def assert_matched(values: list[dict]) -> None:
    reference = values[0]
    for value in values[1:]:
        for key in ("split", "clips", "pixel_stride", "decoder_output_shape", "clean_latent_shape"):
            if value[key] != reference[key]:
                raise ValueError(f"evaluation mismatch at {key}: {reference[key]} vs {value[key]}")
        for metric in METRICS:
            if value[metric]["points"] != reference[metric]["points"]:
                raise ValueError(f"mask/pair population mismatch for {metric}")
        protocol = value["training_protocol"]
        ref_protocol = reference["training_protocol"]
        for key in ("steps", "max_clips", "pair_sampling", "num_query_pairs",
                    "backbone_readout", "wan_hidden_layers", "motion_slots",
                    "trainable_mode", "decoder_seed"):
            if protocol[key] != ref_protocol[key]:
                raise ValueError(f"training protocol mismatch at {key}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-seed2026", required=True)
    parser.add_argument("--anchor-seed2027", required=True)
    parser.add_argument("--source-seed2026", required=True)
    parser.add_argument("--source-seed2027", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    anchor = [load(args.anchor_seed2026, "anchor"), load(args.anchor_seed2027, "anchor")]
    source = [load(args.source_seed2026, "source"), load(args.source_seed2027, "source")]
    assert_matched(anchor + source)

    result = {
        "protocol": "h007_matched_source_frame_coordinate_ablation",
        "evaluation": {
            "split": "validation", "clips": 147, "pixel_stride": 16,
            "pairs_per_clip": 441, "pair_chunk": 8,
            "executable": "evaluate_dense4d.py",
            "metric": "metric-space Euclidean EPE after checkpoint-specific denormalization",
            "common_anchor_rigid_audit_max_abs_epe": max(
                value["coordinate_invariance_max_abs_epe"]
                for value in anchor + source
            ),
        },
        "runs": {
            "anchor_seed2026": args.anchor_seed2026,
            "anchor_seed2027": args.anchor_seed2027,
            "source_seed2026": args.source_seed2026,
            "source_seed2027": args.source_seed2027,
        },
        "metrics": {},
    }
    for metric in METRICS:
        anchor_values = np.asarray([value[metric]["epe"] for value in anchor], dtype=np.float64)
        source_values = np.asarray([value[metric]["epe"] for value in source], dtype=np.float64)
        anchor_mean = float(anchor_values.mean())
        source_mean = float(source_values.mean())
        result["metrics"][metric] = {
            "anchor_seed2026": float(anchor_values[0]),
            "anchor_seed2027": float(anchor_values[1]),
            "source_seed2026": float(source_values[0]),
            "source_seed2027": float(source_values[1]),
            "anchor_mean_std": [anchor_mean, float(anchor_values.std(ddof=1))],
            "source_mean_std": [source_mean, float(source_values.std(ddof=1))],
            "source_change_percent": 100.0 * (source_mean / anchor_mean - 1.0),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("H007_COORDINATE_AGGREGATE_OK", flush=True)


if __name__ == "__main__":
    main()
