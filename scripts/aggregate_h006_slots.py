#!/usr/bin/env python3
"""Aggregate the matched two-seed H006 motion-slot ablations."""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics


METRICS = {
    "pointmap": "pointmap",
    "first_frame": "first_frame_tracking",
    "arbitrary_all_st": "arbitrary_all_st",
    "arbitrary_occluded": "arbitrary_all_st_occluded_valid",
    "late_appearing": "late_appearing",
    "tracking": "tracking",
}


def _records(paths: list[str]) -> list[dict]:
    if len(paths) != 2:
        raise ValueError("every H006 variant requires exactly two seed records")
    return [json.loads(pathlib.Path(path).read_text()) for path in paths]


def _summarize(records: list[dict]) -> dict:
    result = {}
    for output_name, record_name in METRICS.items():
        values = [float(record[record_name]["epe"]) for record in records]
        result[output_name] = {
            "values": values,
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("baseline", "dense_only", "zero", "drop", "pair", "pair_zero_init"):
        parser.add_argument(f"--{name.replace('_', '-')}", nargs=2, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    inputs = {
        name: getattr(args, name)
        for name in ("baseline", "dense_only", "zero", "drop", "pair", "pair_zero_init")
    }
    variants = {name: _summarize(_records(paths)) for name, paths in inputs.items()}
    baseline = variants["baseline"]
    relative = {}
    for name, metrics in variants.items():
        if name == "baseline":
            continue
        relative[name] = {
            metric: 100.0 * (baseline[metric]["mean"] - value["mean"]) / baseline[metric]["mean"]
            for metric, value in metrics.items()
        }
    result = {
        "protocol": {
            "seeds": [2026, 2027],
            "train_clips": 32,
            "steps": 1536,
            "holdout_split": "train",
            "holdout_indices": [80, 111],
            "pixel_stride": 16,
            "positive_relative_percent_means_lower_epe": True,
        },
        "inputs": inputs,
        "variants": variants,
        "relative_improvement_vs_learned_slots8_percent": relative,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("H006_AGGREGATION_OK", flush=True)


if __name__ == "__main__":
    main()
