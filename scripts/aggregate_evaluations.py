#!/usr/bin/env python3
"""Aggregate per-seed evaluation JSON files into mean/std tables."""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["compact", "full"], required=True)
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    records = [json.loads(pathlib.Path(x).read_text()) for x in args.inputs]
    metrics = sorted(set().union(*(r["metrics"] for r in records)))
    aggregate = {}
    for key in metrics:
        values = [float(r["metrics"][key]) for r in records if key in r["metrics"]]
        if not values:
            continue
        aggregate[key] = {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    result = {
        "mode": args.mode,
        "seed_count": len(records),
        "inputs": args.inputs,
        "metrics_mean_std": aggregate,
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"AGGREGATION_OK: {output}", flush=True)


if __name__ == "__main__":
    main()
