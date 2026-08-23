#!/usr/bin/env python3
"""Fail-closed preflight audit for all three 256px training caches."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from worldbridge.data import DATASET_NAMES, load_training_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples-per-dataset", type=int, default=4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    datasets = load_training_datasets(config)
    report = {"status": "pass", "resolution": 256, "datasets": {}, "failures": []}
    for dataset_id, name in enumerate(DATASET_NAMES):
        dataset = datasets[name]
        if not len(dataset):
            report["failures"].append(f"{name}: empty train split")
            continue
        indices = np.linspace(0, len(dataset) - 1, min(args.samples_per_dataset, len(dataset)), dtype=int)
        checks = []
        occluded_candidate = False
        for sample_number, index in enumerate(indices):
            source = int(np.random.default_rng([args.seed, dataset_id, sample_number]).integers(21))
            if not hasattr(dataset, "source_all_targets_with_visibility"):
                report["failures"].append(f"{name}: visibility audit API missing")
                break
            xyz, valid, visible = dataset.source_all_targets_with_visibility(int(index), source)
            latent = dataset.clean_latent(int(index))
            if xyz.shape != (21, 3, 256, 256) or valid.shape != (21, 256, 256):
                report["failures"].append(f"{name}/{index}: geometry shape {xyz.shape}/{valid.shape}")
                continue
            if latent.shape != (16, 6, 32, 32) or latent.dtype != np.float32:
                report["failures"].append(f"{name}/{index}: latent {latent.shape}/{latent.dtype}")
            if not np.isfinite(xyz[valid[:, None].repeat(3, axis=1)]).all():
                report["failures"].append(f"{name}/{index}: non-finite valid XYZ")
            diagonal_count = int(valid[source].sum())
            eligible = np.flatnonzero(valid.reshape(21, -1).any(axis=1))
            if diagonal_count == 0 or not len(eligible):
                report["failures"].append(f"{name}/{index}: no diagonal/eligible target")
            occluded_candidate |= bool((valid & ~visible).any())
            checks.append({
                "index": int(index), "source": source, "diagonal_valid": diagonal_count,
                "eligible_targets": eligible.tolist(), "latent_shape": list(latent.shape),
            })
        if not occluded_candidate:
            report["failures"].append(f"{name}: audited samples contain no occluded-valid point")
        report["datasets"][name] = {"clips": len(dataset), "samples": checks,
                                     "off_diagonal_valid_seen": occluded_candidate}
    report["status"] = "pass" if not report["failures"] else "fail"
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit("THREE_DATASET_256_AUDIT_FAILED")
    print("THREE_DATASET_256_AUDIT_OK", flush=True)


if __name__ == "__main__":
    main()
