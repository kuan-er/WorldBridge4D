#!/usr/bin/env python3
"""Aggregate independent H005 layer-selection splits without using final validation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.layer_analysis import mrmr_layers, standardized_lower_is_better


def rank_order(values: np.ndarray) -> np.ndarray:
    """Return zero-based ascending ranks, with deterministic index tie breaking."""
    return np.argsort(np.argsort(values, kind="stable"), kind="stable")


def aggregate(records: list[dict], count: int, redundancy_weight: float) -> dict:
    if len(records) < 2:
        raise ValueError("at least two independent selection splits are required")
    layers = records[0]["layers"]
    if layers != list(range(len(layers))):
        raise ValueError("all-layer records are required")
    probes = []
    correspondences = []
    ckas = []
    split_selected = []
    for record in records:
        if record["layers"] != layers:
            raise ValueError("selection splits have different layer sets")
        probe = np.asarray(
            [row["validation_pointmap_epe"] for row in record["layer_probes"]], dtype=np.float64
        )
        correspondence = np.asarray(
            [row["all_epe_px"] for row in record["correspondence"]], dtype=np.float64
        )
        cka = np.asarray(record["cka"], dtype=np.float64)
        if cka.shape != (len(layers), len(layers)):
            raise ValueError("selection split has an invalid CKA matrix")
        probes.append(standardized_lower_is_better(probe))
        correspondences.append(standardized_lower_is_better(correspondence))
        ckas.append(cka)
        split_selected.append(record["mrmr_selected_layers"])
    probe_z = np.stack(probes)
    correspondence_z = np.stack(correspondences)
    aggregate_probe = probe_z.mean(0)
    aggregate_correspondence = correspondence_z.mean(0)
    aggregate_cka = np.stack(ckas).mean(0)
    selected = mrmr_layers(
        aggregate_probe, aggregate_correspondence, aggregate_cka,
        count=count, redundancy_weight=redundancy_weight,
    )
    probe_ranks = np.stack([rank_order(value) for value in probe_z])
    correspondence_ranks = np.stack([rank_order(value) for value in correspondence_z])
    combined_ranks = probe_ranks + correspondence_ranks
    frequency = {
        str(layer): int(sum(layer in selection for selection in split_selected))
        for layer in layers
    }
    rows = []
    for layer in layers:
        rows.append({
            "layer": layer,
            "probe_z_by_split": probe_z[:, layer].tolist(),
            "correspondence_z_by_split": correspondence_z[:, layer].tolist(),
            "mean_probe_z": float(aggregate_probe[layer]),
            "mean_correspondence_z": float(aggregate_correspondence[layer]),
            "mean_combined_rank": float(combined_ranks[:, layer].mean()),
            "probe_rank_by_split": probe_ranks[:, layer].tolist(),
            "correspondence_rank_by_split": correspondence_ranks[:, layer].tolist(),
            "split_mrmr_frequency": frequency[str(layer)],
        })
    rows.sort(key=lambda row: (row["mean_combined_rank"], row["layer"]))
    return {
        "selection_records": [record["checkpoint"] for record in records],
        "split_count": len(records),
        "layers": layers,
        "split_mrmr_selected_layers": split_selected,
        "aggregate_mrmr_selected_layers": selected,
        "aggregate_best_probe_layers": np.argsort(aggregate_probe, kind="stable").tolist(),
        "aggregate_best_correspondence_layers": np.argsort(aggregate_correspondence, kind="stable").tolist(),
        "aggregate_cka": aggregate_cka.tolist(),
        "mrmr_redundancy_weight": redundancy_weight,
        "layers_by_mean_combined_rank": rows,
        "method": "mean within-split standardized lower-is-better scores plus mean CKA",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mrmr-count", type=int, default=5)
    parser.add_argument("--mrmr-redundancy-weight", type=float, default=0.5)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="worldbridge4d")
    parser.add_argument("--wandb-entity", default="zhaigong2023-sjtu-hpc-center")
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text()) for path in args.inputs]
    result = aggregate(records, args.mrmr_count, args.mrmr_redundancy_weight)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    if args.wandb:
        import wandb
        run = wandb.init(
            project=args.wandb_project, entity=args.wandb_entity,
            group="h005-layer-visualization-selection", job_type="aggregate-selection",
            name=os.getenv("PRL_RUN_ID", "h005-layer-selection-aggregate"), config=vars(args),
        )
        run.summary["aggregate_mrmr/selected_layers"] = ",".join(map(str, result["aggregate_mrmr_selected_layers"]))
        run.summary["split_count"] = result["split_count"]
        artifact = wandb.Artifact(f"h005-layer-selection-aggregate-{run.id}", type="analysis")
        artifact.add_file(str(output))
        run.log_artifact(artifact)
        run.finish()
    print(json.dumps({
        "aggregate_mrmr_selected_layers": result["aggregate_mrmr_selected_layers"],
        "aggregate_best_probe_layers": result["aggregate_best_probe_layers"][:5],
        "aggregate_best_correspondence_layers": result["aggregate_best_correspondence_layers"][:5],
        "output": str(output),
    }, indent=2))
    print("H005_LAYER_SELECTION_AGGREGATE_OK", flush=True)


if __name__ == "__main__":
    main()
