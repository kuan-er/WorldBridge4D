"""Exact matched-history analysis for the H031 clip=1 versus clip=5 300-update fork."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import wandb

from worldbridge.utils.io import atomic_json

ENTITY_PROJECT = "zhaigong2023-sjtu-hpc-center/worldbridge4d"
RUNS = {"clip1": "zypk1th1", "clip5": "p0ob9xcb"}
START, END = 185501, 185800
DATASETS = {0: "kubric", 1: "pointodyssey", 2: "dynamic_replica"}


def history(api: wandb.Api, run_id: str):
    run = api.run(f"{ENTITY_PROJECT}/{run_id}")
    frame = run.history(samples=20000, pandas=True)
    frame = frame[(frame.global_step >= START) & (frame.global_step <= END)].copy()
    frame = frame.sort_values("global_step").drop_duplicates("global_step").set_index("global_step")
    assert list(frame.index.astype(int)) == list(range(START, END + 1))
    return run, frame


def stats(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()), "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)), "max": float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    api = wandb.Api(timeout=60)
    run1, clip1 = history(api, RUNS["clip1"])
    run5, clip5 = history(api, RUNS["clip5"])
    assert np.array_equal(clip1["train/dataset"].astype(int), clip5["train/dataset"].astype(int))
    for key in ("train/loss", "train/raw_epe_m"):
        assert float(clip1.loc[START, key]) == float(clip5.loc[START, key])

    datasets = {}
    for dataset_id, name in DATASETS.items():
        steps = clip1.index[clip1["train/dataset"].astype(int) == dataset_id]
        item = {"updates": int(len(steps))}
        for metric, short in (
            ("train/raw_epe_m", "epe_m"), ("train/loss", "loss"),
            ("train/gradient_norm", "preclip_gradient_norm"),
        ):
            base = clip1.loc[steps, metric].astype(float).to_numpy()
            test = clip5.loc[steps, metric].astype(float).to_numpy()
            item[short] = {
                "clip1_mean": float(base.mean()), "clip5_mean": float(test.mean()),
                "clip5_relative_percent": float((test.mean() / base.mean() - 1.0) * 100.0),
                "clip5_lower_fraction": float((test < base).mean()),
                "paired_mean_delta": float((test - base).mean()),
            }
        for count in (25, 50):
            tail = steps[-count:]
            base = clip1.loc[tail, "train/raw_epe_m"].astype(float).to_numpy()
            test = clip5.loc[tail, "train/raw_epe_m"].astype(float).to_numpy()
            item[f"last_{count}_dataset_updates_epe_m"] = {
                "clip1_mean": float(base.mean()), "clip5_mean": float(test.mean()),
                "clip5_relative_percent": float((test.mean() / base.mean() - 1.0) * 100.0),
            }
        datasets[name] = item

    windows = {}
    for lower in (185501, 185601, 185701):
        upper = lower + 99
        block = {}
        for dataset_id, name in DATASETS.items():
            steps = [step for step in clip1.index if lower <= step <= upper and int(clip1.loc[step, "train/dataset"]) == dataset_id]
            base = clip1.loc[steps, "train/raw_epe_m"].astype(float).to_numpy()
            test = clip5.loc[steps, "train/raw_epe_m"].astype(float).to_numpy()
            block[name] = {
                "updates": len(steps), "clip1_mean": float(base.mean()),
                "clip5_mean": float(test.mean()),
                "clip5_relative_percent": float((test.mean() / base.mean() - 1.0) * 100.0),
            }
        windows[f"{lower}-{upper}"] = block

    clipping = {}
    for label, frame, threshold in (("clip1", clip1, 1.0), ("clip5", clip5, 5.0)):
        clipping[label] = {}
        for dataset_id, name in DATASETS.items():
            values = frame.loc[frame["train/dataset"].astype(int) == dataset_id, "train/gradient_norm"].astype(float).to_numpy()
            clipping[label][name] = {
                **stats(values), "threshold": threshold,
                "clipped_fraction": float((values > threshold).mean()),
                "mean_postclip_norm": float(np.minimum(values, threshold).mean()),
                "mean_clip_coefficient": float(np.minimum(1.0, threshold / values).mean()),
            }

    report = {
        "event": "H031_CLIP5_300_MATCHED_COMPARISON",
        "steps": [START, END], "matched_updates": END - START + 1,
        "source_checkpoint_step": 185500,
        "source_checkpoint_sha256": "c3bc7f90e3b1cb4c8fbc1c5e9b0036a967d832c8fce16dc1ddb513e2aface402",
        "runs": {
            "clip1": {"run_id": RUNS["clip1"], "url": run1.url, "state": run1.state},
            "clip5": {"run_id": RUNS["clip5"], "url": run5.url, "state": run5.state},
        },
        "initial_replay": {
            "loss_exact": float(clip1.loc[START, "train/loss"]),
            "epe_exact": float(clip1.loc[START, "train/raw_epe_m"]),
            "dataset_sequence_exact": True,
        },
        "datasets": datasets, "hundred_step_windows": windows, "clipping": clipping,
        "endpoint": {
            "dataset": DATASETS[int(clip1.loc[END, "train/dataset"])],
            "clip1_epe_m": float(clip1.loc[END, "train/raw_epe_m"]),
            "clip5_epe_m": float(clip5.loc[END, "train/raw_epe_m"]),
            "clip1_loss": float(clip1.loc[END, "train/loss"]),
            "clip5_loss": float(clip5.loc[END, "train/loss"]),
        },
        "conclusion": "clip5 is worse on all three matched training distributions over 300 updates; reject an immediate mid-run clip1-to-clip5 switch",
        "scope": "matched online training batches, not fixed held-out evaluation or a from-scratch clip5 test",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, report)
    print(report["conclusion"])


if __name__ == "__main__":
    main()
