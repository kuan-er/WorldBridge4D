"""Paired causal evaluation of the source-RGB pyramid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from ..data.constants import DATASET_NAMES
from ..data.factory import load_training_dataset
from ..data.sampling import sample_eligible_targets, source_with_eligible_targets
from ..models.factory import build_real_model, precision_dtype
from ..text_conditions import load_inference_text_condition
from ..trainer.lazy_vae import warm_lazy_latents
from ..utils.io import atomic_json
from .metrics import masked_metrics, paired_bootstrap_ci

MODES = ("normal", "alpha_zero", "wrong_rgb")

def choose_plans(dataset, dataset_name: str, count: int, seed: int,
                 targets_per_source: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), DATASET_NAMES.index(dataset_name), 92500,
    ]))
    plans = []
    for index in rng.permutation(len(dataset)):
        source_order = rng.permutation(21)
        try:
            source, _xyz, valid = source_with_eligible_targets(
                dataset, int(index), source_order, min_targets=targets_per_source,
            )
        except ValueError:
            continue
        targets = sample_eligible_targets(valid, targets_per_source, rng)
        plans.append({
            "index": int(index), "source": int(source),
            "targets": [int(value) for value in targets],
        })
        if len(plans) == count:
            break
    if len(plans) != count:
        raise RuntimeError(
            f"{dataset_name} yielded only {len(plans)}/{count} eligible evaluation clips"
        )
    if len({plan["index"] for plan in plans}) != len(plans):
        raise RuntimeError("counterfactual evaluation plans must use distinct clips")
    return plans


def rgb_tensor(dataset, plan: dict[str, Any], device: torch.device,
               dtype: torch.dtype) -> torch.Tensor:
    value = dataset.source_rgb(plan["index"], plan["source"])
    if value.shape != (256, 256, 3) or value.dtype != np.uint8:
        raise RuntimeError(f"source RGB must be uint8 [256,256,3], got {value.dtype} {value.shape}")
    tensor = torch.from_numpy(value)[None].permute(0, 3, 1, 2).to(
        device, dtype=dtype, non_blocking=True,
    )
    return tensor / 127.5 - 1.0


def decode_targets(model, z4d: torch.Tensor, source: int, targets: list[int],
                   pyramid: dict[int, torch.Tensor], target_chunk: int,
                   device: torch.device) -> torch.Tensor:
    outputs = []
    for start in range(0, len(targets), target_chunk):
        chunk = targets[start:start + target_chunk]
        source_tensor = torch.full(
            (1, len(chunk)), int(source), device=device, dtype=torch.long,
        )
        target_tensor = torch.tensor(chunk, device=device, dtype=torch.long)[None]
        outputs.append(model.decoder(
            z4d, source_tensor, target_tensor, source_pyramid=pyramid,
        ).normalized_xyz)
    return torch.cat(outputs, dim=1)[0].float()

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES, default=list(DATASET_NAMES))
    parser.add_argument("--samples-per-dataset", type=int, default=16)
    parser.add_argument("--targets-per-source", type=int, default=19)
    parser.add_argument("--target-chunk", type=int, default=19)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.samples_per_dataset < 2:
        raise ValueError("samples-per-dataset must be at least two for wrong-RGB derangement")
    if not 1 <= args.targets_per_source <= 21:
        raise ValueError("targets-per-source must be in [1,21]")
    if args.target_chunk < 1:
        raise ValueError("target-chunk must be positive")

    config = yaml.safe_load(Path(args.config).read_text())
    if not bool(config.get("source_rgb_pyramid", False)):
        raise ValueError("counterfactual evaluation requires source_rgb_pyramid")
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", mmap=True, weights_only=True,
    )
    if checkpoint.get("config", {}).get("prompts") != config.get("prompts"):
        raise ValueError("checkpoint/config prompt mismatch")
    state = checkpoint.get("training_state", {})
    checkpoint_step = int(state.get("global_step", -1))
    if checkpoint_step < 0:
        raise ValueError("checkpoint lacks global_step")

    device = torch.device(args.device)
    dtype = precision_dtype(config["precision"])
    # Formal training uses sparse, checksum-bound lazy VAE caches. Select the
    # fixed evaluation plans first and populate only their missing latents
    # before constructing the 1.3B model, avoiding VAE/model co-residency.
    datasets = {
        name: load_training_dataset(config, name, allow_missing_latents=True)
        for name in args.datasets
    }
    plans_by_dataset = {
        name: choose_plans(
            datasets[name], name, args.samples_per_dataset, args.seed,
            args.targets_per_source,
        )
        for name in args.datasets
    }
    required = {name: [] for name in DATASET_NAMES}
    for name, plans in plans_by_dataset.items():
        required[name] = [plan["index"] for plan in plans]
    lazy_counts = warm_lazy_latents(config, datasets, required, device, rank=0)

    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    fusions = model.decoder.upsampler.source_fusions
    expected_fusions = (
        {"32", "64", "128", "256"}
        if bool(config.get("source_rgb_fusion_32", False))
        else {"64", "128", "256"}
    )
    if set(fusions) != expected_fusions:
        raise RuntimeError(f"unexpected source fusion scales: {sorted(fusions)}")
    alpha_values = {name: float(fusion.alpha.detach().float().cpu()) for name, fusion in fusions.items()}
    mean = torch.as_tensor(checkpoint["coordinate_mean"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    scale = torch.as_tensor(checkpoint["coordinate_scale"], device=device, dtype=torch.float32).view(1, 3, 1, 1)
    beta = float(config.get("smooth_l1_beta", 0.05))
    training_prompt_metadata = state.get("prompt_metadata")
    results: dict[str, Any] = {}

    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=dtype, enabled=device.type == "cuda",
    ):
        for dataset_number, dataset_name in enumerate(args.datasets):
            dataset = datasets[dataset_name]
            condition, prompt_metadata = load_inference_text_condition(config, dataset_name)
            if training_prompt_metadata is not None \
                    and training_prompt_metadata.get(dataset_name) != prompt_metadata:
                raise ValueError(f"checkpoint prompt metadata differs for {dataset_name}")
            condition = condition.to(device, dtype=dtype)
            plans = plans_by_dataset[dataset_name]
            sums = {mode: {"loss": 0.0, "epe": 0.0, "points": 0} for mode in MODES}
            samples = []
            for sample_number, plan in enumerate(plans):
                wrong_plan = plans[(sample_number + 1) % len(plans)]
                if wrong_plan["index"] == plan["index"]:
                    raise RuntimeError("wrong-RGB plan was not deranged")
                latent = torch.from_numpy(dataset.clean_latent(plan["index"]))[None].to(
                    device, dtype=dtype, non_blocking=True,
                )
                xyz, valid_np = dataset.source_all_targets(plan["index"], plan["source"])
                targets = plan["targets"]
                target = torch.from_numpy(xyz[targets]).to(device, dtype=torch.float32)
                target = (target - mean) / scale
                valid = torch.from_numpy(valid_np[targets]).to(device=device, dtype=torch.bool)
                z4d = model.backbone(latent, condition)
                normal_pyramid = model.decoder.encode_source_rgb(
                    rgb_tensor(dataset, plan, device, dtype)
                )
                wrong_pyramid = model.decoder.encode_source_rgb(
                    rgb_tensor(dataset, wrong_plan, device, dtype)
                )
                predictions = {
                    "normal": decode_targets(
                        model, z4d, plan["source"], targets, normal_pyramid,
                        args.target_chunk, device,
                    ),
                    "wrong_rgb": decode_targets(
                        model, z4d, plan["source"], targets, wrong_pyramid,
                        args.target_chunk, device,
                    ),
                }
                saved_alpha = {name: fusion.alpha.detach().clone() for name, fusion in fusions.items()}
                try:
                    for fusion in fusions.values():
                        fusion.alpha.zero_()
                    predictions["alpha_zero"] = decode_targets(
                        model, z4d, plan["source"], targets, normal_pyramid,
                        args.target_chunk, device,
                    )
                finally:
                    for name, fusion in fusions.items():
                        fusion.alpha.copy_(saved_alpha[name])
                sample_metrics: dict[str, Any] = {
                    "index": plan["index"], "source": plan["source"],
                    "targets": targets, "wrong_rgb_index": wrong_plan["index"],
                    "wrong_rgb_source": wrong_plan["source"],
                }
                for mode in MODES:
                    loss_sum, epe_sum, points = masked_metrics(
                        predictions[mode], target, valid, scale, beta,
                    )
                    sums[mode]["loss"] += loss_sum
                    sums[mode]["epe"] += epe_sum
                    sums[mode]["points"] += points
                    sample_metrics[mode] = {
                        "loss": loss_sum / (3 * points),
                        "raw_epe_m": epe_sum / points,
                    }
                normal = predictions["normal"]
                for mode in ("alpha_zero", "wrong_rgb"):
                    displacement = torch.linalg.vector_norm(
                        (normal - predictions[mode]) * scale, dim=1,
                    )
                    sample_metrics[f"normal_vs_{mode}_output_delta_m"] = float(
                        (displacement * valid).sum() / valid.sum()
                    )
                samples.append(sample_metrics)
                del latent, z4d, normal_pyramid, wrong_pyramid, predictions

            aggregate = {}
            for mode in MODES:
                points = sums[mode]["points"]
                aggregate[mode] = {
                    "loss": sums[mode]["loss"] / (3 * points),
                    "raw_epe_m": sums[mode]["epe"] / points,
                    "valid_points": points,
                }
            comparisons = {}
            for mode in ("alpha_zero", "wrong_rgb"):
                epe_deltas = [
                    sample["normal"]["raw_epe_m"] - sample[mode]["raw_epe_m"]
                    for sample in samples
                ]
                loss_deltas = [
                    sample["normal"]["loss"] - sample[mode]["loss"]
                    for sample in samples
                ]
                comparisons[f"normal_minus_{mode}"] = {
                    "aggregate_epe_delta_m": (
                        aggregate["normal"]["raw_epe_m"] - aggregate[mode]["raw_epe_m"]
                    ),
                    "aggregate_epe_percent": 100.0 * (
                        aggregate["normal"]["raw_epe_m"] / aggregate[mode]["raw_epe_m"] - 1.0
                    ),
                    "paired_sample_epe_delta_m_mean": float(np.mean(epe_deltas)),
                    "paired_sample_epe_delta_m_95ci": paired_bootstrap_ci(
                        epe_deltas, args.seed + dataset_number * 10 + (1 if mode == "alpha_zero" else 2),
                    ),
                    "paired_sample_loss_delta_mean": float(np.mean(loss_deltas)),
                    "normal_win_fraction_epe": float(np.mean(np.asarray(epe_deltas) < 0.0)),
                }
            results[dataset_name] = {
                "samples": samples, "aggregate": aggregate,
                "comparisons": comparisons, "prompt_metadata": prompt_metadata,
            }
            print(json.dumps({
                "dataset": dataset_name, "aggregate": aggregate,
                "comparisons": comparisons,
            }), flush=True)

    payload = {
        "checkpoint": str(checkpoint_path), "checkpoint_step": checkpoint_step,
        "config": str(Path(args.config).resolve()), "seed": args.seed,
        "samples_per_dataset": args.samples_per_dataset,
        "targets_per_source": args.targets_per_source,
        "target_chunk": args.target_chunk, "source_rgb_alpha": alpha_values,
        "lazy_vae": lazy_counts, "results": results,
    }
    atomic_json(Path(args.output), payload)
    print("SOURCE_RGB_COUNTERFACTUAL_256_OK", flush=True)
