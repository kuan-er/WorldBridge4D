"""Dataset-aware 256px inference with the corresponding formal Wan prompt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..models.factory import build_real_model, precision_dtype
from ..text_conditions import load_inference_text_condition
from ..data.constants import DATASET_NAMES
from ..data.factory import load_training_dataset


def atomic_save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASET_NAMES)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--source", type=int, required=True)
    parser.add_argument("--targets", type=int, nargs="*", default=list(range(21)))
    parser.add_argument("--target-chunk", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not 0 <= args.source < 21:
        raise ValueError("source must be in [0,20]")
    targets = [int(value) for value in args.targets]
    if not targets or len(set(targets)) != len(targets) or any(value < 0 or value >= 21 for value in targets):
        raise ValueError("targets must be unique frame indices in [0,20]")
    if args.target_chunk < 1:
        raise ValueError("target chunk must be positive")
    config = yaml.safe_load(Path(args.config).read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    checkpoint_config = checkpoint.get("config", {})
    if checkpoint_config.get("prompts") != config.get("prompts"):
        raise ValueError("checkpoint/config prompt mismatch; inference must use training prompts")
    condition, prompt_metadata = load_inference_text_condition(config, args.dataset)
    training_prompt_metadata = checkpoint.get("training_state", {}).get("prompt_metadata")
    if training_prompt_metadata is not None and training_prompt_metadata.get(args.dataset) != prompt_metadata:
        raise ValueError("checkpoint prompt metadata differs from inference condition metadata")
    dataset = load_training_dataset(config, args.dataset)
    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index {args.index} outside {args.dataset} split of size {len(dataset)}")
    device = torch.device(args.device)
    dtype = precision_dtype(config["precision"])
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    latent = torch.from_numpy(dataset.clean_latent(args.index))[None].to(device, dtype=dtype)
    condition = condition.to(device, dtype=dtype)
    source_rgb = None
    if bool(config.get("source_rgb_pyramid", False)):
        source_rgb_np = dataset.source_rgb(args.index, args.source)
        if source_rgb_np.shape != (256, 256, 3) or source_rgb_np.dtype != np.uint8:
            raise RuntimeError(
                f"source RGB must be uint8 [256,256,3], got "
                f"{source_rgb_np.dtype} {source_rgb_np.shape}"
            )
        source_rgb = torch.from_numpy(source_rgb_np)[None].permute(0, 3, 1, 2).to(
            device, dtype=dtype,
        )
        source_rgb = source_rgb / 127.5 - 1.0
    outputs = []
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == "cuda"):
        # Encode the video once with the dataset-specific prompt, then chunk
        # only target-dependent decoder work across all requested targets.
        z4d = model.backbone(latent, condition)
        source_pyramid = (
            model.decoder.encode_source_rgb(source_rgb)
            if source_rgb is not None else None
        )
        for start in range(0, len(targets), args.target_chunk):
            chunk = targets[start:start + args.target_chunk]
            source_tensor = torch.full((1, len(chunk)), args.source, device=device, dtype=torch.long)
            target_tensor = torch.tensor(chunk, device=device, dtype=torch.long)[None]
            prediction = model.decoder(
                z4d, source_tensor, target_tensor, source_pyramid=source_pyramid,
            ).normalized_xyz
            outputs.append(prediction.float().cpu())
    normalized = torch.cat(outputs, dim=1)[0]
    mean = torch.as_tensor(checkpoint["coordinate_mean"], dtype=torch.float32).reshape(1, 3, 1, 1)
    scale = torch.as_tensor(checkpoint["coordinate_scale"], dtype=torch.float32).reshape(1, 3, 1, 1)
    metric = normalized * scale + mean
    output = Path(args.output)
    atomic_save(output, {
        "normalized_xyz": normalized, "xyz_meters": metric,
        "source": args.source, "targets": targets, "dataset": args.dataset,
        "clip_index": args.index, "prompt_metadata": prompt_metadata,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    })
    summary = {
        "output": str(output.resolve()), "dataset": args.dataset, "index": args.index,
        "source": args.source, "targets": targets, "shape": list(metric.shape),
        "prompt": prompt_metadata["prompt"],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("THREE_DATASET_256_INFERENCE_OK", flush=True)
