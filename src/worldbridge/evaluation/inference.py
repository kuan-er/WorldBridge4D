"""Dataset-aware 256px inference with default dataset-GT Sim(3) alignment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from ..models.factory import build_real_model, precision_dtype
from ..data.text_conditions import load_inference_text_condition
from ..data.constants import DATASET_NAMES
from ..data.factory import load_dataset
from .metrics import align_sim3_to_ground_truth

PERSISTENT_RUN_ROOT = Path("/data/WorldBridge4D-runs")
DEFAULT_OUTPUT_ROOT = PERSISTENT_RUN_ROOT / "inference-step100000"


def inference_output_path(
    dataset: str, index: int, source: int, targets: list[int],
    *, output: str | None = None, output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> Path:
    """Resolve every inference artifact under the persistent run root."""
    if output is None:
        target_label = "all" if targets == list(range(21)) else "t" + "-".join(
            f"{target:02d}" for target in targets
        )
        path = Path(output_root) / (
            f"{dataset}-index{int(index):06d}-source{int(source):02d}-{target_label}.pt"
        )
    else:
        path = Path(output)
    path = path.expanduser().resolve(strict=False)
    persistent = PERSISTENT_RUN_ROOT.resolve(strict=False)
    try:
        path.relative_to(persistent)
    except ValueError as exc:
        raise ValueError(
            f"inference output must be under persistent root {persistent}, got {path}"
        ) from exc
    if path.suffix != ".pt":
        raise ValueError(f"inference output must use a .pt suffix, got {path}")
    return path


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
    parser.add_argument("--output")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--no-sim3", action="store_true",
        help="save metric predictions without dataset-GT Sim(3) alignment",
    )
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
    # Inference may consume the complete immutable shard tier or the audited
    # per-clip lazy tier. Missing data still fails when clean_latent() reads the
    # requested index; this flag only avoids rejecting a valid lazy-only cache.
    dataset = load_dataset(
        config, args.dataset, split="train", allow_missing_latents=True,
    )
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
    metric_raw = normalized * scale + mean
    if args.no_sim3:
        metric = metric_raw
        sim3: dict[str, object] = {
            "enabled": False,
            "method": "none",
            "warning": "raw metric prediction; no dataset-ground-truth alignment",
        }
    else:
        target_xyz, target_valid = dataset.source_all_targets(args.index, args.source)
        target_xyz = torch.from_numpy(target_xyz[targets]).float()
        target_valid = torch.from_numpy(target_valid[targets]).bool()
        metric, sim3 = align_sim3_to_ground_truth(metric_raw, target_xyz, target_valid)
    output = inference_output_path(
        args.dataset, args.index, args.source, targets,
        output=args.output, output_root=args.output_root,
    )
    atomic_save(output, {
        "normalized_xyz": normalized,
        "xyz_meters_raw": metric_raw,
        "xyz_meters": metric,
        "sim3": sim3,
        "source": args.source, "targets": targets, "dataset": args.dataset,
        "clip_index": args.index, "prompt_metadata": prompt_metadata,
        "checkpoint": str(Path(args.checkpoint).resolve()),
    })
    summary = {
        "output": str(output), "dataset": args.dataset, "index": args.index,
        "source": args.source, "targets": targets, "shape": list(metric.shape),
        "sim3": sim3, "prompt": prompt_metadata["prompt"],
    }
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("THREE_DATASET_256_INFERENCE_OK", flush=True)
