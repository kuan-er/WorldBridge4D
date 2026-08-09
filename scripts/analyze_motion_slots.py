#!/usr/bin/env python3
"""Measure diversity and temporal identity of learned structured motion slots."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d_runtime import build_real_model, encode_clean_video_latents, precision_dtype


def _effective_rank(values: torch.Tensor) -> float:
    values = values.float()
    values = values - values.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(values)
    energy = singular.square()
    if float(energy.sum()) == 0.0:
        return 0.0
    probabilities = energy / energy.sum()
    return float(torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()))


def _clip_metrics(motion: torch.Tensor, dense: torch.Tensor) -> dict[str, float]:
    # motion [T,M,C], dense [C,T,H,W]
    motion = motion.float()
    frames, slots, channels = motion.shape
    normalized = torch.nn.functional.normalize(motion, dim=-1)
    similarities = torch.einsum("tmc,tnc->tmn", normalized, normalized)
    off_diagonal = ~torch.eye(slots, dtype=torch.bool, device=motion.device)[None]
    within_frame_cosine = similarities.masked_select(off_diagonal).mean()

    if frames > 1:
        adjacent = torch.einsum("tmc,tnc->tmn", normalized[:-1], normalized[1:])
        identity = adjacent.diagonal(dim1=-2, dim2=-1).mean()
        cross_identity = adjacent.masked_select(off_diagonal.expand(frames - 1, -1, -1)).mean()
    else:
        identity = motion.new_zeros(())
        cross_identity = motion.new_zeros(())

    global_centered = motion - motion.mean(dim=(0, 1), keepdim=True)
    slot_centered = motion - motion.mean(dim=1, keepdim=True)
    total_energy = global_centered.square().mean().clamp_min(1e-12)
    slot_variance_fraction = slot_centered.square().mean() / total_energy
    return {
        "motion_rms": float(motion.square().mean().sqrt()),
        "dense_rms": float(dense.float().square().mean().sqrt()),
        "within_frame_off_diagonal_cosine": float(within_frame_cosine),
        "adjacent_same_slot_cosine": float(identity),
        "adjacent_other_slot_cosine": float(cross_identity),
        "temporal_identity_margin": float(identity - cross_identity),
        "global_effective_rank": _effective_rank(motion.reshape(frames * slots, channels)),
        "within_frame_centered_effective_rank": _effective_rank(slot_centered.reshape(frames * slots, channels)),
        "slot_variance_fraction": float(slot_variance_fraction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train", choices=("train", "validation"))
    parser.add_argument("--dataset-offset", type=int, default=80)
    parser.add_argument("--max-clips", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.dataset_offset < 0 or args.max_clips < 1:
        raise ValueError("dataset offset must be non-negative and max clips must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = checkpoint["config"]
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    slots = int(getattr(model.backbone, "motion_slots", 0))
    if slots < 2:
        raise ValueError("slot diversity diagnostics require at least two motion slots")
    dtype = precision_dtype(config["precision"])

    dataset = MOViFDataset(
        config["data_root"], split=args.split, clip_length=int(config["clip_length"]),
        clip_start=int(config.get("clip_start", 0)),
        max_examples=args.dataset_offset + args.max_clips, seed=int(config["seed"]),
    )
    samples = [dataset[index] for index in range(args.dataset_offset, args.dataset_offset + args.max_clips)]
    latents = encode_clean_video_latents(samples, config["wan_root"], device)
    rows: list[dict[str, float]] = []
    motions: list[torch.Tensor] = []
    with torch.inference_mode():
        for latent in latents:
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                z4d = model.backbone(latent.to(device=device, dtype=dtype))
            motion = z4d.motion[0].float().cpu()
            motions.append(motion)
            rows.append(_clip_metrics(motion, z4d.dense[0].float().cpu()))

    stacked = torch.stack(motions)
    clip_centered = stacked - stacked.mean(dim=0, keepdim=True)
    clip_signal_rms = float(clip_centered.square().mean().sqrt())
    total_rms = float(stacked.square().mean().sqrt())
    keys = rows[0].keys()
    aggregate = {
        key: {"mean": float(np.mean([row[key] for row in rows])),
              "std": float(np.std([row[key] for row in rows], ddof=1)) if len(rows) > 1 else 0.0}
        for key in keys
    }
    result = {
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "split": args.split,
        "dataset_indices": [args.dataset_offset, args.dataset_offset + args.max_clips - 1],
        "clips": args.max_clips,
        "motion_slots": slots,
        "motion_shape": list(stacked.shape),
        "aggregate": aggregate,
        "clip_signal_rms": clip_signal_rms,
        "clip_signal_fraction_of_motion_rms": clip_signal_rms / max(total_rms, 1e-12),
        "per_clip": rows,
        "environment": {"torch": torch.__version__},
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("MOTION_SLOT_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
