#!/usr/bin/env python3
"""Bounded exhaustive `(s,t)` XYZ evaluation for dense 4D models."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d_data import CoordinateStats, DynamicPointmapCache
from worldbridge.dense4d_runtime import build_real_model, encode_clean_video_latents, precision_dtype


class Accumulator:
    def __init__(self):
        self.points = 0
        self.epe_sum = 0.0
        self.absolute_sum = 0.0

    def add(self, prediction: np.ndarray, target: np.ndarray, mask: np.ndarray):
        mask = np.asarray(mask, bool)
        if not np.any(mask):
            return
        difference = np.asarray(prediction) - np.asarray(target)
        self.points += int(mask.sum())
        self.epe_sum += float(np.linalg.norm(difference, axis=0)[mask].sum())
        self.absolute_sum += float(np.abs(difference)[:, mask].sum())

    def result(self):
        return {
            "points": self.points,
            "epe": self.epe_sum / self.points if self.points else None,
            "xyz_mae": self.absolute_sum / (3 * self.points) if self.points else None,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-clips", type=int, default=2)
    parser.add_argument("--dataset-offset", type=int, default=0,
                        help="start index within the requested split")
    parser.add_argument("--split", default="validation", choices=("train", "validation"))
    parser.add_argument("--pair-chunk", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pixel-stride", type=int, default=1,
                        help="evaluate every Nth pixel; use 16 for the prior H001 metric protocol")
    parser.add_argument("--drop-hidden-layer", type=int,
                        help="zero-shot structured-readout diagnostic: suppress one fused Wan block")
    parser.add_argument(
        "--motion-memory-mode", default="learned", choices=("learned", "zero", "drop"),
        help="zero-shot structured-slot diagnostic; zero preserves token count, drop removes slot tokens",
    )
    args = parser.parse_args()
    if args.pixel_stride < 1:
        raise ValueError("--pixel-stride must be >= 1")
    if args.dataset_offset < 0:
        raise ValueError("--dataset-offset must be non-negative")
    if args.max_clips < 1:
        raise ValueError("--max-clips must be positive")
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = checkpoint["config"]
    stats = CoordinateStats(checkpoint["coordinate_mean"], checkpoint["coordinate_scale"])
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    if args.drop_hidden_layer is not None:
        hidden_layers = tuple(getattr(model.backbone, "hidden_layers", ()))
        if args.drop_hidden_layer not in hidden_layers:
            raise ValueError(f"cannot drop block {args.drop_hidden_layer}; checkpoint layers are {hidden_layers}")
        with torch.no_grad():
            model.backbone.layer_logits[hidden_layers.index(args.drop_hidden_layer)] = -100.0
    dtype = precision_dtype(config["precision"])

    dataset = MOViFDataset(
        config["data_root"], split=args.split, clip_length=int(config["clip_length"]),
        clip_start=int(config.get("clip_start", 0)),
        max_examples=args.dataset_offset + args.max_clips, seed=int(config["seed"]),
    )
    samples = [dataset[index] for index in range(
        args.dataset_offset, args.dataset_offset + args.max_clips
    )]
    latents = encode_clean_video_latents(samples, config["wan_root"], device)
    groups = defaultdict(Accumulator)
    gap_groups = defaultdict(Accumulator)
    cache = DynamicPointmapCache(
        max_entries=2, depth_tolerance=float(config.get("depth_tolerance", 0.05)),
        depth_relative_tolerance=float(config.get("depth_relative_tolerance", 0.01)),
    )
    forward_seconds = []
    z4d_shape = None
    z4d_motion_shape = None

    with torch.inference_mode():
        for sample, latent in zip(samples, latents):
            start = time.time()
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                z4d = model.backbone(latent.to(device=device, dtype=dtype))
            forward_seconds.append(time.time() - start)
            z4d_shape = list(z4d.shape)
            z4d_motion_shape = list(z4d.motion.shape) if hasattr(z4d, "motion") else None
            if args.motion_memory_mode != "learned":
                if not hasattr(z4d, "motion"):
                    raise ValueError("motion-memory diagnostics require a structured Z4D checkpoint")
                if args.motion_memory_mode == "zero":
                    z4d.motion = torch.zeros_like(z4d.motion)
                else:
                    z4d.include_motion = False
            for source in range(sample.num_frames):
                dynamic = cache.get(sample, source)
                ids = sample.segmentation[source]
                first_visible = np.full(ids.shape, sample.num_frames, dtype=np.int64)
                for instance_id in np.unique(ids):
                    if 0 < instance_id <= sample.num_instances:
                        seen = np.flatnonzero(sample.instance_visibility[instance_id - 1] > 0)
                        if len(seen):
                            first_visible[ids == instance_id] = int(seen[0])
                late_appearing = (
                    (first_visible > 0) & (first_visible < sample.num_frames)
                    & (source >= first_visible)
                )
                for target_start in range(0, sample.num_frames, args.pair_chunk):
                    targets = np.arange(target_start, min(target_start + args.pair_chunk, sample.num_frames))
                    sources = np.full(len(targets), source, dtype=np.int64)
                    with torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype == torch.bfloat16):
                        output = model.decoder(
                            z4d, torch.from_numpy(sources)[None].to(device),
                            torch.from_numpy(targets)[None].to(device),
                        ).normalized_xyz
                    prediction = output.float().cpu().numpy()[0]
                    prediction = prediction * stats.scale[None, :, None, None] + stats.mean[None, :, None, None]
                    for local_index, target in enumerate(targets):
                        target = int(target)
                        stride = int(args.pixel_stride)
                        truth = dynamic.xyz[target].transpose(2, 0, 1).reshape(3, -1)[:, ::stride]
                        valid = dynamic.valid[target].reshape(-1)[::stride]
                        visible = dynamic.visible[target].reshape(-1)[::stride]
                        pred = prediction[local_index].reshape(3, -1)[:, ::stride]
                        late = late_appearing.reshape(-1)[::stride]

                        groups["arbitrary_all_st"].add(pred, truth, valid)
                        groups["arbitrary_all_st_visible"].add(pred, truth, valid & visible)
                        groups["arbitrary_all_st_occluded_valid"].add(pred, truth, valid & ~visible)
                        gap_groups[abs(target - source)].add(pred, truth, valid)
                        if source == 0:
                            groups["first_frame_tracking"].add(pred, truth, valid)
                            groups["first_frame_tracking_visible"].add(pred, truth, valid & visible)
                            groups["first_frame_tracking_occluded_valid"].add(pred, truth, valid & ~visible)
                        if np.any(late):
                            groups["late_appearing"].add(pred, truth, valid & late)
                            groups["late_appearing_visible"].add(pred, truth, valid & visible & late)
                            groups["late_appearing_occluded_valid"].add(pred, truth, valid & ~visible & late)
                        if source == target:
                            groups["pointmap"].add(pred, truth, valid)
                        else:
                            groups["tracking"].add(pred, truth, valid)
                            groups["tracking_visible"].add(pred, truth, valid & visible)
                            groups["tracking_occluded_valid"].add(pred, truth, valid & ~visible)
                            groups["tracking_source_zero" if source == 0 else "tracking_source_gt_zero"].add(
                                pred, truth, valid
                            )

    result = {
        "split": args.split, "clips": len(samples), "dataset_offset": args.dataset_offset,
        "dataset_indices": [args.dataset_offset, args.dataset_offset + len(samples) - 1],
        "pixel_stride": int(args.pixel_stride),
        "checkpoint": str(pathlib.Path(args.checkpoint).resolve()),
        "drop_hidden_layer": args.drop_hidden_layer,
        "motion_memory_mode": args.motion_memory_mode,
        "effective_layer_weights": getattr(model.backbone, "layer_weights", lambda: torch.empty(0))().detach().cpu().tolist(),
        "clean_latent_shape": list(latents[0].shape), "z4d_shape": z4d_shape,
        "z4d_motion_shape": z4d_motion_shape,
        "decoder_query_shape": [
            1, args.pair_chunk, int(model.decoder.query_coordinates.shape[0]), int(config["query_dim"])
        ],
        "decoder_output_shape": [1, args.pair_chunk, 3, int(config["image_size"]), int(config["image_size"])],
        "pointmap": groups["pointmap"].result(),
        "first_frame_tracking": groups["first_frame_tracking"].result(),
        "first_frame_tracking_visible": groups["first_frame_tracking_visible"].result(),
        "first_frame_tracking_occluded_valid": groups["first_frame_tracking_occluded_valid"].result(),
        "arbitrary_all_st": groups["arbitrary_all_st"].result(),
        "arbitrary_all_st_visible": groups["arbitrary_all_st_visible"].result(),
        "arbitrary_all_st_occluded_valid": groups["arbitrary_all_st_occluded_valid"].result(),
        "late_appearing": groups["late_appearing"].result(),
        "late_appearing_visible": groups["late_appearing_visible"].result(),
        "late_appearing_occluded_valid": groups["late_appearing_occluded_valid"].result(),
        "tracking": groups["tracking"].result(),
        "tracking_visible": groups["tracking_visible"].result(),
        "tracking_occluded_valid": groups["tracking_occluded_valid"].result(),
        "tracking_source_zero": groups["tracking_source_zero"].result(),
        "tracking_source_gt_zero": groups["tracking_source_gt_zero"].result(),
        "epe_by_temporal_gap": {str(gap): gap_groups[gap].result() for gap in sorted(gap_groups)},
        "mean_wan_forward_seconds": float(np.mean(forward_seconds)),
        "flow_timestep": 0,
        "z4d_transform": getattr(model.backbone, "z4d_transform", "unknown"),
    }
    output = pathlib.Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print("DENSE4D_EVAL_OK", flush=True)


if __name__ == "__main__":
    main()
