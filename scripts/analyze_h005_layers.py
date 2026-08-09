#!/usr/bin/env python3
"""Visualize all Wan blocks and run held-out frozen geometry/correspondence probes."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from worldbridge.data import MOViFDataset
from worldbridge.dense4d_runtime import build_real_model, encode_clean_video_latents
from worldbridge.geometry import GeometryBuilder
from worldbridge.layer_analysis import (
    linear_cka_matrix, mrmr_layers, native_frame_indices, robust_rgb, token_center_pixels,
)


def parse_layers(text: str, total: int) -> list[int]:
    if text.strip().lower() == "all":
        return list(range(total))
    values = sorted({int(value) for value in text.split(",") if value.strip()})
    if not values or values[0] < 0 or values[-1] >= total:
        raise ValueError(f"layers must be within [0,{total - 1}]")
    return values


def probe_targets(sample, grid_shape, mean, scale):
    native_time, grid_height, grid_width = grid_shape
    frames = native_frame_indices(sample.num_frames, native_time)
    centers = token_center_pixels(sample.height, sample.width, grid_height, grid_width)
    pointmaps, valid = GeometryBuilder(sample).pointmaps()
    u, v = centers[:, 0], centers[:, 1]
    metric = np.stack([pointmaps[frame, v, u] for frame in frames], axis=0).reshape(-1, 3)
    mask = np.stack([valid[frame, v, u] for frame in frames], axis=0).reshape(-1)
    normalized = (metric - mean[None]) / scale[None]
    return normalized.astype(np.float32), metric.astype(np.float32), mask.astype(bool), frames, centers


def fit_layer_probes(features, targets, train_count, mean, scale, alpha):
    results = []
    layers = features[0].shape[0]
    for layer in range(layers):
        x_train = np.concatenate([
            features[index][layer][targets[index][2]] for index in range(train_count)
        ]).astype(np.float32)
        y_train = np.concatenate([
            targets[index][0][targets[index][2]] for index in range(train_count)
        ]).astype(np.float32)
        x_validation = np.concatenate([
            features[index][layer][targets[index][2]] for index in range(train_count, len(features))
        ]).astype(np.float32)
        y_validation_normalized = np.concatenate([
            targets[index][0][targets[index][2]] for index in range(train_count, len(features))
        ]).astype(np.float32)
        y_validation_metric = np.concatenate([
            targets[index][1][targets[index][2]] for index in range(train_count, len(features))
        ]).astype(np.float32)
        scaler = StandardScaler(copy=False)
        x_train = scaler.fit_transform(x_train)
        x_validation = scaler.transform(x_validation)
        probe = Ridge(alpha=float(alpha), solver="lsqr", tol=1e-4, max_iter=300)
        probe.fit(x_train, y_train)
        train_prediction = probe.predict(x_train) * scale[None] + mean[None]
        train_metric = y_train * scale[None] + mean[None]
        validation_prediction = probe.predict(x_validation) * scale[None] + mean[None]
        train_epe = float(np.linalg.norm(train_prediction - train_metric, axis=-1).mean())
        validation_epe = float(np.linalg.norm(validation_prediction - y_validation_metric, axis=-1).mean())
        validation_mae = float(np.abs(validation_prediction - y_validation_metric).mean())
        results.append({
            "layer": layer,
            "train_points": int(len(x_train)),
            "validation_points": int(len(x_validation)),
            "train_pointmap_epe": train_epe,
            "validation_pointmap_epe": validation_epe,
            "validation_xyz_mae": validation_mae,
            "coefficient_l2": float(np.linalg.norm(probe.coef_)),
        })
    return results


def correspondence_diagnostics(samples, features, grid_shape, device):
    layers = features[0].shape[0]
    native_time, grid_height, grid_width = grid_shape
    physical_frames = native_frame_indices(samples[0].num_frames, native_time)
    centers = token_center_pixels(samples[0].height, samples[0].width, grid_height, grid_width)
    sums = {name: np.zeros(layers, np.float64) for name in ("all", "visible", "occluded_valid")}
    counts = {name: np.zeros(layers, np.int64) for name in sums}
    correct = {name: np.zeros(layers, np.int64) for name in sums}
    for sample, cached in zip(samples, features):
        value = torch.from_numpy(cached).to(device=device, dtype=torch.float32).reshape(
            layers, native_time, grid_height * grid_width, -1
        )
        value = F.normalize(value, dim=-1)
        geometry = GeometryBuilder(sample)
        for source_native, source_frame in enumerate(physical_frames):
            trajectory, visible, valid = geometry.trajectory(int(source_frame), centers)
            for target_native, target_frame in enumerate(physical_frames):
                if source_native == target_native:
                    continue
                uv, view_depth, _ = geometry.project_trajectory(trajectory[:, int(target_frame)], int(target_frame))
                in_frame = (
                    valid[:, int(target_frame)] & np.isfinite(uv).all(axis=-1) & (view_depth > 0)
                    & (uv[:, 0] >= 0) & (uv[:, 0] < sample.width)
                    & (uv[:, 1] >= 0) & (uv[:, 1] < sample.height)
                )
                similarity = torch.einsum(
                    "lsd,ltd->lst", value[:, source_native], value[:, target_native]
                )
                prediction = similarity.argmax(dim=-1).cpu().numpy()
                prediction_uv = centers[prediction]
                error = np.linalg.norm(prediction_uv - uv[None], axis=-1)
                expected_u = np.floor(uv[:, 0] * grid_width / sample.width).astype(np.int64).clip(0, grid_width - 1)
                expected_v = np.floor(uv[:, 1] * grid_height / sample.height).astype(np.int64).clip(0, grid_height - 1)
                expected = expected_v * grid_width + expected_u
                masks = {
                    "all": in_frame,
                    "visible": in_frame & visible[:, int(target_frame)],
                    "occluded_valid": in_frame & ~visible[:, int(target_frame)],
                }
                for name, mask in masks.items():
                    if not np.any(mask):
                        continue
                    sums[name] += error[:, mask].sum(axis=1)
                    counts[name] += int(mask.sum())
                    correct[name] += (prediction[:, mask] == expected[None, mask]).sum(axis=1)
    output = []
    for layer in range(layers):
        record = {"layer": layer}
        for name in sums:
            record[f"{name}_points"] = int(counts[name][layer])
            record[f"{name}_epe_px"] = float(sums[name][layer] / max(counts[name][layer], 1))
            record[f"{name}_top1_cell_accuracy"] = float(correct[name][layer] / max(counts[name][layer], 1))
        output.append(record)
    return output


def plot_pca(features, layers, grid_shape, path):
    native_time, grid_height, grid_width = grid_shape
    figure, axes = plt.subplots(len(layers), native_time, figsize=(2.1 * native_time, 2.1 * len(layers)), squeeze=False)
    for row, layer in enumerate(layers):
        pca = PCA(n_components=3, svd_solver="randomized", random_state=0)
        components = pca.fit_transform(features[layer].astype(np.float32))
        rgb = robust_rgb(components.reshape(native_time, grid_height, grid_width, 3))
        for time_index in range(native_time):
            axes[row, time_index].imshow(rgb[time_index], interpolation="nearest")
            axes[row, time_index].set_xticks([]); axes[row, time_index].set_yticks([])
            if row == 0:
                axes[row, time_index].set_title(f"native t={time_index}")
            if time_index == 0:
                axes[row, time_index].set_ylabel(f"block {layer}")
    figure.suptitle("Wan hidden-state PCA RGB (per-block PCA)")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_cka(cka, path):
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(cka, vmin=0, vmax=1, cmap="magma")
    axis.set_xlabel("Wan block"); axis.set_ylabel("Wan block")
    axis.set_title("Linear CKA across all Wan blocks")
    figure.colorbar(image, ax=axis, fraction=0.046)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_layer_metrics(probes, correspondence, selected, path):
    layers = np.arange(len(probes))
    probe = np.array([row["validation_pointmap_epe"] for row in probes])
    matching = np.array([row["all_epe_px"] for row in correspondence])
    visible = np.array([row["visible_epe_px"] for row in correspondence])
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(layers, probe, marker="o", ms=3, label="linear pointmap probe")
    axes[0].set_ylabel("EPE (m)"); axes[0].legend(); axes[0].grid(alpha=0.25)
    axes[1].plot(layers, matching, marker="o", ms=3, label="all valid")
    axes[1].plot(layers, visible, marker="o", ms=3, label="visible")
    for axis in axes:
        for layer in selected:
            axis.axvline(layer, color="tab:green", alpha=0.18)
    axes[1].set_ylabel("Correspondence EPE (px)"); axes[1].set_xlabel("Wan block")
    axes[1].legend(); axes[1].grid(alpha=0.25)
    figure.suptitle("Held-out frozen layer diagnostics")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_correspondence_heatmaps(sample, features, layers, grid_shape, path):
    native_time, grid_height, grid_width = grid_shape
    centers = token_center_pixels(sample.height, sample.width, grid_height, grid_width)
    physical = native_frame_indices(sample.num_frames, native_time)
    source_native, target_native = 0, native_time - 1
    source_frame, target_frame = int(physical[source_native]), int(physical[target_native])
    segmentation = sample.segmentation[source_frame, centers[:, 1], centers[:, 0]]
    dynamic = np.array([
        0 < instance <= sample.num_instances and sample.instance_dynamic[instance - 1]
        for instance in segmentation
    ])
    candidates = np.flatnonzero(dynamic)
    source_token = int(candidates[0] if len(candidates) else len(centers) // 2)
    geometry = GeometryBuilder(sample)
    trajectory, _, valid = geometry.trajectory(source_frame, centers[[source_token]])
    uv, _, _ = geometry.project_trajectory(trajectory[:, target_frame], target_frame)
    figure, axes = plt.subplots(1, len(layers), figsize=(3.0 * len(layers), 3.0), squeeze=False)
    for column, layer in enumerate(layers):
        value = torch.from_numpy(features[layer].astype(np.float32)).reshape(native_time, -1, features.shape[-1])
        value = F.normalize(value, dim=-1)
        similarity = (value[source_native, source_token] @ value[target_native].T).numpy().reshape(grid_height, grid_width)
        prediction = int(similarity.argmax())
        axis = axes[0, column]
        image = axis.imshow(similarity, cmap="viridis", interpolation="nearest")
        axis.scatter([prediction % grid_width], [prediction // grid_width], marker="x", c="red", label="feature NN")
        if bool(valid[0, target_frame]):
            axis.scatter([
                uv[0, 0] * grid_width / sample.width - 0.5
            ], [uv[0, 1] * grid_height / sample.height - 0.5], marker="+", c="white", label="GT")
        axis.set_title(f"block {layer}"); axis.set_xticks([]); axis.set_yticks([])
        if column == 0:
            axis.legend(fontsize=7, loc="lower right")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(f"Cosine correspondence: physical frame {source_frame} → {target_frame}")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def motion_attention_diagnostics(model, selected_hidden, grid_shape, output_dir):
    backbone = model.backbone
    device = next(backbone.parameters()).device
    dtype = next(backbone.parameters()).dtype
    hidden = [
        torch.from_numpy(selected_hidden[index].astype(np.float32))[None].to(device=device, dtype=dtype)
        for index in backbone.hidden_layers
    ]
    projected = torch.stack([
        projection(value) for projection, value in zip(backbone.layer_projections, hidden)
    ], dim=0)
    layer_weights = backbone.layer_logits.float().softmax(0)
    fused = (projected * layer_weights.to(dtype=projected.dtype).reshape(-1, 1, 1, 1)).sum(0)
    native_time, grid_height, grid_width = grid_shape
    native = fused.reshape(1, native_time, grid_height, grid_width, backbone.geometry_dim)
    native = native + backbone.native_frame_embedding[None, :, None, None, :]
    query = (
        backbone.motion_frame_embedding[:, None] + backbone.motion_slot_embedding[None]
    ).reshape(1, backbone.num_frames * backbone.motion_slots, backbone.geometry_dim)
    with torch.inference_mode():
        _, attention = backbone.motion_attention(
            query, native.reshape(1, -1, backbone.geometry_dim), native.reshape(1, -1, backbone.geometry_dim),
            need_weights=True, average_attn_weights=False,
        )
    attention = attention.float().mean(1)[0].reshape(
        backbone.num_frames, backbone.motion_slots, native_time, grid_height, grid_width
    ).cpu().numpy()
    temporal = attention.sum(axis=(-1, -2))
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), squeeze=False)
    for slot, axis in enumerate(axes.reshape(-1)):
        image = axis.imshow(temporal[:, slot], aspect="auto", origin="lower", cmap="magma")
        axis.set_title(f"slot {slot}"); axis.set_xlabel("native time"); axis.set_ylabel("physical frame")
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle("Motion-slot temporal attention mass")
    figure.tight_layout(); figure.savefig(output_dir / "motion_slot_temporal_attention.png", dpi=180); plt.close(figure)

    frames = [0, backbone.num_frames // 2, backbone.num_frames - 1]
    figure, axes = plt.subplots(len(frames), backbone.motion_slots, figsize=(2 * backbone.motion_slots, 2 * len(frames)), squeeze=False)
    for row, frame in enumerate(frames):
        for slot in range(backbone.motion_slots):
            spatial = attention[frame, slot].sum(0)
            axes[row, slot].imshow(spatial, cmap="viridis", interpolation="nearest")
            axes[row, slot].set_xticks([]); axes[row, slot].set_yticks([])
            if row == 0: axes[row, slot].set_title(f"slot {slot}")
            if slot == 0: axes[row, slot].set_ylabel(f"frame {frame}")
    figure.suptitle("Motion-slot spatial attention (summed over native time)")
    figure.tight_layout(); figure.savefig(output_dir / "motion_slot_spatial_attention.png", dpi=180); plt.close(figure)

    probability = attention.reshape(backbone.num_frames, backbone.motion_slots, -1)
    entropy = -(probability * np.log(np.maximum(probability, 1e-12))).sum(-1) / np.log(probability.shape[-1])
    expected_native_time = (temporal * np.arange(native_time)[None, None]).sum(-1)
    return {
        "layer_weights": layer_weights.cpu().tolist(),
        "normalized_attention_entropy_mean": float(entropy.mean()),
        "normalized_attention_entropy_by_slot": entropy.mean(0).tolist(),
        "expected_native_time_by_physical_frame_and_slot": expected_native_time.tolist(),
    }


def init_wandb(args):
    if not args.wandb:
        return None
    import wandb
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity,
        group="h005-layer-visualization-selection", job_type="analysis",
        name=os.getenv("PRL_RUN_ID", "h005-layer-analysis"),
        config=vars(args), reinit="return_previous",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default="/dataset/MOVi-F")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--probe-train-clips", type=int, default=8)
    parser.add_argument("--probe-validation-clips", type=int, default=4)
    parser.add_argument(
        "--dataset-offset", type=int, default=0,
        help="start index in the train split; use a disjoint offset for layer selection",
    )
    parser.add_argument("--layers", default="all")
    parser.add_argument("--pca-layers", default="5,11,17,23,29")
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--mrmr-count", type=int, default=5)
    parser.add_argument("--mrmr-redundancy-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="worldbridge4d")
    parser.add_argument("--wandb-entity", default="zhaigong2023-sjtu-hpc-center")
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = checkpoint["config"]
    mean = np.asarray(checkpoint["coordinate_mean"], np.float32)
    scale = np.asarray(checkpoint["coordinate_scale"], np.float32)
    total_clips = args.probe_train_clips + args.probe_validation_clips
    if args.dataset_offset < 0:
        raise ValueError("dataset offset must be non-negative")
    dataset = MOViFDataset(
        args.data_root, split="train", clip_length=int(config["clip_length"]),
        clip_start=int(config.get("clip_start", 0)),
        max_examples=args.dataset_offset + total_clips, seed=args.seed,
    )
    samples = [dataset[args.dataset_offset + index] for index in range(total_clips)]
    device = torch.device(args.device)
    latents = encode_clean_video_latents(samples, config["wan_root"], device)
    model = build_real_model(config, device)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.eval()
    total_layers = len(model.backbone.mapping.dit.blocks)
    layers = parse_layers(args.layers, total_layers)
    if layers != list(range(total_layers)):
        raise ValueError("scientific all-layer comparison requires --layers all")
    pca_layers = parse_layers(args.pca_layers, total_layers)
    features = []
    targets = []
    grid_shape = None
    with torch.inference_mode():
        for latent, sample in zip(latents, samples):
            value, actual_grid = model.backbone.mapping.forward_hidden_layers(
                latent.to(device=device, dtype=next(model.parameters()).dtype),
                torch.zeros(1, device=device, dtype=next(model.parameters()).dtype), tuple(layers),
            )
            grid_shape = actual_grid if grid_shape is None else grid_shape
            if actual_grid != grid_shape:
                raise RuntimeError("Wan hidden grid changed across clips")
            features.append(torch.stack(value, dim=0)[:, 0].to(dtype=torch.float16).cpu().numpy())
            targets.append(probe_targets(sample, grid_shape, mean, scale))
    probes = fit_layer_probes(features, targets, args.probe_train_clips, mean, scale, args.ridge_alpha)
    validation_samples = samples[args.probe_train_clips:]
    validation_features = features[args.probe_train_clips:]
    correspondence = correspondence_diagnostics(validation_samples, validation_features, grid_shape, device)
    cka = np.mean([
        linear_cka_matrix(value.astype(np.float32)) for value in validation_features
    ], axis=0)
    probe_scores = [row["validation_pointmap_epe"] for row in probes]
    correspondence_scores = [row["all_epe_px"] for row in correspondence]
    selected = mrmr_layers(
        probe_scores, correspondence_scores, cka, args.mrmr_count, args.mrmr_redundancy_weight
    )

    plot_pca(validation_features[0], pca_layers, grid_shape, output_dir / "hidden_pca_rgb.png")
    plot_cka(cka, output_dir / "layer_cka.png")
    plot_layer_metrics(probes, correspondence, selected, output_dir / "layer_probe_metrics.png")
    plot_correspondence_heatmaps(
        validation_samples[0], validation_features[0], pca_layers, grid_shape,
        output_dir / "correspondence_heatmaps.png",
    )
    motion = motion_attention_diagnostics(model, validation_features[0], grid_shape, output_dir)
    layer_weights = dict(zip(model.backbone.hidden_layers, motion["layer_weights"]))
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar([str(layer) for layer in model.backbone.hidden_layers], motion["layer_weights"])
    axis.set_xlabel("Wan block"); axis.set_ylabel("softmax fusion weight")
    axis.set_title("Trained H005 layer gates")
    figure.tight_layout(); figure.savefig(output_dir / "trained_layer_weights.png", dpi=180); plt.close(figure)

    result = {
        "checkpoint": args.checkpoint,
        "data_split": "train",
        "probe_train_clips": args.probe_train_clips,
        "probe_validation_clips": args.probe_validation_clips,
        "dataset_offset": args.dataset_offset,
        "probe_train_indices": [args.dataset_offset, args.dataset_offset + args.probe_train_clips - 1],
        "probe_validation_indices": [
            args.dataset_offset + args.probe_train_clips,
            args.dataset_offset + total_clips - 1,
        ],
        "layers": layers,
        "grid_shape": list(grid_shape),
        "native_physical_frames": native_frame_indices(int(config["clip_length"]), grid_shape[0]).tolist(),
        "ridge_alpha": args.ridge_alpha,
        "layer_probes": probes,
        "correspondence": correspondence,
        "cka": cka.tolist(),
        "mrmr_selected_layers": selected,
        "mrmr_redundancy_weight": args.mrmr_redundancy_weight,
        "trained_checkpoint_layer_weights": {str(key): value for key, value in layer_weights.items()},
        "motion_attention": motion,
        "visualization_clip": validation_samples[0].video_name,
        "elapsed_seconds": time.time() - start,
        "environment": {
            "torch": torch.__version__, "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
        },
    }
    (output_dir / "layer_analysis.json").write_text(json.dumps(result, indent=2))
    run = init_wandb(args)
    if run is not None:
        import wandb
        table = wandb.Table(columns=[
            "layer", "probe_epe_m", "correspondence_epe_px", "visible_epe_px", "occluded_epe_px"
        ])
        for probe, match in zip(probes, correspondence):
            table.add_data(
                probe["layer"], probe["validation_pointmap_epe"], match["all_epe_px"],
                match["visible_epe_px"], match["occluded_valid_epe_px"],
            )
        run.log({
            "layer_metrics": table,
            "mrmr/selected_layers": ",".join(map(str, selected)),
            "diagnostics/pca": wandb.Image(str(output_dir / "hidden_pca_rgb.png")),
            "diagnostics/cka": wandb.Image(str(output_dir / "layer_cka.png")),
            "diagnostics/probes": wandb.Image(str(output_dir / "layer_probe_metrics.png")),
            "diagnostics/correspondence": wandb.Image(str(output_dir / "correspondence_heatmaps.png")),
            "diagnostics/motion_temporal": wandb.Image(str(output_dir / "motion_slot_temporal_attention.png")),
            "diagnostics/motion_spatial": wandb.Image(str(output_dir / "motion_slot_spatial_attention.png")),
        })
        artifact = wandb.Artifact(f"h005-layer-analysis-{run.id}", type="analysis")
        artifact.add_dir(str(output_dir)); run.log_artifact(artifact); run.finish()
    print(json.dumps({
        "mrmr_selected_layers": selected,
        "best_probe_layers": np.argsort(probe_scores)[:5].tolist(),
        "best_correspondence_layers": np.argsort(correspondence_scores)[:5].tolist(),
        "elapsed_seconds": result["elapsed_seconds"],
        "output": str(output_dir / "layer_analysis.json"),
    }, indent=2))
    print("H005_LAYER_ANALYSIS_OK", flush=True)


if __name__ == "__main__":
    main()
