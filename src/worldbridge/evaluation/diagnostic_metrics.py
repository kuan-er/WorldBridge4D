"""CPU, prediction-independent populations for matched 4D diagnostics."""
from __future__ import annotations

import numpy as np
import torch

from .benchmark import align_if_possible


def edge_distance(edge: np.ndarray) -> np.ndarray:
    """Match the historical OpenCV L2/5 distance transform; empty edges are infinite."""
    import cv2
    edge = np.asarray(edge, dtype=bool)
    if edge.ndim != 2:
        raise ValueError("edge must be 2D")
    if not edge.any():
        return np.full(edge.shape, np.inf, np.float32)
    return cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, 5)


def depth_discontinuity(depth, valid, relative_threshold=0.05):
    """Historical 95k definition: mark both sides, including depth-validity changes."""
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(depth) & (depth > 0)
    if depth.ndim != 2 or depth.shape != valid.shape:
        raise ValueError("depth and valid must be matching 2D arrays")
    logs = np.zeros_like(depth)
    logs[valid] = np.log(depth[valid])
    edge = np.zeros_like(valid)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        changed = ((valid[a] & valid[b]) &
                   (np.abs(logs[a] - logs[b]) > np.log1p(relative_threshold)))
        changed |= valid[a] ^ valid[b]
        edge[a] |= changed
        edge[b] |= changed
    return edge


def segmentation_discontinuity(segmentation):
    segmentation = np.asarray(segmentation)
    edge = np.zeros(segmentation.shape, bool)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        changed = segmentation[a] != segmentation[b]
        edge[a] |= changed
        edge[b] |= changed
    return edge


def statistics(values, mask):
    values, mask = np.asarray(values), np.asarray(mask, bool)
    selected = values[mask]
    if not np.isfinite(selected).all():
        raise ValueError("non-finite value in fixed GT metric population")
    count = int(selected.size)
    return {"count": count, "sum": float(selected.sum(dtype=np.float64)),
            "mean": float(selected.mean(dtype=np.float64)) if count else None}


def stratified_epe(prediction, target, valid, visible, source, distance, *,
                   boundary_px=2, interior_px=10, minimum_frames=5):
    prediction, target = np.asarray(prediction), np.asarray(target)
    valid, visible = np.asarray(valid, bool), np.asarray(visible, bool)
    if prediction.shape != target.shape or target.ndim != 4 or target.shape[1] != 3:
        raise ValueError("prediction and target must match [T,3,H,W]")
    if valid.shape != (len(target), *target.shape[-2:]) or visible.shape != valid.shape:
        raise ValueError("validity shape mismatch")
    if not np.isfinite(prediction).all() or not np.isfinite(target.transpose(0, 2, 3, 1)[valid]).all():
        raise ValueError("non-finite XYZ; do not silently remove failed predictions")
    error = np.linalg.norm(prediction - target, axis=1)
    gap = np.abs(np.arange(len(target)) - int(source))
    spatial = {"all": np.ones(distance.shape, bool), "boundary": distance <= boundary_px,
               "interior": distance > interior_px}
    temporal = {"all": np.ones(len(target), bool), "pointmap": gap == 0,
                "tracking": gap > 0, "gap1_4": (gap >= 1) & (gap <= 4),
                "gap5_7": (gap >= 5) & (gap <= 7), "gap8plus": gap >= 8}
    visibility = {"all": np.ones_like(valid), "visible": visible, "occluded": ~visible}
    # Source-visible pixels whose persistent trajectory was not visible at frame 0.
    # This is an operational diagnostic, not an object-level late-appearance label.
    late = (valid[source] & visible[source] & valid[0] & ~visible[0]) if source > 0 else np.zeros(distance.shape, bool)
    spatial["not_visible_at_frame0"] = late
    result = {}
    for time_name, time_mask in temporal.items():
        for space_name, space_mask in spatial.items():
            for vis_name, vis_mask in visibility.items():
                mask = valid & time_mask[:, None, None] & space_mask[None] & vis_mask
                result[f"{time_name}/{space_name}/{vis_name}"] = statistics(error, mask)
    pair_means = [statistics(error[t], valid[t])["mean"] for t in range(len(target))]
    counts = valid.sum(axis=0)
    track_error = np.where(valid, error, 0).sum(axis=0, dtype=np.float64) / counts.clip(min=1)
    track_eligible = counts >= minimum_frames
    track = {name: statistics(track_error, track_eligible & mask) for name, mask in spatial.items()}
    displacement_mask = valid & valid[source][None] & (gap > 0)[:, None, None]
    displacement_error = np.linalg.norm(
        (prediction - prediction[source:source + 1]) - (target - target[source:source + 1]), axis=1)
    displacement = {name: statistics(displacement_error, displacement_mask & mask[None])
                    for name, mask in spatial.items()}
    aligned, alignment = align_if_possible(
        torch.from_numpy(prediction), torch.from_numpy(target), torch.from_numpy(valid), enabled=True)
    aligned_error = np.linalg.norm(aligned.numpy() - target, axis=1)
    return {"groups": result, "per_target_epe_m": pair_means,
            "per_target_valid_points": valid.sum(axis=(1, 2)).tolist(),
            "track_mean_epe": track, "displacement_epe": displacement,
            "sim3_epe": statistics(aligned_error, valid), "sim3": alignment}


def cross_instance_neighbor_proxy(prediction, target, valid, segmentation, source,
                                  *, radius=4, min_separation=0.05):
    """GT-fixed nearest different-instance source pixel, NOT a true identity classifier.

    One competitor is selected by source pixel distance (ties dy/dx), never by
    prediction error. Report whether its GT target is closer than the correct
    GT target, only where both tracks are valid and separated in 3D.
    """
    height, width = segmentation.shape
    yy, xx = np.indices((height, width))
    neighbor_y, neighbor_x = yy.copy(), xx.copy()
    chosen = np.zeros((height, width), bool)
    offsets = sorted((dy * dy + dx * dx, dy, dx)
                     for dy in range(-radius, radius + 1)
                     for dx in range(-radius, radius + 1)
                     if 0 < dy * dy + dx * dx <= radius * radius)
    for _, dy, dx in offsets:
        ny, nx = yy + dy, xx + dx
        inside = (ny >= 0) & (ny < height) & (nx >= 0) & (nx < width)
        ny, nx = ny.clip(0, height - 1), nx.clip(0, width - 1)
        take = (~chosen & inside & valid[source] & valid[source, ny, nx]
                & (segmentation != segmentation[ny, nx]))
        neighbor_y[take], neighbor_x[take] = ny[take], nx[take]
        chosen |= take
    own = target.transpose(0, 2, 3, 1)
    other = own[:, neighbor_y, neighbor_x]
    pred = prediction.transpose(0, 2, 3, 1)
    axis = other - own
    separation2 = np.square(axis).sum(axis=-1)
    mask = (valid & valid[:, neighbor_y, neighbor_x] & chosen[None]
            & (separation2 >= min_separation ** 2))
    own_dist = np.linalg.norm(pred - own, axis=-1)
    other_dist = np.linalg.norm(pred - other, axis=-1)
    projection = ((pred - own) * axis).sum(axis=-1) / separation2.clip(min=1e-12)
    return {
        "definition": "nearest_source_pixel_different_instance_gt_competitor",
        "radius_px": radius, "minimum_separation_m": min_separation,
        "neighbor_closer": statistics((other_dist < own_dist).astype(float), mask),
        "toward_neighbor": statistics((projection > 0).astype(float), mask),
        "between_surfaces": statistics(((projection > 0) & (projection < 1)).astype(float), mask),
    }


def parent_balanced_indices(rows, count, seed):
    """Select without model errors or cache availability; round-robin over parents."""
    rng = np.random.default_rng(seed)
    groups = {}
    for index, row in enumerate(rows):
        key = str(row.get("parent_id", row.get("scene_id", row.get("stream", row["clip_id"]))))
        groups.setdefault(key, []).append(index)
    keys = sorted(groups)
    rng.shuffle(keys)
    for values in groups.values():
        rng.shuffle(values)
    selected = []
    round_index = 0
    while len(selected) < min(count, len(rows)):
        for key in keys:
            if round_index < len(groups[key]):
                selected.append((groups[key][round_index], key))
                if len(selected) == min(count, len(rows)):
                    break
        round_index += 1
    return selected
