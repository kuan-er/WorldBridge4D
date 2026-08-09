"""Diagnostics and selection helpers for Wan hidden-state geometry layers."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def native_frame_indices(num_frames: int, native_frames: int) -> np.ndarray:
    """Map native causal latent positions to nearest physical-frame indices."""
    if num_frames < 1 or native_frames < 1:
        raise ValueError("frame counts must be positive")
    values = np.rint(np.linspace(0, num_frames - 1, native_frames)).astype(np.int64)
    if np.any(np.diff(values) <= 0) and native_frames > 1:
        raise ValueError("native frames cannot map uniquely to the requested physical frames")
    return values


def token_center_pixels(height: int, width: int, grid_height: int, grid_width: int) -> np.ndarray:
    """Return integer image pixels nearest each regular token-cell center, in row-major order."""
    if min(height, width, grid_height, grid_width) < 1:
        raise ValueError("image and grid dimensions must be positive")
    u = (np.arange(grid_width, dtype=np.float64) + 0.5) * width / grid_width - 0.5
    v = (np.arange(grid_height, dtype=np.float64) + 0.5) * height / grid_height - 0.5
    vv, uu = np.meshgrid(v, u, indexing="ij")
    centers = np.stack((np.rint(uu), np.rint(vv)), axis=-1).astype(np.int64).reshape(-1, 2)
    centers[:, 0] = centers[:, 0].clip(0, width - 1)
    centers[:, 1] = centers[:, 1].clip(0, height - 1)
    return centers


def centered_gram(features: np.ndarray) -> np.ndarray:
    """Return a double-centered linear Gram matrix in float64 for stable CKA."""
    value = np.asarray(features, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] < 2:
        raise ValueError("features must be [samples,channels] with at least two samples")
    gram = value @ value.T
    return gram - gram.mean(0, keepdims=True) - gram.mean(1, keepdims=True) + gram.mean()


def linear_cka_matrix(layer_features: np.ndarray) -> np.ndarray:
    """Compute pairwise linear CKA for ``[layers,samples,channels]`` features."""
    value = np.asarray(layer_features)
    if value.ndim != 3:
        raise ValueError("layer_features must be [layers,samples,channels]")
    grams = np.stack([centered_gram(layer) for layer in value], axis=0)
    flat = grams.reshape(grams.shape[0], -1)
    norms = np.linalg.norm(flat, axis=1).clip(1e-12)
    result = (flat @ flat.T) / (norms[:, None] * norms[None, :])
    return np.clip(result, -1.0, 1.0)


def robust_rgb(components: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Independently percentile-normalize three PCA components to display RGB."""
    value = np.asarray(components, dtype=np.float32)
    if value.shape[-1] != 3:
        raise ValueError("PCA components must end in three channels")
    flat = value.reshape(-1, 3)
    lo = np.percentile(flat, lower, axis=0)
    hi = np.percentile(flat, upper, axis=0)
    scale = np.maximum(hi - lo, 1e-6)
    return np.clip((value - lo) / scale, 0.0, 1.0)


def standardized_lower_is_better(values: Sequence[float]) -> np.ndarray:
    value = np.asarray(values, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError("selection scores must be a finite vector")
    scale = value.std()
    return (value - value.mean()) / (scale if scale > 1e-12 else 1.0)


def mrmr_layers(
    probe_epe: Sequence[float],
    correspondence_epe: Sequence[float],
    cka: np.ndarray,
    count: int,
    redundancy_weight: float = 0.5,
) -> list[int]:
    """Greedy lower-is-better relevance selection with a CKA redundancy penalty."""
    probe = standardized_lower_is_better(probe_epe)
    correspondence = standardized_lower_is_better(correspondence_epe)
    relevance = probe + correspondence
    similarity = np.asarray(cka, dtype=np.float64)
    layers = len(relevance)
    if similarity.shape != (layers, layers):
        raise ValueError("CKA shape does not match score vectors")
    if not 1 <= count <= layers:
        raise ValueError("selection count is outside available layers")
    selected: list[int] = []
    available = set(range(layers))
    while len(selected) < count:
        def objective(index: int) -> tuple[float, int]:
            redundancy = max((similarity[index, other] for other in selected), default=0.0)
            return float(relevance[index] + redundancy_weight * redundancy), index
        winner = min(available, key=objective)
        selected.append(winner)
        available.remove(winner)
    return selected
