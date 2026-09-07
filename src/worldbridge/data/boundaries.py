"""Prediction-independent source-grid GT boundary bands; CPU NumPy only."""
from __future__ import annotations

import numpy as np


def source_boundary_band(depth, depth_valid, segmentation=None, *, radius_px=2,
                         relative_jump=0.05):
    """Union of two-sided depth/validity jumps and available dense instance edges.

    Integer-grid Euclidean dilation, not a square max-pool and not RGB edges.
    PO/DR supply depth only; do not invent dense instance labels from sparse tracks.
    The resulting source mask is shared across targets, without visibility filtering.
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.asarray(depth_valid, dtype=bool)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError('boundary depth/validity must be matching 2D maps')
    if int(radius_px) != radius_px or not 0 <= radius_px <= 8:
        raise ValueError('boundary radius must be an integer in [0,8]')
    if not np.isfinite(relative_jump) or relative_jump <= 0:
        raise ValueError('boundary relative depth jump must be finite and positive')
    valid = valid & np.isfinite(depth) & (depth > 0)
    logs = np.zeros_like(depth)
    logs[valid] = np.log(depth[valid])
    seg = None if segmentation is None else np.asarray(segmentation)
    if seg is not None and (seg.shape != depth.shape or seg.dtype.kind not in 'iu'):
        raise ValueError('dense instance labels must be integer and match source depth')
    edge = np.zeros(depth.shape, dtype=bool)
    for axis in (0, 1):
        a, b = [slice(None)] * 2, [slice(None)] * 2
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        changed = (valid[a] & valid[b] & (np.abs(logs[a] - logs[b]) > np.log1p(relative_jump)))
        changed |= valid[a] ^ valid[b]
        if seg is not None:
            changed |= seg[a] != seg[b]
        edge[a] |= changed
        edge[b] |= changed
    radius = int(radius_px)
    padded = np.pad(edge, radius)
    height, width = edge.shape
    band = np.zeros_like(edge)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius:
                band |= padded[radius + dy:radius + dy + height,
                               radius + dx:radius + dx + width]
    return band
