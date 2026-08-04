"""Block-friendly 3D and target-camera reprojection metrics."""
from __future__ import annotations
from collections import defaultdict
import numpy as np


def _accumulator():
    return {"n": 0, "euclidean_sum": 0.0, "mae_sum": np.zeros(3, np.float64), "reprojection_sum": 0.0}


def update_metrics(store: dict, pred: np.ndarray, target: np.ndarray, valid: np.ndarray,
                   visible: np.ndarray, reprojection: np.ndarray | None = None,
                   group: str = "all") -> None:
    pred = np.asarray(pred); target = np.asarray(target); valid = np.asarray(valid, bool); visible = np.asarray(visible, bool)
    for suffix, mask in (("all", valid), ("visible", valid & visible), ("occluded", valid & ~visible)):
        key = f"{group}/{suffix}"
        a = store.setdefault(key, _accumulator())
        if not np.any(mask):
            continue
        diff = pred[mask] - target[mask]
        a["n"] += int(mask.sum())
        a["euclidean_sum"] += float(np.linalg.norm(diff, axis=-1).sum())
        a["mae_sum"] += np.abs(diff).sum(axis=0)
        if reprojection is not None:
            a["reprojection_sum"] += float(np.asarray(reprojection)[mask].sum())


def finalize_metrics(store: dict) -> dict[str, float | int]:
    out = {}
    for key, a in sorted(store.items()):
        n = max(1, a["n"])
        out[f"{key}/n"] = a["n"]
        out[f"{key}/endpoint_error"] = a["euclidean_sum"] / n
        out[f"{key}/mae_x"] = a["mae_sum"][0] / n
        out[f"{key}/mae_y"] = a["mae_sum"][1] / n
        out[f"{key}/mae_z"] = a["mae_sum"][2] / n
        out[f"{key}/reprojection_px"] = a["reprojection_sum"] / n
    return out
