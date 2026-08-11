from __future__ import annotations

import numpy as np

import worldbridge.pointodyssey as pointodyssey
from worldbridge.pointodyssey import PointOdysseyDataset, T


def test_source_anchor_requires_visibility_but_targets_only_require_validity(monkeypatch):
    """An occluded source cannot define a pixel identity; target occlusion can."""
    dataset = PointOdysseyDataset.__new__(PointOdysseyDataset)
    dataset.rows = [{"source_scene": "/unused", "start": 0}]

    tracks = 2
    annotation = {
        "trajs_2d": np.broadcast_to(
            np.array([[300.0, 100.0], [400.0, 100.0]], dtype=np.float32),
            (T, tracks, 2),
        ).copy(),
        "trajs_3d": np.ones((T, tracks, 3), dtype=np.float32),
        "valids": np.ones((T, tracks), dtype=bool),
        "visibs": np.ones((T, tracks), dtype=bool),
        "intrinsics": np.broadcast_to(np.eye(3), (T, 3, 3)).copy(),
        "extrinsics": np.broadcast_to(np.eye(4), (T, 4, 4)).copy(),
    }
    # Track 0 is valid but hidden at the source and must be rejected entirely.
    annotation["visibs"][0, 0] = False
    # Track 1 is visible at source but hidden at target 1. It remains supervised.
    annotation["visibs"][1, 1] = False
    dataset._load = lambda _row: annotation
    monkeypatch.setattr(
        pointodyssey,
        "_depth",
        lambda _path: (np.ones((128, 128), np.float32), np.ones((128, 128), bool)),
    )

    _, valid, visible = dataset.source_all_targets_with_visibility(0, source=0)
    pixels = []
    for u, v in annotation["trajs_2d"][0]:
        x = int(np.rint((u - pointodyssey.CROP_X + 0.5) * 128 / pointodyssey.CROP_SIZE - 0.5))
        y = int(np.rint((v - pointodyssey.CROP_Y + 0.5) * 128 / pointodyssey.CROP_SIZE - 0.5))
        pixels.append((y, x))

    hidden_source_pixel, visible_source_pixel = pixels
    assert not valid[:, hidden_source_pixel[0], hidden_source_pixel[1]].any()
    assert valid[:, visible_source_pixel[0], visible_source_pixel[1]].all()
    assert not visible[1, visible_source_pixel[0], visible_source_pixel[1]]
    assert valid[1, visible_source_pixel[0], visible_source_pixel[1]]
