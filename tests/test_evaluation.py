from __future__ import annotations

from pathlib import Path

import pytest
import torch

from worldbridge.evaluation.inference import DEFAULT_OUTPUT_ROOT, inference_output_path
from worldbridge.evaluation.metrics import align_sim3_to_ground_truth


def test_joint_sim3_recovers_metric_trajectory():
    torch.manual_seed(17)
    prediction = torch.randn(3, 3, 5, 4)
    angle = torch.tensor(0.43)
    c, s = torch.cos(angle), torch.sin(angle)
    rotation = torch.tensor([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    scale = 2.75
    translation = torch.tensor([1.5, -0.25, 4.0]).reshape(1, 3, 1, 1)
    target = scale * torch.einsum("ij,kjhw->kihw", rotation, prediction) + translation
    valid = torch.ones(3, 5, 4, dtype=torch.bool)
    valid[1, 0, :2] = False

    aligned, metadata = align_sim3_to_ground_truth(prediction, target, valid)

    torch.testing.assert_close(aligned, target, atol=2e-5, rtol=2e-5)
    assert metadata["enabled"] is True
    assert metadata["points"] == int(valid.sum())
    assert metadata["scale"] == pytest.approx(scale, abs=1e-6)
    assert metadata["rmse_after_m"] < 1e-6


def test_inference_output_defaults_to_persistent_directory():
    path = inference_output_path("kubric", 7, 3, list(range(21)))
    assert path == (DEFAULT_OUTPUT_ROOT / "kubric-index000007-source03-all.pt").resolve()
    assert path.is_relative_to("/data/WorldBridge4D-runs")


def test_inference_output_rejects_ephemeral_or_non_pt_paths():
    with pytest.raises(ValueError, match="persistent root"):
        inference_output_path("kubric", 0, 0, [0], output="/tmp/prediction.pt")
    with pytest.raises(ValueError, match=".pt suffix"):
        inference_output_path(
            "kubric", 0, 0, [0],
            output="/data/WorldBridge4D-runs/inference/prediction.bin",
        )
