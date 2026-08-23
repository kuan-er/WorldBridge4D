from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from worldbridge.evaluation.benchmark import aggregate, align_if_possible, validated_output_root
from worldbridge.evaluation.inference import DEFAULT_OUTPUT_ROOT, inference_output_path
from worldbridge.evaluation.metrics import align_sim3_to_ground_truth
from worldbridge.evaluation.rendering import select_error_tracks, select_worst_unique_clips


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


def test_render_selection_uses_worst_source_from_distinct_clips(tmp_path):
    clips = tmp_path / "kubric" / "clips"
    clips.mkdir(parents=True)
    for index, rows in enumerate((([1.0], [4.0]), ([3.0], [2.0]))):
        matrix = [[None] * 21 for _ in range(21)]
        for source, values in enumerate(rows):
            matrix[source][0] = values[0]
        (clips / f"clip_{index:06d}.json").write_text(json.dumps({
            "index": index, "clip_id": f"clip-{index}",
            "raw": {"source_target_mean_epe_m": matrix},
        }))
    plans = select_worst_unique_clips(tmp_path, 2)
    assert [(p["index"], p["source"]) for p in plans] == [(0, 1), (1, 0)]


def test_error_track_selection_is_spatially_separated():
    error = torch.zeros(3, 8, 8).numpy()
    valid = torch.ones(3, 8, 8, dtype=torch.bool).numpy()
    error[:, 1, 1], error[:, 1, 2], error[:, 6, 6] = 5, 4, 3
    assert select_error_tracks(error, valid, 2, 3, 3.0) == [(1, 1), (6, 6)]


def test_exhaustive_alignment_skips_sources_without_valid_points():
    prediction = torch.zeros(21, 3, 2, 2)
    target = torch.ones_like(prediction)
    valid = torch.zeros(21, 2, 2, dtype=torch.bool)
    aligned, metadata = align_if_possible(prediction, target, valid, enabled=True)
    assert aligned is prediction
    assert metadata == {
        "enabled": False,
        "method": "proper_umeyama_prediction_to_ground_truth",
        "reason": "insufficient_valid_points",
        "points": 0,
    }


def test_exhaustive_aggregate_preserves_pair_macro_and_point_weighting():
    matrix = [[None for _ in range(21)] for _ in range(21)]
    counts = [[0 for _ in range(21)] for _ in range(21)]
    matrix[0][0], counts[0][0] = 1.0, 10
    matrix[0][1], counts[0][1] = 3.0, 30
    record = {"sim3": {
        "source_target_mean_epe_m": matrix,
        "source_target_valid_points": counts,
    }}
    result = aggregate([record], "sim3")
    assert result["macro_source_target_mean_epe_m"] == pytest.approx(2.0)
    assert result["point_weighted_mean_epe_m"] == pytest.approx(2.5)
    assert result["source_target_mean_epe_m"][0][:2] == [1.0, 3.0]


def test_evaluation_output_rejects_ephemeral_root():
    assert validated_output_root(
        "/data/WorldBridge4D-runs/evaluation-step100000", "kubric"
    ).is_relative_to("/data/WorldBridge4D-runs")
    with pytest.raises(ValueError, match="evaluation output"):
        validated_output_root("/tmp/evaluation", "kubric")


def test_inference_output_rejects_ephemeral_or_non_pt_paths():
    with pytest.raises(ValueError, match="persistent root"):
        inference_output_path("kubric", 0, 0, [0], output="/tmp/prediction.pt")
    with pytest.raises(ValueError, match=".pt suffix"):
        inference_output_path(
            "kubric", 0, 0, [0],
            output="/data/WorldBridge4D-runs/inference/prediction.bin",
        )
