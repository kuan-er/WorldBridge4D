from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("review", Path(__file__).resolve().parents[1] /
                                              "research/analysis/h030_compare_boundary_logs.py")
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


def row(step, count=2, epe=1.0):
    return {"global_step": step, "train/loss": 0.2, "train/dataset": 0, "train/pairs": 120,
            "system/world_size": 2, "train/lr_factor": 1.0, "train/boundary_multiplier": 2.0,
            "train/boundary_uses_dense_instance_gt": 1, "train/boundary_pair_mean_fraction": 0.2,
            "train/boundary_valid_fraction": 0.2, "sampling/source_0": 8, "train/clips_seen_total": step * 8,
            "train/xyz_loss": 0.1, "train/raw_epe_m": epe, "train/boundary_weighted_xyz_loss": 0.15,
            "train/gradient_norm": 3.0, "train/boundary_raw_epe_m": epe,
            "train/boundary_valid_points": count, "train/nonboundary_raw_epe_m": 0.25,
            "train/nonboundary_valid_points": 10, "train/boundary_occluded_raw_epe_m": 0.0,
            "train/boundary_occluded_valid_points": 0}


def test_macro_pointweighted_and_empty_populations_are_distinct():
    a = {1: row(1, 2, 1), 2: row(2, 6, 3)}
    b = deepcopy(a)
    for r in b.values(): r["train/boundary_raw_epe_m"] *= 2
    result = review.compare_rows(a, b)
    boundary = result["datasets"]["kubric"]["strata"]["boundary"]
    assert boundary["points"] == 8 and boundary["eligible_batches"] == 2
    assert boundary["batch_macro"]["baseline_mean"] == 2
    assert boundary["reconstructed_point_weighted"]["baseline_mean"] == 2.5
    assert boundary["reconstructed_point_weighted"]["change_pct"] == 100
    empty = result["datasets"]["kubric"]["strata"]["occluded_boundary"]
    assert empty["points"] == 0 and empty["empty_batches"] == 2
    assert all(v is None for v in empty["batch_macro"].values())
    assert all(v is None for v in empty["reconstructed_point_weighted"].values())
    assert result["datasets"]["pointodyssey"]["paired_logged_batches"] == 0


@pytest.mark.parametrize("key,value", [("train/boundary_valid_points", 3),
    ("train/nonboundary_valid_points", 9), ("train/boundary_occluded_valid_points", 1),
    ("sampling/source_0", 7), ("train/clips_seen_total", 9), ("train/lr_factor", 0.5),
    ("train/boundary_raw_epe_m", float("nan")), ("train/raw_epe_m", -0.1),
    ("train/boundary_valid_points", True), ("train/boundary_occluded_raw_epe_m", 1)])
def test_changed_counts_fingerprints_invalid_and_empty_metrics_rejected(key, value):
    a = {1: row(1)}; b = deepcopy(a); b[1][key] = value
    with pytest.raises(ValueError): review.compare_rows(a, b)


def test_coverage_and_duplicate_rows():
    a = {1: row(1)}
    with pytest.raises(ValueError): review.compare_rows(a, {2: row(2)})
    line = lambda r: ('[stdout] ' + json.dumps(r) + '\n').encode()
    assert review.parse_rows(line(row(1)) * 2, 0, 2) == a
    with pytest.raises(ValueError): review.parse_rows(line(row(1)) + line(row(1, epe=2)), 0, 2)


def test_frozen_prefix_hashes_roundtrip_and_no_overwrite(tmp_path):
    rows = [row(i) for i in range(1, 5)]
    data = ''.join('[stdout] ' + json.dumps(r) + '\n' for r in rows).encode()
    identity = {"log_prefix_sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
    manifest = {"baseline": identity, "candidate": identity, "paired_logged_updates": 4,
                "windows": [{"start": 1, "end": 4}]}
    (tmp_path / "summary.json").write_text(json.dumps(manifest))
    for label in ("baseline", "candidate"):
        (tmp_path / f"{label}-console-prefix.log").write_bytes(data)
    output = tmp_path / "review.json"
    review.analyze(tmp_path, output, 0, 4)
    result = json.loads(output.read_text())
    assert result["full_window"]["paired_logged_updates"] == 4
    assert [r["paired_logged_updates"] for r in result["halves"]] == [2, 2]
    with pytest.raises(ValueError, match="fresh"): review.analyze(tmp_path, output, 0, 4)
    (tmp_path / "candidate-console-prefix.log").write_bytes(data + b'changed')
    with pytest.raises(ValueError, match="identity"):
        review.analyze(tmp_path, tmp_path / "another.json", 0, 4)
