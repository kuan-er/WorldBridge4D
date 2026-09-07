import json

import numpy as np
import pytest
import torch

from worldbridge.evaluation.diagnostic_metrics import (
    cross_instance_neighbor_proxy, depth_discontinuity, edge_distance,
    parent_balanced_indices, segmentation_discontinuity, stratified_epe,
)
from worldbridge.evaluation.diagnostics import decode, output_root
from worldbridge.models import DenseQueryDecoder, StructuredZ4D


def scene():
    yy, xx = np.indices((6, 8))
    xyz = np.stack((xx, yy, -np.ones_like(xx))).astype(np.float32)
    target = np.broadcast_to(xyz, (21, 3, 6, 8)).copy()
    valid = np.ones((21, 6, 8), bool)
    distance = np.broadcast_to(np.arange(8), (6, 8)).astype(float)
    return target, valid, distance


def test_edges_mark_both_sides_and_missing_depth():
    depth = np.ones((5, 8), np.float32)
    depth[:, 4:] = 2
    valid = np.ones_like(depth, bool)
    edge = depth_discontinuity(depth, valid)
    assert edge[:, 3:5].all() and not edge[:, :3].any()
    valid[1, 1] = False
    assert depth_discontinuity(depth, valid)[1, 2]
    np.testing.assert_array_equal(segmentation_discontinuity(depth.astype(int)), edge)
    assert np.isinf(edge_distance(np.zeros_like(edge))).all()
    assert np.all(edge_distance(edge)[edge] == 0)


def test_fixed_masks_keep_occluded_and_report_diagonal_separately():
    target, valid, distance = scene()
    visible = valid.copy()
    visible[1:] = False
    prediction = target.copy()
    prediction[1:, 0] += 2
    out = stratified_epe(prediction, target, valid, visible, 0, distance, interior_px=5)
    assert out["groups"]["pointmap/all/all"]["mean"] == 0
    assert out["groups"]["tracking/all/occluded"]["mean"] == pytest.approx(2)
    assert out["groups"]["tracking/all/visible"]["count"] == 0
    assert out["groups"]["all/all/all"]["count"] == valid.sum()
    assert out["displacement_epe"]["all"]["mean"] == pytest.approx(2)


def test_common_position_bias_cancels_in_displacement():
    target, valid, distance = scene()
    pred = target.copy()
    pred[:, 0] += 3
    out = stratified_epe(pred, target, valid, valid, 0, distance)
    assert out["groups"]["all/all/all"]["mean"] == pytest.approx(3)
    assert out["displacement_epe"]["all"]["mean"] == pytest.approx(0)
    assert out["sim3_epe"]["mean"] < 1e-5


def test_nonfinite_prediction_is_not_silently_masked_out():
    target, valid, distance = scene()
    pred = target.copy()
    pred[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        stratified_epe(pred, target, valid, valid, 0, distance)


def test_track_population_requires_minimum_valid_frames():
    target, valid, distance = scene()
    valid[1:, 0, 0] = False
    out = stratified_epe(target, target, valid, valid, 0, distance)
    assert out["track_mean_epe"]["all"]["count"] == 47
    assert out["groups"]["all/all/all"]["count"] == 21 * 48 - 20


def test_parent_balancing_is_deterministic_and_without_replacement():
    rows = [{"clip_id": f"{p}-{i}", "parent_id": p} for p in ("a", "b", "c") for i in range(8)]
    a = parent_balanced_indices(rows, 10, 20260906)
    assert a == parent_balanced_indices(rows, 10, 20260906)
    assert len({i for i, _ in a}) == 10
    assert len({p for _, p in a[:3]}) == 3


def test_neighbor_proxy_perfect_prediction_is_not_wrong_identity():
    target, valid, _ = scene()
    seg = np.zeros((6, 8), int)
    seg[:, 4:] = 1
    out = cross_instance_neighbor_proxy(target, target, valid, seg, 0)
    assert out["neighbor_closer"]["count"] > 0
    assert out["neighbor_closer"]["mean"] == 0


def test_neighbor_proxy_population_does_not_depend_on_prediction():
    target, valid, _ = scene()
    seg = np.zeros((6, 8), int)
    seg[:, 4:] = 1
    a = cross_instance_neighbor_proxy(target, target, valid, seg, 0)
    pred = target.copy()
    pred[:, 0, :, :4] += 5
    b = cross_instance_neighbor_proxy(pred, target, valid, seg, 0)
    assert a["neighbor_closer"]["count"] == b["neighbor_closer"]["count"]
    assert b["neighbor_closer"]["mean"] > 0


def test_output_root_rejects_worktree_or_scratch():
    with pytest.raises(ValueError, match="persistent"):
        output_root("/tmp/h030_diagnostics")


def test_chunked_decoder_is_equivalent_to_batched_pairs():
    torch.manual_seed(8)
    decoder = DenseQueryDecoder(num_frames=21, latent_shape=(8, 21, 4, 4), query_dim=16,
                                embedding_dim=8, num_layers=1, num_heads=4, upsample_channels=(16, 8),
                                output_size=(8, 8), structured_motion_slots=2, structured_local_queries=True)
    decoder.eval()
    z = StructuredZ4D(torch.randn(1, 8, 21, 4, 4), torch.randn(1, 21, 2, 8))
    with torch.inference_mode():
        batched = decoder(z, torch.tensor([[1, 1, 1]]), torch.tensor([[0, 1, 20]])).normalized_xyz
        pieces = torch.cat([decoder(z, torch.tensor([[1]]), torch.tensor([[t]])).normalized_xyz
                            for t in (0, 1, 20)], dim=1)
    torch.testing.assert_close(batched, pieces, atol=2e-6, rtol=2e-5)


def test_motion_populations_are_gt_fixed_and_partition_tracking():
    target, valid, distance = scene()
    target[1:, 0, 0, 0] += 0.005
    target[1:, 0, 0, 1] += 0.05
    target[1:, 0, 0, 2] += 0.2
    valid[0, 0, 3] = False
    a = stratified_epe(target, target, valid, valid, 0, distance)
    b = stratified_epe(target + 3, target, valid, valid, 0, distance)
    expected = {'motion_le_1cm': 900, 'motion_1to10cm': 20,
                'motion_gt_10cm': 20, 'motion_unavailable': 20}
    for group, count in expected.items():
        key = f'tracking/{group}/all'
        assert a['groups'][key]['count'] == b['groups'][key]['count'] == count
    assert sum(expected.values()) == a['groups']['tracking/all/all']['count']
    with pytest.raises(ValueError, match='motion thresholds'):
        stratified_epe(target, target, valid, valid, 0, distance,
                       motion_static_m=0.2, motion_large_m=0.1)


def test_diagnostic_worker_rejects_old_memory_cap_before_cuda(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from worldbridge.evaluation import diagnostics as d
    spec = {'seed': 1, 'cuda_allocator_budget_gib': 12}
    manifest = {'spec': spec, 'data_config': {}}
    manifest['protocol'] = d.json_digest({'spec': spec, 'data_config': {}})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(d, 'output_root', lambda _: tmp_path)
    monkeypatch.setattr(torch.cuda, 'device_count', lambda: pytest.fail('CUDA reached'))
    with pytest.raises(ValueError, match='without a memory cap'):
        d.run(SimpleNamespace(output_root=str(tmp_path), manifest=str(path), gate=False))


def test_prepare_filters_dataset_and_cohort_before_loading(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from worldbridge.evaluation import diagnostics as d
    selection = tmp_path / 'historical.json'
    selection.write_text('{}')
    spec = {'checkpoints': {}, 'historical_selection': str(selection)}
    train = {'dataset': 'kubric', 'split': 'train', 'cohort': 'historical_train_replay', 'index': 0, 'sources': [0]}
    val = {**train, 'split': 'validation', 'cohort': 'validation_screen', 'index': 7}
    po = {**val, 'dataset': 'pointodyssey'}
    monkeypatch.setattr(d, 'output_root', lambda _: tmp_path)
    monkeypatch.setattr(d, 'protocol_config', lambda _: (spec, {}))
    monkeypatch.setattr(d, 'planned_clips', lambda *_: [train, val, po])
    monkeypatch.setattr(d, 'load_dataset', lambda *a, **kw: object())
    monkeypatch.setattr(d, 'prepare_clip', lambda item, *a: item)
    d.prepare(SimpleNamespace(output_root=str(tmp_path), protocol='unused',
                             datasets=['kubric'], cohorts=['validation_screen'],
                             limit_clips=None, manifest_name='manifest.json'))
    manifest = json.loads((tmp_path / 'manifest.json').read_text())
    assert manifest['planned'] == manifest['ready'] == [val]
    assert not manifest['limited_gate_manifest']

