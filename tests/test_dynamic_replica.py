from __future__ import annotations

import json
import numpy as np
from safetensors.numpy import save_file

import worldbridge.data.datasets.dynamic_replica as dynamic_replica
from worldbridge.data.datasets.dynamic_replica import DynamicReplicaDataset, T


def viewpoint() -> dict:
    return {
        "R": np.eye(3).tolist(),
        "T": [0.0, 0.0, 0.0],
        "focal_length": [700.0 / 360.0, 700.0 / 360.0],
        "principal_point": [0.0, 0.0],
        "intrinsics_format": "ndc_isotropic",
    }


def row() -> dict:
    return {
        "index": 0,
        "clip_id": "synthetic-left-000000",
        "frames": [{"depth": "unused.png", "viewpoint": viewpoint()} for _ in range(T)],
    }


def test_raw_root_override_does_not_mutate_preprocessing_state(tmp_path):
    split = tmp_path / "splits"; split.mkdir()
    (split / "train.jsonl").write_text(json.dumps({"clip_id": "x", "index": 0}) + "\n")
    (tmp_path / "PREPROCESSING_STATE.json").write_text(json.dumps({"raw_root": "/old"}))
    dataset = DynamicReplicaDataset(tmp_path, raw_root="/dataset/data/Dynamic_dataset/dynamic_stereo")
    assert dataset.raw_train_root == dynamic_replica.Path("/dataset/data/Dynamic_dataset/dynamic_stereo/train")


def test_camera_projection_and_half_pixel_resize_are_consistent():
    vp = viewpoint()
    world = np.array([[0.25, -0.5, 4.0], [-0.4, 0.2, 2.0]], dtype=np.float64)
    original = np.stack([
        640.0 - 700.0 * world[:, 0] / world[:, 2],
        360.0 - 700.0 * world[:, 1] / world[:, 2],
    ], axis=1)
    expected = dynamic_replica._transform_uv(original)
    protocol = world * np.array([-1.0, 1.0, -1.0])
    K = dynamic_replica._pixel_intrinsics(vp)
    projected = np.stack([
        K[0, 0] * protocol[:, 0] / (-protocol[:, 2]) + K[0, 2],
        K[1, 2] - K[1, 1] * protocol[:, 1] / (-protocol[:, 2]),
    ], axis=1)
    np.testing.assert_allclose(projected, expected, atol=1e-12)
    c2w = dynamic_replica._camera_to_world(vp)
    recovered = protocol @ c2w[:3, :3].T + c2w[:3, 3]
    np.testing.assert_allclose(recovered, world, atol=1e-12)


def test_source_visibility_anchors_track_but_target_visibility_does_not(monkeypatch):
    dataset = DynamicReplicaDataset.__new__(DynamicReplicaDataset)
    dataset.rows = [row()]
    dataset.raw_train_root = dynamic_replica.Path("/unused")
    tracks = 2
    uv = np.broadcast_to(np.array([[500.0, 300.0], [700.0, 300.0]], np.float32), (T, tracks, 2)).copy()
    world = np.broadcast_to(np.array([[0.1, 0.2, 3.0], [0.3, 0.4, 4.0]], np.float32), (T, tracks, 3)).copy()
    world[:, 1, 0] += np.arange(T, dtype=np.float32) * 0.01
    visible = np.ones((T, tracks), bool)
    visible[0, 0] = False
    visible[1, 1] = False
    dataset._load_clip = lambda _row: {"trajs_2d": uv, "trajs_3d_world": world, "visible": visible}
    monkeypatch.setattr(
        dynamic_replica,
        "_depth",
        lambda *_args: (np.ones((128, 128), np.float32), np.ones((128, 128), bool)),
    )

    xyz, valid, target_visible = dataset.source_all_targets_with_visibility(0, source=0)
    mapped = dynamic_replica._transform_uv(uv[0])
    pixels = [(int(np.rint(v)), int(np.rint(u))) for u, v in mapped]
    hidden_source, visible_source = pixels
    # Valid source depth supervises the diagonal everywhere, but a track that
    # is hidden at source cannot anchor any off-diagonal identity.
    assert valid[0, hidden_source[0], hidden_source[1]]
    assert not valid[1:, hidden_source[0], hidden_source[1]].any()
    assert valid[:, visible_source[0], visible_source[1]].all()
    assert valid[1, visible_source[0], visible_source[1]]
    assert not target_visible[1, visible_source[0], visible_source[1]]
    # Identity source viewpoint converts PyTorch3D [x,y,z] to protocol [-x,y,-z].
    np.testing.assert_allclose(
        xyz[1, :, visible_source[0], visible_source[1]],
        [-world[1, 1, 0], world[1, 1, 1], -world[1, 1, 2]],
    )
    # The diagonal is exactly the canonical resized-depth backprojection.
    assert xyz[0, 2, visible_source[0], visible_source[1]] == -1.0


def test_latent_lookup_uses_global_index_for_split_local_row(tmp_path):
    root = tmp_path
    latent_root = root / "latents" / "wan2.1_1.3b_fp32"
    latent_root.mkdir(parents=True)
    values = np.arange(4 * 16 * 6 * 16 * 16, dtype=np.float32).reshape(4, 16, 6, 16, 16)
    save_file({"latents": values}, latent_root / "shard_00000100_00004.safetensors")
    dataset = DynamicReplicaDataset.__new__(DynamicReplicaDataset)
    dataset.root = root
    dataset.rows = [{"index": 102}]
    np.testing.assert_array_equal(dataset.clean_latent(0), values[2])
