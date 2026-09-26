"""DR 512x512 square cache: contract, resize reader, and cache round-trips."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from worldbridge.data.cache.native import CONTRACT_DR512, NativeLatentCache, json_hash
from worldbridge.data.native_inputs import TRANSFORM_DR512, validate_manifest


def test_dr512_contract_and_transform_mapping():
    assert CONTRACT_DR512 == "wan2.1_dr_center_crop_lanczos_512_fp32_v1"
    assert TRANSFORM_DR512 == "center_crop_720_lanczos_512"


def _synthetic_manifest(tmp_path):
    index = tmp_path / "train.jsonl"
    row = {"clip_id": "c0", "index": 0, "start": 0, "stride": 1,
           "frames": [{"rgb": f"clip0/{j:02d}.png"} for j in range(21)]}
    index.write_text(json.dumps(row) + "\n")
    from worldbridge.data.cache.native import file_sha256
    manifest = {
        "contract": CONTRACT_DR512, "dataset": "dynamic_replica", "split": "train",
        "seed": 1, "index_path": str(index), "index_sha256": file_sha256(index),
        "native_hw": [512, 512], "latent_shape": [16, 6, 64, 64], "frames": 21,
        "transform": TRANSFORM_DR512, "vae_checkpoint": "x", "vae_sha256": "a" * 64,
        "posterior": "mean", "compute_and_storage_dtype": "float32", "tiling": False,
        "source_file_identity_kind": "x", "sources": {}, "records": [
            {"index": 0, "source_row_index": 0, "clip_id": "c0", "row_sha256": "a" * 64,
             "paths": [str(tmp_path / f"raw/{j:02d}.png") for j in range(21)]},
        ],
    }
    manifest["sha256"] = json_hash({k: v for k, v in manifest.items() if k != "sha256"})
    return manifest


def test_validate_manifest_accepts_dr512(tmp_path):
    validate_manifest(_synthetic_manifest(tmp_path))


def test_validate_manifest_rejects_wrong_transform(tmp_path):
    m = _synthetic_manifest(tmp_path)
    m["transform"] = "identity_no_resize_crop_pad_or_temporal_resampling"
    m["sha256"] = json_hash({k: v for k, v in m.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_manifest(m)


def test_dr512_crop_resize_shape_and_region():
    # centre-crop 720x720 from 1280x720 then LANCZOS to 512x512.
    raw = np.zeros((720, 1280, 3), dtype=np.uint8)
    raw[:, 280:1000, 0] = 255  # left crop border is column 280
    im = Image.fromarray(raw)
    im = im.crop((280, 0, 1000, 720)).resize((512, 512), Image.Resampling.LANCZOS)
    out = np.asarray(im, dtype=np.uint8)
    assert out.shape == (512, 512, 3)
    assert out[0, 0, 0] > 0  # column 280 maps to output column 0


def test_native_latent_cache_dr512_roundtrip(tmp_path):
    cache = NativeLatentCache(tmp_path, "dynamic_replica", "b" * 64, (16, 6, 64, 64),
                              "c" * 64, contract=CONTRACT_DR512)
    latent = np.random.default_rng(0).standard_normal((16, 6, 64, 64)).astype(np.float32)
    rgb = {"rgb_shape": [21, 512, 512, 3], "rgb_sha256": "d" * 64}
    assert cache.write(0, "c0", latent, rgb) is True
    got = cache.read(0, "c0", rgb)
    assert np.array_equal(got, latent)
