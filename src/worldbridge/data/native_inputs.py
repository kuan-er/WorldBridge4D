"""Read exact native RGB for existing clip identities, without 256 caches or resizes."""
from __future__ import annotations

from collections import defaultdict
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .cache.native import CONTRACT, file_sha256, json_hash, latent_shape, rgb_identity
from .movif import MOViFDataset


def stat_identity(path: str | Path) -> dict:
    p = Path(path).resolve(strict=True)
    s = p.stat()
    return {"path": str(p), "bytes": s.st_size, "mtime_ns": s.st_mtime_ns}


def external_paths(dataset: str, root: str | Path, row: dict) -> list[Path]:
    if int(row.get("stride", 1)) != 1:
        raise ValueError("native cache retains the existing stride1/T21 cohort")
    root = Path(root)
    if dataset == "pointodyssey":
        scene = root / "train" / Path(row["source_scene"]).name
        return [scene / "rgbs" / f"rgb_{int(row['start']) + j:05d}.jpg" for j in range(21)]
    if dataset == "dynamic_replica":
        if len(row["frames"]) != 21:
            raise ValueError("native cache requires exactly21 frame paths")
        paths = [root / "train" / x["rgb"] for x in row["frames"]]
        if any(not p.resolve().is_relative_to((root / "train").resolve()) for p in paths):
            raise ValueError("RGB path escapes the declared dataset root")
        return paths
    raise ValueError("not an external dataset")


def build_manifest(config: dict, dataset: str, vae_sha256: str) -> dict:
    values = config["datasets"][dataset]
    index_path = Path(values["index"])
    rows = [json.loads(x) for x in index_path.read_text().splitlines() if x]
    if not rows or len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("native source index must contain unique clips")
    if [r["index"] for r in rows] != list(range(len(rows))):
        raise ValueError("native cache preserves the canonical contiguous clip index")
    shape = tuple(values["native_hw"])
    sources = {}
    records = []
    native = None
    if dataset == "kubric":
        native = MOViFDataset(values["raw_root"], split="train", clip_length=21, clip_start=0,
                             max_examples=max(r["raw_index"] for r in rows) + 1)
    for row in rows:
        record = {"index": row["index"], "clip_id": row["clip_id"], "row_sha256": json_hash(row)}
        if int(row.get("stride", 1)) != 1:
            raise ValueError("native cache only admits original stride1")
        if native is not None:
            if row.get("start", 0) != 0:
                raise ValueError("Kubric cache retains original first21 frames")
            path, local = native.records[row["raw_index"]]
            path = str(path.resolve())
            if path not in sources:
                sources[path] = stat_identity(path)
            record.update(path=path, local_record=local, raw_index=row["raw_index"])
        else:
            paths = external_paths(dataset, values["raw_root"], row)
            for path in paths:
                p = str(path.resolve())
                if p not in sources:
                    sources[p] = stat_identity(p)
            record["paths"] = [str(p.resolve()) for p in paths]
        records.append(record)
        if len(records) % 512 == 0:
            print(json.dumps({'event': 'NATIVE_MANIFEST_PROGRESS', 'dataset': dataset,
                              'clips': len(records), 'total': len(rows)}), flush=True)
    result = {"contract": CONTRACT, "dataset": dataset, "split": "train", "seed": config["seed"],
              "index_path": str(index_path), "index_sha256": file_sha256(index_path),
              "native_hw": list(shape), "latent_shape": list(latent_shape(*shape)),
              "frames": 21, "transform": "identity_no_resize_crop_pad_or_temporal_resampling",
              "vae_checkpoint": config["vae_checkpoint"], "vae_sha256": vae_sha256,
              "posterior": "mean", "compute_and_storage_dtype": "float32", "tiling": False,
              "source_file_identity_kind": "path_size_mtime_admission_plus_each_decoded_RGB_SHA256_in_cache",
              "sources": sources, "records": records}
    return {**result, "sha256": json_hash(result)}


def validate_manifest(manifest: dict) -> None:
    unsigned = {k: v for k, v in manifest.items() if k != "sha256"}
    if manifest["sha256"] != json_hash(unsigned) or manifest["contract"] != CONTRACT:
        raise ValueError("native manifest checksum/contract mismatch")
    if (manifest["latent_shape"] != list(latent_shape(*manifest["native_hw"]))
            or manifest["transform"] != "identity_no_resize_crop_pad_or_temporal_resampling"
            or manifest["frames"] != 21 or manifest["tiling"] is not False):
        raise ValueError("native manifest transform mismatch")
    if file_sha256(manifest["index_path"]) != manifest["index_sha256"]:
        raise ValueError("native source index changed")


def _check_source(manifest: dict, path: str) -> None:
    if stat_identity(path) != manifest["sources"][path]:
        raise ValueError("native RGB source file changed after admission")


def decode_kubric_rgb(raw: bytes) -> np.ndarray:
    # TF only parses Example; PNG decoding is PIL uint8 RGB without resizing.
    tf = MOViFDataset._tf()
    tf.config.set_visible_devices([], "GPU")
    ex = tf.train.Example.FromString(raw)
    values = ex.features.feature
    height = int(values["metadata/height"].int64_list.value[0])
    width = int(values["metadata/width"].int64_list.value[0])
    n = int(values["metadata/num_frames"].int64_list.value[0])
    pngs = values["video"].bytes_list.value
    if n < 21 or len(pngs) != n:
        raise ValueError("native Kubric PNG sequence contract mismatch")
    rgb = []
    for payload in pngs[:21]:
        with Image.open(io.BytesIO(payload)) as im:
            if im.size != (width, height):
                raise ValueError("native Kubric metadata/image dimensions differ")
            rgb.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
    return np.stack(rgb)


def iter_rgb(manifest: dict, indices: list[int]):
    """Read each Kubric shard once, avoiding quadratic random-access skip IO."""
    validate_manifest(manifest)
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(manifest["records"]) for i in indices):
        raise ValueError("invalid or duplicate native clip selection")
    expected = [21, *manifest["native_hw"], 3]
    if manifest["dataset"] == "kubric":
        tf = MOViFDataset._tf()
        tf.config.set_visible_devices([], "GPU")
        groups = defaultdict(dict)
        for i in indices:
            r = manifest["records"][i]
            if r["local_record"] in groups[r["path"]]:
                raise ValueError("duplicate native Kubric raw record")
            groups[r["path"]][r["local_record"]] = r
        for path, wanted in sorted(groups.items()):
            _check_source(manifest, path)
            options = tf.data.Options()
            options.threading.private_threadpool_size = 1
            options.threading.max_intra_op_parallelism = 1
            data = tf.data.TFRecordDataset([path], num_parallel_reads=1).with_options(options)
            observed = 0
            for local, raw in enumerate(data.take(max(wanted) + 1)):
                if local not in wanted:
                    continue
                rgb = decode_kubric_rgb(bytes(raw.numpy()))
                if list(rgb.shape) != expected:
                    raise ValueError("native RGB shape differs from admitted manifest")
                observed += 1
                yield wanted[local], rgb, rgb_identity(rgb)
            if observed != len(wanted):
                raise ValueError("native Kubric shard ended before requested records")
            _check_source(manifest, path)
    else:
        for i in indices:
            r = manifest["records"][i]
            frames = []
            for path in r["paths"]:
                _check_source(manifest, path)
                with Image.open(path) as im:
                    if [im.height, im.width] != manifest["native_hw"]:
                        raise ValueError("native RGB shape differs from admitted manifest")
                    frames.append(np.asarray(im.convert("RGB"), dtype=np.uint8))
                _check_source(manifest, path)
            rgb = np.stack(frames)
            if list(rgb.shape) != expected:
                raise ValueError("native RGB shape differs from admitted manifest")
            yield r, rgb, rgb_identity(rgb)
