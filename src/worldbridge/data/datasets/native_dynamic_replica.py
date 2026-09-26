"""Native512 Dynamic Replica training admission, using pre-staged RGB/latents.

The 3D track trajectories are resolution-independent (world-space), so the
canonical XYZ/validity geometry is computed on demand at 512x512 from the
existing trajectory mmap + depth/camera, while RGB and VAE latents are read
from the pre-staged 512x512 caches.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..cache.native import CONTRACT_DR512, NativeLatentCache
from ..cache.native_rgb import NativeRGBCache
from .dynamic_replica import DynamicReplicaDataset


class NativeDynamicReplicaDataset(DynamicReplicaDataset):
    image_size = 512

    def __init__(self, values):
        self.manifest = json.loads(Path(values["native_manifest"]).read_text())
        m = self.manifest
        if m["dataset"] != "dynamic_replica" or m["native_hw"] != [512, 512] or m["contract"] != CONTRACT_DR512:
            raise ValueError("native DR admission is DR512 (center-crop LANCZOS) only")
        super().__init__(
            values["cache_root"], split="train", image_size=512,
            raw_root=values["raw_root"],
            trajectory_cache_root=values.get("trajectory_cache_root"),
            depth_cache_root=values.get("depth_cache_root"),
            trajectory_mmap_root=values.get("trajectory_mmap_root"),
            trajectory_mmap_index=Path(values["cache_root"]) / "splits" / "train.jsonl",
            trajectory_mmap_complete_sha256=values.get("trajectory_mmap_complete_sha256"),
        )
        self.local_rgb = NativeRGBCache(values["native_rgb_root"], m)
        self.local_rgb.require_complete()
        self.local_latents = NativeLatentCache(
            values["native_latent_root"], "dynamic_replica", m["sha256"],
            tuple(m["latent_shape"]), m["vae_sha256"], contract=CONTRACT_DR512,
        )
        report = json.loads((self.local_latents.root / "bulk_complete.json").read_text())
        if report.get("manifest_sha256") != m["sha256"] or report.get("processed") != len(self.rows):
            raise ValueError("native DR latent corpus is incomplete")

    def rgb(self, index: int) -> np.ndarray:
        return self.local_rgb.read(int(index))[0]

    def source_rgb(self, index: int, source: int) -> np.ndarray:
        if not 0 <= int(source) < 21:
            raise ValueError("source outside21 frames")
        return self.rgb(int(index))[int(source)]

    def clean_latent(self, index: int) -> np.ndarray:
        index = int(index)
        _, identity = self.local_rgb.read(index)
        return self.local_latents.read(index, self.rows[index]["clip_id"], identity)

    def set_lazy_vae_sha256(self, value: str) -> None:
        if value != self.manifest["vae_sha256"]:
            raise ValueError("VAE identity mismatch")

    def cache_latent(self, *args):
        raise RuntimeError("native admission never generates missing training inputs")
