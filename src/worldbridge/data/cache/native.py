"""Isolated, write-once native Wan posterior-mean cache (never a 256 fallback)."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

CONTRACT = "wan2.1_native_rgb_posterior_mean_fp32_no_resize_no_tile_v1"
CONTRACT_DR512 = "wan2.1_dr_center_crop_lanczos_512_fp32_v1"


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def latent_shape(height: int, width: int, frames: int = 21) -> tuple[int, int, int, int]:
    # VAE /8 followed by DiT patch /2. No silent spatial or temporal padding.
    if (any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (height, width, frames))
            or frames != 21 or min(height, width) <= 0 or height % 16 or width % 16):
        raise ValueError("native route requires T21 and positive H/W divisible by16; no implicit transform")
    return 16, 6, height // 8, width // 8


def rgb_identity(rgb: np.ndarray) -> dict:
    if rgb.ndim != 4 or rgb.shape[0] != 21 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("native RGB must be uint8 [21,H,W,3]")
    latent_shape(*rgb.shape[1:3])
    return {"rgb_shape": list(rgb.shape),
            "rgb_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()}


class NativeLatentCache:
    """One safetensors publication contains tensor and all integrity metadata.

    Manifest digest binds the full source index, native shape, VAE and preprocessing.
    Reads require that digest and clip/index identity. Optional RGB identity binds a
    re-decoded input at write/replay. Training must never use a mismatched manifest.
    """
    def __init__(self, root: str | Path, dataset: str, manifest_sha256: str,
                 shape: tuple[int, ...], vae_sha256: str, contract: str = CONTRACT) -> None:
        if dataset not in {"kubric", "pointodyssey", "dynamic_replica"}:
            raise ValueError("unknown native dataset")
        if any(not re.fullmatch(r"[a-f0-9]{64}", v) for v in (manifest_sha256, vae_sha256)):
            raise ValueError("native cache requires SHA256 identities")
        if (len(shape) != 4 or tuple(shape[:2]) != (16, 6)
                or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in shape)):
            raise ValueError("invalid native latent shape")
        latent_shape(int(shape[2]) * 8, int(shape[3]) * 8)
        self.root = Path(root) / dataset / manifest_sha256
        self.shape = tuple(shape)
        self.fixed = {"contract": str(contract), "dataset": dataset,
                      "manifest_sha256": manifest_sha256, "vae_sha256": vae_sha256}

    def path(self, index: int) -> Path:
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or index < 0:
            raise ValueError("invalid native cache index")
        return self.root / f"latent_{int(index):08d}.safetensors"

    def read(self, index: int, clip_id: str, rgb: dict | None = None) -> np.ndarray:
        from safetensors import safe_open
        with safe_open(str(self.path(index)), framework="np") as handle:
            meta = handle.metadata() or {}
            required = {**self.fixed, "index": str(index), "clip_id": clip_id}
            if any(meta.get(k) != v for k, v in required.items()):
                raise ValueError("native latent identity mismatch")
            observed_rgb = json.loads(meta["rgb_identity"])
            if (observed_rgb["rgb_shape"] != [21, self.shape[2] * 8, self.shape[3] * 8, 3]
                    or not re.fullmatch(r"[a-f0-9]{64}", observed_rgb["rgb_sha256"])):
                raise ValueError("invalid native RGB identity")
            if rgb is not None and observed_rgb != rgb:
                raise ValueError("native RGB content changed")
            value = handle.get_tensor("latent")
        if value.shape != self.shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("native latent shape/dtype/finite mismatch")
        if hashlib.sha256(value.tobytes()).hexdigest() != meta["latent_sha256"]:
            raise ValueError("native latent tensor checksum mismatch")
        return value

    def write(self, index: int, clip_id: str, value: np.ndarray, rgb: dict) -> bool:
        from safetensors.numpy import save_file
        if (value.shape != self.shape or value.dtype != np.float32 or not np.isfinite(value).all()
                or rgb.get("rgb_shape") != [21, self.shape[2] * 8, self.shape[3] * 8, 3]
                or not re.fullmatch(r"[a-f0-9]{64}", rgb.get("rgb_sha256", ""))):
            raise ValueError("invalid native latent/RGB publication")
        value = np.ascontiguousarray(value)
        path = self.path(index)
        self.root.mkdir(parents=True, exist_ok=True)
        lockdir = self.root / ".locks"
        lockdir.mkdir(exist_ok=True)
        with (lockdir / f"{index}.lock").open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                previous = self.read(index, clip_id, rgb)
                if not np.array_equal(previous, value):
                    raise ValueError("refuse to overwrite another native latent")
                return False
            temporary = path.with_suffix(f".{os.getpid()}.tmp.safetensors")
            try:
                save_file({"latent": value}, str(temporary), metadata={
                    **self.fixed, "index": str(index), "clip_id": clip_id,
                    "rgb_identity": json.dumps(rgb, sort_keys=True),
                    "latent_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
                })
                with temporary.open("rb") as f:
                    os.fsync(f.fileno())
                os.replace(temporary, path)
                fd = os.open(self.root, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                temporary.unlink(missing_ok=True)
            self.read(index, clip_id, rgb)
            return True
