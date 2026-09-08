"""Write-once native uint8 RGB snapshot, bound to the original latent manifest."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

import numpy as np

from .native import rgb_identity


class NativeRGBCache:
    def __init__(self, root: str | Path, manifest: dict):
        from ..native_inputs import validate_manifest
        validate_manifest(manifest)  # checks local index/hash, never opens raw RGB
        self.manifest = manifest
        self.root = Path(root) / manifest['dataset'] / manifest['sha256']
        self.shape = [21, *manifest['native_hw'], 3]

    def path(self, index: int) -> Path:
        if type(index) is not int or not 0 <= index < len(self.manifest['records']):
            raise ValueError('RGB cache index out of bounds')
        return self.root / f'rgb_{index:08d}.safetensors'

    def metadata(self, index: int) -> dict:
        row = self.manifest['records'][index]
        return {'contract': 'native_rgb_uint8_snapshot_v1',
                'manifest_sha256': self.manifest['sha256'],
                'index': str(index), 'clip_id': row['clip_id'],
                'row_sha256': row['row_sha256']}

    def read(self, index: int) -> tuple[np.ndarray, dict]:
        from safetensors import safe_open
        with safe_open(str(self.path(index)), framework='np') as f:
            meta = f.metadata() or {}
            if any(meta.get(k) != v for k, v in self.metadata(index).items()):
                raise ValueError('RGB cache identity mismatch')
            rgb = f.get_tensor('rgb')
        identity = rgb_identity(rgb)
        if list(rgb.shape) != self.shape or json.loads(meta['rgb_identity']) != identity:
            raise ValueError('RGB cache shape or checksum mismatch')
        return rgb, identity

    def write(self, index: int, rgb: np.ndarray, identity: dict) -> bool:
        from safetensors.numpy import save_file
        path = self.path(index)
        if list(rgb.shape) != self.shape or rgb_identity(rgb) != identity:
            raise ValueError('RGB publication shape/checksum mismatch')
        self.root.mkdir(parents=True, exist_ok=True)
        locks = self.root / '.locks'
        locks.mkdir(exist_ok=True)
        with (locks / f'{index}.lock').open('a+b') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if path.exists():
                old, old_identity = self.read(index)
                if old_identity != identity or not np.array_equal(old, rgb):
                    raise ValueError('refuse to overwrite different RGB snapshot')
                return False
            tmp = path.with_suffix(f'.{os.getpid()}.tmp')
            try:
                save_file({'rgb': np.ascontiguousarray(rgb)}, str(tmp), metadata={
                    **self.metadata(index), 'rgb_identity': json.dumps(identity, sort_keys=True),
                })
                with tmp.open('rb') as f:
                    os.fsync(f.fileno())
                os.replace(tmp, path)
                fd = os.open(self.root, os.O_DIRECTORY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            finally:
                tmp.unlink(missing_ok=True)
        self.read(index)
        return True

    def require_complete(self) -> None:
        report = json.loads((self.root / 'bulk_complete.json').read_text())
        n = len(self.manifest['records'])
        if (report.get('manifest_sha256') != self.manifest['sha256']
                or report.get('processed') != n or report.get('requested') != n):
            raise ValueError('local RGB snapshot is not complete')
        expected = {self.path(i).name for i in range(n)}
        if {p.name for p in self.root.glob('rgb_*.safetensors')} != expected:
            raise ValueError('local RGB publication coverage mismatch')

    def iter_rgb(self, indices):
        for index in indices:
            rgb, identity = self.read(index)
            yield self.manifest['records'][index], rgb, identity
