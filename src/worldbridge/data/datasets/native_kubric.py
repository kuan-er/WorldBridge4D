"""Native512 training admission, using pre-staged RGB/latents/geometry only."""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from .movif256 import MOViF256Dataset
from ..cache.native import NativeLatentCache, file_sha256
from ..cache.native_rgb import NativeRGBCache


class NativeKubricDataset(MOViF256Dataset):
    image_size = 512

    def __init__(self, values):
        self.manifest = json.loads(Path(values['native_manifest']).read_text())
        m = self.manifest
        if m['dataset'] != 'kubric' or m['native_hw'] != [512, 512]:
            raise ValueError('native training admission is Kubric512 only')
        self.rows = [json.loads(x) for x in Path(m['index_path']).read_text().splitlines() if x]
        self.local_rgb = NativeRGBCache(values['native_rgb_root'], m)
        self.local_rgb.require_complete()
        self.local_latents = NativeLatentCache(values['native_latent_root'], 'kubric', m['sha256'],
                                                tuple(m['latent_shape']), m['vae_sha256'])
        report = json.loads((self.local_latents.root / 'bulk_complete.json').read_text())
        if (report.get('manifest_sha256') != m['sha256'] or report.get('processed') != len(self.rows)
                or not report.get('all_existing_training_index_entries_verified')):
            raise ValueError('native latent corpus is incomplete')
        self.geometry_root = Path(values['native_geometry_root'])
        mode = values.get('native_geometry_mode', 'staged')
        if mode not in ('staged', 'verified_cache_or_raw'):
            raise ValueError('unknown native GT reader mode')
        self.demand_reader = None
        if mode == 'verified_cache_or_raw':
            if values.get('native_geometry_full_corpus'):
                raise ValueError('select staged-full or explicit native on-demand GT, not both')
            from ..cache.native_gt_demand import NativeGTDemandReader
            self.demand_reader = NativeGTDemandReader(self.geometry_root, m, self.local_rgb)
            self.geometry_ready = None
            return
        self.geometry_ready = json.loads((self.geometry_root / 'ready.json').read_text())
        if self.geometry_ready['manifest_sha256'] != m['sha256']:
            raise RuntimeError('native geometry manifest mismatch')
        if values.get('native_geometry_full_corpus'):
            if (not self.geometry_ready.get('full_corpus')
                    or self.geometry_ready.get('index_sha256') != m['index_sha256']
                    or set(self.geometry_ready['entries']) != {str(i) for i in range(len(self.rows))}):
                raise RuntimeError('native long training requires the complete GT corpus')

    @lru_cache(maxsize=4)
    def sample(self, index):
        index = int(index)
        if getattr(self, 'demand_reader', None) is not None:
            return self.demand_reader.read(index)
        entry = self.geometry_ready['entries'][str(index)]  # no raw-source or subset fallback
        path = self.geometry_root / entry['file']
        if file_sha256(path) != entry['sha256']:
            raise RuntimeError('native geometry checksum mismatch')
        sample = self._load_compact_sample(path)
        if sample.depth.shape != (21, 512, 512):
            raise ValueError('native GT must be21x512x512, never upsampled256')
        return sample

    @lru_cache(maxsize=2)
    def rgb(self, index):
        return self.local_rgb.read(int(index))[0]

    def source_rgb(self, index, source):
        if not 0 <= int(source) < 21:
            raise ValueError('source outside21 frames')
        return self.rgb(int(index))[int(source)]

    def clean_latent(self, index):
        index = int(index)
        _, identity = self.local_rgb.read(index)
        return self.local_latents.read(index, self.rows[index]['clip_id'], identity)

    def set_lazy_vae_sha256(self, value):
        if value != self.manifest['vae_sha256']:
            raise ValueError('VAE identity mismatch')

    def cache_latent(self, *args):
        raise RuntimeError('native admission never generates missing training inputs')
