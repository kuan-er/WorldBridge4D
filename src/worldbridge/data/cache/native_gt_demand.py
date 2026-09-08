"""Explicit read-only native512 GT: published exact cache or original TFRecord.

A cache miss is allowed only in this user-selected mode. A corrupt published
entry or missing/changed native source is fatal, never an eligibility fallback.
Training never writes to or races the background GT producer's output files.
"""
import json
from pathlib import Path
import time

from .native import file_sha256, rgb_identity
from ..movif import MOViFDataset
from ..native_inputs import _check_source, decode_kubric_rgb, validate_manifest


class NativeGTDemandReader:
    def __init__(self, root, manifest, local_rgb):
        self.root=Path(root);self.manifest=manifest;self.local_rgb=local_rgb
        validate_manifest(manifest)
        # Initialize TF CPU-only before trainer restores its per-rank RNG state.
        self.tf=MOViFDataset._tf();self.tf.config.set_visible_devices([],'GPU')
        self.tf.constant(0)
        self.native=MOViFDataset.__new__(MOViFDataset)
        self.native.clip_length=21;self.native.clip_start=0;self.native.seed=20260812

    def read(self,index):
        from ..datasets.movif256 import MOViF256Dataset
        start=time.perf_counter();index=int(index);row=self.manifest['records'][index]
        try:
            if row['index']!=index:raise RuntimeError('native GT record index mismatch')
            _check_source(self.manifest,row['path'])
            _,identity=self.local_rgb.read(index)
            sidecar=self.root/f'geom_{index:08d}.json'
            if sidecar.is_file():
                entry=json.loads(sidecar.read_text())
                if (entry['manifest_sha256']!=self.manifest['sha256'] or entry['row']!=row
                        or entry['RGB_sha256']!=identity['rgb_sha256']
                        or entry['file']!=f'geom_{index:08d}.npz'):
                    raise RuntimeError('published native GT identity mismatch')
                path=self.root/entry['file']
                if file_sha256(path)!=entry['sha256']:
                    raise RuntimeError('published native GT checksum mismatch')
                sample=MOViF256Dataset._load_compact_sample(path)
                mode='published_native_cache'
            else:
                # Same TensorFlow reader/decoder as the audited producer, with
                # CRC validation and bounded CPU thread pools; no RGB/VAE write.
                options=self.tf.data.Options();options.threading.private_threadpool_size=1
                options.threading.max_intra_op_parallelism=1
                records=self.tf.data.TFRecordDataset([row['path']],num_parallel_reads=1).with_options(options)
                value=next(iter(records.skip(row['local_record']).take(1)),None)
                if value is None:raise RuntimeError('original native GT record is missing')
                raw=bytes(value.numpy())
                if rgb_identity(decode_kubric_rgb(raw))!=identity:
                    raise RuntimeError('original native GT/RGB identity mismatch')
                sample=self.native._decode(raw,row['raw_index'],decode_rgb=False)
                mode='original_native_raw'
            _check_source(self.manifest,row['path'])
            if sample.depth.shape!=(21,512,512) or sample.segmentation.shape!=(21,512,512) or sample.clip_start!=0:
                raise RuntimeError('native GT must retain original21frame512 shape')
            print(json.dumps(dict(event='NATIVE_GT_DEMAND_READ',index=index,mode=mode,seconds=time.perf_counter()-start)),flush=True)
            return sample
        except (ValueError,KeyError,OSError) as exc:
            raise RuntimeError(f'native GT input contract failed at index{index}') from exc
