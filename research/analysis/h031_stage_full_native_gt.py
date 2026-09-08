"""CPU full-corpus lossless native512 GT staging; no RGB/VAE generation."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time

import numpy as np

from worldbridge.data.cache.native import file_sha256, rgb_identity
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.datasets.movif256 import MOViF256Dataset
from worldbridge.data.movif import MOViFDataset
from worldbridge.data.native_inputs import _check_source, decode_kubric_rgb, validate_manifest
from worldbridge.utils.io import atomic_json

STOP=False
RESERVE=68719476736
CHECKPOINT_BUDGET=38400000000


def stop(*_):
    global STOP
    STOP=True


def write_sample(path, sample):
    """Compressed representation only: verify every field byte-exact after load."""
    path=Path(path);temporary=path.with_suffix('.tmp.npz')
    arrays=asdict(sample)
    if sample.depth.shape!=(21,512,512) or sample.segmentation.shape!=(21,512,512) or sample.clip_start!=0:
        raise ValueError('full GT requires original21frame512 inputs')
    np.savez_compressed(temporary,**arrays)
    with temporary.open('rb') as f:os.fsync(f.fileno())
    with np.load(temporary,allow_pickle=False) as z:
        if set(z.files)!=set(arrays):raise ValueError('GT field mismatch')
        for name,v in arrays.items():
            v=np.asarray(v);a=z[name]
            if a.dtype!=v.dtype or a.shape!=v.shape or a.tobytes()!=v.tobytes():
                raise ValueError(f'lossless GT readback mismatch: {name}')
    if path.exists():raise FileExistsError(path)
    os.replace(temporary,path)
    return {'file':path.name,'sha256':file_sha256(path),'bytes':path.stat().st_size}


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--rgb-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--reuse-bounded',required=True);p.add_argument('--probe',required=True)
    args=p.parse_args();assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    signal.signal(signal.SIGTERM,stop)
    m=json.loads(Path(args.manifest).read_text());validate_manifest(m)
    assert m['native_hw']==[512,512] and len(m['records'])==5737
    rgb=NativeRGBCache(args.rgb_root,m);rgb.require_complete()
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    lock=(root/'producer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (root/'ready.json').exists()
    probe=json.loads(Path(args.probe).read_text());assert probe['event']=='NATIVE_GT_COMPRESSION_PROBE_OK'
    estimate_per_clip=probe['full5737_estimate_max_sample_bytes']/5737
    oldroot=Path(args.reuse_bounded);old=json.loads((oldroot/'ready.json').read_text());assert old['manifest_sha256']==m['sha256']
    report=dict(manifest_sha256=m['sha256'],index_sha256=m['index_sha256'],full_corpus=True,
                geometry_transform='native512_no_resize_original_radial_depth_and_validity',
                compression='lossless_npz_every_field_byte_exact',seed=20260812,entries={},training_ready=False)
    start=time.perf_counter();groups=defaultdict(dict)
    for row in m['records']:
        key=str(row['index']);sidecar=root/f'geom_{row["index"]:08d}.json'
        if sidecar.exists():
            entry=json.loads(sidecar.read_text())
            assert entry['manifest_sha256']==m['sha256'] and entry['row']==row
            assert file_sha256(root/entry['file'])==entry['sha256']
            _check_source(m,row['path']);report['entries'][key]=entry
        else:groups[row['path']][row['local_record']]=row
    tf=MOViFDataset._tf();tf.config.set_visible_devices([],'GPU')
    native=MOViFDataset.__new__(MOViFDataset);native.clip_length=21;native.clip_start=0;native.seed=20260812
    for path,wanted in sorted(groups.items()):
        _check_source(m,path)
        options=tf.data.Options();options.threading.private_threadpool_size=1;options.threading.max_intra_op_parallelism=1
        records=tf.data.TFRecordDataset([path],num_parallel_reads=1).with_options(options)
        for local,raw_tensor in enumerate(records.take(max(wanted)+1)):
            if local not in wanted:continue
            if STOP:
                print(json.dumps(dict(event='NATIVE_GT_FULL_CHECKPOINTED_STOP',count=len(report['entries']))),flush=True);return 3
            remaining=5737-len(report['entries'])
            if shutil.disk_usage(root).free<RESERVE+CHECKPOINT_BUDGET+remaining*estimate_per_clip*1.2:
                raise OSError('full native GT would violate remaining payload +20%, checkpoint budget,64GiB reserve')
            row=wanted[local];key=str(row['index']);raw=bytes(raw_tensor.numpy())
            _,identity=rgb.read(row['index']);assert rgb_identity(decode_kubric_rgb(raw))==identity
            sample=native._decode(raw,row['raw_index'],decode_rgb=False)
            if key in old['entries']:
                prev=old['entries'][key];assert file_sha256(oldroot/prev['file'])==prev['sha256']
                with np.load(oldroot/prev['file'],allow_pickle=False) as z:
                    for name,value in asdict(sample).items():
                        value=np.asarray(value);assert z[name].dtype==value.dtype and z[name].tobytes()==value.tobytes()
            _check_source(m,path)
            destination=root/f'geom_{row["index"]:08d}.npz'
            # Interrupted writes without an identity sidecar are never accepted.
            if destination.exists():raise RuntimeError(f'uncommitted GT output requires explicit repair: {destination}')
            entry=write_sample(destination,sample)
            entry.update(manifest_sha256=m['sha256'],row=row,RGB_sha256=identity['rgb_sha256'])
            atomic_json(destination.with_suffix('.json'),entry);report['entries'][key]=entry
            print(json.dumps(dict(event='NATIVE_GT_FULL_PROGRESS',count=len(report['entries']),total=5737,index=row['index'],bytes=entry['bytes'],elapsed_seconds=time.perf_counter()-start)),flush=True)
        _check_source(m,path)
    assert set(report['entries'])=={str(i) for i in range(5737)}
    validate_manifest(m)
    report.update(count=5737,payload_bytes=sum(e['bytes'] for e in report['entries'].values()),elapsed_seconds=time.perf_counter()-start)
    atomic_json(root/'ready.json',report)
    print(json.dumps(dict(event='NATIVE_GT_FULL_OK',count=5737,payload_bytes=report['payload_bytes'],elapsed_seconds=report['elapsed_seconds'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
