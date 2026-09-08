"""Write-once DR trajectory archives expanded to shared, frame-addressable NPY.

No numerical conversion: original NPY member bytes are copied from the ZIP,
CRC-checked by zipfile, hashed while extracting, then independently read back.
Overlapping21-frame clips share the same files; no dense XYZ is duplicated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from .native import file_sha256
from ...utils.io import atomic_json

MEMBERS = ('traj_3d_world.npy', 'traj_2d.npy', 'verts_inds_vis.npy', 'instances.npy', 'paths.npy')
RESERVE = 68719476736


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def source_stat(path):
    s = Path(path).stat()
    return dict(bytes=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino)


def validate_arrays(directory):
    arrays = {name: np.load(Path(directory)/name, mmap_mode='r', allow_pickle=False)
              for name in MEMBERS}
    world = arrays['traj_3d_world.npy']; uv = arrays['traj_2d.npy']
    vis = arrays['verts_inds_vis.npy']; instances = arrays['instances.npy']
    if (world.ndim != 3 or world.shape[-1] != 3 or uv.shape != (*world.shape[:2],2)
            or vis.shape != world.shape[:2] or instances.shape != (world.shape[1],)):
        raise ValueError('unexpected DR trajectory array shapes')
    if (world.dtype.kind != 'f' or uv.dtype.kind != 'f'
            or vis.dtype not in (np.dtype('bool'), np.dtype('uint8'))):
        raise ValueError('unexpected DR trajectory dtypes')
    paths = json.loads(str(arrays['paths.npy']))
    if len(paths) != world.shape[0] or len(set(paths)) != len(paths):
        raise ValueError('DR trajectory path/frame identity mismatch')
    return {name: dict(shape=list(v.shape), dtype=str(v.dtype)) for name,v in arrays.items()}


def verify_stream(directory, *, index_sha256=None):
    directory = Path(directory)
    report = json.loads((directory/'complete.json').read_text())
    if index_sha256 is not None and report['index_sha256'] != index_sha256:
        raise ValueError('trajectory index identity mismatch')
    if set(report['members']) != set(MEMBERS):
        raise ValueError('trajectory member coverage mismatch')
    for name, entry in report['members'].items():
        if file_sha256(directory/name) != entry['sha256']:
            raise ValueError(f'trajectory mmap checksum mismatch: {directory/name}')
    if validate_arrays(directory) != report['arrays']:
        raise ValueError('trajectory array metadata mismatch')
    return report


def convert_stream(source, root, index_sha256):
    source = Path(source); root = Path(root)
    if source.suffix != '.npz' or source.stem != Path(source.stem).name:
        raise ValueError('invalid trajectory archive name')
    root.mkdir(parents=True,exist_ok=True)
    destination = root/source.stem
    before = source_stat(source)
    if destination.exists():
        report = verify_stream(destination,index_sha256=index_sha256)
        if report['source_sha256'] != file_sha256(source):
            raise ValueError('existing mmap belongs to a different source archive')
        return report,False
    with zipfile.ZipFile(source) as archive:
        if len(archive.namelist()) != len(MEMBERS) or set(archive.namelist()) != set(MEMBERS):
            raise ValueError('unexpected archive members')
        needed = sum(archive.getinfo(name).file_size for name in MEMBERS)
        if shutil.disk_usage(root).free < RESERVE+needed*1.2:
            raise OSError('trajectory conversion would violate64GiB reserve')
        source_sha = file_sha256(source)
        temporary = Path(tempfile.mkdtemp(prefix=f'.{source.stem}.',dir=root))
        try:
            report = dict(version=1,stream=source.stem,index_sha256=index_sha256,
                          source_sha256=source_sha,source_stat=before,
                          transform='exact_NPY_member_bytes_no_cast_or_resample',members={})
            for name in MEMBERS:
                digest = hashlib.sha256(); count = 0
                with archive.open(name) as reader, (temporary/name).open('xb') as writer:
                    while chunk := reader.read(4*1024*1024):
                        writer.write(chunk); digest.update(chunk); count += len(chunk)
                    writer.flush(); os.fsync(writer.fileno())
                if count != archive.getinfo(name).file_size or file_sha256(temporary/name) != digest.hexdigest():
                    raise ValueError('trajectory NPY readback mismatch')
                report['members'][name] = dict(bytes=count,sha256=digest.hexdigest())
            report['arrays'] = validate_arrays(temporary)
            if source_stat(source) != before:
                raise ValueError('trajectory source changed during conversion')
            atomic_json(temporary/'complete.json',report)
            fsync_directory(temporary)
            # Single managed producer; never replace any previously published stream.
            if destination.exists(): raise FileExistsError(destination)
            os.rename(temporary,destination); fsync_directory(root)
        finally:
            if temporary.exists(): shutil.rmtree(temporary)
    return report,True


def load_clip(directory, frame_paths, *, verified_report):
    """Read only selected frames from a previously hash-verified shared stream."""
    directory = Path(directory)
    if verified_report['stream'] != directory.name:
        raise ValueError('trajectory stream identity mismatch')
    paths = json.loads(str(np.load(directory/'paths.npy',allow_pickle=False)))
    lookup = {path:i for i,path in enumerate(paths)}
    indices = [lookup[path] for path in frame_paths]
    result = {}
    for key,filename in [('trajs_3d_world','traj_3d_world.npy'),('trajs_2d','traj_2d.npy'),
                         ('visible','verts_inds_vis.npy')]:
        result[key] = np.load(directory/filename,mmap_mode='r',allow_pickle=False)[indices]
    # Match the existing NPZ loader's uint8 -> bool interpretation; files stay byte-exact.
    result['visible'] = result['visible'].astype(bool, copy=False)
    instances = np.load(directory/'instances.npy',mmap_mode='r',allow_pickle=False)
    result['instances'] = np.broadcast_to(instances,(len(indices),len(instances)))
    for value in result.values(): value.setflags(write=False)
    return result
