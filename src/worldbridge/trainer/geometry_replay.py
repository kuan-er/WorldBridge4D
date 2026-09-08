"""Hash-bound, bounded geometry replay. No RNG consumption or live-data fallback."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from ..data.cache.native import file_sha256
from ..utils.io import atomic_json


def request_for(prefetch, step, slot, dataset_name, index):
    return dict(seed=prefetch.seed, rank=prefetch.rank, step=int(step), slot=int(slot),
                dataset=dataset_name, index=int(index), slots=prefetch.slots_per_rank,
                K=prefetch.targets_per_source, RGB=prefetch.use_source_rgb,
                cycle=prefetch.cycle_enabled and dataset_name in prefetch.cycle_dataset_names,
                boundary=prefetch.boundary_supervision, contrast=prefetch.edge_contrast_enabled)


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pack_tree(value, arrays):
    if isinstance(value, np.ndarray):
        key = f'a{len(arrays)}'
        arrays[key] = np.array(value, copy=True, order='C')
        return {'array': key}
    if isinstance(value, dict):
        return {'dict': [[k, pack_tree(v, arrays)] for k, v in value.items()]}
    if isinstance(value, (tuple, list)):
        return {'tuple' if isinstance(value, tuple) else 'list': [pack_tree(v, arrays) for v in value]}
    if isinstance(value, np.generic):
        # Preserve NumPy scalar dtype without pickle.
        return {'scalar': pack_tree(np.asarray(value), arrays)}
    if value is None or isinstance(value, (str, bool, int, float)):
        return {'value': value}
    raise TypeError(f'unsupported geometry value: {type(value)}')


def unpack_tree(desc, arrays):
    if 'array' in desc:
        return arrays[desc['array']].copy()
    if 'scalar' in desc:
        return unpack_tree(desc['scalar'], arrays).reshape(())[()]
    if 'dict' in desc:
        return {k: unpack_tree(v, arrays) for k, v in desc['dict']}
    for kind, constructor in [('tuple', tuple), ('list', list)]:
        if kind in desc:
            return constructor(unpack_tree(v, arrays) for v in desc[kind])
    return desc['value']


def assert_exact(a, b):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and a.keys() == b.keys()
        for k in a: assert_exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b): assert_exact(x, y)
    else:
        assert type(a) is type(b) and a == b


def write_entry(root, request, value):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    key = request_key(request)
    arrays = {}; tree = pack_tree(value, arrays)
    payload_bytes = sum(v.nbytes for v in arrays.values())
    if shutil.disk_usage(root).free < 68719476736 + payload_bytes * 1.2:
        raise OSError('geometry replay would violate64GiB reserve')
    path = root / f'{key}.npz'
    if path.exists():
        raise FileExistsError(path)
    temporary = root / f'.{key}.{os.getpid()}.tmp'
    try:
        with temporary.open('xb') as f:
            np.savez(f, **arrays)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    entry = dict(request=request, file=path.name, sha256=file_sha256(path), tree=tree,
                 payload_bytes=payload_bytes)
    atomic_json(root / f'{key}.json', entry)
    assert_exact(value, read_entry(root, entry))
    return entry


def read_entry(root, entry):
    filename = entry['file']
    if Path(filename).name != filename:
        raise ValueError('geometry replay entry is not a local basename')
    path = Path(root) / filename
    if file_sha256(path) != entry['sha256']:
        raise ValueError(f'geometry replay checksum mismatch: {path}')
    with np.load(path, allow_pickle=False) as arrays:
        return unpack_tree(entry['tree'], arrays)


class GeometryReplay:
    def __init__(self, root, config_sha256, *, indexes=None, expected_count=None):
        self.root = Path(root)
        report = json.loads((self.root / 'complete.json').read_text())
        if report['config_sha256'] != config_sha256 or report['count'] != len(report['entries']):
            raise ValueError('geometry replay config/coverage mismatch')
        if indexes is not None and report.get('indexes') != indexes:
            raise ValueError('geometry replay dataset index identity mismatch')
        if expected_count is not None and report['count'] != expected_count:
            raise ValueError('geometry replay coverage is incomplete')
        self.entries = report['entries']

    def load(self, request):
        start = time.perf_counter()
        entry = self.entries[request_key(request)]  # missing is fatal, never live fallback
        if entry['request'] != request:
            raise ValueError('geometry replay request mismatch')
        return read_entry(self.root, entry), time.perf_counter() - start


def input_ready(group, *, rank, step, micro, dataset, clips, timeout_seconds=120):
    """CPU-only readiness synchronization before any FSDP forward collective."""
    import torch.distributed as dist
    context = dict(rank=rank, step=step, micro=micro, dataset=dataset, clips=clips)
    print(json.dumps(dict(event='CPU_INPUT_READY_ENTER', **context)), flush=True)
    try:
        dist.monitored_barrier(group=group, timeout=timedelta(seconds=timeout_seconds),
                               wait_all_ranks=True)
    except RuntimeError as e:
        raise RuntimeError(f'CPU input readiness failed: {context}') from e
    print(json.dumps(dict(event='CPU_INPUT_READY_ALL', **context)), flush=True)
