"""Managed CPU-only DR trajectory mmap producer; preserves original archives."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import time
import zipfile

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.cache.trajectory_mmap import RESERVE,convert_stream
from worldbridge.utils.io import atomic_json

STOP = False

def stop(*_):
    global STOP
    STOP = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source',required=True);p.add_argument('--index',required=True)
    p.add_argument('--output',required=True);p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    signal.signal(signal.SIGTERM,stop)
    source=Path(args.source);root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    index_sha=file_sha256(args.index)
    rows=[json.loads(line) for line in Path(args.index).read_text().splitlines() if line]
    names=sorted({str(row['stream']) for row in rows})
    assert len(rows)==6090 and len(names)==435
    assert all(Path(name).name==name and name not in ('.','..') for name in names)
    if args.smoke:names=[names[0],names[len(names)//2],names[-1]]
    with (root/'.producer.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        remaining=0
        for name in names:
            if not (root/name).exists():
                with zipfile.ZipFile(source/(name+'.npz')) as z:
                    remaining+=sum(v.file_size for v in z.infolist())
        if shutil.disk_usage(root).free < RESERVE+remaining*1.2:
            raise OSError('full remaining DR mmap payload plus20pct headroom would violate64GiB reserve')
        print(json.dumps(dict(event='DR_MMAP_BEGIN',streams=len(names),remaining_bytes=remaining,
            index_sha256=index_sha,seed=20260908,source_unchanged=True,training_switch=False)),flush=True)
        reports={};new=0;start=time.perf_counter()
        for name in names:
            if STOP:
                print(json.dumps(dict(event='DR_MMAP_CHECKPOINTED_STOP',streams=len(reports),written=new)),flush=True)
                return 3
            entry,written=convert_stream(source/(name+'.npz'),root,index_sha)
            reports[name]=entry;new+=int(written)
            print(json.dumps(dict(event='DR_MMAP_PROGRESS',processed=len(reports),total=len(names),
                written=new,stream=name,elapsed_seconds=time.perf_counter()-start)),flush=True)
        assert file_sha256(args.index)==index_sha
        report=dict(event='DR_MMAP_SMOKE_OK' if args.smoke else 'DR_MMAP_BULK_OK',
            index_sha256=index_sha,clips=6090,streams=len(reports),written=new,
            reused=len(reports)-new,entries=reports,source_archives_deleted=False,
            training_switch=False,elapsed_seconds=time.perf_counter()-start)
        atomic_json(root/('smoke_complete.json' if args.smoke else 'bulk_complete.json'),report)
        print(json.dumps({k:v for k,v in report.items() if k!='entries'}),flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
