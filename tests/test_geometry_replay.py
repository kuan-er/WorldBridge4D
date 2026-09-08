import copy
from datetime import timedelta
import json
from pathlib import Path
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from worldbridge.trainer.geometry_replay import (GeometryReplay, assert_exact, input_ready,
    pack_tree, read_entry, request_for, request_key, unpack_tree, write_entry)
from worldbridge.trainer.batching import GeometryPrefetcher


def value():
    return (3, 2, np.arange(21*3*2*2,dtype=np.float32).reshape(21,3,2,2),
            np.ones((21,2,2),bool), np.zeros((2,2,3),np.uint8),
            np.ones((21,2,2),bool), {'K':np.eye(3,dtype=np.float64),
            'nested':(None,True,2.0,np.float32(0.25)), 'zero':np.array(2.)},None,None)


def test_tree_roundtrip():
    arrays = {}; desc = pack_tree(value(),arrays)
    assert_exact(value(),unpack_tree(json.loads(json.dumps(desc)),arrays))


def test_replay_exact_missing_and_tamper(tmp_path, monkeypatch):
    # Unit fixtures do not require a64GiB scratch disk.
    import worldbridge.trainer.geometry_replay as module
    from types import SimpleNamespace
    monkeypatch.setattr(module.shutil,'disk_usage',lambda _: SimpleNamespace(free=2**40))
    req = {'step':150000,'slot':0,'rank':0,'dataset':'dynamic_replica','index':3}
    entry = write_entry(tmp_path,req,value())
    (tmp_path/'complete.json').write_text(json.dumps(dict(config_sha256='cfg',count=1,
        entries={request_key(req):entry})))
    replay = GeometryReplay(tmp_path,'cfg'); before = np.random.get_state()
    actual, elapsed = replay.load(req); assert_exact(value(),actual); assert elapsed >= 0
    after = np.random.get_state(); np.testing.assert_array_equal(before[1],after[1])
    with pytest.raises(ValueError,match='config'): GeometryReplay(tmp_path,'wrong')
    with pytest.raises(KeyError): replay.load(dict(req,slot=1))
    with pytest.raises(FileExistsError): write_entry(tmp_path,req,value())
    with (tmp_path/entry['file']).open('ab') as f: f.write(b'bad')
    with pytest.raises(ValueError,match='checksum'): replay.load(req)


class Dataset:
    def __len__(self): return 20
    def clip_id(self,index): return f'clip{index}'
    def source_all_targets_with_visibility(self,*_):
        raise AssertionError('replay must not call live geometry')


class Replay:
    def load(self,req):
        assert req['K'] == 15 and req['slots'] == 4 and req['cycle']
        return value(),0.0


def test_prefetch_preserves_plans_and_rng_without_live_reads():
    datasets = {name:Dataset() for name in ['kubric','pointodyssey','dynamic_replica']}
    options = dict(seed=20260812,rank=0,accumulation=4,microbatch_per_gpu=1,
        targets_per_source=15,use_source_rgb=True,cycle_enabled=True,
        cycle_dataset_names=tuple(datasets),start_step=150000,target_steps=150001,depth=1,workers=1)
    p = GeometryPrefetcher(datasets,**options,geometry_replay=Replay())
    try:
        p.refill(); planned = p.pop(150000)
        from worldbridge.data.sampling import deterministic_sample_plan
        for slot,(index,source,rng) in enumerate(planned.sample_plans):
            expected = deterministic_sample_plan(planned.dataset,planned.dataset_name,20260812,150000,slot,0,4)
            assert (index,source) == expected[:2]
            assert rng.bit_generator.state == expected[2].bit_generator.state
            assert_exact(planned.geometry_futures[slot].result()[0],value())
    finally: p.close()


def gloo_worker(rank,path,result,delayed):
    dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2,
                            timeout=timedelta(seconds=10))
    try:
        if delayed and rank == 1:
            time.sleep(3)
            return
        try:
            input_ready(dist.group.WORLD,rank=rank,step=150000,micro=0,
                        dataset='dynamic_replica',clips=[rank],timeout_seconds=1)
        except RuntimeError as e:
            assert delayed and rank == 0 and 'CPU input readiness failed' in str(e)
            Path(result).write_text(str(e))
        else:
            assert not delayed
    finally: dist.destroy_process_group()


@pytest.mark.parametrize('delayed',[False,True])
def test_gloo_input_ready_and_diagnostic_timeout(tmp_path,delayed):
    mp.spawn(gloo_worker,args=(str(tmp_path/'init'),str(tmp_path/'result'),delayed),
             nprocs=2,join=True)
    if delayed: assert 'clips' in (tmp_path/'result').read_text()
