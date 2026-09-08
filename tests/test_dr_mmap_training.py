import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.cache.trajectory_mmap import convert_stream
from worldbridge.data.cache.trajectory_mmap_reader import TrajectoryMmapReader
from worldbridge.data.datasets.dynamic_replica import DynamicReplicaDataset
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.geometry_replay import assert_exact

CONFIG = 'configs/h031_k512_po256_dr256_b1_a4_k5_dr_mmap.yaml'


def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr('worldbridge.data.cache.trajectory_mmap.shutil.disk_usage', lambda _: SimpleNamespace(free=2**40))
    geometry = tmp_path/'geometry'; (geometry/'splits').mkdir(parents=True)
    vp = dict(intrinsics_format='ndc_isotropic', focal_length=[2,2], principal_point=[0,0], R=np.eye(3).tolist(), T=[0,0,0])
    frames = [dict(trajectory=f'stream/{i}.pth', viewpoint=vp, depth='unused', rgb='unused') for i in range(23)]
    rows = [dict(index=i,clip_id=f'clip{i}',stream='stream',frames=frames[start:start+21]) for i,start in enumerate((0,2))]
    index=geometry/'splits/train.jsonl';index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (geometry/'PREPROCESSING_STATE.json').write_text(json.dumps({'raw_root':str(tmp_path)}))
    source=tmp_path/'stream.npz'
    world=np.ones((23,4,3),np.float32);world[...,2]=3
    uv=np.broadcast_to(np.array([[400,100],[500,200],[600,300],[700,400]],np.float32),(23,4,2))
    visible=np.ones((23,4),np.uint8);visible[5,0]=0
    np.savez_compressed(source,traj_3d_world=world,traj_2d=uv,verts_inds_vis=visible,instances=np.arange(4),paths=np.array(json.dumps([f['trajectory'] for f in frames])))
    root=tmp_path/'mmap';report,_=convert_stream(source,root,file_sha256(index))
    marker=root/'bulk_complete.json';marker.write_text(json.dumps(dict(event='DR_MMAP_BULK_OK', index_sha256=file_sha256(index), clips=2, streams=1, entries={'stream':report})))
    sha=file_sha256(marker)
    monkeypatch.setattr('worldbridge.data.datasets.dynamic_replica._depth', lambda path,image_size,cache_root: (np.full((image_size,image_size),3,np.float32),np.ones((image_size,image_size),bool)))
    old=DynamicReplicaDataset(geometry,trajectory_cache_root=tmp_path)
    new=DynamicReplicaDataset(geometry,trajectory_cache_root=tmp_path,trajectory_mmap_root=root,trajectory_mmap_complete_sha256=sha)
    return old,new,source,root,index,sha


def test_adapter_exact_annotations_cameras_all_sources_no_npz_fallback(tmp_path,monkeypatch):
    old,new,*_=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(new,'_load_stream',lambda _:pytest.fail('mmap must not use legacy stream loader'))
    for i in range(2):
        assert_exact(old._load_clip(old.rows[i]),new._load_clip(new.rows[i]))
        assert_exact(old.cycle_camera(i),new.cycle_camera(i))
        for source in range(21):
            assert_exact(old.source_all_targets_with_visibility(i,source),new.source_all_targets_with_visibility(i,source))
    assert len(new._trajectory_mmap._verified)==1 and len(new._streams)==0


@pytest.mark.parametrize('mutation',['member','source','missing','frame','index','marker'])
def test_mmap_contract_errors_are_fatal_not_eligibility_fallback(tmp_path,monkeypatch,mutation):
    old,new,source,root,index,sha=fixture(tmp_path,monkeypatch)
    reader=new._trajectory_mmap
    reader.read(new.rows[0])
    if mutation=='member':
        with (root/'stream/traj_2d.npy').open('ab') as f:f.write(b'x')
    elif mutation=='source':
        with source.open('ab') as f:f.write(b'x')
    elif mutation=='missing':(root/'stream/verts_inds_vis.npy').unlink()
    elif mutation=='frame':new.rows[1]['frames'][0]['trajectory']='wrong'
    elif mutation=='index':
        index.write_text(index.read_text()+'\n')
        with pytest.raises(RuntimeError):TrajectoryMmapReader(root,index,source.parent,sha)
        return
    else:
        with pytest.raises(RuntimeError):TrajectoryMmapReader(root,index,source.parent,'wrong')
        return
    with pytest.raises(RuntimeError):reader.read(new.rows[1])


def test_k5_exact_scientific_delta():
    a=yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k9_timeout900.yaml').read_text())
    b=yaml.safe_load(Path(CONFIG).read_text());validate_config(b,2)
    assert b.pop('native_kubric512_b1_a4_k5') is True
    a.pop('native_kubric512_b1_a4_k9')
    assert b['targets_per_source']==5 and b['microbatch_per_gpu']==1 and b['gradient_accumulation']==4
    b['targets_per_source']=a['targets_per_source'];b['tracking']=a['tracking']
    dr=b['datasets']['dynamic_replica'];assert dr.pop('trajectory_mmap_root') and dr.pop('trajectory_mmap_complete_sha256')
    assert a==b


@pytest.mark.parametrize('key,value', [('targets_per_source',9),('targets_per_source',7),('native_kubric512_b1_a4_k9',True),('native_kubric512_b1_a4_k15',True),('microbatch_per_gpu',2),('gradient_accumulation',2),('max_steps',155000)])
def test_k5_profile_fails_closed(key,value):
    cfg=yaml.safe_load(Path(CONFIG).read_text());cfg[key]=value
    with pytest.raises(ValueError):validate_config(cfg,2)
