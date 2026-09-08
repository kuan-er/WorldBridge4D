import json
from types import SimpleNamespace

import numpy as np
import pytest

from worldbridge.data.cache.trajectory_mmap import convert_stream,verify_stream,load_clip,MEMBERS


def fixture(tmp_path,monkeypatch):
    monkeypatch.setattr('worldbridge.data.cache.trajectory_mmap.shutil.disk_usage',
                        lambda _:SimpleNamespace(free=2**40))
    arrays=dict(traj_3d_world=np.arange(5*4*3,dtype=np.float32).reshape(5,4,3),
                traj_2d=np.arange(5*4*2,dtype=np.float32).reshape(5,4,2),
                verts_inds_vis=np.ones((5,4),np.uint8),instances=np.arange(4,dtype=np.int64),
                paths=np.array(json.dumps([f'{i}.pth' for i in range(5)])))
    source=tmp_path/'stream.npz';np.savez_compressed(source,**arrays)
    return source,tmp_path/'mmap',arrays


def test_exact_shared_clip_and_resume(tmp_path,monkeypatch):
    source,root,arrays=fixture(tmp_path,monkeypatch)
    report,written=convert_stream(source,root,'idx');assert written
    directory=root/'stream';checked=verify_stream(directory,index_sha256='idx')
    for name in MEMBERS:
        a=np.load(directory/name,allow_pickle=False)
        assert a.dtype==arrays[name[:-4]].dtype
        np.testing.assert_array_equal(a,arrays[name[:-4]])
    clip=load_clip(directory,['3.pth','1.pth'],verified_report=checked)
    np.testing.assert_array_equal(clip['trajs_3d_world'],arrays['traj_3d_world'][[3,1]])
    np.testing.assert_array_equal(clip['visible'],arrays['verts_inds_vis'][[3,1]])
    assert clip['visible'].dtype == np.bool_
    assert all(not v.flags.writeable for v in clip.values())
    _,written=convert_stream(source,root,'idx');assert not written
    assert source.exists()
    with pytest.raises(ValueError,match='index'):convert_stream(source,root,'wrong')


def test_corrupt_and_changed_source_refused(tmp_path,monkeypatch):
    source,root,arrays=fixture(tmp_path,monkeypatch)
    convert_stream(source,root,'idx')
    with (root/'stream'/'traj_2d.npy').open('ab') as f:f.write(b'bad')
    with pytest.raises(ValueError,match='checksum'):verify_stream(root/'stream')


def test_no_space_or_unexpected_member_no_publication(tmp_path,monkeypatch):
    source,root,arrays=fixture(tmp_path,monkeypatch)
    monkeypatch.setattr('worldbridge.data.cache.trajectory_mmap.shutil.disk_usage',
                        lambda _:SimpleNamespace(free=1))
    with pytest.raises(OSError,match='reserve'):convert_stream(source,root,'idx')
    assert not (root/'stream').exists()
    arrays['unexpected']=np.ones(1);np.savez_compressed(source,**arrays)
    with pytest.raises(ValueError,match='members'):convert_stream(source,root,'idx')


def test_source_replacement_refused(tmp_path,monkeypatch):
    source,root,arrays=fixture(tmp_path,monkeypatch)
    convert_stream(source,root,'idx')
    arrays['traj_3d_world']+=1;np.savez_compressed(source,**arrays)
    with pytest.raises(ValueError,match='different source'):convert_stream(source,root,'idx')
