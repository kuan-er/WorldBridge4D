import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.data.cache.native_gt_demand import NativeGTDemandReader


CONFIG='configs/h031_k512_po256_dr256_k5_10k_ondemand.yaml'


def test_on_demand_is_reader_only_delta():
    a=yaml.safe_load(Path('configs/h031_k512_po256_dr256_k5_150010_to160010.yaml').read_text())
    b=yaml.safe_load(Path(CONFIG).read_text());validate_config(b,2)
    assert b['datasets']['kubric']['native_geometry_mode']=='verified_cache_or_raw'
    b['datasets']['kubric'].pop('native_geometry_mode');b['datasets']['kubric']['native_geometry_full_corpus']=True
    b['tracking']=a['tracking'];assert a==b


def test_on_demand_cannot_silently_change_capacity_or_bypass_modes():
    b=yaml.safe_load(Path(CONFIG).read_text());b['datasets']['kubric']['native_geometry_full_corpus']=True
    with pytest.raises(ValueError):validate_config(b,2)
    b=yaml.safe_load(Path('configs/h031_k512_po256_dr256_b1_a4_k5_dr_mmap.yaml').read_text())
    b['datasets']['kubric']['native_geometry_mode']='verified_cache_or_raw'
    with pytest.raises(ValueError):validate_config(b,2)


@pytest.mark.parametrize('case',['checksum','identity','missing_payload'])
def test_corrupt_publication_does_not_fall_back_to_raw(tmp_path,monkeypatch,case):
    row={'index':0,'path':'source','local_record':0,'raw_index':0}
    reader=NativeGTDemandReader.__new__(NativeGTDemandReader)
    reader.root=tmp_path;reader.manifest={'records':[row],'sha256':'manifest'}
    reader.local_rgb=SimpleNamespace(read=lambda index:(None,{'rgb_sha256':'rgb'}))
    monkeypatch.setattr('worldbridge.data.cache.native_gt_demand._check_source',lambda *args:None)
    entry=dict(manifest_sha256='manifest',row=row,RGB_sha256='rgb',file='geom_00000000.npz',sha256='wrong')
    if case!='missing_payload':(tmp_path/entry['file']).write_bytes(b'bad')
    if case=='identity':entry['manifest_sha256']='wrong'
    (tmp_path/'geom_00000000.json').write_text(json.dumps(entry))
    # There is deliberately no TF/native decoder: reaching raw would be a bug.
    with pytest.raises(RuntimeError):reader.read(0)
