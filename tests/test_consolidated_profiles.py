"""Keep H030256 B2K9 and H031 native profiles isolated after consolidation."""
from pathlib import Path
import pytest
import yaml
from worldbridge.trainer.config import validate_config

@pytest.mark.parametrize('name',[
    'h031_k512_b1_a4_k9_mix50_gpu67_to200000.yaml',
    'h031_k512_b1_a4_k11_mix50_gpu67_to200000.yaml',
    'h031_k512_b2_a2_k5_mix50_gpu67_to200000.yaml',
    'h031_k512_b2_a2_k9_mix50_gpu67_to200000.yaml',
])
def test_native_profiles_reject_legacy_cycle_k9(name):
    cfg=yaml.safe_load((Path('configs')/name).read_text())
    validate_config(cfg,2)
    cfg['cycle_b2_a2_k9']=True
    with pytest.raises(ValueError):validate_config(cfg,2)
