"""Native512 FULL-mode capacity route validation and unfreeze semantics."""
from pathlib import Path

import pytest
import torch
from torch import nn
import yaml

from worldbridge.models.worldbridge import DenseQueryWanModel
from worldbridge.trainer.config import validate_config


def config():
    return yaml.safe_load(Path('configs/h031_k512_full_k3_capacity.yaml').read_text())


def config_k9():
    return yaml.safe_load(Path('configs/h031_k512_full_k9_capacity.yaml').read_text())


def config_unfreeze():
    return yaml.safe_load(Path('configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml').read_text())


def test_full_capacity_config_and_decoder_only_control():
    validate_config(config(), 2)
    validate_config(config_k9(), 2)
    validate_config(config_unfreeze(), 2)
    old = yaml.safe_load(Path('configs/h031_k512_b1_a4_k11_mix50_gpu67_to200000.yaml').read_text())
    validate_config(old, 2)


def test_unfreeze_requires_dr_native_paths_and_full_route():
    cfg = config_unfreeze()
    cfg['native_kubric512_full'] = False
    with pytest.raises(ValueError):
        validate_config(cfg, 2)
    cfg = config_unfreeze()
    del cfg['datasets']['dynamic_replica']['native_manifest']
    with pytest.raises(ValueError):
        validate_config(cfg, 2)


@pytest.mark.parametrize('key,value', [
    ('trainable_mode', 'decoder_only'),
    ('trainable_mode', 'source_rgb_plus_wan_decoder'),
    ('precision', 'fp32'),
    ('fsdp_master_precision', 'model'),
    ('targets_per_source', 11),
    ('microbatch_per_gpu', 2),
    ('gradient_accumulation', 2),
    ('dataset_mix_counts', {'kubric': 7, 'pointodyssey': 6, 'dynamic_replica': 7}),
    ('cycle_reprojection_weight', 1.0),
    ('cycle_reprojection_datasets', ['kubric']),
    ('source_edge_contrast_weight', 0.01),
])
def test_invalid_full_capacity_protocol(key, value):
    cfg = config(); cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg, 2)


def test_full_capacity_rejects_lr_restart_and_legacy_native_flags():
    cfg = config()
    cfg['lr_restart'] = {
        'start_step': 0, 'end_step': 3, 'warmup_steps': 1, 'schedule': 'warmup_hold',
        'group_learning_rates': {'dense_decoder': 3e-6, 'source_rgb_decay': 3e-6,
                                 'source_rgb_no_decay': 3e-6},
    }
    with pytest.raises(ValueError):
        validate_config(cfg, 2)
    cfg = config()
    cfg['native_kubric512_b1_a4_k9'] = True
    with pytest.raises(ValueError):
        validate_config(cfg, 2)


def test_full_mode_unfreezes_wan_geometry_and_decoder_but_not_bypassed():
    class TinyBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = nn.Linear(2, 2)
            self.adapter = nn.Linear(2, 2)
            self.bypassed = nn.Parameter(torch.ones(1))
            self.adapter_parameters = list(self.adapter.parameters())
            self.bypassed_parameters = [self.bypassed]

    class TinyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.old = nn.Linear(2, 2)

    model = DenseQueryWanModel(TinyBackbone(), TinyDecoder())
    model.configure_trainable("full")
    assert all(parameter.requires_grad for parameter in model.backbone.dit.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.adapter.parameters())
    assert not model.backbone.bypassed.requires_grad
    assert all(parameter.requires_grad for parameter in model.decoder.parameters())
