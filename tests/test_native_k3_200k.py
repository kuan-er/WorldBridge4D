"""Explicit200k extension retains the existing K3 full-resume/LR contract."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest
import yaml
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import apply_lr_restart_schedule, dataset_for_step

CONFIG = 'configs/h031_k512_k3_mix50_gpu56_to200000.yaml'
PARENT = 'configs/h031_k512_k3_mix50_gpu25_to170000.yaml'

def config():
    return yaml.safe_load(Path(CONFIG).read_text())

def test_only_authorized_extension_and_operational_metadata_change():
    c = config(); old = yaml.safe_load(Path(PARENT).read_text())
    validate_config(c, 2); validate_config(old, 2)
    assert {k for k in c.keys() | old.keys() if c.get(k) != old.get(k)} == {
        'native_kubric512_k3_mix_200k', 'max_steps', 'lr_restart', 'checkpoint_steps',
        'graceful_stop_hours', 'tracking'}
    restart = copy.deepcopy(old['lr_restart']); restart['end_step'] = 200000
    assert c['lr_restart'] == restart
    assert c['max_steps'] == 200000
    assert c['targets_per_source'] == 3
    assert c['microbatch_per_gpu'] * c['gradient_accumulation'] * 2 == 8
    assert c['selected_checkpoint_step'] == 152768
    assert c['distributed_timeout_seconds'] == 900
    assert c['checkpoint_every_after'] == 500 and c['checkpoint_keep_last'] == 3
    assert c['checkpoint_steps'][-1] == 200000
    for step in (169499, 169500, 169639, 170000, 199999):
        assert dataset_for_step(step, c['seed'], c['dataset_mix_counts']) == dataset_for_step(
            step, old['seed'], old['dataset_mix_counts'])

@pytest.mark.parametrize('key,value', [
    ('native_kubric512_k3_mix_200k', False), ('max_steps', 200001), ('max_steps', 170000),
    ('native_kubric512_k3_mix_resume', False), ('targets_per_source', 5),
    ('native_kubric512_k9_mix_170k', False), ('gradient_accumulation', 2)])
def test_reject_unsupported_extensions(key, value):
    c = config(); c[key] = value
    with pytest.raises(ValueError): validate_config(c, 2)

@pytest.mark.parametrize('key,value', [('end_step', 170000), ('start_step', 169500),
                                      ('warmup_steps', 0), ('schedule', 'cosine')])
def test_no_lr_restart_or_implicit_horizon_change(key, value):
    c = config(); c['lr_restart'][key] = value
    with pytest.raises(ValueError): validate_config(c, 2)

def test_hold_no_jump_through_old_and_new_endpoints_no_optimizer_state_mutation():
    c = config(); old = yaml.safe_load(Path(PARENT).read_text())
    groups = [dict(name=k, lr=0.0, _base_lr=1e-5) for k in c['lr_restart']['group_learning_rates']]
    sentinel = {'step': 169500, 'exp_avg': object(), 'exp_avg_sq': object()}
    optimizer = SimpleNamespace(param_groups=groups, state=sentinel)
    for step in (169501, 169639, 170000, 170001, 199999, 200000):
        assert apply_lr_restart_schedule(optimizer, step, c['lr_restart']) == 1.0
        assert all(g['lr'] == g['_base_lr'] == 3e-6 for g in groups)
        assert optimizer.state is sentinel and sentinel['step'] == 169500
        if step <= 170000:
            assert apply_lr_restart_schedule(optimizer, step, old['lr_restart']) == 1.0
    with pytest.raises(ValueError): apply_lr_restart_schedule(optimizer, 200001, c['lr_restart'])
