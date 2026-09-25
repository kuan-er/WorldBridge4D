"""Recovery changes only endpoint/cadence and strict existing W&B tracking."""
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.tracking import init_wandb

ROOT = Path(__file__).resolve().parents[1]


def test_recovery_config_delta():
    base = yaml.safe_load((ROOT / 'configs/h031_k512_dr512_full_k9_unfreeze_183000.yaml').read_text())
    cfg = yaml.safe_load((ROOT / 'configs/h031_k512_dr512_full_k9_resume191000_to200000.yaml').read_text())
    base['max_steps'] = 200000
    base['lr_restart']['end_step'] = 200000
    base['checkpoint_steps'] = list(range(192000, 200001, 1000))
    base['checkpoint_every_after'] = 1000
    base['tracking']['resume'] = 'must'
    assert cfg == base
    validate_config(cfg, 2)
    assert cfg['gradient_clip'] == 1
    assert cfg['lr_restart']['start_step'] == cfg['selected_checkpoint_step'] == 183000


@pytest.fixture
def tracking(monkeypatch):
    run = Mock(id='zypk1th1', resumed=True)
    api = SimpleNamespace(init=Mock(return_value=run), util=SimpleNamespace(generate_id=lambda: 'new'))
    monkeypatch.setitem(sys.modules, 'wandb', api)
    monkeypatch.setenv('WANDB_MODE', 'online')
    monkeypatch.setenv('WANDB_API_KEY', 'unit-test-placeholder')
    return api, run


def test_strict_existing_id(tmp_path, tracking):
    api, run = tracking
    (tmp_path / 'wandb_run_id').write_text('zypk1th1\n')
    assert init_wandb({'tracking': {'resume': 'must'}}, tmp_path, 0, False) is run
    assert api.init.call_args.kwargs['resume'] == 'must'
    assert api.init.call_args.kwargs['id'] == 'zypk1th1'


def test_strict_requires_id(tmp_path, tracking):
    with pytest.raises(ValueError, match='existing wandb_run_id'):
        init_wandb({'tracking': {'resume': 'must'}}, tmp_path, 0, False)
    tracking[0].init.assert_not_called()


def test_strict_rejects_offline(tmp_path, tracking, monkeypatch):
    (tmp_path / 'wandb_run_id').write_text('zypk1th1')
    monkeypatch.setenv('WANDB_MODE', 'offline')
    with pytest.raises(ValueError, match='online'):
        init_wandb({'tracking': {'resume': 'must'}}, tmp_path, 0, False)


def test_strict_rejects_fresh_run(tmp_path, tracking):
    (tmp_path / 'wandb_run_id').write_text('zypk1th1')
    tracking[1].resumed = False
    with pytest.raises(RuntimeError, match='did not resume'):
        init_wandb({'tracking': {'resume': 'must'}}, tmp_path, 0, False)
    tracking[1].finish.assert_called_once_with(exit_code=1)


def test_rng_matches_fsdp_capture_contract(monkeypatch):
    import importlib
    import torch
    from worldbridge.trainer.checkpoint import capture_rng_state
    monkeypatch.syspath_prepend(str(ROOT / 'research/analysis'))
    check_rng = importlib.import_module('h031_resume191000_to200000').check_rng
    rng = capture_rng_state(include_cuda=False)
    assert 'numpy_generator' not in rng
    rng['torch_cuda'] = torch.zeros(16, dtype=torch.uint8)  # CPU-only CUDA state stand-in
    check_rng(rng)
    for key in ('python', 'numpy_global', 'torch_cpu', 'torch_cuda'):
        broken = dict(rng)
        del broken[key]
        with pytest.raises(AssertionError):
            check_rng(broken)


def test_default_unchanged(tmp_path, tracking):
    init_wandb({}, tmp_path, 0, False)
    assert tracking[0].init.call_args.kwargs['resume'] == 'allow'
