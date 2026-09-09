from collections import Counter
from pathlib import Path
import random
import pickle

import numpy as np
import pytest
import torch
import yaml

from worldbridge.data.constants import DATASET_NAMES, MIX_CYCLE
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import (dataset_for_step, deterministic_dataset_schedule,
                                             validated_mix_counts)
from worldbridge.trainer.lazy_vae import required_latent_requests
from worldbridge.trainer.batching import GeometryPrefetcher

CONFIG = 'configs/h031_k512_k9_mix50_152768_to154768.yaml'
COUNTS = dict(kubric=10, pointodyssey=5, dynamic_replica=5)


def cfg():
    return yaml.safe_load(Path(CONFIG).read_text())


def test_new_config_and_only_authorized_delta():
    new = cfg()
    validate_config(new, 2)
    old = yaml.safe_load(Path('configs/h031_k512_po256_dr256_k5_10k_ondemand.yaml').read_text())
    allowed = {'native_kubric512_b1_a4_k5', 'native_kubric512_b1_a4_k9',
               'native_kubric512_k5_10k', 'native_kubric512_k9_mix_trial',
               'dataset_mix_counts', 'targets_per_source', 'max_steps',
               'selected_checkpoint_step', 'selected_checkpoint_path',
               'selected_checkpoint_sha256', 'checkpoint_steps', 'tracking'}
    assert {k for k in new.keys() | old.keys() if new.get(k) != old.get(k)} == allowed
    assert new['lr_restart'] == old['lr_restart']
    assert new['max_steps'] - new['selected_checkpoint_step'] == 2000
    assert new['targets_per_source'] * new['microbatch_per_gpu'] * new['gradient_accumulation'] * 2 == 72


@pytest.mark.parametrize('key,value', [('native_kubric512_b1_a4_k9', False),
    ('native_kubric512_k5_10k', True), ('targets_per_source', 5), ('max_steps', 154769),
    ('native_capacity_test_only', True), ('dataset_mix_counts', dict(kubric=12, pointodyssey=4, dynamic_replica=4))])
def test_invalid_trial_rejected(key, value):
    c = cfg(); c[key] = value
    with pytest.raises(ValueError): validate_config(c, 2)


@pytest.mark.parametrize('bad', [dict(kubric=10, pointodyssey=5),
    dict(kubric=10, pointodyssey=5, dynamic_replica=4),
    dict(kubric=10, pointodyssey=5.0, dynamic_replica=5),
    dict(kubric=20, pointodyssey=0, dynamic_replica=0)])
def test_invalid_composition_rejected(bad):
    with pytest.raises(ValueError): validated_mix_counts(bad)


@pytest.mark.parametrize('seed', [0, 1, 20260812])
def test_legacy_schedule_is_exactly_preserved(seed):
    expected = list(MIX_CYCLE)
    np.random.default_rng(np.random.SeedSequence([seed, 20])).shuffle(expected)
    assert deterministic_dataset_schedule(seed) == tuple(expected)
    assert deterministic_dataset_schedule(seed, dict(kubric=7, pointodyssey=6, dynamic_replica=7)) == tuple(expected)


def test_new_schedule_exact_budget_resume_rng_and_ranks():
    before = pickle.dumps((random.getstate(), np.random.get_state()))
    torch_before = torch.get_rng_state().clone()
    full = [dataset_for_step(i, 20260812, COUNTS) for i in range(152768, 154768)]
    assert Counter(full) == dict(kubric=1000, pointodyssey=500, dynamic_replica=500)
    assert full[117:] == [dataset_for_step(i, 20260812, COUNTS) for i in range(152885, 154768)]
    assert pickle.dumps((random.getstate(), np.random.get_state())) == before
    assert torch.equal(torch_before, torch.get_rng_state())


class FakeDataset:
    rows = []
    def __len__(self): return 101


def test_prefetch_and_latent_plans_agree_with_new_mix(monkeypatch):
    import worldbridge.trainer.batching as batching
    monkeypatch.setattr(batching, 'load_geometry', lambda *args: (args[:2], 0.0))
    datasets = {name: FakeDataset() for name in DATASET_NAMES}
    for rank in (0, 1):
        requests = required_latent_requests(datasets, 20260812, 152768, 152788,
                                            rank, 4, 1, dataset_mix_counts=COUNTS)
        prefetcher = GeometryPrefetcher(datasets, seed=20260812, rank=rank,
            accumulation=4, microbatch_per_gpu=1, targets_per_source=9,
            use_source_rgb=True, start_step=152768, target_steps=152788,
            depth=1, workers=2, dataset_mix_counts=COUNTS)
        try:
            prefetcher.refill()
            actual = []
            for step in range(152768, 152788):
                p = prefetcher.pop(step)
                actual.extend((step, p.dataset_name, i) for i in p.clip_indices)
            assert actual == requests
        finally:
            prefetcher.close()
