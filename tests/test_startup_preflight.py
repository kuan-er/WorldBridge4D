import json
import pickle
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.numpy import save_file

from worldbridge.data.constants import DATASET_NAMES
from worldbridge.data.cache.native import CONTRACT, NativeLatentCache, file_sha256, json_hash, rgb_identity
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.datasets.native_kubric import NativeKubricDataset
from worldbridge.trainer.lazy_vae import required_latent_indices
from worldbridge.trainer.preflight import preflight_end_step, validate_startup_latents


@pytest.mark.parametrize('start,end,n,expected', [(152768,154768,5,152773),
    (152768,154768,0,154768), (152768,152770,5,152770), (0,2000,5,5),
    (152885,154768,3,152888)])
def test_bound_is_relative_to_resume_and_capped(start, end, n, expected):
    assert preflight_end_step(start, end, n) == expected


@pytest.mark.parametrize('bad', [-1, True, 1.5, '5', None])
def test_invalid_bounds(bad):
    with pytest.raises(ValueError): preflight_end_step(1, 10, bad)


class FakeDataset:
    rows = []
    def __init__(self): self.reads = []
    def __len__(self): return 1001
    def clean_latent(self, index): self.reads.append(index)


@pytest.mark.parametrize('rank', [0, 1])
@pytest.mark.parametrize('mix', [None, dict(kubric=10, pointodyssey=5, dynamic_replica=5)])
def test_default_reads_only_five_updates_with_same_plan_and_rng(rank, mix):
    datasets = {name: FakeDataset() for name in DATASET_NAMES}
    rng = pickle.dumps((random.getstate(), np.random.get_state()))
    torch_rng = torch.get_rng_state().clone()
    report = validate_startup_latents(datasets, 20260812, 152768, 154768, rank, 4,
                                     dataset_mix_counts=mix)
    expected = required_latent_indices(datasets, 20260812, 152768, 152773, rank, 4,
                                       dataset_mix_counts=mix)
    assert {name: d.reads for name, d in datasets.items()} == expected
    assert sum(report['counts'].values()) <= 20
    assert report['execution_end'] == 154768 and report['end'] == 152773
    assert not report['all_planned_payloads_checked']
    assert pickle.dumps((random.getstate(), np.random.get_state())) == rng
    assert torch.equal(torch_rng, torch.get_rng_state())


def test_full_override_and_progress_logging(capsys):
    datasets = {name: FakeDataset() for name in DATASET_NAMES}
    report = validate_startup_latents(datasets, 20260812, 152768, 152788, 0, 4, updates=0)
    assert report['scope'] == 'full' and report['all_planned_payloads_checked']
    expected = required_latent_indices(datasets, 20260812, 152768, 152788, 0, 4)
    assert {name: d.reads for name, d in datasets.items()} == expected
    logs = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert any(x['event'] == 'startup_latent_payload_validation_progress' for x in logs)


@pytest.fixture
def native(tmp_path):
    index = tmp_path / 'index.jsonl'; index.write_text('{}\n{}\n')
    rows = [dict(index=i, clip_id=f'clip{i}', row_sha256=str(i)*64) for i in range(2)]
    manifest = dict(contract=CONTRACT, dataset='kubric', native_hw=[16,32],
                    latent_shape=[16,6,2,4], frames=21, tiling=False,
                    transform='identity_no_resize_crop_pad_or_temporal_resampling',
                    index_path=str(index), index_sha256=file_sha256(index), records=rows)
    manifest['sha256'] = json_hash(manifest)
    d = object.__new__(NativeKubricDataset)
    d.rows = rows
    d.local_rgb = NativeRGBCache(tmp_path/'rgb', manifest)
    d.local_latents = NativeLatentCache(tmp_path/'latent', 'kubric', manifest['sha256'],
                                       (16,6,2,4), 'a'*64)
    rgb = np.zeros((21,16,32,3), dtype=np.uint8)
    latent = np.zeros((16,6,2,4), dtype=np.float32)
    for i in range(2):
        d.local_rgb.write(i, rgb, rgb_identity(rgb))
        d.local_latents.write(i, f'clip{i}', latent, rgb_identity(rgb))
    return d


@pytest.mark.parametrize('kind', ['missing_rgb', 'missing_latent', 'corrupt_rgb', 'corrupt_latent'])
def test_out_of_prefix_failure_is_fatal_at_actual_use(native, monkeypatch, kind):
    import worldbridge.trainer.preflight as preflight
    monkeypatch.setattr(preflight, 'required_latent_indices', lambda *a, **k: {'kubric': [0]})
    member = 'rgb' if kind.endswith('rgb') else 'latent'
    cache = native.local_rgb if member == 'rgb' else native.local_latents
    path = cache.path(1)
    if kind.startswith('missing'):
        path.unlink()
    else:
        with safe_open(str(path), framework='np') as f:
            value = f.get_tensor(member).copy(); metadata = f.metadata()
        value.flat[0] = 1
        save_file({member: value}, str(path), metadata=metadata)
    validate_startup_latents({'kubric': native}, 20260812, 10, 2000, 0, 1)
    with pytest.raises((FileNotFoundError, ValueError)):
        native.clean_latent(1)  # same strict call as the actual batch path, no fallback


def test_prefix_failure_aborts_not_skipped(native, monkeypatch):
    import worldbridge.trainer.preflight as preflight
    monkeypatch.setattr(preflight, 'required_latent_indices', lambda *a, **k: {'kubric': [0]})
    native.local_latents.path(0).unlink()
    with pytest.raises(FileNotFoundError):
        validate_startup_latents({'kubric': native}, 20260812, 10, 2000, 0, 1)


def test_wiring_keeps_catalog_and_actual_input_checks_before_forward():
    s = Path('src/worldbridge/trainer/trainer.py').read_text()
    catalog = s.index('datasets = load_training_datasets(')
    preflight = s.index('startup_preflight = validate_startup_latents(')
    loop = s.index('for step in range(start_step, execution_end):')
    read = s.index('dataset.clean_latent(value[0])', loop)
    readiness = s.index('input_ready(input_ready_group', read)
    forward = s.index('prediction, z4d, model_output = fsdp(', readiness)
    assert catalog < preflight < loop < read < readiness < forward
    assert 'updates=args.startup_preflight_updates, dataset_mix_counts=dataset_mix_counts' in s
    assert 'default=DEFAULT_PREFLIGHT_UPDATES' in s
    assert 'target_steps=execution_end,' in s  # geometry prefetch still spans full execution
    assert '"startup_preflight": startup_preflight' in s
