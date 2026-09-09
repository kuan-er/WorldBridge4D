"""CPU admission for the explicitly authorized K9 + 50/25/25 joint trial."""
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import yaml
from worldbridge.data.cache.native import file_sha256
from worldbridge.data.factory import load_training_datasets
from worldbridge.data.sampling import deterministic_sample_plan, sample_eligible_targets
from worldbridge.trainer.batching import load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step, deterministic_dataset_schedule
from worldbridge.utils.io import atomic_json

CONFIG = 'configs/h031_k512_k9_mix50_152768_to154768.yaml'
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k9-mix50-cpu-gate-20260909.json')
assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
cfg = yaml.safe_load(Path(CONFIG).read_text())
validate_config(cfg, 2)
origin = Path(cfg['selected_checkpoint_path'])
handoff = json.loads((origin.parent / 'handoff_152768_review.json').read_text())
assert handoff['event'] == 'H031_152768_HANDOFF_FULL_STATE_OK'
assert handoff['sha256'] == cfg['selected_checkpoint_sha256']
assert handoff['step'] == cfg['selected_checkpoint_step'] == 152768
assert file_sha256(origin) == cfg['selected_checkpoint_sha256']
counts = cfg['dataset_mix_counts']
start, end, seed = cfg['selected_checkpoint_step'], cfg['max_steps'], cfg['seed']
assert end - start == 2000
sequence = [dataset_for_step(i, seed, counts) for i in range(start, end)]
assert [sequence.count(n) for n in counts] == [1000, 500, 500]
assert shutil.disk_usage('/data/WorldBridge4D-runs').free >= 68719476736 + 38400000000
began = time.perf_counter()
datasets = load_training_datasets(cfg, allow_missing_latents=False)
requests = []
for rank in (0, 1):
    for name in counts:
        step = start + sequence.index(name)
        dataset = datasets[name]
        index, _, rng = deterministic_sample_plan(dataset, name, seed, step, 0, rank, 4)
        permutation = np.random.default_rng(np.random.SeedSequence([seed, step, 0, rank, 771])).permutation(21)
        fallback_seed = int(np.random.SeedSequence([seed, step, 0, rank, 772]).generate_state(1)[0])
        value, seconds = load_geometry(dataset, index, permutation, 9, fallback_seed,
                                        True, True)
        actual_index, source, xyz, valid, rgb, visible, camera, boundary, contrast = value
        targets = sample_eligible_targets(valid, 9, rng)
        height = 512 if name == 'kubric' else 256
        assert valid.shape == (21, height, height)
        assert len(set(targets.tolist())) == 9
        assert all(valid[t].any() for t in targets)
        assert rgb.shape == (height, height, 3) and rgb.dtype == np.uint8
        assert visible is not None and camera is not None
        assert boundary is None and contrast is None
        latent = dataset.clean_latent(actual_index)
        assert tuple(latent.shape) == (16, 6, height // 8, height // 8)
        row = dict(rank=rank, step=step, dataset=name, requested_index=int(index),
                   actual_index=int(actual_index), source=int(source), targets=targets.tolist(),
                   latent_shape=list(latent.shape), geometry_shape=list(xyz.shape), seconds=seconds)
        requests.append(row)
        print(json.dumps(dict(event='H031_K9_MIX_REAL_INPUT_OK', **row)), flush=True)
report = dict(event='H031_K9_MIX_CPU_OK', config_sha256=file_sha256(CONFIG),
              origin_sha256=cfg['selected_checkpoint_sha256'], start=start, end=end,
              counts_per20=counts, cycle=deterministic_dataset_schedule(seed, counts),
              cases=requests, elapsed_seconds=time.perf_counter()-began,
              scientific_delta='joint_K5_to9_and35_30_35_to50_25_25_not_exact_old_sampler_replay')
atomic_json(OUTPUT, report)
print(json.dumps(report), flush=True)
