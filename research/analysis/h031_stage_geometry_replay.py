"""CPU-only materialization of the exact bounded two-rank geometry plan."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import yaml

from worldbridge.data.cache.native import file_sha256
from worldbridge.data.factory import load_training_datasets
from worldbridge.data.sampling import deterministic_sample_plan, sample_eligible_targets
from worldbridge.trainer.batching import GeometryPrefetcher, load_geometry
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.geometry_replay import (GeometryReplay, assert_exact, read_entry,
                                                request_for, request_key, write_entry)
from worldbridge.trainer.lazy_vae import set_lazy_vae_identity
from worldbridge.trainer.schedulers import dataset_for_step
from worldbridge.utils.io import atomic_json

STOP = False

def stop(*_):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    signal.signal(signal.SIGTERM, stop)
    cfg = yaml.safe_load(Path(args.config).read_text()); validate_config(cfg, 2)
    assert cfg['native_kubric512_b1_a4_k15'] and cfg['max_steps'] == 150010
    prior = json.loads((Path(cfg['datasets']['kubric']['native_geometry_root']) / 'cpu_complete.json').read_text())
    assert prior['config_sha256'] == file_sha256(args.config)
    assert prior['origin_sha256'] == cfg['selected_checkpoint_sha256']
    assert file_sha256(cfg['selected_checkpoint_path']) == cfg['selected_checkpoint_sha256']
    assert prior['optimizer_states'] == 193 and prior['decoder_checkpoint_names_shapes_exact']
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'complete.json').exists()
    indexes = {name: file_sha256(Path(values['cache_root']) / 'splits/train.jsonl')
               for name, values in cfg['datasets'].items()}
    datasets = load_training_datasets(cfg)
    set_lazy_vae_identity(datasets, file_sha256(cfg['vae_checkpoint']))
    report = dict(version=1, config_sha256=file_sha256(args.config), indexes=indexes,
                  origin_sha256=prior['origin_sha256'], seed=cfg['seed'], start=150000,
                  stop=150010, world=2, B=1, A=4, K=15, entries={}, count=0,
                  scope='bounded80_geometry_results_not_full_corpus', training_ready=False)
    start = time.perf_counter()
    # Serial preparation avoids the concurrent stream-cache eviction/lock traffic
    # observed in the failed GPU run. It does not change the sampling algorithm.
    for rank in range(2):
        prefetch = GeometryPrefetcher(datasets, seed=cfg['seed'], rank=rank, accumulation=4,
            microbatch_per_gpu=1, targets_per_source=15, use_source_rgb=True, cycle_enabled=True,
            cycle_dataset_names=('kubric','pointodyssey','dynamic_replica'), start_step=150000,
            target_steps=150010, depth=1, workers=1)
        try:
            for step in range(150000,150010):
                name = dataset_for_step(step, cfg['seed']); dataset = datasets[name]
                for slot in range(4):
                    if STOP:
                        print('GEOMETRY_REPLAY_CHECKPOINTED_STOP', flush=True)
                        return 3
                    index, _, rng = deterministic_sample_plan(dataset, name, cfg['seed'], step, slot, rank, 4)
                    request = request_for(prefetch, step, slot, name, index); key = request_key(request)
                    sidecar = root / f'{key}.json'
                    if sidecar.exists():
                        entry = json.loads(sidecar.read_text()); assert entry['request'] == request
                        assert entry['config_sha256'] == report['config_sha256'] and entry['indexes'] == indexes
                        geometry = read_entry(root, entry)
                    else:
                        perm = np.random.default_rng(np.random.SeedSequence([cfg['seed'],step,slot,rank,771])).permutation(21)
                        fallback = int(np.random.SeedSequence([cfg['seed'],step,slot,rank,772]).generate_state(1)[0])
                        geometry, _ = load_geometry(dataset,index,perm,15,fallback,True,True)
                        entry = write_entry(root,request,geometry)
                        entry.update(config_sha256=report['config_sha256'], indexes=indexes)
                        atomic_json(sidecar,entry)
                    replayed = read_entry(root,entry); assert_exact(geometry,replayed)
                    selected, source, xyz, valid, rgb, visible, camera, boundary, contrast = geometry
                    hw = 512 if name == 'kubric' else 256
                    assert xyz.shape == (21,3,hw,hw) and valid.shape == visible.shape == (21,hw,hw)
                    assert rgb.shape == (hw,hw,3) and rgb.dtype == np.uint8 and camera is not None
                    assert boundary is None and contrast is None
                    rng2 = copy.deepcopy(rng)
                    targets = sample_eligible_targets(valid,15,rng)
                    np.testing.assert_array_equal(targets,sample_eligible_targets(replayed[3],15,rng2))
                    assert rng.bit_generator.state == rng2.bit_generator.state
                    latent = dataset.clean_latent(selected)
                    assert latent.shape == (16,6,hw//8,hw//8) and np.isfinite(latent).all()
                    reverse = int(targets[np.argmax(np.abs(targets-source))])
                    reverse_rgb = dataset.source_rgb(selected,reverse)
                    assert reverse_rgb.shape == (hw,hw,3) and reverse_rgb.dtype == np.uint8
                    report['entries'][key] = entry; report['count'] += 1
                    print(json.dumps(dict(event='GEOMETRY_REPLAY_STAGED',count=report['count'],rank=rank,
                        step=step,slot=slot,dataset=name,index=selected,source=source,
                        elapsed_seconds=time.perf_counter()-start)),flush=True)
        finally:
            prefetch.close()
    assert report['count'] == 80
    for name, values in cfg['datasets'].items():
        assert file_sha256(Path(values['cache_root'])/'splits/train.jsonl') == indexes[name]
    # Final publication only after all80 exact readbacks and target RNG checks.
    report['payload_bytes'] = sum(e['payload_bytes'] for e in report['entries'].values())
    report['elapsed_seconds'] = time.perf_counter()-start
    atomic_json(root/'complete.json',report)
    replay = GeometryReplay(root,report['config_sha256'])
    assert len(replay.entries) == 80
    print(json.dumps(dict(event='GEOMETRY_REPLAY_CPU_OK',count=80,payload_bytes=report['payload_bytes'],
                          elapsed_seconds=report['elapsed_seconds'])),flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
