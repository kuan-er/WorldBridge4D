"""CPU-only required gate: unit contracts plus three real planned training samples."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

FILES = [
    'test_source_edge_contrast', 'test_boundary_supervision', 'test_rgb1x_continuation',
    'test_rgb1x_control', 'test_cycle0_continuation', 'test_fp32_cycle0',
    'test_fp32_master', 'test_xyz_only_k15', 'test_native_k15', 'test_native_reuse_10k',
    'test_runtime_controls', 'test_lr_restart', 'test_soft_torchrun', 'test_training256',
    'test_diagnostics', 'test_evaluation', 'test_package_layout',
]
CONFIG = Path('configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast01.yaml')
ROOT = Path('/data/WorldBridge4D-runs/h030-edgecontrast01-preflight-20260907')
OUTPUT = Path('/data/WorldBridge4D-runs/h030-150k-to-155k-boundary2x-edgecontrast01-gpu23-b2-k15')
SUCCESS_MARKER = 'EDGE_CONTRAST01_PREFLIGHT_OK'


def main():
    subprocess.run([sys.executable, '-m', 'pytest', '-q', *['tests/' + f + '.py' for f in FILES]], check=True)
    subprocess.run([sys.executable, '-m', 'compileall', '-q', 'src', 'tests/test_source_edge_contrast.py'], check=True)
    subprocess.run(['git', 'diff', '--check'], check=True)
    subprocess.run(['/opt/node-v22.19.0-linux-x64/bin/node', '-e',
                    "const fs=require('fs');const req=require('module').createRequire('/data/WorldBridge4D/.pi/npm/node_modules/pi-research-loop/package.json');for(const p of process.argv.slice(1))req('yaml').parse(fs.readFileSync(p,'utf8'));",
                    str(CONFIG), 'research/STATE.yaml'], check=True)
    import numpy as np
    import torch
    import yaml
    from worldbridge.data.factory import load_dataset
    from worldbridge.data.sampling import deterministic_sample_plan, sample_eligible_targets
    from worldbridge.trainer.batching import load_geometry
    from worldbridge.trainer.schedulers import dataset_for_step
    from worldbridge.trainer.objective import source_edge_contrast_loss
    from worldbridge.utils.io import atomic_json
    torch.set_num_threads(2)
    cfg = yaml.safe_load(CONFIG.read_text()); seed = cfg['seed']
    checkpoint = torch.load(cfg['selected_checkpoint_path'], map_location='cpu', mmap=True, weights_only=False)
    assert checkpoint['format'] == 3 and checkpoint['training_state']['global_step'] == 150000
    assert checkpoint['training_state']['world_size'] == len(checkpoint['training_state']['rng_states']) == 2
    assert checkpoint['training_state']['clips_seen'] == {'kubric':420000, 'pointodyssey':360000, 'dynamic_replica':420000}
    assert len(checkpoint['optimizer']['state']) == 193
    ages = [int(v['step']) for v in checkpoint['optimizer']['state'].values()]
    assert [min(ages), max(ages)] == [50000, 65000] and checkpoint['config']['seed'] == seed
    seen = set(); samples = []
    for step in range(150000, 150020):
        name = dataset_for_step(step, seed)
        if name in seen: continue
        seen.add(name)
        ds = load_dataset(cfg, name, allow_missing_latents=True)
        index, _, rng = deterministic_sample_plan(ds, name, seed, step, 0, 0, 4)
        perm = np.random.default_rng(np.random.SeedSequence([seed, step, 0, 0, 771])).permutation(21)
        fallback = int(np.random.SeedSequence([seed, step, 0, 0, 772]).generate_state(1)[0])
        a, _ = load_geometry(ds, index, perm, 15, fallback, True, True, cfg['boundary_supervision'])
        b, _ = load_geometry(ds, index, perm, 15, fallback, True, True, cfg['boundary_supervision'], True)
        assert a[:2] == b[:2] and a[8] is None
        for i in [2, 3, 4, 5, 7]: np.testing.assert_array_equal(a[i], b[i])
        assert a[6].keys() == b[6].keys()
        for key in a[6]: np.testing.assert_array_equal(a[6][key], b[6][key])
        ra, rb = deepcopy(rng), deepcopy(rng)
        targets = sample_eligible_targets(a[3], 15, ra)
        np.testing.assert_array_equal(targets, sample_eligible_targets(b[3], 15, rb))
        assert ra.bit_generator.state == rb.bit_generator.state
        e = b[8]; assert e.dtype == bool and e.shape == (2, 256, 256)
        assert not e[0, -1].any() and not e[1, :, -1].any()
        target = torch.from_numpy(b[2][targets][None]).float()
        valid = torch.from_numpy(b[3][targets][None])
        yy, xx = torch.meshgrid(torch.arange(256), torch.arange(256), indexing='ij')
        p = (target + (yy * .002 + xx * .003)[None, None, None]).requires_grad_()
        loss, count, pairs = source_edge_contrast_loss(p, target, valid, torch.from_numpy(e[None]))
        assert count > 0 and pairs > 0 and loss > 0 and torch.isfinite(loss)
        loss.backward(); assert p.grad is not None and torch.isfinite(p.grad).all()
        # Real validity/visibility coverage, never a visibility exclusion.
        occluded_edges = 0
        v, vis = b[3][targets], b[5][targets]
        for axis in (0, 1):
            lo, hi = [slice(None)] * 2, [slice(None)] * 2
            lo[axis], hi[axis] = slice(None, -1), slice(1, None)
            lo, hi = (Ellipsis, *lo), (Ellipsis, *hi)
            mask = e[axis][lo][None] & v[lo] & v[hi]
            occluded_edges += int((mask & (~vis[lo] | ~vis[hi])).sum())
        row = dict(dataset=name, planned_step=step+1, index=b[0], clip_id=ds.rows[b[0]]['clip_id'],
                   source=b[1], targets=targets.tolist(), source_contrast_edges=int(e.sum()),
                   valid_target_edges=int(count), eligible_pairs=int(pairs),
                   edges_with_occluded_valid_endpoint=occluded_edges,
                   edges_sha256=hashlib.sha256(e.tobytes()).hexdigest(),
                   all_existing_arrays_masks_selection_and_rng_equal=True)
        samples.append(row); print(json.dumps({'event':'contrast_real_data_smoke', **row}), flush=True)
        del ds, a, b, p, target, valid
    assert len(samples) == 3 and not torch.cuda.is_initialized()
    assert not OUTPUT.exists()
    atomic_json(ROOT / 'preflight.json', dict(config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
                checkpoint_sha256=cfg['selected_checkpoint_sha256'], checkpoint_step=150000,
                source_edge_contrast_weight=cfg['source_edge_contrast_weight'],
                adam_age_range=[min(ages), max(ages)], samples=samples,
                scope='CPU_loss_gradient_and_real_GT_sampling_gate_not_quality_or_GPU_memory_proof'))
    print(SUCCESS_MARKER, flush=True)


if __name__ == '__main__':
    main()
