"""Read-only full-state/input gate for ONE diagnostic-only K11 SIGSEGV retry."""
import json
import os
from pathlib import Path
import sys
import torch
import yaml
from worldbridge.data.cache.native import file_sha256, NativeLatentCache
from worldbridge.data.cache.native_rgb import NativeRGBCache
from worldbridge.data.native_inputs import _check_source
from worldbridge.trainer.config import validate_config
from worldbridge.trainer.schedulers import dataset_for_step

BASE = 'configs/h031_k512_k11_mix50_prefix5_to170000.yaml'
CONFIG = 'configs/h031_k512_k11_mix50_prefix5_nostalltrace_to170000.yaml'
ROOT = Path('/data/WorldBridge4D-runs/h031-k11-segv-recovery-gate-20260909')
OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-k11-mix50-prefix5-nostalltrace-to170000-gpu23-20260909')
HANDOFF = Path('/data/WorldBridge4D-runs/h031-k11-capacity-handoff-20260909')
FAILED_OUTPUT = Path('/data/WorldBridge4D-runs/h031-k512-k11-mix50-prefix5-to170000-gpu23-20260909')

def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    cfg = yaml.safe_load(Path(CONFIG).read_text()); base = yaml.safe_load(Path(BASE).read_text())
    validate_config(cfg, 2)
    assert set(cfg) == set(base)
    assert [k for k in cfg if cfg[k] != base[k]] == ['runtime_stall_traceback_seconds']
    assert base['runtime_stall_traceback_seconds'] == 180 and cfg['runtime_stall_traceback_seconds'] == 0
    for pid in (1002652,1002653,1002110):
        assert not Path(f'/proc/{pid}').exists(), 'old worker PID still exists; review ownership before retry'
    assert not list(FAILED_OUTPUT.rglob('*.pt')), 'new checkpoint exists: review newest state instead of rolling back'
    assert not OUTPUT.exists()
    h = json.loads((HANDOFF/'complete.json').read_text())
    assert h['checkpoint_sha256'] == '291c2b889af85a3e089394d204de7b53d2978e8b3201db6bd416b9613f170fcb'
    source = HANDOFF/'resume.pt'; origin = Path(cfg['selected_checkpoint_path'])
    assert file_sha256(source) == h['checkpoint_sha256']
    assert file_sha256(origin) == cfg['selected_checkpoint_sha256']
    c = torch.load(source,map_location='cpu',mmap=True,weights_only=True)
    o = torch.load(origin,map_location='cpu',mmap=True,weights_only=True)
    s = c['training_state']
    assert s['global_step'] == 152774 and s['world_size'] == 2 and len(s['rng_states']) == 2
    assert s['dataset_cycle_offset'] == 14 and s['dataset_mix_phase_origin'] == 152768
    assert s['dataset_mix_counts'] == cfg['dataset_mix_counts']
    assert c['config']['lr_restart'] == cfg['lr_restart']
    assert set(c['model']) == set(o['model']) and all(v.shape == o['model'][k].shape for k,v in c['model'].items())
    assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for k,v in c['model'].items() if k.startswith('decoder.'))
    a,b = c['optimizer']['state'], o['optimizer']['state']
    assert len(a) == len(b) == 193 and set(a) == set(b)
    assert all(int(a[k]['step']) == int(b[k]['step'])+6 for k in a)
    assert all(v['exp_avg'].dtype == v['exp_avg_sq'].dtype == torch.float32 and torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in a.values())
    for name,n in o['training_state']['clips_seen'].items():
        assert s['clips_seen'][name] == n+8*sum(dataset_for_step(i,cfg['seed'],cfg['dataset_mix_counts']) == name for i in range(152768,152774))
    m = json.loads(Path('/data/WorldBridge4D-runs/h031-native-gpu6-handoff-20260908/kubric/kubric.json').read_text())
    rgb = NativeRGBCache('/data/WorldBridge4D-persistent/worldbridge4d-native-rgb-v1',m)
    lat = NativeLatentCache('/data/WorldBridge4D-persistent/worldbridge4d-native-vae-v1','kubric',m['sha256'],tuple(m['latent_shape']),m['vae_sha256'])
    checked = []
    for i in range(3840,3848):
        row = m['records'][i]; _check_source(m,row['path'])
        _, identity = rgb.read(i)
        value = lat.read(i,row['clip_id'],identity)
        assert list(value.shape) == [16,6,64,64]
        checked.append(i)
    report = dict(event='H031_K11_SEGV_GATE_OK', source=str(source.resolve()), checkpoint_sha256=h['checkpoint_sha256'],
        origin=str(origin), origin_sha256=cfg['selected_checkpoint_sha256'], resume_step=152774,
        last_completed_old_run=152860, unsaved_updates_to_replay=86, config=CONFIG, config_sha256=file_sha256(CONFIG),
        baseline_config_sha256=file_sha256(BASE), only_config_delta={'runtime_stall_traceback_seconds':[180,0]},
        model_optimizer_gate='193FP32Adam_plus6_vs152768_names_shapes_decoder_finite_world2_RNG2_counters_LR',
        checked_rgb_latent_indices=checked, input_scope='both_rank_planned_step152860_RGB_latent_strict_SHA_source_identity_not_GT_geometry_or_entire_future_stage',
        fault='rank0_SIGSEGV_before_elastic_cleanup_no_confirmed_OOM',
        hypothesis='periodic_trace_dump_truncated_mid_frame_candidate_not_proven_cause_not_WandB_blame',
        core_available=False, dmesg='permission_denied', scientific_protocol_unchanged=True,
        seed=cfg['seed'], python=sys.version, action='ONE_K11_retry_no_K9_fallback_no_loop_if_recurs', output=str(OUTPUT))
    ROOT.mkdir(exist_ok=False)
    (ROOT/'complete.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__ == '__main__': main()
