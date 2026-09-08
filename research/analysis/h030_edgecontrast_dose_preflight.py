"""Required CPU-only 0.01 dose gate; reuse the original real-input smoke protocol."""
import json
from pathlib import Path
import subprocess

import h030_edgecontrast_preflight as gate


def main():
    import yaml
    from worldbridge.utils.io import atomic_json
    owner = '01a07615-f347-712a-910e-7f7eea16d709'
    for run in ['R-20260907232029-421c94', 'R-20260907231637-82ea48']:
        meta = yaml.safe_load((Path('/data/WorldBridge4D/runs') / run / 'run.yaml').read_text())
        assert meta['owner']['session_id'] == owner and meta['status'] == 'succeeded'
    changed = subprocess.check_output(['git', 'diff', '--name-only',
        '42869cccfccc316f79268e2b887b1ff78574e3e8', 'HEAD', '--', 'src'], text=True).splitlines()
    assert changed == ['src/worldbridge/trainer/config.py']
    gate.CONFIG = Path('configs/h030_150k_to_155k_gpu23_b2_k15_boundary2x_edgecontrast0p01.yaml')
    gate.ROOT = Path('/data/WorldBridge4D-runs/h030-edgecontrast0p01-preflight-20260907')
    gate.OUTPUT = Path('/data/WorldBridge4D-runs/h030-150k-to-155k-boundary2x-edgecontrast0p01-gpu23-b2-k15')
    gate.FILES += ['test_edgecontrast_dose', 'test_boundary_log_review', 'test_crossed_validation']
    gate.SUCCESS_MARKER = 'EDGE_CONTRAST0P01_REAL_INPUT_CHECKS_DONE'
    assert not gate.ROOT.exists()
    gate.main()
    report = json.loads((gate.ROOT / 'preflight.json').read_text())
    previous = json.loads(Path('/data/WorldBridge4D-runs/h030-edgecontrast01-preflight-20260907/preflight.json').read_text())
    assert report['source_edge_contrast_weight'] == .01
    assert report['checkpoint_sha256'] == previous['checkpoint_sha256']
    assert report['samples'] == previous['samples']
    report.update(previous_gate='R-20260907184251-dab560', same_three_real_samples_as_weight01=True,
                  only_runtime_source_delta='config_validator_admits_fixed001_no_model_loss_or_data_code_change')
    atomic_json(gate.ROOT / 'preflight.json', report)
    print(json.dumps(report), flush=True)
    print('EDGE_CONTRAST0P01_PREFLIGHT_OK', flush=True)


if __name__ == '__main__':
    main()
