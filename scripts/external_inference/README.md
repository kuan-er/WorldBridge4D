# External official-native inference wrappers

This directory contains the standalone VDPM/4RC integration wrappers used by the
three-dataset official-native evaluation protocol. They are not production
WorldBridge4D training entry points and must not receive GT geometry as model
input.

## Layout

- `vdpm_infer_clip.py`: one-clip VDPM inference.
- `vdpm_dataset_worker.py`: resumable VDPM index-range worker.
- `vdpm_smoke.py`: real-checkpoint VDPM smoke test.
- `4rc_infer_clip.py`: one-clip 4RC inference.
- `4rc_dataset_worker.py`: resumable 4RC index-range worker.
- `align_predictions_sim3.py`: deterministic evaluator-side Sim(3) alignment.

The wrappers locate the WorldBridge4D source tree relative to this directory.
Third-party repositories default to `/data/WorldBridge4D-inference`; override
that machine-specific root when necessary:

```bash
export WORLDBRIDGE4D_INFERENCE_ROOT=/path/to/WorldBridge4D-inference
```

Run wrappers from the repository root, for example:

```bash
python scripts/external_inference/vdpm_infer_clip.py --help
python scripts/external_inference/vdpm_dataset_worker.py --help
python scripts/external_inference/4rc_infer_clip.py --help
python scripts/external_inference/4rc_dataset_worker.py --help
python scripts/external_inference/align_predictions_sim3.py --help
```

Historical PRL run manifests retain their original top-level script paths for
auditability; new runs must use the paths in this directory.
