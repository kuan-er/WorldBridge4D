# WorldBridge4D environment

## Supported baseline

The validated environment is Linux x86_64, Python 3.10.13, CUDA-enabled PyTorch
2.10.0+cu126, TensorFlow CPU 2.15.1, and the direct-package versions in
`constraints-known-good-cu126.txt`. The training target is an NVIDIA 80 GiB
GPU; CUDA/PyTorch wheels must match the host driver and GPU.

TensorFlow is used only for read-only MOVi-F TFRecord parsing. Install
`tensorflow-cpu`, not the GPU TensorFlow package, so dataset workers cannot
reserve training GPU memory.

## Fresh environment on the current CUDA 12.6 host

```bash
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip

# Install the CUDA-enabled wheel before the general requirements. For another
# CUDA/driver combination, select the matching official PyTorch index instead.
python -m pip install --index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.10.0'
python -m pip install -r requirements.txt \
  -c constraints-known-good-cu126.txt

PYTHONPATH=src python scripts/check_environment.py --require-cuda
PYTHONPATH=src pytest -q
```

If the target machine cannot use CUDA 12.6, do not blindly reuse the CUDA
constraint file. Install a compatible PyTorch wheel first, then install
`requirements.txt` without the CUDA constraint file (or make a new constraint
file that records the selected wheel and CUDA build).

## Non-Python inputs

Python packages do not contain the model or data. The training machine also
needs:

```text
/dataset/MOVi-F                         # read-only MOVi-F TFRecords
/dataset/Wan2.1-T2V-1.3B                # Wan VAE/DiT and UMT5 files
/tmp/worldbridge_dense4d/wan_empty_condition.pt
/tmp/worldbridge_dense4d/coordinate_stats_train_source.npz
```

`wan_empty_condition.pt` is generated once by
`scripts/create_wan_empty_text_condition.py` and requires a checked-out native
Wan2.1 source tree via `WAN_SOURCE_ROOT` or `--wan-source`. The source tree and
checkpoint files are external inputs and must be recorded by commit/checksum,
not added to Git.

For W&B, run `wandb login` on the machine or provide `WANDB_API_KEY` in the
process environment. Never put the key in Git or in a committed `.env` file.

## Environment gate

`PYTHONPATH=src python scripts/check_environment.py` checks every direct Python
package, the Diffusers Wan APIs, TensorFlow TFRecord access, the manifest schema,
and project imports. Add `--require-cuda` for a training host; it additionally
checks CUDA availability, BF16 support and NVML. This gate does not load model
weights or require the dataset, so it is safe to run before data provisioning.
