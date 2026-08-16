#!/usr/bin/env bash
# Five-GPU offline VAE latent pre-encoding for the 256px three-dataset route.
# Uses GPUs 0,4,5,6,7 (the low-utilization cards) to encode missing latents in
# parallel, then exits -- it does NOT construct FSDP or train.
#
# Unlike the training launcher, this does not require exclusive GPUs: encoding
# only needs ~2.4 GiB of free memory per rank, so it only checks free memory.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp_2gpu_k16_100k.yaml}"
GPUS="${GPUS:-0,4,5,6,7}"
NPROC="${NPROC:-5}"
STEPS="${STEPS:-100000}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
GPU_FREE_MIN_MIB="${GPU_FREE_MIN_MIB:-3000}"

[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 2; }

IFS=',' read -ra GPU_LIST <<< "$GPUS"
[[ "${#GPU_LIST[@]}" == "$NPROC" ]] || { echo "GPU list count != NPROC" >&2; exit 2; }

for gpu in "${GPU_LIST[@]}"; do
  read -r index free util < <(
    nvidia-smi --id="$gpu" --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr ',' ' '
  )
  if [[ "$index" != "$gpu" ]] || (( free < GPU_FREE_MIN_MIB )); then
    echo "GPU $gpu has insufficient free memory: free=${free}MiB util=${util}%" >&2
    exit 2
  fi
  echo "[preflight] GPU $gpu available, free=${free}MiB util=${util}%"
done

# Fail early if a read-only raw mount has disappeared.
python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml
config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text())
missing = []
for name in ("kubric", "pointodyssey", "dynamic_replica"):
    root = Path(config["datasets"][name]["raw_root"])
    if not root.is_dir():
        missing.append(f"{name}={root}")
if missing:
    raise SystemExit("required raw dataset mounts are unavailable: " + ", ".join(missing))
PY

# Authoritative precomputed latents must remain on durable storage. Refuse
# symlinked or scratch-backed roots before any rank starts encoding.
python "$ROOT/scripts/validate_three_dataset_256_cache_roots.py" \
  --config "$CONFIG" --create

# Stage Wan checkpoints and produce the runtime config (with staged paths).
mapfile -t STAGED < <(python "$ROOT/scripts/stage_three_dataset_256_inputs.py" \
  --config "$CONFIG" --staging-root "$STAGING_ROOT")
if [[ "${#STAGED[@]}" -ne 2 ]]; then
  echo "staging helper returned an invalid response" >&2
  exit 2
fi
CONFIG="${STAGED[0]}"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

EXTRA=()
[[ -n "$STEPS" ]] && EXTRA+=(--steps "$STEPS")

exec torchrun --standalone --nproc-per-node="$NPROC" \
  "$ROOT/scripts/precompute_latents_256.py" \
  --config "$CONFIG" "${EXTRA[@]}"
