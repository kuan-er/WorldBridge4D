#!/usr/bin/env bash
# Two-GPU K=16 trajectory using offline two-card parallel VAE pre-encoding
# (--lazy-vae-cache) instead of the streaming pipeline.  The two ranks split
# the missing latents, encode them on their own GPU in parallel, release the
# VAE, and only then start training -- so training reads latents from the
# configured durable cache roots rather than re-reading raw RGB.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp_2gpu_k16_100k.yaml}"
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_two_gpu_b2_k16_a2_100k_seed20260812_precache}"
GPUS="${GPUS:-4,6}"
STEPS="${STEPS:-100000}"
GPU_FREE_MIN_MIB="${GPU_FREE_MIN_MIB:-76000}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
WANDB_NAME="${WANDB_NAME:-worldbridge4d-256-2gpu-b2-k16-a2-precache-100k}"

[[ "$GPUS" == "4,6" ]] || { echo "this trajectory is pinned to physical GPUs 4,6" >&2; exit 2; }
[[ "$STEPS" == "100000" ]] || { echo "this trajectory is frozen at 100000 steps" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 2; }
[[ -f /root/.netrc ]] || { echo "W&B credential file /root/.netrc is missing" >&2; exit 2; }

for gpu in 4 6; do
  read -r index free util < <(
    nvidia-smi --id="$gpu" --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr ',' ' '
  )
  mapfile -t compute_pids < <(
    nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NF && $1 != "[N/A]" {print $1}'
  )
  if [[ "$index" != "$gpu" ]] || (( free < GPU_FREE_MIN_MIB )) || (( ${#compute_pids[@]} != 0 )); then
    echo "GPU $gpu is not exclusively free: free=${free}MiB util=${util}% pids=${compute_pids[*]:-none}" >&2
    exit 2
  fi
  echo "[preflight] GPU $gpu exclusive, free=${free}MiB util=${util}%"
done

mkdir -p "$OUTPUT"
manifest="$OUTPUT/fresh_launch_manifest.json"
[[ ! -e "$manifest" ]] || { echo "fresh launch manifest already exists: $manifest" >&2; exit 2; }
python - "$manifest" "$CONFIG" "$OUTPUT" <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import socket
import sys
path, config, output = sys.argv[1:]
value = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "config": str(Path(config).resolve()),
    "output": str(Path(output).resolve()),
    "physical_gpus": [4, 6],
    "world_size": 2,
    "fsdp": "full_shard",
    "fresh_start": True,
    "seed": 20260812,
    "targets_per_source": 16,
    "microbatch_per_gpu": 2,
    "gradient_accumulation": 2,
    "clips_per_update": 8,
    "pairs_per_update": 128,
    "target_steps": 100000,
    "checkpoint_steps": [50, 100, 250, 500, 1000, 2000, 5000, 10000],
    "checkpoint_every_after": 5000,
    "lazy_mode": "lazy_vae_cache",
    "wandb_required": True,
}
Path(path).write_text(json.dumps(value, indent=2) + "\n")
PY

exec env \
  CONFIG="$CONFIG" \
  OUTPUT="$OUTPUT" \
  GPUS="$GPUS" \
  NPROC=2 \
  STEPS="$STEPS" \
  FRESH_START=1 \
  LAZY_VAE_CACHE=1 \
  STAGE_INPUTS=1 \
  STAGING_ROOT="$STAGING_ROOT" \
  WANDB_MODE=online \
  WANDB_NAME="$WANDB_NAME" \
  bash "$ROOT/scripts/run_three_dataset_256_fsdp.sh"
