#!/usr/bin/env bash
# Fresh experimental four-GPU K=10 trajectory on physical GPUs 2,4,5,6.
# This script never kills GPU owners. The operator must release the cards first.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp_4gpu_k10_10k.yaml}"
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_four_gpu_b2_k10_a1_10k_seed20260812}"
GPUS="${GPUS:-2,4,5,6}"
STEPS="${STEPS:-10000}"
GPU_FREE_MIN_MIB="${GPU_FREE_MIN_MIB:-55000}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
WANDB_NAME="${WANDB_NAME:-worldbridge4d-256-4gpu-b2-k10-a1-fresh-10k}"

if [[ "$GPUS" != "2,4,5,6" ]]; then
  echo "refusing unexpected GPUS=$GPUS; this handoff is pinned to physical GPUs 2,4,5,6" >&2
  exit 2
fi
if [[ "$STEPS" != "10000" ]]; then
  echo "refusing STEPS=$STEPS; this launcher is the frozen fresh 10k trajectory" >&2
  exit 2
fi
if [[ ! -f "$CONFIG" ]]; then
  echo "missing config: $CONFIG" >&2
  exit 2
fi
if [[ ! -f /root/.netrc ]]; then
  echo "W&B credential file /root/.netrc is missing" >&2
  exit 2
fi

IFS=',' read -r -a gpu_list <<< "$GPUS"
if [[ "${#gpu_list[@]}" -ne 4 ]]; then
  echo "expected exactly four physical GPUs, got $GPUS" >&2
  exit 2
fi
for gpu in "${gpu_list[@]}"; do
  read -r index free util < <(
    nvidia-smi --id="$gpu" \
      --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr ',' ' '
  )
  if [[ "$index" != "$gpu" ]]; then
    echo "NVML returned GPU $index while checking requested GPU $gpu" >&2
    exit 2
  fi
  mapfile -t compute_pids < <(
    nvidia-smi --id="$gpu" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | awk 'NF && $1 != "[N/A]" {print $1}'
  )
  if (( free < GPU_FREE_MIN_MIB )); then
    echo "GPU $gpu has insufficient capacity: free=${free}MiB, require >=${GPU_FREE_MIN_MIB}MiB, existing_compute_pids=${compute_pids[*]:-none}" >&2
    exit 2
  fi
  echo "[preflight] GPU $gpu free=${free}MiB util=${util}% existing_compute_pids=${compute_pids[*]:-none}"
done

mkdir -p "$OUTPUT"
manifest="$OUTPUT/fresh_launch_manifest.json"
if [[ -e "$manifest" ]]; then
  echo "fresh launch manifest already exists: $manifest" >&2
  exit 2
fi
python - "$manifest" "$CONFIG" "$OUTPUT" "$GPUS" "$STEPS" <<'PY'
import json
from pathlib import Path
import socket
import sys
from datetime import datetime, timezone

path, config, output, gpus, steps = sys.argv[1:]
value = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(),
    "config": str(Path(config).resolve()),
    "output": str(Path(output).resolve()),
    "physical_gpus": [int(x) for x in gpus.split(",")],
    "world_size": 4,
    "fsdp": "full_shard",
    "fresh_start": True,
    "seed": 20260812,
    "targets_per_source": 10,
    "microbatch_per_gpu": 2,
    "gradient_accumulation": 1,
    "clips_per_update": 8,
    "pairs_per_update": 80,
    "target_steps": int(steps),
    "wandb_required": True,
}
Path(path).write_text(json.dumps(value, indent=2) + "\n")
PY

exec env \
  CONFIG="$CONFIG" \
  OUTPUT="$OUTPUT" \
  GPUS="$GPUS" \
  NPROC=4 \
  STEPS="$STEPS" \
  ALLOW_FOUR_GPU_EXPERIMENT=1 \
  FRESH_START=1 \
  LAZY_VAE_PIPELINE=1 \
  PIPELINE_LOOKAHEAD_STEPS=1 \
  STAGE_INPUTS=1 \
  STAGING_ROOT="$STAGING_ROOT" \
  WANDB_MODE=online \
  WANDB_NAME="$WANDB_NAME" \
  bash "$ROOT/scripts/run_three_dataset_256_fsdp.sh"
