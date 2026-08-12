#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml}"
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_three_dataset_200m}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
STEPS="${STEPS:-}"
LAZY_VAE_CACHE="${LAZY_VAE_CACHE:-0}"
EXTRA=()
if [[ "$NPROC" == "2" ]]; then
  EXTRA+=(--allow-two-gpu-gate)
fi
if [[ -n "$STEPS" ]]; then
  EXTRA+=(--steps "$STEPS")
fi
if [[ "$LAZY_VAE_CACHE" == "1" ]]; then
  EXTRA+=(--lazy-vae-cache)
fi
if [[ -f "$OUTPUT/latest.pt" ]]; then
  EXTRA+=(--resume "$OUTPUT/latest.pt")
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  "$ROOT/scripts/train_three_dataset_256_fsdp.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" "${EXTRA[@]}"
