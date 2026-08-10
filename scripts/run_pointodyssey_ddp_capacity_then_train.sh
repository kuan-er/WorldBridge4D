#!/usr/bin/env bash
# One-step real-model capacity gate followed immediately by the 30k DDP run.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${POINTODYSSEY_CONFIG:-$ROOT/configs/pointodyssey_dense4d_100m_ddp_30k.yaml}"
OUTPUT="${POINTODYSSEY_OUTPUT:-/data/WorldBridge4D-persistent/pointodyssey_dense4d_ddp_30k}"
export CUDA_VISIBLE_DEVICES=2,4,6
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/capacity"

# Descending candidates keep the cards claimed while finding the largest
# successful physical batch.  A failed candidate is only a capacity probe, not
# a model result; the first successful candidate is used for all 30k updates.
CANDIDATES=(8 6 4 3 2 1)
if [[ -n "${POINTODYSSEY_BATCH:-}" ]]; then CANDIDATES=("$POINTODYSSEY_BATCH")
fi
selected=""
for batch in "${CANDIDATES[@]}"; do
  log="$OUTPUT/capacity/batch${batch}.log"
  echo "[capacity] probing batch_size_per_gpu=$batch"
  set +e
  torchrun --standalone --nproc_per_node=3 --master_port="$((29673 + batch))" \
    "$ROOT/scripts/train_pointodyssey_ddp.py" --config "$CONFIG" \
    --output-dir "$OUTPUT/capacity/batch${batch}" --batch-size-per-gpu "$batch" \
    --steps 1 --disable-wandb --no-checkpoint 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 0 )); then
    selected="$batch"
    echo "[capacity] CAPACITY_BATCH_SELECTED=$selected"
    break
  fi
  if ! grep -Eiq 'out of memory|CUBLAS_STATUS_ALLOC_FAILED|CUDA error' "$log"; then
    echo "[capacity] non-capacity failure at batch=$batch; refusing to guess" >&2
    exit "$rc"
  fi
  echo "[capacity] batch=$batch failed capacity probe; trying next lower candidate"
  sleep 5
done
if [[ -z "$selected" ]]; then
  echo "[capacity] no candidate fit on GPUs 2,4,6" >&2
  exit 1
fi

# New process group, same held cards, exact requested 30,000 optimizer steps.
exec torchrun --standalone --nproc_per_node=3 --master_port=29673 \
  "$ROOT/scripts/train_pointodyssey_ddp.py" --config "$CONFIG" \
  --output-dir "$OUTPUT" --batch-size-per-gpu "$selected"
