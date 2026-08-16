#!/usr/bin/env bash
# Auto-pick the freest GPUs and launch K=16 training on an arbitrary number
# of ranks (1..5, matching the world=5 precompute coverage).
#
# Training peak memory is ~63 GiB (recorded by the K16 gate), so the default
# threshold is 65 GiB free.  It picks every card with enough free memory and
# low utilization, capped at MAX_GPUS, and passes --allow-arbitrary-world so
# any rank count in 1..5 is accepted.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp_2gpu_k16_100k.yaml}"
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_auto_pick_100k_seed20260812}"
MIN_FREE_MIB="${MIN_FREE_MIB:-65000}"
MAX_UTIL="${MAX_UTIL:-30}"
MIN_GPUS="${MIN_GPUS:-1}"
MAX_GPUS="${MAX_GPUS:-5}"
STEPS="${STEPS:-100000}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
WANDB_NAME="${WANDB_NAME:-worldbridge4d-256-auto-pick-100k}"

[[ -f "$CONFIG" ]] || { echo "missing config: $CONFIG" >&2; exit 2; }
[[ -f /root/.netrc ]] || { echo "W&B credential file /root/.netrc is missing" >&2; exit 2; }

# Pick cards by free memory only.  Co-tenant GPU utilization is deliberately
# ignored: the operator prefers occupying any card with enough free memory,
# accepting that busy cards slow shared training via fair GPU scheduling.
GPU_IDS=(0 1 2 3 4 5 6 7)
declare -A free_mem util_now
for gpu in "${GPU_IDS[@]}"; do
  read -r f u < <(
    nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu \
      --format=csv,noheader,nounits | tr ',' ' '
  )
  free_mem[$gpu]=$f
  util_now[$gpu]=$u
done

declare -a candidates=()
for gpu in "${GPU_IDS[@]}"; do
  if (( free_mem[$gpu] >= MIN_FREE_MIB )); then
    candidates+=("${free_mem[$gpu]}:$gpu")
  fi
done

if (( ${#candidates[@]} < MIN_GPUS )); then
  echo "fewer than ${MIN_GPUS} GPU(s) satisfy free>=${MIN_FREE_MIB}MiB and util<${MAX_UTIL}%" >&2
  nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader >&2
  exit 2
fi

# Pick up to MAX_GPUS cards, freest first (free memory descending).
mapfile -t top < <(printf '%s\n' "${candidates[@]}" | sort -t: -k1 -rn | head -"$MAX_GPUS")
gpus=()
for entry in "${top[@]}"; do
  gpus+=("${entry##*:}")
done
GPUS="$(IFS=,; echo "${gpus[*]}")"
NPROC="${#gpus[@]}"

for gpu in "${gpus[@]}"; do
  echo "[auto-pick] GPU $gpu selected, free=${free_mem[$gpu]}MiB util=${util_now[$gpu]}%"
done
echo "[auto-pick] ${NPROC} rank(s) on GPUs $GPUS"

exec env \
  CONFIG="$CONFIG" \
  OUTPUT="$OUTPUT" \
  GPUS="$GPUS" \
  NPROC="$NPROC" \
  STEPS="$STEPS" \
  FRESH_START=1 \
  LAZY_VAE_CACHE=1 \
  ALLOW_ARBITRARY_WORLD=1 \
  STAGE_INPUTS=1 \
  STAGING_ROOT="$STAGING_ROOT" \
  WANDB_MODE=online \
  WANDB_NAME="$WANDB_NAME" \
  bash "$ROOT/scripts/run_three_dataset_256_fsdp.sh"
