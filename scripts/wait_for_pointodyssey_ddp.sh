#!/usr/bin/env bash
# Hold the handoff for physical GPUs 2,4,6 without touching any other GPU.
# The initial high-memory PIDs are captured so a different user's later idle
# GPU cannot accidentally trigger the launch.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_ROOT="${POINTODYSSEY_CACHE_ROOT:-/data/WorldBridge4D-persistent/pointodyssey_worldbridge4d_v1}"
CONFIG="${POINTODYSSEY_CONFIG:-$ROOT/configs/pointodyssey_dense4d_100m_ddp_30k.yaml}"
OUTPUT="${POINTODYSSEY_OUTPUT:-/data/WorldBridge4D-persistent/pointodyssey_dense4d_ddp_30k}"
LOG="${POINTODYSSEY_WATCH_LOG:-$OUTPUT/gpu_handoff.log}"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

if [[ "${CUDA_VISIBLE_DEVICES:-}" != "2,4,6" ]]; then
  export CUDA_VISIBLE_DEVICES=2,4,6
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

# Record only substantial jobs present at handoff creation. Small shared CUDA
# contexts (for example the monitoring service) are deliberately ignored.
mapfile -t INITIAL_PIDS < <(nvidia-smi --id=2,4,6 --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2+0 >= 10000 {print $1}' | sort -nu)
printf '[handoff] %s initial high-memory PIDs on 2,4,6: %s\n' "$(date --iso-8601=seconds)" "${INITIAL_PIDS[*]:-none}"

# Build indexes and train-only coordinate statistics immediately on CPU while
# the owner's 30k runs still hold the cards. This never touches CUDA.
if [[ ! -f "$CACHE_ROOT/manifest.json" ]]; then
  echo "[handoff] preparing immutable PointOdyssey index/stat cache: $CACHE_ROOT"
  python "$ROOT/scripts/preprocess_pointodyssey.py" --data-root /dataset/PointOdyssey \
    --output-root "$CACHE_ROOT" --skip-latents --skip-source-hash
fi

pid_present() {
  local pid
  for pid in "${INITIAL_PIDS[@]}"; do
    if nvidia-smi --id=2,4,6 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' | grep -qx "$pid"; then
      return 0
    fi
  done
  return 1
}

ready() {
  local line idx used free util
  local ok=0
  while IFS=',' read -r idx used free util; do
    used="${used//[!0-9]/}"; free="${free//[!0-9]/}"; util="${util//[!0-9]/}"
    # nvidia-smi --id preserves requested order but use index to avoid relying
    # on that implementation detail.
    if [[ "$idx" == "2" || "$idx" == "4" || "$idx" == "6" ]]; then
      if (( free >= 70000 && util <= 5 )); then ok=$((ok + 1)); fi
    fi
  done < <(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits)
  (( ok == 3 )) && ! pid_present
}

while ! ready; do
  nvidia-smi --id=2,4,6 --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits || true
  echo "[handoff] owner jobs still active or GPUs not simultaneously free; next check in 60s"
  sleep 60
done

if [[ -z "${WANDB_API_KEY:-}" ]] && [[ ! -f "$HOME/.netrc" ]]; then
  echo "[handoff] WARNING: no W&B credential found; trainer will keep an offline curve for later 'wandb sync'" >&2
fi
echo "[handoff] GPUs 2,4,6 free at $(date --iso-8601=seconds); launching 3-process DDP now"
exec env CUDA_VISIBLE_DEVICES=2,4,6 PYTHONPATH="$PYTHONPATH" \
  torchrun --standalone --nproc_per_node=3 --master_port="${POINTODYSSEY_MASTER_PORT:-29673}" \
  "$ROOT/scripts/train_pointodyssey_ddp.py" --config "$CONFIG" --output-dir "$OUTPUT"
