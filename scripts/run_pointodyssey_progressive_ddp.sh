#!/usr/bin/env bash
# Start immediately on physical GPU 4, then checkpoint/restart DDP as GPUs 2/6
# become free. PyTorch DDP cannot add a rank in place; each membership change is
# therefore an exact model/optimizer/global-step resume with a larger world.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${POINTODYSSEY_CONFIG:-$ROOT/configs/pointodyssey_dense4d_100m_ddp_30k.yaml}"
OUTPUT="${POINTODYSSEY_OUTPUT:-/data/WorldBridge4D-persistent/pointodyssey_dense4d_ddp_100k}"
STOP_FILE="$OUTPUT/request_scale.stop"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT/capacity"
# A shared stop file and checkpoint make overlapping launchers unsafe. Hold an
# exclusive descriptor for the full capacity/progressive lifecycle so probes or
# retries cannot stop another run or overwrite its exact-resume checkpoint.
exec 9>"$OUTPUT/progressive.lock"
if ! flock -n 9; then
  echo "[progressive] another launcher owns $OUTPUT; refusing concurrent execution" >&2
  exit 73
fi
rm -f "$STOP_FILE"

# Bind the handoff to the current large jobs, while ignoring small monitoring
# contexts. A card is addable only after its original large PID disappears and
# at least 70,000 MiB is free.
declare -A OWNER_PID
for gpu in 2 6; do
  OWNER_PID[$gpu]="$(nvidia-smi --id="$gpu" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F', ' '$2+0 >= 10000 {print $1; exit}')"
  echo "[progressive] GPU $gpu initial owner PID: ${OWNER_PID[$gpu]:-none}"
done

owner_gone() {
  local gpu="$1" pid="${OWNER_PID[$1]}"
  [[ -z "$pid" ]] && return 0
  ! nvidia-smi --id="$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | awk '{print $1}' | grep -qx "$pid"
}
card_free() {
  local gpu="$1" free util
  read -r free util < <(nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr ',' ' ')
  (( free >= 70000 && util <= 5 )) && owner_gone "$gpu"
}

# H020 registers one physical clip per rank. Read that immutable value from the
# YAML and refuse environment/CLI capacity overrides in the formal launcher.
selected="$(python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    print(int(yaml.safe_load(handle)["batch_size_per_gpu"]))
PY
)"
log="$OUTPUT/capacity/gpu4_batch${selected}.log"
echo "[capacity] GPU4 probing fixed batch_size_per_gpu=$selected"
set +e
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 --master_port="$((29740 + selected))" \
  "$ROOT/scripts/train_pointodyssey_ddp.py" --config "$CONFIG" \
  --output-dir "$OUTPUT/capacity/gpu4_batch${selected}" --batch-size-per-gpu "$selected" \
  --steps 1 --disable-wandb --no-checkpoint 2>&1 | tee "$log"
rc=${PIPESTATUS[0]}
set -e
if (( rc != 0 )); then
  echo "[capacity] fixed protocol batch failed; refusing to alter the registered batch" >&2
  exit "$rc"
fi
echo "[capacity] CAPACITY_BATCH_SELECTED=$selected"

visible=(4)
remaining=(2 6)
stage=0
while true; do
  # Add every card that became available before this stage starts.
  next_remaining=()
  for gpu in "${remaining[@]}"; do
    if card_free "$gpu"; then
      visible+=("$gpu")
      echo "[progressive] adding physical GPU $gpu"
    else
      next_remaining+=("$gpu")
    fi
  done
  remaining=("${next_remaining[@]}")
  csv="$(IFS=,; echo "${visible[*]}")"
  world="${#visible[@]}"
  resume_args=()
  [[ -f "$OUTPUT/checkpoint.pt" ]] && resume_args=(--resume "$OUTPUT/checkpoint.pt")
  rm -f "$STOP_FILE"
  echo "[progressive] stage=$stage CUDA_VISIBLE_DEVICES=$csv world_size=$world batch_per_gpu=$selected"
  CUDA_VISIBLE_DEVICES="$csv" WANDB_NAME="pointodyssey-100k-stage${stage}-world${world}" \
    torchrun --standalone --nproc_per_node="$world" --master_port="$((29800 + stage))" \
    "$ROOT/scripts/train_pointodyssey_ddp.py" --config "$CONFIG" --output-dir "$OUTPUT" \
    --batch-size-per-gpu "$selected" --stop-file "$STOP_FILE" "${resume_args[@]}" &
  child=$!

  scale_requested=0
  while kill -0 "$child" 2>/dev/null; do
    if ((${#remaining[@]})); then
      for gpu in "${remaining[@]}"; do
        if card_free "$gpu"; then
          echo "[progressive] GPU $gpu released; requesting checkpointed DDP scale-up"
          touch "$STOP_FILE"; scale_requested=1; break
        fi
      done
    fi
    (( scale_requested )) && break
    sleep 60
  done
  wait "$child"; rc=$?
  (( rc == 0 )) || exit "$rc"
  if (( ! scale_requested )); then
    if [[ -f "$STOP_FILE" ]]; then
      echo "[progressive] unexpected external stop request; checkpoint retained but run is incomplete" >&2
      exit 74
    fi
    echo "[progressive] POINTODYSSEY_PROGRESSIVE_TRAIN_OK"
    exit 0
  fi
  [[ -f "$OUTPUT/checkpoint.pt" ]] || { echo "[progressive] scale checkpoint missing" >&2; exit 1; }
  stage=$((stage + 1))
done
