#!/usr/bin/env bash
# Launch (or resume) Dynamic Replica training.  The default is one process on
# physical GPU 4; set DYNAMIC_REPLICA_GPUS=4,2,6 for a later checkpointed DDP
# handoff.  Do not change batch_size_per_gpu when resuming a checkpoint.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${DYNAMIC_REPLICA_CONFIG:-$ROOT/configs/dynamic_replica_dense4d_capacity_100m_500k.yaml}"
OUTPUT="${DYNAMIC_REPLICA_OUTPUT:-/data/WorldBridge4D-persistent/runs/dynamic_replica_dense4d_500k}"
GPUS="${DYNAMIC_REPLICA_GPUS:-4}"
PORT="${DYNAMIC_REPLICA_MASTER_PORT:-29640}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"
exec 9>"$OUTPUT/training.lock"
if ! flock -n 9; then
  echo "[dynamic-replica] another training launcher owns $OUTPUT" >&2
  exit 73
fi

if [[ ! "$GPUS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid DYNAMIC_REPLICA_GPUS=$GPUS (expected e.g. 4 or 4,2,6)" >&2
  exit 2
fi
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
WORLD="${#GPU_LIST[@]}"
export CUDA_VISIBLE_DEVICES="$GPUS"

# This script is intentionally strict about the registered capacity.  A later
# multi-card run changes world size, not the per-card batch, and resumes the
# optimizer/global-step checkpoint.
BATCH="$(/opt/conda/bin/python - "$CONFIG" <<'PY'
import sys, yaml
with open(sys.argv[1]) as f:
    cfg = yaml.safe_load(f)
value = int(cfg.get("batch_size_per_gpu", cfg.get("batch_size", 1)))
if value != 4:
    raise SystemExit(f"registered Dynamic Replica batch must be 4, got {value}")
print(value)
PY
)"

RESUME=()
if [[ -s "$OUTPUT/checkpoint.pt" ]]; then
  RESUME=(--resume "$OUTPUT/checkpoint.pt")
  echo "[dynamic-replica] resuming $(/opt/conda/bin/python - "$OUTPUT/checkpoint_status.json" 2>/dev/null <<'PY' || true
import json,sys
try: print(json.load(open(sys.argv[1])).get('global_step','unknown'))
except Exception: print('unknown')
PY
)"
fi

# The stop file is a deliberate progressive-DDP handoff request.  Leave it in
# place so an outer watcher will not immediately restart the single-card job.
exec torchrun --standalone --nproc_per_node="$WORLD" --master_port="$PORT" \
  "$ROOT/scripts/train_dynamic_replica_ddp.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" --stop-file "$OUTPUT/request_handoff.stop" \
  "${RESUME[@]}" "$@"
