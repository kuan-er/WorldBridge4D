#!/usr/bin/env bash
# Wait for preprocessing and physical GPU 4, checking capacity once per minute;
# then supervise training and resume from the latest atomic checkpoint after a
# recoverable process failure.  This process is safe to leave running overnight.
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE="${DYNAMIC_REPLICA_CACHE_ROOT:-/data/WorldBridge4D-persistent/datasets/dynamic_stereo_worldbridge4d_v1}"
OUTPUT="${DYNAMIC_REPLICA_OUTPUT:-/data/WorldBridge4D-persistent/runs/dynamic_replica_dense4d_500k}"
CONFIG="${DYNAMIC_REPLICA_CONFIG:-$ROOT/configs/dynamic_replica_dense4d_capacity_100m_500k.yaml}"
LOG="${DYNAMIC_REPLICA_WATCH_LOG:-$OUTPUT/gpu4_watch.log}"
mkdir -p "$OUTPUT"
exec > >(tee -a "$LOG") 2>&1
exec 8>"$OUTPUT/watch.lock"
if ! flock -n 8; then
  echo "[watch] another GPU4 watcher owns $OUTPUT" >&2
  exit 73
fi
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

status() {
  /opt/conda/bin/python - "$1" "$2" "$3" <<'PY'
import datetime as dt, json, sys
path, state, detail = sys.argv[1:]
value = {"status": state, "detail": detail,
         "updated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
tmp = path + ".tmp"
open(tmp, "w").write(json.dumps(value, indent=2) + "\n")
import os
os.replace(tmp, path)
PY
}

free_gpu4() {
  local free util
  read -r free util < <(nvidia-smi --id=4 --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits | tr ',' ' ')
  free="${free//[!0-9]/}"
  util="${util//[!0-9]/}"
  # Leave headroom for CUDA context fragmentation and require the old owner to
  # have actually released the card, rather than only reporting low utilization.
  (( free >= 70000 && util <= 5 ))
}

wait_for_cache() {
  while [[ ! -s "$CACHE/CACHE_COMPLETE.json" ]]; do
    if [[ -s "$CACHE/PIPELINE_FAILED.json" ]]; then
      echo "[watch] preprocessing reported failure:" >&2
      cat "$CACHE/PIPELINE_FAILED.json" >&2
      status "$OUTPUT/watch_status.json" "preprocessing_failed" "see $CACHE/PIPELINE_FAILED.json"
      sleep 60
    else
      local state="missing"
      if [[ -s "$CACHE/PREPROCESSING_STATE.json" ]]; then
        state="$(/opt/conda/bin/python - "$CACHE/PREPROCESSING_STATE.json" <<'PY'
import json,sys
try: print(json.load(open(sys.argv[1])).get('status','unknown'))
except Exception: print('unreadable')
PY
)"
      fi
      echo "[watch] cache not formally ready (state=$state); next check in 60s"
      status "$OUTPUT/watch_status.json" "waiting_for_preprocessing" "$state"
      sleep 60
    fi
  done
  echo "[watch] validated Dynamic Replica cache is ready"
}

wait_for_gpu4() {
  while ! free_gpu4; do
    nvidia-smi --id=4 --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader || true
    status "$OUTPUT/watch_status.json" "waiting_for_gpu4" "GPU4 free>=70000MiB and util<=5%"
    echo "[watch] GPU4 is occupied; next check in 60s"
    sleep 60
  done
  echo "[watch] GPU4 is available at $(date --iso-8601=seconds)"
}

wait_for_cache
if [[ -s "$OUTPUT/TRAINING_COMPLETE.json" ]]; then
  echo "[watch] training already completed; nothing to launch"
  status "$OUTPUT/watch_status.json" "complete" "TRAINING_COMPLETE.json present"
  exit 0
fi

attempt=0
while [[ ! -s "$OUTPUT/TRAINING_COMPLETE.json" ]]; do
  if [[ -s "$OUTPUT/handoff_in_progress" ]]; then
    echo "[watch] external DDP handoff is taking ownership; watcher exits"
    status "$OUTPUT/watch_status.json" "handoff_ready" "handoff_in_progress present"
    exit 0
  fi
  if [[ -s "$OUTPUT/request_handoff.stop" ]]; then
    echo "[watch] handoff stop requested; checkpoint is retained and watcher exits"
    status "$OUTPUT/watch_status.json" "handoff_ready" "request_handoff.stop present"
    exit 0
  fi
  wait_for_gpu4
  attempt=$((attempt + 1))
  status "$OUTPUT/watch_status.json" "launching" "attempt=$attempt"
  echo "[watch] launching Dynamic Replica single-card training (attempt=$attempt)"
  set +e
  DYNAMIC_REPLICA_CONFIG="$CONFIG" DYNAMIC_REPLICA_OUTPUT="$OUTPUT" DYNAMIC_REPLICA_GPUS=4 \
    "$ROOT/scripts/run_dynamic_replica_training.sh" >> "$OUTPUT/training.log" 2>&1
  rc=$?
  set -e
  if [[ -s "$OUTPUT/TRAINING_COMPLETE.json" ]]; then
    echo "[watch] DYNAMIC_REPLICA_TRAINING_COMPLETE"
    status "$OUTPUT/watch_status.json" "complete" "training metrics written"
    exit 0
  fi
  if [[ -s "$OUTPUT/request_handoff.stop" ]]; then
    echo "[watch] training stopped at an optimizer boundary for DDP handoff"
    status "$OUTPUT/watch_status.json" "handoff_ready" "request_handoff.stop present"
    exit 0
  fi
  echo "[watch] trainer exited rc=$rc; checkpoint, logs, and cache are retained"
  status "$OUTPUT/watch_status.json" "trainer_failed" "rc=$rc; retrying after 60s"
  sleep 60
 done
