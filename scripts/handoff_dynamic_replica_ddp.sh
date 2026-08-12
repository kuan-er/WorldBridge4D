#!/usr/bin/env bash
# Request a safe optimizer-boundary checkpoint, then resume with a larger DDP
# membership. Example: DYNAMIC_REPLICA_GPUS=4,2,6 this-script
set -Eeuo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${DYNAMIC_REPLICA_OUTPUT:-/data/WorldBridge4D-persistent/runs/dynamic_replica_dense4d_500k}"
GPUS="${DYNAMIC_REPLICA_GPUS:-4,2,6}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUTPUT"

if [[ -s "$OUTPUT/TRAINING_COMPLETE.json" ]]; then
  echo "training is already complete"
  exit 0
fi
# This marker prevents the still-running GPU4 watcher from relaunching a
# single-card job in the small window between the stop checkpoint and handoff.
touch "$OUTPUT/handoff_in_progress" "$OUTPUT/request_handoff.stop"
trap 'rm -f "$OUTPUT/request_handoff.stop" "$OUTPUT/handoff_in_progress"' EXIT

echo "[handoff] waiting for the current process to save and release checkpoint.pt"
# flock is held by the launcher process, so a lock probe is more reliable than
# guessing which torchrun child is rank zero.
exec 9>"$OUTPUT/training.lock"
until flock -n 9; do sleep 10; done
flock -u 9
if [[ ! -s "$OUTPUT/checkpoint.pt" ]]; then
  echo "[handoff] checkpoint.pt was not produced" >&2
  exit 1
fi

rm -f "$OUTPUT/request_handoff.stop"
echo "[handoff] checkpoint ready; launching CUDA_VISIBLE_DEVICES=$GPUS"
DYNAMIC_REPLICA_OUTPUT="$OUTPUT" DYNAMIC_REPLICA_GPUS="$GPUS" \
  "$ROOT/scripts/run_dynamic_replica_training.sh" "$@"
