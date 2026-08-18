#!/usr/bin/env bash
# Wait for an audited handoff checkpoint and a stable exclusive GPU1/4 window,
# then resume the K19 trajectory with optimized prefetch through step 150k.
# This script never signals or kills the current trainer.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_OUTPUT="${SOURCE_OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_step45005_k19_gpu14_to100k}"
HANDOFF_DIR="${HANDOFF_DIR:-$SOURCE_OUTPUT/optimized_150k_handoff}"
MARKER="${MARKER:-$HANDOFF_DIR/HANDOFF_READY.json}"
OUTPUT="${OUTPUT:-$SOURCE_OUTPUT}"
LOG="${LOG:-$SOURCE_OUTPUT/gpu14_optimized_150k_watch.log}"
LOCK="${LOCK:-/tmp/worldbridge4d-gpu1-4-optimized-150k.lock}"
MIN_FREE_MIB="${MIN_FREE_MIB:-76000}"
MAX_UTIL="${MAX_UTIL:-5}"
POLL_SECONDS="${POLL_SECONDS:-5}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"
TARGET_STEPS="${TARGET_STEPS:-150000}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
WANDB_NAME="${WANDB_NAME:-worldbridge4d-k19-gpu14-optimized-150k}"
LAUNCH_MARKER="$HANDOFF_DIR/LAUNCH_STARTED.json"

[[ "$TARGET_STEPS" == "150000" ]] || { echo "target is frozen at 150000 steps" >&2; exit 2; }
[[ "$POLL_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "POLL_SECONDS must be positive" >&2; exit 2; }
[[ "$STABLE_SECONDS" =~ ^[1-9][0-9]*$ ]] || { echo "STABLE_SECONDS must be positive" >&2; exit 2; }
mkdir -p "$HANDOFF_DIR" "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[watch] another GPU1/4 handoff watcher owns $LOCK" >&2
  exit 73
fi
[[ ! -e "$LAUNCH_MARKER" ]] || {
  echo "[watch] launch marker already exists; refusing a duplicate: $LAUNCH_MARKER" >&2
  exit 74
}

export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

echo "[watch] waiting for an explicitly audited handoff marker; current training is untouched"
while [[ ! -s "$MARKER" ]]; do
  echo "[watch] no $MARKER; next check in 30s"
  sleep 30
done

# Full SHA-256 verification is intentionally completed before waiting for GPUs,
# so no accelerator is left idle while a 9+ GiB checkpoint is audited/staged.
python "$ROOT/scripts/prepare_three_dataset_256_gpu14_handoff.py" \
  --verify-marker "$MARKER" >/dev/null
mapfile -t HANDOFF < <(python - "$MARKER" "$TARGET_STEPS" <<'PY'
import json, pathlib, sys
marker = json.loads(pathlib.Path(sys.argv[1]).read_text())
if marker["physical_gpus"] != [1, 4] or marker["world_size"] != 2:
    raise SystemExit("marker is not pinned to physical GPUs 1,4")
if marker["target_steps"] != int(sys.argv[2]):
    raise SystemExit("marker target-step mismatch")
for key in ("checkpoint", "config", "checkpoint_dir", "completed_step", "checkpoint_sha256"):
    print(marker[key])
PY
)
[[ "${#HANDOFF[@]}" -eq 5 ]] || { echo "invalid handoff marker response" >&2; exit 2; }
RESUME_CHECKPOINT="${HANDOFF[0]}"
CONFIG="${HANDOFF[1]}"
CHECKPOINT_DIR="${HANDOFF[2]}"
RESUME_STEP="${HANDOFF[3]}"
CHECKPOINT_SHA256="${HANDOFF[4]}"
CHECKPOINT_IDENTITY="$(stat -Lc '%d:%i:%s:%Y' "$RESUME_CHECKPOINT")"

[[ -f /root/.netrc || -n "${WANDB_API_KEY:-}" ]] || {
  echo "[watch] online W&B credential is missing" >&2; exit 2;
}
[[ -f "$OUTPUT/wandb_run_id" ]] || {
  echo "[watch] existing W&B run identity is missing: $OUTPUT/wandb_run_id" >&2; exit 2;
}

# Validate all raw mounts and durable cache roots before cards become available.
python - "$CONFIG" <<'PY'
from pathlib import Path
import sys, yaml
config = yaml.safe_load(Path(sys.argv[1]).read_text())
missing = []
for name in ("kubric", "pointodyssey", "dynamic_replica"):
    root = Path(config["datasets"][name]["raw_root"])
    if not root.is_dir():
        missing.append(f"{name}={root}")
if missing:
    raise SystemExit("missing raw dataset mounts: " + ", ".join(missing))
PY
python "$ROOT/scripts/validate_three_dataset_256_cache_roots.py" --config "$CONFIG" --create

# Stage the config/model inputs and immutable resume checkpoint now, while the
# current owner still occupies GPU1/4. The final launcher therefore performs no
# multi-gigabyte staging inside the GPU acquisition race window.
mapfile -t STAGED < <(python "$ROOT/scripts/stage_three_dataset_256_inputs.py" \
  --config "$CONFIG" --resume "$RESUME_CHECKPOINT" --staging-root "$STAGING_ROOT")
[[ "${#STAGED[@]}" -eq 2 ]] || { echo "staging helper returned an invalid response" >&2; exit 2; }
STAGED_CONFIG="${STAGED[0]}"
STAGED_RESUME="${STAGED[1]}"
[[ -f "$STAGED_CONFIG" && -f "$STAGED_RESUME" ]] || { echo "staged inputs disappeared" >&2; exit 2; }

gpu_ready() {
  local gpu index free util
  for gpu in 1 4; do
    read -r index free util < <(
      nvidia-smi --id="$gpu" --query-gpu=index,memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr ',' ' '
    )
    index="${index//[!0-9]/}"; free="${free//[!0-9]/}"; util="${util//[!0-9]/}"
    [[ "$index" == "$gpu" ]] || return 1
    (( free >= MIN_FREE_MIB && util <= MAX_UTIL )) || return 1
    mapfile -t pids < <(
      nvidia-smi --id="$gpu" --query-compute-apps=pid \
        --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF && $1 != "[N/A]" {print $1}'
    )
    (( ${#pids[@]} == 0 )) || return 1
  done
}

stable=0
while (( stable < STABLE_SECONDS )); do
  if gpu_ready; then
    stable=$((stable + POLL_SECONDS))
    echo "[watch] GPU1/4 exclusively idle for ${stable}/${STABLE_SECONDS}s"
  else
    stable=0
    nvidia-smi --id=1,4 --query-gpu=index,memory.used,memory.free,utilization.gpu \
      --format=csv,noheader || true
    echo "[watch] GPU1/4 not simultaneously exclusive; stability timer reset"
  fi
  (( stable >= STABLE_SECONDS )) || sleep "$POLL_SECONDS"
done

# Close the last avoidable TOCTOU window: revalidate checkpoint identity,
# marker metadata, and both cards immediately before torchrun.
python "$ROOT/scripts/prepare_three_dataset_256_gpu14_handoff.py" \
  --verify-marker "$MARKER" --skip-checksum >/dev/null
[[ "$(stat -Lc '%d:%i:%s:%Y' "$RESUME_CHECKPOINT")" == "$CHECKPOINT_IDENTITY" ]] || {
  echo "[watch] immutable checkpoint identity changed" >&2; exit 2;
}
gpu_ready || { echo "[watch] GPU1/4 changed ownership during final preflight" >&2; exit 75; }

python - "$LAUNCH_MARKER" "$RESUME_STEP" "$CHECKPOINT_SHA256" "$CONFIG" <<'PY'
from datetime import datetime, timezone
import json, os, pathlib, socket, sys
path, step, checksum, config = sys.argv[1:]
value = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "host": socket.gethostname(), "physical_gpus": [1, 4], "world_size": 2,
    "resume_step": int(step), "checkpoint_sha256": checksum,
    "target_steps": 150000, "config": str(pathlib.Path(config).resolve()),
}
p = pathlib.Path(path); tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
tmp.write_text(json.dumps(value, indent=2) + "\n")
os.replace(tmp, p)
PY

echo "[watch] exclusive window passed; launching exact K19 resume step=$RESUME_STEP to 150000"
echo "[watch] advisory flock prevents duplicate project launchers; unmanaged jobs can only be excluded by a scheduler"
exec env \
  CONFIG="$STAGED_CONFIG" \
  OUTPUT="$OUTPUT" \
  CHECKPOINT_DIR="$CHECKPOINT_DIR" \
  DURABLE_CHECKPOINT="$SOURCE_OUTPUT/latest.pt" \
  RESUME_CHECKPOINT="$STAGED_RESUME" \
  GPUS=1,4 \
  NPROC=2 \
  STEPS=150000 \
  FRESH_START=0 \
  LAZY_VAE_CACHE=1 \
  STAGE_INPUTS=0 \
  WANDB_LOG_AFTER_STEP="$RESUME_STEP" \
  WANDB_MODE=online \
  WANDB_NAME="$WANDB_NAME" \
  bash "$ROOT/scripts/run_three_dataset_256_fsdp.sh"
