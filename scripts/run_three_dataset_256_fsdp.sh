#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/worldbridge4d_256_three_dataset_200m_fsdp.yaml}"
OUTPUT="${OUTPUT:-/data/WorldBridge4D-runs/worldbridge4d_256_three_dataset_200m}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$OUTPUT}"
DURABLE_CHECKPOINT="${DURABLE_CHECKPOINT:-}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
WANDB_LOG_AFTER_STEP="${WANDB_LOG_AFTER_STEP:--1}"
POST_RESUME_CHECKSUM_MARKER="${POST_RESUME_CHECKSUM_MARKER:-}"
GPUS="${GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
STEPS="${STEPS:-}"
LAZY_VAE_CACHE="${LAZY_VAE_CACHE:-0}"
LAZY_VAE_PIPELINE="${LAZY_VAE_PIPELINE:-0}"
PIPELINE_LOOKAHEAD_STEPS="${PIPELINE_LOOKAHEAD_STEPS:-16}"
ALLOW_FOUR_GPU_EXPERIMENT="${ALLOW_FOUR_GPU_EXPERIMENT:-0}"
ALLOW_ARBITRARY_WORLD="${ALLOW_ARBITRARY_WORLD:-0}"
FRESH_START="${FRESH_START:-0}"
STAGE_INPUTS="${STAGE_INPUTS:-1}"
STAGING_ROOT="${STAGING_ROOT:-/data/WorldBridge4D-persistent/worldbridge4d_staging/checkpoints}"
EXTRA=()
if [[ "$NPROC" == "2" ]]; then
  EXTRA+=(--allow-two-gpu-gate)
fi
if [[ "$ALLOW_FOUR_GPU_EXPERIMENT" == "1" ]]; then
  if [[ "$NPROC" != "4" ]]; then
    echo "ALLOW_FOUR_GPU_EXPERIMENT=1 requires NPROC=4" >&2
    exit 2
  fi
  EXTRA+=(--allow-four-gpu-experiment)
fi
if [[ "$ALLOW_ARBITRARY_WORLD" == "1" ]]; then
  EXTRA+=(--allow-arbitrary-world)
fi
if [[ -n "$STEPS" ]]; then
  EXTRA+=(--steps "$STEPS")
fi
if [[ "$LAZY_VAE_CACHE" == "1" && "$LAZY_VAE_PIPELINE" == "1" ]]; then
  echo "LAZY_VAE_CACHE and LAZY_VAE_PIPELINE are mutually exclusive" >&2
  exit 2
fi
if [[ "$LAZY_VAE_CACHE" == "1" ]]; then
  EXTRA+=(--lazy-vae-cache)
fi
if [[ "$LAZY_VAE_PIPELINE" == "1" ]]; then
  EXTRA+=(--lazy-vae-pipeline --pipeline-lookahead-steps "$PIPELINE_LOOKAHEAD_STEPS")
fi
RESUME=""
if [[ "$FRESH_START" == "1" ]]; then
  for artifact in train_status.json wandb_run_id; do
    if [[ -e "$OUTPUT/$artifact" ]]; then
      echo "FRESH_START=1 refuses existing trajectory artifact: $OUTPUT/$artifact" >&2
      exit 2
    fi
  done
  if [[ -e "$CHECKPOINT_DIR/latest.pt" ]]; then
    echo "FRESH_START=1 refuses existing trajectory artifact: $CHECKPOINT_DIR/latest.pt" >&2
    exit 2
  fi
  if compgen -G "$CHECKPOINT_DIR/checkpoint-*.pt" >/dev/null; then
    echo "FRESH_START=1 refuses existing checkpoints under $CHECKPOINT_DIR" >&2
    exit 2
  fi
elif [[ -n "$RESUME_CHECKPOINT" ]]; then
  if [[ ! -f "$RESUME_CHECKPOINT" ]]; then
    echo "RESUME_CHECKPOINT does not exist: $RESUME_CHECKPOINT" >&2
    exit 2
  fi
  RESUME="$RESUME_CHECKPOINT"
elif [[ -f "$CHECKPOINT_DIR/latest.pt" ]]; then
  RESUME="$CHECKPOINT_DIR/latest.pt"
fi

# Fail before torchrun/NCCL initialization when a read-only raw mount has
# disappeared or a copied gate config still names an obsolete mount. Every
# dataset needs raw geometry even when all planned VAE latents are cached.
python - "$CONFIG" <<'PY'
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1])
if not config_path.is_file():
    raise SystemExit(f"three-dataset config is missing: {config_path}")
config = yaml.safe_load(config_path.read_text())
missing = []
for name in ("kubric", "pointodyssey", "dynamic_replica"):
    root = Path(config["datasets"][name]["raw_root"])
    if not root.is_dir():
        missing.append(f"{name}={root}")
if missing:
    raise SystemExit(
        "required raw dataset mounts are unavailable; not launching FSDP: "
        + ", ".join(missing)
    )
PY

# Training consumes authoritative cached model inputs. Never allow a dangling,
# symlinked, or /tmp-backed lazy tier to masquerade as a durable precompute.
python "$ROOT/scripts/validate_three_dataset_256_cache_roots.py" \
  --config "$CONFIG" --create

if [[ "$STAGE_INPUTS" == "1" ]]; then
  STAGE_ARGS=(--config "$CONFIG" --staging-root "$STAGING_ROOT")
  if [[ -n "$RESUME" ]]; then
    STAGE_ARGS+=(--resume "$RESUME")
  fi
  mapfile -t STAGED < <(python "$ROOT/scripts/stage_three_dataset_256_inputs.py" "${STAGE_ARGS[@]}")
  if [[ "${#STAGED[@]}" -ne 2 ]]; then
    echo "staging helper returned an invalid response" >&2
    exit 2
  fi
  CONFIG="${STAGED[0]}"
  RESUME="${STAGED[1]}"
fi
if [[ -n "$RESUME" ]]; then
  EXTRA+=(--resume "$RESUME")
fi
EXTRA+=(--checkpoint-dir "$CHECKPOINT_DIR")
EXTRA+=(--wandb-log-after-step "$WANDB_LOG_AFTER_STEP")
if [[ -n "$POST_RESUME_CHECKSUM_MARKER" ]]; then
  [[ -n "$RESUME" ]] || { echo "post-resume checksum requires a resume checkpoint" >&2; exit 2; }
  EXTRA+=(--post-resume-checksum-marker "$POST_RESUME_CHECKSUM_MARKER")
fi
if [[ -n "$DURABLE_CHECKPOINT" ]]; then
  EXTRA+=(--durable-checkpoint "$DURABLE_CHECKPOINT")
fi

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  "$ROOT/scripts/train_three_dataset_256_fsdp.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" "${EXTRA[@]}"
