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

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec torchrun --standalone --nproc-per-node="$NPROC" \
  "$ROOT/scripts/train_three_dataset_256_fsdp.py" \
  --config "$CONFIG" --output-dir "$OUTPUT" "${EXTRA[@]}"
