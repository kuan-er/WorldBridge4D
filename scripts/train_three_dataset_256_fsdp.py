#!/usr/bin/env python3
"""Thin CLI for the WorldBridge4D FSDP trainer."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from worldbridge.trainer.config import validate_config
from worldbridge.trainer.distributed import clip_optimizer_grad_norm_, wan_block_auto_wrap_policy
from worldbridge.trainer.fsdp_checkpoint import (
    extend_optimizer_state_for_rgb, filter_optimizer_state_for_trainable,
    load_initial_model_weights, load_model_state_for_resume, prune_periodic_checkpoints,
    update_latest_checkpoint,
)
from worldbridge.trainer.lazy_vae import (
    lazy_latent_owner, pipeline_work_for_rank, required_latent_indices, warm_lazy_latents,
)
from worldbridge.trainer.optimizer import apply_fresh_group_warmup
from worldbridge.trainer.trainer import WorldBridgeTrainer, main

if __name__ == "__main__":
    WorldBridgeTrainer().fit()
