"""Training statistics and optional experiment tracking."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

def load_stats(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(config["mixture_coordinate_stats"])
    if not path.is_file():
        raise FileNotFoundError(f"train-only mixture coordinate statistics missing: {path}")
    with np.load(path) as values:
        mean = np.asarray(values["mean"], np.float32).reshape(3)
        scale = np.asarray(values["scale"], np.float32).reshape(3)
        if str(values.get("coordinate_frame", "")) != "source":
            raise ValueError("mixture stats must use the source camera frame")
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("invalid mixture coordinate statistics")
    return mean, scale


def init_wandb(config: dict[str, Any], output: Path, rank: int, disabled: bool):
    tracking = config.get("tracking", {})
    if rank != 0 or disabled or not tracking.get("enabled", True) or os.environ.get("WANDB_MODE") == "disabled":
        return None
    import wandb
    run_id_path = output / "wandb_run_id"
    resume_policy = tracking.get("resume", "allow")
    if resume_policy not in {"allow", "must"}:
        raise ValueError("tracking.resume must be allow or must")
    if resume_policy == "must" and not run_id_path.is_file():
        raise ValueError("strict W&B resume requires an existing wandb_run_id")
    run_id = run_id_path.read_text().strip() if run_id_path.exists() else wandb.util.generate_id()
    run_id_path.write_text(run_id + "\n")
    mode = os.environ.get("WANDB_MODE", "online")
    if mode == "online" and not os.environ.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
        mode = "offline"
    if resume_policy == "must" and mode != "online":
        raise ValueError("strict W&B resume requires authenticated online mode")
    run = wandb.init(
        id=run_id, resume=resume_policy, mode=mode, project=tracking.get("project", "worldbridge4d"),
        entity=tracking.get("entity"), group=tracking.get("group"), tags=tracking.get("tags"),
        name=os.environ.get("WANDB_NAME", "worldbridge4d-256-three-dataset-200m"), config=config,
    )
    if resume_policy == "must" and (run.id != run_id or not run.resumed):
        run.finish(exit_code=1)
        raise RuntimeError("W&B did not resume the requested existing run")
    run.define_metric("global_step")
    run.define_metric("train/*", step_metric="global_step")
    run.define_metric("train/loss_by_dataset/*", step_metric="global_step")
    run.define_metric("train/raw_epe_m_by_dataset/*", step_metric="global_step")
    run.define_metric("train/cycle_reprojection_loss_by_dataset/*", step_metric="global_step")
    run.define_metric("train/xyz_loss_by_dataset/*", step_metric="global_step")
    run.define_metric("train/weighted_cycle_reprojection_loss_by_dataset/*", step_metric="global_step")
    run.define_metric("train/cycle_reprojection_loss_ratio_by_dataset/*", step_metric="global_step")
    run.define_metric("train/cycle_reprojection_scale_by_dataset/*", step_metric="global_step")
    run.define_metric("system/*", step_metric="global_step")
    run.define_metric("sampling/*", step_metric="global_step")
    return run
