"""Checkpoint and deterministic RNG state utilities."""
from __future__ import annotations

import copy
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from ..models.worldbridge import DenseQueryWanModel

def capture_rng_state(numpy_generator: np.random.Generator | None = None,
                      include_cuda: bool = True) -> dict[str, Any]:
    """Capture restart-safe Python, NumPy, and Torch RNG state."""
    numpy_global = np.random.get_state()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": {
            "bit_generator": numpy_global[0],
            # PyTorch 2.4/2.5 cannot serialize TypedStorage(torch.uint32).
            # int64 is weights_only-safe and restore_rng_state casts back to
            # NumPy's required uint32 representation exactly.
            "state": torch.from_numpy(numpy_global[1].astype(np.int64, copy=True)),
            "position": int(numpy_global[2]),
            "has_gauss": int(numpy_global[3]),
            "cached_gaussian": float(numpy_global[4]),
        },
        "torch_cpu": torch.get_rng_state(),
    }
    if numpy_generator is not None:
        state["numpy_generator"] = copy.deepcopy(numpy_generator.bit_generator.state)
    if include_cuda and torch.cuda.is_available():
        # Dense4D is single-device; storing only the active device keeps exact
        # resume portable across different CUDA_VISIBLE_DEVICES layouts.
        state["torch_cuda"] = torch.cuda.get_rng_state()
    return state


def restore_rng_state(state: dict[str, Any], numpy_generator: np.random.Generator | None = None) -> None:
    """Restore state captured by :func:`capture_rng_state`."""
    required = {"python", "numpy_global", "torch_cpu"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"checkpoint RNG state is missing {missing}")
    random.setstate(state["python"])
    numpy_global = state["numpy_global"]
    np.random.set_state((
        str(numpy_global["bit_generator"]),
        torch.as_tensor(numpy_global["state"]).cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_global["position"]), int(numpy_global["has_gauss"]),
        float(numpy_global["cached_gaussian"]),
    ))
    if numpy_generator is not None:
        if "numpy_generator" not in state:
            raise ValueError("checkpoint RNG state has no local NumPy generator")
        numpy_generator.bit_generator.state = copy.deepcopy(state["numpy_generator"])
    torch.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8).cpu())
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(torch.as_tensor(cuda_state, dtype=torch.uint8).cpu())


def save_checkpoint(path: str | Path, model: DenseQueryWanModel, config: dict[str, Any],
                    coordinate_mean: np.ndarray, coordinate_scale: np.ndarray,
                    extra: dict[str, Any] | None = None,
                    optimizer: torch.optim.Optimizer | None = None,
                    training_state: dict[str, Any] | None = None) -> Path:
    """Atomically save model state and optional exact-resume state."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": 2,
        "model": model.state_dict(), "config": config,
        # Plain lists keep weights_only=True checkpoint validation safe on
        # PyTorch 2.6+; no NumPy reconstruction globals are needed.
        "coordinate_mean": np.asarray(coordinate_mean, np.float32).tolist(),
        "coordinate_scale": np.asarray(coordinate_scale, np.float32).tolist(),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if training_state is not None:
        payload["training_state"] = training_state
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path
