"""Learning-rate and logging schedules."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.constants import DATASET_NAMES, MIX_CYCLE

def deterministic_dataset_schedule(seed: int) -> tuple[str, ...]:
    """One seeded 20-update cycle containing exactly 7/6/7 datasets."""
    values = list(MIX_CYCLE)
    np.random.default_rng(np.random.SeedSequence([int(seed), 20])).shuffle(values)
    assert values.count("kubric") == 7 and values.count("pointodyssey") == 6
    assert values.count("dynamic_replica") == 7
    return tuple(values)


def dataset_for_step(global_step: int, seed: int) -> str:
    if int(global_step) < 0:
        raise ValueError("global_step cannot be negative")
    # Every cycle has the same shuffled composition. This makes resume a pure
    # function of global_step while retaining the exact requested ratio.
    return deterministic_dataset_schedule(seed)[int(global_step) % 20]


def cosine_learning_rate_factor(update_number: int, warmup_steps: int,
                                horizon_steps: int) -> float:
    update_number = int(update_number)
    warmup_steps = int(warmup_steps)
    horizon_steps = int(horizon_steps)
    if update_number < 1 or warmup_steps < 0 or horizon_steps <= warmup_steps:
        raise ValueError("invalid cosine schedule arguments")
    if warmup_steps and update_number <= warmup_steps:
        return update_number / warmup_steps
    progress = min(1.0, (update_number - warmup_steps) / (horizon_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def extended_cosine_learning_rate_factor(
    update_number: int,
    warmup_steps: int,
    original_horizon_steps: int,
    extension_start_step: int | None = None,
    extension_horizon_steps: int | None = None,
) -> float:
    """Extend a running cosine schedule without an LR jump at the handoff.

    Replacing a 100k horizon with 150k at resume would increase the learning
    rate immediately.  Instead, retain the original factor through the handoff
    and cosine-decay that factor to zero over the added interval.
    """
    if extension_start_step is None and extension_horizon_steps is None:
        return cosine_learning_rate_factor(
            update_number, warmup_steps, original_horizon_steps,
        )
    if extension_start_step is None or extension_horizon_steps is None:
        raise ValueError("both cosine extension steps must be configured")
    extension_start_step = int(extension_start_step)
    extension_horizon_steps = int(extension_horizon_steps)
    if not warmup_steps < extension_start_step < extension_horizon_steps:
        raise ValueError("invalid cosine extension interval")
    if extension_start_step >= original_horizon_steps:
        raise ValueError("cosine extension must start before the original horizon")
    if update_number <= extension_start_step:
        return cosine_learning_rate_factor(
            update_number, warmup_steps, original_horizon_steps,
        )
    start_factor = cosine_learning_rate_factor(
        extension_start_step, warmup_steps, original_horizon_steps,
    )
    progress = min(
        1.0,
        (int(update_number) - extension_start_step)
        / (extension_horizon_steps - extension_start_step),
    )
    return start_factor * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    update_number: int,
    warmup_steps: int,
    horizon_steps: int,
    extension_start_step: int | None = None,
    extension_horizon_steps: int | None = None,
) -> float:
    factor = extended_cosine_learning_rate_factor(
        update_number, warmup_steps, horizon_steps,
        extension_start_step, extension_horizon_steps,
    )
    for group in optimizer.param_groups:
        base_lr = float(group.setdefault("_base_lr", group["lr"]))
        group["lr"] = base_lr * factor
    return factor


def training_diagnostic_due(
    completed_step: int,
    start_step: int,
    target_steps: int,
    dataset_name: str,
    schedule_step: int,
    diagnostic_every: int,
    ensure_dataset_coverage: bool,
    last_logged_cycle: dict[str, int],
) -> bool:
    """Return whether to aggregate/log this update.

    A cadence of five aliases the fixed 20-step mixture cycle and previously
    omitted Dynamic Replica almost completely.  Coverage mode logs at least one
    update from every dataset in every mixture cycle, independent of cadence.
    """
    if dataset_name not in DATASET_NAMES:
        raise ValueError(f"unknown diagnostic dataset: {dataset_name}")
    if diagnostic_every < 1:
        raise ValueError("diagnostic_every must be positive")
    regular = (
        completed_step == start_step + 1
        or completed_step % diagnostic_every == 0
        or completed_step == target_steps
    )
    cycle = int(schedule_step) // len(MIX_CYCLE)
    coverage = (
        ensure_dataset_coverage
        and last_logged_cycle.get(dataset_name) != cycle
    )
    return regular or coverage
