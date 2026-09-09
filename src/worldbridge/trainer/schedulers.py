"""Learning-rate and logging schedules."""
from __future__ import annotations

import math

import numpy as np
import torch

from ..data.constants import DATASET_NAMES, MIX_CYCLE

def validated_mix_counts(counts=None) -> dict[str, int]:
    """Explicit 20-update composition; absent config preserves legacy7/6/7."""
    if counts is None:
        return {name: MIX_CYCLE.count(name) for name in DATASET_NAMES}
    if (not isinstance(counts, dict) or set(counts) != set(DATASET_NAMES)
            or any(type(v) is not int or v < 1 for v in counts.values())
            or sum(counts.values()) != 20):
        raise ValueError('dataset_mix_counts requires three positive integer counts totaling20')
    return {name: counts[name] for name in DATASET_NAMES}


def deterministic_dataset_schedule(seed: int, dataset_mix_counts=None) -> tuple[str, ...]:
    """One seeded 20-update cycle, with a byte-compatible legacy default."""
    counts = validated_mix_counts(dataset_mix_counts)
    values = (list(MIX_CYCLE) if counts == validated_mix_counts() else
              [name for name in DATASET_NAMES for _ in range(counts[name])])
    np.random.default_rng(np.random.SeedSequence([int(seed), 20])).shuffle(values)
    assert all(values.count(name) == counts[name] for name in DATASET_NAMES)
    return tuple(values)


def dataset_for_step(global_step: int, seed: int, dataset_mix_counts=None) -> str:
    if int(global_step) < 0:
        raise ValueError("global_step cannot be negative")
    # Every cycle has the same shuffled composition. This makes resume a pure
    # function of global_step while retaining the exact requested ratio.
    return deterministic_dataset_schedule(seed, dataset_mix_counts)[int(global_step) % 20]


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


def apply_lr_restart_schedule(optimizer: torch.optim.Optimizer, update_number: int,
                              restart: dict) -> float:
    """Explicit warmup/hold phase after an exhausted schedule, preserving Adam state.

    Absolute group rates override restored legacy ``_base_lr`` values. The phase
    origin is fixed in config, not the latest resume step, so checkpoint replay
    never restarts warmup or silently reuses the zero-ended cosine trajectory.
    """
    position = int(update_number) - int(restart["start_step"])
    if position < 1 or int(update_number) > int(restart["end_step"]):
        raise ValueError("update outside the declared LR restart phase")
    warmup = int(restart["warmup_steps"])
    if warmup < 1:
        raise ValueError("LR restart requires positive warmup")
    rates = {str(k): float(v) for k, v in restart["group_learning_rates"].items()}
    if {str(group.get("name")) for group in optimizer.param_groups} != set(rates):
        raise ValueError("optimizer groups do not match LR restart rates")
    factor = min(1.0, position / warmup)
    for group in optimizer.param_groups:
        rate = rates[str(group["name"])]
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("LR restart rates must be finite and positive")
        group["_base_lr"] = rate
        group["lr"] = rate * factor
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
