"""Deterministic clip and source-target sampling."""
from __future__ import annotations

import numpy as np

from .constants import DATASET_NAMES
from .types import TrainingDataset

def source_with_eligible_targets(dataset: TrainingDataset, index: int,
                                 sources: np.ndarray, min_targets: int = 1
                                 ) -> tuple[int, np.ndarray, np.ndarray]:
    """Return the first source with at least ``min_targets`` supervised pairs."""
    min_targets = int(min_targets)
    if min_targets < 1:
        raise ValueError("min_targets must be positive")
    sources = np.asarray(sources, dtype=np.int64).reshape(-1)
    for source in sources:
        xyz, valid = dataset.source_all_targets(int(index), int(source))
        eligible = np.asarray(valid, dtype=bool).reshape(21, -1).any(axis=1)
        if int(eligible.sum()) >= min_targets:
            return int(source), xyz, valid
    raise ValueError(
        f"clip index {index} has no source with {min_targets} eligible targets"
    )


def sample_eligible_targets(valid: np.ndarray, k: int,
                            rng: np.random.Generator) -> np.ndarray:
    """Uniform without-replacement targets among pairs with >=1 valid point."""
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 3:
        raise ValueError("valid must be [T,H,W]")
    eligible = np.flatnonzero(valid.reshape(valid.shape[0], -1).any(axis=1))
    k = int(k)
    if k < 1:
        raise ValueError("K must be positive")
    if len(eligible) < k:
        raise ValueError(
            f"selected source has {len(eligible)} eligible targets, fewer than K={k}"
        )
    return np.asarray(rng.choice(eligible, size=k, replace=False), dtype=np.int64)


def deterministic_sample_plan(dataset: TrainingDataset, dataset_name: str,
                              seed: int, global_step: int, microstep: int,
                              rank: int, microsteps_per_rank: int = 2
                              ) -> tuple[int, int, np.random.Generator]:
    """Plan one clip/source from only checkpointed counters and rank."""
    if not len(dataset):
        raise ValueError(f"empty dataset: {dataset_name}")
    dataset_id = DATASET_NAMES.index(dataset_name)
    sequence = np.random.SeedSequence([
        int(seed), int(global_step), int(microstep), int(rank), dataset_id,
    ])
    rng = np.random.default_rng(sequence)
    rows = getattr(dataset, "rows", [])
    if dataset_name != "kubric" and rows and all("parent_id" in row for row in rows):
        # The parent->members mapping is dataset-static; build it once and
        # reuse it across the ~400k per-rank plan calls instead of rebuilding
        # it (and re-scanning every row) on each call.
        cache = getattr(dataset, "_parent_index_cache", None)
        if cache is None:
            parents: dict[str, list[int]] = {}
            for index, row in enumerate(rows):
                parents.setdefault(str(row["parent_id"]), []).append(index)
            names = sorted(parents)
            cache = (names, parents)
            dataset._parent_index_cache = cache
        names, parents = cache
        # All ranks and accumulation microsteps stay in one parent/scene for
        # this update; rank-local RNG chooses different clips inside the block.
        parent_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(global_step), dataset_id, 991,
        ]))
        parent = names[int(parent_rng.integers(len(names)))]
        members = parents[parent]
        base = int(parent_rng.integers(len(members)))
        index = members[(base + rank * int(microsteps_per_rank) + int(microstep)) % len(members)]
    else:
        base_rng = np.random.default_rng(np.random.SeedSequence([
            int(seed), int(global_step), dataset_id, 557,
        ]))
        base = int(base_rng.integers(len(dataset)))
        index = (base + rank * int(microsteps_per_rank) + int(microstep)) % len(dataset)
    source = int(rng.integers(21))
    return index, source, rng
