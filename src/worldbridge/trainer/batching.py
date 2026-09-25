"""Deterministic geometry planning and bounded background prefetch."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from ..data.sampling import deterministic_sample_plan, source_with_eligible_targets
from ..data.types import TrainingDataset
from .schedulers import dataset_for_step, validated_mix_counts

SamplePlan = tuple[int, int, np.random.Generator]
GeometryValue = tuple[
    int, int, np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any] | None,
]
TimedGeometryValue = tuple[GeometryValue, float]


@dataclass
class PlannedStep:
    """One deterministic update with its geometry reads already submitted."""

    step: int
    dataset_name: str
    dataset: TrainingDataset
    sample_plans: list[SamplePlan]
    geometry_futures: list[Future[TimedGeometryValue]]

    @property
    def clip_indices(self) -> list[int]:
        return [index for index, _source, _rng in self.sample_plans]


def load_geometry(
    dataset: TrainingDataset,
    index: int,
    source_permutation: np.ndarray,
    required_targets: int,
    fallback_seed: int,
    use_source_rgb: bool,
    camera_supervision: bool = False,
) -> TimedGeometryValue:
    """Load one eligible clip/source pair and report worker execution time."""
    task_started = time.perf_counter()
    candidates = [int(index)]
    fallback_rng = np.random.default_rng(int(fallback_seed))
    fallback_order = fallback_rng.permutation(len(dataset))
    candidates.extend(int(value) for value in fallback_order if int(value) != int(index))
    last_error = None
    for candidate_number, candidate in enumerate(candidates):
        candidate_sources = (
            source_permutation
            if candidate_number == 0
            else fallback_rng.permutation(21)
        )
        try:
            # One shared contract for every dataset: the chosen source yields all
            # 21 target maps, which the camera path needs for eligible-target
            # sampling, diagonal selection and camera metadata.
            source, xyz, valid = source_with_eligible_targets(
                dataset, candidate, candidate_sources,
                min_targets=required_targets, require_diagonal=camera_supervision,
            )
        except ValueError as error:
            last_error = error
            continue
        # RGB failures are data-contract errors, not a reason to alter the
        # deterministic geometry fallback clip/source.
        source_rgb = dataset.source_rgb(candidate, source) if use_source_rgb else None
        camera = None
        if camera_supervision:
            from .camera_objective import validate_supervision_camera
            camera = dataset.supervision_camera(candidate)
            validate_supervision_camera(camera, *valid.shape[-2:])
        return (
            (candidate, source, xyz, valid, source_rgb, camera),
            time.perf_counter() - task_started,
        )


class GeometryPrefetcher:
    """Maintain an ordered queue of deterministic geometry-read futures."""

    def __init__(
        self,
        datasets: Mapping[str, TrainingDataset],
        *,
        seed: int,
        rank: int,
        accumulation: int,
        microbatch_per_gpu: int,
        targets_per_source: int,
        use_source_rgb: bool,
        camera_supervision: bool = False,
        start_step: int,
        target_steps: int,
        depth: int,
        workers: int,
        dataset_mix_counts=None,
    ) -> None:
        self.datasets = datasets
        self.dataset_mix_counts = validated_mix_counts(dataset_mix_counts)
        self.seed = int(seed)
        self.rank = int(rank)
        self.slots_per_rank = int(accumulation) * int(microbatch_per_gpu)
        self.targets_per_source = int(targets_per_source)
        self.use_source_rgb = bool(use_source_rgb)
        self.camera_supervision = bool(camera_supervision)
        self.target_steps = int(target_steps)
        self.depth = int(depth)
        self.next_step = int(start_step)
        self.pending: deque[PlannedStep] = deque()
        self.pool = ThreadPoolExecutor(
            max_workers=int(workers),
            thread_name_prefix="three-dataset-geometry",
        )

    def _plan_step(self, step: int) -> PlannedStep:
        dataset_name = dataset_for_step(step, self.seed, self.dataset_mix_counts)
        dataset = self.datasets[dataset_name]
        sample_plans = [
            deterministic_sample_plan(
                dataset, dataset_name, self.seed, step, slot, self.rank,
                self.slots_per_rank,
            )
            for slot in range(self.slots_per_rank)
        ]
        geometry_futures = [
            self.pool.submit(
                load_geometry,
                dataset,
                index,
                np.random.default_rng(np.random.SeedSequence([
                    self.seed, step, slot, self.rank, 771,
                ])).permutation(21),
                self.targets_per_source,
                int(np.random.SeedSequence([
                    self.seed, step, slot, self.rank, 772,
                ]).generate_state(1)[0]),
                self.use_source_rgb,
                self.camera_supervision,
            )
            for slot, (index, _source, _rng) in enumerate(sample_plans)
        ]
        return PlannedStep(
            step, dataset_name, dataset, sample_plans, geometry_futures,
        )

    def refill(self) -> None:
        while len(self.pending) < self.depth and self.next_step < self.target_steps:
            self.pending.append(self._plan_step(self.next_step))
            self.next_step += 1

    def pop(self, expected_step: int) -> PlannedStep:
        if not self.pending:
            raise RuntimeError("geometry prefetch queue is empty")
        planned = self.pending.popleft()
        if planned.step != int(expected_step):
            raise RuntimeError(
                f"prefetch plan order mismatch: {planned.step} != {expected_step}"
            )
        self.refill()
        return planned

    def close(self) -> None:
        self.pool.shutdown(wait=True)
