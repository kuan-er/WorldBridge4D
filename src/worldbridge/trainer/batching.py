"""Deterministic geometry planning and bounded background prefetch."""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Any, Mapping

import numpy as np

from ..data.boundaries import source_boundary_band, source_contrast_edges
from ..data.sampling import deterministic_sample_plan, source_with_eligible_targets
from ..data.types import TrainingDataset
from .schedulers import dataset_for_step, validated_mix_counts

SamplePlan = tuple[int, int, np.random.Generator]
GeometryValue = tuple[
    int, int, np.ndarray, np.ndarray, np.ndarray | None,
    np.ndarray | None, dict[str, Any] | None, np.ndarray | None, np.ndarray | None,
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
    use_cycle: bool = False,
    boundary_supervision: Mapping[str, Any] | None = None,
    edge_contrast_enabled: bool = False,
) -> TimedGeometryValue:
    """Load one eligible clip/source pair and report worker execution time."""
    task_started = time.perf_counter()
    if edge_contrast_enabled and boundary_supervision is None:
        raise ValueError('edge contrast requires the existing GT boundary context')
    candidates = [int(index)]
    fallback_rng = np.random.default_rng(int(fallback_seed))
    fallback_order = fallback_rng.permutation(len(dataset))
    candidates.extend(int(value) for value in fallback_order if int(value) != int(index))
    last_error = None
    visibility_loader = getattr(dataset, "source_all_targets_with_visibility", None) if use_cycle else None
    camera_loader = getattr(dataset, "cycle_camera", None) if use_cycle else None
    if use_cycle and (visibility_loader is None or camera_loader is None):
        raise ValueError(
            "cycle reprojection requires source visibility and camera metadata"
        )
    for candidate_number, candidate in enumerate(candidates):
        candidate_sources = (
            source_permutation
            if candidate_number == 0
            else fallback_rng.permutation(21)
        )
        try:
            visible = None
            if use_cycle:
                source = None
                for source_candidate in candidate_sources:
                    xyz_candidate, valid_candidate, visible_candidate = visibility_loader(
                        candidate, int(source_candidate)
                    )
                    eligible = np.asarray(valid_candidate, dtype=bool).reshape(21, -1).any(axis=1)
                    if int(eligible.sum()) >= int(required_targets):
                        source = int(source_candidate)
                        xyz, valid, visible = xyz_candidate, valid_candidate, visible_candidate
                        break
                if source is None:
                    raise ValueError(
                        f"clip {candidate} has no source with {required_targets} eligible targets"
                    )
            else:
                source, xyz, valid = source_with_eligible_targets(
                    dataset, candidate, candidate_sources,
                    min_targets=required_targets,
                )
        except ValueError as error:
            last_error = error
            continue
        # RGB failures are data-contract errors, not a reason to alter the
        # deterministic geometry fallback clip/source.
        source_rgb = dataset.source_rgb(candidate, source) if use_source_rgb else None
        camera = camera_loader(candidate) if use_cycle else None
        boundary = contrast_edges = None
        if boundary_supervision is not None:
            loader = getattr(dataset, 'source_boundary_context', None)
            if loader is None:
                raise ValueError('boundary supervision requires GT source boundary context')
            # Like RGB failures, a boundary contract failure MUST NOT select a
            # different clip/source or alter the deterministic fallback/RNG path.
            depth, depth_valid, segmentation = loader(candidate, source)
            boundary = source_boundary_band(
                depth, depth_valid, segmentation,
                radius_px=boundary_supervision['radius_px'],
                relative_jump=boundary_supervision['depth_relative_jump'],
            )
            if boundary.shape != valid.shape[-2:]:
                raise ValueError('source boundary and XYZ grids differ')
            if edge_contrast_enabled:
                contrast_edges = source_contrast_edges(
                    depth, depth_valid, valid[source], segmentation,
                    relative_jump=boundary_supervision['depth_relative_jump'],
                )
        return (
            (candidate, source, xyz, valid, source_rgb, visible, camera, boundary, contrast_edges),
            time.perf_counter() - task_started,
        )
    raise ValueError(
        f"dataset has no clip/source with K={required_targets} eligible targets"
    ) from last_error


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
        cycle_enabled: bool = False,
        cycle_dataset_names: tuple[str, ...] = ("kubric",),
        boundary_supervision: Mapping[str, Any] | None = None,
        edge_contrast_enabled: bool = False,
        start_step: int,
        target_steps: int,
        depth: int,
        workers: int,
        geometry_replay=None,
        dataset_mix_counts=None,
    ) -> None:
        self.datasets = datasets
        self.geometry_replay = geometry_replay
        self.dataset_mix_counts = validated_mix_counts(dataset_mix_counts)
        self.seed = int(seed)
        self.rank = int(rank)
        self.slots_per_rank = int(accumulation) * int(microbatch_per_gpu)
        self.targets_per_source = int(targets_per_source)
        self.use_source_rgb = bool(use_source_rgb)
        self.cycle_enabled = bool(cycle_enabled)
        self.cycle_dataset_names = frozenset(str(name) for name in cycle_dataset_names)
        self.boundary_supervision = None if boundary_supervision is None else dict(boundary_supervision)
        self.edge_contrast_enabled = bool(edge_contrast_enabled)
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
        if self.geometry_replay is not None:
            from .geometry_replay import request_for
            futures = [self.pool.submit(self.geometry_replay.load,
                        request_for(self, step, slot, dataset_name, index))
                       for slot, (index, _source, _rng) in enumerate(sample_plans)]
            return PlannedStep(step, dataset_name, dataset, sample_plans, futures)
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
                self.cycle_enabled and dataset_name in self.cycle_dataset_names,
                self.boundary_supervision,
                self.edge_contrast_enabled,
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

    def quiesce(self, current: PlannedStep) -> int:
        """Finish submitted CPU reads before GPU compute, without replanning.

        The main thread is the only submitter. Waiting for current and lookahead
        futures leaves workers idle until the next pop/refill; results, caches,
        source/target RNG and queue order are retained, not consumed or replaced.
        """
        futures = list(current.geometry_futures)
        for planned in self.pending:
            futures.extend(planned.geometry_futures)
        for future in futures:
            future.result()  # propagate input failures; never eligibility fallback
        return len(futures)

    def close(self) -> None:
        self.pool.shutdown(wait=True)
