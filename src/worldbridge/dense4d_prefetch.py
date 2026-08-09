"""Deterministic source-centric target plans and bounded CPU prefetch."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import queue
import threading
import time
from typing import Sequence

import numpy as np
import torch

from .data import MOViSample
from .dense4d_data import CoordinateStats, DynamicPointmapCache, dense_pair_targets


@dataclass(frozen=True)
class SourceCentricPlan:
    """Main-thread-owned sampling plan for one optimizer update."""

    step: int
    sample_indices: np.ndarray  # [B]
    source: np.ndarray  # [B,T], one source repeated for all target times
    target: np.ndarray  # [B,T], 0..T-1 for every clip


@dataclass
class SourceCentricBatch:
    """One bounded batch; NumPy arrays remain available for exact diagnostics."""

    plan: SourceCentricPlan
    normalized_xyz: np.ndarray  # [B,T,3,H,W]
    metric_xyz: np.ndarray  # [B,T,3,H,W]
    visible: np.ndarray  # [B,T,H,W], M
    valid: np.ndarray  # [B,T,H,W], A
    normalized_xyz_cpu: torch.Tensor
    source_cpu: torch.Tensor
    target_cpu: torch.Tensor
    visible_cpu: torch.Tensor
    valid_cpu: torch.Tensor
    geometry_seconds_sum: float
    geometry_seconds_max: float

    def to_device(self, device: torch.device | str) -> dict[str, torch.Tensor]:
        device = torch.device(device)
        return {
            "source": self.source_cpu.to(device=device, non_blocking=True),
            "target": self.target_cpu.to(device=device, non_blocking=True),
            "target_xyz": self.normalized_xyz_cpu.to(device=device, non_blocking=True),
            "visible": self.visible_cpu.to(device=device, non_blocking=True),
            "valid": self.valid_cpu.to(device=device, non_blocking=True),
        }


def make_source_centric_plan(num_samples: int, batch_size: int, num_frames: int,
                             global_step: int) -> SourceCentricPlan:
    """Create the complete plan on the caller/main thread.

    Clip positions advance in deterministic dataset order.  A clip's source is
    ``(clip_index + visit_index) % num_frames``; therefore successive visits
    rotate a clip uniformly through source times, independent of worker timing.
    """
    num_samples, batch_size, num_frames, global_step = map(
        int, (num_samples, batch_size, num_frames, global_step)
    )
    if num_samples < 1 or batch_size < 1 or num_frames < 1 or global_step < 0:
        raise ValueError("num_samples, batch_size, num_frames must be positive and global_step non-negative")
    positions = global_step * batch_size + np.arange(batch_size, dtype=np.int64)
    sample_indices = positions % num_samples
    visit_indices = positions // num_samples
    source_values = (sample_indices + visit_indices) % num_frames
    source = np.broadcast_to(source_values[:, None], (batch_size, num_frames)).copy()
    target = np.broadcast_to(np.arange(num_frames, dtype=np.int64)[None, :],
                             (batch_size, num_frames)).copy()
    return SourceCentricPlan(global_step, sample_indices.copy(), source, target)


def _pin(array: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    # pin_memory is a CUDA-only optimization; keeping the CPU fallback makes
    # the synchronization smoke runnable on a CPU-only test worker.
    return tensor.pin_memory() if torch.cuda.is_available() else tensor


class _WorkerGeometry:
    """Thread-local geometry cache; no DynamicPointmapCache is shared."""

    def __init__(self, stats: CoordinateStats, depth_tolerance: float,
                 depth_relative_tolerance: float):
        self.stats = stats
        self.depth_tolerance = float(depth_tolerance)
        self.depth_relative_tolerance = float(depth_relative_tolerance)
        self.local = threading.local()

    def cache(self) -> DynamicPointmapCache:
        cache = getattr(self.local, "cache", None)
        if cache is None:
            # A source-centric item needs one source map and all 21 targets.
            cache = DynamicPointmapCache(
                max_entries=1,
                depth_tolerance=self.depth_tolerance,
                depth_relative_tolerance=self.depth_relative_tolerance,
            )
            self.local.cache = cache
        return cache

    def __call__(self, sample: MOViSample, source: np.ndarray,
                 target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        start = time.perf_counter()
        normalized, metric, visible, valid = dense_pair_targets(
            sample, source, target, self.stats, self.cache()
        )
        return normalized, metric, visible, valid, time.perf_counter() - start


def _make_batch(plan: SourceCentricPlan, results: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]) -> SourceCentricBatch:
    normalized = np.stack([result[0] for result in results], axis=0).astype(np.float32, copy=False)
    metric = np.stack([result[1] for result in results], axis=0).astype(np.float32, copy=False)
    visible = np.stack([result[2] for result in results], axis=0).astype(bool, copy=False)
    valid = np.stack([result[3] for result in results], axis=0).astype(bool, copy=False)
    geometry_times = [float(result[4]) for result in results]
    expected = (len(plan.sample_indices), len(plan.target[0]))
    if normalized.ndim != 5 or normalized.shape[:2] != expected or normalized.shape[2] != 3:
        raise ValueError(f"source-centric XYZ shape must be [B,T,3,H,W], got {normalized.shape}")
    if visible.ndim != 4 or visible.shape[:2] != expected:
        raise ValueError(f"source-centric visible shape must be [B,T,H,W], got {visible.shape}")
    if valid.shape != visible.shape:
        raise ValueError(f"source-centric valid shape {valid.shape} != visible {visible.shape}")
    return SourceCentricBatch(
        plan=plan,
        normalized_xyz=normalized,
        metric_xyz=metric,
        visible=visible,
        valid=valid,
        normalized_xyz_cpu=_pin(normalized),
        source_cpu=_pin(plan.source.astype(np.int64, copy=False)),
        target_cpu=_pin(plan.target.astype(np.int64, copy=False)),
        visible_cpu=_pin(visible),
        valid_cpu=_pin(valid),
        geometry_seconds_sum=float(sum(geometry_times)),
        geometry_seconds_max=float(max(geometry_times, default=0.0)),
    )


def build_source_centric_batch(samples: Sequence[MOViSample], stats: CoordinateStats,
                               plan: SourceCentricPlan, *,
                               depth_tolerance: float = 0.05,
                               depth_relative_tolerance: float = 0.01) -> SourceCentricBatch:
    """Synchronous reference implementation used for equality verification."""
    worker = _WorkerGeometry(stats, depth_tolerance, depth_relative_tolerance)
    results = [worker(samples[int(index)], plan.source[row], plan.target[row])
               for row, index in enumerate(plan.sample_indices)]
    return _make_batch(plan, results)


class SourceCentricPrefetcher:
    """Eight-thread CPU geometry producer with a bounded queue of depth two."""

    def __init__(self, samples: Sequence[MOViSample], stats: CoordinateStats, *,
                 workers: int = 8, queue_depth: int = 2,
                 depth_tolerance: float = 0.05,
                 depth_relative_tolerance: float = 0.01):
        if int(workers) != 8:
            raise ValueError("H004 source-centric protocol fixes workers=8")
        if int(queue_depth) != 2:
            raise ValueError("H004 source-centric protocol fixes queue_depth=2")
        self.samples = samples
        self._worker_geometry = _WorkerGeometry(stats, depth_tolerance, depth_relative_tolerance)
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="h004-geometry")
        self._ready: queue.Queue[tuple[SourceCentricPlan, list[Future]]] = queue.Queue(maxsize=2)
        self._closed = False

    def submit(self, plan: SourceCentricPlan) -> float:
        """Submit a main-thread-created plan; blocking is bounded by queue depth."""
        if self._closed:
            raise RuntimeError("prefetcher is closed")
        start = time.perf_counter()
        futures = [self._executor.submit(
            self._worker_geometry,
            self.samples[int(index)], plan.source[row], plan.target[row],
        ) for row, index in enumerate(plan.sample_indices)]
        self._ready.put((plan, futures))
        return time.perf_counter() - start

    def next(self) -> tuple[SourceCentricBatch, float]:
        """Return the oldest batch and CPU wait time for its completion."""
        if self._closed:
            raise RuntimeError("prefetcher is closed")
        start = time.perf_counter()
        plan, futures = self._ready.get()
        results = [future.result() for future in futures]
        return _make_batch(plan, results), time.perf_counter() - start

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "SourceCentricPrefetcher":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def source_centric_loss_weights(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fixed pair weights: 1/3 total diagonal and 2/3 total off-diagonal."""
    source = np.asarray(source)
    target = np.asarray(target)
    if source.shape != target.shape or source.ndim != 2:
        raise ValueError("source and target must have matching [B,K] shape")
    diagonal = source == target
    off_diagonal = ~diagonal
    weights = np.zeros(source.shape, dtype=np.float32)
    for row in range(source.shape[0]):
        diagonal_count = int(diagonal[row].sum())
        off_count = int(off_diagonal[row].sum())
        if diagonal_count:
            weights[row, diagonal[row]] = (1.0 / 3.0) / diagonal_count
        if off_count:
            weights[row, off_diagonal[row]] = (2.0 / 3.0) / off_count
    return weights


def weighted_masked_pair_smooth_l1(prediction: torch.Tensor, target: torch.Tensor,
                                    validity: torch.Tensor, pair_weights: torch.Tensor,
                                    beta: float = 0.05) -> torch.Tensor:
    """Validity-masked XYZ loss with fixed per-update diagonal/off-diagonal mass."""
    if prediction.shape != target.shape or prediction.ndim != 5 or prediction.shape[2] != 3:
        raise ValueError("prediction/target must match [B,K,3,H,W]")
    if validity.shape != prediction.shape[:2] + prediction.shape[-2:]:
        raise ValueError("validity must be [B,K,H,W]")
    if pair_weights.shape != prediction.shape[:2]:
        raise ValueError("pair_weights must be [B,K]")
    error = torch.nn.functional.smooth_l1_loss(prediction, target, beta=beta, reduction="none").sum(dim=2)
    mask = validity.to(dtype=error.dtype)
    per_pair = (error * mask).sum(dim=(-2, -1)) / mask.sum(dim=(-2, -1)).clamp_min(1.0)
    return (per_pair * pair_weights.to(dtype=per_pair.dtype)).sum(dim=1).mean()
