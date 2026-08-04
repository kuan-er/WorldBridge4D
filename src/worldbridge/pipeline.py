"""Blockwise conversion from MOVi geometry to model inputs and balanced queries."""
from __future__ import annotations
from collections.abc import Iterable
import numpy as np
import torch

from .data import MOViSample
from .geometry import GeometryBuilder
from .models import WorldLatentModel


def train_coordinate_stats(samples: Iterable[MOViSample], depth_tolerance: float = 0.05,
                           depth_relative_tolerance: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """Coordinate moments from training pointmaps only (never validation/test)."""
    count = 0
    total = np.zeros(3, np.float64)
    total2 = np.zeros(3, np.float64)
    for sample in samples:
        p, valid = GeometryBuilder(sample, depth_tolerance, depth_relative_tolerance).pointmaps()
        x = p[valid]
        count += len(x); total += x.sum(axis=0, dtype=np.float64); total2 += np.square(x, dtype=np.float64).sum(axis=0)
    if count == 0:
        raise ValueError("no valid train coordinates")
    mean = total / count
    scale = np.sqrt(np.maximum(total2 / count - mean * mean, 1e-6))
    return mean.astype(np.float32), scale.astype(np.float32)


def _compact_inputs(geom: GeometryBuilder, device: torch.device, block_size: int):
    sample = geom.sample; T,H,W = sample.num_frames, sample.height, sample.width
    p, pv = geom.pointmaps()
    blocks = [geom.trajectory_block(0, start, min(start + block_size, H*W))[:3]
              for start in range(0, H*W, block_size)]
    x = np.concatenate([b[0] for b in blocks], axis=0).reshape(H,W,T,3).transpose(2,0,1,3)
    m = np.concatenate([b[1] for b in blocks], axis=0).reshape(H,W,T).transpose(2,0,1)
    v = np.concatenate([b[2] for b in blocks], axis=0).reshape(H,W,T).transpose(2,0,1)
    pointmaps = torch.from_numpy(p.transpose(0,3,1,2))[None].to(device)
    point_valid = torch.from_numpy(pv)[None].to(device)
    return pointmaps, point_valid, torch.from_numpy(x)[None].to(device), torch.from_numpy(m)[None].to(device), torch.from_numpy(v)[None].to(device)


def encode_sample(model: WorldLatentModel, sample: MOViSample, device: torch.device,
                  block_size: int = 16384, depth_tolerance: float = 0.05,
                  depth_relative_tolerance: float = 0.01) -> tuple[torch.Tensor, GeometryBuilder]:
    """Encode one clip; Full trajectories are generated one source/block at a time."""
    geom = GeometryBuilder(sample, depth_tolerance, depth_relative_tolerance)
    T,H,W = sample.num_frames, sample.height, sample.width
    if model.mode == "compact":
        p, pv, x, m, v = _compact_inputs(geom, device, block_size)
        return model(pointmaps=p, point_valid=pv, anchor_values=x, anchor_visible=m,
                     anchor_valid=v, source_length=T), geom
    sources = []
    for s in range(T):
        chunks = []
        for start in range(0, H*W, block_size):
            x, m, v, _ = geom.trajectory_block(s, start, min(start + block_size, H*W))
            xt = torch.from_numpy(x).to(device)
            mt = torch.from_numpy(m).to(device)
            vt = torch.from_numpy(v).to(device)
            chunks.append(model.encode_trajectory(xt, mt, vt))
        sources.append(torch.cat(chunks, dim=0).reshape(H,W,-1))
    h = torch.stack(sources, dim=0)[None]
    return model(trajectory_features=h, source_length=T), geom


def sample_balanced_queries(geom: GeometryBuilder, mode: str, num_queries: int,
                            rng: np.random.Generator) -> dict[str, np.ndarray]:
    T,H,W = geom.sample.num_frames, geom.sample.height, geom.sample.width
    if mode == "compact":
        counts = [num_queries // 2, num_queries - num_queries // 2]
        groups = np.concatenate([np.zeros(counts[0], np.int64), np.ones(counts[1], np.int64)])
    else:
        a = num_queries // 3; counts = [a, a, num_queries - 2*a]
        groups = np.concatenate([np.zeros(counts[0], np.int64), np.ones(counts[1], np.int64), np.full(counts[2],2,np.int64)])
    source = np.empty(num_queries, np.int64); target = np.empty(num_queries, np.int64)
    sel = groups == 0; source[sel] = rng.integers(T,size=sel.sum()); target[sel] = source[sel]
    sel = groups == 1; source[sel] = 0; target[sel] = rng.integers(T,size=sel.sum())
    sel = groups == 2
    if sel.any():
        source[sel] = rng.integers(T,size=sel.sum()); target[sel] = rng.integers(T,size=sel.sum())
    uv = np.stack([rng.integers(W,size=num_queries), rng.integers(H,size=num_queries)],axis=-1).astype(np.int64)
    x, visible, valid = geom.query(source,uv)
    row = np.arange(num_queries)
    target_x = x[row,target]
    target_visible = visible[row,target]
    target_valid = valid[row,target]
    uv01 = uv.astype(np.float32); uv01[:,0] /= max(W-1,1); uv01[:,1] /= max(H-1,1)
    return {"source":source, "target":target, "uv":uv, "uv01":uv01, "x":target_x,
            "visible":target_visible, "valid":target_valid, "groups":groups}


def query_tensors(query: dict[str,np.ndarray], device: torch.device):
    source = torch.from_numpy(query["source"].astype(np.float32))[None].to(device)
    target = torch.from_numpy(query["target"].astype(np.float32))[None].to(device)
    uv01 = torch.from_numpy(query["uv01"])[None].to(device)
    x = torch.from_numpy(query["x"]).to(device)
    valid = torch.from_numpy(query["valid"]).to(device)
    groups = torch.from_numpy(query["groups"]).to(device)
    return source,target,uv01,x,valid,groups
