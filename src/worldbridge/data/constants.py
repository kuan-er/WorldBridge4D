"""Shared constants for the three-dataset route."""
from pathlib import Path

DATASET_NAMES = ("kubric", "pointodyssey", "dynamic_replica")
PERSISTENT_CACHE_ROOT = Path("/data/WorldBridge4D-persistent/worldbridge4d-cache")
PERSISTENT_CACHE_V2_ROOT = Path("/data/WorldBridge4D-persistent/worldbridge4d-cache-v2")
KUBRIC_GEOMETRY_CACHE = PERSISTENT_CACHE_ROOT / "kubric_geometry"
KUBRIC_GEOMETRY_MMAP_CACHE = PERSISTENT_CACHE_V2_ROOT / "kubric_geometry_mmap"
POINTODYSSEY_ANNO_CACHE = PERSISTENT_CACHE_ROOT / "anno"
POINTODYSSEY_ANNO_NPY_CACHE = PERSISTENT_CACHE_ROOT / "anno_npy"
POINTODYSSEY_DEPTH_CACHE = PERSISTENT_CACHE_ROOT / "depth" / "pointodyssey"
DYNAMIC_REPLICA_DEPTH_CACHE = PERSISTENT_CACHE_ROOT / "depth" / "dynamic_replica"
DYNAMIC_REPLICA_TRAJECTORY_CACHE = PERSISTENT_CACHE_ROOT / "trajectories"
KUBRIC_MMAP_FIELDS = ("depth", "depth_valid", "segmentation")
MIX_CYCLE = (
    "kubric", "pointodyssey", "dynamic_replica", "kubric", "dynamic_replica",
    "pointodyssey", "kubric", "dynamic_replica", "pointodyssey", "kubric",
    "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica", "pointodyssey",
    "kubric", "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica",
)
