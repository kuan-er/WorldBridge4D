"""Shared constants for the three-dataset route."""
from pathlib import Path

DATASET_NAMES = ("kubric", "pointodyssey", "dynamic_replica")
KUBRIC_GEOMETRY_CACHE = Path("/tmp/worldbridge4d-cache/kubric_geometry")
KUBRIC_MMAP_FIELDS = ("depth", "depth_valid", "segmentation")
MIX_CYCLE = (
    "kubric", "pointodyssey", "dynamic_replica", "kubric", "dynamic_replica",
    "pointodyssey", "kubric", "dynamic_replica", "pointodyssey", "kubric",
    "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica", "pointodyssey",
    "kubric", "dynamic_replica", "pointodyssey", "kubric", "dynamic_replica",
)
