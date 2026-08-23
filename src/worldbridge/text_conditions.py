"""Strict dataset-aware Wan text-condition loading for training and inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .models.factory import load_text_condition
from .data.constants import DATASET_NAMES


def file_sha256(path: str | Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while value := stream.read(chunk):
            digest.update(value)
    return digest.hexdigest()


def load_dataset_text_conditions(config: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Load all formal conditions and prove they match configured prompts.

    No empty/implicit fallback is allowed. The same function is shared by
    training and inference so evaluation cannot silently use a different text
    condition from the corresponding dataset's training updates.
    """
    paths = config.get("text_conditions")
    prompts = config.get("prompts")
    if not isinstance(paths, dict) or not isinstance(prompts, dict):
        raise ValueError("formal 256 route requires text_conditions and prompts mappings")
    metadata_path = Path(paths.get("metadata", ""))
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Wan text-condition metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    result: dict[str, torch.Tensor] = {}
    for name in DATASET_NAMES:
        if name not in paths or name not in prompts:
            raise ValueError(f"missing formal prompt/condition for {name}")
        path = Path(paths[name])
        condition = load_text_condition(path).cpu()
        record = metadata.get(name, {})
        if record.get("dataset") != name or record.get("prompt") != prompts[name]:
            raise ValueError(f"prompt metadata mismatch for {name}")
        expected_sha = record.get("condition_sha256")
        actual_sha = file_sha256(path)
        if expected_sha != actual_sha:
            raise ValueError(
                f"condition checksum mismatch for {name}: metadata={expected_sha}, actual={actual_sha}"
            )
        if record.get("shape") != [1, 512, 4096]:
            raise ValueError(f"condition metadata shape mismatch for {name}: {record.get('shape')}")
        result[name] = condition
    return result, metadata


def load_inference_text_condition(config: dict[str, Any], dataset: str
                                  ) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return exactly the condition associated with an inference dataset."""
    dataset = str(dataset).lower()
    if dataset not in DATASET_NAMES:
        raise ValueError(f"dataset must be one of {DATASET_NAMES}, got {dataset!r}")
    conditions, metadata = load_dataset_text_conditions(config)
    return conditions[dataset], metadata[dataset]
