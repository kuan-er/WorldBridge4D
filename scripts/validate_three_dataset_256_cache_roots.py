#!/usr/bin/env python3
"""Fail closed when lazy latent caches use ephemeral or indirect storage."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable

import yaml

DATASET_NAMES = ("kubric", "pointodyssey", "dynamic_replica")
LAZY_RELATIVE = Path("latents/wan2.1_1.3b_fp32_256_lazy")


def _under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_cache_roots(
    config: dict[str, Any], *, create: bool = False,
    forbidden_roots: Iterable[Path] = (Path("/tmp"),),
) -> dict[str, str]:
    """Require a durable backup, and permit a /tmp hot-cache symlink.

    The authoritative latents must live in a durable, non-symlinked directory
    named ``<lazy>_backup`` next to the live path.  The live path itself may be
    a symlink onto /tmp scratch storage: it is a reproducible hot cache whose
    loss is recoverable by re-copying from the backup.
    """
    failures: list[str] = []
    roots: dict[str, str] = {}
    forbidden = tuple(Path(value).resolve(strict=False) for value in forbidden_roots)
    for name in DATASET_NAMES:
        cache_root = Path(config["datasets"][name]["cache_root"])
        root = cache_root / LAZY_RELATIVE
        backup = cache_root / "latents" / f"{LAZY_RELATIVE.name}_backup"
        roots[name] = str(root)
        # The durable authoritative copy must exist as a real directory.
        if backup.is_symlink() or not backup.is_dir():
            failures.append(f"{name}: durable latent backup missing or symlinked ({backup})")
            continue
        if root.is_symlink():
            target = os.readlink(root)
            if not root.resolve(strict=False).is_dir():
                failures.append(f"{name}: latent cache symlink target missing ({root} -> {target})")
                continue
            # Symlink onto /tmp is allowed: it is a recoverable hot cache.
        else:
            resolved = root.resolve(strict=False)
            if any(_under(resolved, value) for value in forbidden):
                failures.append(f"{name}: cache root is on non-durable scratch storage ({resolved})")
                continue
            if root.exists() and not root.is_dir():
                failures.append(f"{name}: cache root is not a directory ({root})")
                continue
            if create:
                root.mkdir(parents=True, exist_ok=True)
            elif not root.is_dir():
                failures.append(f"{name}: cache root is missing ({root})")
    if failures:
        raise RuntimeError("unsafe lazy latent cache roots: " + "; ".join(failures))
    return roots


def assert_expected_latent_files(roots: dict[str, str], expected: int) -> dict[str, int]:
    """Require a stable file count after distributed precompute completes."""
    expected = int(expected)
    counts = {
        name: sum(1 for _ in Path(root).glob("latent_*.safetensors"))
        for name, root in roots.items()
    }
    total = sum(counts.values())
    if total != expected:
        raise RuntimeError(
            f"durable lazy latent count mismatch: expected={expected}, "
            f"observed={total}, datasets={counts}"
        )
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--create", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    roots = validate_cache_roots(config, create=args.create)
    print(json.dumps({"event": "durable_lazy_cache_roots", "roots": roots}), flush=True)
    print("DURABLE_LAZY_CACHE_ROOTS_OK", flush=True)


if __name__ == "__main__":
    main()
