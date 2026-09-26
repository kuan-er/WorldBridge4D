#!/usr/bin/env python3
"""Unified data preparation, cache, and checkpoint maintenance CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

COMMANDS = {
    "audit": "worldbridge.data.commands.audit",
    "cache-roots": "worldbridge.data.commands.cache_roots",
    "compact-dynamic-replica": "worldbridge.data.commands.compact_dynamic_replica",
    "compact-kubric": "worldbridge.data.commands.compact_kubric",
    "compute-stats": "worldbridge.data.commands.compute_stats",
    "convert-anno": "worldbridge.data.commands.convert_anno",
    "convert-kubric-mmap": "worldbridge.data.commands.convert_kubric_mmap",
    "copy-anno": "worldbridge.data.commands.copy_anno",
    "copy-depth": "worldbridge.data.commands.copy_depth",
    "precompute-latents": "worldbridge.data.commands.precompute_latents",
    "preprocess-dynamic-replica": "worldbridge.data.commands.preprocess_dynamic_replica",
    "preprocess-pointodyssey": "worldbridge.data.commands.preprocess_pointodyssey",
    "preprocess-three-dataset": "worldbridge.data.commands.preprocess_three_dataset",
    "text-conditions": "worldbridge.data.commands.text_conditions",
    "check-environment": "worldbridge.utils.commands.check_environment",
    "compare-replay": "worldbridge.trainer.commands.compare_replay",
    "recover-status": "worldbridge.trainer.commands.recover_status",
    "validate-checkpoint": "worldbridge.trainer.commands.validate_checkpoint",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=sorted(COMMANDS))
    args, remainder = parser.parse_known_args()
    sys.argv = [f"{Path(sys.argv[0]).name} {args.command}", *remainder]
    runpy.run_module(COMMANDS[args.command], run_name="__main__")


if __name__ == "__main__":
    main()
